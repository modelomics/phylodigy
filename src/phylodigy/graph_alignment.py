"""Auditable primitive indel candidates between computation graphs.

This module aligns exact structural node labels and emits replay operations.
Adding or removing an endpoint-local object describes only the mechanical
direction from the selected left endpoint to the right endpoint.  It is not a
claim about ancestry, historical gains, or historical losses.

The bounded canonical-sequence solver is deterministic and symmetric under
endpoint reversal, but it is not a general minimum graph-edit solver.  Every
result therefore carries a rigorous lower bound and its replayable upper bound.
Objective optimality is reported only by endpoint validation, never trusted
from a serialized scalar.
"""

from __future__ import annotations

import math
import re
from array import array
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from .canonical import canonical_json, content_digest, freeze_json, normalize_json
from .computation_graph import (
    ComputationGraph,
    GraphEdge,
    GraphNode,
    GraphProfile,
    _canonical_node_colors,
    _canonical_topological_order,
    _edge_label,
    _operation_is_commutative,
)


GRAPH_ALIGNMENT_SCHEMA_VERSION = "1.0.0"
GRAPH_ALIGNMENT_ALGORITHM = "canonical-sequence-exact-label-indel.v1"
COMPARISON_ORIENTATION = "comparison_direction_only"
PRIMITIVE_INDEL_OBJECTIVE = "exact_label_primitive_node_edge_edit.v1"
LOWER_BOUND_METHOD = "exact_node_and_edge_label_multiset.v1"


class AlignmentResourceLimitError(ValueError):
    """Raised when the bounded alignment matrix would exceed its budget."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: Any, name: str) -> str:
    text = _nonempty(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return text


def _positive_cost(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def _finite_sum(values: Sequence[float], name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class AlignmentParameters:
    """Symmetric objective costs plus a separate deterministic solver budget."""

    node_indel_cost: float = 1.0
    edge_indel_cost: float = 1.0
    max_matrix_cells: int = 1_000_000
    algorithm: str = GRAPH_ALIGNMENT_ALGORITHM

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_indel_cost",
            _positive_cost(self.node_indel_cost, "node_indel_cost"),
        )
        object.__setattr__(
            self,
            "edge_indel_cost",
            _positive_cost(self.edge_indel_cost, "edge_indel_cost"),
        )
        if (
            isinstance(self.max_matrix_cells, bool)
            or not isinstance(self.max_matrix_cells, int)
        ):
            raise TypeError("max_matrix_cells must be an integer")
        if self.max_matrix_cells <= 0:
            raise ValueError("max_matrix_cells must be positive")
        if self.algorithm != GRAPH_ALIGNMENT_ALGORITHM:
            raise ValueError(
                f"algorithm must be {GRAPH_ALIGNMENT_ALGORITHM!r}"
            )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    @property
    def cost_model_digest(self) -> str:
        return content_digest(self.cost_model_dict())

    @property
    def solver_configuration_digest(self) -> str:
        return content_digest(self.solver_configuration_dict())

    def cost_model_dict(self) -> dict[str, Any]:
        return {
            "edge_indel_cost": self.edge_indel_cost,
            "node_indel_cost": self.node_indel_cost,
            "objective": PRIMITIVE_INDEL_OBJECTIVE,
        }

    def solver_configuration_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "max_matrix_cells": self.max_matrix_cells,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "edge_indel_cost": self.edge_indel_cost,
            "max_matrix_cells": self.max_matrix_cells,
            "node_indel_cost": self.node_indel_cost,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AlignmentParameters":
        if not isinstance(raw, Mapping):
            raise TypeError("alignment parameters must be a mapping")
        return cls(**dict(raw))


@dataclass(frozen=True)
class StructuralNode:
    """Observation-free node payload local to one endpoint graph."""

    ref: str
    operation: str
    attributes: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref", _nonempty(self.ref, "node ref"))
        object.__setattr__(
            self, "operation", _nonempty(self.operation, "node operation")
        )
        normalized = normalize_json(dict(self.attributes))
        if not isinstance(normalized, dict):  # pragma: no cover - defensive
            raise TypeError("node attributes must normalize to an object")
        object.__setattr__(self, "attributes", freeze_json(normalized))

    @property
    def structural_signature(self) -> str:
        return content_digest(self.label_dict())

    def label_dict(self) -> dict[str, Any]:
        return {
            "attributes": normalize_json(self.attributes),
            "operation": self.operation,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributes": normalize_json(self.attributes),
            "operation": self.operation,
            "ref": self.ref,
            "structural_signature": self.structural_signature,
        }

    @classmethod
    def from_graph_node(cls, node: GraphNode) -> "StructuralNode":
        return cls(node.node_id, node.operation, node.attributes)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StructuralNode":
        result = cls(
            ref=str(raw["ref"]),
            operation=str(raw["operation"]),
            attributes=raw.get("attributes", {}),
        )
        supplied = raw.get("structural_signature")
        if supplied is not None and str(supplied) != result.structural_signature:
            raise ValueError("node structural signature mismatch")
        return result


@dataclass(frozen=True)
class StructuralEdge:
    """One endpoint-local directed, position-aware edge."""

    source: str
    target: str
    position: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _nonempty(self.source, "edge source"))
        object.__setattr__(self, "target", _nonempty(self.target, "edge target"))
        object.__setattr__(
            self, "position", _nonempty(self.position, "edge position")
        )

    @property
    def ref(self) -> str:
        return f"{self.source}->{self.target}@{self.position}"

    def to_dict(self) -> dict[str, str]:
        return {
            "position": self.position,
            "ref": self.ref,
            "source": self.source,
            "target": self.target,
        }

    @classmethod
    def from_graph_edge(cls, edge: GraphEdge) -> "StructuralEdge":
        return cls(edge.source, edge.target, edge.position)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StructuralEdge":
        result = cls(
            source=str(raw["source"]),
            target=str(raw["target"]),
            position=str(raw["position"]),
        )
        supplied = raw.get("ref")
        if supplied is not None and str(supplied) != result.ref:
            raise ValueError("edge ref mismatch")
        return result


@dataclass(frozen=True)
class NodeMatch:
    """One exact structural-label correspondence."""

    left_ref: str
    right_ref: str
    structural_signature: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_ref", _nonempty(self.left_ref, "left ref"))
        object.__setattr__(self, "right_ref", _nonempty(self.right_ref, "right ref"))
        object.__setattr__(
            self,
            "structural_signature",
            _digest(self.structural_signature, "node structural signature"),
        )

    def invert(self) -> "NodeMatch":
        return NodeMatch(self.right_ref, self.left_ref, self.structural_signature)

    def to_dict(self) -> dict[str, str]:
        return {
            "left_ref": self.left_ref,
            "right_ref": self.right_ref,
            "structural_signature": self.structural_signature,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NodeMatch":
        return cls(
            left_ref=str(raw["left_ref"]),
            right_ref=str(raw["right_ref"]),
            structural_signature=str(raw["structural_signature"]),
        )


@dataclass(frozen=True)
class EdgeMatch:
    """One structurally equivalent edge correspondence used during replay."""

    left: StructuralEdge
    right: StructuralEdge
    canonical_port_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.left, StructuralEdge) or not isinstance(
            self.right, StructuralEdge
        ):
            raise TypeError("edge matches require StructuralEdge endpoints")
        object.__setattr__(
            self,
            "canonical_port_label",
            _nonempty(self.canonical_port_label, "canonical port label"),
        )

    def invert(self) -> "EdgeMatch":
        return EdgeMatch(self.right, self.left, self.canonical_port_label)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_port_label": self.canonical_port_label,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EdgeMatch":
        return cls(
            left=StructuralEdge.from_dict(raw["left"]),
            right=StructuralEdge.from_dict(raw["right"]),
            canonical_port_label=str(raw["canonical_port_label"]),
        )


@dataclass(frozen=True)
class AlignmentAmbiguity:
    """A local structural orbit underidentified by this representative solver."""

    structural_signature: str
    left_refs: tuple[str, ...]
    right_refs: tuple[str, ...]
    matched_count: int
    kind: str = "repeated_exact_local_context"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "structural_signature",
            _digest(self.structural_signature, "ambiguity structural signature"),
        )
        left = tuple(sorted({_nonempty(item, "left ambiguity ref") for item in self.left_refs}))
        right = tuple(sorted({_nonempty(item, "right ambiguity ref") for item in self.right_refs}))
        if len(left) <= 1 and len(right) <= 1:
            raise ValueError("an ambiguity requires a repeated endpoint label")
        if isinstance(self.matched_count, bool) or not isinstance(self.matched_count, int):
            raise TypeError("matched_count must be an integer")
        if self.matched_count < 0 or self.matched_count > min(len(left), len(right)):
            raise ValueError("matched_count is incompatible with ambiguity refs")
        if self.kind != "repeated_exact_local_context":
            raise ValueError("unsupported solver-underidentification kind")
        object.__setattr__(self, "left_refs", left)
        object.__setattr__(self, "right_refs", right)

    def invert(self) -> "AlignmentAmbiguity":
        return AlignmentAmbiguity(
            self.structural_signature,
            self.right_refs,
            self.left_refs,
            self.matched_count,
            self.kind,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "left_refs": list(self.left_refs),
            "matched_count": self.matched_count,
            "right_refs": list(self.right_refs),
            "structural_signature": self.structural_signature,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AlignmentAmbiguity":
        return cls(
            structural_signature=str(raw["structural_signature"]),
            left_refs=tuple(str(item) for item in raw.get("left_refs", [])),
            right_refs=tuple(str(item) for item in raw.get("right_refs", [])),
            matched_count=raw["matched_count"],
            kind=str(raw.get("kind", "repeated_exact_local_context")),
        )


@dataclass(frozen=True)
class AddNodeOperation:
    node: StructuralNode
    cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.node, StructuralNode):
            raise TypeError("add-node operation requires a StructuralNode")
        object.__setattr__(self, "cost", _positive_cost(self.cost, "operation cost"))

    @property
    def kind(self) -> str:
        return "add_node"

    @property
    def operation_id(self) -> str:
        return content_digest(self.to_dict(include_id=False))

    def invert(self) -> "RemoveNodeOperation":
        return RemoveNodeOperation(self.node, self.cost)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {"cost": self.cost, "kind": self.kind, "node": self.node.to_dict()}
        if include_id:
            result["id"] = self.operation_id
        return result


@dataclass(frozen=True)
class RemoveNodeOperation:
    node: StructuralNode
    cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.node, StructuralNode):
            raise TypeError("remove-node operation requires a StructuralNode")
        object.__setattr__(self, "cost", _positive_cost(self.cost, "operation cost"))

    @property
    def kind(self) -> str:
        return "remove_node"

    @property
    def operation_id(self) -> str:
        return content_digest(self.to_dict(include_id=False))

    def invert(self) -> AddNodeOperation:
        return AddNodeOperation(self.node, self.cost)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {"cost": self.cost, "kind": self.kind, "node": self.node.to_dict()}
        if include_id:
            result["id"] = self.operation_id
        return result


@dataclass(frozen=True)
class AddEdgeOperation:
    edge: StructuralEdge
    cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.edge, StructuralEdge):
            raise TypeError("add-edge operation requires a StructuralEdge")
        object.__setattr__(self, "cost", _positive_cost(self.cost, "operation cost"))

    @property
    def kind(self) -> str:
        return "add_edge"

    @property
    def operation_id(self) -> str:
        return content_digest(self.to_dict(include_id=False))

    def invert(self) -> "RemoveEdgeOperation":
        return RemoveEdgeOperation(self.edge, self.cost)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {"cost": self.cost, "edge": self.edge.to_dict(), "kind": self.kind}
        if include_id:
            result["id"] = self.operation_id
        return result


@dataclass(frozen=True)
class RemoveEdgeOperation:
    edge: StructuralEdge
    cost: float

    def __post_init__(self) -> None:
        if not isinstance(self.edge, StructuralEdge):
            raise TypeError("remove-edge operation requires a StructuralEdge")
        object.__setattr__(self, "cost", _positive_cost(self.cost, "operation cost"))

    @property
    def kind(self) -> str:
        return "remove_edge"

    @property
    def operation_id(self) -> str:
        return content_digest(self.to_dict(include_id=False))

    def invert(self) -> AddEdgeOperation:
        return AddEdgeOperation(self.edge, self.cost)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {"cost": self.cost, "edge": self.edge.to_dict(), "kind": self.kind}
        if include_id:
            result["id"] = self.operation_id
        return result


# Short internal spellings keep the replay implementation readable without
# exposing historical polarity through the public API.
NodeInsertion = AddNodeOperation
NodeDeletion = RemoveNodeOperation
EdgeInsertion = AddEdgeOperation
EdgeDeletion = RemoveEdgeOperation

GraphAlignmentOperation: TypeAlias = (
    AddNodeOperation | RemoveNodeOperation | AddEdgeOperation | RemoveEdgeOperation
)


def _operation_from_dict(raw: Mapping[str, Any]) -> GraphAlignmentOperation:
    kind = raw.get("kind")
    cost = raw.get("cost")
    if kind == "add_node":
        result: GraphAlignmentOperation = NodeInsertion(
            StructuralNode.from_dict(raw["node"]), cost
        )
    elif kind == "remove_node":
        result = NodeDeletion(StructuralNode.from_dict(raw["node"]), cost)
    elif kind == "add_edge":
        result = EdgeInsertion(StructuralEdge.from_dict(raw["edge"]), cost)
    elif kind == "remove_edge":
        result = EdgeDeletion(StructuralEdge.from_dict(raw["edge"]), cost)
    else:
        raise ValueError(f"unsupported graph replay operation kind: {kind!r}")
    supplied = raw.get("id")
    if supplied is not None and str(supplied) != result.operation_id:
        raise ValueError("graph replay operation ID mismatch")
    return result


def _operation_sort_key(operation: GraphAlignmentOperation) -> tuple[int, str]:
    order = {
        "remove_edge": 0,
        "remove_node": 1,
        "add_node": 2,
        "add_edge": 3,
    }
    return order[operation.kind], operation.operation_id


@dataclass(frozen=True)
class GraphAlignment:
    """One bounded, replayable exact-label alignment candidate."""

    left_structural_digest: str
    right_structural_digest: str
    parameters: AlignmentParameters
    node_matches: tuple[NodeMatch, ...]
    edge_matches: tuple[EdgeMatch, ...]
    underidentified_mappings: tuple[AlignmentAmbiguity, ...]
    operations: tuple[GraphAlignmentOperation, ...]
    lower_bound_node_operations: int
    lower_bound_edge_operations: int
    representative_traceback_tie_count: int = 0
    algorithm: str = GRAPH_ALIGNMENT_ALGORITHM
    schema_version: str = GRAPH_ALIGNMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_structural_digest",
            _digest(self.left_structural_digest, "left structural digest"),
        )
        object.__setattr__(
            self,
            "right_structural_digest",
            _digest(self.right_structural_digest, "right structural digest"),
        )
        if not isinstance(self.parameters, AlignmentParameters):
            raise TypeError("parameters must be AlignmentParameters")
        if self.algorithm != GRAPH_ALIGNMENT_ALGORITHM:
            raise ValueError(f"algorithm must be {GRAPH_ALIGNMENT_ALGORITHM!r}")
        if self.parameters.algorithm != self.algorithm:
            raise ValueError("parameter and alignment algorithms must match")
        if self.schema_version != GRAPH_ALIGNMENT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {GRAPH_ALIGNMENT_SCHEMA_VERSION!r}"
            )
        node_matches = tuple(self.node_matches)
        if any(not isinstance(item, NodeMatch) for item in node_matches):
            raise TypeError("node_matches must contain NodeMatch objects")
        node_matches = tuple(
            sorted(node_matches, key=lambda item: (item.left_ref, item.right_ref))
        )
        if len({item.left_ref for item in node_matches}) != len(node_matches):
            raise ValueError("left node matches must be injective")
        if len({item.right_ref for item in node_matches}) != len(node_matches):
            raise ValueError("right node matches must be injective")
        edge_matches = tuple(self.edge_matches)
        if any(not isinstance(item, EdgeMatch) for item in edge_matches):
            raise TypeError("edge_matches must contain EdgeMatch objects")
        edge_matches = tuple(
            sorted(
                edge_matches,
                key=lambda item: (item.left.ref, item.right.ref),
            )
        )
        if len({item.left.ref for item in edge_matches}) != len(edge_matches):
            raise ValueError("left edge matches must be injective")
        if len({item.right.ref for item in edge_matches}) != len(edge_matches):
            raise ValueError("right edge matches must be injective")
        underidentified = tuple(self.underidentified_mappings)
        if any(not isinstance(item, AlignmentAmbiguity) for item in underidentified):
            raise TypeError(
                "underidentified_mappings must contain AlignmentAmbiguity objects"
            )
        underidentified = tuple(
            sorted(
                underidentified,
                key=lambda item: (
                    item.structural_signature,
                    item.left_refs,
                    item.right_refs,
                ),
            )
        )
        operations = tuple(self.operations)
        if any(
            not isinstance(item, (NodeInsertion, NodeDeletion, EdgeInsertion, EdgeDeletion))
            for item in operations
        ):
            raise TypeError("operations must contain primitive replay operations")
        operations = tuple(sorted(operations, key=_operation_sort_key))
        operation_ids = [item.operation_id for item in operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("graph replay operations must be unique")
        for operation_type, label in (
            (NodeDeletion, "left node removals"),
            (NodeInsertion, "right node additions"),
            (EdgeDeletion, "left edge removals"),
            (EdgeInsertion, "right edge additions"),
        ):
            refs = [
                item.node.ref
                if isinstance(item, (NodeInsertion, NodeDeletion))
                else item.edge.ref
                for item in operations
                if isinstance(item, operation_type)
            ]
            if len(refs) != len(set(refs)):
                raise ValueError(f"{label} must reference unique objects")
        matched_left_nodes = {item.left_ref for item in node_matches}
        matched_right_nodes = {item.right_ref for item in node_matches}
        removed_left_nodes = {
            item.node.ref for item in operations if isinstance(item, NodeDeletion)
        }
        added_right_nodes = {
            item.node.ref for item in operations if isinstance(item, NodeInsertion)
        }
        matched_left_edges = {item.left.ref for item in edge_matches}
        matched_right_edges = {item.right.ref for item in edge_matches}
        removed_left_edges = {
            item.edge.ref for item in operations if isinstance(item, EdgeDeletion)
        }
        added_right_edges = {
            item.edge.ref for item in operations if isinstance(item, EdgeInsertion)
        }
        if matched_left_nodes & removed_left_nodes:
            raise ValueError("a left node cannot be both matched and removed")
        if matched_right_nodes & added_right_nodes:
            raise ValueError("a right node cannot be both matched and added")
        if matched_left_edges & removed_left_edges:
            raise ValueError("a left edge cannot be both matched and removed")
        if matched_right_edges & added_right_edges:
            raise ValueError("a right edge cannot be both matched and added")
        for operation in operations:
            expected = (
                self.parameters.node_indel_cost
                if isinstance(operation, (NodeInsertion, NodeDeletion))
                else self.parameters.edge_indel_cost
            )
            if operation.cost != expected:
                raise ValueError("operation cost does not match the cost model")
        for name in (
            "lower_bound_node_operations",
            "lower_bound_edge_operations",
            "representative_traceback_tie_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.lower_bound_node_operations > self.node_operation_count:
            raise ValueError("node-operation lower bound exceeds the candidate")
        if self.lower_bound_edge_operations > self.edge_operation_count:
            raise ValueError("edge-operation lower bound exceeds the candidate")
        if not math.isfinite(self.total_cost) or not math.isfinite(self.lower_bound):
            raise ValueError("aggregate alignment costs must be finite")
        object.__setattr__(self, "node_matches", node_matches)
        object.__setattr__(self, "edge_matches", edge_matches)
        object.__setattr__(self, "underidentified_mappings", underidentified)
        object.__setattr__(self, "operations", operations)

    @property
    def node_operation_count(self) -> int:
        return sum(
            isinstance(item, (NodeInsertion, NodeDeletion))
            for item in self.operations
        )

    @property
    def edge_operation_count(self) -> int:
        return sum(
            isinstance(item, (EdgeInsertion, EdgeDeletion))
            for item in self.operations
        )

    @property
    def total_cost(self) -> float:
        return _finite_sum(
            [item.cost for item in self.operations], "alignment total cost"
        )

    @property
    def upper_bound(self) -> float:
        return self.total_cost

    @property
    def lower_bound(self) -> float:
        return _finite_sum(
            [
                self.parameters.node_indel_cost
                * self.lower_bound_node_operations,
                self.parameters.edge_indel_cost
                * self.lower_bound_edge_operations,
            ],
            "alignment lower bound",
        )

    @property
    def bound_closed(self) -> bool:
        return (
            self.lower_bound_node_operations == self.node_operation_count
            and self.lower_bound_edge_operations == self.edge_operation_count
        )

    @property
    def solver_mapping_underidentified(self) -> bool:
        return bool(
            self.underidentified_mappings
            or self.representative_traceback_tie_count
        )

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict(include_digest=False))

    def invert(self) -> "GraphAlignment":
        return GraphAlignment(
            left_structural_digest=self.right_structural_digest,
            right_structural_digest=self.left_structural_digest,
            parameters=self.parameters,
            node_matches=tuple(item.invert() for item in self.node_matches),
            edge_matches=tuple(item.invert() for item in self.edge_matches),
            underidentified_mappings=tuple(
                item.invert() for item in self.underidentified_mappings
            ),
            operations=tuple(item.invert() for item in self.operations),
            lower_bound_node_operations=self.lower_bound_node_operations,
            lower_bound_edge_operations=self.lower_bound_edge_operations,
            representative_traceback_tie_count=self.representative_traceback_tie_count,
        )

    def validate_against(
        self,
        left: ComputationGraph | GraphProfile,
        right: ComputationGraph | GraphProfile,
    ) -> dict[str, Any]:
        """Recompute endpoint-derived bounds and verify structural replay."""

        return validate_graph_alignment(left, right, self)

    def _unmatched_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "unmatched_left_edges": [
                item.edge.to_dict()
                for item in self.operations
                if isinstance(item, EdgeDeletion)
            ],
            "unmatched_left_nodes": [
                item.node.to_dict()
                for item in self.operations
                if isinstance(item, NodeDeletion)
            ],
            "unmatched_right_edges": [
                item.edge.to_dict()
                for item in self.operations
                if isinstance(item, EdgeInsertion)
            ],
            "unmatched_right_nodes": [
                item.node.to_dict()
                for item in self.operations
                if isinstance(item, NodeInsertion)
            ],
        }

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "algorithm": self.algorithm,
            "ancestral_interpretation": "none",
            "artifact_type": "phylodigy.graph_alignment",
            "bounds": {
                "closed": self.bound_closed,
                "lower": self.lower_bound,
                "lower_bound_edge_operations": self.lower_bound_edge_operations,
                "lower_bound_method": LOWER_BOUND_METHOD,
                "lower_bound_node_operations": self.lower_bound_node_operations,
                "upper": self.upper_bound,
            },
            "candidate_optimality": "not_asserted_without_endpoint_validation",
            "cost_model": self.parameters.cost_model_dict(),
            "cost_model_digest": self.parameters.cost_model_digest,
            "edge_matches": [item.to_dict() for item in self.edge_matches],
            "left_structural_digest": self.left_structural_digest,
            "node_matches": [item.to_dict() for item in self.node_matches],
            "objective": PRIMITIVE_INDEL_OBJECTIVE,
            "operations": [item.to_dict() for item in self.operations],
            "parameters": self.parameters.to_dict(),
            "replay_direction": {
                "ancestral_interpretation": "none",
                "from_endpoint": "left",
                "to_endpoint": "right",
            },
            "right_structural_digest": self.right_structural_digest,
            "schema_version": self.schema_version,
            "solver_configuration": {
                **self.parameters.solver_configuration_dict(),
                "representative_traceback_tie_count": self.representative_traceback_tie_count,
            },
            "solver_configuration_digest": self.parameters.solver_configuration_digest,
            "solver_mapping_underidentified": self.solver_mapping_underidentified,
            "solver_underidentification_scope": "canonical_sequence_representative_only",
            "underidentified_mappings": [
                item.to_dict() for item in self.underidentified_mappings
            ],
            "total_cost": self.total_cost,
            **self._unmatched_dict(),
        }
        if include_digest:
            result["digest"] = content_digest(result)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphAlignment":
        if not isinstance(raw, Mapping):
            raise TypeError("graph alignment must be a mapping")
        if raw.get("artifact_type") != "phylodigy.graph_alignment":
            raise ValueError("artifact_type must be 'phylodigy.graph_alignment'")
        if raw.get("objective") != PRIMITIVE_INDEL_OBJECTIVE:
            raise ValueError("unsupported primitive indel objective")
        bounds = raw.get("bounds")
        solver = raw.get("solver_configuration")
        if not isinstance(bounds, Mapping) or not isinstance(solver, Mapping):
            raise TypeError("bounds and solver_configuration must be mappings")
        if bounds.get("lower_bound_method") != LOWER_BOUND_METHOD:
            raise ValueError("unsupported lower-bound method")
        if raw.get("ancestral_interpretation") != "none":
            raise ValueError("graph alignment cannot assert ancestral interpretation")
        result = cls(
            left_structural_digest=str(raw["left_structural_digest"]),
            right_structural_digest=str(raw["right_structural_digest"]),
            parameters=AlignmentParameters.from_mapping(raw["parameters"]),
            node_matches=tuple(
                NodeMatch.from_dict(item) for item in raw.get("node_matches", [])
            ),
            edge_matches=tuple(
                EdgeMatch.from_dict(item) for item in raw.get("edge_matches", [])
            ),
            underidentified_mappings=tuple(
                AlignmentAmbiguity.from_dict(item)
                for item in raw.get("underidentified_mappings", [])
            ),
            operations=tuple(
                _operation_from_dict(item) for item in raw.get("operations", [])
            ),
            lower_bound_node_operations=bounds["lower_bound_node_operations"],
            lower_bound_edge_operations=bounds["lower_bound_edge_operations"],
            representative_traceback_tie_count=solver.get(
                "representative_traceback_tie_count", 0
            ),
            algorithm=str(raw.get("algorithm", "")),
            schema_version=str(raw.get("schema_version", "")),
        )
        supplied = normalize_json(raw)
        expected = normalize_json(result.to_dict())
        if set(supplied) != set(expected):
            missing = sorted(set(expected) - set(supplied))
            unexpected = sorted(set(supplied) - set(expected))
            raise ValueError(
                "graph alignment fields mismatch: "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )
        for key in expected:
            if canonical_json(supplied[key]) != canonical_json(expected[key]):
                raise ValueError(f"graph alignment {key} mismatch")
        return result


def _canonicalized_copy(graph: ComputationGraph) -> ComputationGraph:
    """Re-run the graph core's name-free canonical labeling."""

    incoming: dict[str, list[tuple[str, str]]] = {
        node.node_id: [] for node in graph.nodes
    }
    for edge in graph.edges:
        incoming[edge.target].append((edge.source, edge.position))
    prepared = [
        {
            "attributes": normalize_json(node.attributes),
            "inputs": sorted(incoming[node.node_id]),
            "name": node.node_id,
            "observations": normalize_json(node.observations),
            "operation": node.operation,
        }
        for node in graph.nodes
    ]
    by_name = {str(item["name"]): item for item in prepared}
    colors = _canonical_node_colors(prepared, by_name)
    topological = _canonical_topological_order(prepared, by_name, colors)
    identifiers = {
        source: f"n{index:06d}" for index, source in enumerate(topological)
    }
    nodes = tuple(
        GraphNode(
            identifiers[source],
            str(by_name[source]["operation"]),
            by_name[source]["attributes"],
            by_name[source]["observations"],
        )
        for source in topological
    )
    nodes_by_id = {node.node_id: node for node in nodes}
    pending: dict[str, list[tuple[str, str]]] = {}
    for target in topological:
        target_id = identifiers[target]
        for source, position in by_name[target]["inputs"]:
            pending.setdefault(target_id, []).append((identifiers[source], position))
    edges: list[GraphEdge] = []
    for target in sorted(pending):
        target_node = nodes_by_id[target]
        if _operation_is_commutative(target_node.attributes):
            for index, (source, _) in enumerate(sorted(pending[target])):
                edges.append(GraphEdge(source, target, f"operand:{index:06d}"))
        else:
            for source, position in pending[target]:
                edges.append(GraphEdge(source, target, position))
    return ComputationGraph(nodes, tuple(edges), graph.schema_version)


def _as_graph(
    value: ComputationGraph | GraphProfile,
    *,
    require_canonical: bool = False,
) -> ComputationGraph:
    if isinstance(value, GraphProfile):
        graph = value.graph
    elif isinstance(value, ComputationGraph):
        graph = value
    else:
        raise TypeError(
            "graph alignment expects ComputationGraph or GraphProfile objects"
        )
    if require_canonical:
        canonical = _canonicalized_copy(graph)
        if canonical.structural_digest != graph.structural_digest:
            raise ValueError(
                "alignment inputs must use the graph core's canonical node labeling"
            )
    return graph


def _node_token(node: GraphNode) -> str:
    return canonical_json(
        {"attributes": normalize_json(node.attributes), "operation": node.operation}
    )


def _lcs_matches(
    left: Sequence[GraphNode],
    right: Sequence[GraphNode],
    *,
    max_matrix_cells: int,
) -> tuple[list[tuple[int, int]], int]:
    rows = len(left) + 1
    columns = len(right) + 1
    cells = rows * columns
    if cells > max_matrix_cells:
        raise AlignmentResourceLimitError(
            f"alignment matrix requires {cells} cells; budget is {max_matrix_cells}"
        )
    left_tokens = [_node_token(item) for item in left]
    right_tokens = [_node_token(item) for item in right]
    table = [array("I", [0]) * columns for _ in range(rows)]
    for left_index in range(len(left) - 1, -1, -1):
        row = table[left_index]
        next_row = table[left_index + 1]
        for right_index in range(len(right) - 1, -1, -1):
            best = max(next_row[right_index], row[right_index + 1])
            if left_tokens[left_index] == right_tokens[right_index]:
                best = max(best, 1 + next_row[right_index + 1])
            row[right_index] = best

    matches: list[tuple[int, int]] = []
    tie_count = 0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        current = table[left_index][right_index]
        candidates: list[str] = []
        if (
            left_tokens[left_index] == right_tokens[right_index]
            and 1 + table[left_index + 1][right_index + 1] == current
        ):
            candidates.append("match")
        if table[left_index + 1][right_index] == current:
            candidates.append("skip_left")
        if table[left_index][right_index + 1] == current:
            candidates.append("skip_right")
        if len(candidates) > 1:
            tie_count += 1
        # This representative is presentation-only. Endpoint canonicalization
        # plus canonical internal orientation makes it deterministic; ties stay
        # visible in the alignment diagnostics.
        if "match" in candidates:
            matches.append((left_index, right_index))
            left_index += 1
            right_index += 1
        elif "skip_left" in candidates:
            left_index += 1
        else:
            right_index += 1
    return matches, tie_count


def _canonical_port_label(
    edge: GraphEdge,
    nodes: Mapping[str, GraphNode],
) -> str:
    target = nodes[edge.target]
    return _edge_label(target.operation, edge.position, target.attributes)


def _lower_bound_counts(
    left: ComputationGraph,
    right: ComputationGraph,
) -> tuple[int, int]:
    left_nodes = {node.node_id: node for node in left.nodes}
    right_nodes = {node.node_id: node for node in right.nodes}
    left_labels = Counter(_node_token(node) for node in left.nodes)
    right_labels = Counter(_node_token(node) for node in right.nodes)
    possible_node_matches = sum(
        min(left_labels[label], right_labels[label])
        for label in set(left_labels) | set(right_labels)
    )
    node_gap = len(left.nodes) + len(right.nodes) - 2 * possible_node_matches

    def edge_labels(
        graph: ComputationGraph, nodes: Mapping[str, GraphNode]
    ) -> Counter[tuple[str, str, str]]:
        return Counter(
            (
                _node_token(nodes[edge.source]),
                _node_token(nodes[edge.target]),
                _canonical_port_label(edge, nodes),
            )
            for edge in graph.edges
        )

    left_edge_labels = edge_labels(left, left_nodes)
    right_edge_labels = edge_labels(right, right_nodes)
    possible_edge_matches = sum(
        min(left_edge_labels[label], right_edge_labels[label])
        for label in set(left_edge_labels) | set(right_edge_labels)
    )
    edge_gap = len(left.edges) + len(right.edges) - 2 * possible_edge_matches
    return node_gap, edge_gap


def _align_canonical_orientation(
    left: ComputationGraph,
    right: ComputationGraph,
    parameters: AlignmentParameters,
) -> GraphAlignment:
    if left.structural_digest == right.structural_digest:
        pairs = list(zip(range(len(left.nodes)), range(len(right.nodes))))
        tie_count = 0
    else:
        pairs, tie_count = _lcs_matches(
            left.nodes,
            right.nodes,
            max_matrix_cells=parameters.max_matrix_cells,
        )
    left_nodes = {node.node_id: node for node in left.nodes}
    right_nodes = {node.node_id: node for node in right.nodes}
    node_matches = tuple(
        NodeMatch(
            left.nodes[left_index].node_id,
            right.nodes[right_index].node_id,
            StructuralNode.from_graph_node(left.nodes[left_index]).structural_signature,
        )
        for left_index, right_index in pairs
    )
    mapping = {item.left_ref: item.right_ref for item in node_matches}
    matched_left_nodes = set(mapping)
    matched_right_nodes = set(mapping.values())

    operations: list[GraphAlignmentOperation] = []
    for node in left.nodes:
        if node.node_id not in matched_left_nodes:
            operations.append(
                NodeDeletion(
                    StructuralNode.from_graph_node(node), parameters.node_indel_cost
                )
            )
    for node in right.nodes:
        if node.node_id not in matched_right_nodes:
            operations.append(
                NodeInsertion(
                    StructuralNode.from_graph_node(node), parameters.node_indel_cost
                )
            )

    right_edge_candidates: dict[tuple[str, str, str], list[GraphEdge]] = {}
    for edge in right.edges:
        key = (
            edge.source,
            edge.target,
            _canonical_port_label(edge, right_nodes),
        )
        right_edge_candidates.setdefault(key, []).append(edge)
    for candidates in right_edge_candidates.values():
        candidates.sort(key=lambda item: item.edge_id)
    used_right_edges: set[str] = set()
    edge_matches: list[EdgeMatch] = []
    for edge in left.edges:
        candidate: GraphEdge | None = None
        canonical_port = _canonical_port_label(edge, left_nodes)
        if edge.source in mapping and edge.target in mapping:
            key = (mapping[edge.source], mapping[edge.target], canonical_port)
            candidate = next(
                (
                    item
                    for item in right_edge_candidates.get(key, [])
                    if item.edge_id not in used_right_edges
                ),
                None,
            )
        if candidate is None:
            operations.append(
                EdgeDeletion(
                    StructuralEdge.from_graph_edge(edge), parameters.edge_indel_cost
                )
            )
        else:
            used_right_edges.add(candidate.edge_id)
            edge_matches.append(
                EdgeMatch(
                    StructuralEdge.from_graph_edge(edge),
                    StructuralEdge.from_graph_edge(candidate),
                    canonical_port,
                )
            )
    for edge in right.edges:
        if edge.edge_id not in used_right_edges:
            operations.append(
                EdgeInsertion(
                    StructuralEdge.from_graph_edge(edge), parameters.edge_indel_cost
                )
            )

    def local_contexts(
        graph: ComputationGraph,
        nodes: Mapping[str, GraphNode],
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        incoming: dict[str, list[tuple[str, str]]] = {
            node_id: [] for node_id in nodes
        }
        outgoing: dict[str, list[tuple[str, str]]] = {
            node_id: [] for node_id in nodes
        }
        for graph_edge in graph.edges:
            port = _canonical_port_label(graph_edge, nodes)
            incoming[graph_edge.target].append(
                (port, _node_token(nodes[graph_edge.source]))
            )
            outgoing[graph_edge.source].append(
                (port, _node_token(nodes[graph_edge.target]))
            )
        by_ref: dict[str, str] = {}
        groups: dict[str, list[str]] = {}
        for node_id, node in nodes.items():
            signature = content_digest(
                {
                    "incoming": sorted(incoming[node_id]),
                    "node": _node_token(node),
                    "outgoing": sorted(outgoing[node_id]),
                }
            )
            by_ref[node_id] = signature
            groups.setdefault(signature, []).append(node_id)
        return by_ref, groups

    left_context_by_ref, left_context_groups = local_contexts(
        left, left_nodes
    )
    right_context_by_ref, right_context_groups = local_contexts(
        right, right_nodes
    )
    matched_context_counts = Counter(
        left_context_by_ref[item.left_ref]
        for item in node_matches
        if left_context_by_ref[item.left_ref]
        == right_context_by_ref[item.right_ref]
    )
    underidentified = tuple(
        AlignmentAmbiguity(
            signature,
            tuple(left_context_groups[signature]),
            tuple(right_context_groups[signature]),
            matched_context_counts[signature],
        )
        for signature in sorted(
            set(left_context_groups) & set(right_context_groups)
        )
        if len(left_context_groups[signature]) > 1
        or len(right_context_groups[signature]) > 1
    )
    lower_node_count, lower_edge_count = _lower_bound_counts(left, right)
    alignment = GraphAlignment(
        left_structural_digest=left.structural_digest,
        right_structural_digest=right.structural_digest,
        parameters=parameters,
        node_matches=node_matches,
        edge_matches=tuple(edge_matches),
        underidentified_mappings=underidentified,
        operations=tuple(operations),
        lower_bound_node_operations=lower_node_count,
        lower_bound_edge_operations=lower_edge_count,
        representative_traceback_tie_count=tie_count,
    )
    validate_graph_alignment(left, right, alignment)
    return alignment


def align_graphs(
    left: ComputationGraph | GraphProfile,
    right: ComputationGraph | GraphProfile,
    *,
    parameters: AlignmentParameters | Mapping[str, Any] | None = None,
) -> GraphAlignment:
    """Return a deterministic, replayable primitive graph alignment.

    The endpoint order determines only the mechanical replay direction. The
    solver always works in digest order and inverts afterward, guaranteeing
    that ``align_graphs(b, a)`` is the exact inverse of ``align_graphs(a, b)``.
    """

    left_graph = _as_graph(left, require_canonical=True)
    right_graph = _as_graph(right, require_canonical=True)
    if parameters is None:
        selected = AlignmentParameters()
    elif isinstance(parameters, AlignmentParameters):
        selected = parameters
    elif isinstance(parameters, Mapping):
        selected = AlignmentParameters.from_mapping(parameters)
    else:
        raise TypeError("parameters must be AlignmentParameters, a mapping, or None")
    if left_graph.structural_digest <= right_graph.structural_digest:
        return _align_canonical_orientation(left_graph, right_graph, selected)
    return _align_canonical_orientation(right_graph, left_graph, selected).invert()


def apply_graph_alignment(
    graph: ComputationGraph | GraphProfile,
    alignment: GraphAlignment,
) -> ComputationGraph:
    """Replay ``alignment`` and return an observation-free target structure."""

    if not isinstance(alignment, GraphAlignment):
        raise TypeError("alignment must be a GraphAlignment")
    source = _as_graph(graph, require_canonical=True)
    if source.structural_digest != alignment.left_structural_digest:
        raise ValueError("alignment left endpoint does not match the source graph")
    source_nodes = {node.node_id: node for node in source.nodes}
    mapping = {item.left_ref: item.right_ref for item in alignment.node_matches}
    deletion_nodes = {
        item.node.ref: item.node
        for item in alignment.operations
        if isinstance(item, NodeDeletion)
    }
    insertion_nodes = {
        item.node.ref: item.node
        for item in alignment.operations
        if isinstance(item, NodeInsertion)
    }
    if set(source_nodes) != set(mapping) | set(deletion_nodes):
        raise ValueError(
            "alignment must match or remove every source node exactly once"
        )
    if set(mapping) & set(deletion_nodes):
        raise ValueError("a source node cannot be both matched and removed")
    for ref, payload in deletion_nodes.items():
        if ref not in source_nodes:
            raise ValueError("node removal references a missing source node")
        if StructuralNode.from_graph_node(source_nodes[ref]) != payload:
            raise ValueError("node removal payload does not match the source graph")

    target_nodes: dict[str, GraphNode] = {}
    signature_by_source = {
        ref: StructuralNode.from_graph_node(node).structural_signature
        for ref, node in source_nodes.items()
    }
    for match in alignment.node_matches:
        source_node = source_nodes.get(match.left_ref)
        if source_node is None:
            raise ValueError("node match references a missing source node")
        if signature_by_source[match.left_ref] != match.structural_signature:
            raise ValueError("node match signature does not match the source node")
        if match.right_ref in target_nodes:
            raise ValueError("node matches must produce unique target refs")
        target_nodes[match.right_ref] = GraphNode(
            match.right_ref,
            source_node.operation,
            source_node.attributes,
        )
    for ref, payload in insertion_nodes.items():
        if ref in target_nodes:
            raise ValueError("node addition collides with a matched target ref")
        target_nodes[ref] = GraphNode(ref, payload.operation, payload.attributes)

    deletion_edges = {
        item.edge.ref: item.edge
        for item in alignment.operations
        if isinstance(item, EdgeDeletion)
    }
    insertion_edges = {
        item.edge.ref: item.edge
        for item in alignment.operations
        if isinstance(item, EdgeInsertion)
    }
    source_edges = {
        edge.edge_id: StructuralEdge.from_graph_edge(edge) for edge in source.edges
    }
    if not set(deletion_edges).issubset(source_edges):
        raise ValueError("edge removal references a missing source edge")
    for ref, payload in deletion_edges.items():
        if source_edges[ref] != payload:
            raise ValueError("edge removal payload does not match the source graph")

    matched_left_edges = {item.left.ref for item in alignment.edge_matches}
    if set(source_edges) != set(deletion_edges) | matched_left_edges:
        raise ValueError("alignment must match or remove every source edge exactly once")
    if set(deletion_edges) & matched_left_edges:
        raise ValueError("a source edge cannot be both matched and removed")
    target_edges: dict[str, GraphEdge] = {}
    for match in alignment.edge_matches:
        source_payload = source_edges.get(match.left.ref)
        if source_payload is None or source_payload != match.left:
            raise ValueError("edge match does not resolve against the source graph")
        if (
            mapping.get(match.left.source) != match.right.source
            or mapping.get(match.left.target) != match.right.target
        ):
            raise ValueError("edge match endpoints disagree with the node mapping")
        source_edge = GraphEdge(
            match.left.source, match.left.target, match.left.position
        )
        source_port = _canonical_port_label(source_edge, source_nodes)
        if source_port != match.canonical_port_label:
            raise ValueError("edge match canonical port label is stale")
        target_edge = GraphEdge(
            match.right.source, match.right.target, match.right.position
        )
        target_port = _canonical_port_label(target_edge, target_nodes)
        if target_port != match.canonical_port_label:
            raise ValueError("edge match target port is not structurally equivalent")
        if target_edge.edge_id in target_edges:
            raise ValueError("edge matches must produce unique target edges")
        target_edges[target_edge.edge_id] = target_edge
    for payload in insertion_edges.values():
        if payload.source not in target_nodes or payload.target not in target_nodes:
            raise ValueError("edge addition references a missing target node")
        edge = GraphEdge(payload.source, payload.target, payload.position)
        if edge.edge_id in target_edges:
            raise ValueError("edge addition duplicates an existing target edge")
        target_edges[edge.edge_id] = edge

    result = ComputationGraph(
        nodes=tuple(target_nodes.values()),
        edges=tuple(target_edges.values()),
        schema_version=source.schema_version,
    )
    if result.structural_digest != alignment.right_structural_digest:
        raise ValueError("replayed alignment does not produce its pinned target graph")
    _as_graph(result, require_canonical=True)
    return result


def validate_graph_alignment(
    left: ComputationGraph | GraphProfile,
    right: ComputationGraph | GraphProfile,
    alignment: GraphAlignment,
) -> dict[str, Any]:
    """Validate endpoint-derived bounds and replay, then report scoped proof."""

    if not isinstance(alignment, GraphAlignment):
        raise TypeError("alignment must be a GraphAlignment")
    left_graph = _as_graph(left, require_canonical=True)
    right_graph = _as_graph(right, require_canonical=True)
    if left_graph.structural_digest != alignment.left_structural_digest:
        raise ValueError("alignment does not bind the supplied left endpoint")
    if right_graph.structural_digest != alignment.right_structural_digest:
        raise ValueError("alignment does not bind the supplied right endpoint")
    lower_node_count, lower_edge_count = _lower_bound_counts(
        left_graph, right_graph
    )
    if (
        lower_node_count != alignment.lower_bound_node_operations
        or lower_edge_count != alignment.lower_bound_edge_operations
    ):
        raise ValueError("alignment lower-bound components do not match endpoints")
    replayed = apply_graph_alignment(left_graph, alignment)
    if replayed.structural_digest != right_graph.structural_digest:
        raise ValueError("alignment replay does not reproduce the right endpoint")
    result = {
        "alignment_digest": alignment.digest,
        "bound_components_validated": True,
        "endpoints_validated": True,
        "objective": PRIMITIVE_INDEL_OBJECTIVE,
        "objective_optimality_proven": alignment.bound_closed,
        "replay_validated": True,
    }
    result["digest"] = content_digest(result)
    return result


SolverMappingUnderidentification = AlignmentAmbiguity


__all__ = [
    "AddEdgeOperation",
    "AddNodeOperation",
    "AlignmentParameters",
    "AlignmentResourceLimitError",
    "COMPARISON_ORIENTATION",
    "EdgeMatch",
    "GRAPH_ALIGNMENT_ALGORITHM",
    "GRAPH_ALIGNMENT_SCHEMA_VERSION",
    "GraphAlignment",
    "GraphAlignmentOperation",
    "LOWER_BOUND_METHOD",
    "NodeMatch",
    "PRIMITIVE_INDEL_OBJECTIVE",
    "RemoveEdgeOperation",
    "RemoveNodeOperation",
    "SolverMappingUnderidentification",
    "StructuralEdge",
    "StructuralNode",
    "align_graphs",
    "apply_graph_alignment",
    "validate_graph_alignment",
]
