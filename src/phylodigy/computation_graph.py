"""Framework-neutral architectural genomes derived only from computation graphs.

The executable graph is the structural observation.  There is no feature
catalog, architecture vocabulary, or named architecture detector in this
module.  Framework frontends lower nodes into FX-like records; unknown
operations are retained under an open canonical label.  Generic algorithms
then discover anonymous neighborhoods and fork/join regions with stable,
content-derived identifiers.

The multiscale fingerprint and its weighted-L1 distance are scalable alignment
signals.  They are a metric on fingerprint vectors and a *pseudometric* on
graphs: different graphs can have the same Weisfeiler-Lehman features.  They are
not a substitute for the event-weighted graph edit model intended for final
phylogenetic branch lengths.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .canonical import canonical_json, content_digest, freeze_json, normalize_json


GRAPH_SCHEMA_VERSION = "2.0.0"
GRAPH_PROFILE_VERSION = "2"

__all__ = [
    "GRAPH_PROFILE_VERSION",
    "GRAPH_SCHEMA_VERSION",
    "ComputationGraph",
    "FingerprintScale",
    "GraphCharacter",
    "GraphCharacterOccurrence",
    "GraphDistance",
    "GraphEdge",
    "GraphProfile",
    "GraphRegion",
    "GraphNode",
    "StructuralFingerprint",
    "build_graph_profile",
    "compare_graphs",
    "computation_graph_from_fx",
    "computation_graph_from_records",
    "discover_graph_characters",
    "discover_graph_regions",
    "normalize_operator",
    "structural_fingerprint",
]


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _normalize_radii(radii: Iterable[int]) -> tuple[int, ...]:
    values = tuple(radii)
    if any(isinstance(radius, bool) or not isinstance(radius, int) for radius in values):
        raise TypeError("radii must contain integers")
    requested = tuple(sorted(set(values)))
    if not requested or requested[0] < 0:
        raise ValueError("radii must contain non-negative integers")
    return requested


def _character_id(
    generator: str,
    structural_signature: str,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    return content_digest(
        {
            "generator": generator,
            "parameters": normalize_json(dict(parameters or {})),
            "structural_signature": structural_signature,
        }
    )


def _occurrence_id(
    character_id: str,
    node_ids: Iterable[str],
    edge_ids: Iterable[str],
) -> str:
    return content_digest(
        {
            "character_id": character_id,
            "edge_ids": sorted(set(edge_ids)),
            "node_ids": sorted(set(node_ids)),
        }
    )


def _operator_token(value: Any) -> str:
    """Return a stable, open-ended spelling for a framework target."""

    text = str(value or "").strip().lower()
    # Callable reprs otherwise contain process-specific addresses.
    text = re.sub(r"\s+at\s+0x[0-9a-f]+", "", text)
    text = text.strip("<>")
    text = re.sub(r"^(?:built-in function|built-in method|function|method)\s+", "", text)
    text = text.replace("::", ".").replace("/", ".")
    text = re.sub(r"[^a-z0-9_]+", ".", text)
    text = re.sub(r"\.+", ".", text).strip(".")
    return text or "unknown"


def normalize_operator(
    op: Any,
    target: Any | None = None,
    module_type: Any | None = None,
) -> str:
    """Lower a framework operation without classifying an architecture.

    The vocabulary is open: unrecognized targets survive as normalized labels.
    No target spelling is translated into a project-maintained semantic class.
    """

    kind = _operator_token(op)
    target_token = _operator_token(target)
    module_token = _operator_token(module_type)
    if kind == "placeholder":
        return "graph.input"
    if kind == "output":
        return "graph.output"
    if kind == "get_attr":
        # Attribute paths are program names, not structural identities.
        return "data.attribute"
    if kind == "call_module":
        if module_type is None or module_token == "unknown":
            return "module.opaque"
        torch_module = re.fullmatch(r"torch\.nn\.modules\.[^.]+\.(.+)", module_token)
        if torch_module:
            # torch.nn.Linear and its implementation module path denote the
            # same public primitive across PyTorch releases.
            module_token = f"torch.nn.{torch_module.group(1)}"
        return f"module.{module_token}"
    if kind == "call_function":
        return f"function.{target_token}"
    if kind == "call_method":
        return f"method.{target_token}"
    if target is None:
        return f"operation.{kind}"
    return f"operation.{kind}.{target_token}"


@dataclass(frozen=True)
class GraphNode:
    """One canonical operation node."""

    node_id: str
    operation: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    observations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _nonempty(self.node_id, "node_id"))
        object.__setattr__(self, "operation", _nonempty(self.operation, "operation"))
        normalized = normalize_json(dict(self.attributes))
        if not isinstance(normalized, dict):  # pragma: no cover - defensive
            raise TypeError("node attributes must normalize to an object")
        observed = normalize_json(dict(self.observations))
        if not isinstance(observed, dict):  # pragma: no cover - defensive
            raise TypeError("node observations must normalize to an object")
        object.__setattr__(self, "attributes", freeze_json(normalized))
        object.__setattr__(self, "observations", freeze_json(observed))

    def to_dict(self) -> dict[str, Any]:
        result = {"id": self.node_id, "operation": self.operation}
        if self.attributes:
            result["attributes"] = normalize_json(self.attributes)
        if self.observations:
            result["observations"] = normalize_json(self.observations)
        return result

    def structural_dict(self) -> dict[str, Any]:
        """Return the character-bearing content, excluding trace observations."""

        result = {"id": self.node_id, "operation": self.operation}
        if self.attributes:
            result["attributes"] = normalize_json(self.attributes)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphNode":
        return cls(
            node_id=str(raw["id"]),
            operation=str(raw["operation"]),
            attributes=raw.get("attributes", {}),
            observations=raw.get("observations", {}),
        )


@dataclass(frozen=True)
class GraphEdge:
    """A directed data-flow edge with a stable input position."""

    source: str
    target: str
    position: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _nonempty(self.source, "edge source"))
        object.__setattr__(self, "target", _nonempty(self.target, "edge target"))
        object.__setattr__(self, "position", _nonempty(self.position, "edge position"))

    @property
    def edge_id(self) -> str:
        return f"{self.source}->{self.target}@{self.position}"

    def to_dict(self) -> dict[str, str]:
        return {
            "position": self.position,
            "source": self.source,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphEdge":
        return cls(
            source=str(raw["source"]),
            target=str(raw["target"]),
            position=str(raw["position"]),
        )


@dataclass(frozen=True)
class ComputationGraph:
    """A canonical, directed computation graph."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    schema_version: str = GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GRAPH_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {GRAPH_SCHEMA_VERSION!r}, "
                f"got {self.schema_version!r}"
            )
        ordered_nodes = tuple(sorted(self.nodes, key=lambda item: item.node_id))
        node_ids = [item.node_id for item in ordered_nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node IDs must be unique")
        known = set(node_ids)
        ordered_edges = tuple(
            sorted(self.edges, key=lambda item: (item.source, item.target, item.position))
        )
        edge_ids = [item.edge_id for item in ordered_edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("graph edges must be unique by source, target, and position")
        for edge in ordered_edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"edge references an unknown node: {edge.edge_id}")
        object.__setattr__(self, "nodes", ordered_nodes)
        object.__setattr__(self, "edges", ordered_edges)
        _assert_acyclic(self)

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict())

    @property
    def structural_digest(self) -> str:
        """Digest used by character comparison, excluding probe observations."""

        return content_digest(
            {
                "edges": [item.to_dict() for item in self.edges],
                "nodes": [item.structural_dict() for item in self.nodes],
                "schema_version": self.schema_version,
            }
        )

    def canonical_json(self, *, indent: int | None = None) -> str:
        return canonical_json(self.to_dict(), indent=indent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": [item.to_dict() for item in self.edges],
            "nodes": [item.to_dict() for item in self.nodes],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ComputationGraph":
        return cls(
            nodes=tuple(GraphNode.from_dict(item) for item in raw.get("nodes", [])),
            edges=tuple(GraphEdge.from_dict(item) for item in raw.get("edges", [])),
            schema_version=str(raw.get("schema_version", GRAPH_SCHEMA_VERSION)),
        )


def _assert_acyclic(graph: ComputationGraph) -> None:
    successors: dict[str, list[str]] = {node.node_id: [] for node in graph.nodes}
    indegree = {node.node_id: 0 for node in graph.nodes}
    for edge in graph.edges:
        successors[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in sorted(successors[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(graph.nodes):
        raise ValueError("computation graph must be acyclic")


def _record_inputs(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    explicit = record.get("input_edges")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        result: list[tuple[str, str]] = []
        for index, raw in enumerate(explicit):
            if isinstance(raw, Mapping):
                source = raw.get("source", raw.get("name", ""))
                position = raw.get("position", raw.get("slot", f"arg:{index:06d}"))
            else:
                source = raw
                position = f"arg:{index:06d}"
            result.append((str(source), str(position)))
        return result
    inputs = record.get("inputs", ())
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise TypeError("record inputs must be a sequence")
    return [(str(source), f"arg:{index:06d}") for index, source in enumerate(inputs)]


def _operation_is_commutative(attributes: Mapping[str, Any]) -> bool:
    properties = attributes.get("operator_properties", {})
    return isinstance(properties, Mapping) and properties.get("commutative") is True


def _edge_label(
    target_operation: str,
    position: str,
    attributes: Mapping[str, Any] | None = None,
) -> str:
    """Return an input-port label using only frontend-declared algebra.

    ``target_operation`` remains an argument to keep call sites explicit about
    which endpoint owns the port. The core contains no operator-name table;
    commutativity is honored only when the graph record declares
    ``attributes.operator_properties.commutative``.
    """

    del target_operation
    if _operation_is_commutative(attributes or {}):
        return "operand"
    return position


def _canonical_node_colors(
    prepared: Sequence[Mapping[str, Any]],
    by_name: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Refine name-free colors from both sides until the partition is stable.

    A single ancestor pass followed by a descendant pass misses correlations
    that cross the two directions.  Synchronous refinement propagates those
    distinctions until no color cell can be split further.  Remaining cells
    are resolved by :func:`_canonical_topological_order`, never by record
    names, indices, or emission order.
    """

    incoming: dict[str, list[tuple[str, str]]] = {
        str(item["name"]): list(item["inputs"]) for item in prepared
    }
    outgoing: dict[str, list[tuple[str, str]]] = {
        str(item["name"]): [] for item in prepared
    }
    base: dict[str, Any] = {}
    for item in prepared:
        name = str(item["name"])
        base[name] = {
            "attributes": item["attributes"],
            "operation": item["operation"],
        }
        for source, position in item["inputs"]:
            outgoing[source].append(
                (
                    _edge_label(
                        str(item["operation"]), position, item["attributes"]
                    ),
                    name,
                )
            )

    def ranks(signatures: Mapping[str, Any]) -> dict[str, str]:
        encoded = {name: canonical_json(value) for name, value in signatures.items()}
        rank_by_signature = {
            signature: f"c{index:06d}"
            for index, signature in enumerate(sorted(set(encoded.values())))
        }
        return {name: rank_by_signature[encoded[name]] for name in encoded}

    colors = ranks(base)
    for _ in range(len(prepared)):
        signatures = {}
        for name, item in by_name.items():
            signatures[name] = {
                "base": base[name],
                "color": colors[name],
                "in": sorted(
                    (
                        _edge_label(
                            str(item["operation"]), position, item["attributes"]
                        ),
                        colors[source],
                    )
                    for source, position in incoming[name]
                ),
                "out": sorted(
                    (position, colors[target])
                    for position, target in outgoing[name]
                ),
            }
        refined = ranks(signatures)
        # Including the previous color makes refinement monotone.  Once the
        # number of cells stops growing, the partition is stable.
        if len(set(refined.values())) == len(set(colors.values())):
            return refined
        colors = refined
    return colors


def _canonical_topological_order(
    prepared: Sequence[Mapping[str, Any]],
    by_name: Mapping[str, Mapping[str, Any]],
    colors: Mapping[str, str],
) -> tuple[str, ...]:
    """Return an emission-order-independent canonical topological labeling.

    Stable color refinement resolves almost all model graphs.  When distinct,
    non-automorphic nodes remain in the same ready color cell, this function
    explicitly explores the tied choices and selects the lexicographically
    smallest structural serialization.  Exact twins are safely collapsed and
    ordered by observations, avoiding factorial work for wide symmetric fans.
    The bounded search fails closed instead of silently using source order.
    """

    incoming: dict[str, list[tuple[str, str]]] = {
        str(item["name"]): list(item["inputs"]) for item in prepared
    }
    outgoing: dict[str, list[tuple[str, str]]] = {
        str(item["name"]): [] for item in prepared
    }
    successors: dict[str, set[str]] = {name: set() for name in by_name}
    indegree = {name: 0 for name in by_name}
    for item in prepared:
        target = str(item["name"])
        unique_sources = {source for source, _ in item["inputs"]}
        indegree[target] = len(unique_sources)
        for source, position in item["inputs"]:
            label = _edge_label(
                str(item["operation"]), position, item["attributes"]
            )
            outgoing[source].append((label, target))
            successors[source].add(target)

    def certificate(order: Sequence[str]) -> tuple[str, str]:
        identifiers = {name: index for index, name in enumerate(order)}
        nodes = [
            {
                "attributes": by_name[name]["attributes"],
                "operation": by_name[name]["operation"],
            }
            for name in order
        ]
        edges: list[tuple[int, int, str]] = []
        for target in order:
            target_index = identifiers[target]
            item = by_name[target]
            raw_edges = [
                (identifiers[source], target_index, position)
                for source, position in item["inputs"]
            ]
            if _operation_is_commutative(item["attributes"]):
                for operand, (source_index, _, _) in enumerate(sorted(raw_edges)):
                    edges.append(
                        (source_index, target_index, f"operand:{operand:06d}")
                    )
            else:
                edges.extend(raw_edges)
        structural = canonical_json({"edges": sorted(edges), "nodes": nodes})
        observed = canonical_json(
            [by_name[name]["observations"] for name in order]
        )
        return structural, observed

    def exact_twin_key(name: str) -> tuple[Any, ...]:
        return (
            canonical_json(by_name[name]["attributes"]),
            str(by_name[name]["operation"]),
            tuple(sorted(incoming[name])),
            tuple(sorted(outgoing[name])),
        )

    best: tuple[tuple[str, str], tuple[str, ...]] | None = None
    explored = 0
    search_limit = 100_000

    def visit(
        prefix: list[str],
        remaining_indegree: dict[str, int],
        ready: set[str],
    ) -> None:
        nonlocal best, explored
        while ready:
            minimum_color = min(colors[name] for name in ready)
            tied = [name for name in ready if colors[name] == minimum_color]

            # Members with literally identical structural neighborhoods are
            # related by an obvious automorphism.  Only their observation order
            # can affect the full (non-structural) graph serialization.
            representatives: dict[tuple[Any, ...], str] = {}
            for name in tied:
                key = exact_twin_key(name)
                current = representatives.get(key)
                if current is None or (
                    canonical_json(by_name[name]["observations"]), name
                ) < (
                    canonical_json(by_name[current]["observations"]), current
                ):
                    representatives[key] = name
            choices = sorted(representatives.values())
            if len(choices) > 1:
                for choice in choices:
                    explored += 1
                    if explored > search_limit:
                        raise ValueError(
                            "graph canonical labeling exceeded its ambiguity "
                            "budget; provide additional executable structure"
                        )
                    next_prefix = prefix + [choice]
                    next_indegree = dict(remaining_indegree)
                    next_ready = set(ready)
                    next_ready.remove(choice)
                    for target in successors[choice]:
                        next_indegree[target] -= 1
                        if next_indegree[target] == 0:
                            next_ready.add(target)
                    visit(next_prefix, next_indegree, next_ready)
                return

            choice = choices[0]
            prefix.append(choice)
            ready.remove(choice)
            for target in successors[choice]:
                remaining_indegree[target] -= 1
                if remaining_indegree[target] == 0:
                    ready.add(target)

        if len(prefix) != len(prepared):
            raise ValueError("FX-like records must describe an acyclic graph")
        candidate = (certificate(prefix), tuple(prefix))
        if best is None or candidate[0] < best[0]:
            best = candidate

    visit(
        [],
        dict(indegree),
        {name for name, degree in indegree.items() if degree == 0},
    )
    if best is None:  # pragma: no cover - empty ready set is a cycle
        raise ValueError("FX-like records must describe an acyclic graph")
    return best[1]


def computation_graph_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    include_tensor_metadata: bool = True,
) -> ComputationGraph:
    """Build a canonical graph from open, frontend-neutral node records.

    Required fields are ``name``, ``op``, ``target`` and ``inputs`` (or
    ``input_edges``).  ``target_type`` should be supplied for ``call_module``
    records so module-path renames cannot affect operation identity.
    """

    prepared: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TypeError("graph records must be mappings")
        name = _nonempty(str(raw.get("name", "")), "record name")
        if name in by_name:
            raise ValueError(f"duplicate graph record name: {name!r}")
        raw_attributes = raw.get("attributes", {})
        if not isinstance(raw_attributes, Mapping):
            raise TypeError("graph record attributes must be a mapping")
        normalized_attributes = normalize_json(dict(raw_attributes))
        if not isinstance(normalized_attributes, dict):  # pragma: no cover
            raise TypeError("graph record attributes must normalize to an object")
        attributes: dict[str, Any] = normalized_attributes
        raw_observations = raw.get("observations", {})
        if not isinstance(raw_observations, Mapping):
            raise TypeError("graph record observations must be a mapping")
        normalized_observations = normalize_json(dict(raw_observations))
        if not isinstance(normalized_observations, dict):  # pragma: no cover
            raise TypeError("graph record observations must normalize to an object")
        observations: dict[str, Any] = normalized_observations
        if include_tensor_metadata and raw.get("tensor_meta") is not None:
            observations["tensor_meta"] = normalize_json(raw.get("tensor_meta"))
        item = {
            "name": name,
            "operation": normalize_operator(
                raw.get("op", "operation"),
                raw.get("target"),
                raw.get("target_type", raw.get("module_type")),
            ),
            "inputs": _record_inputs(raw),
            "attributes": attributes,
            "observations": observations,
        }
        prepared.append(item)
        by_name[name] = item

    for item in prepared:
        for source, _ in item["inputs"]:
            if source not in by_name:
                raise ValueError(
                    f"graph record {item['name']!r} references unknown input {source!r}"
                )

    colors = _canonical_node_colors(prepared, by_name)
    topological = _canonical_topological_order(prepared, by_name, colors)

    canonical_ids = {
        source_name: f"n{index:06d}" for index, source_name in enumerate(topological)
    }
    nodes: list[GraphNode] = []
    for source_name in topological:
        item = by_name[source_name]
        nodes.append(
            GraphNode(
                node_id=canonical_ids[source_name],
                operation=item["operation"],
                attributes=item["attributes"],
                observations=item["observations"],
            )
        )

    nodes_by_id = {node.node_id: node for node in nodes}
    pending_edges: list[tuple[str, str, str]] = []
    for target_name in topological:
        target_id = canonical_ids[target_name]
        for source_name, position in by_name[target_name]["inputs"]:
            pending_edges.append((canonical_ids[source_name], target_id, position))

    edges: list[GraphEdge] = []
    by_target: dict[str, list[tuple[str, str]]] = {}
    for source, target, position in pending_edges:
        by_target.setdefault(target, []).append((source, position))
    for target in sorted(by_target):
        incoming = by_target[target]
        if _operation_is_commutative(nodes_by_id[target].attributes):
            # Argument order for a commutative primitive is incidental.
            for index, (source, _) in enumerate(sorted(incoming)):
                edges.append(GraphEdge(source, target, f"operand:{index:06d}"))
        else:
            for source, position in incoming:
                edges.append(GraphEdge(source, target, position))
    return ComputationGraph(nodes=tuple(nodes), edges=tuple(edges))


def computation_graph_from_fx(
    records: Iterable[Mapping[str, Any]],
    *,
    include_tensor_metadata: bool = True,
) -> ComputationGraph:
    """Compatibility alias for the generic record frontend."""

    return computation_graph_from_records(
        records, include_tensor_metadata=include_tensor_metadata
    )


@dataclass(frozen=True)
class GraphRegion:
    """One anonymous fork/join occurrence discovered from topology alone.

    ``structural_signature`` deliberately excludes graph-local node IDs.  It is
    therefore shared by structurally identical occurrences in other artifacts.
    ``occurrence_id`` includes the concrete support and distinguishes repeated
    copies inside one graph.  Human terminology belongs in paper annotations,
    never in this object.
    """

    entry_id: str
    exit_id: str
    branch_paths: tuple[tuple[str, ...], ...]
    support_edges: tuple[str, ...]
    structural_signature: str
    discovery: str = "fork_join.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_id", _nonempty(self.entry_id, "region entry"))
        object.__setattr__(self, "exit_id", _nonempty(self.exit_id, "region exit"))
        discovery = _nonempty(self.discovery, "discovery")
        if discovery != "fork_join.v1":
            raise ValueError("discovery must be 'fork_join.v1' for this schema")
        object.__setattr__(self, "discovery", discovery)
        paths = tuple(sorted((tuple(path) for path in self.branch_paths), key=lambda p: p))
        if len(paths) < 2:
            raise ValueError("a fork/join region requires at least two branches")
        if any(len(path) < 2 for path in paths):
            raise ValueError("every fork/join branch must contain an edge")
        if any(path[0] != self.entry_id or path[-1] != self.exit_id for path in paths):
            raise ValueError("every branch must run from the region entry to its exit")
        signature = str(self.structural_signature)
        if re.fullmatch(r"[0-9a-f]{64}", signature) is None:
            raise ValueError("structural_signature must be a SHA-256 digest")
        object.__setattr__(self, "branch_paths", paths)
        object.__setattr__(self, "support_edges", tuple(sorted(set(self.support_edges))))
        object.__setattr__(self, "structural_signature", signature)

    @property
    def character_id(self) -> str:
        return _character_id(
            "fork_join_region.v1", self.structural_signature
        )

    @property
    def occurrence_id(self) -> str:
        return _occurrence_id(
            self.character_id,
            self.support_nodes,
            self.support_edges,
        )

    @property
    def support_nodes(self) -> tuple[str, ...]:
        return tuple(sorted({node for path in self.branch_paths for node in path}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_paths": [list(path) for path in self.branch_paths],
            "character_id": self.character_id,
            "discovery": self.discovery,
            "entry_id": self.entry_id,
            "exit_id": self.exit_id,
            "id": self.occurrence_id,
            "structural_signature": self.structural_signature,
            "support_edges": list(self.support_edges),
            "support_nodes": list(self.support_nodes),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphRegion":
        result = cls(
            entry_id=str(raw["entry_id"]),
            exit_id=str(raw["exit_id"]),
            branch_paths=tuple(
                tuple(str(node) for node in path) for path in raw["branch_paths"]
            ),
            support_edges=tuple(str(edge) for edge in raw.get("support_edges", [])),
            structural_signature=str(raw["structural_signature"]),
            discovery=str(raw.get("discovery", "fork_join.v1")),
        )
        supplied = raw.get("id")
        if supplied is not None and str(supplied) != result.occurrence_id:
            raise ValueError("graph region occurrence ID does not match its content")
        supplied_character = raw.get("character_id")
        if supplied_character is not None and str(supplied_character) != result.character_id:
            raise ValueError("graph region character ID does not match its signature")
        return result


def _graph_maps(
    graph: ComputationGraph,
) -> tuple[dict[str, GraphNode], dict[str, list[GraphEdge]], dict[str, list[GraphEdge]]]:
    nodes = {node.node_id: node for node in graph.nodes}
    incoming = {node_id: [] for node_id in nodes}
    outgoing = {node_id: [] for node_id in nodes}
    for edge in graph.edges:
        incoming[edge.target].append(edge)
        outgoing[edge.source].append(edge)
    for values in (incoming, outgoing):
        for node_id in values:
            values[node_id].sort(key=lambda edge: edge.edge_id)
    return nodes, incoming, outgoing


def _is_data_flow_edge(edge: GraphEdge) -> bool:
    """Return whether an edge carries executable values rather than a resource.

    Framework frontends may add generic ``resource:*`` incidence edges to
    preserve parameter/module sharing.  Those edges are structural characters,
    but they are not alternate tensor paths and must not fabricate fork/join
    regions.
    """

    return not edge.position.startswith("resource:")


def _ancestor_distances(
    start: str, incoming: Mapping[str, Sequence[GraphEdge]]
) -> dict[str, int]:
    distances = {start: 0}
    queue = deque([start])
    while queue:
        target = queue.popleft()
        for edge in incoming[target]:
            candidate = distances[target] + 1
            if edge.source not in distances or candidate < distances[edge.source]:
                distances[edge.source] = candidate
                queue.append(edge.source)
    return distances


def _shortest_path(
    source: str,
    target: str,
    outgoing: Mapping[str, Sequence[GraphEdge]],
) -> tuple[str, ...] | None:
    if source == target:
        return (source,)
    queue = deque([source])
    previous: dict[str, str | None] = {source: None}
    while queue:
        current = queue.popleft()
        for edge in outgoing[current]:
            if edge.target in previous:
                continue
            previous[edge.target] = current
            if edge.target == target:
                path = [target]
                while path[-1] != source:
                    parent = previous[path[-1]]
                    if parent is None:  # pragma: no cover - source handled above
                        return None
                    path.append(parent)
                return tuple(reversed(path))
            queue.append(edge.target)
    return None


def _path_edge_ids(path: Sequence[str], edges: Sequence[GraphEdge]) -> list[str]:
    by_pair: dict[tuple[str, str], list[GraphEdge]] = {}
    for edge in edges:
        if not _is_data_flow_edge(edge):
            continue
        by_pair.setdefault((edge.source, edge.target), []).append(edge)
    result: list[str] = []
    for source, target in zip(path, path[1:]):
        candidates = sorted(by_pair[(source, target)], key=lambda edge: edge.edge_id)
        result.append(candidates[0].edge_id)
    return result


def _region_signature(
    graph: ComputationGraph,
    branch_paths: Sequence[Sequence[str]],
) -> str:
    """Hash region content without using graph-local node identifiers."""

    nodes, _, _ = _graph_maps(graph)
    edge_by_pair: dict[tuple[str, str], list[GraphEdge]] = {}
    for edge in graph.edges:
        if not _is_data_flow_edge(edge):
            continue
        edge_by_pair.setdefault((edge.source, edge.target), []).append(edge)
    path_signatures: list[list[Any]] = []
    for path in branch_paths:
        encoded: list[Any] = []
        for index, node_id in enumerate(path):
            node = nodes[node_id]
            item: dict[str, Any] = {
                "attributes": node.attributes,
                "operation": node.operation,
            }
            if index:
                source = path[index - 1]
                candidates = sorted(
                    edge_by_pair[(source, node_id)], key=lambda edge: edge.edge_id
                )
                item["input_port"] = _edge_label(
                    node.operation, candidates[0].position, node.attributes
                )
            encoded.append(item)
        path_signatures.append(encoded)
    # The branches are a set.  Non-commutative semantics remain represented by
    # each branch's final input-port label.
    return content_digest(
        {
            "discovery": "fork_join.v1",
            "paths": sorted(path_signatures, key=canonical_json),
        }
    )


def discover_graph_regions(graph: ComputationGraph) -> tuple[GraphRegion, ...]:
    """Enumerate anonymous fork/join regions with no architecture classifier.

    Every operation with multiple data-flow inputs is considered. Generic
    parameter/module resource edges remain structural fingerprint evidence but
    are not mistaken for executable branches. For each pair of input branches,
    the nearest common ancestor whose paths are internally disjoint defines one
    occurrence. The join operator is data in the region signature; it is never
    translated into a feature name such as ``residual`` or ``attention``.
    """

    nodes, all_incoming, all_outgoing = _graph_maps(graph)
    incoming = {
        node_id: [edge for edge in edges if _is_data_flow_edge(edge)]
        for node_id, edges in all_incoming.items()
    }
    outgoing = {
        node_id: [edge for edge in edges if _is_data_flow_edge(edge)]
        for node_id, edges in all_outgoing.items()
    }
    regions: list[GraphRegion] = []
    for join_id in sorted(nodes):
        join_operator = nodes[join_id].operation
        if join_operator in {"graph.input", "graph.output", "data.attribute"}:
            continue
        predecessors = sorted({edge.source for edge in incoming[join_id]})
        if len(predecessors) < 2:
            continue
        # One region per predecessor pair keeps high-arity joins explicit.
        for left_index, left in enumerate(predecessors):
            for right in predecessors[left_index + 1 :]:
                left_ancestors = _ancestor_distances(left, incoming)
                right_ancestors = _ancestor_distances(right, incoming)
                candidates = set(left_ancestors) & set(right_ancestors)
                chosen: tuple[str, tuple[str, ...], tuple[str, ...]] | None = None
                ranking = sorted(
                    candidates,
                    key=lambda item: (
                        max(left_ancestors[item], right_ancestors[item]),
                        left_ancestors[item] + right_ancestors[item],
                        item,
                    ),
                )
                for entry in ranking:
                    left_path = _shortest_path(entry, left, outgoing)
                    right_path = _shortest_path(entry, right, outgoing)
                    if left_path is None or right_path is None:
                        continue
                    # Apart from the entry, the two branches must be distinct.
                    if set(left_path[1:]) & set(right_path[1:]):
                        continue
                    chosen = (entry, left_path + (join_id,), right_path + (join_id,))
                    break
                if chosen is None:
                    continue
                entry, left_path, right_path = chosen
                support = _path_edge_ids(left_path, graph.edges) + _path_edge_ids(
                    right_path, graph.edges
                )
                paths = (left_path, right_path)
                regions.append(
                    GraphRegion(
                        entry_id=entry,
                        exit_id=join_id,
                        branch_paths=paths,
                        support_edges=tuple(support),
                        structural_signature=_region_signature(graph, paths),
                    )
                )
    unique = {region.occurrence_id: region for region in regions}
    return tuple(unique[key] for key in sorted(unique))


@dataclass(frozen=True)
class FingerprintScale:
    radius: int
    features: Mapping[str, int]

    def __post_init__(self) -> None:
        if isinstance(self.radius, bool) or not isinstance(self.radius, int) or self.radius < 0:
            raise ValueError("fingerprint radius must be a non-negative integer")
        normalized: dict[str, int] = {}
        for key, value in self.features.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("fingerprint counts must be integers")
            count = value
            if count < 0:
                raise ValueError("fingerprint counts cannot be negative")
            if count:
                normalized[_nonempty(str(key), "fingerprint key")] = count
        object.__setattr__(self, "features", freeze_json(dict(sorted(normalized.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {"features": dict(self.features), "radius": self.radius}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FingerprintScale":
        return cls(radius=raw["radius"], features=raw.get("features", {}))


@dataclass(frozen=True)
class StructuralFingerprint:
    scales: tuple[FingerprintScale, ...]
    scheme: str = "directed-wl-counts.v1"

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.scales, key=lambda item: item.radius))
        radii = [item.radius for item in ordered]
        if len(radii) != len(set(radii)):
            raise ValueError("fingerprint radii must be unique")
        object.__setattr__(self, "scales", ordered)
        object.__setattr__(self, "scheme", _nonempty(self.scheme, "fingerprint scheme"))

    def scale(self, radius: int) -> FingerprintScale:
        for item in self.scales:
            if item.radius == radius:
                return item
        raise KeyError(radius)

    def to_dict(self) -> dict[str, Any]:
        return {"scales": [item.to_dict() for item in self.scales], "scheme": self.scheme}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StructuralFingerprint":
        return cls(
            scales=tuple(FingerprintScale.from_dict(item) for item in raw.get("scales", [])),
            scheme=str(raw.get("scheme", "directed-wl-counts.v1")),
        )


def _hash_label(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _wl_color_maps(
    graph: ComputationGraph,
    requested: Sequence[int],
) -> dict[int, dict[str, str]]:
    """Return each node's anonymous directed-neighborhood digest by radius."""

    nodes, incoming, outgoing = _graph_maps(graph)
    colors = {
        node_id: _hash_label(
            ["operator", node.operation, "attributes", node.attributes]
        )
        for node_id, node in nodes.items()
    }
    result: dict[int, dict[str, str]] = {}
    for radius in range(max(requested) + 1):
        if radius in requested:
            result[radius] = dict(colors)
        if radius == max(requested):
            break
        refined: dict[str, str] = {}
        for node_id in sorted(nodes):
            predecessor_colors = [
                (
                    _edge_label(
                        nodes[node_id].operation,
                        edge.position,
                        nodes[node_id].attributes,
                    ),
                    colors[edge.source],
                )
                for edge in incoming[node_id]
            ]
            successor_colors = [
                (
                    _edge_label(
                        nodes[edge.target].operation,
                        edge.position,
                        nodes[edge.target].attributes,
                    ),
                    colors[edge.target],
                )
                for edge in outgoing[node_id]
            ]
            refined[node_id] = _hash_label(
                {
                    "in": sorted(predecessor_colors),
                    "out": sorted(successor_colors),
                    "self": colors[node_id],
                }
            )
        colors = refined
    return result


def structural_fingerprint(
    graph: ComputationGraph,
    *,
    radii: Iterable[int] = (0, 1, 2, 3),
) -> StructuralFingerprint:
    """Count direction-aware Weisfeiler-Lehman neighborhoods."""

    requested = _normalize_radii(radii)
    color_maps = _wl_color_maps(graph, requested)
    return StructuralFingerprint(
        tuple(
            FingerprintScale(radius, Counter(color_maps[radius].values()))
            for radius in requested
        )
    )


@dataclass(frozen=True)
class GraphCharacterOccurrence:
    """Concrete graph support for one anonymous structural character."""

    character_id: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "character_id", _nonempty(self.character_id, "character_id")
        )
        nodes = tuple(sorted({_nonempty(str(item), "occurrence node") for item in self.node_ids}))
        edges = tuple(sorted({_nonempty(str(item), "occurrence edge") for item in self.edge_ids}))
        if not nodes and not edges:
            raise ValueError("a character occurrence must reference graph support")
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "edge_ids", edges)

    @property
    def occurrence_id(self) -> str:
        return _occurrence_id(self.character_id, self.node_ids, self.edge_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "edge_ids": list(self.edge_ids),
            "id": self.occurrence_id,
            "node_ids": list(self.node_ids),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphCharacterOccurrence":
        result = cls(
            character_id=str(raw["character_id"]),
            node_ids=tuple(str(item) for item in raw.get("node_ids", [])),
            edge_ids=tuple(str(item) for item in raw.get("edge_ids", [])),
        )
        supplied = raw.get("id")
        if supplied is not None and str(supplied) != result.occurrence_id:
            raise ValueError("character occurrence ID does not match its support")
        return result


@dataclass(frozen=True)
class GraphCharacter:
    """A content-addressed character generated without a semantic catalog."""

    generator: str
    structural_signature: str
    occurrences: tuple[GraphCharacterOccurrence, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "generator", _nonempty(self.generator, "character generator"))
        signature = str(self.structural_signature)
        if re.fullmatch(r"[0-9a-f]{64}", signature) is None:
            raise ValueError("character structural_signature must be a SHA-256 digest")
        occurrences = tuple(
            sorted(self.occurrences, key=lambda item: item.occurrence_id)
        )
        if not occurrences:
            raise ValueError("a graph character must have at least one occurrence")
        occurrence_ids = [item.occurrence_id for item in occurrences]
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise ValueError("graph character occurrences must be unique")
        normalized = normalize_json(dict(self.parameters))
        if not isinstance(normalized, dict):  # pragma: no cover - defensive
            raise TypeError("character parameters must normalize to an object")
        object.__setattr__(self, "structural_signature", signature)
        object.__setattr__(self, "occurrences", occurrences)
        object.__setattr__(self, "parameters", freeze_json(normalized))
        expected_id = _character_id(self.generator, signature, normalized)
        if any(item.character_id != expected_id for item in occurrences):
            raise ValueError(
                "every occurrence must reference its containing character"
            )

    @property
    def character_id(self) -> str:
        return _character_id(
            self.generator,
            self.structural_signature,
            self.parameters,
        )

    @property
    def count(self) -> int:
        return len(self.occurrences)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "generator": self.generator,
            "id": self.character_id,
            "occurrences": [item.to_dict() for item in self.occurrences],
            "parameters": normalize_json(self.parameters),
            "structural_signature": self.structural_signature,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphCharacter":
        result = cls(
            generator=str(raw["generator"]),
            structural_signature=str(raw["structural_signature"]),
            occurrences=tuple(
                GraphCharacterOccurrence.from_dict(item)
                for item in raw.get("occurrences", [])
            ),
            parameters=raw.get("parameters", {}),
        )
        if raw.get("id") is not None and str(raw["id"]) != result.character_id:
            raise ValueError("graph character ID does not match its content")
        if raw.get("count") is not None and int(raw["count"]) != result.count:
            raise ValueError("graph character count does not match its occurrences")
        return result


def discover_graph_characters(
    graph: ComputationGraph,
    *,
    regions: Sequence[GraphRegion] | None = None,
    radii: Iterable[int] = (0, 1, 2, 3),
) -> tuple[GraphCharacter, ...]:
    """Generate anonymous multiscale and region characters from ``graph``.

    Radius-zero characters are operators/layers with their structural
    attributes.  Larger radii are directed neighborhoods.  Identical regions
    and neighborhoods are automatically represented as repeated occurrences.
    """

    requested = _normalize_radii(radii)
    colors = _wl_color_maps(graph, requested)
    characters: list[GraphCharacter] = []
    for radius in requested:
        grouped: dict[str, list[str]] = {}
        for node_id, signature in colors[radius].items():
            grouped.setdefault(signature, []).append(node_id)
        for signature, node_ids in sorted(grouped.items()):
            character_id = _character_id(
                "directed_wl_neighborhood.v1",
                signature,
                {"radius": radius},
            )
            characters.append(
                GraphCharacter(
                    generator="directed_wl_neighborhood.v1",
                    structural_signature=signature,
                    occurrences=tuple(
                        GraphCharacterOccurrence(
                            character_id=character_id,
                            node_ids=(node_id,),
                        )
                        for node_id in sorted(node_ids)
                    ),
                    parameters={"radius": radius},
                )
            )

    discovered_regions = tuple(regions) if regions is not None else discover_graph_regions(graph)
    region_groups: dict[str, list[GraphRegion]] = {}
    for region in discovered_regions:
        region_groups.setdefault(region.structural_signature, []).append(region)
    for signature, occurrences in sorted(region_groups.items()):
        character_id = _character_id("fork_join_region.v1", signature)
        characters.append(
            GraphCharacter(
                generator="fork_join_region.v1",
                structural_signature=signature,
                occurrences=tuple(
                    GraphCharacterOccurrence(
                        character_id=character_id,
                        node_ids=region.support_nodes,
                        edge_ids=region.support_edges,
                    )
                    for region in occurrences
                ),
            )
        )
    return tuple(sorted(characters, key=lambda item: item.character_id))


@dataclass(frozen=True)
class GraphDistance:
    """A metric in dynamic-character space (a graph pseudometric)."""

    distance: float
    per_radius: Mapping[int, float]
    region_distance: float = 0.0
    radius_weights: Mapping[int, float] = field(default_factory=dict)
    region_weight: float = 1.0
    left_digest: str = ""
    right_digest: str = ""
    distance_signature: str = ""
    scheme: str = "weighted-l1-on-directed-wl-counts.v1"

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, (int, float)):
            raise TypeError("graph distance must be numeric")
        value = float(self.distance)
        if not math.isfinite(value) or value < 0:
            raise ValueError("graph distance must be finite and non-negative")
        normalized_per_radius: dict[int, float] = {}
        for key, item in sorted(self.per_radius.items()):
            if isinstance(key, bool) or not isinstance(key, int):
                raise TypeError("per-radius distance keys must be integers")
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError("per-radius graph distances must be numeric")
            radius_value = float(item)
            if not math.isfinite(radius_value) or radius_value < 0:
                raise ValueError(
                    "per-radius graph distances must be finite and non-negative"
                )
            normalized_per_radius[key] = radius_value
        object.__setattr__(self, "distance", value)
        object.__setattr__(
            self, "per_radius", MappingProxyType(dict(normalized_per_radius))
        )
        if isinstance(self.region_distance, bool) or not isinstance(
            self.region_distance, (int, float)
        ):
            raise TypeError("region distance must be numeric")
        region_distance = float(self.region_distance)
        if not math.isfinite(region_distance) or region_distance < 0:
            raise ValueError("region distance must be finite and non-negative")
        if isinstance(self.region_weight, bool) or not isinstance(
            self.region_weight, (int, float)
        ):
            raise TypeError("region weight must be numeric")
        region_weight = float(self.region_weight)
        if not math.isfinite(region_weight) or region_weight <= 0:
            raise ValueError("region weight must be finite and strictly positive")
        object.__setattr__(self, "region_distance", region_distance)
        object.__setattr__(self, "region_weight", region_weight)
        normalized_weights: dict[int, float] = {}
        for key, item in sorted(self.radius_weights.items()):
            if isinstance(key, bool) or not isinstance(key, int):
                raise TypeError("radius weight keys must be integers")
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError("radius weights must be numeric")
            weight = float(item)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("radius weights must be finite and strictly positive")
            normalized_weights[key] = weight
        if normalized_weights and set(normalized_weights) != set(normalized_per_radius):
            raise ValueError("radius weights must match per-radius distances")
        object.__setattr__(
            self, "radius_weights", MappingProxyType(dict(normalized_weights))
        )
        left_digest = _nonempty(str(self.left_digest), "left graph digest")
        right_digest = _nonempty(str(self.right_digest), "right graph digest")
        object.__setattr__(self, "left_digest", left_digest)
        object.__setattr__(self, "right_digest", right_digest)
        signature = str(self.distance_signature)
        if re.fullmatch(r"[0-9a-f]{64}", signature) is None:
            raise ValueError("distance_signature must be a SHA-256 digest")
        object.__setattr__(self, "distance_signature", signature)
        object.__setattr__(self, "scheme", _nonempty(self.scheme, "distance scheme"))
        expected = sum(
            normalized_weights.get(radius, 1.0) * difference
            for radius, difference in normalized_per_radius.items()
        ) + region_weight * region_distance
        if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "graph distance must equal its weighted per-radius and region components"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "distance_signature": self.distance_signature,
            "left_digest": self.left_digest,
            "per_radius": {str(key): value for key, value in self.per_radius.items()},
            "region_distance": self.region_distance,
            "region_weight": self.region_weight,
            "radius_weights": {
                str(key): value for key, value in self.radius_weights.items()
            },
            "right_digest": self.right_digest,
            "scheme": self.scheme,
            "space": "dynamic_character_metric_graph_pseudometric",
        }


def _as_graph(value: ComputationGraph | "GraphProfile") -> ComputationGraph:
    return value.graph if isinstance(value, GraphProfile) else value


def compare_graphs(
    left: ComputationGraph | "GraphProfile",
    right: ComputationGraph | "GraphProfile",
    *,
    radii: Iterable[int] = (0, 1, 2, 3),
    radius_weights: Mapping[int, float] | None = None,
    region_weight: float = 1.0,
) -> GraphDistance:
    """Return raw weighted L1 distance; no pairwise size normalization is used.

    All selected components receive strictly positive weight, making this a
    metric on the generated character-count vectors.  It remains only a
    pseudometric on graphs because finite-radius WL and region counts are not a
    complete graph invariant.
    """

    requested = _normalize_radii(radii)
    left_fingerprint = structural_fingerprint(_as_graph(left), radii=requested)
    right_fingerprint = structural_fingerprint(_as_graph(right), radii=requested)
    weights = {radius: 1.0 for radius in requested}
    region_weight = float(region_weight)
    if not math.isfinite(region_weight) or region_weight <= 0:
        raise ValueError("region weight must be finite and strictly positive")
    if radius_weights is not None:
        for radius in requested:
            weights[radius] = float(radius_weights.get(radius, 1.0))
            if not math.isfinite(weights[radius]) or weights[radius] <= 0:
                raise ValueError("radius weights must be finite and strictly positive")
    per_radius: dict[int, float] = {}
    total = 0.0
    for radius in requested:
        left_features = left_fingerprint.scale(radius).features
        right_features = right_fingerprint.scale(radius).features
        difference = float(
            sum(
                abs(left_features.get(key, 0) - right_features.get(key, 0))
                for key in set(left_features) | set(right_features)
            )
        )
        per_radius[radius] = difference
        total += weights[radius] * difference
    left_graph = _as_graph(left)
    right_graph = _as_graph(right)
    left_regions = left.regions if isinstance(left, GraphProfile) else discover_graph_regions(left_graph)
    right_regions = right.regions if isinstance(right, GraphProfile) else discover_graph_regions(right_graph)
    left_region_counts = Counter(item.structural_signature for item in left_regions)
    right_region_counts = Counter(item.structural_signature for item in right_regions)
    region_difference = float(
        sum(
            abs(left_region_counts.get(key, 0) - right_region_counts.get(key, 0))
            for key in set(left_region_counts) | set(right_region_counts)
        )
    )
    total += region_weight * region_difference
    return GraphDistance(
        distance=total,
        per_radius=per_radius,
        region_distance=region_difference,
        radius_weights=weights,
        region_weight=region_weight,
        left_digest=left_graph.structural_digest,
        right_digest=right_graph.structural_digest,
        distance_signature=content_digest(
            {
                "fingerprint_scheme": left_fingerprint.scheme,
                "radii": list(requested),
                "radius_weights": {
                    str(radius): weights[radius] for radius in requested
                },
                "region_scheme": "fork_join_region.v1",
                "region_weight": region_weight,
                "scheme": "weighted-l1-on-dynamic-graph-characters.v1",
            }
        ),
        scheme="weighted-l1-on-dynamic-graph-characters.v1",
    )


@dataclass(frozen=True)
class GraphProfile:
    """A graph plus only automatically generated structural characters."""

    graph: ComputationGraph
    regions: tuple[GraphRegion, ...]
    characters: tuple[GraphCharacter, ...]
    fingerprint: StructuralFingerprint
    frontend: str = "generic"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    profile_version: str = GRAPH_PROFILE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "frontend", _nonempty(self.frontend, "frontend"))
        if self.profile_version != GRAPH_PROFILE_VERSION:
            raise ValueError(
                f"profile_version must be {GRAPH_PROFILE_VERSION!r}, "
                f"got {self.profile_version!r}"
            )
        ordered_regions = tuple(
            sorted(self.regions, key=lambda item: item.occurrence_id)
        )
        ordered_characters = tuple(
            sorted(self.characters, key=lambda item: item.character_id)
        )
        object.__setattr__(self, "regions", ordered_regions)
        object.__setattr__(self, "characters", ordered_characters)
        normalized = normalize_json(dict(self.metadata))
        if not isinstance(normalized, dict):  # pragma: no cover - defensive
            raise TypeError("graph profile metadata must normalize to an object")
        object.__setattr__(self, "metadata", freeze_json(normalized))

        radii = tuple(scale.radius for scale in self.fingerprint.scales)
        expected_fingerprint = structural_fingerprint(self.graph, radii=radii)
        if self.fingerprint != expected_fingerprint:
            raise ValueError("graph profile fingerprint does not match its graph")
        expected_regions = discover_graph_regions(self.graph)
        if ordered_regions != expected_regions:
            raise ValueError("graph profile regions do not match its graph")
        expected_characters = discover_graph_characters(
            self.graph, regions=expected_regions, radii=radii
        )
        if ordered_characters != expected_characters:
            raise ValueError("graph profile characters do not match its graph")

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict(include_digest=False))

    def canonical_json(self, *, indent: int | None = None) -> str:
        return canonical_json(self.to_dict(), indent=indent)

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "artifact_type": "phylodigy.computation_graph_profile",
            "characters": [item.to_dict() for item in self.characters],
            "fingerprint": self.fingerprint.to_dict(),
            "frontend": self.frontend,
            "profile_version": self.profile_version,
            "graph": self.graph.to_dict(),
            "regions": [item.to_dict() for item in self.regions],
        }
        if self.metadata:
            result["metadata"] = normalize_json(self.metadata)
        if include_digest:
            result["digest"] = content_digest(result)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GraphProfile":
        result = cls(
            graph=ComputationGraph.from_dict(raw["graph"]),
            regions=tuple(GraphRegion.from_dict(item) for item in raw.get("regions", [])),
            characters=tuple(
                GraphCharacter.from_dict(item) for item in raw.get("characters", [])
            ),
            fingerprint=StructuralFingerprint.from_dict(raw["fingerprint"]),
            frontend=str(raw.get("frontend", "generic")),
            metadata=raw.get("metadata", {}),
            profile_version=str(raw.get("profile_version", GRAPH_PROFILE_VERSION)),
        )
        supplied = raw.get("digest")
        if supplied is not None and str(supplied) != result.digest:
            raise ValueError("graph profile digest mismatch")
        return result


def build_graph_profile(
    records: Iterable[Mapping[str, Any]],
    *,
    frontend: str = "torch.fx",
    metadata: Mapping[str, Any] | None = None,
    radii: Iterable[int] = (0, 1, 2, 3),
    include_tensor_metadata: bool = True,
) -> GraphProfile:
    """Lower records and derive all currently supported graph characters."""

    graph = computation_graph_from_records(
        records, include_tensor_metadata=include_tensor_metadata
    )
    regions = discover_graph_regions(graph)
    return GraphProfile(
        graph=graph,
        regions=regions,
        characters=discover_graph_characters(graph, regions=regions, radii=radii),
        fingerprint=structural_fingerprint(graph, radii=radii),
        frontend=frontend,
        metadata=metadata or {},
    )
