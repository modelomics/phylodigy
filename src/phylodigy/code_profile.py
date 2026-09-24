"""Safe static source provenance extraction.

This module intentionally does not recognize named model techniques. It has no
semantic catalog or architecture classifier. Python, JSON, and TOML inputs are converted to generic directed source
graphs with an open operator vocabulary. The graph records are provenance and
possible tracing aids; an executable computation graph remains the authoritative
model structure.

Repository code is never imported, evaluated, or executed.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import posixpath
import tokenize
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:  # Python 3.11+; the project also supports Python 3.10.
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 only
    tomllib = None  # type: ignore[assignment]

from .canonical import canonical_json, content_digest, freeze_json, normalize_json


__all__ = [
    "EXTRACTOR_VERSION",
    "SCAN_POLICY_VERSION",
    "SourceManifest",
    "StaticSourceGraph",
    "SOURCE_GRAPH_SCHEMA_VERSION",
    "SourceScanPolicy",
    "extract_code_profile",
    "extract_source_profile",
    "source_tree_digest",
]


EXTRACTOR_VERSION = "2.0.0"
SCAN_POLICY_VERSION = "2.0.0"
SOURCE_TREE_FORMAT = "phylodigy.source-tree.v2"
SOURCE_GRAPH_SCHEMA_VERSION = "phylodigy.source-graph.v1"
SOURCE_MANIFEST_VERSION = "1.0.0"
SOURCE_MANIFEST_ARTIFACT_TYPE = "phylodigy.source_manifest"


_MANDATORY_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".old",
        "__pycache__",
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        "vendor",
        "vendors",
        "third_party",
        "third-party",
        "external",
    }
)


@dataclass(frozen=True)
class SourceScanPolicy:
    """Serializable safety and resource bounds for source traversal."""

    include_suffixes: tuple[str, ...] = (".json", ".py", ".toml", ".yaml", ".yml")
    exclude_dir_names: tuple[str, ...] = tuple(sorted(_MANDATORY_EXCLUDED_DIRS))
    exclude_file_suffixes: tuple[str, ...] = (".old", ".orig", ".rej")
    max_file_bytes: int = 2 * 1024 * 1024
    max_diagnostics: int = 512
    max_graph_nodes: int = 2048
    max_graph_edges: int = 8192
    record_unsupported_files: bool = True

    def __post_init__(self) -> None:
        suffixes = tuple(sorted({_normalize_suffix(item) for item in self.include_suffixes}))
        if not suffixes:
            raise ValueError("include_suffixes cannot be empty")
        excluded_dirs = tuple(
            sorted(_MANDATORY_EXCLUDED_DIRS | {str(item) for item in self.exclude_dir_names})
        )
        excluded_suffixes = tuple(
            sorted({_normalize_suffix(item) for item in self.exclude_file_suffixes})
        )
        for name in (
            "max_file_bytes",
            "max_diagnostics",
            "max_graph_nodes",
            "max_graph_edges",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "include_suffixes", suffixes)
        object.__setattr__(self, "exclude_dir_names", excluded_dirs)
        object.__setattr__(self, "exclude_file_suffixes", excluded_suffixes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exclude_dir_names": list(self.exclude_dir_names),
            "exclude_file_suffixes": list(self.exclude_file_suffixes),
            "include_suffixes": list(self.include_suffixes),
            "max_diagnostics": self.max_diagnostics,
            "max_file_bytes": self.max_file_bytes,
            "max_graph_edges": self.max_graph_edges,
            "max_graph_nodes": self.max_graph_nodes,
            "record_unsupported_files": self.record_unsupported_files,
        }


@dataclass(frozen=True)
class _Unit:
    path: str
    suffix: str
    size: int
    digest: str
    data: bytes | None
    status: str


@dataclass(frozen=True)
class StaticSourceGraph:
    """A non-executable syntax/data graph used only as tracing provenance."""

    documents: tuple[Mapping[str, Any], ...]
    truncated: bool = False
    schema_version: str = SOURCE_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_GRAPH_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SOURCE_GRAPH_SCHEMA_VERSION!r}"
            )
        normalized = tuple(
            normalize_json(dict(item))
            for item in sorted(self.documents, key=canonical_json)
        )
        if not all(isinstance(item, dict) for item in normalized):
            raise TypeError("source graph documents must normalize to objects")
        object.__setattr__(
            self, "documents", tuple(freeze_json(item) for item in normalized)
        )
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be boolean")

    @property
    def node_count(self) -> int:
        return sum(len(item.get("graph", {}).get("nodes", ())) for item in self.documents)

    @property
    def edge_count(self) -> int:
        return sum(len(item.get("graph", {}).get("edges", ())) for item in self.documents)

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "document_count": len(self.documents),
            "documents": [normalize_json(item) for item in self.documents],
            "edge_count": self.edge_count,
            "node_count": self.node_count,
            "schema_version": self.schema_version,
            "truncated": self.truncated,
        }
        if include_digest:
            result["digest"] = content_digest(result)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "StaticSourceGraph":
        result = cls(
            documents=tuple(raw.get("documents", ())),
            truncated=bool(raw.get("truncated", False)),
            schema_version=str(raw.get("schema_version", SOURCE_GRAPH_SCHEMA_VERSION)),
        )
        for field_name, actual in (
            ("document_count", len(result.documents)),
            ("node_count", result.node_count),
            ("edge_count", result.edge_count),
        ):
            if raw.get(field_name) is not None and int(raw[field_name]) != actual:
                raise ValueError(f"source graph {field_name} does not match its documents")
        supplied = raw.get("digest")
        if supplied is not None and str(supplied) != result.digest:
            raise ValueError("source graph digest mismatch")
        return result


@dataclass(frozen=True)
class SourceManifest:
    """Source provenance that is explicitly not an architectural genome."""

    artifact_id: str
    source_graph: StaticSourceGraph
    name: str = ""
    release_date: str | None = None
    date_min: str | None = None
    date_max: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = SOURCE_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        object.__setattr__(self, "artifact_id", self.artifact_id.strip())
        if not isinstance(self.source_graph, StaticSourceGraph):
            raise TypeError("source_graph must be a StaticSourceGraph")
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        object.__setattr__(self, "name", self.name.strip())
        for field_name in ("release_date", "date_min", "date_max"):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                    raise ValueError(f"{field_name} must use canonical YYYY-MM-DD form")
        if self.date_min and self.date_max and self.date_min > self.date_max:
            raise ValueError("date_min cannot be after date_max")
        normalized = normalize_json(dict(self.metadata or {}))
        if not isinstance(normalized, dict):
            raise TypeError("metadata must normalize to an object")
        object.__setattr__(self, "metadata", freeze_json(normalized))
        if self.version != SOURCE_MANIFEST_VERSION:
            raise ValueError(f"version must be {SOURCE_MANIFEST_VERSION!r}")

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        artifact: dict[str, Any] = {"id": self.artifact_id}
        if self.name:
            artifact["name"] = self.name
        for key in ("release_date", "date_min", "date_max"):
            if getattr(self, key):
                artifact[key] = getattr(self, key)
        if self.metadata:
            artifact["metadata"] = normalize_json(self.metadata)
        result: dict[str, Any] = {
            "artifact": artifact,
            "artifact_type": SOURCE_MANIFEST_ARTIFACT_TYPE,
            "source_graph": self.source_graph.to_dict(),
            "version": self.version,
        }
        if include_digest:
            result["digest"] = content_digest(result)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceManifest":
        if raw.get("artifact_type") != SOURCE_MANIFEST_ARTIFACT_TYPE:
            raise ValueError(f"artifact_type must be {SOURCE_MANIFEST_ARTIFACT_TYPE!r}")
        artifact = raw["artifact"]
        result = cls(
            artifact_id=str(artifact["id"]),
            source_graph=StaticSourceGraph.from_dict(raw["source_graph"]),
            name=str(artifact.get("name", "")),
            release_date=artifact.get("release_date"),
            date_min=artifact.get("date_min"),
            date_max=artifact.get("date_max"),
            metadata=artifact.get("metadata", {}),
            version=str(raw.get("version", SOURCE_MANIFEST_VERSION)),
        )
        if raw.get("digest") is not None and str(raw["digest"]) != result.digest:
            raise ValueError("source manifest digest mismatch")
        return result


def extract_code_profile(
    source_path: str | os.PathLike[str],
    *,
    artifact_id: str,
    name: str = "",
    release_date: str | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
    policy: SourceScanPolicy | None = None,
) -> SourceManifest:
    """Build a profile containing a generic structural source graph."""

    selected = policy or SourceScanPolicy()
    units, diagnostics, stats = _scan_path(Path(source_path), selected)
    return _profile_from_units(
        units,
        diagnostics,
        stats,
        artifact_id=artifact_id,
        name=name,
        release_date=release_date,
        date_min=date_min,
        date_max=date_max,
        policy=selected,
    )


def extract_source_profile(
    sources: Mapping[str, str | bytes],
    *,
    artifact_id: str,
    name: str = "",
    release_date: str | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
    policy: SourceScanPolicy | None = None,
) -> SourceManifest:
    """Pure in-memory variant of :func:`extract_code_profile`."""

    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping")
    selected = policy or SourceScanPolicy()
    units, diagnostics, stats = _units_from_mapping(sources, selected)
    return _profile_from_units(
        units,
        diagnostics,
        stats,
        artifact_id=artifact_id,
        name=name,
        release_date=release_date,
        date_min=date_min,
        date_max=date_max,
        policy=selected,
    )


def source_tree_digest(
    source_path: str | os.PathLike[str], *, policy: SourceScanPolicy | None = None
) -> str:
    """Return a digest of included normalized paths and file content."""

    selected = policy or SourceScanPolicy()
    units, _diagnostics, _stats = _scan_path(Path(source_path), selected)
    return _tree_digest(units)


def _profile_from_units(
    units: Sequence[_Unit],
    diagnostics: Sequence[Mapping[str, Any]],
    stats: Mapping[str, int],
    *,
    artifact_id: str,
    name: str,
    release_date: str | None,
    date_min: str | None,
    date_max: str | None,
    policy: SourceScanPolicy,
) -> SourceManifest:
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("artifact_id must be a non-empty string")

    documents: list[dict[str, Any]] = []
    parse_diagnostics = [dict(item) for item in diagnostics]
    analyzed = 0
    node_total = 0
    edge_total = 0
    truncated = False

    for unit in sorted(units, key=lambda item: item.path):
        if unit.data is None:
            continue
        try:
            text = _decode(unit.data, unit.suffix).replace("\r\n", "\n").replace("\r", "\n")
        except (SyntaxError, UnicodeDecodeError, LookupError):
            parse_diagnostics.append(
                {"code": "decode_failed", "path": unit.path, "stage": "decode"}
            )
            continue
        try:
            graph = _parse_source_graph(text, unit.suffix)
        except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
            parse_diagnostics.append(
                {
                    "code": "parse_failed",
                    "detail": type(exc).__name__,
                    "path": unit.path,
                    "stage": "parse",
                }
            )
            continue
        if graph is None:
            parse_diagnostics.append(
                {
                    "code": "syntax_not_structurally_parsed",
                    "path": unit.path,
                    "stage": "parse",
                }
            )
            continue
        analyzed += 1
        nodes = graph["nodes"]
        edges = graph["edges"]
        available_nodes = max(0, policy.max_graph_nodes - node_total)
        available_edges = max(0, policy.max_graph_edges - edge_total)
        kept_nodes = nodes[:available_nodes]
        kept_ids = {node["id"] for node in kept_nodes}
        kept_edges = [
            edge
            for edge in edges
            if edge["source"] in kept_ids and edge["target"] in kept_ids
        ][:available_edges]
        if len(kept_nodes) != len(nodes) or len(kept_edges) != len(edges):
            truncated = True
        node_total += len(kept_nodes)
        edge_total += len(kept_edges)
        structural = {
            "edges": kept_edges,
            "frontend": graph["frontend"],
            "nodes": kept_nodes,
        }
        documents.append(
            {
                "content_sha256": unit.digest,
                "graph": structural,
                "graph_digest": content_digest(structural),
                # Path is provenance only and excluded from graph_digest.
                "path": unit.path,
            }
        )
        if node_total >= policy.max_graph_nodes or edge_total >= policy.max_graph_edges:
            truncated = truncated or any(
                remaining.data is not None
                for remaining in units
                if remaining.path > unit.path
            )
            break

    source_graph = StaticSourceGraph(
        documents=tuple(documents),
        truncated=truncated,
    )
    ordered_diagnostics = sorted(parse_diagnostics, key=canonical_json)
    metadata = {
        "diagnostic_count": len(ordered_diagnostics),
        "diagnostics": ordered_diagnostics[: policy.max_diagnostics],
        "diagnostics_truncated": len(ordered_diagnostics) > policy.max_diagnostics,
        "extraction_policy": "generic_source_provenance_only",
        "extractor_version": EXTRACTOR_VERSION,
        "scan_policy": policy.to_dict(),
        "scan_policy_version": SCAN_POLICY_VERSION,
        "source_tree_digest": _tree_digest(units),
        "source_tree_file_count": len(units),
        "source_tree_format": SOURCE_TREE_FORMAT,
        "statistics": {**dict(sorted(stats.items())), "analyzed_file_count": analyzed},
    }
    return SourceManifest(
        artifact_id=artifact_id,
        source_graph=source_graph,
        name=name,
        release_date=release_date,
        date_min=date_min,
        date_max=date_max,
        metadata=normalize_json(metadata),
    )


def _parse_source_graph(text: str, suffix: str) -> dict[str, Any] | None:
    if suffix == ".py":
        return _python_graph(ast.parse(text, mode="exec", type_comments=True))
    if suffix == ".json":
        return _data_graph(json.loads(text, parse_constant=_reject_constant), "json")
    if suffix == ".toml" and tomllib is not None:
        return _data_graph(tomllib.loads(text), "toml")
    # YAML deliberately has no keyword scan. Without a complete parser it is
    # retained in the source manifest but cannot contribute structural claims.
    return None


def _python_graph(tree: ast.AST) -> dict[str, Any]:
    """Encode parser-provided structure without interpreting symbol meanings."""

    ordered: list[ast.AST] = []
    seen: set[int] = set()
    for candidate in ast.walk(tree):
        token = id(candidate)
        if token not in seen:
            seen.add(token)
            ordered.append(candidate)
    ids = {id(node): f"n{index:06d}" for index, node in enumerate(ordered)}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for node in ordered:
        record: dict[str, Any] = {
            "id": ids[id(node)],
            "kind": _ast_kind(node),
            "operation": _ast_operation(node),
        }
        attributes = _ast_attributes(node)
        if attributes:
            record["attributes"] = attributes
        nodes.append(record)
        for field_name, value in ast.iter_fields(node):
            children = [value] if isinstance(value, ast.AST) else (
                [item for item in value if isinstance(item, ast.AST)]
                if isinstance(value, list)
                else []
            )
            for position, child in enumerate(children):
                edges.append(
                    {
                        "kind": "syntax",
                        "position": f"{field_name}:{position:06d}",
                        "source": ids[id(child)],
                        "target": ids[id(node)],
                    }
                )
    return {
        "edges": sorted(edges, key=canonical_json),
        "frontend": "python.ast",
        "nodes": sorted(nodes, key=lambda item: item["id"]),
    }


def _ast_kind(node: ast.AST) -> str:
    if isinstance(node, ast.stmt):
        return "control"
    if isinstance(node, ast.expr):
        return "operator"
    if isinstance(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
        return "operator_token"
    return "syntax"


def _ast_operation(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        target = _call_target(node.func)
        return f"python.call:{target}" if target else "python.call"
    return f"python.ast:{type(node).__name__.casefold()}"


def _call_target(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_target(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _ast_attributes(node: ast.AST) -> dict[str, Any]:
    """Return generic arity/type attributes, never curated symbol meanings."""

    attributes: dict[str, Any] = {}
    if isinstance(node, ast.Call):
        attributes["keyword_count"] = len(node.keywords)
        attributes["positional_count"] = len(node.args)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        attributes["parameter_count"] = len(node.args.args)
    elif isinstance(node, ast.Constant):
        value = node.value
        attributes["literal_type"] = "null" if value is None else type(value).__name__
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        attributes["element_count"] = len(getattr(node, "elts", getattr(node, "keys", ())))
    return attributes


def _data_graph(value: Any, frontend: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def visit(item: Any) -> str:
        node_id = f"n{len(nodes):06d}"
        record: dict[str, Any] = {
            "id": node_id,
            "kind": "data",
            "operation": _data_operation(item),
        }
        if isinstance(item, Mapping):
            record["attributes"] = {"entry_count": len(item)}
        elif isinstance(item, (list, tuple)):
            record["attributes"] = {"element_count": len(item)}
        nodes.append(record)
        if isinstance(item, Mapping):
            for position, (key, child) in enumerate(sorted(item.items(), key=lambda pair: str(pair[0]))):
                child_id = visit(child)
                edges.append(
                    {
                        "field": str(key),
                        "kind": "field",
                        "position": f"{position:06d}",
                        "source": child_id,
                        "target": node_id,
                    }
                )
        elif isinstance(item, (list, tuple)):
            for position, child in enumerate(item):
                child_id = visit(child)
                edges.append(
                    {
                        "kind": "item",
                        "position": f"{position:06d}",
                        "source": child_id,
                        "target": node_id,
                    }
                )
        return node_id

    visit(value)
    return {"edges": edges, "frontend": frontend, "nodes": nodes}


def _data_operation(value: Any) -> str:
    if isinstance(value, Mapping):
        return "data.object"
    if isinstance(value, (list, tuple)):
        return "data.array"
    if value is None:
        return "data.null"
    if isinstance(value, bool):
        return "data.boolean"
    if isinstance(value, (int, float)):
        return "data.number"
    if isinstance(value, str):
        return "data.string"
    return f"data.{type(value).__name__.casefold()}"


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is unsupported: {value}")


def _decode(data: bytes, suffix: str) -> str:
    if suffix == ".py":
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        return data.decode(encoding)
    return data.decode("utf-8-sig")


def _normalize_suffix(value: Any) -> str:
    suffix = str(value).strip().lower()
    if not suffix:
        raise ValueError("file suffixes must be non-empty")
    return suffix if suffix.startswith(".") else f".{suffix}"


def _normalize_path(raw: Any) -> str:
    if not isinstance(raw, (str, os.PathLike)):
        raise TypeError("source paths must be strings or path-like values")
    path = os.fspath(raw).replace("\\", "/")
    if not path or path.startswith("/") or any(part == ".." for part in path.split("/")):
        raise ValueError(f"source path must be relative and contained: {path!r}")
    normalized = posixpath.normpath(path)
    if normalized in {"", "."} or normalized.startswith("../"):
        raise ValueError(f"invalid relative source path: {path!r}")
    return normalized


def _excluded(path: str, policy: SourceScanPolicy) -> bool:
    parts = PurePosixPath(path).parts
    if any(part in policy.exclude_dir_names for part in parts[:-1]):
        return True
    filename = parts[-1]
    return filename == ".old" or filename.endswith(".old") or any(
        filename.endswith(suffix) for suffix in policy.exclude_file_suffixes
    )


def _new_stats() -> dict[str, int]:
    return {
        "candidate_file_count": 0,
        "oversize_file_count": 0,
        "supported_file_count": 0,
        "symlink_count": 0,
        "unreadable_file_count": 0,
        "unsupported_file_count": 0,
    }


def _units_from_mapping(
    sources: Mapping[str, str | bytes], policy: SourceScanPolicy
) -> tuple[list[_Unit], list[dict[str, Any]], dict[str, int]]:
    normalized: dict[str, str | bytes] = {}
    for raw_path, raw_data in sources.items():
        path = _normalize_path(raw_path)
        if path in normalized:
            raise ValueError(f"normalized source-path collision: {path!r}")
        if not isinstance(raw_data, (str, bytes)):
            raise TypeError(f"source {path!r} must contain str or bytes")
        normalized[path] = raw_data
    units: list[_Unit] = []
    diagnostics: list[dict[str, Any]] = []
    stats = _new_stats()
    for path in sorted(normalized):
        if _excluded(path, policy):
            continue
        stats["candidate_file_count"] += 1
        suffix = PurePosixPath(path).suffix.casefold()
        if suffix not in policy.include_suffixes:
            stats["unsupported_file_count"] += 1
            if policy.record_unsupported_files:
                diagnostics.append(
                    {"code": "unsupported_extension", "path": path, "stage": "scan"}
                )
            continue
        data = normalized[path]
        payload = data.encode("utf-8") if isinstance(data, str) else data
        digest = hashlib.sha256(payload).hexdigest()
        stats["supported_file_count"] += 1
        if len(payload) > policy.max_file_bytes:
            stats["oversize_file_count"] += 1
            diagnostics.append({"code": "file_too_large", "path": path, "stage": "read"})
            units.append(_Unit(path, suffix, len(payload), digest, None, "oversize"))
        else:
            units.append(_Unit(path, suffix, len(payload), digest, payload, "readable"))
    return units, diagnostics, stats


def _scan_path(
    source_path: Path, policy: SourceScanPolicy
) -> tuple[list[_Unit], list[dict[str, Any]], dict[str, int]]:
    if not source_path.exists():
        raise FileNotFoundError(os.fspath(source_path))
    if source_path.is_symlink():
        raise ValueError("source_path cannot be a symlink")
    units: list[_Unit] = []
    diagnostics: list[dict[str, Any]] = []
    stats = _new_stats()

    def consider(path: Path, relative: str) -> None:
        normalized = _normalize_path(relative)
        if _excluded(normalized, policy):
            return
        stats["candidate_file_count"] += 1
        suffix = PurePosixPath(normalized).suffix.casefold()
        if suffix not in policy.include_suffixes:
            stats["unsupported_file_count"] += 1
            if policy.record_unsupported_files:
                diagnostics.append(
                    {"code": "unsupported_extension", "path": normalized, "stage": "scan"}
                )
            return
        stats["supported_file_count"] += 1
        try:
            size = path.stat().st_size
            if size > policy.max_file_bytes:
                digest = _file_digest(path)
                stats["oversize_file_count"] += 1
                diagnostics.append(
                    {"code": "file_too_large", "path": normalized, "stage": "read"}
                )
                units.append(_Unit(normalized, suffix, size, digest, None, "oversize"))
                return
            data = path.read_bytes()
        except OSError:
            stats["unreadable_file_count"] += 1
            diagnostics.append({"code": "read_failed", "path": normalized, "stage": "read"})
            return
        units.append(
            _Unit(normalized, suffix, len(data), hashlib.sha256(data).hexdigest(), data, "readable")
        )

    if source_path.is_file():
        consider(source_path, source_path.name)
    elif source_path.is_dir():
        for directory, names, filenames in os.walk(source_path, followlinks=False):
            names[:] = sorted(name for name in names if name not in policy.exclude_dir_names)
            base = Path(directory)
            for filename in sorted(filenames):
                path = base / filename
                relative = path.relative_to(source_path).as_posix()
                if path.is_symlink():
                    if not _excluded(relative, policy):
                        stats["symlink_count"] += 1
                        diagnostics.append(
                            {"code": "symlink_skipped", "path": relative, "stage": "scan"}
                        )
                    continue
                consider(path, relative)
    else:
        raise ValueError("source_path must be a regular file or directory")
    return sorted(units, key=lambda item: item.path), diagnostics, stats


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(units: Sequence[_Unit]) -> str:
    return content_digest(
        {
            "files": [
                {
                    "content_sha256": item.digest,
                    "path": item.path,
                    "status": item.status,
                }
                for item in sorted(units, key=lambda row: row.path)
            ],
            "format": SOURCE_TREE_FORMAT,
        }
    )
