"""Typed Arrow/Parquet storage for graphs, trees, and analysis tables.

Arrow is the in-memory representation.  Parquet is the durable representation
because it is compressed, predicate-friendly, and directly indexable by the
Hugging Face Dataset Viewer.  Complex provider or frontend payloads are kept as
canonical CBOR only when they cannot be represented as stable public columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

from .canonical import normalize_json
from .computation_graph import ComputationGraph, GraphEdge, GraphNode


COLUMNAR_SCHEMA_VERSION = "1"


class ColumnarUnavailableError(RuntimeError):
    """Raised when optional columnar-storage dependencies are unavailable."""


class ColumnarFormatError(ValueError):
    """Raised when a Parquet artifact is malformed or semantically inconsistent."""


def _libraries() -> tuple[Any, Any, Any]:
    try:
        import cbor2
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ColumnarUnavailableError(
            "columnar artifacts require the optional 'pyarrow' and 'cbor2' "
            "dependencies; install 'phylodigy[columnar]'"
        ) from exc
    return pa, pq, cbor2


def canonical_cbor(value: Any) -> bytes:
    """Encode one normalized value with RFC 8949 canonical map ordering."""

    _, _, cbor2 = _libraries()
    return bytes(cbor2.dumps(normalize_json(value), canonical=True))


def decode_cbor(payload: bytes | bytearray | memoryview) -> Any:
    _, _, cbor2 = _libraries()
    try:
        return normalize_json(cbor2.loads(bytes(payload)))
    except Exception as exc:
        raise ColumnarFormatError(f"invalid canonical CBOR payload: {exc}") from exc


def semantic_digest(value: Any) -> str:
    """Hash semantic content independently of Parquet physical layout."""

    return hashlib.sha256(canonical_cbor(value)).hexdigest()


def _metadata(table_name: str) -> dict[bytes, bytes]:
    return {
        b"phylodigy.schema_version": COLUMNAR_SCHEMA_VERSION.encode("ascii"),
        b"phylodigy.table": table_name.encode("ascii"),
    }


def _schema(table_name: str, fields: Iterable[Any]) -> Any:
    pa, _, _ = _libraries()
    return pa.schema(list(fields), metadata=_metadata(table_name))


def _write_table(path: Path, rows: list[dict[str, Any]], schema: Any) -> None:
    pa, pq, _ = _libraries()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            data_page_version="1.0",
            row_group_size=65_536,
            use_dictionary=True,
            version="2.6",
            write_page_index=True,
        )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_rows(path: Path, table_name: str) -> list[dict[str, Any]]:
    _, pq, _ = _libraries()
    if not path.is_file():
        raise FileNotFoundError(f"missing Parquet table: {path}")
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise ColumnarFormatError(f"could not read Parquet table {path}: {exc}") from exc
    metadata = table.schema.metadata or {}
    actual_name = metadata.get(b"phylodigy.table", b"").decode("utf-8", "replace")
    actual_version = metadata.get(b"phylodigy.schema_version", b"").decode(
        "utf-8", "replace"
    )
    if actual_name != table_name:
        raise ColumnarFormatError(
            f"expected table {table_name!r} in {path}, found {actual_name!r}"
        )
    if actual_version != COLUMNAR_SCHEMA_VERSION:
        raise ColumnarFormatError(
            f"unsupported columnar schema {actual_version!r} in {path}"
        )
    return table.to_pylist()


def _search_text(value: Any) -> str:
    tokens: list[str] = []

    def visit(item: Any) -> None:
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, (str, int, float)):
            text = str(item).strip()
            if text:
                tokens.append(text)
            return
        if isinstance(item, Mapping):
            for key in sorted(item):
                visit(key)
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(normalize_json(value))
    return " ".join(tokens)


def _graph_schemas() -> dict[str, Any]:
    pa, _, _ = _libraries()
    return {
        "graphs": _schema(
            "graphs",
            (
                pa.field("graph_id", pa.string(), nullable=False),
                pa.field("schema_version", pa.string(), nullable=False),
                pa.field("graph_digest", pa.string(), nullable=False),
                pa.field("structural_digest", pa.string(), nullable=False),
                pa.field("vertex_count", pa.int64(), nullable=False),
                pa.field("edge_count", pa.int64(), nullable=False),
            ),
        ),
        "graph_vertices": _schema(
            "graph_vertices",
            (
                pa.field("graph_id", pa.string(), nullable=False),
                pa.field("node_id", pa.string(), nullable=False),
                pa.field("operation", pa.string(), nullable=False),
                pa.field("structural_attributes_cbor", pa.binary(), nullable=False),
                pa.field("observations_cbor", pa.binary(), nullable=False),
                pa.field("search_text", pa.string(), nullable=False),
            ),
        ),
        "graph_edges": _schema(
            "graph_edges",
            (
                pa.field("graph_id", pa.string(), nullable=False),
                pa.field("edge_id", pa.string(), nullable=False),
                pa.field("source_id", pa.string(), nullable=False),
                pa.field("target_id", pa.string(), nullable=False),
                pa.field("position", pa.string(), nullable=False),
            ),
        ),
    }


def write_graph_tables(
    graphs: Mapping[str, ComputationGraph] | Iterable[tuple[str, ComputationGraph]],
    destination: str | Path,
) -> dict[str, Path]:
    """Write normalized graph, vertex, and edge Parquet tables."""

    items = list(graphs.items() if isinstance(graphs, Mapping) else graphs)
    items.sort(key=lambda item: item[0])
    if not items:
        raise ValueError("at least one computation graph is required")
    if any(not isinstance(key, str) or not key.strip() for key, _ in items):
        raise ValueError("graph IDs must be non-empty strings")
    if len({key for key, _ in items}) != len(items):
        raise ValueError("graph IDs must be unique")
    if any(not isinstance(graph, ComputationGraph) for _, graph in items):
        raise TypeError("graphs must contain ComputationGraph values")

    graph_rows: list[dict[str, Any]] = []
    vertex_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for graph_id, graph in items:
        graph_rows.append(
            {
                "edge_count": len(graph.edges),
                "graph_digest": graph.digest,
                "graph_id": graph_id,
                "schema_version": graph.schema_version,
                "structural_digest": graph.structural_digest,
                "vertex_count": len(graph.nodes),
            }
        )
        for node in graph.nodes:
            attributes = normalize_json(node.attributes)
            observations = normalize_json(node.observations)
            vertex_rows.append(
                {
                    "graph_id": graph_id,
                    "node_id": node.node_id,
                    "observations_cbor": canonical_cbor(observations),
                    "operation": node.operation,
                    "search_text": _search_text(
                        {
                            "attributes": attributes,
                            "observations": observations,
                            "operation": node.operation,
                        }
                    ),
                    "structural_attributes_cbor": canonical_cbor(attributes),
                }
            )
        for edge in graph.edges:
            edge_rows.append(
                {
                    "edge_id": edge.edge_id,
                    "graph_id": graph_id,
                    "position": edge.position,
                    "source_id": edge.source,
                    "target_id": edge.target,
                }
            )

    root = Path(destination)
    schemas = _graph_schemas()
    paths = {
        "graphs": root / "graphs.parquet",
        "graph_vertices": root / "graph_vertices.parquet",
        "graph_edges": root / "graph_edges.parquet",
    }
    _write_table(paths["graphs"], graph_rows, schemas["graphs"])
    _write_table(paths["graph_vertices"], vertex_rows, schemas["graph_vertices"])
    _write_table(paths["graph_edges"], edge_rows, schemas["graph_edges"])
    return paths


def read_graph_tables(source: str | Path) -> dict[str, ComputationGraph]:
    """Reconstruct and digest-check computation graphs from Parquet tables."""

    root = Path(source)
    graph_rows = _read_rows(root / "graphs.parquet", "graphs")
    vertex_rows = _read_rows(root / "graph_vertices.parquet", "graph_vertices")
    edge_rows = _read_rows(root / "graph_edges.parquet", "graph_edges")
    vertices: dict[str, list[GraphNode]] = {}
    edges: dict[str, list[GraphEdge]] = {}
    for row in vertex_rows:
        graph_id = str(row["graph_id"])
        attributes = decode_cbor(row["structural_attributes_cbor"])
        observations = decode_cbor(row["observations_cbor"])
        if not isinstance(attributes, Mapping) or not isinstance(observations, Mapping):
            raise ColumnarFormatError("graph vertex CBOR fields must decode to mappings")
        vertices.setdefault(graph_id, []).append(
            GraphNode(
                node_id=str(row["node_id"]),
                operation=str(row["operation"]),
                attributes=attributes,
                observations=observations,
            )
        )
    for row in edge_rows:
        graph_id = str(row["graph_id"])
        edges.setdefault(graph_id, []).append(
            GraphEdge(
                source=str(row["source_id"]),
                target=str(row["target_id"]),
                position=str(row["position"]),
            )
        )
    result: dict[str, ComputationGraph] = {}
    for row in sorted(graph_rows, key=lambda item: str(item["graph_id"])):
        graph_id = str(row["graph_id"])
        if graph_id in result:
            raise ColumnarFormatError(f"duplicate graph ID: {graph_id}")
        graph = ComputationGraph(
            nodes=tuple(vertices.pop(graph_id, ())),
            edges=tuple(edges.pop(graph_id, ())),
            schema_version=str(row["schema_version"]),
        )
        if graph.digest != row["graph_digest"]:
            raise ColumnarFormatError(f"graph digest mismatch: {graph_id}")
        if graph.structural_digest != row["structural_digest"]:
            raise ColumnarFormatError(f"structural graph digest mismatch: {graph_id}")
        if len(graph.nodes) != row["vertex_count"] or len(graph.edges) != row["edge_count"]:
            raise ColumnarFormatError(f"graph row count mismatch: {graph_id}")
        result[graph_id] = graph
    if vertices or edges:
        unknown = sorted(set(vertices) | set(edges))
        raise ColumnarFormatError(f"graph rows reference unknown graph IDs: {unknown}")
    return result


def _lineage_schemas() -> dict[str, Any]:
    pa, _, _ = _libraries()
    return {
        "phylogenies": _schema(
            "phylogenies",
            (
                pa.field("tree_id", pa.string(), nullable=False),
                pa.field("lineage_digest", pa.string(), nullable=False),
                pa.field("analysis_version", pa.string(), nullable=False),
                pa.field("method", pa.string(), nullable=False),
                pa.field("distance_model", pa.string(), nullable=False),
                pa.field("normalization", pa.string(), nullable=False),
                pa.field("taxon_count", pa.int64(), nullable=False),
                pa.field("branch_count", pa.int64(), nullable=False),
                pa.field("distance_domain_valid", pa.bool_()),
                pa.field("metric_within_tolerance", pa.bool_()),
                pa.field("additive_within_tolerance", pa.bool_()),
                pa.field("tree_metric_within_tolerance", pa.bool_()),
                pa.field("maximum_four_point_violation", pa.float64()),
                pa.field("analysis_cbor", pa.binary(), nullable=False),
            ),
        ),
        "tree_edges": _schema(
            "tree_edges",
            (
                pa.field("tree_id", pa.string(), nullable=False),
                pa.field("parent_id", pa.string(), nullable=False),
                pa.field("child_id", pa.string(), nullable=False),
                pa.field("branch_length", pa.float64(), nullable=False),
                pa.field("raw_branch_length", pa.float64(), nullable=False),
                pa.field("length_clamped", pa.bool_(), nullable=False),
                pa.field("child_is_taxon", pa.bool_(), nullable=False),
            ),
        ),
        "taxa": _schema(
            "taxa",
            (
                pa.field("tree_id", pa.string(), nullable=False),
                pa.field("model_id", pa.string(), nullable=False),
                pa.field("genome_digest", pa.string()),
                pa.field("graph_digest", pa.string()),
                pa.field("date_min", pa.date32()),
                pa.field("date_max", pa.date32()),
            ),
        ),
        "character_states": _schema(
            "character_states",
            (
                pa.field("tree_id", pa.string(), nullable=False),
                pa.field("model_id", pa.string(), nullable=False),
                pa.field("character_id", pa.string(), nullable=False),
                pa.field("count", pa.int64(), nullable=False),
            ),
        ),
        "distances": _schema(
            "distances",
            (
                pa.field("tree_id", pa.string(), nullable=False),
                pa.field("left_id", pa.string(), nullable=False),
                pa.field("right_id", pa.string(), nullable=False),
                pa.field("distance", pa.float64(), nullable=False),
                pa.field("scheme", pa.string(), nullable=False),
                pa.field("comparison_digest", pa.string()),
                pa.field("comparison_cbor", pa.binary(), nullable=False),
            ),
        ),
    }


def _date_value(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ColumnarFormatError(f"invalid lineage calendar date: {value!r}") from exc


def write_lineage_tables(
    lineage: Mapping[str, Any], destination: str | Path
) -> dict[str, Path]:
    """Write one rich lineage artifact as normalized searchable Parquet tables."""

    normalized = normalize_json(dict(lineage))
    if normalized.get("artifact_type") != "phylodigy.architecture_lineage":
        raise ValueError("lineage artifact_type must be phylodigy.architecture_lineage")
    tree = normalized.get("tree")
    if not isinstance(tree, Mapping):
        raise ValueError("lineage must contain a tree mapping")
    tree_id = str(tree.get("digest") or semantic_digest(tree))
    taxa = [str(item) for item in tree.get("taxa", ())]
    taxon_set = set(taxa)
    edges = tree.get("edges") or []
    if not isinstance(edges, list):
        raise ValueError("lineage tree edges must be an array")
    diagnostics = tree.get("tree_likeness") or {}
    if not isinstance(diagnostics, Mapping):
        raise ValueError("tree_likeness must be a mapping")
    lineage_digest = str(normalized.get("digest") or semantic_digest(normalized))

    phylogeny_rows = [
        {
            "additive_within_tolerance": diagnostics.get("additive_within_tolerance"),
            "analysis_cbor": canonical_cbor(normalized),
            "analysis_version": str(normalized.get("analysis_version") or "unknown"),
            "branch_count": len(edges),
            "distance_domain_valid": diagnostics.get("distance_domain_valid"),
            "distance_model": str(tree.get("distance_model") or "unknown"),
            "lineage_digest": lineage_digest,
            "maximum_four_point_violation": diagnostics.get(
                "maximum_four_point_violation"
            ),
            "method": str(tree.get("method") or "unknown"),
            "metric_within_tolerance": diagnostics.get("metric_within_tolerance"),
            "normalization": str(tree.get("normalization") or "none"),
            "taxon_count": len(taxa),
            "tree_id": tree_id,
            "tree_metric_within_tolerance": diagnostics.get(
                "tree_metric_within_tolerance"
            ),
        }
    ]
    tree_edge_rows = []
    for raw in edges:
        if not isinstance(raw, Mapping):
            raise ValueError("each lineage tree edge must be a mapping")
        child = str(raw["child"])
        tree_edge_rows.append(
            {
                "branch_length": float(raw["branch_length"]),
                "child_id": child,
                "child_is_taxon": child in taxon_set,
                "length_clamped": bool(raw.get("length_clamped", False)),
                "parent_id": str(raw["parent"]),
                "raw_branch_length": float(
                    raw.get("raw_branch_length", raw["branch_length"])
                ),
                "tree_id": tree_id,
            }
        )
    tree_edge_rows.sort(key=lambda row: (row["parent_id"], row["child_id"]))

    genomes = {
        str(item["id"]): item
        for item in normalized.get("genomes", ())
        if isinstance(item, Mapping) and item.get("id")
    }
    taxon_rows = []
    for model_id in sorted(taxa):
        genome = genomes.get(model_id, {})
        taxon_rows.append(
            {
                "date_max": _date_value(genome.get("date_max")),
                "date_min": _date_value(genome.get("date_min")),
                "genome_digest": genome.get("digest"),
                "graph_digest": genome.get("graph_digest"),
                "model_id": model_id,
                "tree_id": tree_id,
            }
        )

    matrix = normalized.get("character_matrix") or {}
    character_ids = list(matrix.get("character_ids") or [])
    counts_by_artifact = matrix.get("counts_by_artifact") or {}
    state_rows = []
    for model_id in sorted(counts_by_artifact):
        counts = list(counts_by_artifact[model_id])
        if len(counts) != len(character_ids):
            raise ValueError(f"character vector length mismatch for {model_id}")
        for character_id, count in zip(character_ids, counts, strict=True):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("character counts must be non-negative integers")
            if count:
                state_rows.append(
                    {
                        "character_id": str(character_id),
                        "count": count,
                        "model_id": str(model_id),
                        "tree_id": tree_id,
                    }
                )

    distance_rows = []
    for comparison in normalized.get("comparisons", ()):
        if not isinstance(comparison, Mapping):
            raise ValueError("lineage comparisons must be mappings")
        graph_distance = comparison.get("graph_distance") or {}
        left_id = str((comparison.get("left") or {}).get("id") or "")
        right_id = str((comparison.get("right") or {}).get("id") or "")
        if not left_id or not right_id:
            raise ValueError("lineage comparisons require left and right IDs")
        left_id, right_id = sorted((left_id, right_id))
        distance_rows.append(
            {
                "comparison_cbor": canonical_cbor(comparison),
                "comparison_digest": comparison.get("digest"),
                "distance": float(graph_distance["distance"]),
                "left_id": left_id,
                "right_id": right_id,
                "scheme": str(graph_distance.get("scheme") or "unknown"),
                "tree_id": tree_id,
            }
        )
    distance_rows.sort(key=lambda row: (row["left_id"], row["right_id"]))

    root = Path(destination)
    schemas = _lineage_schemas()
    paths = {name: root / f"{name}.parquet" for name in schemas}
    _write_table(paths["phylogenies"], phylogeny_rows, schemas["phylogenies"])
    _write_table(paths["tree_edges"], tree_edge_rows, schemas["tree_edges"])
    _write_table(paths["taxa"], taxon_rows, schemas["taxa"])
    _write_table(paths["character_states"], state_rows, schemas["character_states"])
    _write_table(paths["distances"], distance_rows, schemas["distances"])
    return paths


def read_lineage_tables(source: str | Path) -> dict[str, Any]:
    """Read the lossless lineage payload and verify every normalized table."""

    root = Path(source)
    phylogenies = _read_rows(root / "phylogenies.parquet", "phylogenies")
    if len(phylogenies) != 1:
        raise ColumnarFormatError("expected exactly one phylogeny row")
    row = phylogenies[0]
    lineage = decode_cbor(row["analysis_cbor"])
    if not isinstance(lineage, dict):
        raise ColumnarFormatError("lineage CBOR must decode to a mapping")
    tree = lineage.get("tree") or {}
    tree_id = str(tree.get("digest") or semantic_digest(tree))
    if tree_id != row["tree_id"]:
        raise ColumnarFormatError("tree ID does not match the lossless lineage payload")
    lineage_digest = str(lineage.get("digest") or semantic_digest(lineage))
    if lineage_digest != row["lineage_digest"]:
        raise ColumnarFormatError("lineage digest does not match the stored payload")

    expected_edges = {
        (str(item["parent"]), str(item["child"]))
        for item in tree.get("edges", ())
    }
    edge_rows = _read_rows(root / "tree_edges.parquet", "tree_edges")
    actual_edges = {(str(item["parent_id"]), str(item["child_id"])) for item in edge_rows}
    if expected_edges != actual_edges:
        raise ColumnarFormatError("normalized tree edge table does not match lineage")
    taxa_rows = _read_rows(root / "taxa.parquet", "taxa")
    if {str(item["model_id"]) for item in taxa_rows} != set(tree.get("taxa", ())):
        raise ColumnarFormatError("normalized taxa table does not match lineage")
    _read_rows(root / "character_states.parquet", "character_states")
    _read_rows(root / "distances.parquet", "distances")
    return lineage


__all__ = [
    "COLUMNAR_SCHEMA_VERSION",
    "ColumnarFormatError",
    "ColumnarUnavailableError",
    "canonical_cbor",
    "decode_cbor",
    "read_graph_tables",
    "read_lineage_tables",
    "semantic_digest",
    "write_graph_tables",
    "write_lineage_tables",
]
