"""Build a local Hugging Face dataset repository from Parquet tables.

This module only prepares files on the local filesystem.  It deliberately has
no dependency on ``huggingface_hub`` and never creates or updates a remote
dataset repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .columnar import ColumnarFormatError, ColumnarUnavailableError


@dataclass(frozen=True)
class _TableSpec:
    config_name: str
    source_kind: str
    source_names: tuple[str, ...]
    omit_model_payload: bool = False


# Keep the public scientific tables in the requested Dataset Viewer order.  The
# final four configurations expose the remaining registry tables under names
# that describe their contents rather than their storage implementation.
_TABLE_SPECS = (
    _TableSpec("models", "registry", ("current_records", "models"), True),
    _TableSpec("observations", "registry", ("observations",)),
    _TableSpec("papers", "registry", ("papers",)),
    _TableSpec(
        "model_paper_links",
        "registry",
        ("current_model_papers", "model_paper_links"),
    ),
    _TableSpec("snapshots", "registry", ("snapshots",)),
    _TableSpec("graph_vertices", "graph", ("graph_vertices",)),
    _TableSpec("graph_edges", "graph", ("graph_edges",)),
    _TableSpec("phylogenies", "lineage", ("phylogenies",)),
    _TableSpec("tree_edges", "lineage", ("tree_edges",)),
    _TableSpec("taxa", "lineage", ("taxa",)),
    _TableSpec("character_states", "lineage", ("character_states",)),
    _TableSpec("distances", "lineage", ("distances",)),
    _TableSpec("model_metrics", "registry", ("current_metrics", "model_metrics")),
    _TableSpec("paper_links", "registry", ("paper_links",)),
    _TableSpec("paper_metrics", "registry", ("paper_metrics",)),
    _TableSpec("coverage", "registry", ("provider_checkpoints", "coverage")),
)


def _parquet_library() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional environment
        raise ColumnarUnavailableError(
            "Hugging Face dataset export requires the optional 'pyarrow' "
            "dependency; install 'phylodigy[columnar]'"
        ) from exc
    return pq


def _require_directory(path: str | Path, label: str) -> Path:
    directory = Path(path)
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {directory}")
    return directory


def _table_files(root: Path, source_names: tuple[str, ...]) -> tuple[Path, ...]:
    """Find one flat table or a directory of shards under a source root."""

    for source_name in source_names:
        candidates: list[Path] = []
        flat_file = root / f"{source_name}.parquet"
        if flat_file.is_file():
            candidates.append(flat_file)
        for table_directory in (root / source_name, root / "data" / source_name):
            if table_directory.is_dir():
                candidates.extend(
                    path
                    for path in table_directory.rglob("*.parquet")
                    if path.is_file()
                )
        if candidates:
            return tuple(sorted(set(candidates), key=lambda path: path.as_posix()))
    return ()


def _validate_parquet(path: Path, pq: Any) -> None:
    try:
        pq.ParquetFile(path).schema_arrow
    except Exception as exc:
        raise ColumnarFormatError(f"invalid Parquet source {path}: {exc}") from exc


def _copy_parquet(source: Path, destination: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_models(source: Path, destination: Path, pq: Any) -> None:
    """Write the viewer-facing models shard without its opaque raw payload."""

    try:
        table = pq.read_table(source)
    except Exception as exc:
        raise ColumnarFormatError(
            f"could not read models table {source}: {exc}"
        ) from exc
    if "payload_cbor" in table.column_names:
        table = table.drop(["payload_cbor"])
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
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
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _dataset_card(config_names: tuple[str, ...]) -> str:
    lines = ["---", "configs:"]
    for config_name in config_names:
        lines.append(f"  - config_name: {config_name}")
        if config_name == "models":
            lines.append("    default: true")
        lines.extend(
            (
                "    data_files:",
                "      - split: full",
                f"        path: data/{config_name}/*.parquet",
            )
        )
    lines.extend(
        (
            "---",
            "",
            "# Phylodigy dataset",
            "",
            "Searchable registry, computation-graph, and phylogeny tables.",
            "Each configuration is a normalized Parquet table.",
            "",
        )
    )
    return "\n".join(lines)


def export_huggingface_dataset(
    registry_directory: str | Path,
    destination: str | Path,
    *,
    graph_tables: str | Path | None = None,
    lineage_tables: str | Path | None = None,
) -> dict[str, tuple[Path, ...]]:
    """Create a local, upload-ready Hugging Face dataset repository.

    ``registry_directory`` is a Parquet registry produced by Phylodigy.
    ``graph_tables`` and ``lineage_tables`` may point to the directories
    produced by :func:`write_graph_tables` and :func:`write_lineage_tables`.
    Only tables that exist are included.  The destination must be absent or
    empty so an earlier export cannot leave stale configurations behind.

    The returned mapping contains the written Parquet shards for each dataset
    configuration.  This function performs no network or upload operations.
    """

    registry_root = _require_directory(registry_directory, "registry_directory")
    graph_root = (
        _require_directory(graph_tables, "graph_tables")
        if graph_tables is not None
        else None
    )
    lineage_root = (
        _require_directory(lineage_tables, "lineage_tables")
        if lineage_tables is not None
        else None
    )
    roots = {
        "registry": registry_root,
        "graph": graph_root,
        "lineage": lineage_root,
    }

    output_root = Path(destination)
    if output_root.exists():
        if not output_root.is_dir():
            raise NotADirectoryError(f"destination is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise FileExistsError(f"destination is not empty: {output_root}")

    pq = _parquet_library()
    discovered: list[tuple[_TableSpec, tuple[Path, ...]]] = []
    for spec in _TABLE_SPECS:
        source_root = roots[spec.source_kind]
        if source_root is None:
            continue
        files = _table_files(source_root, spec.source_names)
        if not files:
            continue
        for path in files:
            _validate_parquet(path, pq)
        discovered.append((spec, files))

    if not any(spec.config_name == "models" for spec, _ in discovered):
        raise FileNotFoundError(
            f"registry has no current_records.parquet models table: {registry_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, tuple[Path, ...]] = {}
    for spec, sources in discovered:
        config_directory = output_root / "data" / spec.config_name
        config_directory.mkdir(parents=True, exist_ok=False)
        destinations: list[Path] = []
        for index, source in enumerate(sources):
            target = config_directory / f"part-{index:05d}.parquet"
            if spec.omit_model_payload:
                _write_models(source, target, pq)
            else:
                _copy_parquet(source, target)
            _validate_parquet(target, pq)
            destinations.append(target)
        written[spec.config_name] = tuple(destinations)

    config_names = tuple(spec.config_name for spec, _ in discovered)
    (output_root / "README.md").write_text(
        _dataset_card(config_names), encoding="utf-8", newline="\n"
    )
    return written


__all__ = ["export_huggingface_dataset"]
