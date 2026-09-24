"""Build graph-derived lineage reports over a complete Modelome entry snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .canonical import content_digest
from .compact_lineage import infer_compact_lineage
from .lineage import LineageAnalysisConfig, infer_lineage_network
from .modelome_binding import bind_modelome_profiles
from .modelome_input import read_modelome_entries
from .modelome_profiles import load_modelome_profiles
from .modelome_relations import build_modelome_relations


def plan_modelome_tree(entries_path: str | Path) -> dict[str, Any]:
    """Inventory every entry and its declared resources without extracting code."""
    result = build_modelome_tree(entries_path)
    result["artifact_type"] = "phylodigy.modelome_plan"
    result["status"] = "planned"
    result.pop("digest")
    result["digest"] = content_digest(result)
    return result


def build_modelome_tree(
    entries_path: str | Path,
    *,
    profiles_dir: str | Path | None = None,
    bindings: Mapping[str, str] | None = None,
    max_taxa: int = 100,
    collapse_identical: bool = False,
    config: LineageAnalysisConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer the observed subset, retaining coverage of the entire entry corpus.

    Every source entry remains in ``modelome.entries``. Unobserved graphs do
    not acquire invented tree tips or distances. Exact inference is bounded
    before pairwise work; callers may explicitly raise ``max_taxa``.
    """
    if isinstance(max_taxa, bool) or not isinstance(max_taxa, int) or max_taxa < 2:
        raise ValueError("max_taxa must be an integer of at least 2")
    if not isinstance(collapse_identical, bool):
        raise TypeError("collapse_identical must be boolean")
    selected = (
        config
        if isinstance(config, LineageAnalysisConfig)
        else (
            LineageAnalysisConfig.from_mapping(config)
            if config is not None
            else LineageAnalysisConfig()
        )
    )
    source = read_modelome_entries(entries_path)
    entries = sorted(source["entries"], key=lambda item: item["id"])
    profiles = load_modelome_profiles(profiles_dir) if profiles_dir is not None else {}
    bound = bind_modelome_profiles(entries, profiles, bindings=bindings)
    genomes = bound["genomes"]
    if not collapse_identical and len(genomes) > max_taxa:
        raise ValueError(
            f"{len(genomes)} profiled entries exceed max_taxa={max_taxa}; "
            "exact diagnostics scale quartically. Raise --max-taxa explicitly "
            "only when the required computation is acceptable."
        )
    if len(genomes) >= 2:
        result = (
            infer_compact_lineage(genomes, config=selected, max_graphs=max_taxa)
            if collapse_identical
            else infer_lineage_network(genomes, config=selected)
        )
        result.pop("digest")
        result["status"] = "inferred"
    else:
        result = {
            "artifact_type": "phylodigy.modelome_tree",
            "status": "insufficient_profiles",
            "tree": None,
            "comparisons": [],
            "config": selected.to_dict(),
        }
    result["modelome"] = {
        "schema_version": "1",
        "source_digest": source["source_digest"],
        "manifest": source["manifest"],
        "integrity": "verified_bundle"
        if source["manifest"] is not None
        else "unverified_file",
        "entries": entries,
        "coverage": bound["coverage"],
        "counts": {
            "entries": len(entries),
            "profiled": len(genomes),
            "unprofiled": len(entries) - len(genomes),
        },
        "complete_profile_coverage": len(genomes) == len(entries),
        "unused_profile_ids": bound["unused_profile_ids"],
        "declared_relations": build_modelome_relations(entries),
        "max_taxa": max_taxa,
        "collapse_identical": collapse_identical,
    }
    result["digest"] = content_digest(result)
    return result


__all__ = ["build_modelome_tree", "plan_modelome_tree"]
