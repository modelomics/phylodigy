"""Neighbor joining over distinct observed graphs, retaining every artifact leaf."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .canonical import content_digest, normalize_json
from .lineage import (
    LINEAGE_ANALYSIS_VERSION,
    LineageAnalysisConfig,
    infer_lineage_network,
)
from .profile_comparison import diagnose_tree_likeness, selected_character_counts
from .schema import ArchitecturalGenome


def infer_compact_lineage(
    profiles: Iterable[ArchitecturalGenome],
    *,
    config: LineageAnalysisConfig | Mapping[str, Any] | None = None,
    max_graphs: int = 100,
    evidence: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Infer on exact structural representatives and expand duplicate leaves.

    Every distinct graph has one vote in neighbor joining, irrespective of
    checkpoint multiplicity. This is a different estimator from artifact-level
    NJ on non-additive distances. The compact comparisons and character matrix
    are lossless representations of the original input distances and states;
    the fitted tree itself need not reproduce non-additive input distances.
    """
    if (
        isinstance(max_graphs, bool)
        or not isinstance(max_graphs, int)
        or max_graphs < 1
    ):
        raise ValueError("max_graphs must be a positive integer")
    genomes = list(profiles)
    if any(not isinstance(item, ArchitecturalGenome) for item in genomes):
        raise TypeError("profiles must contain ArchitecturalGenome objects")
    genomes.sort(key=lambda item: item.artifact_id)
    if len(genomes) < 2:
        raise ValueError("at least two architectural genomes are required")
    ids = [item.artifact_id for item in genomes]
    if len(set(ids)) != len(ids):
        raise ValueError("artifact IDs must be unique")
    selected = (
        config
        if isinstance(config, LineageAnalysisConfig)
        else (
            LineageAnalysisConfig.from_mapping(config)
            if config is not None
            else LineageAnalysisConfig()
        )
    )
    groups: dict[str, list[ArchitecturalGenome]] = {}
    for genome in genomes:
        groups.setdefault(genome.structural_digest, []).append(genome)
    if len(groups) > max_graphs:
        raise ValueError(
            f"{len(groups)} distinct graphs exceed max_graphs={max_graphs}; "
            "exact representative diagnostics still scale quartically"
        )
    representatives = sorted(
        (group[0] for group in groups.values()), key=lambda item: item.artifact_id
    )
    if len(representatives) > 1:
        result = infer_lineage_network(
            representatives, config=selected, evidence=evidence
        )
        result.pop("digest")
    else:
        counts = selected_character_counts(representatives[0], radii=selected.radii)
        characters = sorted(counts)
        external = [normalize_json(dict(item)) for item in evidence]
        external.sort(key=content_digest)
        result = {
            "analysis_version": LINEAGE_ANALYSIS_VERSION,
            "config": selected.to_dict(),
            "comparisons": [],
            "external_evidence": external,
            "character_matrix": {
                "character_ids": characters,
                "counts_by_artifact": {ids[0]: [counts[key] for key in characters]},
                "source": "operator_layer_graph_only",
            },
            "structural_evidence_boundary": {
                "paper_can_create_characters": False,
                "structural_source": "operator_layer_graph",
            },
            "tree": {
                "artifact_type": "phylodigy.architecture_phylogeny",
                "distance_model": "weighted_l1_dynamic_graph_characters",
                "normalization": "none",
                "edges": [],
                "tree_likeness": diagnose_tree_likeness(
                    [], tolerance=selected.four_point_tolerance
                ),
            },
        }
    aliases = {
        item.artifact_id: group[0].artifact_id
        for group in groups.values()
        for item in group
    }
    tree = result["tree"]
    edges = [dict(edge) for edge in tree["edges"]]
    used_ids = set(ids) | {edge[side] for edge in edges for side in ("parent", "child")}
    # The representative-only inference cannot reserve IDs of omitted aliases.
    # Rename any inferred internal node that collides with an original artifact.
    internal_ids = {edge["parent"] for edge in edges}
    renames = {}
    for index, node in enumerate(sorted(internal_ids & set(ids))):
        replacement = f"__phylodigy_internal__:quotient:{index}"
        while replacement in used_ids:
            replacement += ":"
        used_ids.add(replacement)
        renames[node] = replacement
    for edge in edges:
        for side in ("parent", "child"):
            edge[side] = renames.get(edge[side], edge[side])
    for limb in (
        tree["tree_likeness"].get("neighbor_joining", {}).get("negative_limbs", [])
    ):
        for side in ("parent", "child"):
            limb[side] = renames.get(limb[side], limb[side])

    def attachment_id(index: int) -> str:
        value = f"__phylodigy_internal__:duplicate:{index}"
        while value in used_ids:
            value += ":"
        used_ids.add(value)
        return value

    groups_report = []
    for index, representative in enumerate(representatives):
        group = groups[representative.structural_digest]
        members = [item.artifact_id for item in group]
        groups_report.append(
            {
                "graph_digest": representative.structural_digest,
                "representative_id": representative.artifact_id,
                "artifact_ids": members,
            }
        )
        if len(group) == 1:
            continue
        parent = attachment_id(index)
        for edge in edges:
            if edge["child"] == representative.artifact_id:
                edge["child"] = parent
                break
        for limb in (
            tree["tree_likeness"].get("neighbor_joining", {}).get("negative_limbs", [])
        ):
            if limb["child"] == representative.artifact_id:
                limb["child"] = parent
        edges.extend(
            {
                "parent": parent,
                "child": member,
                "branch_length": 0.0,
                "raw_branch_length": 0.0,
                "length_clamped": False,
            }
            for member in members
        )
    tree.update(
        edges=sorted(edges, key=lambda edge: (edge["parent"], edge["child"])),
        taxa=ids,
        method="neighbor_joining_unique_graphs",
    )
    diagnostics = tree["tree_likeness"]
    diagnostics.update(
        scope="structural_representatives",
        representative_count=len(representatives),
        artifact_count=len(genomes),
        four_point_assessment="exhaustive"
        if len(representatives) >= 4
        else "not_assessed",
    )
    if len(representatives) < 4:
        diagnostics["additive_within_tolerance"] = None
        diagnostics["tree_metric_within_tolerance"] = None
    tree.pop("digest", None)
    tree["digest"] = content_digest(tree)
    result.update(
        artifact_type="phylodigy.compact_architecture_lineage",
        representation_version="1",
        comparison_scope="structural_representatives",
        structural_groups=groups_report,
        genomes=[
            {
                "id": item.artifact_id,
                "digest": item.digest,
                "graph_digest": item.structural_digest,
                "date_min": item.date_min,
                "date_max": item.date_max,
            }
            for item in genomes
        ],
    )
    result["character_matrix"]["artifact_representatives"] = aliases
    result["character_matrix"]["scope"] = "structural_representatives"
    result["digest"] = content_digest(result)
    return result


def lineage_distance_lookup(lineage: Mapping[str, Any]) -> Callable[[str, str], float]:
    """Return a scalar distance lookup without expanding a quadratic matrix."""
    known = set(lineage["tree"]["taxa"])
    aliases = lineage.get("character_matrix", {}).get("artifact_representatives", {})
    distances = {
        tuple(sorted((item["left"]["id"], item["right"]["id"]))): item[
            "graph_distance"
        ]["distance"]
        for item in lineage["comparisons"]
    }

    def distance(left: str, right: str) -> float:
        if left not in known or right not in known:
            raise KeyError("distance endpoints must be known artifact IDs")
        left, right = aliases.get(left, left), aliases.get(right, right)
        if left == right:
            return 0.0
        return distances[tuple(sorted((left, right)))]

    return distance


__all__ = ["infer_compact_lineage", "lineage_distance_lookup"]
