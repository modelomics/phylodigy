"""Phylogenetic analysis over graph-derived architectural genomes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .canonical import content_digest, normalize_json
from .profile_comparison import (
    build_vertical_backbone,
    pairwise_profile_comparisons,
    selected_character_counts,
)
from .schema import ArchitecturalGenome


LINEAGE_ANALYSIS_VERSION = "2"


@dataclass(frozen=True)
class LineageAnalysisConfig:
    """Numerical settings for graph-character phylogeny."""

    radii: tuple[int, ...] = (0, 1, 2, 3)
    radius_weights: Mapping[int, float] = field(default_factory=dict)
    region_weight: float = 1.0
    four_point_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in self.radii):
            raise TypeError("radii must contain integers")
        radii = tuple(sorted(set(self.radii)))
        if not radii or radii[0] < 0:
            raise ValueError("radii must contain non-negative integers")
        weights: dict[int, float] = {}
        for radius, raw in self.radius_weights.items():
            if isinstance(radius, bool) or not isinstance(radius, int):
                raise TypeError("radius weight keys must be integers")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError("radius weights must be numeric")
            value = float(raw)
            if radius not in radii:
                raise ValueError("radius weights may only reference selected radii")
            if not math.isfinite(value) or value <= 0:
                raise ValueError("radius weights must be finite and strictly positive")
            weights[radius] = value
        region_weight = float(self.region_weight)
        tolerance = float(self.four_point_tolerance)
        if not math.isfinite(region_weight) or region_weight <= 0:
            raise ValueError("region_weight must be finite and strictly positive")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("four_point_tolerance must be finite and non-negative")
        object.__setattr__(self, "radii", radii)
        object.__setattr__(
            self,
            "radius_weights",
            MappingProxyType(dict(sorted(weights.items()))),
        )
        object.__setattr__(self, "region_weight", region_weight)
        object.__setattr__(self, "four_point_tolerance", tolerance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "four_point_tolerance": self.four_point_tolerance,
            "radii": list(self.radii),
            "radius_weights": {
                str(key): value for key, value in self.radius_weights.items()
            },
            "region_weight": self.region_weight,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> LineageAnalysisConfig:
        """Load a config from an in-memory or JSON-decoded mapping.

        JSON object keys are necessarily strings, while the runtime model uses
        integer radii.  Only canonical non-negative decimal keys are converted;
        ambiguous spellings remain validation errors.
        """

        values = dict(raw)
        raw_weights = values.get("radius_weights", {})
        if not isinstance(raw_weights, Mapping):
            raise TypeError("radius_weights must be a mapping")
        weights: dict[int, Any] = {}
        for raw_radius, weight in raw_weights.items():
            if isinstance(raw_radius, bool):
                raise TypeError("radius weight keys must be integers")
            if isinstance(raw_radius, int):
                radius = raw_radius
            elif isinstance(raw_radius, str):
                if (
                    not raw_radius
                    or any(character not in "0123456789" for character in raw_radius)
                    or (len(raw_radius) > 1 and raw_radius.startswith("0"))
                ):
                    raise TypeError(
                        "JSON radius weight keys must be canonical non-negative integers"
                    )
                radius = int(raw_radius)
            else:
                raise TypeError("radius weight keys must be integers")
            if radius in weights:
                raise ValueError("duplicate radius weight after JSON key conversion")
            weights[radius] = weight
        values["radius_weights"] = weights
        return cls(**values)


def infer_lineage_network(
    profiles: Iterable[ArchitecturalGenome],
    *,
    config: LineageAnalysisConfig | Mapping[str, Any] | None = None,
    evidence: Iterable[Mapping[str, Any]] = (),
    **removed_options: Any,
) -> dict[str, Any]:
    """Infer a graph-character distance phylogeny and retain external evidence.

    ``evidence`` may contain chronology, citation, or provenance records. It is
    serialized beside the tree but never changes graph characters or pairwise
    structural distances. Downstream homology/convergence models can use that
    separate channel explicitly.
    """

    if removed_options:
        names = ", ".join(sorted(removed_options))
        raise TypeError(f"unsupported non-graph lineage options: {names}")
    if config is None:
        selected = LineageAnalysisConfig()
    elif isinstance(config, LineageAnalysisConfig):
        selected = config
    elif isinstance(config, Mapping):
        selected = LineageAnalysisConfig.from_mapping(config)
    else:
        raise TypeError("config must be LineageAnalysisConfig, a mapping, or None")
    genomes = tuple(sorted(profiles, key=lambda item: item.artifact_id))
    if len(genomes) < 2:
        raise ValueError("at least two architectural genomes are required")
    if len({item.artifact_id for item in genomes}) != len(genomes):
        raise ValueError("artifact IDs must be unique")
    comparisons = pairwise_profile_comparisons(
        genomes,
        radii=selected.radii,
        radius_weights=selected.radius_weights,
        region_weight=selected.region_weight,
    )
    tree = build_vertical_backbone(
        comparisons=comparisons,
        tolerance=selected.four_point_tolerance,
    )
    selected_counts = {
        genome.artifact_id: selected_character_counts(
            genome, radii=selected.radii
        )
        for genome in genomes
    }
    character_ids = sorted(
        {key for counts in selected_counts.values() for key in counts}
    )
    character_matrix = {
        genome.artifact_id: [
            selected_counts[genome.artifact_id].get(character_id, 0)
            for character_id in character_ids
        ]
        for genome in genomes
    }
    external_evidence = [normalize_json(dict(item)) for item in evidence]
    external_evidence.sort(key=lambda item: content_digest(item))
    result: dict[str, Any] = {
        "analysis_version": LINEAGE_ANALYSIS_VERSION,
        "artifact_type": "phylodigy.architecture_lineage",
        "character_matrix": {
            "character_ids": character_ids,
            "counts_by_artifact": character_matrix,
            "source": "operator_layer_graph_only",
        },
        "comparisons": [item.to_dict() for item in comparisons],
        "config": selected.to_dict(),
        "external_evidence": external_evidence,
        "genomes": [
            {
                "date_max": item.date_max,
                "date_min": item.date_min,
                "digest": item.digest,
                "graph_digest": item.structural_digest,
                "id": item.artifact_id,
            }
            for item in genomes
        ],
        "structural_evidence_boundary": {
            "paper_can_create_characters": False,
            "structural_source": "operator_layer_graph",
        },
        "tree": tree,
    }
    result["digest"] = content_digest(result)
    return result


__all__ = [
    "LINEAGE_ANALYSIS_VERSION",
    "LineageAnalysisConfig",
    "infer_lineage_network",
]
