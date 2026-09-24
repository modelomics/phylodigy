"""Evidence-grounded contact records with graph diagnostics.

Graph similarity never creates or orients an edge in this module. Vertical
history belongs to the graph-character phylogeny in :mod:`phylodigy.lineage`.
Only explicit, endpoint-grounded provenance, awareness, or character-transfer
records may create a contact edge. Graph distance and anonymous-character
overlap are retained beside each pair solely as diagnostics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

from .canonical import canonical_json, content_digest, freeze_json, normalize_json
from .computation_graph import compare_graphs
from .schema import ArchitecturalGenome


CausalRole = Literal["provenance", "character_transfer", "awareness"]
CAUSAL_ROLES = frozenset({"provenance", "character_transfer", "awareness"})


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _strength(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("strength must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("strength must be between zero and one")
    return result


@dataclass(frozen=True)
class ContactEvidence:
    """One explicit relationship record between represented artifacts."""

    source_id: str
    target_id: str
    strength: float = 1.0
    kind: str = "explicit_provenance"
    character_id: str | None = None
    locator: str = ""
    evidence_tier: str = "curated"
    causal_role: CausalRole | str = "provenance"
    character_value: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        object.__setattr__(self, "kind", _text(self.kind, "kind"))
        object.__setattr__(self, "strength", _strength(self.strength))
        if not isinstance(self.locator, str):
            raise TypeError("locator must be a string")
        object.__setattr__(self, "locator", self.locator.strip())
        if self.source_id == self.target_id:
            raise ValueError("contact evidence endpoints must be different")
        if self.character_id is not None:
            object.__setattr__(
                self, "character_id", _text(self.character_id, "character_id")
            )
        if self.evidence_tier not in {"code", "paper", "curated"}:
            raise ValueError("evidence_tier must be code, paper, or curated")
        role = str(self.causal_role).strip().casefold()
        if role not in CAUSAL_ROLES:
            raise ValueError(f"causal_role must be one of {sorted(CAUSAL_ROLES)!r}")
        if role == "character_transfer" and self.character_id is None:
            raise ValueError("character_transfer evidence requires character_id")
        if role != "character_transfer" and self.character_id is not None:
            raise ValueError("character_id is only valid for character_transfer evidence")
        if self.character_value is not None and self.character_id is None:
            raise ValueError("character_value requires character_id")
        object.__setattr__(self, "causal_role", role)
        object.__setattr__(self, "character_value", freeze_json(self.character_value))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "causal_role": self.causal_role,
            "evidence_tier": self.evidence_tier,
            "kind": self.kind,
            "source_id": self.source_id,
            "strength": self.strength,
            "target_id": self.target_id,
        }
        if self.character_id is not None:
            result["character_id"] = self.character_id
        if self.character_value is not None:
            result["character_value"] = normalize_json(self.character_value)
        if self.locator:
            result["locator"] = self.locator
        return result

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ContactEvidence":
        return cls(
            source_id=raw.get("source_id", raw.get("source")),
            target_id=raw.get("target_id", raw.get("target")),
            strength=raw.get("strength", raw.get("confidence", 1.0)),
            kind=raw.get("kind", "explicit_provenance"),
            character_id=raw.get("character_id"),
            locator=str(raw.get("locator", "")),
            evidence_tier=str(raw.get("evidence_tier", "curated")),
            causal_role=raw.get("causal_role", "provenance"),
            character_value=raw.get("character_value"),
        )


@dataclass(frozen=True)
class ContactInferenceConfig:
    """Output controls that cannot turn similarity into contact evidence."""

    include_unlinked_pair_diagnostics: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.include_unlinked_pair_diagnostics, bool):
            raise ValueError("include_unlinked_pair_diagnostics must be boolean")


def infer_contact_network(
    profiles: Sequence[ArchitecturalGenome | Mapping[str, Any]],
    *,
    edge_evidence: (
        Iterable[ContactEvidence | Mapping[str, Any]]
        | Mapping[tuple[str, str], float]
        | None
    ) = None,
    config: ContactInferenceConfig | None = None,
) -> dict[str, Any]:
    """Build dated contact edges only from explicit grounded evidence.

    Empty evidence always yields an empty edge set and parent forest, regardless
    of graph similarity. Inadmissibly dated evidence is retained in diagnostics
    but creates no edge.
    """

    settings = config or ContactInferenceConfig()
    genomes = tuple(
        sorted((_as_genome(item) for item in profiles), key=lambda item: item.artifact_id)
    )
    by_id = {item.artifact_id: item for item in genomes}
    if len(by_id) != len(genomes):
        raise ValueError("artifact ids must be unique")
    evidence = _normalize_evidence(edge_evidence)
    unknown = sorted(
        {
            endpoint
            for item in evidence
            for endpoint in (item.source_id, item.target_id)
            if endpoint not in by_id
        }
    )
    if unknown:
        raise ValueError("edge evidence references unknown artifacts: " + ", ".join(unknown))
    _validate_character_evidence(evidence, by_id)

    evidence_by_pair: dict[tuple[str, str], list[ContactEvidence]] = {}
    for item in evidence:
        evidence_by_pair.setdefault((item.source_id, item.target_id), []).append(item)

    pair_diagnostics: list[dict[str, Any]] = []
    temporal_exclusions: list[dict[str, Any]] = []
    for target in genomes:
        for source in genomes:
            if source.artifact_id == target.artifact_id:
                continue
            pair = (source.artifact_id, target.artifact_id)
            pair_evidence = evidence_by_pair.get(pair, ())
            if not settings.include_unlinked_pair_diagnostics and not pair_evidence:
                continue
            temporal = _temporal_relation(source, target)
            diagnostic = _pair_diagnostic(source, target, pair_evidence, temporal)
            pair_diagnostics.append(diagnostic)
            if pair_evidence and temporal != "earlier":
                temporal_exclusions.append(
                    {
                        "evidence": [item.to_dict() for item in pair_evidence],
                        "reason": temporal,
                        "source_id": source.artifact_id,
                        "target_id": target.artifact_id,
                    }
                )

    edges: list[dict[str, Any]] = []
    admitted_evidence: dict[tuple[str, str], tuple[ContactEvidence, ...]] = {}
    for pair, items in sorted(evidence_by_pair.items()):
        source, target = by_id[pair[0]], by_id[pair[1]]
        if _temporal_relation(source, target) != "earlier":
            continue
        ordered = tuple(sorted(items, key=lambda item: canonical_json(item.to_dict())))
        admitted_evidence[pair] = ordered
        for role in sorted({item.causal_role for item in ordered}):
            role_evidence = tuple(item for item in ordered if item.causal_role == role)
            edge: dict[str, Any] = {
                "edge_type": "transfer" if role == "character_transfer" else role,
                "evidence": [item.to_dict() for item in role_evidence],
                "source_id": pair[0],
                "target_id": pair[1],
            }
            if role == "character_transfer":
                edge["character_ids"] = sorted(
                    {item.character_id for item in role_evidence if item.character_id}
                )
            edges.append(edge)

    result: dict[str, Any] = {
        "artifact_type": "phylodigy.graph_contact_network",
        "artifact_version": "2.0.0",
        "character_sources": _character_sources(genomes, admitted_evidence),
        "configuration": asdict(settings),
        "diagnostics": {
            "missing_dates": sorted(
                item.artifact_id
                for item in genomes
                if item.date_min is None or item.date_max is None
            ),
            "pair_diagnostics": sorted(
                pair_diagnostics,
                key=lambda item: (item["target_id"], item["source_id"]),
            ),
            "temporal_exclusions": sorted(temporal_exclusions, key=canonical_json),
        },
        "edge_evidence": [item.to_dict() for item in evidence],
        "edges": sorted(edges, key=canonical_json),
        "method": {
            "contact_rule": "explicit grounded evidence only",
            "distance_rule": "graph distance and overlap are diagnostics only",
            "paper_rule": "paper evidence cannot alter a genome or create a structural parent",
            "temporal_rule": "source.date_max must be strictly earlier than target.date_min",
            "vertical_history": "inferred separately by graph-character phylogeny",
        },
        "nodes": [_node(item) for item in genomes],
        # Retained as an explicit boundary assertion for downstream readers.
        "primary_parent_forest": [],
    }
    normalized = normalize_json(result)
    normalized["digest"] = content_digest(normalized)
    return normalize_json(normalized)


def build_contact_network(
    profiles: Sequence[ArchitecturalGenome | Mapping[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    """Alias for :func:`infer_contact_network`."""

    return infer_contact_network(profiles, **kwargs)


def _as_genome(raw: ArchitecturalGenome | Mapping[str, Any]) -> ArchitecturalGenome:
    if isinstance(raw, ArchitecturalGenome):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("profiles must contain ArchitecturalGenome objects or mappings")
    return ArchitecturalGenome.from_dict(raw)


def _normalize_evidence(
    raw: (
        Iterable[ContactEvidence | Mapping[str, Any]]
        | Mapping[tuple[str, str], float]
        | None
    ),
) -> tuple[ContactEvidence, ...]:
    if raw is None:
        return ()
    items: list[ContactEvidence] = []
    if isinstance(raw, Mapping):
        for pair, strength in raw.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("pair-weight evidence must use (source_id, target_id) keys")
            items.append(ContactEvidence(str(pair[0]), str(pair[1]), strength=float(strength)))
    else:
        for item in raw:
            items.append(
                item if isinstance(item, ContactEvidence) else ContactEvidence.from_mapping(item)
            )
    unique = {canonical_json(item.to_dict()): item for item in items}
    return tuple(unique[key] for key in sorted(unique))


def _validate_character_evidence(
    evidence: Sequence[ContactEvidence],
    by_id: Mapping[str, ArchitecturalGenome],
) -> None:
    for item in evidence:
        if item.causal_role != "character_transfer":
            continue
        source = by_id[item.source_id]
        target = by_id[item.target_id]
        if (
            item.character_id not in source.character_counts
            or item.character_id not in target.character_counts
        ):
            raise ValueError(
                "character_transfer evidence is not grounded in both endpoint "
                f"graphs: {item.source_id}->{item.target_id}/{item.character_id}"
            )
        if (
            item.character_value is not None
            and canonical_json(item.character_value)
            != canonical_json(target.character_counts[item.character_id])
        ):
            raise ValueError(
                "character_transfer value does not match the target graph's "
                f"derived count: {item.target_id}/{item.character_id}"
            )


def _pair_diagnostic(
    source: ArchitecturalGenome,
    target: ArchitecturalGenome,
    evidence: Sequence[ContactEvidence],
    temporal_relation: str,
) -> dict[str, Any]:
    left = source.character_counts
    right = target.character_counts
    shared = sorted(set(left) & set(right))
    shared_mass = sum(min(left[item], right[item]) for item in shared)
    total_mass = sum(left.values()) + sum(right.values())
    similarity = 1.0 if total_mass == 0 else 2.0 * shared_mass / total_mass
    return {
        "admitted_contact_evidence": bool(evidence and temporal_relation == "earlier"),
        "evidence": [
            item.to_dict() for item in sorted(evidence, key=lambda item: canonical_json(item.to_dict()))
        ],
        "gap_days": _gap_days(source, target),
        "graph_distance": compare_graphs(source.graph_profile, target.graph_profile).to_dict(),
        "graph_similarity": round(float(similarity), 12),
        "shared_character_count": len(shared),
        "shared_character_mass": shared_mass,
        "shared_character_ids": shared,
        "source_id": source.artifact_id,
        "target_id": target.artifact_id,
        "temporal_relation": temporal_relation,
    }


def _character_sources(
    genomes: Sequence[ArchitecturalGenome],
    evidence: Mapping[tuple[str, str], Sequence[ContactEvidence]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for target in genomes:
        for character_id, count in sorted(target.character_counts.items()):
            donors = sorted(
                source
                for (source, child), items in evidence.items()
                if child == target.artifact_id
                and any(
                    item.causal_role == "character_transfer"
                    and item.character_id == character_id
                    for item in items
                )
            )
            record: dict[str, Any] = {
                "character_id": character_id,
                "count": count,
                "origin_hypothesis": "transfer" if donors else "first_observed",
                "target_id": target.artifact_id,
            }
            if donors:
                record["source_ids"] = donors
                if len(donors) == 1:
                    record["source_id"] = donors[0]
            records.append(record)
    return records


def _temporal_relation(source: ArchitecturalGenome, target: ArchitecturalGenome) -> str:
    source_max = _date(source.date_max or source.release_date)
    target_min = _date(target.date_min or target.release_date)
    if source_max is None or target_min is None:
        return "missing_date"
    return "earlier" if source_max < target_min else "not_strictly_earlier"


def _gap_days(source: ArchitecturalGenome, target: ArchitecturalGenome) -> int | None:
    source_max = _date(source.date_max or source.release_date)
    target_min = _date(target.date_min or target.release_date)
    if source_max is None or target_min is None:
        return None
    return max(0, (target_min - source_max).days)


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _node(genome: ArchitecturalGenome) -> dict[str, Any]:
    return {
        "character_count": len(genome.character_counts),
        "date_max": genome.date_max,
        "date_min": genome.date_min,
        "graph_digest": genome.structural_digest,
        "id": genome.artifact_id,
        "kind": genome.artifact_kind,
        "name": genome.name,
        "release_date": genome.release_date,
    }


__all__ = [
    "CAUSAL_ROLES",
    "CausalRole",
    "ContactEvidence",
    "ContactInferenceConfig",
    "build_contact_network",
    "infer_contact_network",
]
