"""Canonical schema for a graph-derived architectural genome.

An architectural genome contains an executable operator/layer graph and the
anonymous characters generated from it. It has no slot for a hand-authored
feature call or ontology. Human terminology and provenance live in separate
paper annotations that must point back to content already present here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

from .canonical import canonical_json, content_digest, freeze_json, normalize_json
from .computation_graph import GraphProfile


SCHEMA_VERSION = "3.0.0"
ARCHITECTURAL_GENOME_ARTIFACT_TYPE = "phylodigy.architectural_genome"
PROFILE_ARTIFACT_TYPE = ARCHITECTURAL_GENOME_ARTIFACT_TYPE


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _date_or_none(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be an ISO date or None")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field_name} must use canonical YYYY-MM-DD form")
    return value


@dataclass(frozen=True)
class ArchitecturalGenome:
    """The graph-derived structural record for one immutable artifact."""

    artifact_id: str
    graph_profile: GraphProfile
    artifact_kind: str = "model"
    name: str = ""
    release_date: str | None = None
    date_min: str | None = None
    date_max: str | None = None
    extractor_name: str = "unknown"
    extractor_version: str = "0"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "artifact_kind", _nonempty(self.artifact_kind, "artifact_kind"))
        object.__setattr__(self, "extractor_name", _nonempty(self.extractor_name, "extractor_name"))
        object.__setattr__(self, "extractor_version", _nonempty(self.extractor_version, "extractor_version"))
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        object.__setattr__(self, "name", self.name.strip())
        if not isinstance(self.graph_profile, GraphProfile):
            raise TypeError("graph_profile must be a GraphProfile")
        release_date = _date_or_none(self.release_date, "release_date")
        date_min = _date_or_none(self.date_min, "date_min") or release_date
        date_max = _date_or_none(self.date_max, "date_max") or release_date
        if date_min and date_max and date_min > date_max:
            raise ValueError("date_min cannot be after date_max")
        object.__setattr__(self, "release_date", release_date)
        object.__setattr__(self, "date_min", date_min)
        object.__setattr__(self, "date_max", date_max)
        normalized = normalize_json(dict(self.metadata))
        if not isinstance(normalized, dict):  # pragma: no cover - defensive
            raise TypeError("metadata must normalize to an object")
        object.__setattr__(self, "metadata", freeze_json(normalized))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )

    @property
    def structural_digest(self) -> str:
        """Digest of graph structure, excluding frontend and paper metadata.

        Characters and regions are verified deterministic functions of this
        graph during :class:`GraphProfile` construction.
        """

        return self.graph_profile.graph.structural_digest

    @property
    def character_counts(self) -> dict[str, int]:
        """Anonymous character-state vector used by graph comparisons."""

        return {item.character_id: item.count for item in self.graph_profile.characters}

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        artifact: dict[str, Any] = {"id": self.artifact_id, "kind": self.artifact_kind}
        if self.name:
            artifact["name"] = self.name
        for key in ("release_date", "date_min", "date_max"):
            value = getattr(self, key)
            if value:
                artifact[key] = value
        if self.metadata:
            artifact["metadata"] = normalize_json(self.metadata)
        result: dict[str, Any] = {
            "artifact": artifact,
            "artifact_type": ARCHITECTURAL_GENOME_ARTIFACT_TYPE,
            "extractor": {"name": self.extractor_name, "version": self.extractor_version},
            "graph_profile": self.graph_profile.to_dict(),
            "schema_version": self.schema_version,
        }
        if include_digest:
            result["digest"] = content_digest(result)
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ArchitecturalGenome":
        artifact_type = raw.get("artifact_type")
        if artifact_type != ARCHITECTURAL_GENOME_ARTIFACT_TYPE:
            raise ValueError(
                f"artifact_type must be {ARCHITECTURAL_GENOME_ARTIFACT_TYPE!r}, "
                f"got {artifact_type!r}"
            )
        artifact = raw["artifact"]
        extractor = raw.get("extractor", {})
        result = cls(
            artifact_id=str(artifact["id"]),
            graph_profile=GraphProfile.from_dict(raw["graph_profile"]),
            artifact_kind=str(artifact.get("kind", "model")),
            name=str(artifact.get("name", "")),
            release_date=artifact.get("release_date"),
            date_min=artifact.get("date_min"),
            date_max=artifact.get("date_max"),
            extractor_name=str(extractor.get("name", "unknown")),
            extractor_version=str(extractor.get("version", "0")),
            metadata=artifact.get("metadata", {}),
            schema_version=str(raw.get("schema_version", SCHEMA_VERSION)),
        )
        supplied = raw.get("digest")
        if supplied is not None and str(supplied) != result.digest:
            raise ValueError("architectural genome digest does not match its content")
        return result

    @property
    def digest(self) -> str:
        return content_digest(self.to_dict(include_digest=False))

    def canonical_json(self, *, indent: int | None = None) -> str:
        return canonical_json(self.to_dict(), indent=indent)


ArtifactProfile = ArchitecturalGenome


def merge_profiles(*profiles: ArchitecturalGenome) -> ArchitecturalGenome:
    """Merge metadata only when every input has the identical graph profile.

    Paper annotations are intentionally not merged here. They remain separate
    evidence artifacts so text can never mutate the structural genome.
    """

    if not profiles:
        raise ValueError("at least one architectural genome is required")
    first = profiles[0]
    if any(item.artifact_id != first.artifact_id for item in profiles[1:]):
        raise ValueError("cannot merge genomes for different artifacts")
    if any(item.graph_profile.digest != first.graph_profile.digest for item in profiles[1:]):
        raise ValueError(
            "cannot merge different graph profiles; rebuild them with the same "
            "frontend observations and character radii first"
        )
    names = {item.name for item in profiles if item.name}
    release_dates = {item.release_date for item in profiles if item.release_date}
    if len(names) > 1 or len(release_dates) > 1:
        raise ValueError("artifact identity metadata conflicts across genomes")
    date_mins = [item.date_min for item in profiles if item.date_min]
    date_maxs = [item.date_max for item in profiles if item.date_max]
    values_by_key: dict[str, dict[str, Any]] = {}
    for item in profiles:
        for key, value in item.metadata.items():
            values_by_key.setdefault(key, {})[canonical_json(value)] = value
    metadata: dict[str, Any] = {}
    conflicts: dict[str, list[Any]] = {}
    for key in sorted(values_by_key):
        values = values_by_key[key]
        if len(values) == 1:
            metadata[key] = next(iter(values.values()))
        else:
            conflicts[key] = [values[token] for token in sorted(values)]
    if conflicts:
        metadata["merge_conflicts"] = conflicts
    kinds = {item.artifact_kind for item in profiles}
    return ArchitecturalGenome(
        artifact_id=first.artifact_id,
        graph_profile=first.graph_profile,
        artifact_kind=next(iter(kinds)) if len(kinds) == 1 else "composite",
        name=next(iter(names), first.name),
        release_date=next(iter(release_dates), first.release_date),
        date_min=min(date_mins) if date_mins else None,
        date_max=max(date_maxs) if date_maxs else None,
        extractor_name="phylodigy.merge",
        extractor_version="1",
        metadata=metadata,
    )


__all__ = [
    "ARCHITECTURAL_GENOME_ARTIFACT_TYPE",
    "ArtifactProfile",
    "ArchitecturalGenome",
    "PROFILE_ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "merge_profiles",
]
