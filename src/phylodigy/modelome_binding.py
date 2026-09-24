"""Explicitly bind catalog entries to architectural genome profiles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from .schema import ArchitecturalGenome


def bind_modelome_profiles(
    entries: Iterable[Mapping[str, Any]],
    profiles: Mapping[str, ArchitecturalGenome],
    *,
    bindings: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind profiles by exact IDs or explicit entry-to-profile assignments.

    Names, URLs, provider-native IDs, and other descriptive fields are never
    used to infer identity. Returned genomes use catalog entry IDs so they can
    be placed in the modelome tree without changing the profile itself.
    """

    if not isinstance(profiles, Mapping):
        raise TypeError("profiles must be a mapping of profile IDs to genomes")
    for profile_id, profile in profiles.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError("profile IDs must be non-empty strings")
        if not isinstance(profile, ArchitecturalGenome):
            raise TypeError(f"profile {profile_id!r} is not an ArchitecturalGenome")
        if profile.artifact_id != profile_id:
            raise ValueError(
                f"profile mapping key {profile_id!r} does not match its artifact ID "
                f"{profile.artifact_id!r}"
            )

    entry_list = list(entries)
    entry_ids: list[str] = []
    for entry in entry_list:
        if not isinstance(entry, Mapping):
            raise TypeError("entries must contain mappings")
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError("every catalog entry must have a non-empty id")
        entry_ids.append(entry_id)
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("catalog entry IDs must be unique")

    if bindings is not None and not isinstance(bindings, Mapping):
        raise TypeError("bindings must be a mapping or None")
    assignments = dict(bindings) if bindings is not None else {}
    known_entries = set(entry_ids)
    for entry_id, profile_id in assignments.items():
        if not isinstance(entry_id, str) or not entry_id or entry_id not in known_entries:
            raise ValueError(f"binding references unknown entry ID {entry_id!r}")
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError(f"binding for {entry_id!r} has an invalid profile ID")
        if profile_id not in profiles:
            raise ValueError(
                f"binding for {entry_id!r} references unknown profile ID {profile_id!r}"
            )
    bound_profile_ids = list(assignments.values())
    if len(bound_profile_ids) != len(set(bound_profile_ids)):
        raise ValueError("a profile cannot be explicitly assigned to multiple entries")

    genomes: list[ArchitecturalGenome] = []
    coverage: list[dict[str, str | None]] = []
    used_profile_ids: set[str] = set()
    for entry_id in entry_ids:
        explicit_profile_id = assignments.get(entry_id)
        profile_id = explicit_profile_id
        if profile_id is None and entry_id in profiles:
            profile_id = entry_id
        if profile_id is None:
            coverage.append(
                {
                    "entry_id": entry_id,
                    "status": "missing_profile",
                    "profile_id": None,
                    "reason": "no exact artifact ID match or explicit binding",
                }
            )
            continue
        if profile_id in used_profile_ids:
            raise ValueError(f"profile {profile_id!r} is assigned more than once")
        used_profile_ids.add(profile_id)
        genomes.append(replace(profiles[profile_id], artifact_id=entry_id))
        coverage.append(
            {
                "entry_id": entry_id,
                "status": "bound",
                "profile_id": profile_id,
                "reason": "explicit binding" if explicit_profile_id else "exact artifact ID match",
            }
        )

    return {
        "genomes": genomes,
        "coverage": coverage,
        "unused_profile_ids": sorted(set(profiles) - used_profile_ids),
    }


__all__ = ["bind_modelome_profiles"]
