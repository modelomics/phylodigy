"""Resolve portable Modelome evidence to exact Hugging Face commits.

This module is deliberately offline. It accepts only explicit provider
identifiers; display names and URLs are not identity evidence.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from .huggingface_profile import _valid_huggingface_repo_id

_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def resolve_modelome_targets(
    entries: Iterable[Mapping[str, Any]],
    *,
    pins: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve each entry only when exact repo and 40-hex revision evidence exists.

    A pin chooses among evidence already present on that entry. It cannot add a
    repository or revision that the entry did not declare.
    """
    if pins is not None and not isinstance(pins, Mapping):
        raise TypeError("pins must be a mapping or None")
    entry_list = list(entries)
    known: dict[str, Mapping[str, Any]] = {}
    for entry in entry_list:
        if not isinstance(entry, Mapping):
            raise TypeError("entries must contain mappings")
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError("every entry must have a non-empty id")
        if entry_id in known:
            raise ValueError(f"duplicate entry id: {entry_id}")
        known[entry_id] = entry
    pin_values = dict(pins or {})
    for entry_id, pin in pin_values.items():
        if not isinstance(entry_id, str) or entry_id not in known:
            raise ValueError(f"pin references unknown entry ID {entry_id!r}")
        if not isinstance(pin, Mapping):
            raise ValueError(f"pin for {entry_id!r} must be a mapping")
        repo_id, revision = pin.get("repo_id"), pin.get("revision")
        if not _valid_target(repo_id, revision):
            raise ValueError(
                f"pin for {entry_id!r} must contain a valid repo_id and 40-hex revision"
            )

    result = []
    for entry in sorted(entry_list, key=lambda row: row["id"]):
        entry_id = entry["id"]
        candidates, has_supported_evidence, has_hf_evidence = _entry_candidates(entry)
        pin = pin_values.get(entry_id)
        if pin is not None:
            desired = {"repo_id": pin["repo_id"], "revision": pin["revision"].lower()}
            selected = next((item for item in candidates if item == desired), None)
            if selected is None:
                raise ValueError(
                    f"pin for {entry_id!r} does not match an evidenced candidate"
                )
            result.append(
                {
                    "entry_id": entry_id,
                    "status": "ready",
                    "reason": "selected evidenced candidate by pin",
                    "target": selected,
                    "candidates": candidates,
                }
            )
        elif len(candidates) == 1:
            result.append(
                {
                    "entry_id": entry_id,
                    "status": "ready",
                    "reason": "one exact Hugging Face repository and revision candidate",
                    "target": candidates[0],
                    "candidates": candidates,
                }
            )
        elif len(candidates) > 1:
            result.append(
                {
                    "entry_id": entry_id,
                    "status": "ambiguous_reference",
                    "reason": "multiple exact Hugging Face repository and revision candidates",
                    "target": None,
                    "candidates": candidates,
                }
            )
        elif has_hf_evidence and not has_supported_evidence:
            result.append(
                {
                    "entry_id": entry_id,
                    "status": "unsupported_reference",
                    "reason": "Hugging Face identifiers do not form a supported exact repository and 40-hex revision pair",
                    "target": None,
                    "candidates": [],
                }
            )
        else:
            result.append(
                {
                    "entry_id": entry_id,
                    "status": "missing_pinned_reference",
                    "reason": "no exact Hugging Face repository and 40-hex revision evidence",
                    "target": None,
                    "candidates": [],
                }
            )
    return result


def _entry_candidates(
    entry: Mapping[str, Any],
) -> tuple[list[dict[str, str]], bool, bool]:
    all_ids = _identifiers(entry.get("identifiers", ()))
    hf_seen = any(namespace.startswith("huggingface:") for namespace, _ in all_ids)
    candidates: set[tuple[str, str]] = set()

    # A fully qualified revision value (repo@commit) is exact evidence by
    # itself. Releases and entry-level identifiers both contribute candidates.
    releases = entry.get("releases", ())
    if isinstance(releases, (list, tuple)):
        for release in releases:
            if not isinstance(release, Mapping):
                continue
            ids = _identifiers(release.get("identifiers", ()))
            hf_seen |= any(namespace.startswith("huggingface:") for namespace, _ in ids)
            candidates.update(_revision_targets(ids))
    candidates.update(_revision_targets(all_ids))
    targets = [
        {"repo_id": repo, "revision": revision} for repo, revision in sorted(candidates)
    ]
    return targets, bool(candidates), hf_seen


def _identifiers(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in value:
        if (
            isinstance(item, Mapping)
            and isinstance(item.get("namespace"), str)
            and isinstance(item.get("value"), str)
        ):
            result.append((item["namespace"], item["value"].strip()))
    return result


def _revision_targets(identifiers: list[tuple[str, str]]) -> set[tuple[str, str]]:
    revisions = {
        value for namespace, value in identifiers if namespace == "huggingface:revision"
    }
    result = set()
    for value in revisions:
        if "@" not in value:
            continue
        repo, sha = value.rsplit("@", 1)
        if _valid_repo(repo) and _SHA.fullmatch(sha):
            result.add((repo, sha.lower()))
    return result


def _valid_repo(value: Any) -> bool:
    return _valid_huggingface_repo_id(value)


def _valid_target(repo_id: Any, revision: Any) -> bool:
    return (
        _valid_repo(repo_id)
        and isinstance(revision, str)
        and bool(_SHA.fullmatch(revision))
    )


__all__ = ["resolve_modelome_targets"]
