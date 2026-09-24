"""Canonical, content-addressed serialization helpers.

Persisted artifact profiles must compare byte-for-byte across runs. This
module deliberately avoids timestamps, random identifiers, and platform
specific representations.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any


def normalize_json(value: Any) -> Any:
    """Return *value* as a recursively canonical JSON-compatible object."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("non-finite floats are not canonical JSON values")
        # Collapse negative zero, which otherwise has two textual encodings.
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        return {key: normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [normalize_json(item) for item in value]
        return sorted(normalized, key=lambda item: canonical_json(item))
    if hasattr(value, "to_dict"):
        return normalize_json(value.to_dict())
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def freeze_json(value: Any) -> Any:
    """Return a recursively immutable canonical JSON value.

    Content-addressed dataclasses must not retain caller-owned dictionaries or
    lists: mutating either after construction would otherwise change a digest
    while leaving derived fields stale.  ``MappingProxyType`` and tuples keep
    the public values mapping/sequence-like without introducing a custom data
    model.  Call :func:`normalize_json` to obtain a fresh JSON-compatible copy.
    """

    normalized = normalize_json(value)

    def freeze(item: Any) -> Any:
        if isinstance(item, dict):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, list):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(normalized)


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Serialize using a stable key order and compact UTF-8 representation."""
    normalized = normalize_json(value)
    if indent is None:
        return json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        sort_keys=True,
    )


def content_digest(value: Any) -> str:
    """Return the SHA-256 digest of the canonical JSON representation."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(value, indent=2) + "\n", encoding="utf-8")
