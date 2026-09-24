"""Read portable Modelome entry corpora without importing Modelome."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from phylodigy.canonical import content_digest


class ModelomeInputError(ValueError):
    """The supplied Modelome entry corpus is malformed or fails verification."""


_FORMAT = "modelome-entry-corpus-v1"
_LIST_FIELDS = ("aliases", "identifiers", "tags", "members", "resources", "releases", "model_relations")


def read_modelome_entries(path: str | Path) -> dict[str, Any]:
    """Read an entry JSON/JSONL file or verified portable bundle directory.

    Directory inputs are verified against their manifest. Standalone files have
    no signed provenance; their source digest is the SHA-256 of the input bytes.
    """
    source = Path(path)
    bundled = source.is_dir()
    entries_path = source / "entries.jsonl" if bundled else source
    try:
        raw = entries_path.read_bytes()
    except OSError as error:
        raise ModelomeInputError(f"cannot read Modelome entries: {entries_path}") from error
    entries = _parse_entries(raw, allow_empty=bundled)
    _validate_entries(entries)
    digest = hashlib.sha256(raw).hexdigest()
    manifest: dict[str, Any] | None = None
    if bundled:
        try:
            manifest_raw = (source / "manifest.json").read_bytes()
            manifest = json.loads(manifest_raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelomeInputError("bundle manifest.json is missing or invalid") from error
        if not isinstance(manifest, dict):
            raise ModelomeInputError("bundle manifest must be an object")
        if manifest.get("format") != _FORMAT:
            raise ModelomeInputError("unsupported Modelome bundle format")
        files = manifest.get("files")
        if not isinstance(files, dict) or files.get("entries.jsonl") != digest:
            raise ModelomeInputError("entries.jsonl SHA-256 does not match manifest")
        if manifest.get("entry_count") != len(entries):
            raise ModelomeInputError("entry_count does not match entries.jsonl")
        if manifest.get("entry_sha256") != content_digest(entries):
            raise ModelomeInputError("entry_sha256 does not match parsed entries")
    return {"entries": entries, "manifest": manifest, "source_digest": digest}


def _parse_entries(raw: bytes, *, allow_empty: bool = False) -> list[Any]:
    try:
        text = raw.decode("utf-8")
        if not text.strip() and allow_empty:
            return []
        if not text.strip():
            raise ModelomeInputError("Modelome entry input is empty")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelomeInputError("Modelome entry input is not valid JSON or JSONL") from error
    if isinstance(parsed, dict):
        if "entries" in parsed:
            parsed = parsed["entries"]
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        raise ModelomeInputError("Modelome input must contain a JSON array or JSONL rows")
    return parsed


def _validate_entries(entries: list[Any]) -> None:
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ModelomeInputError(f"entry {index} must be an object")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ModelomeInputError(f"entry {index} is missing a valid id")
        if identifier in seen:
            raise ModelomeInputError(f"duplicate entry id: {identifier}")
        seen.add(identifier)
        if not isinstance(entry.get("canonical_name"), str) or not entry["canonical_name"].strip():
            raise ModelomeInputError(f"entry {index} is missing a valid canonical_name")
        for field in _LIST_FIELDS:
            if field in entry and not isinstance(entry[field], list):
                raise ModelomeInputError(f"entry {index} field {field} must be an array")
        # Entry payloads are intentionally preserved as supplied. Validate they
        # are JSON values without inferring identity from names or evidence.
        try:
            json.dumps(entry, allow_nan=False, ensure_ascii=False).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as error:
            raise ModelomeInputError(f"entry {index} contains invalid JSON values") from error


__all__ = ["ModelomeInputError", "read_modelome_entries"]
