from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phylodigy.canonical import content_digest
from phylodigy.modelome_input import ModelomeInputError, read_modelome_entries


ENTRY = {
    "id": "entry:one",
    "canonical_name": "One",
    "aliases": [],
    "identifiers": [],
    "tags": [],
    "members": [],
    "resources": [],
    "releases": [],
    "model_relations": [],
}


def write_bundle(root: Path, entries: list[dict], *, manifest_changes=None) -> dict:
    """Write the portable bundle contract: JSONL bytes plus manifest digests."""
    raw = b"".join(
        (json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for entry in entries
    )
    (root / "entries.jsonl").write_bytes(raw)
    manifest = {
        "format": "modelome-entry-corpus-v1",
        "entry_count": len(entries),
        "entry_sha256": content_digest(entries),
        "files": {"entries.jsonl": hashlib.sha256(raw).hexdigest()},
    }
    manifest.update(manifest_changes or {})
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


class ModelomeInputEdgeTests(unittest.TestCase):
    def test_rejects_each_manifest_integrity_failure(self):
        cases = (
            ({"format": "other"}, "format"),
            ({"entry_count": 2}, "entry_count"),
            ({"entry_sha256": "0" * 64}, "entry_sha256"),
            ({"files": {}}, "SHA-256"),
            ({"files": {"entries.jsonl": "0" * 64}}, "SHA-256"),
        )
        for change, message in cases:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                write_bundle(Path(directory), [ENTRY], manifest_changes=change)
                with self.assertRaisesRegex(ModelomeInputError, message):
                    read_modelome_entries(directory)

    def test_rejects_missing_or_non_object_manifest(self):
        for manifest_text in (None, "[]", "not json"):
            with self.subTest(manifest_text=manifest_text), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_bundle(root, [ENTRY])
                if manifest_text is None:
                    (root / "manifest.json").unlink()
                else:
                    (root / "manifest.json").write_text(manifest_text, encoding="utf-8")
                with self.assertRaises(ModelomeInputError):
                    read_modelome_entries(root)

    def test_rejects_malformed_json_and_jsonl(self):
        malformed_inputs = (b"", b" \n", b"\xff", b"{\"id\":", b"{}\nnot-json\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entries.jsonl"
            for raw in malformed_inputs:
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    with self.assertRaises(ModelomeInputError):
                        read_modelome_entries(path)

    def test_rejects_structurally_invalid_entries(self):
        invalid = (
            ["row"],
            [{"id": "entry:x"}],
            [{"id": "entry:x", "canonical_name": "X", "aliases": "not-an-array"}],
            [{"id": "entry:x", "canonical_name": "X"}, {"id": "entry:x", "canonical_name": "Y"}],
            [{"id": " ", "canonical_name": "X"}],
            [{"id": "entry:x", "canonical_name": " "}],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entries.json"
            for entries in invalid:
                with self.subTest(entries=entries):
                    path.write_text(json.dumps(entries), encoding="utf-8")
                    with self.assertRaises(ModelomeInputError):
                        read_modelome_entries(path)

    def test_rejects_non_finite_json_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entries.json"
            path.write_text(
                '[{"id":"entry:x","canonical_name":"X","evidence":NaN}]',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ModelomeInputError, "invalid JSON values"):
                read_modelome_entries(path)


if __name__ == "__main__":
    unittest.main()
