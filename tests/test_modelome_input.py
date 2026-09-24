from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from phylodigy.canonical import content_digest
from phylodigy.modelome_input import ModelomeInputError, read_modelome_entries


ENTRY = {
    "id": "entry:x",
    "canonical_name": "Example",
    "aliases": [],
    "identifiers": [],
    "tags": [],
    "members": [],
    "resources": [{"url": "https://example.test", "evidence": {"raw": [1, 2]}}],
    "releases": [],
    "model_relations": [],
}


class ModelomeInputTests(unittest.TestCase):
    def test_reads_json_array_and_preserves_nested_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entries.json"
            path.write_text(json.dumps([ENTRY]), encoding="utf-8")
            expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            report = read_modelome_entries(path)
        self.assertEqual(report["entries"][0]["resources"], ENTRY["resources"])
        self.assertIsNone(report["manifest"])
        self.assertEqual(report["source_digest"], expected_digest)

    def test_verifies_bundle_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = (json.dumps(ENTRY, sort_keys=True) + "\n").encode()
            (root / "entries.jsonl").write_bytes(raw)
            manifest = {
                "format": "modelome-entry-corpus-v1",
                "entry_count": 1,
                "entry_sha256": content_digest([ENTRY]),
                "files": {"entries.jsonl": hashlib.sha256(raw).hexdigest()},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            report = read_modelome_entries(root)
        self.assertEqual(report["manifest"], manifest)
        self.assertEqual(report["entries"], [ENTRY])

    def test_rejects_duplicate_or_missing_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entries.jsonl"
            path.write_text(json.dumps(ENTRY) + "\n" + json.dumps(ENTRY), encoding="utf-8")
            with self.assertRaisesRegex(ModelomeInputError, "duplicate"):
                read_modelome_entries(path)
            path.write_text(json.dumps({"canonical_name": "No ID"}), encoding="utf-8")
            with self.assertRaisesRegex(ModelomeInputError, "id"):
                read_modelome_entries(path)

    def test_rejects_tampered_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = (json.dumps(ENTRY) + "\n").encode()
            (root / "entries.jsonl").write_bytes(raw)
            (root / "manifest.json").write_text(json.dumps({
                "format": "modelome-entry-corpus-v1", "entry_count": 1,
                "entry_sha256": "bad", "files": {"entries.jsonl": hashlib.sha256(raw).hexdigest()},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ModelomeInputError, "entry_sha256"):
                read_modelome_entries(root)

    def test_accepts_valid_empty_bundle_but_not_plain_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = b""
            (root / "entries.jsonl").write_bytes(raw)
            manifest = {
                "format": "modelome-entry-corpus-v1",
                "entry_count": 0,
                "entry_sha256": content_digest([]),
                "files": {"entries.jsonl": hashlib.sha256(raw).hexdigest()},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(read_modelome_entries(root)["entries"], [])
            plain = Path(directory) / "empty.jsonl"
            plain.write_bytes(raw)
            with self.assertRaisesRegex(ModelomeInputError, "empty"):
                read_modelome_entries(plain)

    def test_rejects_lone_surrogate_as_invalid_json_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entries.json"
            path.write_text(
                '[{"id":"entry:x","canonical_name":"\\ud800"}]',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ModelomeInputError, "invalid JSON values"):
                read_modelome_entries(path)


if __name__ == "__main__":
    unittest.main()
