from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from phylodigy.code_profile import (
    SourceManifest,
    SourceScanPolicy,
    extract_code_profile,
    extract_source_profile,
    source_tree_digest,
)


class InMemorySourceGraphTests(unittest.TestCase):
    def test_mapping_order_encoding_and_newlines_are_deterministic(self):
        left = extract_source_profile(
            {
                "b.json": '{"values": [1, 2]}\n',
                "a.py": "def f(x):\r\n    return x + 1\r\n",
            },
            artifact_id="model:a",
        )
        right = extract_source_profile(
            {
                "a.py": b"# coding: utf-8\ndef f(x):\n    return x + 1\n",
                "b.json": '{"values": [1, 2]}',
            },
            artifact_id="model:a",
        )

        # Content and encoding differences remain provenance, but the parsed
        # structural graph is invariant to file insertion order and newlines.
        left_graphs = [item["graph"] for item in left.source_graph.documents]
        right_graphs = [item["graph"] for item in right.source_graph.documents]
        self.assertEqual(left_graphs, right_graphs)

    def test_source_parser_emits_generic_syntax_not_named_techniques(self):
        profile = extract_source_profile(
            {
                "model.py": """
class ResidualAttentionTransformer:
    def forward(self, value):
        return mystery_op(value, axis=-1)
""",
                "config.json": json.dumps(
                    {
                        "architecture": "FamousFamily",
                        "num_attention_heads": 32,
                        "normalization": "RMSNorm",
                    }
                ),
            },
            artifact_id="model:names-are-not-features",
        )
        raw = profile.to_dict()
        text = json.dumps(raw)

        self.assertEqual(
            profile.metadata["extraction_policy"],
            "generic_source_provenance_only",
        )
        self.assertEqual(raw["artifact_type"], "phylodigy.source_manifest")
        self.assertNotIn("graph_profile", raw)
        self.assertNotIn("traits", raw)
        self.assertNotIn("ontology", raw)
        self.assertNotIn("architecture.attention", text)
        self.assertNotIn("architecture.normalization", text)
        self.assertTrue(
            all(
                node["operation"].startswith("python.")
                or node["operation"].startswith("data.")
                for document in profile.source_graph.documents
                for node in document["graph"]["nodes"]
            )
        )

    def test_arbitrary_unknown_call_survives_as_open_operator_identity(self):
        profile = extract_source_profile(
            {"program.py": "result = vendor.experimental.quantum_gate(value)\n"},
            artifact_id="model:unknown",
        )
        operations = {
            node["operation"]
            for document in profile.source_graph.documents
            for node in document["graph"]["nodes"]
        }

        self.assertIn(
            "python.call:vendor.experimental.quantum_gate",
            operations,
        )
        self.assertFalse(any("family" in operation for operation in operations))

    def test_parse_failures_are_diagnostics_not_negative_features(self):
        profile = extract_source_profile(
            {
                "broken.py": "def broken(:\n",
                "opaque.yaml": "architecture: named-but-unparsed\n",
                "valid.py": "x = 1\n",
            },
            artifact_id="model:partial",
        )
        codes = {item["code"] for item in profile.metadata["diagnostics"]}

        self.assertIn("parse_failed", codes)
        self.assertIn("syntax_not_structurally_parsed", codes)
        self.assertNotIn("traits", profile.to_dict())
        self.assertGreater(profile.source_graph.node_count, 0)

    def test_resource_bounds_are_explicit_and_deterministic(self):
        policy = SourceScanPolicy(max_graph_nodes=3, max_graph_edges=12)
        profile = extract_source_profile(
            {"program.py": "\n".join(f"v{i} = {i}" for i in range(20))},
            artifact_id="model:bounded",
            policy=policy,
        )

        source_graph = profile.source_graph
        self.assertTrue(source_graph.truncated)
        self.assertLessEqual(source_graph.node_count, policy.max_graph_nodes)

    def test_dates_and_graph_schema_round_trip(self):
        profile = extract_source_profile(
            {"program.py": "value = f(input)\n"},
            artifact_id="model:dated",
            release_date="2026-02-03",
        )
        restored = SourceManifest.from_dict(profile.to_dict())

        self.assertEqual(restored, profile)
        self.assertEqual(restored.release_date, "2026-02-03")
        self.assertIsNone(restored.date_min)
        self.assertIsNone(restored.date_max)
        self.assertTrue(
            all(
                document["graph"]["frontend"] in {"python.ast", "json", "toml"}
                for document in profile.source_graph.documents
            )
        )


class RepositorySourceGraphTests(unittest.TestCase):
    def test_relocated_checkout_preserves_structural_digest(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir) / "checkout-a"
            second = Path(second_dir) / "renamed-checkout"
            first.mkdir()
            second.mkdir()
            for root in (first, second):
                (root / "program.py").write_text(
                    "def compute(x):\n    return novel_operator(x)\n",
                    encoding="utf-8",
                )

            left = extract_code_profile(first, artifact_id="model:a")
            right = extract_code_profile(second, artifact_id="model:a")

        self.assertEqual(left.source_graph.digest, right.source_graph.digest)
        self.assertEqual(left.metadata["source_tree_digest"], right.metadata["source_tree_digest"])

    def test_ignored_trees_and_size_policy_do_not_create_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "program.py").write_text("x = f(y)\n", encoding="utf-8")
            (root / ".old").mkdir()
            (root / ".old" / "named.py").write_text(
                "class TransformerAttention: pass\n", encoding="utf-8"
            )
            (root / "large.py").write_text("x" * 200, encoding="utf-8")
            policy = SourceScanPolicy(max_file_bytes=32)
            profile = extract_code_profile(root, artifact_id="model:a", policy=policy)

            self.assertEqual(source_tree_digest(root, policy=policy), profile.metadata["source_tree_digest"])

        text = json.dumps(profile.to_dict()).lower()
        self.assertNotIn("transformerattention", text)
        self.assertTrue(
            any(item["code"] == "file_too_large" for item in profile.metadata["diagnostics"])
        )


if __name__ == "__main__":
    unittest.main()
