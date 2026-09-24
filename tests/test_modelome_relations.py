from __future__ import annotations

import unittest

from phylodigy.modelome_relations import build_modelome_relations


class ModelomeRelationsTests(unittest.TestCase):
    def test_resolves_only_exact_target_entry_ids_and_keeps_evidence(self):
        declaration = {
            "predicate": "fine_tuned_from",
            "confidence": 0.8,
            "target": {"entry_id": "entry:b", "name": "B"},
            "locator": "registry.json#relation-1",
        }
        report = build_modelome_relations([
            {"id": "entry:a", "model_relations": [declaration]},
            {"id": "entry:b", "canonical_name": "renamed"},
            {"id": "entry:c", "model_relations": [
                {"predicate": "derived_from", "target": {"entry_id": None, "name": "B"}}
            ]},
        ])

        self.assertEqual(report["nodes"], [{"id": "entry:a"}, {"id": "entry:b"}, {"id": "entry:c"}])
        self.assertEqual(len(report["edges"]), 1)
        self.assertEqual(report["edges"][0]["source"], "entry:a")
        self.assertEqual(report["edges"][0]["target"], "entry:b")
        self.assertEqual(report["edges"][0]["predicate"], "fine_tuned_from")
        self.assertEqual(report["edges"][0]["evidence"], declaration)
        self.assertEqual(report["unresolved"][0]["source"], "entry:c")
        self.assertEqual(report["unresolved"][0]["reason"], "target_entry_id_unresolved")

    def test_missing_target_stays_unresolved_even_when_name_matches(self):
        report = build_modelome_relations([
            {"id": "a", "canonical_name": "model B", "model_relations": [
                {"predicate": "parent", "target": {"name": "model B", "entry_id": "other"}}
            ]}
        ])
        self.assertEqual(report["edges"], [])
        self.assertEqual(report["unresolved"][0]["target"], "other")

    def test_cycles_are_retained_and_output_is_deterministic(self):
        entries = [
            {"id": "b", "model_relations": [{"predicate": "related", "target": {"entry_id": "a"}}]},
            {"id": "a", "model_relations": [{"predicate": "related", "target": {"entry_id": "b"}}]},
        ]
        expected = build_modelome_relations(entries)
        self.assertEqual(len(expected["edges"]), 2)
        self.assertEqual(build_modelome_relations(reversed(entries)), expected)

    def test_rejects_invalid_entry_and_relation_shapes(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            build_modelome_relations([{"id": "same"}, {"id": "same"}])
        with self.assertRaisesRegex(ValueError, "target must be an object"):
            build_modelome_relations([{"id": "a", "model_relations": [
                {"predicate": "related", "target": None}
            ]}])
        with self.assertRaisesRegex(ValueError, "predicate must be a non-empty string"):
            build_modelome_relations([{"id": "a", "model_relations": [
                {"predicate": "  ", "target": {"entry_id": "a"}}
            ]}])


if __name__ == "__main__":
    unittest.main()
