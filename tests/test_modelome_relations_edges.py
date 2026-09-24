from __future__ import annotations

import unittest

from phylodigy.modelome_relations import build_modelome_relations


class ModelomeRelationEdgeCasesTests(unittest.TestCase):
    def test_output_order_is_deterministic_across_entry_and_relation_order(self):
        entries = [
            {
                "id": "z",
                "model_relations": [
                    {"predicate": "same", "target": {"entry_id": "a"}, "rank": 2},
                    {"predicate": "same", "target": {"entry_id": "a"}, "rank": 1},
                    {"predicate": "other", "target": {"entry_id": "missing"}},
                ],
            },
            {"id": "a"},
        ]
        reordered = [
            {"id": "a"},
            {
                "id": "z",
                "model_relations": list(reversed(entries[0]["model_relations"])),
            },
        ]

        self.assertEqual(
            build_modelome_relations(entries), build_modelome_relations(reordered)
        )

    def test_repeated_predicates_and_duplicate_declarations_are_preserved(self):
        declaration = {"predicate": "derived_from", "target": {"entry_id": "b"}}
        report = build_modelome_relations([
            {"id": "a", "model_relations": [declaration, dict(declaration)]},
            {"id": "b"},
        ])

        self.assertEqual(len(report["edges"]), 2)
        self.assertEqual([edge["predicate"] for edge in report["edges"]], [
            "derived_from", "derived_from"
        ])
        self.assertEqual([edge["evidence"] for edge in report["edges"]], [
            declaration, declaration
        ])

    def test_dangling_ids_are_unresolved_without_name_based_matching(self):
        report = build_modelome_relations([
            {
                "id": "a",
                "canonical_name": "Known by name",
                "model_relations": [{
                    "predicate": "related_to",
                    "target": {"entry_id": "unknown-id", "name": "Known by name"},
                }],
            },
        ])

        self.assertEqual(report["edges"], [])
        self.assertEqual(len(report["unresolved"]), 1)
        self.assertEqual(report["unresolved"][0]["target"], "unknown-id")
        self.assertEqual(report["unresolved"][0]["reason"], "target_entry_id_unresolved")

    def test_evidence_is_preserved_as_supplied_for_resolved_and_dangling_edges(self):
        resolved = {
            "predicate": "fine_tuned_from",
            "target": {"entry_id": "b", "name": "B"},
            "confidence": 0.75,
            "sources": [{"uri": "example:paper", "pages": [2, 5]}],
            "extensions": {"reviewer": "human"},
        }
        dangling = {
            "predicate": "derived_from",
            "target": {"entry_id": "c"},
            "evidence": {"quote": "verbatim", "offset": [4, 11]},
        }
        report = build_modelome_relations([
            {"id": "a", "model_relations": [resolved, dangling]},
            {"id": "b"},
        ])

        self.assertEqual(report["edges"][0]["evidence"], resolved)
        self.assertEqual(report["unresolved"][0]["evidence"], dangling)

    def test_self_edges_and_cycles_are_retained(self):
        report = build_modelome_relations([
            {"id": "a", "model_relations": [
                {"predicate": "self", "target": {"entry_id": "a"}},
                {"predicate": "next", "target": {"entry_id": "b"}},
            ]},
            {"id": "b", "model_relations": [
                {"predicate": "next", "target": {"entry_id": "a"}},
            ]},
        ])

        self.assertEqual(
            [(edge["source"], edge["target"]) for edge in report["edges"]],
            [("a", "a"), ("a", "b"), ("b", "a")],
        )
        self.assertEqual(report["unresolved"], [])

    def test_non_object_target_raises_a_clear_error(self):
        for bad_target in (None, "b", ["b"], 17):
            with self.subTest(target=bad_target):
                with self.assertRaisesRegex(ValueError, "target must be an object"):
                    build_modelome_relations([
                        {"id": "a", "model_relations": [{
                            "predicate": "derived_from", "target": bad_target,
                        }]},
                    ])


if __name__ == "__main__":
    unittest.main()
