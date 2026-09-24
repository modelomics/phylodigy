from __future__ import annotations

import unittest

from phylodigy.newick import lineage_to_newick


class NewickTests(unittest.TestCase):
    def test_serializes_inferred_tree_deterministically_and_quotes_labels(self):
        lineage = {
            "tree": {
                "edges": [
                    {"parent": "root", "child": "z", "branch_length": 1.0},
                    {"parent": "root", "child": "a,b", "branch_length": 0.25},
                    {"parent": "root", "child": "O'Brien", "branch_length": 0},
                ]
            }
        }
        self.assertEqual(
            lineage_to_newick(lineage), "('O''Brien':0,'a,b':0.25,z:1);"
        )
        self.assertEqual(lineage_to_newick(lineage), lineage_to_newick(lineage))

    def test_rejects_invalid_edges_and_branch_lengths(self):
        invalid = [
            {"tree": {"edges": []}},
            {"tree": {"edges": [{"parent": "x", "child": "y"}]}},
            {"tree": {"edges": [{"parent": "x", "child": "y", "branch_length": float("nan")}]}},
            {"tree": {"edges": [
                {"parent": "r", "child": "x", "branch_length": 1},
                {"parent": "s", "child": "x", "branch_length": 1},
            ]}},
        ]
        for lineage in invalid:
            with self.subTest(lineage=lineage), self.assertRaises(ValueError):
                lineage_to_newick(lineage)

    def test_rejects_cycles(self):
        lineage = {"tree": {"edges": [
            {"parent": "root", "child": "leaf", "branch_length": 1},
            {"parent": "leaf", "child": "root", "branch_length": 1},
        ]}}
        with self.assertRaises(ValueError):
            lineage_to_newick(lineage)


if __name__ == "__main__":
    unittest.main()
