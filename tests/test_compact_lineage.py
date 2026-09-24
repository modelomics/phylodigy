from __future__ import annotations

import unittest
from dataclasses import replace
from itertools import combinations

from phylodigy.compact_lineage import infer_compact_lineage, lineage_distance_lookup
from phylodigy.computation_graph import build_graph_profile
from phylodigy.newick import lineage_to_newick
from phylodigy.profile_comparison import compare_profiles
from phylodigy.schema import ArchitecturalGenome


def _genome(identifier: str, operation: str = "vendor.alpha", **metadata):
    graph = build_graph_profile(
        [
            {"name": "input", "op": "placeholder", "target": "input", "inputs": []},
            {
                "name": "feature",
                "op": "call_function",
                "target": operation,
                "inputs": ["input"],
            },
            {
                "name": "output",
                "op": "output",
                "target": "output",
                "inputs": ["feature"],
            },
        ],
        radii=(0,),
    )
    return ArchitecturalGenome(identifier, graph, metadata=metadata)


class CompactLineageTests(unittest.TestCase):
    def test_collapses_identical_graphs_but_keeps_every_artifact_leaf(self):
        profiles = [
            _genome("model:a"),
            _genome("model:b"),
            _genome("model:c", "vendor.beta"),
        ]

        result = infer_compact_lineage(profiles)

        self.assertEqual(
            result["artifact_type"], "phylodigy.compact_architecture_lineage"
        )
        self.assertEqual(result["comparison_scope"], "structural_representatives")
        self.assertEqual(result["tree"]["method"], "neighbor_joining_unique_graphs")
        self.assertEqual(
            set(result["tree"]["taxa"]), {item.artifact_id for item in profiles}
        )
        self.assertEqual(len(result["structural_groups"]), 2)
        repeated_group = next(
            group
            for group in result["structural_groups"]
            if len(group["artifact_ids"]) == 2
        )
        self.assertEqual(repeated_group["artifact_ids"], ["model:a", "model:b"])
        self.assertEqual(len(result["comparisons"]), 1)
        self.assertEqual(
            set(result["character_matrix"]["counts_by_artifact"]),
            {group["representative_id"] for group in result["structural_groups"]},
        )
        self.assertEqual(
            set(result["character_matrix"]["artifact_representatives"]),
            {item.artifact_id for item in profiles},
        )
        self.assertEqual(
            result["character_matrix"]["artifact_representatives"]["model:a"],
            result["character_matrix"]["artifact_representatives"]["model:b"],
        )
        leaf_edges = [
            edge
            for edge in result["tree"]["edges"]
            if edge["child"] in {"model:a", "model:b", "model:c"}
        ]
        self.assertEqual(len(leaf_edges), 3)
        self.assertTrue(
            all(
                edge["branch_length"] == 0
                for edge in leaf_edges
                if edge["child"] in {"model:a", "model:b"}
            )
        )

    def test_grouping_ignores_metadata_and_output_is_order_invariant(self):
        shared = _genome("model:first")
        alias = replace(
            shared,
            artifact_id="model:second",
            name="A different display name",
            release_date="2024-01-01",
            metadata={"source": "catalog"},
        )
        distinct = _genome("model:third", "vendor.beta")

        left = infer_compact_lineage([shared, alias, distinct])
        right = infer_compact_lineage([distinct, alias, shared])

        self.assertEqual(left["structural_groups"], right["structural_groups"])
        self.assertEqual(left["comparisons"], right["comparisons"])
        self.assertEqual(left["tree"], right["tree"])

    def test_all_identical_graphs_make_zero_star_without_comparisons(self):
        profiles = [_genome(f"model:{index}") for index in range(4)]

        result = infer_compact_lineage(profiles)

        self.assertEqual(len(result["structural_groups"]), 1)
        self.assertEqual(result["comparisons"], [])
        self.assertEqual(
            result["tree"]["taxa"], [item.artifact_id for item in profiles]
        )
        self.assertEqual(len(result["tree"]["edges"]), len(profiles))
        self.assertTrue(
            all(edge["branch_length"] == 0 for edge in result["tree"]["edges"])
        )

    def test_graph_changes_form_distinct_groups_and_graph_limit_preflights(self):
        profiles = [_genome("model:one"), _genome("model:two", "vendor.beta")]

        result = infer_compact_lineage(profiles, max_graphs=2)

        self.assertEqual(len(result["structural_groups"]), 2)
        self.assertEqual(len(result["comparisons"]), 1)
        with self.assertRaisesRegex(ValueError, "max_graphs"):
            infer_compact_lineage(profiles, max_graphs=1)

    def test_requires_two_artifacts_and_positive_graph_limit(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            infer_compact_lineage([_genome("model:only")])
        with self.assertRaisesRegex(ValueError, "max_graphs"):
            infer_compact_lineage(
                [_genome("model:a"), _genome("model:b")], max_graphs=0
            )


    def test_alias_distance_lookup_matches_every_original_pair(self):
        profiles = [
            _genome("model:a"), _genome("model:b"), _genome("model:c"),
            _genome("model:d", "vendor.beta"), _genome("model:e", "vendor.gamma"),
        ]
        lookup = lineage_distance_lookup(infer_compact_lineage(profiles))

        for left, right in combinations(profiles, 2):
            expected = compare_profiles(left, right).graph_distance.distance
            self.assertEqual(lookup(left.artifact_id, right.artifact_id), expected)
        self.assertEqual(lookup("model:a", "model:b"), 0.0)

    def test_additive_two_group_tree_preserves_patristic_distances_and_newick_tips(self):
        profiles = [
            _genome("model_a"), _genome("model_b"),
            _genome("model_c", "vendor.beta"), _genome("model_d", "vendor.beta"),
        ]
        result = infer_compact_lineage(profiles)

        for left, right in combinations(profiles, 2):
            expected = compare_profiles(left, right).graph_distance.distance
            self.assertAlmostEqual(
                _patristic_distance(result["tree"], left.artifact_id, right.artifact_id),
                expected,
            )
        newick = lineage_to_newick(result)
        for profile in profiles:
            self.assertIn(profile.artifact_id, newick)
        diagnostics = result["tree"]["tree_likeness"]
        self.assertEqual(diagnostics["four_point_assessment"], "not_assessed")
        self.assertIsNone(diagnostics["additive_within_tolerance"])

    def test_reserved_internal_id_collision_is_remapped_without_losing_tip(self):
        reserved = "__phylodigy_internal__:000000"
        profiles = [_genome("!a"), _genome(reserved), _genome("b", "vendor.beta")]

        result = infer_compact_lineage(profiles)

        self.assertEqual(set(result["tree"]["taxa"]), {item.artifact_id for item in profiles})
        self.assertTrue(any(edge["child"] == reserved for edge in result["tree"]["edges"]))
        internal_parents = {edge["parent"] for edge in result["tree"]["edges"]} - set(result["tree"]["taxa"])
        self.assertNotIn(reserved, internal_parents)
        self.assertIn("'" + reserved + "'", lineage_to_newick(result))


def _patristic_distance(tree, left, right):
    adjacency = {}
    for edge in tree["edges"]:
        adjacency.setdefault(edge["parent"], []).append((edge["child"], edge["branch_length"]))
        adjacency.setdefault(edge["child"], []).append((edge["parent"], edge["branch_length"]))
    pending = [(left, None, 0.0)]
    while pending:
        node, previous, distance = pending.pop()
        if node == right:
            return distance
        pending.extend(
            (neighbor, node, distance + length)
            for neighbor, length in adjacency[node]
            if neighbor != previous
        )
    raise AssertionError(f"no path between {left} and {right}")


if __name__ == "__main__":
    unittest.main()
