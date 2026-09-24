from __future__ import annotations

import itertools
import unittest

from phylodigy.computation_graph import GraphDistance, build_graph_profile
from phylodigy.profile_comparison import (
    ProfileComparison,
    build_vertical_backbone,
    compare_profiles,
    diagnose_tree_likeness,
    distance_matrix,
    pairwise_profile_comparisons,
)
from phylodigy.schema import ArchitecturalGenome


def genome(artifact_id, operations, *, date=None):
    records = [
        {"index": 0, "name": "input", "op": "placeholder", "target": "input"}
    ]
    previous = "input"
    for index, operation in enumerate(operations, start=1):
        name = f"operation_{index}"
        records.append(
            {
                "index": index,
                "inputs": [previous],
                "name": name,
                "op": "call_function",
                "target": operation,
            }
        )
        previous = name
    records.append(
        {
            "index": len(records),
            "inputs": [previous],
            "name": "output",
            "op": "output",
            "target": "output",
        }
    )
    return ArchitecturalGenome(
        artifact_id,
        build_graph_profile(records),
        release_date=date,
    )


def synthetic_comparison(
    left,
    right,
    value,
    *,
    left_digest=None,
    right_digest=None,
    distance_signature="a" * 64,
):
    left_digest = left_digest or f"graph:{left}"
    right_digest = right_digest or f"graph:{right}"
    value = float(value)
    return ProfileComparison(
        left,
        right,
        left_digest,
        right_digest,
        GraphDistance(
            value,
            {0: value},
            radius_weights={0: 1.0},
            left_digest=left_digest,
            right_digest=right_digest,
            distance_signature=distance_signature,
        ),
        {},
        {},
        {},
    )


def synthetic_comparisons(ids, distances):
    return tuple(
        synthetic_comparison(
            left,
            right,
            distances[tuple(sorted((left, right)))],
        )
        for left, right in itertools.combinations(ids, 2)
    )


class GraphOnlyProfileComparisonTests(unittest.TestCase):
    def test_comparison_uses_only_graph_generated_characters(self):
        left = genome("named:Transformer", ["vendor.alpha", "vendor.beta"])
        right = genome("named:ResNet", ["vendor.alpha", "vendor.gamma"])
        result = compare_profiles(left, right)

        self.assertGreater(result.distance, 0.0)
        self.assertTrue(result.shared_characters)
        self.assertTrue(result.left_only_characters)
        self.assertTrue(result.right_only_characters)
        self.assertEqual(result.left_digest, left.structural_digest)
        self.assertEqual(result.right_digest, right.structural_digest)
        self.assertEqual(result.left_digest, result.graph_distance.left_digest)
        self.assertEqual(result.right_digest, result.graph_distance.right_digest)
        self.assertNotEqual(result.left_digest, left.digest)
        raw = result.to_dict()
        self.assertNotIn("traits", raw)
        self.assertNotIn("ontology", raw)
        self.assertNotIn("paper", raw)
        for character_id in (
            *result.shared_characters,
            *result.left_only_characters,
            *result.right_only_characters,
        ):
            self.assertEqual(len(character_id), 64)

    def test_artifact_names_do_not_change_structural_distance(self):
        common = ["vendor.alpha", "vendor.beta"]
        left = genome("family:a", common)
        renamed = genome("totally-different:b", common)

        comparison = compare_profiles(left, renamed)
        self.assertEqual(comparison.distance, 0.0)
        self.assertFalse(comparison.left_only_characters)
        self.assertFalse(comparison.right_only_characters)

    def test_removed_non_graph_options_fail_explicitly(self):
        left = genome("a", ["vendor.alpha"])
        right = genome("b", ["vendor.beta"])
        for option in (
            {"ontology": object()},
            {"partition_weights": {"paper": 1}},
            {"trait_priors": {"named": "rare"}},
        ):
            with self.subTest(option=next(iter(option))):
                with self.assertRaisesRegex(TypeError, "unsupported non-graph"):
                    compare_profiles(left, right, **option)

    def test_pairwise_order_and_distance_matrix_are_deterministic(self):
        profiles = (
            genome("c", ["vendor.alpha", "vendor.gamma"]),
            genome("a", ["vendor.alpha"]),
            genome("b", ["vendor.alpha", "vendor.beta"]),
        )
        first = pairwise_profile_comparisons(profiles)
        second = pairwise_profile_comparisons(reversed(profiles))

        self.assertEqual(first, second)
        self.assertEqual(
            [(item.left_id, item.right_id) for item in first],
            [("a", "b"), ("a", "c"), ("b", "c")],
        )
        matrix = distance_matrix(profiles)
        self.assertEqual(matrix["artifact_ids"], ["a", "b", "c"])
        self.assertEqual(matrix["normalization"], "none")
        self.assertEqual(matrix["matrix"][0][0], 0.0)
        self.assertEqual(matrix["matrix"][0][1], matrix["matrix"][1][0])

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            pairwise_profile_comparisons(
                [genome("same", ["vendor.alpha"]), genome("same", ["vendor.beta"])]
            )

    def test_comparison_rejects_endpoint_digest_mismatch(self):
        with self.assertRaisesRegex(ValueError, "match graph-distance endpoints"):
            ProfileComparison(
                "a",
                "b",
                "wrong:a",
                "graph:b",
                GraphDistance(
                    1,
                    {0: 1},
                    radius_weights={0: 1},
                    left_digest="graph:a",
                    right_digest="graph:b",
                    distance_signature="a" * 64,
                ),
                {},
                {},
                {},
            )


class TreeLikenessTests(unittest.TestCase):
    def test_additive_quartet_satisfies_four_point_condition(self):
        # Tree split ab|cd with pendant lengths 1 and an internal length 2.
        comparisons = synthetic_comparisons(
            ("a", "b", "c", "d"),
            {
                ("a", "b"): 2,
                ("c", "d"): 2,
                ("a", "c"): 4,
                ("a", "d"): 4,
                ("b", "c"): 4,
                ("b", "d"): 4,
            },
        )
        diagnostic = diagnose_tree_likeness(comparisons)

        self.assertTrue(diagnostic["additive_within_tolerance"])
        self.assertEqual(diagnostic["violation_count"], 0)
        tree = build_vertical_backbone(comparisons=comparisons)
        self.assertEqual(tree["method"], "neighbor_joining")
        self.assertEqual(tree["normalization"], "none")
        self.assertEqual(tree["taxa"], ["a", "b", "c", "d"])
        self.assertTrue(tree["tree_likeness"]["metric_within_tolerance"])
        self.assertEqual(
            tree["tree_likeness"]["neighbor_joining"]["negative_limb_count"],
            0,
        )

    def test_nonadditive_quartet_is_reported_not_hidden(self):
        comparisons = synthetic_comparisons(
            ("a", "b", "c", "d"),
            {
                ("a", "b"): 1,
                ("c", "d"): 1,
                ("a", "c"): 2,
                ("b", "d"): 2,
                ("a", "d"): 5,
                ("b", "c"): 1,
            },
        )
        diagnostic = diagnose_tree_likeness(comparisons)

        self.assertFalse(diagnostic["additive_within_tolerance"])
        self.assertEqual(diagnostic["violation_count"], 1)
        self.assertGreater(diagnostic["maximum_four_point_violation"], 0.0)

    def test_incomplete_matrix_is_rejected(self):
        comparisons = synthetic_comparisons(
            ("a", "b", "c"),
            {("a", "b"): 1, ("a", "c"): 2, ("b", "c"): 1},
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            diagnose_tree_likeness(comparisons[:-1])

    def test_mixed_distance_algorithms_are_rejected(self):
        comparisons = list(
            synthetic_comparisons(
                ("a", "b", "c"),
                {("a", "b"): 1, ("a", "c"): 2, ("b", "c"): 1},
            )
        )
        original = comparisons[-1]
        comparisons[-1] = synthetic_comparison(
            original.left_id,
            original.right_id,
            original.distance,
            left_digest=original.left_digest,
            right_digest=original.right_digest,
            distance_signature="b" * 64,
        )

        with self.assertRaisesRegex(ValueError, "algorithm signature"):
            build_vertical_backbone(comparisons=comparisons)

    def test_one_artifact_id_cannot_bind_multiple_structural_digests(self):
        comparisons = list(
            synthetic_comparisons(
                ("a", "b", "c"),
                {("a", "b"): 1, ("a", "c"): 2, ("b", "c"): 1},
            )
        )
        comparisons[1] = synthetic_comparison(
            "a", "c", 2, left_digest="graph:a:other"
        )

        with self.assertRaisesRegex(ValueError, "conflicting structural digests: a"):
            diagnose_tree_likeness(comparisons)

    def test_duplicate_unordered_pairs_are_always_rejected(self):
        comparisons = list(
            synthetic_comparisons(
                ("a", "b", "c"),
                {("a", "b"): 1, ("a", "c"): 2, ("b", "c"): 1},
            )
        )
        comparisons.append(synthetic_comparison("b", "a", 1))

        messages = []
        for ordered in (comparisons, list(reversed(comparisons))):
            with self.assertRaisesRegex(ValueError, "duplicate comparison pairs") as error:
                diagnose_tree_likeness(ordered)
            messages.append(str(error.exception))
        self.assertEqual(messages[0], messages[1])

    def test_internal_node_ids_never_collide_with_taxa(self):
        collision = "__phylodigy_internal__:000000"
        comparisons = synthetic_comparisons(
            (collision, "b", "c"),
            {
                tuple(sorted((collision, "b"))): 2,
                tuple(sorted((collision, "c"))): 2,
                ("b", "c"): 2,
            },
        )
        tree = build_vertical_backbone(comparisons=comparisons)
        internal_ids = {edge["parent"] for edge in tree["edges"]}

        self.assertEqual(set(tree["taxa"]), {collision, "b", "c"})
        self.assertTrue(internal_ids.isdisjoint(tree["taxa"]))

    def test_tolerance_is_validated_even_without_quartets(self):
        comparisons = synthetic_comparisons(("a", "b"), {("a", "b"): 1})
        for tolerance in (-1, float("inf"), float("nan"), True):
            for function in (diagnose_tree_likeness, build_vertical_backbone):
                with self.subTest(
                    function=function.__name__, tolerance=tolerance
                ):
                    with self.assertRaisesRegex(ValueError, "tolerance"):
                        if function is build_vertical_backbone:
                            function(
                                comparisons=comparisons, tolerance=tolerance
                            )
                        else:
                            function(comparisons, tolerance=tolerance)

    def test_triangle_violation_and_negative_nj_limb_are_explicit(self):
        comparisons = synthetic_comparisons(
            ("a", "b", "c"),
            {("a", "b"): 1, ("a", "c"): 1, ("b", "c"): 3},
        )
        diagnostic = diagnose_tree_likeness(comparisons)

        self.assertFalse(diagnostic["metric_within_tolerance"])
        self.assertFalse(diagnostic["additive_within_tolerance"])
        self.assertEqual(diagnostic["triangle_count"], 1)
        self.assertEqual(diagnostic["triangle_check_count"], 3)
        self.assertEqual(diagnostic["triangle_violation_count"], 1)
        self.assertEqual(diagnostic["maximum_triangle_violation"], 1.0)

        tree = build_vertical_backbone(comparisons=comparisons)
        nj = tree["tree_likeness"]["neighbor_joining"]
        self.assertEqual(nj["negative_limb_count"], 1)
        self.assertEqual(nj["negative_limb_count_beyond_tolerance"], 1)
        self.assertFalse(tree["tree_likeness"]["tree_metric_within_tolerance"])
        self.assertEqual(
            sum(edge["length_clamped"] for edge in tree["edges"]),
            nj["negative_limb_count"],
        )

    def test_nonfinite_and_negative_distances_are_diagnosed_and_not_built(self):
        for value, finite, nonnegative in (
            (-1.0, True, False),
            (float("inf"), False, True),
            (float("nan"), False, False),
        ):
            with self.subTest(value=value):
                comparisons = list(
                    synthetic_comparisons(("a", "b"), {("a", "b"): 1})
                )
                object.__setattr__(comparisons[0].graph_distance, "distance", value)
                diagnostic = diagnose_tree_likeness(comparisons)
                self.assertEqual(diagnostic["distances_finite"], finite)
                self.assertEqual(
                    diagnostic["distances_nonnegative"], nonnegative
                )
                self.assertFalse(diagnostic["distance_domain_valid"])
                self.assertEqual(len(diagnostic["invalid_distances"]), 1)
                with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                    build_vertical_backbone(comparisons=comparisons)


if __name__ == "__main__":
    unittest.main()
