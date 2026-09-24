from __future__ import annotations

import unittest

from phylodigy.computation_graph import build_graph_profile
from phylodigy.lineage import LineageAnalysisConfig, infer_lineage_network
from phylodigy.schema import ArchitecturalGenome


def genome(artifact_id, *operations, date=None):
    records = [
        {"index": 0, "name": "input", "op": "placeholder", "target": "input"}
    ]
    previous = "input"
    for index, operation in enumerate(operations, 1):
        name = f"n{index}"
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


class LineageConfigTests(unittest.TestCase):
    def test_configuration_is_canonical_and_validated(self):
        config = LineageAnalysisConfig(
            radii=(3, 1, 1, 0),
            radius_weights={1: 2},
            region_weight=3,
            four_point_tolerance=0.01,
        )

        self.assertEqual(config.radii, (0, 1, 3))
        self.assertEqual(
            config.to_dict(),
            {
                "four_point_tolerance": 0.01,
                "radii": [0, 1, 3],
                "radius_weights": {"1": 2.0},
                "region_weight": 3.0,
            },
        )
        self.assertEqual(
            LineageAnalysisConfig.from_mapping(config.to_dict()),
            config,
        )

        for weights in ({"01": 2}, {"1.5": 2}, {True: 2}, {1: 2, "1": 3}):
            with self.subTest(weights=weights):
                with self.assertRaises((TypeError, ValueError)):
                    LineageAnalysisConfig.from_mapping(
                        {"radii": [0, 1], "radius_weights": weights}
                    )

        for kwargs in (
            {"radii": ()},
            {"radii": (-1,)},
            {"radius_weights": {9: 1}},
            {"radius_weights": {0: 0}},
            {"region_weight": 0},
            {"four_point_tolerance": -1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LineageAnalysisConfig(**kwargs)


class GraphOnlyLineageTests(unittest.TestCase):
    def setUp(self):
        self.genomes = (
            genome("model:a", "vendor.alpha", date="2020-01-01"),
            genome("model:b", "vendor.alpha", "vendor.beta", date="2021-01-01"),
            genome("model:c", "vendor.alpha", "vendor.gamma", date="2022-01-01"),
        )

    def test_output_contains_graph_character_matrix_and_neighbor_joining_tree(self):
        result = infer_lineage_network(self.genomes)

        self.assertEqual(result["artifact_type"], "phylodigy.architecture_lineage")
        self.assertEqual(result["analysis_version"], "2")
        self.assertEqual(result["tree"]["method"], "neighbor_joining")
        self.assertEqual(result["tree"]["normalization"], "none")
        self.assertEqual(
            result["character_matrix"]["source"],
            "operator_layer_graph_only",
        )
        self.assertFalse(
            result["structural_evidence_boundary"]["paper_can_create_characters"]
        )
        self.assertEqual(len(result["digest"]), 64)
        self.assertNotIn("traits", result)
        self.assertNotIn("ontology", result)

    def test_character_matrix_is_derived_exactly_from_genomes(self):
        result = infer_lineage_network(self.genomes)
        character_ids = result["character_matrix"]["character_ids"]
        counts = result["character_matrix"]["counts_by_artifact"]

        for item in self.genomes:
            self.assertEqual(
                counts[item.artifact_id],
                [item.character_counts.get(character_id, 0) for character_id in character_ids],
            )

    def test_analysis_radii_not_stored_profile_radii_define_matrix(self):
        records = (
            {"name": "x", "op": "placeholder", "target": "x"},
            {
                "name": "y",
                "op": "call_function",
                "target": "vendor.alpha",
                "inputs": ["x"],
            },
            {"name": "out", "op": "output", "target": "out", "inputs": ["y"]},
        )
        left = ArchitecturalGenome(
            "left",
            build_graph_profile(records, radii=(0,)),
            release_date="2020-01-01",
        )
        right = ArchitecturalGenome(
            "right",
            build_graph_profile(records, radii=(0, 1, 2, 3)),
            release_date="2021-01-01",
        )

        result = infer_lineage_network(
            (left, right), config=LineageAnalysisConfig(radii=(0,))
        )
        counts = result["character_matrix"]["counts_by_artifact"]
        self.assertEqual(counts["left"], counts["right"])
        self.assertEqual(result["comparisons"][0]["graph_distance"]["distance"], 0.0)

    def test_external_evidence_is_retained_but_cannot_change_structure(self):
        without = infer_lineage_network(self.genomes)
        with_evidence = infer_lineage_network(
            self.genomes,
            evidence=(
                {"kind": "paper_label", "label": "a human name"},
                {"kind": "repository_provenance", "source": "model:a"},
            ),
        )

        self.assertEqual(without["comparisons"], with_evidence["comparisons"])
        self.assertEqual(without["tree"], with_evidence["tree"])
        self.assertEqual(without["character_matrix"], with_evidence["character_matrix"])
        self.assertNotEqual(without["digest"], with_evidence["digest"])

    def test_input_and_evidence_order_do_not_change_output(self):
        evidence = (
            {"kind": "z", "value": 1},
            {"kind": "a", "value": 2},
        )
        first = infer_lineage_network(self.genomes, evidence=evidence)
        second = infer_lineage_network(reversed(self.genomes), evidence=reversed(evidence))

        self.assertEqual(first, second)

    def test_removed_trait_and_partition_options_fail_explicitly(self):
        for option in (
            {"trait_priors": {"named": "rare"}},
            {"ontology": object()},
            {"partition_weights": {"paper": 1}},
            {"bundle_groups": {"named": "bundle"}},
        ):
            with self.subTest(option=next(iter(option))):
                with self.assertRaisesRegex(TypeError, "unsupported non-graph"):
                    infer_lineage_network(self.genomes, **option)

    def test_at_least_two_unique_genomes_are_required(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            infer_lineage_network(self.genomes[:1])
        with self.assertRaisesRegex(ValueError, "unique"):
            infer_lineage_network((self.genomes[0], self.genomes[0]))


if __name__ == "__main__":
    unittest.main()
