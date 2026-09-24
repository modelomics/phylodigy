from __future__ import annotations

import unittest

from phylodigy.computation_graph import build_graph_profile
from phylodigy.schema import ArchitecturalGenome, ArtifactProfile, merge_profiles


def graph(*operators):
    records = [
        {"index": 0, "name": "input", "op": "placeholder", "target": "input"}
    ]
    previous = "input"
    for index, operator in enumerate(operators, start=1):
        name = f"node_{index}"
        records.append(
            {
                "index": index,
                "inputs": [previous],
                "name": name,
                "op": "call_function",
                "target": operator,
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
    return build_graph_profile(records)


class ArchitecturalGenomeSchemaTests(unittest.TestCase):
    def test_artifact_profile_is_graph_only_compatibility_spelling(self):
        self.assertIs(ArtifactProfile, ArchitecturalGenome)
        genome = ArchitecturalGenome("model:a", graph("vendor.alpha"))
        raw = genome.to_dict()

        self.assertEqual(raw["artifact_type"], "phylodigy.architectural_genome")
        self.assertEqual(raw["schema_version"], "3.0.0")
        self.assertIn("graph_profile", raw)
        self.assertNotIn("traits", raw)
        self.assertNotIn("ontology", raw)

    def test_round_trip_binds_derived_graph_and_digest(self):
        genome = ArchitecturalGenome(
            "model:a",
            graph("vendor.alpha", "vendor.beta"),
            release_date="2026-01-02",
            extractor_name="test.frontend",
            extractor_version="1",
            metadata={"z": 2, "a": 1},
        )
        restored = ArchitecturalGenome.from_dict(genome.to_dict())

        self.assertEqual(restored, genome)
        self.assertEqual(restored.digest, genome.digest)
        self.assertEqual(restored.date_min, "2026-01-02")
        self.assertEqual(restored.date_max, "2026-01-02")

        tampered = genome.to_dict()
        tampered["artifact"]["name"] = "changed"
        with self.assertRaisesRegex(ValueError, "digest"):
            ArchitecturalGenome.from_dict(tampered)

    def test_content_addressed_metadata_is_immutable_and_name_is_text(self):
        genome = ArchitecturalGenome(
            "model:a",
            graph("vendor.alpha"),
            metadata={"nested": {"probes": ["a"]}},
        )
        digest = genome.digest
        with self.assertRaises(TypeError):
            genome.metadata["new"] = True
        with self.assertRaises(TypeError):
            genome.metadata["nested"]["probes"] = ("b",)
        self.assertEqual(genome.digest, digest)
        with self.assertRaisesRegex(TypeError, "name"):
            ArchitecturalGenome("model:b", graph(), name=7)

    def test_character_counts_are_generated_from_graph_profile(self):
        genome = ArchitecturalGenome("model:a", graph("vendor.alpha"))

        self.assertEqual(
            genome.character_counts,
            {
                character.character_id: character.count
                for character in genome.graph_profile.characters
            },
        )
        self.assertTrue(genome.character_counts)

    def test_merge_requires_identical_graphs(self):
        first = ArchitecturalGenome(
            "model:a",
            graph("vendor.alpha"),
            metadata={"source": "first"},
        )
        same = ArchitecturalGenome(
            "model:a",
            graph("vendor.alpha"),
            metadata={"probe": "second"},
        )
        changed = ArchitecturalGenome("model:a", graph("vendor.beta"))

        merged = merge_profiles(first, same)
        self.assertEqual(merged.structural_digest, first.structural_digest)
        self.assertEqual(merged.metadata, {"probe": "second", "source": "first"})
        with self.assertRaisesRegex(ValueError, "different graph profiles"):
            merge_profiles(first, changed)

    def test_merge_rejects_different_character_radii_deterministically(self):
        records = (
            {"name": "x", "op": "placeholder", "target": "x"},
            {
                "name": "y",
                "op": "call_function",
                "target": "vendor.op",
                "inputs": ["x"],
            },
            {"name": "out", "op": "output", "target": "out", "inputs": ["y"]},
        )
        radius_zero = ArchitecturalGenome(
            "model:a", build_graph_profile(records, radii=(0,))
        )
        multiscale = ArchitecturalGenome(
            "model:a", build_graph_profile(records, radii=(0, 1))
        )

        with self.assertRaisesRegex(ValueError, "different graph profiles"):
            merge_profiles(radius_zero, multiscale)
        with self.assertRaisesRegex(ValueError, "different graph profiles"):
            merge_profiles(multiscale, radius_zero)

    def test_metadata_conflicts_are_order_independent_and_keep_every_value(self):
        profiles = tuple(
            ArchitecturalGenome(
                "model:a", graph("vendor.alpha"), metadata={"probe": value}
            )
            for value in ("third", "first", "second")
        )

        forward = merge_profiles(*profiles)
        reverse = merge_profiles(*reversed(profiles))

        self.assertEqual(forward.metadata, reverse.metadata)
        self.assertNotIn("probe", forward.metadata)
        self.assertEqual(
            set(forward.metadata["merge_conflicts"]["probe"]),
            {"first", "second", "third"},
        )

    def test_invalid_dates_and_artifact_type_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "date_min"):
            ArchitecturalGenome(
                "model:a",
                graph(),
                date_min="2026-02-01",
                date_max="2026-01-01",
            )

        raw = ArchitecturalGenome("model:a", graph()).to_dict()
        raw["artifact_type"] = "phylodigy.old_trait_profile"
        with self.assertRaisesRegex(ValueError, "artifact_type"):
            ArchitecturalGenome.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
