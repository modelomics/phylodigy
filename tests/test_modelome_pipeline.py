from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from phylodigy.computation_graph import build_graph_profile
from phylodigy.io import write_profile
from phylodigy.modelome import build_modelome_tree, plan_modelome_tree
from phylodigy.schema import ArchitecturalGenome


def _entry(identifier: str, name: str, **extra):
    return {
        "id": identifier,
        "canonical_name": name,
        "aliases": [],
        "identifiers": [],
        "tags": [],
        "members": [],
        "resources": [],
        "releases": [],
        "model_relations": [],
        **extra,
    }


def _genome(identifier: str, operation: str = "vendor.alpha"):
    profile = build_graph_profile(
        [
            {"name": "input", "op": "placeholder", "target": "input", "inputs": []},
            {"name": "feature", "op": "call_function", "target": operation, "inputs": ["input"]},
            {"name": "output", "op": "output", "target": "output", "inputs": ["feature"]},
        ],
        radii=(0,),
    )
    return ArchitecturalGenome(identifier, profile)


def _write_genomes(directory: Path, genomes):
    for genome in genomes:
        write_profile(genome, directory / f"{genome.artifact_id.replace(':', '-')}.json")


class ModelomePipelineTests(unittest.TestCase):
    def test_plan_counts_all_entries_and_rejects_taxa_limit_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entries.json"
            entries = [_entry(f"catalog:{index}", f"Model {index}") for index in range(3)]
            path.write_text(json.dumps(entries), encoding="utf-8")

            plan = plan_modelome_tree(path)

            self.assertEqual(plan["modelome"]["counts"]["entries"], 3)

            profiles_dir = Path(directory) / "profiles"
            profiles_dir.mkdir()
            genomes = [_genome(f"catalog:{index}", f"vendor.op{index}") for index in range(3)]
            _write_genomes(profiles_dir, genomes)
            with self.assertRaisesRegex(ValueError, "max_taxa"):
                build_modelome_tree(path, profiles_dir=profiles_dir, max_taxa=2)

    def test_build_keeps_every_model_and_reports_profile_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries_path = root / "entries.json"
            profiles_dir = root / "profiles"
            profiles_dir.mkdir()
            genomes = [_genome("model:first"), _genome("model:second", "vendor.beta")]
            entries = [
                _entry(genomes[0].artifact_id, "Renamed first", resources=[{"url": "https://example.test"}]),
                _entry("catalog:unprofiled", "No architecture profile"),
                _entry(genomes[1].artifact_id, "Renamed second"),
            ]
            entries_path.write_text(json.dumps(entries), encoding="utf-8")
            _write_genomes(profiles_dir, genomes)

            result = build_modelome_tree(entries_path, profiles_dir=profiles_dir)

        entries_by_id = sorted(entries, key=lambda item: item["id"])
        self.assertEqual(result["modelome"]["entries"], entries_by_id)
        self.assertEqual(result["modelome"]["counts"], {"entries": 3, "profiled": 2, "unprofiled": 1})
        self.assertEqual([row["entry_id"] for row in result["modelome"]["coverage"]], [row["id"] for row in entries_by_id])
        expected_status = ["bound" if entry["id"] in {g.artifact_id for g in genomes} else "missing_profile" for entry in entries_by_id]
        self.assertEqual([row["status"] for row in result["modelome"]["coverage"]], expected_status)
        self.assertEqual(result["status"], "inferred")
        self.assertIsNotNone(result["tree"])
        self.assertEqual(set(result["tree"]["taxa"]), {genomes[0].artifact_id, genomes[1].artifact_id})

    def test_graph_tree_is_independent_of_catalog_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles_dir = root / "profiles"
            profiles_dir.mkdir()
            genomes = [_genome(f"model:{index}", f"vendor.op{index}") for index in range(3)]
            _write_genomes(profiles_dir, genomes)
            ids = [genome.artifact_id for genome in genomes]
            plain = [_entry(identifier, f"Model {index}") for index, identifier in enumerate(ids)]
            decorated = [
                _entry(
                    identifier,
                    f"Unrelated name {index}",
                    releases=[{"date": f"203{index}-01-01"}],
                    resources=[{"url": f"https://example.test/{index}"}],
                    tags=[f"tag-{index}"],
                )
                for index, identifier in enumerate(ids)
            ]
            first_path = root / "plain.json"
            second_path = root / "decorated.json"
            first_path.write_text(json.dumps(plain), encoding="utf-8")
            second_path.write_text(json.dumps(decorated), encoding="utf-8")

            first = build_modelome_tree(first_path, profiles_dir=profiles_dir)
            second = build_modelome_tree(second_path, profiles_dir=profiles_dir)

        self.assertEqual(first["tree"], second["tree"])
        self.assertEqual(first["comparisons"], second["comparisons"])

    def test_one_profile_reports_insufficient_profiles_without_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries_path = root / "entries.json"
            profiles_dir = root / "profiles"
            profiles_dir.mkdir()
            genome = _genome("model:one")
            entries = [_entry(genome.artifact_id, "One model"), _entry("catalog:missing", "Missing")]
            entries_path.write_text(json.dumps(entries), encoding="utf-8")
            _write_genomes(profiles_dir, [genome])

            result = build_modelome_tree(entries_path, profiles_dir=profiles_dir)

        self.assertEqual(result["status"], "insufficient_profiles")
        self.assertIsNone(result["tree"])
        self.assertEqual(result["modelome"]["counts"]["profiled"], 1)
        self.assertEqual(result["modelome"]["counts"]["unprofiled"], 1)

    def test_compact_mode_uses_graph_limit_and_retains_all_bound_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries_path = root / "entries.json"
            profiles_dir = root / "profiles"
            profiles_dir.mkdir()
            genomes = [
                _genome("model:a"),
                _genome("model:b"),
                _genome("model:c", "vendor.beta"),
            ]
            entries = [_entry(genome.artifact_id, genome.artifact_id) for genome in genomes]
            entries_path.write_text(json.dumps(entries), encoding="utf-8")
            _write_genomes(profiles_dir, genomes)

            with self.assertRaisesRegex(ValueError, "max_taxa"):
                build_modelome_tree(entries_path, profiles_dir=profiles_dir, max_taxa=2)
            compact = build_modelome_tree(
                entries_path,
                profiles_dir=profiles_dir,
                max_taxa=2,
                collapse_identical=True,
            )

        self.assertEqual(compact["status"], "inferred")
        self.assertEqual(compact["tree"]["method"], "neighbor_joining_unique_graphs")
        self.assertEqual(compact["modelome"]["counts"], {"entries": 3, "profiled": 3, "unprofiled": 0})
        self.assertEqual(len(compact["tree"]["taxa"]), 3)
        self.assertEqual(compact["modelome"]["collapse_identical"], True)


if __name__ == "__main__":
    unittest.main()
