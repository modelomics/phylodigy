from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phylodigy.computation_graph import build_graph_profile
from phylodigy.io import ProfileReadError, write_profile
from phylodigy.modelome_profiles import load_modelome_profiles
from phylodigy.schema import ArchitecturalGenome


def _genome(artifact_id: str, *, kind: str = "model") -> ArchitecturalGenome:
    graph_profile = build_graph_profile(
        [
            {"name": "x", "op": "placeholder", "target": "x", "inputs": []},
            {"name": "out", "op": "output", "target": "out", "inputs": ["x"]},
        ],
        radii=(0,),
    )
    return ArchitecturalGenome(artifact_id, graph_profile, artifact_kind=kind)


class ModelomeProfileLoaderTests(unittest.TestCase):
    def test_loads_one_profile_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "one.json"
            write_profile(_genome("model:one"), path)

            result = load_modelome_profiles(path)

        self.assertEqual(set(result), {"model:one"})
        self.assertIsInstance(result["model:one"], ArchitecturalGenome)

    def test_recursively_loads_json_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_profile(_genome("model:one"), root / "one.json")
            nested = root / "nested"
            nested.mkdir()
            write_profile(_genome("model:two"), nested / "two.json")
            (nested / "ignored.txt").write_text("not JSON", encoding="utf-8")

            result = load_modelome_profiles(root)

        self.assertEqual(set(result), {"model:one", "model:two"})

    def test_empty_directory_returns_empty_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(load_modelome_profiles(temporary), {})

    def test_ignores_explicit_popularity_failure_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_profile(_genome("model:one"), root / "model.genome.json")
            (root / "broken.failure.json").write_text("not a profile", encoding="utf-8")

            result = load_modelome_profiles(root)

        self.assertEqual(set(result), {"model:one"})

    def test_rejects_duplicate_artifact_ids_with_both_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a.json"
            second = root / "b.json"
            write_profile(_genome("model:same"), first)
            write_profile(_genome("model:same"), second)

            with self.assertRaisesRegex(ValueError, "duplicate artifact ID.*a.json.*b.json"):
                load_modelome_profiles(root)

    def test_rejects_non_genome_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "paper.json"
            path.write_text('{"artifact_type":"phylodigy.paper_profile"}', encoding="utf-8")

            with self.assertRaises(ProfileReadError):
                load_modelome_profiles(path)

    def test_rejects_non_model_genome(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.json"
            write_profile(_genome("dataset:one", kind="dataset"), path)

            with self.assertRaisesRegex(ValueError, "requires artifact kind 'model'"):
                load_modelome_profiles(path)


if __name__ == "__main__":
    unittest.main()
