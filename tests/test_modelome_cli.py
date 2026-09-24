from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRY = {
    "id": "entry:example",
    "canonical_name": "Example model",
    "aliases": [],
    "identifiers": [],
    "tags": ["text-generation"],
    "members": [],
    "resources": [],
    "releases": [],
    "model_relations": [],
}


class ModelomeCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "phylodigy.cli", *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def write_entries(self, directory: Path) -> Path:
        path = directory / "entries.json"
        path.write_text(json.dumps([ENTRY]), encoding="utf-8")
        return path

    def test_modelome_plan_writes_json_and_preserves_entries_and_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            entries = self.write_entries(Path(temporary))
            result = self.run_cli("modelome-plan", str(entries), "-o", "-")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        output = json.loads(result.stdout)
        self.assertEqual(output["modelome"]["entries"], [ENTRY])
        self.assertIsInstance(output["modelome"]["coverage"], list)
        self.assertIsInstance(output["modelome"]["counts"], dict)

    def test_modelome_tree_without_profiles_reports_insufficient_profiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = self.write_entries(root)
            result = self.run_cli(
                "modelome-tree",
                str(entries),
                "--max-taxa",
                "5",
                "-o",
                "-",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        output = json.loads(result.stdout)
        modelome = output["modelome"]
        self.assertEqual(modelome["entries"], [ENTRY])
        self.assertIsInstance(modelome["coverage"], list)
        self.assertIsNone(output["tree"])
        self.assertEqual(output["status"], "insufficient_profiles")

    def test_modelome_plan_reports_bad_source_on_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            result = self.run_cli("modelome-plan", str(missing), "-o", "-")

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("error:", result.stderr)

    def test_modelome_tree_reports_bad_bindings_on_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = self.write_entries(root)
            profiles = root / "profiles"
            profiles.mkdir()
            bindings = root / "bindings.json"
            bindings.write_text("{bad json", encoding="utf-8")
            result = self.run_cli(
                "modelome-tree",
                str(entries),
                "--profiles-dir",
                str(profiles),
                "--bindings",
                str(bindings),
                "--max-taxa",
                "5",
                "--newick",
                str(root / "tree.nwk"),
                "-o",
                "-",
            )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("error:", result.stderr)

    def test_modelome_tree_rejects_nonpositive_max_taxa(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = self.write_entries(root)
            profiles = root / "profiles"
            profiles.mkdir()
            result = self.run_cli(
                "modelome-tree",
                str(entries),
                "--profiles-dir",
                str(profiles),
                "--bindings",
                str(root / "bindings.json"),
                "--max-taxa",
                "0",
                "--newick",
                str(root / "tree.nwk"),
                "-o",
                "-",
            )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("error:", result.stderr)

    def test_modelome_tree_writes_newick_with_json_to_separate_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_json = root / "tree.json"
            output_newick = root / "tree.nwk"
            result = self.run_cli(
                "modelome-tree",
                str(ROOT / "examples/modelome/entries"),
                "--profiles-dir",
                str(ROOT / "examples/modelome/profiles"),
                "--max-taxa",
                "5",
                "-o",
                str(output_json),
                "--newick",
                str(output_newick),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, "")
            output = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(output["status"], "inferred")
            self.assertIsNotNone(output["tree"])
            newick = output_newick.read_text(encoding="utf-8")
            self.assertTrue(newick.endswith(";\n"), newick)

    def test_modelome_tree_rejects_newick_without_a_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = self.write_entries(root)
            result = self.run_cli(
                "modelome-tree",
                str(entries),
                "--newick",
                str(root / "tree.nwk"),
                "-o",
                "-",
            )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("cannot write Newick", result.stderr)

    def test_modelome_tree_rejects_json_and_newick_stdout_collision(self):
        result = self.run_cli(
            "modelome-tree",
            str(ROOT / "examples/modelome/entries"),
            "--profiles-dir",
            str(ROOT / "examples/modelome/profiles"),
            "-o",
            "-",
            "--newick",
            "-",
        )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("cannot write JSON and Newick to standard output together", result.stderr)


if __name__ == "__main__":
    unittest.main()
