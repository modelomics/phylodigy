from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ROOT / "examples/modelome/entries"


class ModelomeExtractCliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        )
        return subprocess.run(
            [sys.executable, "-m", "phylodigy.cli", *args],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_extract_plan_uses_entry_fixture_offline(self):
        with tempfile.TemporaryDirectory() as temporary:
            pins = Path(temporary) / "pins.json"
            pins.write_text("{}", encoding="utf-8")
            result = self.run_cli(
                "modelome-extract-plan", str(ENTRIES), "--pins", str(pins), "-o", "-"
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        plan = json.loads(result.stdout)
        self.assertEqual(plan["artifact_type"], "phylodigy.modelome_extraction_plan")
        self.assertEqual(len(plan["jobs"]), 3)
        self.assertEqual(plan["counts"], {"missing_pinned_reference": 3})

    def test_extract_without_ready_targets_is_offline_and_emits_machine_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "profiles"
            result = self.run_cli(
                "modelome-extract",
                str(ENTRIES),
                "--output-dir",
                str(output_dir),
                "-o",
                "-",
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "finished")
        self.assertFalse(report["complete_profile_coverage"])
        self.assertEqual(report["counts"], {"missing_pinned_reference": 3})

    def test_extract_rejects_invalid_controls_without_attempting_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.run_cli(
                "modelome-extract",
                str(ENTRIES),
                "--output-dir",
                str(Path(temporary) / "profiles"),
                "--max-models",
                "0",
                "-o",
                "-",
            )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("max_models must be a positive integer", result.stderr)

    def test_extract_plan_rejects_invalid_pins_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            pins = Path(temporary) / "pins.json"
            pins.write_text("{invalid", encoding="utf-8")
            result = self.run_cli(
                "modelome-extract-plan", str(ENTRIES), "--pins", str(pins), "-o", "-"
            )

        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
