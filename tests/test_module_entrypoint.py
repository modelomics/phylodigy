from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class ModuleEntrypointTests(unittest.TestCase):
    def run_module(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(SRC), env.get("PYTHONPATH", "")) if part
        )
        return subprocess.run(
            [sys.executable, "-m", "phylodigy", *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_version_runs_from_source_checkout(self):
        result = self.run_module("--version")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "phylodigy 0.2.0")
        self.assertEqual(result.stderr, "")

    def test_help_lists_commands(self):
        result = self.run_module("--help")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: phylodigy", result.stdout)
        self.assertIn("build-genome", result.stdout)
        self.assertIn("infer-lineage", result.stdout)
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
