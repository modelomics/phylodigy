from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from phylodigy.computation_graph import build_graph_profile
from phylodigy.modelome import build_modelome_tree
from phylodigy.modelome_extraction import extract_modelome_profiles
from phylodigy.modelome_profiles import load_modelome_profiles
from phylodigy.schema import ArchitecturalGenome


REV_A = "a" * 40
REV_B = "b" * 40


def _entry(entry_id: str, *, revisions=(REV_A,), extra_identifiers=()):
    return {
        "id": entry_id,
        "canonical_name": entry_id,
        "aliases": [],
        "identifiers": [
            {
                "namespace": "huggingface:revision",
                "value": f"org/{entry_id.replace(':', '-')}@{revision}",
            }
            for revision in revisions
        ]
        + list(extra_identifiers),
        "tags": [],
        "members": [],
        "resources": [],
        "releases": [],
        "model_relations": [],
    }


def _genome(entry_id: str, op: str = "vendor.alpha"):
    graph = build_graph_profile(
        [
            {"name": "input", "op": "placeholder", "target": "input", "inputs": []},
            {
                "name": "feature",
                "op": "call_function",
                "target": op,
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
    return ArchitecturalGenome(entry_id, graph)


class ModelomeExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.entries_path = self.root / "entries.json"
        self.output_dir = self.root / "runs"

    def tearDown(self):
        self.temporary.cleanup()

    def write_entries(self, entries):
        self.entries_path.write_text(json.dumps(entries), encoding="utf-8")

    def run_extraction(self, **kwargs):
        return extract_modelome_profiles(self.entries_path, self.output_dir, **kwargs)

    def stable_runtime(self):
        return {"torch": "test", "transformers": "test", "huggingface-hub": "test"}

    def test_bounded_run_resumes_success_and_continues_pending_jobs(self):
        entries = [_entry(f"model:{index}") for index in range(3)]
        self.write_entries(entries)

        def extract(job, policy):
            return _genome(job["entry_id"], f"vendor.op{job['entry_id'][-1]}")

        with (
            patch("phylodigy.modelome_extraction._runtime", self.stable_runtime),
            patch(
                "phylodigy.modelome_extraction._run_extraction", side_effect=extract
            ) as run,
        ):
            first = self.run_extraction(max_models=1)
            second = self.run_extraction(max_models=1)

        self.assertEqual(first["counts"], {"pending": 2, "succeeded": 1})
        self.assertEqual(first["attempted"], 1)
        self.assertEqual(second["counts"], {"succeeded": 2, "pending": 1})
        self.assertEqual(second["attempted"], 1)
        self.assertEqual(run.call_count, 2)
        statuses = {
            row["entry_id"]: (row["status"], row["resumed"])
            for row in second["entries"]
        }
        self.assertEqual(statuses["model:0"], ("succeeded", True))
        self.assertEqual(statuses["model:1"], ("succeeded", False))

    def test_cache_key_changes_with_policy_source_and_pin(self):
        self.write_entries([_entry("model:one", revisions=(REV_A, REV_B))])
        with (
            patch("phylodigy.modelome_extraction._runtime", self.stable_runtime),
            patch(
                "phylodigy.modelome_extraction._run_extraction",
                side_effect=lambda job, policy: _genome(job["entry_id"]),
            ) as run,
        ):
            base = self.run_extraction(
                pins={"model:one": {"repo_id": "org/model-one", "revision": REV_A}}
            )
            changed_policy = self.run_extraction(
                pins={"model:one": {"repo_id": "org/model-one", "revision": REV_A}},
                max_graph_nodes=99,
            )
            changed_pin = self.run_extraction(
                pins={"model:one": {"repo_id": "org/model-one", "revision": REV_B}}
            )
            self.write_entries(
                [
                    _entry(
                        "model:one",
                        revisions=(REV_A, REV_B),
                        extra_identifiers=[
                            {"namespace": "local:note", "value": "changed"}
                        ],
                    )
                ]
            )
            changed_source = self.run_extraction(
                pins={"model:one": {"repo_id": "org/model-one", "revision": REV_A}}
            )

        self.assertEqual(
            len(
                {
                    base["run_key"],
                    changed_policy["run_key"],
                    changed_pin["run_key"],
                    changed_source["run_key"],
                }
            ),
            4,
        )
        self.assertEqual(run.call_count, 4)

    def test_failure_retry_unsupported_pending_and_wrong_id_not_cached(self):
        ready = [_entry(f"model:{index}") for index in range(3)]
        unsupported = _entry(
            "model:unsupported",
            revisions=(),
            extra_identifiers=[
                {"namespace": "huggingface:repo_id", "value": "org/repo"}
            ],
        )
        missing = _entry("model:missing", revisions=())
        self.write_entries(ready + [unsupported, missing])
        attempts = []

        def fail(job, policy):
            attempts.append(job["entry_id"])
            raise RuntimeError("transient")

        with (
            patch("phylodigy.modelome_extraction._runtime", self.stable_runtime),
            patch("phylodigy.modelome_extraction._run_extraction", side_effect=fail),
        ):
            first = self.run_extraction(max_models=1)
            cached = self.run_extraction(max_models=1)
            retried = self.run_extraction(max_models=1, retry_failures=True)

        self.assertEqual(first["counts"]["failed"], 1)
        self.assertEqual(first["counts"]["pending"], 2)
        self.assertEqual(first["counts"]["unsupported_reference"], 1)
        self.assertEqual(first["counts"]["missing_pinned_reference"], 1)
        self.assertTrue(
            next(row for row in cached["entries"] if row["entry_id"] == attempts[0])[
                "resumed"
            ]
        )
        # The cached failure is not rerun; the freed budget advances to the next job.
        self.assertEqual(cached["attempted"], 1)
        self.assertEqual(retried["attempted"], 1)

        wrong_output = self.root / "wrong"
        self.output_dir = wrong_output
        self.write_entries([_entry("model:wrong")])
        with (
            patch("phylodigy.modelome_extraction._runtime", self.stable_runtime),
            patch(
                "phylodigy.modelome_extraction._run_extraction",
                return_value=_genome("model:other"),
            ),
        ):
            wrong = self.run_extraction()
        self.assertEqual(wrong["counts"], {"failed": 1})
        self.assertEqual(list(Path(wrong["profiles_dir"]).glob("*.genome.json")), [])

    def test_invalid_failure_cache_is_retried(self):
        self.write_entries([_entry("model:one")])
        for index, contents in enumerate(
            ("{", "{}", '{"error": "old", "run_key": "wrong"}')
        ):
            with self.subTest(contents=contents):
                self.output_dir = self.root / f"corrupt-{index}"
                with (
                    patch(
                        "phylodigy.modelome_extraction._runtime", self.stable_runtime
                    ),
                    patch(
                        "phylodigy.modelome_extraction._run_extraction",
                        side_effect=[
                            RuntimeError("temporary failure"),
                            _genome("model:one"),
                        ],
                    ) as run,
                ):
                    first = self.run_extraction()
                    failure = next(
                        (Path(first["profiles_dir"]).parent / "failures").glob("*.json")
                    )
                    failure.write_text(contents)
                    second = self.run_extraction()
                self.assertEqual(second["counts"], {"succeeded": 1})
                self.assertEqual(run.call_count, 2)
                self.assertFalse(failure.exists())

    def test_successful_profiles_load_and_feed_modelome_tree(self):
        entries = [_entry("model:one"), _entry("model:two")]
        self.write_entries(entries)

        def extract(job, policy):
            return _genome(job["entry_id"], "vendor." + job["entry_id"].split(":")[1])

        with (
            patch("phylodigy.modelome_extraction._runtime", self.stable_runtime),
            patch("phylodigy.modelome_extraction._run_extraction", side_effect=extract),
        ):
            report = self.run_extraction()
        profiles = load_modelome_profiles(report["profiles_dir"])
        tree = build_modelome_tree(
            self.entries_path, profiles_dir=report["profiles_dir"]
        )
        self.assertEqual(set(profiles), {"model:one", "model:two"})
        self.assertEqual(tree["status"], "inferred")
        self.assertEqual(tree["modelome"]["counts"]["profiled"], 2)

    def test_run_extraction_timeout_terminates_and_closes_process(self):
        events = []

        class FakeProcess:
            exitcode = None

            def start(self):
                events.append("start")

            def join(self, timeout=None):
                events.append(("join", timeout))

            def is_alive(self):
                return "terminated" not in events

            def terminate(self):
                events.append("terminated")

            def kill(self):
                events.append("killed")

            def close(self):
                events.append("closed")

        class FakeContext:
            def Process(self, **kwargs):
                return FakeProcess()

        class FakeMultiprocessing:
            def get_context(self, method):
                self.method = method
                return FakeContext()

        fake_mp = FakeMultiprocessing()
        with (
            patch("phylodigy.modelome_extraction.multiprocessing", fake_mp),
            self.assertRaisesRegex(TimeoutError, "exceeded"),
        ):
            from phylodigy.modelome_extraction import _run_extraction

            _run_extraction({"entry_id": "model:x"}, {"per_model_timeout": 0.01})
        self.assertEqual(fake_mp.method, "spawn")
        self.assertEqual(
            events, ["start", ("join", 0.01), "terminated", ("join", 5), "closed"]
        )


if __name__ == "__main__":
    unittest.main()
