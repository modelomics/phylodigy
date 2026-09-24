from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from phylodigy.computation_graph import build_graph_profile
from phylodigy.huggingface_profile import (
    HuggingFaceProfileError,
    _valid_huggingface_repo_id,
    extract_huggingface_genome,
)
from phylodigy.schema import ArchitecturalGenome


class HuggingFaceProfileTests(unittest.TestCase):
    revision = "a" * 40

    def test_repository_length_limit_excludes_namespace(self):
        self.assertTrue(_valid_huggingface_repo_id("namespace/" + "a" * 96))
        self.assertFalse(_valid_huggingface_repo_id("namespace/" + "a" * 97))

    def setUp(self):
        graph = build_graph_profile(
            [
                {"name": "x", "op": "placeholder", "target": "x", "inputs": []},
                {"name": "out", "op": "output", "target": "out", "inputs": ["x"]},
            ],
            radii=(0,),
        )
        self.genome = ArchitecturalGenome("entry:one", graph, name="org/model")

    def test_verifies_commit_and_returns_pinned_graph_profile(self):
        info = SimpleNamespace(sha=self.revision, gated=False)
        with (
            patch(
                "phylodigy.huggingface_profile._hub_model_info", return_value=info
            ) as verify,
            patch(
                "phylodigy.popularity_corpus._model_from_entry",
                return_value=(object(), {"x": 1}, "TinyModel"),
            ) as load,
            patch(
                "phylodigy.huggingface_profile.extract_architectural_genome",
                return_value=self.genome,
            ) as extract,
            patch(
                "phylodigy.huggingface_profile._runtime_versions",
                return_value={
                    "torch": "2",
                    "transformers": "4",
                    "huggingface_hub": "0.25",
                },
            ),
        ):
            result = extract_huggingface_genome(
                "org/model", self.revision, artifact_id="entry:one", radii=(0,)
            )

        verify.assert_called_once_with("org/model", self.revision)
        entry = load.call_args.args[0]
        self.assertEqual(entry["hub_id"], "org/model")
        self.assertEqual(entry["hub_revision"], self.revision)
        self.assertIs(entry["gated"], False)
        self.assertIs(entry["requires_custom_code"], False)
        extract.assert_called_once()
        provenance = result.metadata["huggingface"]
        self.assertEqual(provenance["repo_id"], "org/model")
        self.assertEqual(provenance["revision"], self.revision)
        self.assertFalse(provenance["weights_loaded"])
        self.assertFalse(provenance["trust_remote_code"])

    def test_accepts_unscoped_hugging_face_repository_id(self):
        with (
            patch(
                "phylodigy.huggingface_profile._hub_model_info",
                return_value=SimpleNamespace(sha=self.revision, gated=False),
            ) as verify,
            patch(
                "phylodigy.popularity_corpus._model_from_entry",
                return_value=(object(), {"x": 1}, "TinyModel"),
            ),
            patch(
                "phylodigy.huggingface_profile.extract_architectural_genome",
                return_value=self.genome,
            ),
            patch("phylodigy.huggingface_profile._runtime_versions", return_value={}),
        ):
            extract_huggingface_genome(
                "_model_name", self.revision, artifact_id="entry:one", radii=(0,)
            )
        verify.assert_called_once_with("_model_name", self.revision)

    def test_rejects_non_exact_repo_or_revision_without_hub_call(self):
        with patch("phylodigy.huggingface_profile._hub_model_info") as verify:
            for repo_id, revision in (
                ("https://huggingface.co/org/model", self.revision),
                ("org/../model", self.revision),
                ("org/invalid--repo", self.revision),
                ("org/model", "main"),
                ("org/model", "a" * 39),
            ):
                with self.subTest(repo_id=repo_id, revision=revision):
                    with self.assertRaises(ValueError):
                        extract_huggingface_genome(
                            repo_id, revision, artifact_id="entry"
                        )
            verify.assert_not_called()

    def test_normalizes_hex_revision_before_verification(self):
        with (
            patch(
                "phylodigy.huggingface_profile._hub_model_info",
                return_value=SimpleNamespace(sha=self.revision, gated=False),
            ) as verify,
            patch(
                "phylodigy.popularity_corpus._model_from_entry",
                return_value=(object(), {"x": 1}, "TinyModel"),
            ),
            patch(
                "phylodigy.huggingface_profile.extract_architectural_genome",
                return_value=self.genome,
            ),
            patch("phylodigy.huggingface_profile._runtime_versions", return_value={}),
        ):
            result = extract_huggingface_genome(
                "org/model", self.revision.upper(), artifact_id="entry:one", radii=(0,)
            )
        verify.assert_called_once_with("org/model", self.revision)
        self.assertEqual(result.metadata["huggingface"]["revision"], self.revision)

    def test_invalid_radii_are_rejected_before_hub_call(self):
        with patch("phylodigy.huggingface_profile._hub_model_info") as verify:
            with self.assertRaises((TypeError, ValueError)):
                extract_huggingface_genome(
                    "org/model", self.revision, artifact_id="entry", radii=()
                )
            verify.assert_not_called()

    def test_gated_repo_is_rejected_before_model_construction(self):
        with (
            patch(
                "phylodigy.huggingface_profile._hub_model_info",
                return_value=SimpleNamespace(sha=self.revision, gated=True),
            ),
            patch("phylodigy.popularity_corpus._model_from_entry") as load,
        ):
            with self.assertRaisesRegex(HuggingFaceProfileError, "gated"):
                extract_huggingface_genome(
                    "org/model", self.revision, artifact_id="entry"
                )
            load.assert_not_called()

    def test_private_repo_is_rejected_before_model_construction(self):
        with (
            patch(
                "phylodigy.huggingface_profile._hub_model_info",
                return_value=SimpleNamespace(
                    sha=self.revision, gated=False, private=True
                ),
            ),
            patch("phylodigy.popularity_corpus._model_from_entry") as load,
        ):
            with self.assertRaisesRegex(HuggingFaceProfileError, "private"):
                extract_huggingface_genome(
                    "org/model", self.revision, artifact_id="entry"
                )
            load.assert_not_called()

    def test_mismatched_returned_sha_is_rejected(self):
        with (
            patch(
                "phylodigy.huggingface_profile._hub_model_info",
                return_value=SimpleNamespace(sha="b" * 40, gated=False),
            ),
            patch("phylodigy.popularity_corpus._model_from_entry") as load,
        ):
            with self.assertRaisesRegex(HuggingFaceProfileError, "returned revision"):
                extract_huggingface_genome(
                    "org/model", self.revision, artifact_id="entry"
                )
            load.assert_not_called()

    def test_graph_node_limit_is_enforced(self):
        info = SimpleNamespace(sha=self.revision, gated=False)
        with (
            patch("phylodigy.huggingface_profile._hub_model_info", return_value=info),
            patch(
                "phylodigy.popularity_corpus._model_from_entry",
                return_value=(object(), {"x": 1}, "TinyModel"),
            ),
            patch(
                "phylodigy.huggingface_profile.extract_architectural_genome",
                return_value=self.genome,
            ),
            patch("phylodigy.huggingface_profile._runtime_versions", return_value={}),
        ):
            with self.assertRaisesRegex(HuggingFaceProfileError, "limit is 1"):
                extract_huggingface_genome(
                    "org/model",
                    self.revision,
                    artifact_id="entry:one",
                    max_graph_nodes=1,
                )


if __name__ == "__main__":
    unittest.main()
