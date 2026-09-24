from __future__ import annotations

import unittest

from phylodigy.modelome_targets import resolve_modelome_targets


SHA_A = "a" * 40
SHA_B = "b" * 40


def ident(namespace: str, value: str) -> dict[str, str]:
    return {"namespace": namespace, "value": value}


class ModelomeTargetTests(unittest.TestCase):
    def test_resolves_exact_entry_level_repo_and_revision(self):
        entries = [
            {
                "id": "entry:1",
                "identifiers": [
                    ident("huggingface:model", "org/model"),
                    ident("huggingface:revision", f"org/model@{SHA_A.upper()}"),
                ],
            }
        ]
        self.assertEqual(
            resolve_modelome_targets(entries),
            [
                {
                    "entry_id": "entry:1",
                    "status": "ready",
                    "reason": "one exact Hugging Face repository and revision candidate",
                    "target": {"repo_id": "org/model", "revision": SHA_A},
                    "candidates": [{"repo_id": "org/model", "revision": SHA_A}],
                }
            ],
        )

    def test_upstream_release_revision_uses_its_qualified_repo(self):
        entry = {
            "id": "entry:1",
            "identifiers": [
                ident("huggingface:model", "org/model"),
            ],
            "releases": [
                {
                    "identifiers": [
                        ident("huggingface:revision", f"org/model@{SHA_B}"),
                    ]
                }
            ],
        }
        row = resolve_modelome_targets([entry])[0]
        self.assertEqual(row["target"], {"repo_id": "org/model", "revision": SHA_B})
        self.assertEqual(
            row["candidates"], [{"repo_id": "org/model", "revision": SHA_B}]
        )

    def test_unscoped_repository_in_exact_revision_is_supported(self):
        row = resolve_modelome_targets(
            [
                {
                    "id": "entry:bare",
                    "identifiers": [
                        ident("huggingface:revision", f"_model_name@{SHA_A}")
                    ],
                }
            ]
        )[0]
        self.assertEqual(row["status"], "ready")
        self.assertEqual(row["target"], {"repo_id": "_model_name", "revision": SHA_A})

    def test_entry_and_release_revision_candidates_are_unioned(self):
        entry = {
            "id": "entry:1",
            "identifiers": [
                ident("huggingface:model", "org/model"),
                ident("huggingface:revision", f"org/model@{SHA_A}"),
            ],
            "releases": [
                {
                    "identifiers": [
                        ident("huggingface:revision", f"org/model@{SHA_B}"),
                    ]
                }
            ],
        }
        row = resolve_modelome_targets([entry])[0]
        self.assertEqual(row["status"], "ambiguous_reference")
        self.assertEqual(
            row["candidates"],
            [
                {"repo_id": "org/model", "revision": SHA_A},
                {"repo_id": "org/model", "revision": SHA_B},
            ],
        )

    def test_multiple_exact_pairs_are_ambiguous_and_pin_selects_only_evidence(self):
        entry = {
            "id": "entry:1",
            "identifiers": [
                ident("huggingface:model", "org/model"),
                ident("huggingface:revision", f"org/model@{SHA_A}"),
                ident("huggingface:revision", f"org/model@{SHA_B}"),
            ],
        }
        row = resolve_modelome_targets([entry])[0]
        self.assertEqual(row["status"], "ambiguous_reference")
        self.assertIsNone(row["target"])
        selected = resolve_modelome_targets(
            [entry], pins={"entry:1": {"repo_id": "org/model", "revision": SHA_B}}
        )[0]
        self.assertEqual(selected["status"], "ready")
        self.assertEqual(
            selected["target"], {"repo_id": "org/model", "revision": SHA_B}
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_modelome_targets(
                [entry],
                pins={"entry:1": {"repo_id": "org/model", "revision": "c" * 40}},
            )

    def test_unpinned_missing_and_unsupported_are_explicit(self):
        entries = [
            {"id": "none", "canonical_name": "A famous model"},
            {
                "id": "bad",
                "identifiers": [
                    ident("huggingface:model", "org/model"),
                    ident("huggingface:revision", "main"),
                ],
            },
        ]
        rows = resolve_modelome_targets(entries)
        by_id = {row["entry_id"]: row for row in rows}
        self.assertEqual(by_id["none"]["status"], "missing_pinned_reference")
        self.assertEqual(by_id["none"]["candidates"], [])
        self.assertEqual(by_id["bad"]["status"], "unsupported_reference")
        self.assertIsNone(by_id["bad"]["target"])

    def test_pin_cannot_reference_unknown_entry_or_be_malformed(self):
        with self.assertRaisesRegex(ValueError, "unknown entry"):
            resolve_modelome_targets(
                [], pins={"ghost": {"repo_id": "org/model", "revision": SHA_A}}
            )
        with self.assertRaisesRegex(ValueError, "valid repo_id"):
            resolve_modelome_targets(
                [{"id": "entry"}],
                pins={"entry": {"repo_id": "model", "revision": "bad"}},
            )

    def test_non_huggingface_names_and_urls_do_not_resolve(self):
        row = resolve_modelome_targets(
            [
                {
                    "id": "entry",
                    "canonical_name": "org/model",
                    "aliases": ["org/model"],
                    "resources": [
                        {"url": f"https://huggingface.co/org/model/commit/{SHA_A}"}
                    ],
                    "identifiers": [ident("doi", f"https://doi.org/{SHA_A}")],
                }
            ]
        )[0]
        self.assertEqual(row["status"], "missing_pinned_reference")

    def test_output_rows_are_sorted_by_entry_id(self):
        rows = resolve_modelome_targets([{"id": "z"}, {"id": "a"}])
        self.assertEqual([row["entry_id"] for row in rows], ["a", "z"])


if __name__ == "__main__":
    unittest.main()
