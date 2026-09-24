from __future__ import annotations

import copy
import unittest

from phylodigy.popularity_corpus import (
    fetch_citation_evidence,
    fetch_popularity_manifest,
    validate_popularity_manifest,
)


class PopularityManifestTests(unittest.TestCase):
    def test_ranking_is_frozen_by_likes_and_citations_are_external(self):
        listing = [
            {
                "downloads": 10,
                "gated": False,
                "id": "example/lower",
                "library_name": "transformers",
                "likes": 20,
                "pipeline_tag": "text-generation",
                "sha": "b" * 40,
                "tags": ["transformers", "arxiv:1234.56789", "license:mit"],
            },
            {
                "downloads": 5,
                "gated": False,
                "id": "example/higher",
                "library_name": "transformers",
                "likes": 30,
                "pipeline_tag": "text-generation",
                "sha": "a" * 40,
                "tags": ["transformers", "custom_code"],
            },
        ]

        def fetch(url):
            if "huggingface.co" in url:
                return listing
            return {
                "results": [
                    {
                        "cited_by_count": 999,
                        "display_name": "Example paper",
                        "doi": None,
                        "id": "https://openalex.org/W1",
                        "locations": [
                            {
                                "landing_page_url": (
                                    "https://arxiv.org/abs/1234.56789"
                                )
                            }
                        ],
                    }
                ]
            }

        manifest = fetch_popularity_manifest(limit=2, fetch_json=fetch)

        self.assertEqual(
            [item["hub_id"] for item in manifest["entries"]],
            ["example/higher", "example/lower"],
        )
        self.assertEqual(manifest["entries"][1]["maximum_citation_count"], 999)
        self.assertTrue(manifest["entries"][0]["requires_custom_code"])
        self.assertEqual(validate_popularity_manifest(manifest), manifest)

    def test_digest_and_rank_tampering_are_rejected(self):
        listing = [
            {"id": "a/model", "sha": "a" * 40, "likes": 2, "tags": []},
            {"id": "b/model", "sha": "b" * 40, "likes": 1, "tags": []},
        ]
        manifest = fetch_popularity_manifest(
            limit=2,
            include_citations=False,
            fetch_json=lambda _url: listing,
        )

        changed = copy.deepcopy(manifest)
        changed["entries"][0]["likes_at_snapshot"] = 0
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_popularity_manifest(changed)

        changed = copy.deepcopy(manifest)
        changed["entries"][0]["rank"] = 2
        with self.assertRaisesRegex(ValueError, "ranks"):
            validate_popularity_manifest(changed)

    def test_citation_snapshot_is_batched_and_manifest_bound(self):
        listing = [
            {
                "id": "a/model",
                "sha": "a" * 40,
                "likes": 2,
                "tags": ["arxiv:1234.56789"],
            },
            {
                "id": "b/model",
                "sha": "b" * 40,
                "likes": 1,
                "tags": ["arxiv:1234.56789", "arxiv:9999.00001"],
            },
        ]
        manifest = fetch_popularity_manifest(
            limit=2,
            include_citations=False,
            fetch_json=lambda _url: listing,
        )
        requests = []

        def post(url, payload):
            requests.append((url, payload))
            return [
                {
                    "citationCount": 42,
                    "externalIds": {"ArXiv": "1234.56789"},
                    "paperId": "paper-1",
                    "title": "A paper",
                    "url": "https://example.test/paper-1",
                    "year": 2020,
                },
                None,
            ]

        evidence = fetch_citation_evidence(manifest, post_json=post)

        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0][1]["ids"],
            ["ARXIV:1234.56789", "ARXIV:9999.00001"],
        )
        self.assertEqual(evidence["manifest_digest"], manifest["digest"])
        self.assertEqual(evidence["papers"][0]["citation_count"], 42)
        self.assertFalse(evidence["papers"][1]["resolved"])


if __name__ == "__main__":
    unittest.main()
