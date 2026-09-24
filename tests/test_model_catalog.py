from __future__ import annotations

import copy
import unittest

from phylodigy.model_catalog import (
    CatalogQuery,
    DateWindow,
    HuggingFaceCatalogProvider,
    ImportanceFilter,
    RecordFeedCatalogProvider,
    bind_primary_papers,
    discover_catalog,
    select_catalog,
    validate_catalog_snapshot,
)


def feed_records(count=10):
    return [
        {
            "access_kind": "closed" if index == 0 else "public_artifact",
            "created_at": f"2026-08-{20 + index:02d}",
            "entity_kind": "vendor_model" if index == 0 else "model",
            "id": f"model-{index}",
            "metrics": {"adoption": count - index},
            "name": f"Model {index}",
        }
        for index in range(count)
    ]


class QueryContractTests(unittest.TestCase):
    def test_top_k_top_p_and_date_windows_are_strict(self):
        self.assertEqual(ImportanceFilter("citations", top_k=3).top_k, 3)
        self.assertEqual(ImportanceFilter("citations", top_p=0.1).top_p, 0.1)
        for kwargs in (
            {},
            {"top_k": 1, "top_p": 0.5},
            {"top_k": 0},
            {"top_p": 0},
            {"top_p": 1.01},
        ):
            with self.assertRaises((TypeError, ValueError)):
                ImportanceFilter("citations", **kwargs)

        window = DateWindow(
            field="created", since="2026-08-21", until="2026-08-23"
        )
        self.assertTrue(window.includes("2026-08-21T00:00:00Z"))
        self.assertFalse(window.includes("2026-08-23T00:00:00Z"))
        with self.assertRaises(ValueError):
            DateWindow(since="2026-08-23", until="2026-08-23")

    def test_record_feed_supports_exact_top_fraction(self):
        provider = RecordFeedCatalogProvider(
            "vendor", feed_records(), observed_at="2026-09-01"
        )
        query = CatalogQuery(
            provider="vendor",
            importance=ImportanceFilter("adoption", top_p=0.25),
            page_size=2,
        )
        result = discover_catalog(query, provider=provider)

        self.assertEqual(len(result["entries"]), 3)
        self.assertEqual(
            [item["selection"]["importance_score"] for item in result["entries"]],
            [10.0, 9.0, 8.0],
        )
        self.assertEqual(result["scan"]["stopped_reason"], "importance_bound_satisfied")
        self.assertFalse(result["scan"]["population_exhausted"])
        self.assertIsNotNone(result["scan"]["next_cursor"])
        self.assertEqual(validate_catalog_snapshot(result), result)

    def test_record_feed_rejects_duplicate_version_identities(self):
        records = feed_records(2)
        records[1]["version_id"] = "vendor:model-0"
        records[0]["version_id"] = "vendor:model-0"
        with self.assertRaisesRegex(ValueError, "version_id values must be unique"):
            RecordFeedCatalogProvider("vendor", records, observed_at="2026-09-01")

    def test_date_window_is_filtered_before_local_ranking(self):
        provider = RecordFeedCatalogProvider(
            "vendor", feed_records(5), observed_at="2026-09-01"
        )
        query = CatalogQuery(
            provider="vendor",
            date_window=DateWindow(
                field="created", since="2026-08-21", until="2026-08-24"
            ),
            importance=ImportanceFilter("adoption", top_k=2),
            page_size=2,
        )
        result = discover_catalog(query, provider=provider)

        self.assertEqual(
            [item["native_id"] for item in result["entries"]],
            ["model-1", "model-2"],
        )
        self.assertTrue(result["scan"]["population_exhausted"])

    def test_page_budget_can_resume_an_exhaustive_scan(self):
        provider = RecordFeedCatalogProvider(
            "vendor", feed_records(5), observed_at="2026-09-01"
        )
        query = CatalogQuery(provider="vendor", page_size=2)
        first = discover_catalog(query, provider=provider, max_pages=1)
        self.assertFalse(first["scan"]["complete"])
        self.assertEqual(first["scan"]["next_cursor"], "offset:2")

        resumed = discover_catalog(query, provider=provider, resume=first)
        self.assertTrue(resumed["scan"]["complete"])
        self.assertEqual(len(resumed["entries"]), 5)
        self.assertEqual(resumed["scan"]["pages_fetched"], 3)

    def test_huggingface_adapter_uses_lightweight_fields_and_opaque_link(self):
        calls = []

        def fetch(url):
            calls.append(url)
            return (
                [
                    {
                        "createdAt": "2026-08-31T12:00:00Z",
                        "downloads": 7,
                        "downloadsAllTime": 70,
                        "gated": False,
                        "id": "example/model",
                        "lastModified": "2026-08-31T13:00:00Z",
                        "likes": 3,
                        "sha": "a" * 40,
                        "tags": ["arxiv:1234.56789"],
                    }
                ],
                {},
            )

        provider = HuggingFaceCatalogProvider(
            fetch_page=fetch, observed_at="2026-09-01"
        )
        result = discover_catalog(
            CatalogQuery(
                provider="huggingface",
                importance=ImportanceFilter("likes", top_k=1),
            ),
            provider=provider,
        )

        self.assertIn("expand=createdAt", calls[0])
        self.assertIn("sort=likes", calls[0])
        self.assertNotIn("full=true", calls[0])
        self.assertEqual(result["entries"][0]["revision"], "a" * 40)
        self.assertEqual(
            result["entries"][0]["paper_candidates"][0]["identifier"],
            "arxiv:1234.56789",
        )

    def test_digest_tampering_is_rejected(self):
        provider = RecordFeedCatalogProvider(
            "vendor", feed_records(2), observed_at="2026-09-01"
        )
        result = discover_catalog(CatalogQuery(provider="vendor"), provider=provider)
        changed = copy.deepcopy(result)
        changed["entries"][0]["name"] = "tampered"
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_catalog_snapshot(changed)


class PaperGroundingTests(unittest.TestCase):
    def setUp(self):
        provider = RecordFeedCatalogProvider(
            "vendor", feed_records(3), observed_at="2026-09-01"
        )
        self.catalog = discover_catalog(
            CatalogQuery(provider="vendor"), provider=provider
        )

    def test_primary_papers_are_unique_and_paperless_is_explicit(self):
        versions = [item["version_id"] for item in self.catalog["entries"]]
        grounded = bind_primary_papers(
            self.catalog,
            [
                {
                    "citation_count": 100,
                    "citation_provider": "example",
                    "paper_id": "doi:10/example.a",
                    "version_id": versions[0],
                },
                {
                    "citation_count": 50,
                    "citation_provider": "example",
                    "paper_id": "arxiv:1234.56789",
                    "version_id": versions[1],
                },
                {
                    "paperless_reason": "vendor release with no primary paper",
                    "version_id": versions[2],
                },
            ],
            require_complete=True,
        )
        selection = select_catalog(
            grounded, importance=ImportanceFilter("citations", top_k=2)
        )

        self.assertEqual(grounded["paper_coverage"]["bound_primary_paper_count"], 2)
        self.assertEqual(grounded["paper_coverage"]["paperless_exception_count"], 1)
        self.assertEqual(len(selection["entries"]), 2)
        self.assertEqual(selection["entries"][0]["selection"]["importance_score"], 100.0)

    def test_duplicate_primary_paper_is_rejected_by_corpus_policy(self):
        versions = [item["version_id"] for item in self.catalog["entries"]]
        with self.assertRaisesRegex(ValueError, "unique"):
            bind_primary_papers(
                self.catalog,
                [
                    {"paper_id": "doi:10/shared", "version_id": versions[0]},
                    {"paper_id": "doi:10/shared", "version_id": versions[1]},
                ],
                require_unique_papers=True,
            )

    def test_distinct_paper_policy_belongs_to_selection_not_registry_binding(self):
        versions = [item["version_id"] for item in self.catalog["entries"]]
        grounded = bind_primary_papers(
            self.catalog,
            [
                {
                    "citation_count": 10,
                    "paper_id": "doi:10/shared",
                    "version_id": versions[0],
                },
                {
                    "citation_count": 10,
                    "paper_id": "doi:10/shared",
                    "version_id": versions[1],
                },
            ],
        )
        selection = select_catalog(
            grounded,
            importance=ImportanceFilter("citations", top_k=2),
            distinct_primary_papers=True,
            paperless_quota=0,
        )

        self.assertEqual(len(selection["entries"]), 1)
        self.assertEqual(
            len(selection["selection"]["duplicate_primary_paper_exclusions"]), 1
        )


if __name__ == "__main__":
    unittest.main()
