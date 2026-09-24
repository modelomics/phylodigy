from __future__ import annotations

import unittest
from urllib.parse import parse_qs, urlparse

from phylodigy.curated_catalog import EpochCatalogProvider, OpenRouterCatalogProvider
from phylodigy.model_catalog import (
    CatalogQuery,
    ImportanceFilter,
    discover_catalog,
)


EPOCH_CSV = """Model,Domain,Task,Organization,Authors,Publication date,Reference,Link,Citations,Notability criteria,Parameters,Training compute (FLOP),Model accessibility,Base model,Last modified,Open model weights?
Closed Alpha,Language,Chat,Example Labs,A. Author,2024-01-02,Alpha system paper,https://arxiv.org/abs/2401.00001,250.0,Discretionary,1000000,1e20,API access,,2026-08-30 12:00:00+00:00,No
Alpha Small,Language,Chat,Example Labs,A. Author,2024-01-02,Alpha system paper,https://arxiv.org/abs/2401.00001,250.0,Discretionary,100000,1e19,Open weights (unrestricted),,2026-08-30 12:00:00+00:00,Yes
Paperless Beta,Multimodal,Generation,Vendor Inc.,,2025-03-04,Beta launch,https://vendor.example/beta,,Industry,2000000,,Hosted access (no API),Closed Alpha,2026-08-31 01:00:00+00:00,No
"""


class EpochCatalogProviderTests(unittest.TestCase):
    def test_epoch_coalesces_duplicate_source_rows_by_model_name(self):
        duplicate_csv = EPOCH_CSV + "Closed Alpha,,,,,,,,,,,,,,,\n"
        provider = EpochCatalogProvider(
            fetch_csv=lambda _url: duplicate_csv,
            observed_at="2026-08-31T12:00:00Z",
        )
        result = discover_catalog(CatalogQuery(provider="epoch"), provider=provider)

        self.assertEqual(len(result["entries"]), 3)
        closed = next(entry for entry in result["entries"] if entry["name"] == "Closed Alpha")
        self.assertEqual(closed["metadata"]["source_duplicate_row_count"], 2)

    def test_epoch_keeps_closed_models_and_paper_citations_separate(self):
        calls: list[str] = []

        def fetch(url: str) -> str:
            calls.append(url)
            return EPOCH_CSV

        provider = EpochCatalogProvider(
            fetch_csv=fetch, observed_at="2026-08-31T12:00:00Z"
        )
        result = discover_catalog(
            CatalogQuery(
                provider="epoch",
                importance=ImportanceFilter("citations", top_k=2),
                page_size=1,
            ),
            provider=provider,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [entry["name"] for entry in result["entries"]],
            ["Alpha Small", "Closed Alpha"],
        )
        first, second = result["entries"]
        self.assertEqual(first["primary_paper"]["citation_count"], 250)
        self.assertEqual(
            first["primary_paper"]["identifier"],
            second["primary_paper"]["identifier"],
        )
        self.assertNotIn("citations", first["metrics"])
        self.assertEqual(second["access_kind"], "api_access")
        self.assertEqual(second["entity_kind"], "model")
        self.assertTrue(result["scan"]["query_complete"])
        self.assertFalse(result["scan"]["coverage_complete"])

    def test_epoch_exhaustive_snapshot_retains_unresolved_paperless_candidate(self):
        provider = EpochCatalogProvider(
            fetch_csv=lambda _url: EPOCH_CSV,
            observed_at="2026-08-31T12:00:00Z",
        )
        result = discover_catalog(CatalogQuery(provider="epoch"), provider=provider)

        self.assertEqual(len(result["entries"]), 3)
        paperless = next(
            entry for entry in result["entries"] if entry["name"] == "Paperless Beta"
        )
        self.assertIsNone(paperless["primary_paper"])
        self.assertEqual(paperless["paper_status"], "unresolved")
        self.assertEqual(paperless["base_models"], ["Closed Alpha"])
        self.assertEqual(paperless["access_kind"], "hosted_access")
        self.assertTrue(result["scan"]["coverage_complete"])


class OpenRouterCatalogProviderTests(unittest.TestCase):
    def test_openrouter_paginates_current_api_offerings(self):
        calls: list[str] = []
        records = [
            {
                "architecture": {
                    "input_modalities": ["text"],
                    "modality": "text->text",
                    "output_modalities": ["text"],
                    "tokenizer": "GPT",
                },
                "canonical_slug": "vendor/closed-alpha",
                "context_length": 128000,
                "created": 1720000000,
                "description": "Closed API model",
                "id": "vendor/closed-alpha",
                "name": "Closed Alpha",
                "supported_parameters": ["temperature"],
            },
            {
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "modality": "text+image->text",
                    "output_modalities": ["text"],
                    "tokenizer": "Other",
                },
                "canonical_slug": "vendor/beta",
                "context_length": 64000,
                "created": 1730000000,
                "id": "vendor/beta",
                "name": "Beta",
                "supported_parameters": [],
            },
        ]

        def fetch(url: str):
            calls.append(url)
            parameters = parse_qs(urlparse(url).query)
            offset = int(parameters.get("offset", ["0"])[0])
            limit = int(parameters["limit"][0])
            return (
                {
                    "data": records[offset : offset + limit],
                    "links": {},
                    "total_count": len(records),
                },
                {},
            )

        provider = OpenRouterCatalogProvider(
            fetch_page=fetch, observed_at="2026-08-31T12:00:00Z"
        )
        result = discover_catalog(
            CatalogQuery(provider="openrouter", page_size=1, search="vendor"),
            provider=provider,
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [parse_qs(urlparse(url).query)["offset"] for url in calls],
            [["0"], ["1"]],
        )
        self.assertTrue(all("q" in parse_qs(urlparse(url).query) for url in calls))
        self.assertTrue(
            all(
                parse_qs(urlparse(url).query)["output_modalities"] == ["all"]
                for url in calls
            )
        )
        self.assertEqual(len(result["entries"]), 2)
        closed = next(
            entry
            for entry in result["entries"]
            if entry["native_id"] == "vendor/closed-alpha"
        )
        self.assertEqual(closed["entity_kind"], "model_offering")
        self.assertEqual(closed["access_kind"], "api_offering")
        self.assertEqual(closed["metrics"]["context_length"], 128000)
        self.assertEqual(closed["created_at"], "2024-07-03T09:46:40Z")
        self.assertTrue(result["scan"]["coverage_complete"])


if __name__ == "__main__":
    unittest.main()
