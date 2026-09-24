from __future__ import annotations

import unittest
from urllib.parse import parse_qs, urlparse

from phylodigy.model_catalog import (
    CatalogPage,
    CatalogQuery,
    GitHubCatalogProvider,
    ImportanceFilter,
    ModelCatalogError,
    OpenAlexPaperProvider,
    ProviderCapabilities,
    discover_catalog,
)


def _github_repository(name: str, stars: int) -> dict[str, object]:
    return {
        "created_at": "2024-01-02T03:04:05Z",
        "default_branch": "main",
        "fork": False,
        "forks_count": 7,
        "full_name": name,
        "html_url": f"https://github.com/{name}",
        "license": {"spdx_id": "Apache-2.0"},
        "open_issues_count": 2,
        "private": False,
        "pushed_at": "2026-08-30T12:00:00Z",
        "stargazers_count": stars,
        "subscribers_count": 3,
        "topics": ["machine-learning", "models"],
    }


def _openalex_work(
    work_id: str,
    *,
    citations: int,
    fwci: float,
    title: str,
) -> dict[str, object]:
    return {
        "cited_by_count": citations,
        "display_name": title,
        "fwci": fwci,
        "id": f"https://openalex.org/{work_id}",
        "ids": {
            "doi": f"https://doi.org/10.1000/{work_id.casefold()}",
            "openalex": f"https://openalex.org/{work_id}",
            "pmid": f"https://pubmed.ncbi.nlm.nih.gov/{work_id[1:]}",
        },
        "primary_location": {
            "landing_page_url": f"https://papers.example/{work_id}"
        },
        "publication_date": "2024-01-02",
        "updated_date": "2026-08-30T12:34:56Z",
    }


class ProviderContractTests(unittest.TestCase):
    def test_github_result_cap_prevents_false_exhaustive_completeness(self):
        def fetch(_url):
            return (
                {
                    "items": [_github_repository("example/model", 500)],
                    "total_count": 1500,
                },
                {},
            )

        provider = GitHubCatalogProvider(
            fetch_page=fetch, observed_at="2026-08-31T00:00:00Z"
        )
        exhaustive = discover_catalog(
            CatalogQuery(provider="github", search="topic:machine-learning"),
            provider=provider,
        )

        self.assertEqual(
            exhaustive["scan"]["stopped_reason"],
            "provider_result_cap_requires_partition",
        )
        self.assertFalse(exhaustive["scan"]["complete"])
        self.assertFalse(exhaustive["scan"]["query_complete"])
        self.assertFalse(exhaustive["scan"]["coverage_complete"])
        self.assertFalse(exhaustive["scan"]["population_exhausted"])
        self.assertFalse(exhaustive["scan"]["selection_applied"])
        self.assertEqual(exhaustive["scan"]["provider_reported_total"], 1500)
        self.assertEqual(exhaustive["provider_capabilities"]["result_cap"], 1000)

        top_one = discover_catalog(
            CatalogQuery(
                provider="github",
                search="topic:machine-learning",
                importance=ImportanceFilter("github_stars", top_k=1),
            ),
            provider=provider,
        )

        self.assertTrue(top_one["scan"]["query_complete"])
        self.assertFalse(top_one["scan"]["coverage_complete"])
        self.assertTrue(top_one["scan"]["selection_applied"])
        self.assertEqual(
            top_one["scan"]["stopped_reason"], "importance_bound_satisfied"
        )
        self.assertEqual(
            top_one["entries"][0]["selection"]["importance_score"], 500.0
        )

    def test_github_top_fraction_does_not_treat_capped_total_as_exact(self):
        def fetch(_url):
            return (
                {
                    "items": [_github_repository("example/model", 500)],
                    "total_count": 1500,
                },
                {},
            )

        provider = GitHubCatalogProvider(
            fetch_page=fetch, observed_at="2026-08-31T00:00:00Z"
        )
        result = discover_catalog(
            CatalogQuery(
                provider="github",
                search="topic:machine-learning",
                importance=ImportanceFilter("github_stars", top_p=0.1),
            ),
            provider=provider,
        )

        self.assertFalse(result["scan"]["complete"])
        self.assertFalse(result["scan"]["selection_applied"])
        self.assertNotIn("selection", result["entries"][0])
        self.assertEqual(
            result["scan"]["stopped_reason"],
            "provider_result_cap_requires_partition",
        )

    def test_openalex_normalizes_papers_and_applies_citation_top_k(self):
        calls: list[str] = []

        def fetch(url):
            calls.append(url)
            return (
                {
                    "meta": {"count": 2, "next_cursor": None},
                    "results": [
                        _openalex_work(
                            "W2", citations=10, fwci=1.25, title="Lower paper"
                        ),
                        _openalex_work(
                            "W1", citations=80, fwci=4.5, title="Higher paper"
                        ),
                    ],
                },
                {},
            )

        provider = OpenAlexPaperProvider(
            fetch_page=fetch, observed_at="2026-08-31T00:00:00Z"
        )
        result = discover_catalog(
            CatalogQuery(
                provider="openalex",
                search="machine learning model",
                importance=ImportanceFilter("citations", top_k=2),
            ),
            provider=provider,
        )

        parameters = parse_qs(urlparse(calls[0]).query)
        self.assertEqual(parameters["cursor"], ["*"])
        self.assertEqual(parameters["search"], ["machine learning model"])
        self.assertEqual(parameters["sort"], ["cited_by_count:desc"])
        self.assertIn("cited_by_count", parameters["select"][0])
        self.assertEqual(
            [entry["native_id"] for entry in result["entries"]], ["W1", "W2"]
        )
        self.assertEqual(
            [entry["selection"]["importance_score"] for entry in result["entries"]],
            [80.0, 10.0],
        )
        higher = result["entries"][0]
        self.assertEqual(higher["catalog_id"], "openalex:W1")
        self.assertEqual(higher["version_id"], "openalex:W1")
        self.assertEqual(higher["entity_kind"], "paper_candidate")
        self.assertEqual(higher["created_at"], "2024-01-02T00:00:00Z")
        self.assertEqual(higher["modified_at"], "2026-08-30T12:34:56Z")
        self.assertEqual(higher["metrics"], {"citations": 80, "fwci": 4.5})
        self.assertEqual(
            higher["identifiers"],
            [
                "https://doi.org/10.1000/w1",
                "https://openalex.org/W1",
                "https://pubmed.ncbi.nlm.nih.gov/1",
            ],
        )
        self.assertEqual(
            higher["source_locators"],
            ["https://openalex.org/W1", "https://papers.example/W1"],
        )
        self.assertTrue(result["scan"]["query_complete"])
        self.assertTrue(result["scan"]["selection_applied"])
        self.assertFalse(result["scan"]["provider_order_verified"])

    def test_repeated_provider_cursor_is_rejected(self):
        class LoopingProvider:
            name = "loop"
            capabilities = ProviderCapabilities(entity_kinds=("model",))

            def __init__(self):
                self.calls: list[str | None] = []

            def fetch_page(self, _query, cursor=None):
                self.calls.append(cursor)
                index = len(self.calls)
                return CatalogPage(
                    entries=(
                        {
                            "catalog_id": f"loop:model-{index}",
                            "native_id": f"model-{index}",
                            "provider": "loop",
                            "version_id": f"loop:model-{index}",
                        },
                    ),
                    next_cursor="same-cursor",
                )

        provider = LoopingProvider()
        with self.assertRaisesRegex(
            ModelCatalogError, "pagination cursor repeated"
        ):
            discover_catalog(CatalogQuery(provider="loop"), provider=provider)

        self.assertEqual(provider.calls, [None, "same-cursor"])


if __name__ == "__main__":
    unittest.main()
