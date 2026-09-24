"""Curated and commercial catalog adapters.

These providers complement artifact indexes.  Epoch AI contributes curated
historical model records and associated-publication evidence.  OpenRouter
contributes a current catalog of API offerings, including models for which no
public weights or repository exist.

Neither source is treated as a universal ground truth.  Entries keep their
provider-local identity and provenance until a later reconciliation layer
links them to canonical models, papers, artifacts, or offerings.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .canonical import content_digest, normalize_json
from .model_catalog import (
    CatalogPage,
    CatalogQuery,
    HttpPageFetcher,
    ModelCatalogError,
    ProviderCapabilities,
    _instant,
)


EPOCH_MODELS_CSV_URL = "https://epoch.ai/data/all_ai_models.csv"
OPENROUTER_MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"

CsvFetcher = Callable[[str], str]


def _http_csv(url: str) -> str:
    request = Request(
        url,
        headers={
            "Accept": "text/csv",
            "User-Agent": "phylodigy-model-catalog/1",
        },
    )
    with urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8-sig")


def _bearer_json_fetcher(token: str) -> HttpPageFetcher:
    def fetch(url: str) -> tuple[Any, Mapping[str, str]]:
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "phylodigy-model-catalog/1",
            },
        )
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
            headers = {
                key.casefold(): value for key, value in response.headers.items()
            }
        return payload, headers

    return fetch


def _finite_number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _urls(value: Any) -> list[str]:
    return sorted(
        {
            match.rstrip(".,;)")
            for match in re.findall(r"https?://[^\s]+", str(value or ""))
        }
    )


def _publication_identifier(title: str, links: list[str]) -> str:
    for link in links:
        match = re.search(
            r"arxiv\.org/(?:abs|pdf)/([^?#]+)", link, flags=re.IGNORECASE
        )
        if match:
            identifier = re.sub(r"\.pdf$", "", match.group(1), flags=re.IGNORECASE)
            identifier = re.sub(r"v\d+$", "", identifier)
            return f"arxiv:{identifier}"
        match = re.search(r"doi\.org/(10\.[^?#]+)", link, flags=re.IGNORECASE)
        if match:
            return f"doi:{match.group(1).rstrip('/').casefold()}"
        parsed = re.search(r"openreview\.net/(?:forum|pdf)\?id=([^&#]+)", link)
        if parsed:
            return f"openreview:{parsed.group(1)}"
    normalized_title = " ".join(title.casefold().split())
    return "epoch-reference:" + content_digest({"title": normalized_title})


def _epoch_access(value: Any) -> str:
    normalized = " ".join(str(value or "").casefold().split())
    if normalized == "api access":
        return "api_access"
    if normalized == "hosted access (no api)":
        return "hosted_access"
    if normalized == "unreleased":
        return "unreleased"
    if normalized.startswith("open weights"):
        if "unrestricted" in normalized:
            return "open_weights_unrestricted"
        if "non-commercial" in normalized:
            return "open_weights_noncommercial"
        return "open_weights_restricted"
    return "metadata_only"


def _epoch_entry(
    row: Mapping[str, Any], *, observed_at: str
) -> dict[str, Any] | None:
    model_name = str(row.get("Model") or "").strip()
    if not model_name:
        return None
    catalog_id = f"epoch:{model_name}"
    reference = str(row.get("Reference") or "").strip()
    links = _urls(row.get("Link"))
    citation_count = _finite_number(row.get("Citations"))
    if not isinstance(citation_count, int) or citation_count < 0:
        citation_count = None

    paper_identifier = (
        _publication_identifier(reference, links)
        if reference and (links or citation_count is not None)
        else None
    )
    paper_evidence = {
        "links": links,
        "reference": reference,
        "resolution_method": "epoch_associated_publication",
        "source": "epoch_ai_models",
    }
    primary_paper = None
    paper_candidates: list[dict[str, Any]] = []
    paper_status = "unresolved"
    if paper_identifier:
        candidate = {
            "evidence": paper_evidence,
            "identifier": paper_identifier,
            "relationship": "associated_publication",
            "source": "epoch",
        }
        if citation_count is not None:
            primary_paper = {
                "citation_count": citation_count,
                "citation_provider": "epoch",
                "citation_snapshot_at": observed_at,
                "evidence": paper_evidence,
                "identifier": paper_identifier,
            }
            paper_status = "primary_bound"
        else:
            paper_candidates.append(candidate)

    publication_date = str(row.get("Publication date") or "").strip()
    last_modified = str(row.get("Last modified") or "").strip()
    base_model = str(row.get("Base model") or "").strip()
    metadata_fields = {
        "approach": row.get("Approach"),
        "authors": row.get("Authors"),
        "confidence": row.get("Confidence"),
        "country": row.get("Country (of organization)"),
        "domain": row.get("Domain"),
        "foundation_model": row.get("Foundation model"),
        "frontier_model": row.get("Frontier model"),
        "model_accessibility": row.get("Model accessibility"),
        "notability_criteria": row.get("Notability criteria"),
        "open_model_weights": row.get("Open model weights?"),
        "organization": row.get("Organization"),
        "parameters": _finite_number(row.get("Parameters")),
        "reference": reference or None,
        "task": row.get("Task"),
        "training_compute_flop": _finite_number(row.get("Training compute (FLOP)")),
        "source_duplicate_conflicts": row.get("_source_duplicate_conflicts"),
        "source_duplicate_row_count": row.get("_source_duplicate_row_count"),
    }
    metadata = {
        key: value
        for key, value in metadata_fields.items()
        if value not in (None, "")
    }
    return {
        "access_kind": _epoch_access(row.get("Model accessibility")),
        "base_models": [base_model] if base_model else [],
        "catalog_id": catalog_id,
        # Epoch defines this as the associated publication, announcement, or
        # release date.  It is intentionally not called a canonical release date.
        "created_at": (
            _instant(publication_date, name="Publication date")
            if publication_date
            else None
        ),
        "entity_kind": "model",
        "metadata": normalize_json(metadata),
        "metrics": {},
        "modified_at": (
            _instant(last_modified, name="Last modified")
            if last_modified
            else None
        ),
        "name": model_name,
        "native_id": model_name,
        "observed_at": observed_at,
        "paper_candidates": paper_candidates,
        "paper_status": paper_status,
        "primary_paper": primary_paper,
        "provider": "epoch",
        "revision": None,
        "source_locators": links,
        "version_id": catalog_id,
    }


class EpochCatalogProvider:
    """Read Epoch AI's curated daily all-models CSV.

    The provider downloads one modest snapshot and pages it locally.  Citation
    counts remain attached to the associated paper object, never to the model's
    structural or model-level metric fields.
    """

    name = "epoch"
    capabilities = ProviderCapabilities(
        entity_kinds=("model",),
        date_fields=("created", "modified"),
        native_importance_metrics=("citations",),
        exact_total=True,
        stable_snapshot=True,
        coverage_note=(
            "Curated Epoch AI model records from one downloaded CSV snapshot; "
            "coverage is broad but intentionally selective and can lag new releases."
        ),
    )

    def __init__(
        self,
        *,
        fetch_csv: CsvFetcher | None = None,
        endpoint: str = EPOCH_MODELS_CSV_URL,
        observed_at: str | datetime | None = None,
    ) -> None:
        self._fetch = fetch_csv or _http_csv
        self.endpoint = endpoint
        self.observed_at = _instant(
            observed_at or datetime.now(timezone.utc), name="observed_at"
        )
        self._entries: tuple[dict[str, Any], ...] | None = None

    def _load(self) -> tuple[dict[str, Any], ...]:
        if self._entries is not None:
            return self._entries
        text = self._fetch(self.endpoint)
        try:
            reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
            if not reader.fieldnames or "Model" not in reader.fieldnames:
                raise ModelCatalogError("Epoch CSV does not contain a Model column")
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in reader:
                name = str(row.get("Model") or "").strip()
                if name:
                    grouped.setdefault(name, []).append(dict(row))
            rows = []
            for name, duplicates in grouped.items():
                ordered = sorted(
                    duplicates,
                    key=lambda row: (
                        -sum(str(value or "").strip() != "" for value in row.values()),
                        content_digest(row),
                    ),
                )
                merged = dict(ordered[0])
                merged["Model"] = name
                conflicts: dict[str, list[str]] = {}
                for field in reader.fieldnames:
                    values = sorted(
                        {
                            str(row.get(field) or "").strip()
                            for row in ordered
                            if str(row.get(field) or "").strip()
                        }
                    )
                    if not str(merged.get(field) or "").strip() and len(values) == 1:
                        merged[field] = values[0]
                    elif len(values) > 1:
                        conflicts[field] = values
                if len(ordered) > 1:
                    merged["_source_duplicate_row_count"] = len(ordered)
                if conflicts:
                    merged["_source_duplicate_conflicts"] = conflicts
                rows.append(merged)
            entries = [
                entry
                for row in rows
                if (entry := _epoch_entry(row, observed_at=self.observed_at))
                is not None
            ]
        except csv.Error as exc:
            raise ModelCatalogError("Epoch model CSV could not be parsed") from exc
        self._entries = tuple(entries)
        return self._entries

    @staticmethod
    def _offset(cursor: str | None) -> int:
        if cursor is None:
            return 0
        prefix, separator, value = cursor.partition(":")
        if prefix != "offset" or not separator or not value.isdigit():
            raise ModelCatalogError("invalid Epoch cursor")
        return int(value)

    def fetch_page(
        self, query: CatalogQuery, cursor: str | None = None
    ) -> CatalogPage:
        if query.provider != self.name:
            raise ValueError("Epoch provider does not match the query")
        if query.library:
            raise ModelCatalogError("Epoch discovery does not use --library")
        entries = list(self._load())
        if query.search:
            needle = query.search.casefold()
            entries = [
                entry
                for entry in entries
                if needle
                in " ".join(
                    (
                        str(entry.get("name") or ""),
                        str(entry.get("metadata", {}).get("organization") or ""),
                        str(entry.get("metadata", {}).get("reference") or ""),
                        str(entry.get("metadata", {}).get("domain") or ""),
                        str(entry.get("metadata", {}).get("task") or ""),
                    )
                ).casefold()
            ]
        ordered_by = "native_id"
        if query.date_window:
            ordered_by = query.date_window.field
            field = f"{query.date_window.field}_at"
            entries.sort(
                key=lambda entry: (
                    str(entry.get(field) or ""),
                    str(entry["version_id"]),
                ),
                reverse=True,
            )
        elif query.importance:
            if query.importance.metric not in {"citations", "primary_paper.citation_count"}:
                raise ModelCatalogError(
                    f"Epoch cannot rank importance metric {query.importance.metric!r}"
                )
            ordered_by = query.importance.metric

            def citation_key(entry: Mapping[str, Any]) -> tuple[bool, float, str]:
                primary = entry.get("primary_paper")
                value = (
                    primary.get("citation_count")
                    if isinstance(primary, Mapping)
                    else None
                )
                return (
                    value is None,
                    -float(value or 0),
                    str(entry["version_id"]),
                )

            entries.sort(key=citation_key)
        else:
            entries.sort(key=lambda entry: str(entry["version_id"]))

        offset = self._offset(cursor)
        selected = tuple(entries[offset : offset + query.page_size])
        next_offset = offset + len(selected)
        next_cursor = (
            f"offset:{next_offset}" if next_offset < len(entries) else None
        )
        return CatalogPage(
            entries=selected,
            next_cursor=next_cursor,
            total_count=len(entries),
            ordered_by=ordered_by,
            ranking_population_count=(
                sum(entry.get("primary_paper") is not None for entry in entries)
                if query.importance
                else None
            ),
        )


def _openrouter_entry(
    item: Mapping[str, Any], *, observed_at: str
) -> dict[str, Any] | None:
    native_id = str(item.get("id") or "").strip()
    if not native_id:
        return None
    created_raw = item.get("created")
    created_at = None
    if (
        isinstance(created_raw, (int, float))
        and not isinstance(created_raw, bool)
        and math.isfinite(float(created_raw))
    ):
        created_at = _instant(
            datetime.fromtimestamp(float(created_raw), tz=timezone.utc),
            name="created",
        )
    context_length = _finite_number(item.get("context_length"))
    metrics = (
        {"context_length": context_length}
        if isinstance(context_length, (int, float)) and context_length >= 0
        else {}
    )
    details = None
    links = item.get("links")
    if isinstance(links, Mapping) and links.get("details"):
        details = urljoin(OPENROUTER_MODELS_ENDPOINT + "/", str(links["details"]))
    architecture = item.get("architecture")
    return {
        "access_kind": "api_offering",
        "catalog_id": f"openrouter:{native_id}",
        # This is OpenRouter's catalog creation timestamp, not necessarily the
        # developer's first public model release date.
        "created_at": created_at,
        "entity_kind": "model_offering",
        "metadata": normalize_json(
            {
                "architecture": architecture if isinstance(architecture, Mapping) else {},
                "canonical_slug": item.get("canonical_slug"),
                "description": item.get("description"),
                "expiration_date": item.get("expiration_date"),
                "knowledge_cutoff": item.get("knowledge_cutoff"),
                "pricing": item.get("pricing") if isinstance(item.get("pricing"), Mapping) else {},
                "supported_parameters": item.get("supported_parameters") or [],
                "top_provider": item.get("top_provider") if isinstance(item.get("top_provider"), Mapping) else {},
            }
        ),
        "metrics": metrics,
        "modified_at": None,
        "name": str(item.get("name") or native_id),
        "native_id": native_id,
        "observed_at": observed_at,
        "paper_candidates": [],
        "paper_status": "unresolved",
        "primary_paper": None,
        "provider": "openrouter",
        "revision": None,
        "source_locators": [details] if details else [],
        "version_id": f"openrouter:{native_id}",
    }


class OpenRouterCatalogProvider:
    """Discover current cross-vendor API model offerings through OpenRouter."""

    name = "openrouter"
    capabilities = ProviderCapabilities(
        entity_kinds=("model_offering",),
        date_fields=("created",),
        native_importance_metrics=("context_length",),
        exact_total=True,
        stable_snapshot=False,
        coverage_note=(
            "Current OpenRouter API offerings visible to the supplied account; "
            "this is not a complete historical catalog or canonical model list."
        ),
    )

    def __init__(
        self,
        *,
        token: str | None = None,
        fetch_page: HttpPageFetcher | None = None,
        endpoint: str = OPENROUTER_MODELS_ENDPOINT,
        observed_at: str | datetime | None = None,
    ) -> None:
        if fetch_page is None:
            resolved_token = token or os.environ.get("OPENROUTER_API_KEY")
            if not resolved_token:
                raise ModelCatalogError(
                    "OpenRouter discovery requires OPENROUTER_API_KEY"
                )
            fetch_page = _bearer_json_fetcher(resolved_token)
        self._fetch = fetch_page
        self.endpoint = endpoint
        self.observed_at = _instant(
            observed_at or datetime.now(timezone.utc), name="observed_at"
        )

    @staticmethod
    def _offset(cursor: str | None) -> int:
        if cursor is None:
            return 0
        prefix, separator, value = cursor.partition(":")
        if prefix != "offset" or not separator or not value.isdigit():
            raise ModelCatalogError("invalid OpenRouter cursor")
        return int(value)

    def _url(self, query: CatalogQuery, offset: int) -> tuple[str, str]:
        if query.library:
            raise ModelCatalogError("OpenRouter discovery does not use --library")
        if query.date_window and query.date_window.field != "created":
            raise ModelCatalogError(
                "OpenRouter discovery has no model-modification timestamp"
            )
        parameters: dict[str, Any] = {
            "limit": min(query.page_size, 1000),
            "offset": offset,
            # OpenRouter otherwise defaults this endpoint to text-output models.
            "output_modalities": "all",
        }
        if query.search:
            parameters["q"] = query.search
        ordered_by = "provider_default"
        if query.date_window:
            parameters["sort"] = "newest"
            ordered_by = "created"
        elif query.importance:
            if query.importance.metric not in {
                "context_length",
                "metrics.context_length",
            }:
                raise ModelCatalogError(
                    "OpenRouter can currently expose exact scores only for "
                    "context_length; ingest first for other selection profiles"
                )
            parameters["sort"] = "context-high-to-low"
            ordered_by = query.importance.metric
        return f"{self.endpoint}?{urlencode(parameters)}", ordered_by

    def fetch_page(
        self, query: CatalogQuery, cursor: str | None = None
    ) -> CatalogPage:
        if query.provider != self.name:
            raise ValueError("OpenRouter provider does not match the query")
        offset = self._offset(cursor)
        url, ordered_by = self._url(query, offset)
        raw, _headers = self._fetch(url)
        if not isinstance(raw, Mapping) or not isinstance(raw.get("data"), list):
            raise ModelCatalogError("OpenRouter page was not a model-list response")
        entries = tuple(
            entry
            for item in raw["data"]
            if isinstance(item, Mapping)
            if (entry := _openrouter_entry(item, observed_at=self.observed_at))
            is not None
        )
        total_raw = raw.get("total_count")
        total = (
            int(total_raw)
            if isinstance(total_raw, int) and not isinstance(total_raw, bool)
            else None
        )
        if total is not None and total < 0:
            raise ModelCatalogError("OpenRouter returned a negative total_count")
        next_offset = offset + len(raw["data"])
        if not raw["data"] and total is not None and offset < total:
            raise ModelCatalogError("OpenRouter returned an empty page before its total")
        next_cursor = (
            f"offset:{next_offset}"
            if total is not None and next_offset < total
            else None
        )
        return CatalogPage(
            entries=entries,
            next_cursor=next_cursor,
            total_count=total,
            ordered_by=ordered_by,
        )


__all__ = [
    "EPOCH_MODELS_CSV_URL",
    "OPENROUTER_MODELS_ENDPOINT",
    "EpochCatalogProvider",
    "OpenRouterCatalogProvider",
]
