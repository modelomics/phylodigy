"""Provider-neutral, resumable discovery of model and evidence records.

Catalog selection is deliberately separate from phylogenetic inference.  An
importance metric can bound or sample a catalog, but it can never become a
structural character or a branch weight.

Adapters can enumerate curated models, API offerings, public artifacts,
repositories, or paper candidates without forcing those objects into one
identity type. Provider coverage and query completion remain explicit.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from .canonical import content_digest, normalize_json


CATALOG_VERSION = "1"
HF_MODELS_ENDPOINT = "https://huggingface.co/api/models"
GITHUB_SEARCH_ENDPOINT = "https://api.github.com/search/repositories"
OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
MODEL_CATALOG_ARTIFACT_TYPE = "phylodigy.model_catalog_snapshot"
MODEL_SELECTION_ARTIFACT_TYPE = "phylodigy.model_catalog_selection"
PAPER_GROUNDED_CATALOG_ARTIFACT_TYPE = "phylodigy.paper_grounded_model_catalog"


class ModelCatalogError(RuntimeError):
    """Raised when a provider page or catalog operation is invalid."""


@dataclass(frozen=True)
class ProviderCapabilities:
    """Machine-readable bounds on one discovery adapter's coverage."""

    entity_kinds: tuple[str, ...]
    date_fields: tuple[str, ...] = ()
    native_importance_metrics: tuple[str, ...] = ()
    exact_total: bool = False
    stable_snapshot: bool = False
    result_cap: int | None = None
    coverage_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "coverage_note": self.coverage_note,
            "date_fields": list(self.date_fields),
            "entity_kinds": list(self.entity_kinds),
            "exact_total": self.exact_total,
            "native_importance_metrics": list(self.native_importance_metrics),
            "result_cap": self.result_cap,
            "stable_snapshot": self.stable_snapshot,
        }


def _instant(value: str | datetime | date | None, *, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min, tzinfo=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{name} cannot be empty")
        try:
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                parsed = datetime.combine(date.fromisoformat(text), time.min)
            else:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an ISO 8601 date or timestamp"
            ) from exc
    else:
        raise TypeError(f"{name} must be a date, datetime, string, or None")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def _instant_value(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class DateWindow:
    """Inclusive start and exclusive end over one provider timestamp field."""

    field: str = "modified"
    since: str | datetime | date | None = None
    until: str | datetime | date | None = None

    def __post_init__(self) -> None:
        if self.field not in {"created", "modified"}:
            raise ValueError("date-window field must be 'created' or 'modified'")
        since = _instant(self.since, name="since")
        until = _instant(self.until, name="until")
        if since is None and until is None:
            raise ValueError("a date window requires since, until, or both")
        if since is not None and until is not None:
            if _instant_value(since) >= _instant_value(until):
                raise ValueError("since must be earlier than until")
        object.__setattr__(self, "since", since)
        object.__setattr__(self, "until", until)

    def includes(self, value: str | None) -> bool:
        if value is None:
            return False
        observed = _instant_value(_instant(value, name=self.field))
        lower = _instant_value(self.since)
        upper = _instant_value(self.until)
        return bool(
            (lower is None or observed >= lower)
            and (upper is None or observed < upper)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "since": self.since, "until": self.until}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DateWindow":
        return cls(field=str(raw.get("field", "modified")), since=raw.get("since"), until=raw.get("until"))


@dataclass(frozen=True)
class ImportanceFilter:
    """A bounded selection over a public, nonstructural metric.

    ``top_p`` is a fraction of the eligible records that contain the requested
    metric, not nucleus sampling: ``top_p=0.01`` selects the highest-scoring one
    percent of that scorable population. Exact streaming top-p selection
    requires a provider-reported scorable population size; otherwise the
    eligible date window must be exhausted first.
    """

    metric: str
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        metric = str(self.metric).strip()
        if not metric:
            raise ValueError("importance metric must be non-empty")
        if (self.top_k is None) == (self.top_p is None):
            raise ValueError("set exactly one of top_k or top_p")
        if self.top_k is not None:
            if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
                raise ValueError("top_k must be a positive integer")
        if self.top_p is not None:
            if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
                raise TypeError("top_p must be numeric")
            value = float(self.top_p)
            if not math.isfinite(value) or value <= 0 or value > 1:
                raise ValueError("top_p must be greater than zero and at most one")
            object.__setattr__(self, "top_p", value)
        object.__setattr__(self, "metric", metric)

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "top_k": self.top_k, "top_p": self.top_p}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ImportanceFilter":
        return cls(metric=str(raw["metric"]), top_k=raw.get("top_k"), top_p=raw.get("top_p"))


@dataclass(frozen=True)
class CatalogQuery:
    """One reproducible provider query."""

    provider: str = "huggingface"
    importance: ImportanceFilter | None = None
    date_window: DateWindow | None = None
    page_size: int = 100
    library: str | None = None
    search: str | None = None

    def __post_init__(self) -> None:
        provider = str(self.provider).strip().casefold()
        if not provider:
            raise ValueError("provider must be non-empty")
        if isinstance(self.page_size, bool) or not isinstance(self.page_size, int):
            raise TypeError("page_size must be an integer")
        if self.page_size < 1 or self.page_size > 1000:
            raise ValueError("page_size must be from 1 through 1000")
        if self.importance is not None and not isinstance(self.importance, ImportanceFilter):
            raise TypeError("importance must be an ImportanceFilter or None")
        if self.date_window is not None and not isinstance(self.date_window, DateWindow):
            raise TypeError("date_window must be a DateWindow or None")
        object.__setattr__(self, "provider", provider)
        for name in ("library", "search"):
            value = getattr(self, name)
            if value is not None:
                normalized = str(value).strip()
                object.__setattr__(self, name, normalized or None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date_window": self.date_window.to_dict() if self.date_window else None,
            "importance": self.importance.to_dict() if self.importance else None,
            "library": self.library,
            "page_size": self.page_size,
            "provider": self.provider,
            "search": self.search,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CatalogQuery":
        importance = raw.get("importance")
        date_window = raw.get("date_window")
        return cls(
            provider=str(raw.get("provider", "huggingface")),
            importance=(ImportanceFilter.from_mapping(importance) if isinstance(importance, Mapping) else None),
            date_window=(DateWindow.from_mapping(date_window) if isinstance(date_window, Mapping) else None),
            page_size=int(raw.get("page_size", 100)),
            library=raw.get("library"),
            search=raw.get("search"),
        )


@dataclass(frozen=True)
class CatalogPage:
    entries: tuple[Mapping[str, Any], ...]
    next_cursor: str | None = None
    total_count: int | None = None
    ordered_by: str | None = None
    results_complete: bool = True
    ranking_population_count: int | None = None

    def __post_init__(self) -> None:
        if self.total_count is not None and (
            isinstance(self.total_count, bool)
            or not isinstance(self.total_count, int)
            or self.total_count < 0
        ):
            raise ValueError("total_count must be a non-negative integer or None")
        if not isinstance(self.results_complete, bool):
            raise TypeError("results_complete must be a boolean")
        if self.ranking_population_count is not None and (
            isinstance(self.ranking_population_count, bool)
            or not isinstance(self.ranking_population_count, int)
            or self.ranking_population_count < 0
        ):
            raise ValueError(
                "ranking_population_count must be a non-negative integer or None"
            )


class CatalogProvider(Protocol):
    name: str
    capabilities: ProviderCapabilities

    def fetch_page(self, query: CatalogQuery, cursor: str | None = None) -> CatalogPage:
        ...


HttpPageFetcher = Callable[[str], tuple[Any, Mapping[str, str]]]


def _http_json_page(url: str) -> tuple[Any, Mapping[str, str]]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "phylodigy-model-catalog/1",
        },
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
        headers = {key.casefold(): value for key, value in response.headers.items()}
    return payload, headers


def _link_next(value: str | None) -> str | None:
    if not value:
        return None
    for part in value.split(","):
        match = re.match(r'\s*<([^>]+)>\s*;\s*rel="?next"?\s*$', part.strip())
        if match:
            return match.group(1)
    return None


def _header_integer(headers: Mapping[str, str], *names: str) -> int | None:
    lowered = {str(key).casefold(): value for key, value in headers.items()}
    for name in names:
        value = lowered.get(name.casefold())
        if value is not None and str(value).strip().isdigit():
            return int(str(value).strip())
    return None


def _arxiv_candidates(tags: Iterable[Any]) -> list[dict[str, str]]:
    identifiers = sorted(
        {
            str(tag).partition(":")[2]
            for tag in tags
            if isinstance(tag, str)
            and tag.startswith("arxiv:")
            and tag.partition(":")[2]
        }
    )
    return [
        {
            "identifier": f"arxiv:{identifier}",
            "relationship": "candidate",
            "source": "provider_tag",
        }
        for identifier in identifiers
    ]


def _hf_entry(item: Mapping[str, Any], observed_at: str) -> dict[str, Any] | None:
    model_id = str(item.get("id") or item.get("modelId") or "").strip()
    if not model_id:
        return None
    revision = str(item.get("sha") or "").strip() or None
    catalog_id = f"hf:{model_id}"
    version_id = f"{catalog_id}@{revision}" if revision else catalog_id
    tags = tuple(item.get("tags") or ())
    created = _instant(item.get("createdAt"), name="createdAt") if item.get("createdAt") else None
    modified = _instant(item.get("lastModified"), name="lastModified") if item.get("lastModified") else None
    return {
        "access_kind": (
            "private"
            if bool(item.get("private", False))
            else "gated"
            if item.get("gated") not in (False, None)
            else "public_artifact"
        ),
        "base_models": sorted(str(value) for value in (item.get("baseModels") or ())),
        "catalog_id": catalog_id,
        "created_at": created,
        "disabled": bool(item.get("disabled", False)),
        "entity_kind": "model",
        "gated": item.get("gated", False),
        "library": str(item.get("library_name") or "") or None,
        "metrics": {
            "downloads": int(item.get("downloads") or 0),
            "downloads_all_time": int(item.get("downloadsAllTime") or 0),
            "likes": int(item.get("likes") or 0),
        },
        "modified_at": modified,
        "native_id": model_id,
        "name": model_id,
        "observed_at": observed_at,
        "paper_candidates": _arxiv_candidates(tags),
        "pipeline_tag": str(item.get("pipeline_tag") or "") or None,
        "primary_paper": None,
        "private": bool(item.get("private", False)),
        "provider": "huggingface",
        "requires_custom_code": "custom_code" in tags,
        "revision": revision,
        "version_id": version_id,
    }


class HuggingFaceCatalogProvider:
    """Read public model metadata from the Hub API without downloading weights."""

    name = "huggingface"
    capabilities = ProviderCapabilities(
        entity_kinds=("model",),
        date_fields=("created", "modified"),
        native_importance_metrics=("downloads", "likes"),
        exact_total=False,
        stable_snapshot=False,
        coverage_note=(
            "Public Hugging Face model repositories only; repositories are not "
            "necessarily scientifically distinct models."
        ),
    )

    def __init__(
        self,
        *,
        fetch_page: HttpPageFetcher | None = None,
        endpoint: str = HF_MODELS_ENDPOINT,
        observed_at: str | datetime | None = None,
    ) -> None:
        self._fetch = fetch_page or _http_json_page
        self.endpoint = endpoint
        self.observed_at = _instant(
            observed_at or datetime.now(timezone.utc), name="observed_at"
        )

    def _initial_url(self, query: CatalogQuery) -> tuple[str, str]:
        parameters: dict[str, Any] = {
            "expand": [
                "baseModels",
                "createdAt",
                "disabled",
                "downloads",
                "downloadsAllTime",
                "gated",
                "lastModified",
                "library_name",
                "likes",
                "pipeline_tag",
                "private",
                "sha",
                "tags",
            ],
            "limit": query.page_size,
        }
        if query.library:
            parameters["filter"] = query.library
        if query.search:
            parameters["search"] = query.search
        ordered_by = "native_id"
        if query.date_window:
            ordered_by = query.date_window.field
            parameters["sort"] = (
                "created_at" if query.date_window.field == "created" else "last_modified"
            )
        elif query.importance:
            native_metric = {
                "downloads": "downloads",
                "likes": "likes",
                "metrics.downloads": "downloads",
                "metrics.likes": "likes",
            }.get(query.importance.metric)
            if native_metric is None:
                raise ModelCatalogError(
                    f"Hugging Face cannot push down importance metric "
                    f"{query.importance.metric!r}; discover first and use select_catalog"
                )
            ordered_by = query.importance.metric
            parameters["sort"] = native_metric
        return f"{self.endpoint}?{urlencode(parameters, doseq=True)}", ordered_by

    def _validate_cursor(self, cursor: str) -> None:
        expected = urlparse(self.endpoint)
        actual = urlparse(cursor)
        if (
            actual.scheme != expected.scheme
            or actual.netloc != expected.netloc
            or actual.path != expected.path
        ):
            raise ModelCatalogError("provider cursor does not target the configured endpoint")

    def fetch_page(self, query: CatalogQuery, cursor: str | None = None) -> CatalogPage:
        if query.provider != self.name:
            raise ValueError(f"query provider must be {self.name!r}")
        initial_url, ordered_by = self._initial_url(query)
        url = cursor or initial_url
        if cursor:
            self._validate_cursor(cursor)
        raw, headers = self._fetch(url)
        if not isinstance(raw, list):
            raise ModelCatalogError("Hugging Face model page was not a JSON array")
        entries = tuple(
            entry
            for item in raw
            if isinstance(item, Mapping)
            if (entry := _hf_entry(item, self.observed_at)) is not None
        )
        next_cursor = _link_next(
            next(
                (value for key, value in headers.items() if str(key).casefold() == "link"),
                None,
            )
        )
        total = _header_integer(headers, "x-total-count", "x-total")
        return CatalogPage(
            entries=entries,
            next_cursor=next_cursor,
            total_count=total,
            ordered_by=ordered_by,
        )


def normalize_feed_record(
    raw: Mapping[str, Any], *, provider: str, observed_at: str
) -> dict[str, Any]:
    """Normalize one adapter/feed record, including metadata-only closed models."""

    if not isinstance(raw, Mapping):
        raise TypeError("feed records must be objects")
    native_id = str(raw.get("native_id") or raw.get("id") or "").strip()
    if not native_id:
        raise ValueError("feed record requires native_id")
    catalog_id = str(raw.get("catalog_id") or f"{provider}:{native_id}").strip()
    revision = str(raw.get("revision") or "").strip() or None
    version_id = str(
        raw.get("version_id")
        or (f"{catalog_id}@{revision}" if revision else catalog_id)
    ).strip()
    entity_kind = str(raw.get("entity_kind") or "vendor_model").strip()
    if entity_kind not in {
        "model",
        "model_artifact",
        "model_offering",
        "paper_candidate",
        "repository",
        "vendor_model",
    }:
        raise ValueError(f"unsupported feed entity_kind: {entity_kind!r}")
    metrics: dict[str, int | float] = {}
    for key, value in (raw.get("metrics") or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("feed metrics must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("feed metrics must be finite numbers")
        metrics[str(key)] = value
    paper_candidates = []
    for item in raw.get("paper_candidates") or ():
        if isinstance(item, str):
            paper_candidates.append(
                {
                    "identifier": item,
                    "relationship": "candidate",
                    "source": "feed",
                }
            )
        elif isinstance(item, Mapping) and item.get("identifier"):
            paper_candidates.append(normalize_json(dict(item)))
        else:
            raise ValueError("paper candidates require an identifier")
    created_at = (
        _instant(raw.get("created_at"), name="created_at")
        if raw.get("created_at")
        else None
    )
    modified_at = (
        _instant(raw.get("modified_at"), name="modified_at")
        if raw.get("modified_at")
        else None
    )
    result = {
        "access_kind": str(raw.get("access_kind") or "metadata_only"),
        "catalog_id": catalog_id,
        "created_at": created_at,
        "entity_kind": entity_kind,
        "metrics": metrics,
        "modified_at": modified_at,
        "name": str(raw.get("name") or native_id),
        "native_id": native_id,
        "observed_at": observed_at,
        "paper_candidates": sorted(
            paper_candidates, key=lambda item: (item["identifier"], content_digest(item))
        ),
        "primary_paper": normalize_json(raw.get("primary_paper")),
        "provider": provider,
        "revision": revision,
        "source_locators": normalize_json(raw.get("source_locators") or []),
        "version_id": version_id,
    }
    metadata = raw.get("metadata")
    if metadata:
        result["metadata"] = normalize_json(metadata)
    return result


class RecordFeedCatalogProvider:
    """Adapter for vendor, institutional, or curated model feeds.

    This is the escape hatch that makes closed and paper-only models first-class
    without pretending they have a Hub repository or executable revision.
    """

    def __init__(
        self,
        provider: str,
        records: Sequence[Mapping[str, Any]],
        *,
        observed_at: str | datetime | None = None,
        coverage_note: str = "Provider-authored or curated model feed.",
    ) -> None:
        self.name = str(provider).strip().casefold()
        if not self.name or self.name == "huggingface":
            raise ValueError("feed provider requires a non-Hugging-Face name")
        observation = _instant(
            observed_at or datetime.now(timezone.utc), name="observed_at"
        )
        self._entries = tuple(
            normalize_feed_record(raw, provider=self.name, observed_at=observation)
            for raw in records
        )
        identities = [str(entry["version_id"]) for entry in self._entries]
        if len(identities) != len(set(identities)):
            raise ValueError("feed version_id values must be unique")
        self.capabilities = ProviderCapabilities(
            entity_kinds=tuple(sorted({entry["entity_kind"] for entry in self._entries})),
            date_fields=("created", "modified"),
            native_importance_metrics=tuple(
                sorted(
                    {
                        metric
                        for entry in self._entries
                        for metric in entry.get("metrics", {})
                    }
                )
            ),
            exact_total=True,
            stable_snapshot=True,
            coverage_note=coverage_note,
        )

    def fetch_page(self, query: CatalogQuery, cursor: str | None = None) -> CatalogPage:
        if query.provider != self.name:
            raise ValueError("feed provider does not match the query")
        offset = 0
        if cursor:
            prefix, separator, value = cursor.partition(":")
            if prefix != "offset" or not separator or not value.isdigit():
                raise ModelCatalogError("invalid record-feed cursor")
            offset = int(value)
        ordered_by = "native_id"
        entries = list(self._entries)
        if query.date_window:
            ordered_by = query.date_window.field
            field = f"{query.date_window.field}_at"
            entries.sort(key=lambda item: (str(item.get(field) or ""), item["version_id"]), reverse=True)
        elif query.importance:
            ordered_by = query.importance.metric
            def score(item: Mapping[str, Any]) -> tuple[bool, float, str]:
                value = _metric_value(item, query.importance.metric)
                return (value is None, -(value or 0.0), str(item["version_id"]))
            entries.sort(
                key=score
            )
        else:
            entries.sort(key=lambda item: item["version_id"])
        selected = tuple(entries[offset : offset + query.page_size])
        next_offset = offset + len(selected)
        next_cursor = f"offset:{next_offset}" if next_offset < len(entries) else None
        return CatalogPage(
            entries=selected,
            next_cursor=next_cursor,
            total_count=len(entries),
            ordered_by=ordered_by,
            ranking_population_count=(
                sum(
                    _metric_value(entry, query.importance.metric) is not None
                    for entry in entries
                )
                if query.importance
                else None
            ),
        )


def _github_entry(item: Mapping[str, Any], observed_at: str) -> dict[str, Any] | None:
    native_id = str(item.get("full_name") or "").strip()
    if not native_id:
        return None
    catalog_id = f"gh:{native_id}"
    license_record = item.get("license")
    return {
        "access_kind": "public_source" if not item.get("private") else "private",
        "catalog_id": catalog_id,
        "created_at": (
            _instant(item.get("created_at"), name="created_at")
            if item.get("created_at")
            else None
        ),
        "default_branch": str(item.get("default_branch") or "") or None,
        "description": str(item.get("description") or "") or None,
        "entity_kind": "model_artifact",
        "fork": bool(item.get("fork", False)),
        "license": (
            str(license_record.get("spdx_id") or "") or None
            if isinstance(license_record, Mapping)
            else None
        ),
        "metrics": {
            "forks": int(item.get("forks_count") or 0),
            "github_stars": int(item.get("stargazers_count") or 0),
            "open_issues": int(item.get("open_issues_count") or 0),
            "watchers": int(item.get("subscribers_count") or item.get("watchers_count") or 0),
        },
        "modified_at": (
            _instant(item.get("pushed_at") or item.get("updated_at"), name="modified_at")
            if (item.get("pushed_at") or item.get("updated_at"))
            else None
        ),
        "name": native_id,
        "native_id": native_id,
        "observed_at": observed_at,
        "paper_candidates": [],
        "primary_paper": None,
        "provider": "github",
        # Search results do not identify a commit. A revision resolver can add
        # the default-branch SHA before structural source extraction.
        "revision": None,
        "source_locators": [str(item.get("html_url") or "")],
        "topics": sorted(str(value) for value in (item.get("topics") or ())),
        "version_id": catalog_id,
    }


class GitHubCatalogProvider:
    """Discover public model-code repositories through explicit search strata."""

    name = "github"
    capabilities = ProviderCapabilities(
        entity_kinds=("model_artifact",),
        date_fields=("created", "modified"),
        native_importance_metrics=("forks", "github_stars"),
        exact_total=False,
        stable_snapshot=False,
        result_cap=1000,
        coverage_note=(
            "GitHub repositories matching the declared search query; search is "
            "capped at 1,000 results per partition and repository topics are not a "
            "universal model taxonomy."
        ),
    )

    def __init__(
        self,
        *,
        fetch_page: HttpPageFetcher | None = None,
        endpoint: str = GITHUB_SEARCH_ENDPOINT,
        observed_at: str | datetime | None = None,
    ) -> None:
        self._fetch = fetch_page or _http_json_page
        self.endpoint = endpoint
        self.observed_at = _instant(
            observed_at or datetime.now(timezone.utc), name="observed_at"
        )

    def _initial_url(self, query: CatalogQuery) -> tuple[str, str]:
        if not query.search:
            raise ModelCatalogError(
                "GitHub discovery requires an explicit search population"
            )
        terms = [query.search]
        if query.date_window:
            field = "created" if query.date_window.field == "created" else "pushed"
            since = (query.date_window.since or "").partition("T")[0]
            until = (query.date_window.until or "").partition("T")[0]
            if since and until:
                terms.append(f"{field}:{since}..{until}")
            elif since:
                terms.append(f"{field}:>={since}")
            elif until:
                terms.append(f"{field}:<{until}")
        parameters: dict[str, Any] = {
            "order": "desc",
            "per_page": min(query.page_size, 100),
            "q": " ".join(terms),
        }
        ordered_by = "modified"
        if query.importance:
            native = {
                "forks": "forks",
                "github_stars": "stars",
                "metrics.forks": "forks",
                "metrics.github_stars": "stars",
            }.get(query.importance.metric)
            if native is None:
                raise ModelCatalogError(
                    f"GitHub cannot push down importance metric {query.importance.metric!r}"
                )
            parameters["sort"] = native
            ordered_by = query.importance.metric
        else:
            parameters["sort"] = "updated"
        return f"{self.endpoint}?{urlencode(parameters)}", ordered_by

    def _validate_cursor(self, cursor: str) -> None:
        expected = urlparse(self.endpoint)
        actual = urlparse(cursor)
        if (
            actual.scheme != expected.scheme
            or actual.netloc != expected.netloc
            or actual.path != expected.path
        ):
            raise ModelCatalogError("GitHub cursor does not target the search endpoint")

    def fetch_page(self, query: CatalogQuery, cursor: str | None = None) -> CatalogPage:
        if query.provider != self.name:
            raise ValueError("GitHub provider does not match the query")
        initial, ordered_by = self._initial_url(query)
        url = cursor or initial
        if cursor:
            self._validate_cursor(cursor)
        raw, headers = self._fetch(url)
        if not isinstance(raw, Mapping) or not isinstance(raw.get("items"), list):
            raise ModelCatalogError("GitHub search page was not a repository result")
        entries = tuple(
            entry
            for item in raw["items"]
            if isinstance(item, Mapping)
            if (entry := _github_entry(item, self.observed_at)) is not None
        )
        next_cursor = _link_next(
            next(
                (value for key, value in headers.items() if str(key).casefold() == "link"),
                None,
            )
        )
        total = int(raw.get("total_count") or 0)
        return CatalogPage(
            entries=entries,
            next_cursor=next_cursor,
            # GitHub's count may exceed the 1,000-result search cap, so it is
            # descriptive rather than proof that top-p can stop exactly.
            total_count=total,
            ordered_by=ordered_by,
            results_complete=not bool(raw.get("incomplete_results", False)),
        )


def _openalex_entry(item: Mapping[str, Any], observed_at: str) -> dict[str, Any] | None:
    openalex_id = str(item.get("id") or "").strip()
    native_id = openalex_id.rstrip("/").rsplit("/", 1)[-1]
    if not native_id:
        return None
    ids = item.get("ids") if isinstance(item.get("ids"), Mapping) else {}
    identifiers = []
    for scheme in ("doi", "openalex", "pmid"):
        value = ids.get(scheme)
        if value:
            identifiers.append(str(value))
    primary_location = item.get("primary_location")
    source_url = (
        str(primary_location.get("landing_page_url") or "")
        if isinstance(primary_location, Mapping)
        else ""
    )
    return {
        "access_kind": "paper_metadata",
        "catalog_id": f"openalex:{native_id}",
        "created_at": (
            _instant(item.get("publication_date"), name="publication_date")
            if item.get("publication_date")
            else None
        ),
        "entity_kind": "paper_candidate",
        "identifiers": sorted(set(identifiers)),
        "metrics": {
            "citations": int(item.get("cited_by_count") or 0),
            **(
                {"fwci": float(item["fwci"])}
                if isinstance(item.get("fwci"), (int, float))
                and not isinstance(item.get("fwci"), bool)
                and math.isfinite(float(item["fwci"]))
                else {}
            ),
        },
        "modified_at": (
            _instant(item.get("updated_date"), name="updated_date")
            if item.get("updated_date")
            else None
        ),
        "name": str(item.get("display_name") or native_id),
        "native_id": native_id,
        "observed_at": observed_at,
        "paper_candidates": [],
        "primary_paper": None,
        "provider": "openalex",
        "revision": None,
        "source_locators": sorted(set(filter(None, (openalex_id, source_url)))),
        "version_id": f"openalex:{native_id}",
    }


class OpenAlexPaperProvider:
    """Harvest broad scholarly candidates for model-paper resolution."""

    name = "openalex"
    capabilities = ProviderCapabilities(
        entity_kinds=("paper_candidate",),
        date_fields=("created",),
        native_importance_metrics=("citations", "fwci"),
        exact_total=True,
        stable_snapshot=False,
        coverage_note=(
            "Scholarly works matching the declared OpenAlex search/filter, not "
            "pre-resolved model identities; paperless models require other providers."
        ),
    )

    def __init__(
        self,
        *,
        fetch_page: HttpPageFetcher | None = None,
        endpoint: str = OPENALEX_WORKS_ENDPOINT,
        observed_at: str | datetime | None = None,
    ) -> None:
        self._fetch = fetch_page or _http_json_page
        self.endpoint = endpoint
        self.observed_at = _instant(
            observed_at or datetime.now(timezone.utc), name="observed_at"
        )

    def _url(self, query: CatalogQuery, cursor: str | None) -> tuple[str, str]:
        if query.date_window and query.date_window.field == "modified":
            raise ModelCatalogError(
                "OpenAlex paper discovery currently supports publication dates, "
                "not a model-modification date"
            )
        filters = []
        if query.date_window:
            if query.date_window.since:
                filters.append(
                    "from_publication_date:"
                    + query.date_window.since.partition("T")[0]
                )
            if query.date_window.until:
                # OpenAlex's upper filter is inclusive. Local half-open filtering
                # remains authoritative, so requesting this day only over-fetches.
                filters.append(
                    "to_publication_date:"
                    + query.date_window.until.partition("T")[0]
                )
        parameters: dict[str, Any] = {
            "cursor": cursor or "*",
            "per-page": min(query.page_size, 100),
            "select": (
                "id,display_name,publication_date,updated_date,cited_by_count,"
                "fwci,ids,primary_location"
            ),
        }
        if query.search:
            parameters["search"] = query.search
        if filters:
            parameters["filter"] = ",".join(filters)
        ordered_by = "created"
        if query.importance:
            native = {
                "citations": "cited_by_count",
                "fwci": "fwci",
                "metrics.citations": "cited_by_count",
                "metrics.fwci": "fwci",
            }.get(query.importance.metric)
            if native is None:
                raise ModelCatalogError(
                    f"OpenAlex cannot push down importance metric {query.importance.metric!r}"
                )
            parameters["sort"] = f"{native}:desc"
            ordered_by = query.importance.metric
        else:
            parameters["sort"] = "publication_date:desc"
        return f"{self.endpoint}?{urlencode(parameters)}", ordered_by

    def fetch_page(self, query: CatalogQuery, cursor: str | None = None) -> CatalogPage:
        if query.provider != self.name:
            raise ValueError("OpenAlex provider does not match the query")
        url, ordered_by = self._url(query, cursor)
        raw, _headers = self._fetch(url)
        if not isinstance(raw, Mapping) or not isinstance(raw.get("results"), list):
            raise ModelCatalogError("OpenAlex page was not a works result")
        meta = raw.get("meta") if isinstance(raw.get("meta"), Mapping) else {}
        entries = tuple(
            entry
            for item in raw["results"]
            if isinstance(item, Mapping)
            if (entry := _openalex_entry(item, self.observed_at)) is not None
        )
        next_cursor = str(meta.get("next_cursor") or "") or None
        total = meta.get("count")
        return CatalogPage(
            entries=entries,
            next_cursor=next_cursor,
            total_count=(int(total) if isinstance(total, int) and not isinstance(total, bool) else None),
            ordered_by=ordered_by,
        )


def _entry_in_window(entry: Mapping[str, Any], window: DateWindow | None) -> bool:
    if window is None:
        return True
    return window.includes(entry.get(f"{window.field}_at"))


def _entry_before_window(entry: Mapping[str, Any], window: DateWindow) -> bool:
    if window.since is None:
        return False
    raw = entry.get(f"{window.field}_at")
    if raw is None:
        return False
    return _instant_value(_instant(raw, name=window.field)) < _instant_value(window.since)


def _metric_value(entry: Mapping[str, Any], metric: str) -> float | None:
    aliases = {
        "citations": "primary_paper.citation_count",
        "downloads": "metrics.downloads",
        "github_stars": "metrics.github_stars",
        "likes": "metrics.likes",
    }
    if (
        "." not in metric
        and isinstance(entry.get("metrics"), Mapping)
        and metric in entry["metrics"]
    ):
        path = f"metrics.{metric}"
    else:
        path = aliases.get(metric, metric)
    value: Any = entry
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _rank_entries(
    entries: Iterable[Mapping[str, Any]],
    importance: ImportanceFilter,
    *,
    limit_override: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    scored = []
    missing = []
    for raw in entries:
        entry = normalize_json(dict(raw))
        score = _metric_value(entry, importance.metric)
        if score is None:
            missing.append(str(entry.get("version_id") or entry.get("catalog_id") or ""))
        else:
            scored.append((score, str(entry.get("version_id") or ""), entry))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if limit_override is not None:
        limit = limit_override
    elif importance.top_k is not None:
        limit = importance.top_k
    else:
        limit = math.ceil(float(importance.top_p) * len(scored))
    selected = []
    for rank, (score, _identity, entry) in enumerate(scored[:limit], 1):
        copy = dict(entry)
        copy["selection"] = {
            "importance_metric": importance.metric,
            "importance_score": score,
            "rank": rank,
        }
        selected.append(copy)
    return selected, sorted(missing)


def _entry_observation_digest(entry: Mapping[str, Any]) -> str:
    """Compare repeated provider rows without volatile scan annotations."""

    stable = dict(normalize_json(dict(entry)))
    stable.pop("observed_at", None)
    stable.pop("selection", None)
    return content_digest(stable)


def discover_catalog(
    query: CatalogQuery,
    *,
    provider: CatalogProvider | None = None,
    resume: Mapping[str, Any] | None = None,
    max_pages: int | None = None,
) -> dict[str, Any]:
    """Discover a bounded page range or exhaust an eligible provider stream.

    With no importance filter, pagination continues until the provider stream or
    date window is exhausted. ``max_pages`` creates a resumable shard. Top-k can
    stop as soon as a provider-sorted stream supplies enough eligible entries.
    Top-p stops early only when the provider reports the exact scorable
    population for the requested metric and no local date window changes it.
    """

    if not isinstance(query, CatalogQuery):
        raise TypeError("query must be a CatalogQuery")
    if max_pages is not None and (
        isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1
    ):
        raise ValueError("max_pages must be a positive integer or None")
    selected_provider = provider
    if selected_provider is None:
        factories: dict[str, Callable[[], CatalogProvider]] = {
            "github": GitHubCatalogProvider,
            "huggingface": HuggingFaceCatalogProvider,
            "openalex": OpenAlexPaperProvider,
        }
        factory = factories.get(query.provider)
        selected_provider = factory() if factory else None
        if selected_provider is None and query.provider in {"epoch", "openrouter"}:
            from .curated_catalog import EpochCatalogProvider, OpenRouterCatalogProvider

            curated_factories: dict[str, Callable[[], CatalogProvider]] = {
                "epoch": EpochCatalogProvider,
                "openrouter": OpenRouterCatalogProvider,
            }
            selected_provider = curated_factories[query.provider]()
    if selected_provider is None:
        raise ModelCatalogError(f"no provider adapter is configured for {query.provider!r}")
    if selected_provider.name != query.provider:
        raise ValueError("provider adapter does not match the query")
    capabilities = selected_provider.capabilities

    entries: list[dict[str, Any]] = []
    cursor = None
    pages_before = 0
    records_seen = 0
    provider_total = None
    started_at = _instant(datetime.now(timezone.utc), name="started_at")
    if resume is not None:
        frozen = validate_catalog_snapshot(resume)
        if frozen["provider"] != query.provider or frozen["query"] != query.to_dict():
            raise ValueError("resume snapshot provider and query must match")
        continuation = frozen["scan"].get("next_cursor")
        if frozen["scan"].get("query_complete"):
            raise ValueError("cannot resume a query-complete snapshot")
        if not continuation:
            raise ValueError("resume snapshot has no continuation cursor")
        entries = [dict(item) for item in frozen["entries"]]
        cursor = str(continuation)
        pages_before = int(frozen["scan"]["pages_fetched"])
        records_seen = int(frozen["scan"]["records_seen"])
        started_at = frozen["scan"]["started_at"]
        provider_total = frozen["scan"].get("provider_reported_total")

    seen = {
        str(item["version_id"]): _entry_observation_digest(item) for item in entries
    }
    pages_this_run = 0
    stopped_reason = "provider_exhausted"
    complete = False
    population_exhausted = False
    source_exhausted = False
    selection_target = None
    cursors_seen: set[str] = set()
    duplicate_version_conflicts = (
        int(frozen["scan"].get("duplicate_version_conflicts", 0))
        if resume is not None
        else 0
    )
    ranking_population_count = None
    provider_order_verified: bool | None = (
        True if query.importance is not None and query.date_window is None else None
    )
    last_ordered_score: float | None = None
    ordered_missing_seen = False
    if provider_order_verified:
        provider_order_verified = bool(
            frozen["scan"].get("provider_order_verified", True)
            if resume is not None
            else True
        )
        existing_scores = [
            _metric_value(entry, query.importance.metric) for entry in entries
        ]
        finite_existing = [score for score in existing_scores if score is not None]
        last_ordered_score = min(finite_existing) if finite_existing else None
        ordered_missing_seen = any(score is None for score in existing_scores)
    if cursor:
        cursors_seen.add(cursor)
    while True:
        if max_pages is not None and pages_this_run >= max_pages:
            stopped_reason = "page_budget"
            break
        page = selected_provider.fetch_page(query, cursor)
        pages_this_run += 1
        records_seen += len(page.entries)
        if page.total_count is not None:
            provider_total = page.total_count
        if page.ranking_population_count is not None:
            ranking_population_count = page.ranking_population_count
        if (
            provider_order_verified is not None
            and page.ordered_by != query.importance.metric
        ):
            provider_order_verified = False
        older_than_window = False
        for raw in page.entries:
            entry = normalize_json(dict(raw))
            if query.date_window and _entry_before_window(entry, query.date_window):
                older_than_window = True
            if not _entry_in_window(entry, query.date_window):
                continue
            identity = str(entry["version_id"])
            digest = _entry_observation_digest(entry)
            if identity in seen:
                if seen[identity] != digest:
                    duplicate_version_conflicts += 1
                # One discovery snapshot has one membership row per version.
                # The first observation wins; metric history belongs in the
                # persistent registry, not as duplicate catalog taxa.
                continue
            if provider_order_verified:
                score = _metric_value(entry, query.importance.metric)
                if score is None:
                    ordered_missing_seen = True
                elif ordered_missing_seen or (
                    last_ordered_score is not None
                    and score > last_ordered_score
                ):
                    provider_order_verified = False
                else:
                    last_ordered_score = score
            seen[identity] = digest
            entries.append(entry)

        if not page.results_complete:
            # GitHub and similar search APIs can explicitly report that a
            # response is incomplete.  More rows do not make its rank order or
            # population exact, so never turn this page into a completed
            # importance result or coverage checkpoint.
            cursor = page.next_cursor
            stopped_reason = "provider_reported_incomplete_results"
            break

        target = None
        if query.importance and query.date_window is None:
            if (
                page.ordered_by == query.importance.metric
                and provider_order_verified
                and query.importance.top_k is not None
            ):
                target = query.importance.top_k
            elif (
                page.ordered_by == query.importance.metric
                and provider_order_verified
                and ranking_population_count is not None
            ):
                target = math.ceil(
                    query.importance.top_p * ranking_population_count
                )
        scored_entry_count = (
            sum(
                _metric_value(entry, query.importance.metric) is not None
                for entry in entries
            )
            if query.importance
            else 0
        )
        if target is not None and scored_entry_count >= target:
            stopped_reason = "importance_bound_satisfied"
            complete = True
            selection_target = target
            cursor = page.next_cursor
            if cursor is None:
                capped = bool(
                    capabilities.result_cap is not None
                    and provider_total is not None
                    and provider_total > records_seen
                )
                if not capped:
                    population_exhausted = True
                    source_exhausted = True
            break
        if query.date_window and older_than_window and page.ordered_by == query.date_window.field:
            stopped_reason = "date_window_exhausted"
            complete = True
            population_exhausted = True
            cursor = page.next_cursor
            break
        cursor = page.next_cursor
        if not cursor:
            capped = bool(
                capabilities.result_cap is not None
                and provider_total is not None
                and provider_total > records_seen
            )
            complete = not capped
            population_exhausted = not capped
            if capped:
                stopped_reason = "provider_result_cap_requires_partition"
            else:
                source_exhausted = True
            break
        if cursor in cursors_seen:
            raise ModelCatalogError("provider pagination cursor repeated")
        cursors_seen.add(cursor)

    if query.importance and population_exhausted:
        ranking_population_count = sum(
            _metric_value(entry, query.importance.metric) is not None
            for entry in entries
        )
    if query.importance and complete:
        ranked, missing_metric = _rank_entries(
            entries, query.importance, limit_override=selection_target
        )
    else:
        ranked = sorted(entries, key=lambda item: (str(item["catalog_id"]), str(item.get("revision") or "")))
        missing_metric = []
    result: dict[str, Any] = {
        "artifact_type": MODEL_CATALOG_ARTIFACT_TYPE,
        "catalog_version": CATALOG_VERSION,
        "entries": ranked,
        "provider": query.provider,
        "provider_capabilities": capabilities.to_dict(),
        "query": query.to_dict(),
        "scan": {
            "complete": complete,
            "coverage_complete": population_exhausted,
            "eligible_entry_count": len(entries),
            "duplicate_version_conflicts": duplicate_version_conflicts,
            "missing_importance_metric": missing_metric,
            "next_cursor": None if population_exhausted else cursor,
            "pages_fetched": pages_before + pages_this_run,
            "population_exhausted": population_exhausted,
            "provider_reported_total": provider_total,
            "provider_order_verified": provider_order_verified,
            "ranking_population_count": ranking_population_count,
            "query_complete": complete,
            "records_seen": records_seen,
            "selection_applied": bool(query.importance and complete),
            "source_exhausted": source_exhausted,
            "started_at": started_at,
            "stopped_reason": stopped_reason,
            "updated_at": _instant(datetime.now(timezone.utc), name="updated_at"),
        },
        "structural_evidence_boundary": {
            "catalog_fields_can_create_characters": False,
            "importance_can_change_distance": False,
        },
    }
    result["digest"] = content_digest(result)
    return result


def validate_catalog_snapshot(raw: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_json(dict(raw))
    if normalized.get("artifact_type") != MODEL_CATALOG_ARTIFACT_TYPE:
        raise ValueError("not a model catalog snapshot")
    if normalized.get("catalog_version") != CATALOG_VERSION:
        raise ValueError("unsupported model catalog version")
    query = normalized.get("query")
    if not isinstance(query, Mapping):
        raise ValueError("catalog query must be an object")
    CatalogQuery.from_mapping(query)
    entries = normalized.get("entries")
    if not isinstance(entries, list):
        raise ValueError("catalog entries must be an array")
    identities = [
        str(item.get("observation_id") or item.get("version_id") or "")
        for item in entries
        if isinstance(item, Mapping)
    ]
    if len(identities) != len(entries) or any(not item for item in identities):
        raise ValueError("every catalog entry requires a version identity")
    if len(identities) != len(set(identities)):
        raise ValueError("catalog version identities must be unique")
    supplied = normalized.pop("digest", None)
    expected = content_digest(normalized)
    if supplied != expected:
        raise ValueError("model catalog digest does not match its content")
    normalized["digest"] = supplied
    return normalized


def select_catalog(
    catalog: Mapping[str, Any],
    *,
    importance: ImportanceFilter,
    distinct_primary_papers: bool = False,
    paperless_quota: int | None = None,
) -> dict[str, Any]:
    """Select top-k or top-p from an already discovered catalog."""

    if not isinstance(importance, ImportanceFilter):
        raise TypeError("importance must be an ImportanceFilter")
    if catalog.get("artifact_type") == MODEL_CATALOG_ARTIFACT_TYPE:
        frozen = validate_catalog_snapshot(catalog)
    elif catalog.get("artifact_type") == PAPER_GROUNDED_CATALOG_ARTIFACT_TYPE:
        frozen = validate_paper_grounded_catalog(catalog)
    else:
        raise ValueError("unsupported catalog artifact type")
    if paperless_quota is not None and (
        isinstance(paperless_quota, bool)
        or not isinstance(paperless_quota, int)
        or paperless_quota < 0
    ):
        raise ValueError("paperless_quota must be a non-negative integer or None")
    if paperless_quota is not None and not distinct_primary_papers:
        raise ValueError("paperless_quota requires distinct_primary_papers")
    ranked, missing = _rank_entries(
        frozen["entries"], importance, limit_override=len(frozen["entries"])
    )
    duplicate_papers = []
    paperless_excluded = []
    if distinct_primary_papers:
        policy_candidates = []
        seen_papers: set[str] = set()
        paperless_used = 0
        for entry in ranked:
            primary = entry.get("primary_paper")
            paper_id = (
                str(primary.get("identifier"))
                if isinstance(primary, Mapping) and primary.get("identifier")
                else None
            )
            if paper_id:
                if paper_id in seen_papers:
                    duplicate_papers.append(str(entry["version_id"]))
                    continue
                seen_papers.add(paper_id)
            elif entry.get("paper_status") == "paperless_exception":
                if paperless_quota is not None and paperless_used >= paperless_quota:
                    paperless_excluded.append(str(entry["version_id"]))
                    continue
                paperless_used += 1
            else:
                paperless_excluded.append(str(entry["version_id"]))
                continue
            policy_candidates.append(entry)
    else:
        policy_candidates = ranked
    if importance.top_k is not None:
        target = importance.top_k
    else:
        target = math.ceil(importance.top_p * len(policy_candidates))
    selected = policy_candidates[:target]
    for rank, entry in enumerate(selected, 1):
        entry["selection"]["rank"] = rank
    result: dict[str, Any] = {
        "artifact_type": MODEL_SELECTION_ARTIFACT_TYPE,
        "catalog_digest": frozen["digest"],
        "catalog_version": CATALOG_VERSION,
        "entries": selected,
        "selection": {
            **importance.to_dict(),
            "eligible_metric_count": len(frozen["entries"]) - len(missing),
            "missing_metric_count": len(missing),
            "missing_metric_version_ids": missing,
            "distinct_primary_papers": distinct_primary_papers,
            "duplicate_primary_paper_exclusions": sorted(duplicate_papers),
            "paperless_exclusions": sorted(paperless_excluded),
            "paperless_quota": paperless_quota,
            "selected_count": len(selected),
            "tie_breaker": "version_id_ascending",
        },
        "structural_evidence_boundary": {
            "importance_can_create_characters": False,
            "importance_can_change_distance": False,
        },
    }
    result["digest"] = content_digest(result)
    return result


def bind_primary_papers(
    catalog: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = False,
    require_unique_papers: bool = False,
) -> dict[str, Any]:
    """Bind model versions to distinct primary papers or paperless exceptions."""

    frozen = validate_catalog_snapshot(catalog)
    by_version: dict[str, Mapping[str, Any]] = {}
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise TypeError("paper bindings must be objects")
        version_id = str(raw.get("version_id") or "").strip()
        if not version_id:
            raise ValueError("paper binding requires version_id")
        if version_id in by_version:
            raise ValueError(f"duplicate paper binding for {version_id}")
        paper_id = str(raw.get("paper_id") or "").strip() or None
        paperless_reason = str(raw.get("paperless_reason") or "").strip() or None
        if (paper_id is None) == (paperless_reason is None):
            raise ValueError(
                "paper binding requires exactly one of paper_id or paperless_reason"
            )
        citation_count = raw.get("citation_count")
        if citation_count is not None and (
            isinstance(citation_count, bool)
            or not isinstance(citation_count, int)
            or citation_count < 0
        ):
            raise ValueError("citation_count must be a non-negative integer or None")
        by_version[version_id] = normalize_json(dict(raw))

    known_versions = {str(entry["version_id"]) for entry in frozen["entries"]}
    unknown = sorted(set(by_version) - known_versions)
    if unknown:
        raise ValueError(f"paper bindings reference unknown versions: {unknown}")
    entries = []
    unresolved = []
    paper_ids = []
    paperless = []
    for raw in frozen["entries"]:
        entry = dict(raw)
        version_id = str(entry["version_id"])
        binding = by_version.get(version_id)
        if binding is None:
            unresolved.append(version_id)
            entry["primary_paper"] = None
            entry["paper_status"] = "unresolved"
        elif binding.get("paper_id"):
            paper_id = str(binding["paper_id"])
            paper_ids.append(paper_id)
            entry["primary_paper"] = {
                "citation_count": binding.get("citation_count"),
                "citation_provider": binding.get("citation_provider"),
                "citation_snapshot_at": binding.get("citation_snapshot_at"),
                "evidence": binding.get("evidence", {}),
                "identifier": paper_id,
            }
            entry["paper_status"] = "primary_bound"
        else:
            paperless.append(version_id)
            entry["primary_paper"] = None
            entry["paper_status"] = "paperless_exception"
            entry["paperless_reason"] = binding["paperless_reason"]
        entries.append(entry)
    if require_complete and unresolved:
        raise ValueError(f"unresolved primary-paper bindings: {unresolved}")
    duplicates = sorted(
        paper_id for paper_id in set(paper_ids) if paper_ids.count(paper_id) > 1
    )
    if require_unique_papers and duplicates:
        raise ValueError(f"primary papers must be unique: {duplicates}")
    result: dict[str, Any] = {
        "artifact_type": PAPER_GROUNDED_CATALOG_ARTIFACT_TYPE,
        "catalog_digest": frozen["digest"],
        "catalog_version": CATALOG_VERSION,
        "entries": entries,
        "paper_coverage": {
            "bound_primary_paper_count": len(paper_ids),
            "distinct_primary_paper_count": len(set(paper_ids)),
            "model_count": len(entries),
            "paperless_exception_count": len(paperless),
            "paperless_version_ids": sorted(paperless),
            "unresolved_count": len(unresolved),
            "unresolved_version_ids": sorted(unresolved),
        },
        "policy": {
            "require_complete": require_complete,
            "require_unique_papers": require_unique_papers,
        },
        "structural_evidence_boundary": {
            "citations_can_create_characters": False,
            "citations_can_change_distance": False,
        },
    }
    result["digest"] = content_digest(result)
    return result


def validate_paper_grounded_catalog(raw: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_json(dict(raw))
    if normalized.get("artifact_type") != PAPER_GROUNDED_CATALOG_ARTIFACT_TYPE:
        raise ValueError("not a paper-grounded model catalog")
    if normalized.get("catalog_version") != CATALOG_VERSION:
        raise ValueError("unsupported paper-grounded catalog version")
    if not isinstance(normalized.get("entries"), list):
        raise ValueError("paper-grounded catalog entries must be an array")
    supplied = normalized.pop("digest", None)
    expected = content_digest(normalized)
    if supplied != expected:
        raise ValueError("paper-grounded catalog digest does not match its content")
    normalized["digest"] = supplied
    return normalized


__all__ = [
    "CATALOG_VERSION",
    "CatalogPage",
    "CatalogProvider",
    "CatalogQuery",
    "DateWindow",
    "GITHUB_SEARCH_ENDPOINT",
    "GitHubCatalogProvider",
    "HF_MODELS_ENDPOINT",
    "HuggingFaceCatalogProvider",
    "ImportanceFilter",
    "MODEL_CATALOG_ARTIFACT_TYPE",
    "MODEL_SELECTION_ARTIFACT_TYPE",
    "ModelCatalogError",
    "PAPER_GROUNDED_CATALOG_ARTIFACT_TYPE",
    "OPENALEX_WORKS_ENDPOINT",
    "OpenAlexPaperProvider",
    "ProviderCapabilities",
    "RecordFeedCatalogProvider",
    "bind_primary_papers",
    "discover_catalog",
    "normalize_feed_record",
    "select_catalog",
    "validate_catalog_snapshot",
    "validate_paper_grounded_catalog",
]
