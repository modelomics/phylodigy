"""Reproducible popularity-ranked model corpus construction.

Popularity selects candidate artifacts but never contributes structural
characters or distances. The default population is the public Hugging Face
model index filtered to Transformers-compatible artifacts and sorted by likes.
Repository revisions freeze the selected artifacts. Exact arXiv tags can be
enriched with OpenAlex citation records as a separate evidence channel.
"""

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .canonical import content_digest, normalize_json
from .lineage import LineageAnalysisConfig, infer_lineage_network
from .model_profile import extract_architectural_genome
from .schema import ArchitecturalGenome


POPULARITY_CORPUS_VERSION = "1"
HF_MODELS_ENDPOINT = "https://huggingface.co/api/models"
OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_BATCH_ENDPOINT = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
)


class PopularityCorpusError(RuntimeError):
    """Raised when a ranked corpus cannot be fetched, loaded, or traced."""


JsonFetcher = Callable[[str], Any]
JsonPoster = Callable[[str, Mapping[str, Any]], Any]


def _fetch_json(url: str) -> Any:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "phylodigy-popularity-corpus/1",
        },
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == 3:
                break
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else float(attempt + 1)
            time.sleep(max(1.0, min(delay, 10.0)))
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            break
    raise PopularityCorpusError(f"could not fetch {url}: {last_error}") from last_error


def _post_json(url: str, value: Mapping[str, Any]) -> Any:
    request = Request(
        url,
        data=json.dumps(value).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "phylodigy-popularity-corpus/1",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == 3:
                break
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else float(2 ** attempt)
            time.sleep(max(1.0, min(delay, 15.0)))
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            break
    raise PopularityCorpusError(f"could not fetch {url}: {last_error}") from last_error


def _arxiv_ids(tags: Iterable[Any]) -> tuple[str, ...]:
    result = {
        str(tag).partition(":")[2]
        for tag in tags
        if isinstance(tag, str)
        and tag.startswith("arxiv:")
        and tag.partition(":")[2]
    }
    return tuple(sorted(result))


def _license(tags: Iterable[Any]) -> str | None:
    values = sorted(
        str(tag).partition(":")[2]
        for tag in tags
        if isinstance(tag, str) and tag.startswith("license:")
    )
    return values[0] if values else None


def _openalex_arxiv_evidence(arxiv_id: str, fetch_json: JsonFetcher) -> dict[str, Any] | None:
    query = urlencode(
        {
            "search": arxiv_id,
            "per-page": 10,
            "select": "id,display_name,cited_by_count,doi,locations",
        }
    )
    raw = fetch_json(f"{OPENALEX_WORKS_ENDPOINT}?{query}")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("results"), list):
        return None
    expected = arxiv_id.casefold()
    matches: list[Mapping[str, Any]] = []
    for work in raw["results"]:
        if not isinstance(work, Mapping):
            continue
        urls = []
        for location in work.get("locations") or ():
            if isinstance(location, Mapping):
                urls.append(str(location.get("landing_page_url") or "").casefold())
        doi = str(work.get("doi") or "").casefold()
        if any(
            f"arxiv.org/abs/{expected}" in url
            or f"doi.org/10.48550/arxiv.{expected}" in url
            for url in urls
        ) or f"doi.org/10.48550/arxiv.{expected}" in doi:
            matches.append(work)
    if not matches:
        return None
    selected = max(matches, key=lambda item: int(item.get("cited_by_count") or 0))
    return {
        "arxiv_id": arxiv_id,
        "cited_by_count": int(selected.get("cited_by_count") or 0),
        "openalex_id": str(selected.get("id") or ""),
        "title": str(selected.get("display_name") or ""),
    }


def fetch_popularity_manifest(
    *,
    limit: int = 100,
    include_citations: bool = True,
    fetch_json: JsonFetcher | None = None,
) -> dict[str, Any]:
    """Fetch and freeze a top-liked Transformers model manifest.

    The rank is defined only by Hugging Face likes (descending, then repository
    ID for deterministic ties). OpenAlex citation counts are documentary
    enrichment and cannot reorder the corpus or enter graph distance.
    """

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 2 or limit > 1000:
        raise ValueError("limit must be an integer from 2 through 1000")
    fetch = fetch_json or _fetch_json
    parameters = {
        "direction": -1,
        "filter": "transformers",
        "full": "true",
        "limit": limit,
        "sort": "likes",
    }
    source_url = f"{HF_MODELS_ENDPOINT}?{urlencode(parameters)}"
    raw = fetch(source_url)
    if not isinstance(raw, list):
        raise PopularityCorpusError("Hugging Face model listing was not a JSON array")
    candidates = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        hub_id = str(item.get("id") or "").strip()
        revision = str(item.get("sha") or "").strip()
        if not hub_id or not revision:
            continue
        tags = tuple(item.get("tags") or ())
        candidates.append(
            {
                "arxiv_ids": list(_arxiv_ids(tags)),
                "downloads_at_snapshot": int(item.get("downloads") or 0),
                "gated": item.get("gated", False),
                "hub_id": hub_id,
                "hub_revision": revision,
                "library_name": str(item.get("library_name") or ""),
                "license": _license(tags),
                "likes_at_snapshot": int(item.get("likes") or 0),
                "pipeline_tag": str(item.get("pipeline_tag") or ""),
                "requires_custom_code": "custom_code" in tags,
            }
        )
    candidates.sort(key=lambda item: (-item["likes_at_snapshot"], item["hub_id"]))
    candidates = candidates[:limit]
    if len(candidates) < 2:
        raise PopularityCorpusError("fewer than two revision-pinned models were returned")

    citation_cache: dict[str, dict[str, Any] | None] = {}
    entries = []
    for rank, candidate in enumerate(candidates, 1):
        citation_evidence = []
        citation_resolution_errors = []
        if include_citations:
            for arxiv_id in candidate["arxiv_ids"]:
                if arxiv_id not in citation_cache:
                    try:
                        citation_cache[arxiv_id] = _openalex_arxiv_evidence(
                            arxiv_id, fetch
                        )
                    except PopularityCorpusError as exc:
                        citation_cache[arxiv_id] = None
                        citation_resolution_errors.append(
                            {"arxiv_id": arxiv_id, "error": str(exc)}
                        )
                evidence = citation_cache[arxiv_id]
                if evidence is not None:
                    citation_evidence.append(evidence)
        entry = dict(candidate)
        entry.update(
            {
                "artifact_id": (
                    f"hf:{candidate['hub_id']}@{candidate['hub_revision']}"
                ),
                "citation_evidence": citation_evidence,
                "citation_resolution_errors": citation_resolution_errors,
                "maximum_citation_count": max(
                    (item["cited_by_count"] for item in citation_evidence),
                    default=None,
                ),
                "rank": rank,
            }
        )
        entries.append(entry)

    result: dict[str, Any] = {
        "artifact_type": "phylodigy.popularity_corpus",
        "corpus_version": POPULARITY_CORPUS_VERSION,
        "entries": entries,
        "selection": {
            "citation_role": "external_evidence_only",
            "population": "public_huggingface_models_tagged_transformers",
            "primary_metric": "huggingface_likes_descending",
            "requested_limit": limit,
            "selection_source": source_url,
            "tie_breaker": "hub_id_ascending",
        },
        "snapshot_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    result["digest"] = content_digest(result)
    return result


def validate_popularity_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a fetched manifest and its content digest."""

    normalized = normalize_json(dict(raw))
    if normalized.get("artifact_type") != "phylodigy.popularity_corpus":
        raise ValueError("not a phylodigy popularity corpus")
    if normalized.get("corpus_version") != POPULARITY_CORPUS_VERSION:
        raise ValueError("unsupported popularity corpus version")
    entries = normalized.get("entries")
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("popularity corpus requires at least two entries")
    ranks = [item.get("rank") for item in entries if isinstance(item, dict)]
    if ranks != list(range(1, len(entries) + 1)):
        raise ValueError("popularity corpus ranks must be contiguous and ordered")
    supplied = normalized.pop("digest", None)
    expected = content_digest(normalized)
    if supplied != expected:
        raise ValueError("popularity corpus digest does not match its content")
    normalized["digest"] = supplied
    return normalized


def fetch_citation_evidence(
    manifest: Mapping[str, Any],
    *,
    post_json: JsonPoster | None = None,
) -> dict[str, Any]:
    """Resolve every manifest arXiv ID in one Semantic Scholar batch request."""

    frozen = validate_popularity_manifest(manifest)
    arxiv_ids = sorted(
        {
            arxiv_id
            for entry in frozen["entries"]
            for arxiv_id in entry.get("arxiv_ids", ())
        }
    )
    post = post_json or _post_json
    endpoint = (
        f"{SEMANTIC_SCHOLAR_BATCH_ENDPOINT}?"
        + urlencode(
            {
                "fields": (
                    "title,year,citationCount,externalIds,url"
                )
            }
        )
    )
    raw = post(endpoint, {"ids": [f"ARXIV:{item}" for item in arxiv_ids]})
    if not isinstance(raw, list) or len(raw) != len(arxiv_ids):
        raise PopularityCorpusError(
            "Semantic Scholar batch response did not match the requested IDs"
        )
    papers = []
    for arxiv_id, item in zip(arxiv_ids, raw):
        if item is None:
            papers.append({"arxiv_id": arxiv_id, "resolved": False})
            continue
        if not isinstance(item, Mapping):
            raise PopularityCorpusError("invalid Semantic Scholar paper record")
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "citation_count": int(item.get("citationCount") or 0),
                "external_ids": normalize_json(item.get("externalIds") or {}),
                "paper_id": str(item.get("paperId") or ""),
                "resolved": True,
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "year": item.get("year"),
            }
        )
    result: dict[str, Any] = {
        "artifact_type": "phylodigy.citation_snapshot",
        "manifest_digest": frozen["digest"],
        "papers": papers,
        "provider": "semantic_scholar_academic_graph",
        "snapshot_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_endpoint": endpoint,
    }
    result["digest"] = content_digest(result)
    return result


def _require_hf_runtime() -> tuple[Any, Any]:
    try:
        return importlib.import_module("torch"), importlib.import_module("transformers")
    except ImportError as exc:
        raise PopularityCorpusError(
            "popularity-tree extraction requires the 'phylodigy[corpus]' extra"
        ) from exc


def _configure_for_export(config: Any) -> None:
    for name, value in (
        ("return_dict", False),
        ("use_cache", False),
        ("output_attentions", False),
        ("output_hidden_states", False),
    ):
        if hasattr(config, name):
            setattr(config, name, value)


def _model_from_entry(
    entry: Mapping[str, Any], *, max_hidden_layers: int
) -> tuple[Any, Any, str]:
    if entry.get("gated") is not False:
        raise PopularityCorpusError("artifact is gated and no authorization was supplied")
    if entry.get("requires_custom_code"):
        raise PopularityCorpusError("artifact requires untrusted custom code")
    torch, transformers = _require_hf_runtime()
    config = transformers.AutoConfig.from_pretrained(
        entry["hub_id"],
        revision=entry["hub_revision"],
        trust_remote_code=False,
    )
    _configure_for_export(config)
    layer_counts = [
        int(value)
        for name in (
            "num_hidden_layers",
            "n_layer",
            "num_layers",
            "encoder_layers",
            "decoder_layers",
        )
        if (value := getattr(config, name, None)) is not None
        and not isinstance(value, bool)
        and isinstance(value, int)
    ]
    if layer_counts and max(layer_counts) > max_hidden_layers:
        raise PopularityCorpusError(
            f"declared layer count {max(layer_counts)} exceeds limit "
            f"{max_hidden_layers}"
        )
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if not architectures:
        raise PopularityCorpusError("config declares no model architecture")
    architecture = str(architectures[0])
    model_class = getattr(transformers, architecture, None)
    if model_class is None:
        raise PopularityCorpusError(
            f"architecture {architecture!r} requires unavailable or custom code"
        )
    with torch.device("meta"):
        model = model_class(config)
    model.eval()
    dummy_inputs = getattr(model, "dummy_inputs", None)
    if not isinstance(dummy_inputs, Mapping) or not dummy_inputs:
        raise PopularityCorpusError("model supplies no framework dummy inputs")
    return model, dict(dummy_inputs), architecture


def _extract_entry(
    entry: Mapping[str, Any],
    *,
    manifest_digest: str,
    max_graph_nodes: int,
    max_hidden_layers: int,
    radii: Sequence[int],
) -> tuple[ArchitecturalGenome, dict[str, Any]]:
    model, dummy_inputs, architecture = _model_from_entry(
        entry, max_hidden_layers=max_hidden_layers
    )
    extracted = extract_architectural_genome(
        model,
        artifact_id=entry["artifact_id"],
        name=entry["hub_id"],
        example_kwargs=dummy_inputs,
        propagate_shapes=False,
        radii=radii,
    )
    node_count = len(extracted.graph_profile.graph.nodes)
    if node_count > max_graph_nodes:
        raise PopularityCorpusError(
            f"observed graph has {node_count} nodes; limit is {max_graph_nodes}"
        )
    metadata = dict(extracted.metadata)
    metadata["popularity_corpus"] = {
        "architecture": architecture,
        "citation_evidence": entry["citation_evidence"],
        "hub_revision": entry["hub_revision"],
        "likes_at_snapshot": entry["likes_at_snapshot"],
        "manifest_digest": manifest_digest,
        "rank": entry["rank"],
    }
    genome = ArchitecturalGenome(
        artifact_id=extracted.artifact_id,
        graph_profile=extracted.graph_profile,
        artifact_kind=extracted.artifact_kind,
        name=extracted.name,
        extractor_name=extracted.extractor_name,
        extractor_version=extracted.extractor_version,
        metadata=metadata,
    )
    return genome, {
        "architecture": architecture,
        "artifact_id": genome.artifact_id,
        "graph_digest": genome.structural_digest,
        "node_count": node_count,
        "profile_digest": genome.digest,
        "rank": entry["rank"],
    }


def _isolated_extract_worker(
    entry: Mapping[str, Any],
    manifest_digest: str,
    max_graph_nodes: int,
    max_hidden_layers: int,
    radii: tuple[int, ...],
    profile_path: str,
    queue: Any,
) -> None:
    """Extract one model in a killable child process."""

    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                genome, success = _extract_entry(
                    entry,
                    manifest_digest=manifest_digest,
                    max_graph_nodes=max_graph_nodes,
                    max_hidden_layers=max_hidden_layers,
                    radii=radii,
                )
                from .io import write_profile

                write_profile(genome, profile_path)
        queue.put({"success": success})
    except Exception as exc:
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


def extract_popularity_tree(
    manifest: Mapping[str, Any],
    *,
    max_models: int = 100,
    radii: Sequence[int] = (0,),
    max_graph_nodes: int = 12000,
    max_hidden_layers: int = 36,
    per_model_timeout: float = 45.0,
    profiles_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Trace ranked artifacts and infer a tree from every successful graph.

    Gated, custom-code, incompatible, and resource-exceeding artifacts remain
    explicit failures. They are never replaced with configuration-derived or
    synthetic characters.
    """

    frozen = validate_popularity_manifest(manifest)
    if isinstance(max_models, bool) or not isinstance(max_models, int) or max_models < 2:
        raise ValueError("max_models must be an integer of at least two")
    if (
        isinstance(max_graph_nodes, bool)
        or not isinstance(max_graph_nodes, int)
        or max_graph_nodes < 1
    ):
        raise ValueError("max_graph_nodes must be a positive integer")
    if (
        isinstance(max_hidden_layers, bool)
        or not isinstance(max_hidden_layers, int)
        or max_hidden_layers < 1
    ):
        raise ValueError("max_hidden_layers must be a positive integer")
    if (
        isinstance(per_model_timeout, bool)
        or not isinstance(per_model_timeout, (int, float))
        or per_model_timeout <= 0
    ):
        raise ValueError("per_model_timeout must be a positive number")
    selected_radii = tuple(radii)
    if profiles_directory is None:
        import tempfile

        destination = Path(tempfile.mkdtemp(prefix="phylodigy-popularity-"))
    else:
        destination = Path(profiles_directory)
    destination.mkdir(parents=True, exist_ok=True)

    genomes: list[ArchitecturalGenome] = []
    failures = []
    successes = []
    # Import once in the parent so POSIX fork workers spend their budget on
    # model construction and tracing rather than repeatedly importing PyTorch.
    _require_hf_runtime()
    available_methods = multiprocessing.get_all_start_methods()
    start_method = "fork" if "fork" in available_methods else "spawn"
    context = multiprocessing.get_context(start_method)
    policy = {
        "max_graph_nodes": max_graph_nodes,
        "max_hidden_layers": max_hidden_layers,
        "per_model_timeout": float(per_model_timeout),
        "radii": list(selected_radii),
    }
    for entry in frozen["entries"][:max_models]:
        path = destination / f"{entry['rank']:03d}.genome.json"
        failure_path = destination / f"{entry['rank']:03d}.failure.json"
        from .io import read_profile

        if path.exists():
            try:
                genome = read_profile(path)
                if genome.artifact_id != entry["artifact_id"]:
                    raise ValueError("cached profile artifact does not match manifest")
                corpus_metadata = genome.metadata.get("popularity_corpus", {})
                success = {
                    "architecture": corpus_metadata.get("architecture", "unknown"),
                    "artifact_id": genome.artifact_id,
                    "graph_digest": genome.structural_digest,
                    "node_count": len(genome.graph_profile.graph.nodes),
                    "profile_digest": genome.digest,
                    "profile_path": str(path),
                    "rank": entry["rank"],
                    "resumed": True,
                }
                genomes.append(genome)
                successes.append(success)
                continue
            except Exception:
                pass

        if failure_path.exists():
            try:
                from .io import read_json_object

                cached_failure = read_json_object(failure_path)
                if (
                    cached_failure.get("artifact_id") == entry["artifact_id"]
                    and cached_failure.get("manifest_digest") == frozen["digest"]
                    and cached_failure.get("policy") == policy
                ):
                    failure = dict(cached_failure["failure"])
                    failure["resumed"] = True
                    failures.append(failure)
                    continue
            except Exception:
                pass

        queue = context.Queue()
        process = context.Process(
            target=_isolated_extract_worker,
            args=(
                entry,
                frozen["digest"],
                max_graph_nodes,
                max_hidden_layers,
                selected_radii,
                str(path),
                queue,
            ),
        )
        process.start()
        process.join(float(per_model_timeout))
        if process.is_alive():
            process.terminate()
            process.join(5)
            failure = {
                "artifact_id": entry["artifact_id"],
                "error": (
                    "TimeoutError: extraction exceeded "
                    f"{float(per_model_timeout):g} seconds"
                ),
                "rank": entry["rank"],
                "resumed": False,
            }
            failures.append(failure)
            from .io import write_json_data

            write_json_data(
                {
                    "artifact_id": entry["artifact_id"],
                    "failure": failure,
                    "manifest_digest": frozen["digest"],
                    "policy": policy,
                },
                failure_path,
            )
            queue.close()
            continue
        try:
            message = queue.get(timeout=1)
        except Exception:
            message = {
                "error": f"worker exited with status {process.exitcode} without a report"
            }
        finally:
            queue.close()
        if "error" in message:
            failure = {
                "artifact_id": entry["artifact_id"],
                "error": message["error"],
                "rank": entry["rank"],
                "resumed": False,
            }
            failures.append(failure)
            from .io import write_json_data

            write_json_data(
                {
                    "artifact_id": entry["artifact_id"],
                    "failure": failure,
                    "manifest_digest": frozen["digest"],
                    "policy": policy,
                },
                failure_path,
            )
            continue
        genome = read_profile(path)
        success = dict(message["success"])
        success["profile_path"] = str(path)
        success["resumed"] = False
        genomes.append(genome)
        successes.append(success)

    lineage = None
    status = "insufficient_executable_graphs"
    if len(genomes) >= 2:
        external_evidence = (
            {
                "kind": "popularity_corpus_selection",
                "manifest_digest": frozen["digest"],
                "selection": frozen["selection"],
                "snapshot_at": frozen["snapshot_at"],
            },
        )
        lineage = infer_lineage_network(
            genomes,
            config=LineageAnalysisConfig(radii=selected_radii),
            evidence=external_evidence,
        )
        status = "complete"
    result: dict[str, Any] = {
        "artifact_type": "phylodigy.popularity_tree_run",
        "extraction": {
            "attempted": min(max_models, len(frozen["entries"])),
            "failed": failures,
            "failure_count": len(failures),
            "max_graph_nodes": max_graph_nodes,
            "max_hidden_layers": max_hidden_layers,
            "per_model_timeout": float(per_model_timeout),
            "worker_start_method": start_method,
            "succeeded": successes,
            "success_count": len(successes),
        },
        "lineage": lineage,
        "manifest_digest": frozen["digest"],
        "radii": list(selected_radii),
        "status": status,
    }
    result["digest"] = content_digest(result)
    return result


__all__ = [
    "HF_MODELS_ENDPOINT",
    "OPENALEX_WORKS_ENDPOINT",
    "SEMANTIC_SCHOLAR_BATCH_ENDPOINT",
    "POPULARITY_CORPUS_VERSION",
    "PopularityCorpusError",
    "extract_popularity_tree",
    "fetch_citation_evidence",
    "fetch_popularity_manifest",
    "validate_popularity_manifest",
]
