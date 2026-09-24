"""Command line interface for graph-derived architectural genomes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date, timedelta
import sys
from typing import Any

from . import __version__
from .code_profile import extract_code_profile
from .computation_graph import build_graph_profile
from .contact_network import infer_contact_network
from .curated_catalog import EpochCatalogProvider, OpenRouterCatalogProvider
from .graph_alignment import align_graphs
from .io import (
    PhylodigyIOError,
    read_document_text,
    read_json_data,
    read_json_object,
    read_profile,
    write_json_data,
    write_profile,
)
from .lineage import infer_lineage_network
from .model_profile import GraphExtractionError, TorchUnavailableError
from .modelome_cli import add_modelome_commands
from .model_catalog import (
    CatalogQuery,
    DateWindow,
    GitHubCatalogProvider,
    HuggingFaceCatalogProvider,
    ImportanceFilter,
    ModelCatalogError,
    OpenAlexPaperProvider,
    RecordFeedCatalogProvider,
    bind_primary_papers,
    discover_catalog,
    select_catalog,
)
from .paper_profile import extract_paper_profile
from .popularity_corpus import (
    PopularityCorpusError,
    extract_popularity_tree,
    fetch_citation_evidence,
    fetch_popularity_manifest,
)
from .schema import ArchitecturalGenome, merge_profiles
from .toy_tree import build_toy_phylodigital_tree, toy_tree_summary


def _output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="write JSON to PATH instead of standard output; use - for stdout",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phylodigy",
        description=(
            "Extract operator/layer graphs as architectural genomes, derive "
            "anonymous graph characters, and infer graph-character phylogenies."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    add_modelome_commands(commands)

    paper = commands.add_parser(
        "prepare-paper",
        aliases=("extract-paper",),
        help="normalize a paper for graph-targeted annotation",
    )
    paper.add_argument("input", metavar="PAPER", help=".txt/.md/.pdf path, or -")
    paper.add_argument("--artifact-id", required=True)
    paper.add_argument("--artifact-kind", default="paper")
    paper.add_argument("--name", default="")
    paper.add_argument("--release-date")
    paper.add_argument("--date-min")
    paper.add_argument("--date-max")
    paper.add_argument("--source-id")
    paper.add_argument("--identifier", action="append", default=[], metavar="KEY=VALUE")
    _output(paper)
    paper.set_defaults(handler=_prepare_paper)

    graph = commands.add_parser(
        "build-genome",
        help="build an architectural genome from generic graph records",
        description=(
            "Read a JSON array of open node records, canonicalize the operator/layer "
            "graph, and discover anonymous structural characters."
        ),
    )
    graph.add_argument("input", metavar="RECORDS_JSON", help="JSON array path, or -")
    graph.add_argument("--artifact-id", required=True)
    graph.add_argument("--frontend", default="generic")
    graph.add_argument("--name", default="")
    graph.add_argument("--release-date")
    graph.add_argument("--date-min")
    graph.add_argument("--date-max")
    _output(graph)
    graph.set_defaults(handler=_build_genome)

    code = commands.add_parser(
        "extract-code",
        help="create a static source manifest for tracing provenance",
        description=(
            "Parse generic source structure without recognizing techniques. "
            "The result is not an architectural genome and cannot enter graph distance."
        ),
    )
    code.add_argument("input", metavar="SOURCE")
    code.add_argument("--artifact-id", required=True)
    code.add_argument("--name", default="")
    code.add_argument("--release-date")
    code.add_argument("--date-min")
    code.add_argument("--date-max")
    _output(code)
    code.set_defaults(handler=_extract_code)

    merge = commands.add_parser(
        "merge", help="merge metadata for identical architectural genomes"
    )
    merge.add_argument("inputs", nargs="+", metavar="GENOME")
    _output(merge)
    merge.set_defaults(handler=_merge)

    validate = commands.add_parser("validate", help="validate architectural genomes")
    validate.add_argument("inputs", nargs="+", metavar="GENOME")
    validate.add_argument("-q", "--quiet", action="store_true")
    validate.add_argument("-o", "--output", metavar="PATH")
    validate.set_defaults(handler=_validate)

    align = commands.add_parser(
        "align",
        help="generate a bounded primitive graph alignment between two genomes",
        description=(
            "Align exact structural node labels and emit a replayable node/edge "
            "alignment candidate. Add/remove direction is mechanical left-to-right, "
            "not an ancestral gain/loss claim."
        ),
    )
    align.add_argument("left", metavar="LEFT_GENOME")
    align.add_argument("right", metavar="RIGHT_GENOME")
    align.add_argument(
        "--config",
        metavar="JSON",
        help="AlignmentParameters JSON (indel costs and matrix-cell budget)",
    )
    _output(align)
    align.set_defaults(handler=_align)

    network = commands.add_parser(
        "infer-network",
        help="retain explicit dated contact evidence with graph diagnostics",
        description=(
            "Admit explicit provenance, awareness, or graph-grounded character "
            "transfer evidence through the date gate. Graph similarity is diagnostic "
            "only and cannot create contact or parent edges."
        ),
    )
    network.add_argument("inputs", nargs="+", metavar="GENOME")
    network.add_argument("--edge-evidence", metavar="JSON")
    _output(network)
    network.set_defaults(handler=_infer_network)

    lineage = commands.add_parser(
        "infer-lineage",
        help="infer a phylogeny from graph-character distances",
    )
    lineage.add_argument("inputs", nargs="+", metavar="GENOME")
    lineage.add_argument(
        "--config",
        metavar="JSON",
        help="LineageAnalysisConfig JSON (radii, weights, tolerance)",
    )
    lineage.add_argument(
        "--evidence",
        metavar="JSON",
        help="external chronology/citation/provenance records retained outside distance",
    )
    _output(lineage)
    lineage.set_defaults(handler=_infer_lineage)

    toy = commands.add_parser(
        "toy-tree",
        help="build a five-artifact demonstration phylodigital tree",
        description=(
            "Instantiate five executable PyTorch models, trace their observed operator "
            "graphs, and infer a graph-character neighbor-joining tree. Names, dates, "
            "and story labels remain nonstructural."
        ),
    )
    toy.add_argument(
        "--summary",
        action="store_true",
        help="write a short human-readable summary instead of canonical JSON",
    )
    _output(toy)
    toy.set_defaults(handler=_toy_tree)

    rank_models = commands.add_parser(
        "rank-models",
        help="freeze a popularity-ranked public model corpus",
        description=(
            "Select public Transformers model artifacts by Hugging Face likes, pin "
            "repository revisions, and optionally retain OpenAlex citations as "
            "external evidence. Popularity never becomes structural evidence."
        ),
    )
    rank_models.add_argument("--limit", type=int, default=100)
    rank_models.add_argument(
        "--without-citations",
        action="store_true",
        help="skip OpenAlex enrichment of exact arXiv tags",
    )
    _output(rank_models)
    rank_models.set_defaults(handler=_rank_models)

    popularity_tree = commands.add_parser(
        "infer-popularity-tree",
        help="trace a frozen ranked corpus and infer its graph-only tree",
    )
    popularity_tree.add_argument("manifest", metavar="MANIFEST_JSON")
    popularity_tree.add_argument("--max-models", type=int, default=100)
    popularity_tree.add_argument("--max-graph-nodes", type=int, default=12000)
    popularity_tree.add_argument("--max-hidden-layers", type=int, default=36)
    popularity_tree.add_argument("--per-model-timeout", type=float, default=45.0)
    popularity_tree.add_argument(
        "--radii", type=int, nargs="+", default=[0], metavar="RADIUS"
    )
    popularity_tree.add_argument(
        "--profiles-dir",
        metavar="DIRECTORY",
        help="write every successfully extracted genome to this directory",
    )
    _output(popularity_tree)
    popularity_tree.set_defaults(handler=_infer_popularity_tree)

    citations = commands.add_parser(
        "enrich-model-citations",
        help="resolve a ranked manifest's arXiv IDs to citation counts",
    )
    citations.add_argument("manifest", metavar="MANIFEST_JSON")
    _output(citations)
    citations.set_defaults(handler=_enrich_model_citations)

    discover = commands.add_parser(
        "discover-models",
        help="discover a bounded or exhaustive provider model catalog",
        description=(
            "Page through a configured model source. With no importance filter, "
            "the command exhausts the source or --max-pages budget. Date windows "
            "are half-open UTC intervals. Provider metrics select catalog members "
            "but never enter phylogenetic distance."
        ),
    )
    discover.add_argument(
        "--provider",
        default="huggingface",
        help=(
            "epoch, openrouter, huggingface, github, openalex, or a custom feed ID"
        ),
    )
    discover.add_argument(
        "--feed",
        metavar="JSON",
        help="JSON array/object of model records for a vendor or curated feed",
    )
    discover.add_argument("--importance", metavar="METRIC")
    discover_limit = discover.add_mutually_exclusive_group()
    discover_limit.add_argument("--top-k", type=int)
    discover_limit.add_argument("--top-p", type=float, metavar="FRACTION")
    discover.add_argument("--date-field", choices=("created", "modified"), default="created")
    discover.add_argument(
        "--day",
        metavar="YYYY-MM-DD",
        help="UTC calendar day shorthand for --since DAY --until NEXT_DAY",
    )
    discover.add_argument("--since", metavar="ISO_DATE_OR_TIMESTAMP")
    discover.add_argument("--until", metavar="ISO_DATE_OR_TIMESTAMP")
    discover.add_argument("--page-size", type=int, default=100)
    discover.add_argument("--max-pages", type=int)
    discover.add_argument("--library", help="provider library/tag filter")
    discover.add_argument("--search", help="provider search population")
    discover.add_argument("--resume", metavar="SNAPSHOT_JSON")
    _output(discover)
    discover.set_defaults(handler=_discover_models)

    select = commands.add_parser(
        "select-models",
        help="select top-k or top-p from a local discovered catalog",
    )
    select.add_argument("catalog", metavar="CATALOG_JSON")
    select.add_argument("--importance", required=True, metavar="METRIC")
    select_limit = select.add_mutually_exclusive_group(required=True)
    select_limit.add_argument("--top-k", type=int)
    select_limit.add_argument("--top-p", type=float, metavar="FRACTION")
    select.add_argument("--distinct-primary-papers", action="store_true")
    select.add_argument("--paperless-quota", type=int)
    _output(select)
    select.set_defaults(handler=_select_models)

    bind = commands.add_parser(
        "bind-model-papers",
        help="bind catalog models to primary papers or paperless exceptions",
    )
    bind.add_argument("catalog", metavar="CATALOG_JSON")
    bind.add_argument("bindings", metavar="BINDINGS_JSON")
    bind.add_argument("--require-complete", action="store_true")
    bind.add_argument(
        "--require-distinct-primary-papers",
        action="store_true",
        help="apply the demonstration-corpus one-model-per-paper policy",
    )
    _output(bind)
    bind.set_defaults(handler=_bind_model_papers)

    return parser


def _identifiers(items: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not key or not value:
            raise ValueError(f"invalid --identifier {item!r}; expected KEY=VALUE")
        if key in result:
            raise ValueError(f"duplicate --identifier key: {key}")
        result[key] = value
    return result


def _one_stdin(inputs: Sequence[str]) -> None:
    if sum(item == "-" for item in inputs) > 1:
        raise ValueError("standard input (-) may appear at most once")


def _prepare_paper(args: argparse.Namespace) -> int:
    document = extract_paper_profile(
        read_document_text(args.input),
        artifact_id=args.artifact_id,
        artifact_kind=args.artifact_kind,
        name=args.name,
        release_date=args.release_date,
        date_min=args.date_min,
        date_max=args.date_max,
        identifiers=_identifiers(args.identifier),
        source_id=args.source_id,
    )
    write_json_data(document.to_dict(), args.output)
    return 0


def _build_genome(args: argparse.Namespace) -> int:
    records = read_json_data(args.input)
    if not isinstance(records, list) or not all(
        isinstance(item, dict) for item in records
    ):
        raise ValueError("graph records JSON must contain an array of objects")
    graph_profile = build_graph_profile(records, frontend=args.frontend)
    genome = ArchitecturalGenome(
        artifact_id=args.artifact_id,
        graph_profile=graph_profile,
        name=args.name,
        release_date=args.release_date,
        date_min=args.date_min,
        date_max=args.date_max,
        extractor_name="phylodigy.cli.build_genome",
        extractor_version="1",
        metadata={"input_record_count": len(records)},
    )
    write_profile(genome, args.output)
    return 0


def _extract_code(args: argparse.Namespace) -> int:
    manifest = extract_code_profile(
        args.input,
        artifact_id=args.artifact_id,
        name=args.name,
        release_date=args.release_date,
        date_min=args.date_min,
        date_max=args.date_max,
    )
    write_json_data(manifest.to_dict(), args.output)
    return 0


def _merge(args: argparse.Namespace) -> int:
    _one_stdin(args.inputs)
    write_profile(merge_profiles(*(read_profile(item) for item in args.inputs)), args.output)
    return 0


def _validate(args: argparse.Namespace) -> int:
    _one_stdin(args.inputs)
    records: list[dict[str, Any]] = []
    for source in args.inputs:
        try:
            profile = read_profile(source)
        except (OSError, PhylodigyIOError, TypeError, ValueError) as exc:
            records.append({"error": str(exc), "input": source, "valid": False})
        else:
            records.append(
                {
                    "artifact_id": profile.artifact_id,
                    "digest": profile.digest,
                    "input": source,
                    "schema_version": profile.schema_version,
                    "valid": True,
                }
            )
    valid = all(item["valid"] for item in records)
    if not args.quiet:
        write_json_data(
            {
                "artifact_type": "phylodigy.validation_report",
                "files": records,
                "valid": valid,
            },
            args.output,
        )
    return 0 if valid else 1


def _align(args: argparse.Namespace) -> int:
    auxiliary = [args.config] if args.config else []
    _one_stdin([args.left, args.right, *auxiliary])
    left = read_profile(args.left)
    right = read_profile(args.right)
    parameters = read_json_object(args.config) if args.config else None
    alignment = align_graphs(
        left.graph_profile,
        right.graph_profile,
        parameters=parameters,
    )
    write_json_data(alignment.to_dict(), args.output)
    return 0


def _evidence(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    raw = read_json_data(path)
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("evidence JSON must contain an array of objects")
    return raw


def _infer_network(args: argparse.Namespace) -> int:
    auxiliary = [args.edge_evidence] if args.edge_evidence else []
    _one_stdin([*args.inputs, *auxiliary])
    result = infer_contact_network(
        tuple(read_profile(item) for item in args.inputs),
        edge_evidence=_evidence(args.edge_evidence),
    )
    write_json_data(result, args.output)
    return 0


def _infer_lineage(args: argparse.Namespace) -> int:
    auxiliary = [item for item in (args.config, args.evidence) if item]
    _one_stdin([*args.inputs, *auxiliary])
    config = read_json_object(args.config) if args.config else None
    result = infer_lineage_network(
        tuple(read_profile(item) for item in args.inputs),
        config=config,
        evidence=_evidence(args.evidence),
    )
    write_json_data(result, args.output)
    return 0


def _toy_tree(args: argparse.Namespace) -> int:
    result = build_toy_phylodigital_tree()
    if args.summary:
        payload = toy_tree_summary(result) + "\n"
        if args.output is None or args.output == "-":
            sys.stdout.write(payload)
        else:
            from pathlib import Path

            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(payload, encoding="utf-8")
    else:
        write_json_data(result, args.output)
    return 0


def _rank_models(args: argparse.Namespace) -> int:
    result = fetch_popularity_manifest(
        limit=args.limit,
        include_citations=not args.without_citations,
    )
    write_json_data(result, args.output)
    return 0


def _infer_popularity_tree(args: argparse.Namespace) -> int:
    manifest = read_json_object(args.manifest)
    result = extract_popularity_tree(
        manifest,
        max_models=args.max_models,
        max_graph_nodes=args.max_graph_nodes,
        max_hidden_layers=args.max_hidden_layers,
        per_model_timeout=args.per_model_timeout,
        radii=args.radii,
        profiles_directory=args.profiles_dir,
    )
    write_json_data(result, args.output)
    return 0


def _enrich_model_citations(args: argparse.Namespace) -> int:
    result = fetch_citation_evidence(read_json_object(args.manifest))
    write_json_data(result, args.output)
    return 0


def _importance(args: argparse.Namespace) -> ImportanceFilter | None:
    top_k = getattr(args, "top_k", None)
    top_p = getattr(args, "top_p", None)
    metric = getattr(args, "importance", None)
    if top_k is None and top_p is None:
        if metric:
            raise ValueError("--importance requires --top-k or --top-p")
        return None
    if not metric:
        raise ValueError("--top-k/--top-p requires --importance")
    return ImportanceFilter(metric=metric, top_k=top_k, top_p=top_p)


def _date_window(args: argparse.Namespace) -> DateWindow | None:
    since = getattr(args, "since", None)
    until = getattr(args, "until", None)
    day = getattr(args, "day", None)
    if day is not None:
        if since is not None or until is not None:
            raise ValueError("--day cannot be combined with --since or --until")
        try:
            parsed = date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError("--day must use YYYY-MM-DD") from exc
        since = parsed.isoformat()
        until = (parsed + timedelta(days=1)).isoformat()
    if since is None and until is None:
        return None
    return DateWindow(field=args.date_field, since=since, until=until)


def _feed_records(path: str) -> list[dict[str, Any]]:
    raw = read_json_data(path)
    if isinstance(raw, dict):
        raw = raw.get("entries")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("feed JSON must be an array of objects or an object with entries")
    return raw


def _catalog_discovery(args: argparse.Namespace) -> dict[str, Any]:
    provider_name = str(args.provider).strip().casefold()
    query = CatalogQuery(
        provider=provider_name,
        importance=_importance(args),
        date_window=_date_window(args),
        page_size=args.page_size,
        library=args.library,
        search=args.search,
    )
    if provider_name == "huggingface":
        if args.feed:
            raise ValueError("--feed is only valid for a feed provider")
        provider = HuggingFaceCatalogProvider()
    elif provider_name == "github":
        if args.feed:
            raise ValueError("--feed is only valid for a feed provider")
        provider = GitHubCatalogProvider()
    elif provider_name == "openalex":
        if args.feed:
            raise ValueError("--feed is only valid for a feed provider")
        provider = OpenAlexPaperProvider()
    elif provider_name == "epoch":
        if args.feed:
            raise ValueError("--feed is only valid for a feed provider")
        provider = EpochCatalogProvider()
    elif provider_name == "openrouter":
        if args.feed:
            raise ValueError("--feed is only valid for a feed provider")
        provider = OpenRouterCatalogProvider()
    else:
        if not args.feed:
            raise ValueError("custom/vendor providers require --feed JSON")
        provider = RecordFeedCatalogProvider(
            provider_name,
            _feed_records(args.feed),
        )
    resume = read_json_object(args.resume) if args.resume else None
    return discover_catalog(
        query,
        provider=provider,
        resume=resume,
        max_pages=args.max_pages,
    )


def _discover_models(args: argparse.Namespace) -> int:
    write_json_data(_catalog_discovery(args), args.output)
    return 0


def _select_models(args: argparse.Namespace) -> int:
    result = select_catalog(
        read_json_object(args.catalog),
        importance=_importance(args),
        distinct_primary_papers=args.distinct_primary_papers,
        paperless_quota=args.paperless_quota,
    )
    write_json_data(result, args.output)
    return 0


def _bind_model_papers(args: argparse.Namespace) -> int:
    raw = read_json_data(args.bindings)
    if isinstance(raw, dict):
        raw = raw.get("bindings")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("bindings JSON must be an array or an object with bindings")
    result = bind_primary_papers(
        read_json_object(args.catalog),
        raw,
        require_complete=args.require_complete,
        require_unique_papers=args.require_distinct_primary_papers,
    )
    write_json_data(result, args.output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except BrokenPipeError:
        return 0
    except (
        OSError,
        GraphExtractionError,
        ModelCatalogError,
        PhylodigyIOError,
        PopularityCorpusError,
        TorchUnavailableError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
