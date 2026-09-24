---
title: CLI Reference
aliases:
  - Command reference
tags:
  - phylodigy/reference
  - phylodigy/cli
status: implemented
updated: 2026-09-01
---

# CLI reference

The `phylodigy` CLI discovers model metadata and builds architectural genomes
from operator or layer graph records. It infers relationships from
graph-derived characters. It has no fixed structural feature catalog.

Run `phylodigy COMMAND --help` for the exact help installed in the current
environment.

## Global syntax

```console
phylodigy [-h] [--version] COMMAND
```

Most commands write canonical JSON to standard output. `-o PATH` writes to a
file. A command can consume standard input through `-` at most once across its
primary and auxiliary inputs.

## Commands

| Command | Output |
| --- | --- |
| `build-genome` | Graph-derived `ArchitecturalGenome` |
| `prepare-paper` | Unannotated `PaperDocument` |
| `extract-code` | Nonstructural `SourceManifest` |
| `merge` | Metadata merge for identical genomes |
| `validate` | Genome validation report |
| `align` | Bounded, replayable primitive graph alignment |
| `infer-network` | Explicit dated contact records with graph diagnostics |
| `infer-lineage` | Graph-character distance phylogeny |
| `toy-tree` | Five-artifact demonstration lineage |
| `rank-models` | Revision-pinned popularity corpus manifest |
| `infer-popularity-tree` | Graph-only tree from traceable ranked artifacts |
| `enrich-model-citations` | Batched arXiv citation snapshot for a manifest |
| `modelome-plan` | Entry and resource inventory for a Modelome corpus |
| `modelome-tree` | Graph-derived tree with complete Modelome entry coverage |
| `discover-models` | Provider-scoped catalog snapshot |
| `select-models` | Importance-ranked catalog selection |
| `bind-model-papers` | Paper-grounded catalog |

`extract-paper` is an alias for `prepare-paper`.

## `build-genome`

```console
phylodigy build-genome RECORDS_JSON \
  --artifact-id ID \
  [--frontend FRONTEND] \
  [--name NAME] \
  [--release-date YYYY-MM-DD] \
  [--date-min YYYY-MM-DD] \
  [--date-max YYYY-MM-DD] \
  [-o PATH]
```

`RECORDS_JSON` is a JSON array of open graph-node records. Each record supplies
an input index, frontend operation kind, target schema, dynamic inputs, and
optional observed attributes. The command canonicalizes the graph and derives
all regions, multiscale fingerprints, and anonymous characters from it.

The frontend defaults to `generic`. The output can round-trip through
`ArchitecturalGenome.from_dict`; readers recompute all graph-derived fields.

## `prepare-paper`

```console
phylodigy prepare-paper PAPER \
  --artifact-id ID \
  [--artifact-kind KIND] \
  [--name NAME] \
  [--release-date YYYY-MM-DD] \
  [--date-min YYYY-MM-DD] \
  [--date-max YYYY-MM-DD] \
  [--source-id ID] \
  [--identifier KEY=VALUE] ... \
  [-o PATH]
```

`PAPER` accepts UTF-8 text, Markdown, PDF, or `-` for UTF-8 standard input.
Identifiers are repeatable and duplicate keys fail validation.

The result contains normalized text and provenance only. It performs no
semantic recognition. Use the Python annotation API after a graph exists to
bind verbatim paper labels and descriptions to validated graph targets.

## `extract-code`

```console
phylodigy extract-code SOURCE \
  --artifact-id ID \
  [--name NAME] \
  [--release-date YYYY-MM-DD] \
  [--date-min YYYY-MM-DD] \
  [--date-max YYYY-MM-DD] \
  [-o PATH]
```

This command parses generic source structure without executing a repository.
Its `SourceManifest` preserves files, hashes, source syntax graphs, and scan
provenance. It is deliberately not an architectural genome and cannot enter
graph distance or phylogeny.

## `merge`

```console
phylodigy merge GENOME [GENOME ...] [-o PATH]
```

Every input must identify the same artifact and the same structural graph.
The command combines compatible metadata and preserves metadata conflicts
explicitly. It never merges different traces or paper artifacts.

## `validate`

```console
phylodigy validate GENOME [GENOME ...] [-q | -o PATH]
```

Validation checks the architectural-genome schema, supplied content digest,
graph digest, fingerprint, discovered regions, and anonymous characters.

| Exit status | Meaning |
| --- | --- |
| `0` | Every genome is valid |
| `1` | At least one genome failed validation |
| `2` | The command or input is invalid |

Quiet mode emits no report.

## `align`

```console
phylodigy align LEFT_GENOME RIGHT_GENOME \
  [--config JSON] \
  [-o PATH]
```

The command aligns exact structural node labels and emits a content-addressed
`phylodigy.graph_alignment`. It pins both structural endpoint digests and
contains exact-label node and edge matches, solver-scoped underidentification,
neutral unmatched-left/right objects, explicit add/remove replay operations,
the solver budget and representative tie count, and exact-component cost
bounds. Every edge operation is explicit even when its endpoint node is added
or removed. Operator substitutions, rewiring, duplication, and region-level
macro-operations are not part of this version.

The optional configuration object accepts:

| Field | Default | Constraint |
| --- | --- | --- |
| `node_indel_cost` | `1.0` | Finite, strictly positive number |
| `edge_indel_cost` | `1.0` | Finite, strictly positive number |
| `max_matrix_cells` | `1000000` | Positive integer |
| `algorithm` | `canonical-sequence-exact-label-indel.v1` | Exact supported algorithm identifier |

`total_cost` is the replayable candidate's upper bound. `bounds.closed` is not
a self-authenticating proof in a serialized artifact. Only the Python
`validate_graph_alignment` API recomputes both endpoints and can report
`objective_optimality_proven`, scoped to the declared primitive exact-label
objective. This score is not a general graph-edit distance, lineage-matrix
distance, or branch length.

Add and remove describe only the mechanical left-to-right replay direction.
The output serializes `ancestral_interpretation: none`. Reversing the two
genomes yields the inverse equal-cost alignment; it does not convert the
comparison into evidence of ancestral gain or loss.

## `infer-network`

```console
phylodigy infer-network GENOME [GENOME ...] \
  [--edge-evidence JSON] \
  [-o PATH]
```

The command admits explicit provenance, awareness, or character-transfer
records through a strict date gate. Without `--edge-evidence`, it emits no
contact edges and no parent forest. It still reports graph-character overlap
and graph distance as pair diagnostics, but those measurements never create,
orient, or score an edge.

Character-transfer evidence must reference a derived character present in both
endpoint graphs. If it supplies a value, that value must equal the target's
derived count. Evidence cannot add or modify a character. Vertical history is
inferred separately by `infer-lineage`.

## `infer-lineage`

```console
phylodigy infer-lineage GENOME [GENOME ...] \
  [--config JSON] \
  [--evidence JSON] \
  [-o PATH]
```

The command builds a complete graph-character distance matrix, reports
four-point diagnostics, and constructs an unrooted neighbor-joining tree.
External evidence is retained in the output but does not change distances or
tree placement.

The optional configuration object accepts:

| Field | Default | Constraint |
| --- | --- | --- |
| `radii` | `[0, 1, 2, 3]` | Nonempty nonnegative integer set |
| `radius_weights` | `{}` | Positive finite weights for selected radii |
| `region_weight` | `1.0` | Positive finite number |
| `four_point_tolerance` | `1e-9` | Finite nonnegative number |

## Structural boundary

The CLI accepts only observed graph records as structural input. Configuration
keys, source symbols, artifact names, and paper phrases cannot create an
architectural genome or alter its character matrix.

## `toy-tree`

```console
phylodigy toy-tree [--summary] [-o PATH]
```

This command instantiates five small, executable `torch.nn.Module` networks,
traces them with the regular `torch.fx` model frontend, and passes their
observed graphs through the same graph-character lineage pipeline as
`infer-lineage`. Install the `model` extra first. The default output is
canonical lineage JSON; `--summary` emits a compact terminal walkthrough.

## Popularity corpus commands

```console
phylodigy rank-models [--limit 100] [--without-citations] -o MANIFEST
phylodigy enrich-model-citations MANIFEST -o CITATIONS_JSON
phylodigy infer-popularity-tree MANIFEST \
  [--max-models 100] [--max-graph-nodes 12000] [--max-hidden-layers 36] \
  [--per-model-timeout 45] [--radii 0] [--profiles-dir DIRECTORY] -o RUN_JSON
```

`rank-models` selects public Hugging Face artifacts tagged for Transformers by
likes at the snapshot time and pins each repository revision. Exact arXiv tags
may be resolved to OpenAlex citation counts during ranking. For a more efficient
independent snapshot, `enrich-model-citations` resolves all tagged arXiv IDs in
one Semantic Scholar Academic Graph batch. Citations cannot change rank or
distance.

`infer-popularity-tree` constructs standard Transformers classes on PyTorch's
meta device, uses their framework-provided dummy inputs, captures an executable
FX or export graph, and infers a lineage from successful profiles. It downloads
configuration files only, not weights. Every exclusion is retained with its
rank and error; no configuration or popularity field is converted into a
structural character.

## Modelome commands

These commands use a portable Modelome entry corpus to plan coverage and build
a tree from locally observed graph profiles. See [[Build a Modelome Tree]] for
the input preparation and binding workflow.

### `modelome-plan`

```console
phylodigy modelome-plan ENTRIES [-o PATH]
```

`ENTRIES` accepts a bundle directory, `entries.jsonl`, or a JSON/JSONL entry
file. Bundle directories are checked against their manifest. The plan
inventories entries and declared resources; it does not load local profiles.

### `modelome-tree`

```console
phylodigy modelome-tree ENTRIES \
  [--profiles-dir PATH] \
  [--bindings JSON_FILE] \
  [--max-taxa N] \
  [--newick PATH] \
  [-o PATH]
```

The command retains every Modelome entry in coverage and infers from profiles
that match entry IDs exactly or use an explicit `entry_id` to `artifact_id`
mapping. Unprofiled entries remain in coverage without inferred tree tips.
Exact diagnostics are limited to 100 bound profiles by default. `--newick`
writes the inferred tree in Newick format and requires at least two bound
profiles.

## Catalog commands

These independent commands create, rank, and bind local provider catalogs.
They do not require or store a registry directory.

### `discover-models`

```console
phylodigy discover-models \
  [--provider PROVIDER] \
  [--feed JSON] \
  [--importance METRIC (--top-k K | --top-p FRACTION)] \
  [--date-field {created,modified}] \
  [--day YYYY-MM-DD | \
    [--since ISO_DATE_OR_TIMESTAMP] [--until ISO_DATE_OR_TIMESTAMP]] \
  [--page-size N] [--max-pages N] \
  [--library FILTER] [--search QUERY] \
  [--resume SNAPSHOT_JSON] \
  [-o PATH]
```

The built-in providers are `huggingface`, `github`, `openalex`, `epoch`, and
`openrouter`. Any other provider requires `--feed` with a JSON array or an
object that contains `entries`.

GitHub requires `--search`. OpenAlex returns paper candidates instead of model
identities. The Hugging Face adapter does not download weights.

Epoch returns curated model records with associated publication evidence.
It supports `created` and `modified` dates and citation ranking.

OpenRouter returns current routed API offerings. It supports the `created` date
and `context_length` ranking. It requires `OPENROUTER_API_KEY`.

```console
phylodigy discover-models --provider epoch -o epoch-models.json
# Configure OPENROUTER_API_KEY through the process or scheduler secret store.
phylodigy discover-models --provider openrouter -o openrouter-offerings.json
```

With no importance filter, the command reads until the provider ends or the
page budget stops the scan. `--resume` continues a matching incomplete
snapshot.

Date windows use UTC. They include `--since` and exclude `--until`. `--day`
creates the interval from that day to the next day.

`--top-k` selects a bounded record count. `--top-p` selects the highest-scoring
fraction of eligible records that contain the metric and uses
`ceil(FRACTION * N)`. It is not probability sampling.

### `select-models`

```console
phylodigy select-models CATALOG_JSON \
  --importance METRIC \
  (--top-k K | --top-p FRACTION) \
  [--distinct-primary-papers] \
  [--paperless-quota N] \
  [-o PATH]
```

This command ranks one validated local catalog. Records without the selected
metric cannot enter the result.

`--distinct-primary-papers` removes repeated primary-paper identifiers and
unresolved models. It admits explicit paperless exceptions up to the optional
quota.

### `bind-model-papers`

```console
phylodigy bind-model-papers CATALOG_JSON BINDINGS_JSON \
  [--require-complete] \
  [--require-distinct-primary-papers] \
  [-o PATH]
```

Each binding identifies a catalog `version_id`. It must supply exactly one
`paper_id` or `paperless_reason`.

`--require-complete` rejects unresolved models. The distinct-paper option
rejects one paper identifier that binds to multiple models.

## Related notes

- [[Build a Modelome Tree]]
- [[Open Model Workflow]]
- [[Closed Model Workflow]]
- [[Artifact Profile Schema]]
- [[Infer Lineage]]
- [[Infer Network]]
- [[Python API]]
