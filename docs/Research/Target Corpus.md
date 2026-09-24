---
title: Target Corpus
aliases:
  - Corpus manifest
  - Model corpus
tags:
  - phylodigy/research
  - corpus
status: active
updated: 2026-08-31
---

# Target Corpus

This note defines a reproducible sampling plan. A dated candidate list now
implements its paper-grounded selection stage; it is not yet the final
artifact-resolved validation corpus.

Any actual corpus needs a dated manifest because rankings, releases, and evidence access can change.

## Meaning of top

`Top` must identify a population, public metric, snapshot date, and tie rule. No single ranking measures influence, adoption, capability, and openness.

Use one declared selection rule for each stratum. Archive every dynamic selection source when its terms permit archiving.

## Entry requirements

Each corpus entry must satisfy these requirements:

- One stable artifact ID
- One immutable locator or content digest
- A public date or uncertainty interval
- Cited date evidence
- An eligibility decision at the snapshot date
- Explicit code, weight, and paper access labels
- A declared selection stratum and selection rule
- License and access metadata

A family name cannot replace an artifact version. Separate releases need separate entries when their evidence differs.

## Planned strata

| Stratum | Preferred evidence | Research purpose |
| --- | --- | --- |
| Open base model | Executable graph, source provenance, weights | Test broad ancestry recovery |
| Declared fine-tune | Parent declaration and immutable artifacts | Test vertical edge recovery |
| Merge or distillation | Several declared sources | Test reticulate cases |
| Independent implementation | Distinct source history and shared methods | Test convergence controls |
| Closed model | Stable identity and citable paper evidence | Test paper-based placement |
| Method paper | Archived paper and exact citation spans | Test documentary provenance separately |

The final manifest should declare quotas before inference. Quotas should balance validation needs, access levels, families, and parameter scales.

## Required manifest fields

| Field | Scope | Purpose |
| --- | --- | --- |
| `corpus_version` | Corpus | Identify one frozen release |
| `as_of_date` | Corpus | Bound eligibility and source status |
| `selection_rule` | Corpus or stratum | Define candidate selection |
| `selection_source` | Corpus or stratum | Cite the archived source |
| `artifact_id` | Artifact | Identify one DAG node |
| `artifact_kind` | Artifact | Distinguish a model from a paper |
| `immutable_locator` | Artifact | Resolve the exact artifact |
| `content_digest` | Artifact | Detect later content changes |
| `date_min` and `date_max` | Artifact | Preserve temporal uncertainty |
| `date_source` | Artifact | Support the recorded interval |
| `evidence_access` | Artifact | Record code, weights, and paper access |
| `declared_sources` | Artifact | Record known parents or method sources |
| `license` | Artifact | Bound lawful access and redistribution |
| `exclusion_reason` | Candidate | Explain a rejected candidate |

## Sampling procedure

1. Freeze the snapshot date, population, metric, and tie rule.
2. Archive each selection source or record its stable version.
3. Collect candidates that were eligible at the snapshot date.
4. Resolve every candidate to one immutable artifact version.
5. Assign strata and evidence access labels.
6. Deduplicate aliases with recorded equivalence rules.
7. Publish the manifest, exclusions, and source digests before inference.

## Open model path

Record executable-graph, source, configuration, paper, and weight access
separately. Prefer immutable commits and content digests.

Use papers to name and describe validated graph targets, record attribution,
and preserve unverified claims. Let the observed graph decide structural
conflicts.

## Closed model path

A closed model can enter the corpus with a stable identity, public date, and
citable document. Mark unavailable code, weights, and graph evidence
explicitly.

Bind paper semantics to exact spans and citations. Without a validated graph
target, keep each passage as an unverified claim and exclude the artifact from
graph-only distances.

## Target list status

The repository includes
[`epoch-citation-top-100-distinct-papers.json`](../../artifacts/registry/epoch-citation-top-100-distinct-papers.json),
a 2026-08-31 candidate snapshot with 100 model records associated with 100
distinct primary papers and ranked by the papers' recorded citation counts.
The source snapshot is
[`epoch-all-models.json`](../../artifacts/registry/epoch-all-models.json).

This list is suitable for exercising corpus review and evidence collection. It
is not a completed phylogeny manifest: provider-local identities still need
cross-source reconciliation, and entries still need immutable artifact IDs or
an explicit closed-model evidence policy before graph inference. Citation
counts select candidates only. They cannot affect graph characters, distances,
or inferred ancestry.

## Related notes

- [[Research Design]]
- [[Validation Strategy]]
- [[Prior Art and SOTA]]
- [[Artifact Profile]]
- [[Evidence Channels]]
- [[Assumptions and Limits]]
