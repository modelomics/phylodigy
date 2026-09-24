---
title: Contact Network Schema
tags:
  - phylodigy/reference
  - schema
  - phylogeny
status: implemented
updated: 2026-08-29
---

# Contact network schema

`infer_contact_network` emits a canonical
`phylodigy.graph_contact_network`. Only explicit, endpoint-grounded evidence
can create an edge. Anonymous characters and graph distance are reported as
diagnostics and never infer contact or vertical ancestry.

## Top level

| Field | Meaning |
| --- | --- |
| `artifact_type` | `phylodigy.graph_contact_network` |
| `artifact_version` | Exact output contract |
| `nodes` | Genome identities, dates, and graph digests |
| `primary_parent_forest` | Always empty; vertical history belongs to lineage inference |
| `edges` | Explicit dated provenance, awareness, or grounded character-transfer records |
| `character_sources` | Origin hypothesis for every target character |
| `edge_evidence` | Canonical submitted contact-evidence records |
| `configuration` | Diagnostic-output controls only |
| `diagnostics` | Pairwise graph measurements, missing dates, and rejected evidence directions |
| `method` | Machine-readable structural, contact, and temporal boundary |
| `digest` | SHA-256 content digest |

## Pair diagnostics

`diagnostics.pair_diagnostics` can record `source_id`, `target_id`, temporal
relation, date gap, anonymous shared-character IDs and mass, Dice overlap, and
the independent graph-distance object. Set
`ContactInferenceConfig(include_unlinked_pair_diagnostics=False)` to omit pairs
that have no submitted evidence.

Graph similarity is Dice overlap on derived character counts. Graph distance
is the raw multiscale dynamic-character pseudometric and remains a separate
field. Neither value is an edge score, contact detector, orientation rule, or
parent selector. They can support human evaluation of an independently
evidenced contact without creating one.

## Time gate

Submitted evidence is admitted only when `source.date_max` is strictly earlier
than `target.date_min`. Evidence with missing or overlapping dates appears in
`diagnostics.temporal_exclusions` and creates no edge.

## Contact evidence

```json
{
  "source_id": "model:earlier",
  "target_id": "model:later",
  "causal_role": "character_transfer",
  "character_id": "content-derived-id",
  "character_value": 2,
  "kind": "paper_statement",
  "evidence_tier": "paper",
  "strength": 0.8,
  "locator": "paper:later#span=420:487"
}
```

Allowed causal roles are `provenance`, `character_transfer`, and `awareness`.
Evidence tiers are `code`, `paper`, and `curated`. Both endpoint IDs must name
genomes supplied to the same inference call.

A `character_transfer` record must reference a character found in both
endpoint genomes. An optional `character_value` must equal the target's
derived count. Arbitrary IDs and semantic aliases fail validation.

## Edges and character origins

Admitted provenance and awareness records produce `provenance` and `awareness`
edges. Admitted character-transfer records produce a `transfer` edge listing
exact character IDs. No role creates a vertical-parent edge, and
`primary_parent_forest` remains empty. Use `infer_lineage_network` for the
graph-character phylogeny.

For each character occurrence, `character_sources` reports one of:

- `transfer` when grounded donor evidence supports it;
- `first_observed` otherwise, including when similar earlier graphs contain it.

These are deterministic hypotheses, not verified historical causes.

## Related notes

- [[Time-aware Contact DAG]]
- [[Graph-Derived Characters]]
- [[Infer Network]]
