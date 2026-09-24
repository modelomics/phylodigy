---
title: Time-aware Contact DAG
aliases:
  - Lineage network
tags:
  - phylodigy/concept
  - phylogeny
  - network
status: research-model
updated: 2026-08-29
---

# Time-aware Contact DAG

The contact DAG records explicit provenance, awareness, and grounded
character-transfer evidence between dated architectural genomes. It complements
the graph-derived phylogeny but does not contain or infer its vertical history.
Every directed contact edge must satisfy a strict public-time gate.

## Nodes

One node represents one immutable artifact. Model nodes bind compatible graph
profile digests and coverage. A paper artifact may support attribution or
contact, but cannot become a model's structural parent without its own model
graph.

## Separate vertical phylogeny

Vertical history is inferred in the lineage analysis from event-weighted graph
distance and anonymous structural characters. It retains alternative
placements, tree-fit residuals, and bootstrap support.

Neighbor joining produces an unrooted tree from approximately additive
distances. The contact layer neither turns similarity into an edge nor selects
a parent forest. Dates and independently verified evidence govern only the
admission and direction of contact records.

## Lateral signal

A character-transfer edge identifies an anonymous character present in both
endpoint graphs and binds it to explicit evidence. Provenance and awareness
records can create their own contact edges without asserting that a particular
character moved.

Graph similarity, an aligned region, or an edit history can motivate evidence
collection and remain a diagnostic, but cannot create or orient a contact edge.
When explicit evidence cannot distinguish copying from independent
construction, retain that uncertainty. Do not resolve it by assigning a
hand-written recurrence class to a named technique.

## Temporal gate

A directed source is admissible only when its latest possible date is strictly
earlier than the target's earliest possible date. Overlapping or missing
intervals remain directionally unresolved.

Dates reject impossible directions but do not prove influence. Private work may
predate public release.

## Diagnostics

The contact artifact retains:

- graph identities and method boundaries;
- pairwise graph distance, overlap, and shared-character diagnostics;
- missing dates and temporally rejected evidence;
- exact admitted evidence records;
- transfer or `first_observed` character-origin hypotheses.

Triangle and quartet residuals, alternative tree placements, and the vertical
backbone remain in the lineage artifact.

## Related notes

- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Homology and Convergence]]
- [[Citations as Evidence]]
- [[Contact Network Schema]]
