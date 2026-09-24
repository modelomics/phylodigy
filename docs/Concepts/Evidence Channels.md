---
title: Evidence Channels
aliases:
  - Evidence policy
tags:
  - phylodigy/concept
  - evidence
status: experimental
updated: 2026-08-29
---

# Evidence Channels

Evidence channels remain separate because they answer different questions.
Only an observed framework graph supplies primary structural evidence.

| Channel | What it can support | What it cannot do |
| --- | --- | --- |
| Computation graph | Nodes, directed edges, structural call/resource attributes, probe observations, coverage | Name a technique or prove ancestry |
| Source history | File identity, code copying, repository provenance | Create a graph structure from class names |
| Configuration | Probe context, declared dimensions, launch metadata | Infer architecture characters from key names |
| Paper document | Author claims, labels, descriptions, citations | Fabricate unobserved graph nodes or regions |
| Graph annotation | Paper meaning bound to an existing graph target | Change structural identity or distance |
| Citation/provenance | Awareness, attribution, possible contact | Prove implementation or transfer alone |

## Positive graph evidence

A frontend records what it observed under a particular version, probe, and set
of guards. Failed tracing, opaque operations, and uncovered branches remain
explicit. No channel converts non-observation into structural absence.

## Conflict handling

Graph extractions, source history, papers, and annotations can disagree. The
profile retains each record with its digest and provenance. A downstream method
may select compatible traces or compare alternative interpretations, but merge
does not average or overwrite them.

Several paper annotations may attach different labels or descriptions to one
graph target. This is representational evidence about terminology, not a graph
conflict.

## Evidence policy

Structural comparison consumes graph evidence only. Paper and source channels
may affect contact hypotheses, temporal constraints, target annotation, or
confidence in a historical interpretation. They do not receive a numeric
weight inside the graph fingerprint or edit distance.

## Related notes

- [[Artifact Profile]]
- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Citations as Evidence]]
- [[Assumptions and Limits]]
