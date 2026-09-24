---
title: Artifact Profile
aliases:
  - Model profile
  - Digital profile
tags:
  - phylodigy/concept
  - evidence
status: experimental
updated: 2026-08-29
---

# Artifact Profile

An artifact profile binds one immutable model artifact to observed computation
graphs and provenance. It does not contain a manually selected feature vector
and does not imply biological identity.

## Core record

Each architectural genome contains:

- an artifact ID and optional human metadata;
- a date or date interval constraining possible influence;
- exactly one graph profile produced by a labeled frontend and probe;
- extraction and discovery method identities;
- a content digest over the canonical record.

Paper documents and graph-targeted annotations are separate content-addressed
artifacts. They point to a pinned graph digest and cannot mutate this record.

Graph observations distinguish observed, unavailable, opaque, and uncovered
regions. Missing trace coverage is unknown, never a negative feature.

## Structural truth

For executable models, nodes, edges, structural call/resource attributes, and
probe observations come from framework graphs. Generic algorithms derive
anonymous characters from the structural projection; tensor probe metadata is
kept for audit and coverage rather than used as a character.

Architecture-family names, Python class names, configuration keys, source-code
symbols, and paper phrases cannot create structural characters. They may be
retained as provenance or annotations, but never substitute for graph support.

Repository analysis can still provide source-tree identity and homology
evidence. Configuration files can preserve launch and probe context. Neither
channel is a second architecture detector.

## Paper annotation overlay

A paper document is stored independently from the graph. A graph annotation
must target an existing graph object and records a label, description, exact
text span, confidence, and citation provenance.

Annotations are conflict-preserving and many-to-many. The graph's digest and
anonymous character IDs do not change when an annotation is added. A paper for
a closed model may remain useful document evidence, but it cannot synthesize
an unobserved graph.

## Multiple traces

One trace has a bounded capture scope. Store each distinct trace as its own
graph profile with versioned probe and guard metadata. The current genome
envelope contains one such profile. Comparative analyses must disclose how a
study selects or combines compatible traces.

Opaque calls remain nodes. A frontend failure or uncovered branch remains an
explicit limitation, not evidence that a structure is absent.

## Merge rule

Merging architectural genomes is deliberately narrow: artifact IDs and graph
structure must match exactly, and only compatible metadata is combined.
Different traces and paper artifacts remain separate. A merge does not select
one paper label or collapse incompatible graph coverage.

Resolution belongs to a named downstream comparison or inference method. This
keeps sensitivity analysis possible and prevents descriptive metadata from
silently changing structural evidence.

## Interpretation

A shared graph character can reflect inheritance, transfer, convergence,
common engineering constraints, or a collision in an approximate fingerprint.
Paper citations and source history can help distinguish these hypotheses, but
they do not alter the graph observation itself.

## Related notes

- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Evidence Channels]]
- [[Artifact Profile Schema]]
- [[Homology and Convergence]]
- [[Citations as Evidence]]
