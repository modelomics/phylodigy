---
title: Research Design
aliases:
  - Study design
tags:
  - phylodigy/research
  - graph phylogeny
status: proposed
updated: 2026-08-29
---

# Research Design

This study treats model lineage as reconstruction from executable computation
graphs. It does not choose architecture traits in advance. The study's
characters, similarities, and candidate replay operations are generated from
the graphs in the frozen corpus.

## Research questions

1. Which graph representations recover independently documented model ancestry?
2. Which automatically discovered structures remain stable under harmless implementation refactors?
3. Which graph-edit models best separate inheritance, duplication, transfer, and convergence?
4. How well can paper passages name and describe exact graph regions without leaking names into discovery?
5. Where do quartet incompatibilities support a network rather than a tree?

## Unit of analysis

One taxon is one immutable executable artifact version under a documented
frontend and probe suite. Prefer a release tag, commit hash, model digest, or
archived artifact identity. A family name is never a taxon or a character.

## Corpus design

Stratify the corpus across:

- releases with independently known repository ancestry;
- renamed and mechanically refactored implementations;
- independently implemented but functionally similar structures;
- models assembled from copied, merged, adapted, or distilled components;
- models with high-quality papers that refer to exact implementation regions;
- models with opaque operations or partial trace coverage.

Include negative controls where class names or paper vocabulary are similar but
graphs differ, and where graphs are similar but names differ.

## Representation plan

The primary representation is the canonical operator/layer DAG, including
execution-used static arguments and declared tensor-resource relationships.
Probe tensor metadata is retained as nonstructural coverage evidence. From the
structural graph, generic algorithms derive:

- multiscale directed neighborhoods;
- bounded graphlets and path shingles;
- single-entry/single-exit regions;
- repeated and near-duplicate regions;
- cross-graph alignments and neutral unmatched-object comparisons.

Source-history, citation, and paper evidence remain independent channels.
Python names and configuration keys may support provenance but cannot create a
structural character.

## Comparative methods

Evaluate complementary methods:

1. scalable WL and graphlet pseudometrics for retrieval and alignment anchors;
2. bounded exact-label node/edge indel candidates for correspondence diagnosis
   and endpoint replay;
3. future hierarchical, globally optimized event-weighted graph edit distance
   as a branch-length candidate only after optimality and matrix diagnostics;
4. character-based phylogeny over discovered structural signatures;
5. neighbor joining or weighted least squares after additivity diagnostics;
6. split or contact-network methods for incompatible signal.

Do not use UPGMA without an independently justified clock. Do not normalize
branch distances by pairwise graph size.

The version-zero alignment score is the upper bound of one deterministic,
bounded sequence-DP candidate. It is not assumed to be a general minimum graph
edit distance, metric, triangular, additive, or ancestral. A closed serialized
bound is not a proof: only endpoint validation may report optimality for the
declared primitive exact-label objective. Left/right add and remove labels are
replay directions. Repeated or symmetric structures retain solver-scoped
underidentification rather than being silently declared homologous.

## Paper protocol

Graph discovery is frozen before semantic review. Annotators see candidate
graph targets and paper passages, then attach labels and descriptions with
exact spans. A paper phrase cannot add a node, edge, region, or character.

Evaluate target selection, span accuracy, and description agreement separately
from graph discovery. Preserve synonyms and conflicts.

## Expected outputs

- a versioned artifact and probe manifest;
- canonical graph profiles;
- an anonymous structural character matrix;
- audited graph alignments, replay operations, and endpoint-validation reports;
- a distance matrix with additivity diagnostics;
- a vertical tree hypothesis plus incompatible/contact signal;
- paper annotations bound to exact graph targets;
- bootstrap, sensitivity, and negative-control results.

## Related notes

- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Validation Strategy]]
- [[Assumptions and Limits]]
- [[Target Corpus]]
- [[Homology and Convergence]]
