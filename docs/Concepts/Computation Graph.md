---
title: Computation Graph
aliases:
  - Graph profile
tags:
  - phylodigy/concept
  - graph
  - phylogeny
status: experimental
updated: 2026-08-29
---

# Computation graph profile

For an executable model, the normalized operator/layer graph is the primary
structural profile. The extractor has no list of architectures, techniques,
traits, or motifs to look for.

This means a graph is useful before anyone has named its structures. A fork,
two paths, and a later reconvergence are represented exactly as observed.
Whether a paper calls the region a residual connection, skip connection, or
something new is a separate annotation question.

## Representation

A graph profile contains:

- operation nodes with open-vocabulary operator identities;
- directed, position-aware tensor-flow edges;
- generic static call arguments and declared parameter/buffer resource structure;
- optional probe-dependent tensor shape and dtype observations;
- extraction provenance, including frontend version and probe identity;
- deterministic multiscale structural fingerprints;
- universally discovered graph regions and characters with exact support.

Unknown operations remain in the graph. No node is discarded because it is
absent from a vocabulary.

FX is the first frontend. ONNX, exported PyTorch, JAX, TensorFlow, and compiler
IRs can lower into the same records without changing the comparison layer.

## Canonicalization boundary

Canonicalization removes incidental serialization differences while retaining
computational distinctions. Framework-qualified operator schemas are
preferred over project-maintained semantic aliases. Variable names, module
paths, and graph emission order are provenance, not graph identity.

Structural identity is derived from operators, directed connectivity, input
positions, execution-used static arguments, and declared resource structure.
Probe tensor metadata is retained outside that identity. Commutativity or other
algebraic properties may be consumed when supplied by the frontend's operator
schema; the graph layer does not maintain a hand-written list of techniques.

Direction-aware refinement supplies name-free color cells; tied cells use a
bounded canonical-labeling search rather than source order. The multiscale WL
fingerprint remains an intentionally incomplete graph invariant, so it is a
screening pseudometric rather than proof of graph isomorphism.

## Dynamic discovery

The implemented generic discovery pass runs over every graph:

1. rooted directed neighborhood refinement;
2. anonymous fork/join region discovery; and
3. exact grouping of repeated neighborhood and region signatures.

Bounded graphlets, path shingles, and near-duplicate regions remain planned
extensions. The first cross-graph alignment layer uses a bounded sequence
dynamic program over structural nodes and emits exact node/edge matches,
neutral unmatched-object lists, and explicit replay operations. It uses the
same name-free boundary.

Outputs are anonymous, content-derived structural signatures. Discovery never
branches on a technique name or a curated operator sequence. A particular
branch/reconvergence pattern therefore needs no dedicated “residual” detector.
It emerges as one region among all regions.

The output for each character includes its supporting nodes and edges,
multiplicity, locations, discovery parameters, and uncertainty. See
[[Graph-Derived Characters]].

## Paper-grounded meaning

Papers supply names and descriptions only after a graph object exists. An
annotation targets an exact graph, node, edge, region, or graph-derived
character and binds normalized text offsets plus document identity.

This overlay is many-to-many and conflict-preserving. It does not rename the
structural identifier, change a fingerprint, or create a missing structure.
Models without observable graphs can carry unverified document claims, but not
synthetic graph characters.

## Distances for phylogeny

Neighborhood and graphlet distances support fast screening and alignment.
They are pseudometrics on graphs because distinct graphs can share feature
counts.

The first edit layer proposes a replayable primitive node/edge alignment from
graphs rather than selecting changes from an architecture catalog. It is a
deterministic bounded candidate, not a globally minimum general
[graph edit distance](https://en.wikipedia.org/wiki/Graph_edit_distance) or a
phylogenetic branch length. A validated closed bound establishes optimality
only for its declared primitive exact-label objective. Region substitutions,
duplications, and other macro-operations remain future layers.

Forward add and remove replay operations are reversed when the endpoint order
is swapped. They describe comparison mechanics only;
`ancestral_interpretation` is explicitly `none`, and a pairwise unrooted
alignment cannot infer historical gain or loss.

Pairwise size normalization is unsuitable for branch lengths because it erases
accumulated event cost. Keep normalized similarity only for retrieval.

A globally exact symmetric edit metric would still not automatically be a tree
distance. Approximate alignment candidates need not be metrics at all. Test
triangle inequalities and the four-point condition before tree inference. Use
[neighbor joining](https://en.wikipedia.org/wiki/Neighbor_joining) or weighted
least squares when the matrix is approximately additive. Use UPGMA only with
an independently justified clock.

Systematic incompatibility can indicate convergence, copied components,
distillation, or other contact. Retain it in a network or split representation
instead of forcing every relationship into one tree.

The anonymous character matrix supports a second, character-based analysis and
bootstrap resampling over graph regions. Human paper labels are display and
evidence metadata, not states in that matrix.

## Trace scope

No single runtime trace completely describes arbitrary dynamic Python.
Data-dependent control flow, input-specific branches, generated kernels, and
custom operations can remain hidden or opaque.

Comparative studies should define a versioned probe suite and may combine
several labeled traces. Guards and probe identities are part of provenance.
Missing coverage remains unknown and never means absence.

## Selected sources

- Shervashidze and colleagues define the [Weisfeiler-Lehman graph kernel](https://jmlr.org/papers/v12/shervashidze11a.html).
- Shervashidze and colleagues describe [graphlet kernels](http://proceedings.mlr.press/v5/shervashidze09a/shervashidze09a.pdf).
- Bunke and Shearer give a [maximum-common-subgraph metric](https://doi.org/10.1016/S0167-8655(97)00179-7).
- Saitou and Nei introduce [Neighbor Joining](https://consurfdb.tau.ac.il/documents/NJ_1987.pdf).
- Bryant and Moulton introduce [Neighbor-Net](https://www.maths.otago.ac.nz/~dbryant/Papers/04NeighborNet.pdf) for incompatible signal.

## Related notes

- [[Graph-Derived Characters]]
- [[Artifact Profile]]
- [[Profile Comparison API]]
- [[Infer Lineage]]
- [[Homology and Convergence]]
- [[Validation Strategy]]
