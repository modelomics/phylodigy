---
title: Graph-Derived Characters
aliases:
  - Structural characters
  - Dynamic characters
tags:
  - phylodigy/concept
  - graph
  - phylogeny
status: experimental
updated: 2026-08-29
---

# Graph-Derived Characters

Phylodigy does not maintain a catalog of architecture traits, allowed
states, named model families, or feature-specific detectors. The computation
graph is the observation. Structural characters are generated from that graph
for the corpus being analyzed.

This is a strict boundary:

- extraction may record operator identity, execution-used arguments, declared
  resource relationships, probe tensor metadata, and directed data flow
  supplied by a framework frontend;
- generic graph algorithms may enumerate neighborhoods, graphlets, paths,
  branch/reconvergence regions, repeated regions, and edit candidates;
- extraction may not decide that a region is a ResNet block, attention layer,
  normalization scheme, or other manually enumerated technique;
- a paper may attach a human label and description only to a concrete graph
  node, edge, region, or graph-derived character.

## Character identity

A discovered character has a content-derived identifier. Its identity is based
on canonical graph structure and structural call/resource attributes, not a
human name or probe-dependent tensor observation.

At minimum, retain:

- the discovery algorithm and version;
- the graph digest and exact supporting nodes and edges;
- the canonical structural signature;
- multiplicity and locations in the graph;
- observability and probe provenance;
- any uncertainty introduced by opaque operations or incomplete traces.

The character remains usable when no paper describes it. Human annotations
are evidence about the character; they do not redefine its structural
identity.

## Corpus-relative discovery

Characters should be mined without a technique vocabulary. A practical
pipeline is:

1. enumerate generic local and region-level structures in each graph;
2. canonicalize or hash each structure;
3. cluster equivalent or near-equivalent structures across the corpus;
4. retain counts, locations, gains, losses, and transformations as
   phylogenetic characters;
5. align paper passages to those concrete structures when evidence permits.

Discovery parameters are part of the scientific method. Radius, graphlet
size, region-boundary rules, support thresholds, and clustering distance must
be stored with the output. Changing them creates a different character matrix.

## Paper annotations

A paper annotation is an overlay with four independent claims:

- **target**: the exact graph, node, edge, region, or character being discussed;
- **label**: the authors' or curator's human-readable name;
- **semantics**: optional verbatim name, description, claimed function, and
  provenance statement;
- **provenance**: normalized text offsets, excerpt, document digest, and
  confidence.

The system must permit several labels for one structure, one label for several
structures, conflicting descriptions, and an unlabelled structure. It must not
turn a phrase match into a structural observation.

Paper text can help select among graph-alignment candidates, but cannot create
nodes, edges, regions, or characters that were not observed. A passage with no
validated target is retained as an `UnverifiedPaperClaim`. For closed models
with no graph, paper claims remain document evidence and cannot masquerade as a
graph profile.

## Phylogenetic use

Anonymous structural signatures form a character matrix without requiring
predeclared states. Future generic region-level insertion, deletion,
duplication, substitution, and rewiring models may supply an event-weighted
graph distance after alignment and optimality are validated.

Character differences and primitive graph alignments are separate
representations.
One node or edge change can alter many overlapping WL or region characters, so
their costs must not be added together as if they were independent events.

The first bounded alignment layer emits only node/edge matches, neutral
unmatched-left/right lists, and add/remove replay operations. Unequal
structural nodes require explicit node and edge operations; region duplication,
substitution, and rewiring are later models. Its candidate is operational and
direction-relative, with no ancestral interpretation.

Recurrence or convergence rates are estimated from corpus prevalence,
homoplasy, and fitted evolutionary models. They are not assigned by a built-in
ontology. Paper citations may support contact or transfer hypotheses, but they
do not alter the structural distance.

## Related notes

- [[Computation Graph]]
- [[Artifact Profile]]
- [[Citations as Evidence]]
- [[Homology and Convergence]]
- [[Validation Strategy]]
