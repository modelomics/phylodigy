---
title: Assumptions and Limits
aliases:
  - Interpretation limits
  - Threats to validity
tags:
  - phylodigy/research
  - limitations
status: experimental
updated: 2026-08-29
---

# Assumptions and Limits

Phylodigy produces reproducible lineage hypotheses. It does not recover a
complete development history or prove causal influence.

## Artifact identity

An artifact ID should identify one immutable version. Mutable branches, reused
model names, or silently replaced weights can combine unrelated states.

Digests establish content identity for supplied records; they do not
authenticate a publisher or recover deleted/private history.

## Graph observability

No single trace is a complete description of arbitrary software. Dynamic
control flow, generated kernels, distributed execution, custom operations, and
input-specific branches can remain hidden or opaque.

Probe suites improve coverage but introduce a study design choice. Models must
be compared under compatible frontends, versions, probes, guards, and lowering
policies. Missing coverage is unknown, never absence.

Static source and configuration files cannot replace executable graph
evidence. Names can be stale, unused, misleading, or copied. Parameter values,
training data, and functional behavior remain outside the current graph
profile.

## Canonicalization

Open operator labels depend on frontend schemas and versioning. Two frameworks
may expose different primitive boundaries for equivalent computation. Overly
aggressive lowering can erase meaningful differences; overly shallow lowering
can preserve incidental wrappers.

Finite Weisfeiler-Leman refinement is not a complete graph-isomorphism test.
Different graphs may receive identical fingerprints. Canonicalization and
fingerprinting therefore require collision tests and source provenance.

## Dynamic discovery

Removing a fixed feature list does not remove modeling choices. Neighborhood
radii, graphlet size, region boundaries, repetition thresholds, and clustering
distance determine the resulting character matrix.

Automatically discovered structures need not correspond one-to-one with
human techniques or evolutionary events. One software change can affect many
overlapping characters, and one copied region can appear as many primitive
node edits.

Discovery parameters must be frozen before outcome analysis and varied in
sensitivity tests.

## Paper annotation

A paper records claims, not verified implementation. Authors may omit details,
use overloaded names, describe conceptual rather than lowered operations, or
publish after implementation.

Target validation prevents text from fabricating graph structure, but choosing
which observed region a passage describes can remain ambiguous. Preserve
synonyms, conflicts, rejected targets, and annotator confidence.

A closed model can have document evidence without a graph. It cannot receive a
graph-based phylogenetic placement unless a compatible graph becomes
observable.

## Distance and tree assumptions

WL, graphlet, and path-feature distances are graph pseudometrics. They support
retrieval and alignment but are not automatically evolutionary distances.

Graph edit distance depends on the edit vocabulary, alignment approximation,
and fitted costs. A symmetric metric can still fail the four-point condition.
Convergence, reversals, component reuse, distillation, and contact create
shortcuts inconsistent with a single tree.

The first primitive alignment layer uses bounded sequence dynamic programming
to produce a deterministic candidate alignment. Its upper bound is not a
globally minimum general graph edit distance. Even a closed serialized bound is
not self-authenticating: endpoint validation may establish optimality only for
the declared primitive exact-label objective. Candidate costs need not satisfy
the triangle inequality and are not phylogenetic branch lengths.

Add and remove names describe only the chosen left-to-right replay direction.
Reversing the comparison reverses those names. Neither direction identifies an
ancestor, gain, loss, duplication, or historical engineering operation.

Neighbor joining is exact only for additive input. UPGMA additionally assumes
a clock. Systematic residuals and quartet violations must remain visible as
network or split signal.

## Historical interpretation

Public dates impose only a partial order and do not reveal when private work or
idea transfer occurred. The earliest observed graph is not necessarily the
origin of a structure.

Common constraints can produce similar graphs independently. Conversely,
mechanical refactors can make inherited code appear structurally different.
Paper citations can express adoption, comparison, background, or convention
and are not proof of transfer.

## Corpus bias

Open-source and traceable models are overrepresented. Missing artifacts can
create false roots, false independent origins, and misleading branch lengths.
Frontend support may correlate with model community and era.

Every result should disclose exclusions, failed traces, opaque nodes, probe
coverage, and paper-only records.

## Related notes

- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Validation Strategy]]
- [[Homology and Convergence]]
- [[Citations as Evidence]]
- [[Target Corpus]]
