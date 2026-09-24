---
title: Roadmap
aliases:
  - Research roadmap
tags:
  - phylodigy/research
  - roadmap
status: living
updated: 2026-08-30
---

# Roadmap

## Implemented foundation

- [x] Canonical, open-vocabulary computation-graph schema.
- [x] Position-aware directed edges and observed node attributes.
- [x] Stable graph identity across incidental names and emission order.
- [x] Multiscale directed WL fingerprints and auditable distances.
- [x] Generic graph-region and content-derived character records.
- [x] Optional FX lowering with trace and probe provenance.
- [x] Content-addressed serialization and derived-field verification.
- [x] Bounded exact-label alignment with replayable primitive node/edge indel
      candidates, solver-scoped underidentification, and cost bounds.

The former fixed feature catalog, architecture-name detectors, config bridges,
AST symbol rules, module-family classifiers, and paper regex ontology are not
part of the graph model.

## Milestone 1: complete generic discovery

- [ ] Enumerate bounded directed graphlets.
- [ ] Add path and path-pair shingles.
- [ ] Discover single-entry/single-exit regions without operator-specific rules.
- [ ] Discover exact and approximate repeated regions.
- [ ] Emit corpus-level character matrices with support locations.
- [ ] Benchmark invariance, collision rates, and runtime.

## Milestone 2: paper-grounded semantics

- [x] Store normalized paper documents without inferring structural claims.
- [x] Validate annotations against exact graph targets.
- [x] Support labels, descriptions, citations, synonyms, and conflicts.
- [ ] Rank graph targets for a passage without changing graph identity.
- [ ] Evaluate blinded target and span agreement.

## Milestone 3: evolutionary distance

- [x] Emit deterministic primitive node/edge alignment candidates with exact
      endpoint replay and reversal symmetry.
- [x] Serialize endpoint structural digests, separate cost/solver identities,
      solver-scoped underidentification, and exact-component bounds.
- [x] Recompute endpoint-derived bounds and report objective-scoped optimality
      only from explicit endpoint validation.
- [ ] Align graphs hierarchically using fingerprint and region anchors.
- [ ] Generate region-level edit candidates from hierarchical alignment.
- [ ] Add substitution, rewiring, duplication, and other generic macro-event
      models only after they can be derived from aligned structure.
- [ ] Solve or certify globally minimum graph edit costs on bounded instances.
- [ ] Fit symmetric edit costs from documented histories.
- [ ] Audit minimum edit histories separately from the current bounded replay
      operations.
- [ ] Test triangle inequalities and quartet additivity only for the eventual
      declared distance, not for the version-zero candidate score.

## Milestone 4: phylogeny and reticulation

- [x] Infer a neighbor-joining tree from the current graph-character distance.
- [ ] Add weighted-least-squares trees after event-distance implementation.
- [ ] Fit character gain/loss models to anonymous graph signatures.
- [ ] Bootstrap over graph regions and probe subsets.
- [ ] Localize four-point violations to supporting graph changes.
- [ ] Add split-network or tree-plus-contact output for incompatible signal.

## Milestone 5: frontend and coverage expansion

- [ ] Add multi-probe trace unions and control-flow guards.
- [ ] Add exported PyTorch and ONNX frontends.
- [ ] Add JAX, TensorFlow, and compiler-IR frontends.
- [ ] Model opaque calls and incompatible probe coverage explicitly.
- [ ] Publish a frozen, licensed validation corpus.

## Decision gates

Do not add a technique-specific detector to improve one example. A failure must
be addressed by a more general graph representation or discovery algorithm and
evaluated across the corpus.

Do not call a fingerprint distance a branch length without an event model and
additivity diagnostics. Do not force incompatible history into one tree.

## Related notes

- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Research Design]]
- [[Validation Strategy]]
- [[Assumptions and Limits]]
