---
title: Infer Lineage
tags:
  - phylodigy/workflow
  - phylogeny
---

# Infer lineage

Lineage inference begins with compatible computation graphs, not manually
selected feature vectors.

## 1. Freeze inputs

Pin artifact graph digests, frontend versions, probe suites, discovery
parameters, dates, and the corpus manifest. Exclude or label incompatible graph
coverage.

The implemented CLI accepts only architectural-genome JSON:

```console
phylodigy infer-lineage model-a.genome.json model-b.genome.json \
  -o lineage.json
```

Use `--config analysis.json` to select WL radii, radius weights, region weight,
and four-point tolerance. Use `--evidence records.json` only to retain external
records beside the result; those records do not change the matrix or tree.

## 2. Discover corpus characters

Run generic neighborhood, graphlet, path, region, and repetition algorithms.
Store the anonymous character matrix and exact occurrence locations.

## 3. Compare graphs

The implemented lineage command uses its declared graph-character distance.
The primitive indel API can additionally produce audited pairwise candidate
alignments and replay operations, but those candidate costs are not yet the
lineage branch-length matrix.

Use the same alignment parameters for every pair. Retain lower and upper
bounds, their exact integer components, separate cost/solver digests, the
matrix-cell budget, representative traceback ties, and solver-scoped
underidentification. Treat a resource-limit error as an explicit missing
comparison; version zero does not return a truncated alignment. A closed bound
becomes an optimality claim only for the declared primitive objective and only
after endpoint validation.

## 4. Diagnose the matrix

Check symmetry, diagonals, triangle inequalities, missing pairs, quartet
four-point residuals, and least-squares tree fit. Bootstrap over graph regions
and probe strata.

## 5. Infer the tree-like component

Use neighbor joining or weighted least squares when distance is approximately
additive. Fit a separate character model over discovered signatures as a
cross-check. Use dates and verified provenance only for admissible orientation.

## 6. Preserve conflict

Localize systematic tree incompatibility to supporting graph regions or edits.
Combine it with source, citation, and paper-annotation evidence to propose
contact, copied-component, convergence, or unresolved relationships.

## 7. Explain after inference

Join paper labels and descriptions to graph targets for human-readable output.
Do not allow those labels to alter distances, characters, alignments, or tree
placement.

## Required output

- input and method digests;
- compatibility and coverage report;
- anonymous character matrix;
- pairwise distances and, when run separately, bounded graph alignments;
- additivity and tree-fit diagnostics;
- backbone, branch lengths, and bootstrap support;
- localized incompatible/contact signal;
- paper annotations used only for explanation.

## Related notes

- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Profile Comparison API]]
- [[Infer Network]]
- [[Validation Strategy]]
