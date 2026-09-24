---
title: Validation Strategy
aliases:
  - Evaluation plan
tags:
  - phylodigy/research
  - validation
status: proposed
updated: 2026-08-29
---

# Validation Strategy

Validate graph capture, generic discovery, paper annotation, graph comparison,
and phylogenetic inference separately. End-to-end accuracy alone cannot reveal
which layer failed.

## Level 1: artifact integrity

For every persisted object verify:

- canonical serialization is invariant to mapping and valid emission order;
- supplied content digests match serialized content;
- node/edge references and graph acyclicity are valid;
- fingerprints, regions, and characters recompute from the graph;
- annotations resolve against pinned graph and document digests;
- invalid dates, spans, weights, and non-finite values are rejected.

## Level 2: frontend fidelity

Compare each lowered graph with the framework's authoritative IR. Sample nodes,
edges, nested argument positions, static arguments, tensor metadata, parameter
signatures, opaque calls, guards, and outputs.

Use probe suites that exercise alternative shapes and branches. Report node and
edge coverage by probe. Never score an uncovered branch as absent.

Test equivalent programs under variable renaming, module relocation, harmless
wrapper insertion, serialization order changes, and valid commutative operand
swaps when commutativity is declared by the frontend. Also include undeclared
or noncommutative counterexamples that must remain distinct.

## Level 3: dynamic character discovery

Discovery tests must be architecture-name blind. Fixtures use anonymous
operators and verify generic properties:

- every character has exact supporting nodes and edges;
- IDs are content-derived and invariant to incidental names;
- the same algorithm discovers every qualifying region, not just a named case;
- novel operator labels survive extraction and participate in signatures;
- graph changes produce localized character gains, losses, or transformations;
- collision rates are measured for bounded fingerprints and graphlets.

Use held-out graph families when tuning discovery parameters. Do not add a
special rule after inspecting one architecture.

## Level 4: paper annotation

Freeze graph extraction and discovery before reviewers see paper vocabulary.
Construct blinded tasks in which annotators select an existing graph target
and attach a label and description to an exact passage.

Measure:

- document normalization and span accuracy;
- target-selection precision and recall;
- agreement on graph, node, edge, region, and character targets;
- agreement on relation and confidence;
- synonym and conflict retention;
- rate of correctly rejected passages with no supported graph target.

Include adversarial cases where the same name refers to different graphs and
different names refer to equivalent graphs.

## Level 5: graph similarity and alignment

Benchmark on known refactors, copied components, independent
reimplementations, and versioned architecture changes.

Evaluate:

- nearest-neighbor retrieval using WL, graphlet, and path features;
- graph alignment accuracy against independently known correspondences;
- alignment correspondence and replay-operation localization;
- symmetry, non-negativity, identity, and, for declared metrics only, triangle
  behavior;
- sensitivity to probe coverage and opaque nodes;
- computational scaling with graph and corpus size.

Fingerprint collisions are expected and must be reported. They are not graph
isomorphism evidence.

### Primitive indel layer

Validate the first bounded alignment layer independently from fingerprint
distance and phylogeny. Its sequence dynamic program returns a deterministic
candidate alignment and cost bounds; it is not a globally minimum graph edit
distance and is not automatically a metric or branch length.

Use fixtures with an exact expected primitive alignment:

- identical graphs produce no operations, zero candidate cost, and closed
  lower and upper bounds;
- adding one acyclic shortcut between existing nodes produces one edge
  add operation, and the reverse comparison produces its edge remove operation;
- splicing `X` into `A -> B` produces one node add, removal of `A -> B`, and
  addition of `A -> X` plus `X -> B`; the reverse alignment has the same cost
  with every operation inverted;
- unequal structural nodes use explicit node and incident-edge operations in
  this layer; there is no operator-substitution shortcut;
- applying an alignment to its pinned left graph reproduces the pinned right
  graph, and applying the inverse reproduces the left graph.

Incident edges are explicit operations independent from node operations.
Validate replay order: remove affected edges before removing their node, and
add a node before adding edges that reference it. The total candidate cost must
equal the sum of finite, strictly positive primitive costs.

For every fixture, swap comparison direction and verify equal cost, swapped
endpoint digests, add/remove inversion, and successful replay. This is
operational symmetry only. Add and remove describe the chosen left-to-right
replay direction; they do not assert ancestral gain or loss.

Test determinism under source-variable renaming, mapping order, valid parallel
emission order, artifact metadata changes, and probe-only tensor observations.
These changes must not alter the structural endpoints, candidate alignment,
bounds, or solver-underidentification record.

Use two adversarial underidentification fixtures:

- two structurally identical fork branches with one branch extended;
- a repeated chain containing several equally valid locations for one inserted
  occurrence.

Swapping branch names or emission order must not choose a different alignment
by accident. The output must retain solver-scoped underidentification or apply
a content-based canonical tie break. Never use source names to break the tie.

Report the exact integer lower-bound components, candidate upper bound,
matrix-cell budget, representative traceback tie count, and solver-scoped
underidentified mappings. A serialized closed bound is not proof. Recompute
both endpoints with `validate_graph_alignment` and require its
`objective_optimality_proven` result before making an optimality claim for the
declared primitive objective. Exceeding the state budget must fail explicitly;
version zero does not return a truncated alignment. Benchmark matrix growth and
resource-limit failures on repeated and symmetric graphs.

Finally, repeat the fixtures with architecture-like artifact names, class
names, and paper text. Those strings must not appear in structural matches,
operations, costs, or solver mapping decisions. Unknown operator labels remain
valid structural data; no feature catalog is consulted.

## Level 6: phylogenetic recovery

Use histories supported independently by repository ancestry, release notes,
or archived component provenance. Evaluate parent ranking, split recovery,
branch-length error, and bootstrap support.

For each distance matrix report:

- missing and incompatible pairs;
- triangle violations;
- four-point residuals by quartet;
- least-squares tree residuals;
- stability under graph-region and probe bootstrap;
- changes under alternative symmetric edit-cost fits.

Keep reticulate cases in the benchmark. A method should localize incompatible
signal rather than earn credit for forcing it into a tree.

## Negative controls

- Shuffle architecture and class names while holding graphs fixed.
- Shuffle paper labels while holding graph targets fixed.
- Pair similar prose with structurally different models.
- Pair differently named implementations with equivalent graphs.
- Shuffle dates and citation edges.
- Remove individual probes and opaque selected regions.

Structural output must be invariant to the first two controls. Name or prose
leakage into graph characters is a test failure.

## Review protocol

1. Freeze the corpus, frontend versions, probes, and discovery parameters.
2. Hide lineage predictions from annotators.
3. Collect independent graph-target annotations.
4. Adjudicate with recorded evidence while preserving disagreement.
5. Publish exclusions, unresolved coverage, method signatures, and all tested configurations.

## Selected sources

- Felsenstein introduced [bootstrap confidence for phylogenies](https://doi.org/10.1111/j.1558-5646.1985.tb00420.x).
- Shervashidze and colleagues define the [Weisfeiler-Lehman graph kernel](https://jmlr.org/papers/v12/shervashidze11a.html).
- Saitou and Nei introduce [Neighbor Joining](https://consurfdb.tau.ac.il/documents/NJ_1987.pdf).

## Related notes

- [[Research Design]]
- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Assumptions and Limits]]
- [[Testing]]
- [[Target Corpus]]
