---
title: Profile Comparison API
aliases:
  - Graph comparison
tags:
  - phylodigy/reference
  - graph
  - distance
---

# Profile comparison API

Structural comparison consumes computation graphs. It does not compare
predeclared architecture traits or paper-supplied names.

## Screening distance

```python
compare_graphs(
    left,
    right,
    *,
    radii=(0, 1, 2, 3),
    radius_weights=None,
    region_weight=1.0,
) -> GraphDistance
```

The current implementation builds direction-aware WL and discovered-region
count vectors and returns raw weighted L1 distance.

`GraphDistance` serializes:

- total and per-radius distance;
- selected positive finite weights;
- left and right graph digests;
- region distance and its positive finite weight;
- distance scheme and vector-space label;
- an algorithm signature binding the method parameters.

No pairwise size normalization is applied. The distance is symmetric and
triangular in fingerprint space, but only a pseudometric on graphs.

## Comparability

Before comparing graphs, verify compatible:

- frontend and lowering scheme;
- operator schema version;
- probe suite and guards;
- tensor-metadata policy;
- opaque-operation policy;
- fingerprint or discovery method version.

Missing or incompatible coverage must produce an explicit diagnostic, not a
renormalized claim of equivalence.

## Character comparison

Generic discovery emits a corpus-relative character matrix. Character IDs are
canonical structural signatures. Counts, presence, multiplicity, and localized
transformations can be compared without semantic names.

Overlapping graph characters are statistically dependent. Bootstrap units
should therefore use bounded graph regions or probe strata rather than treating
every hash as independent.

## Primitive graph alignment

The first alignment layer is a separate API from `compare_graphs`. It compares
structural node records, constructs node correspondences with bounded sequence
dynamic programming, and emits a replayable `GraphAlignment`.

```python
from phylodigy import (
    AlignmentParameters,
    GraphAlignment,
    align_graphs,
    apply_graph_alignment,
    validate_graph_alignment,
)

alignment = align_graphs(
    left,
    right,
    parameters=AlignmentParameters(
        node_indel_cost=1.0,
        edge_indel_cost=1.0,
        max_matrix_cells=1_000_000,
    ),
)
target_graph = apply_graph_alignment(left, alignment)
validation = validate_graph_alignment(left, right, alignment)
```

Structural node equality uses operator identity and structural attributes. It
excludes canonical node IDs and probe observations. Structural edge equality
uses mapped endpoints, direction, and canonical input position.

Version zero has only four primitive replay-operation kinds:

- add node;
- remove node;
- add edge;
- remove edge.

Edges are explicit operations independent from node operations. A node change
is represented by removing and adding the unequal nodes plus their affected
edges. There is no operator substitution, rewiring, region substitution,
duplication, or other macro-operation in this version.

The alignment pins both endpoint structural digests, structural node and edge
matches, solver-scoped underidentification records, primitive costs, the solver
budget and representative tie count, and exact integer bound components. The
endpoint keys are
`left_structural_digest` and `right_structural_digest`; artifact names and
metadata-bearing genome digests are not structural endpoints.
`apply_graph_alignment` validates the pinned source and replays the ordered
operations to the pinned target.

The serialized artifact exposes a `bounds` object, `total_cost`, neutral
`unmatched_left_*` and `unmatched_right_*` lists, explicit
`replay_direction`, and `ancestral_interpretation: none`. Its
`cost_model_digest` excludes solver choice and resource budget; those have a
separate `solver_configuration_digest`. `GraphAlignment.from_dict()` verifies
the artifact's internal derived values and content digest.

Forward add and remove operations describe left-to-right replay. Reversing the
endpoints reverses those operations at equal cost. This symmetry does not
identify an ancestor or historical gain/loss.

### Exactness boundary

The sequence dynamic program is deterministic and bounded. Its selected
alignment is a candidate upper bound, not a claim of globally minimum graph
edit distance.
The alignment artifact deliberately makes no self-authenticating proof claim,
even when `bounds.closed` is true. Call it optimal only for the declared
primitive exact-label objective when `validate_graph_alignment(left, right,
alignment)` has recomputed the endpoints and returned
`objective_optimality_proven: true`.

Repeated or symmetric structures may permit several equally supported node
correspondences. `solver_mapping_underidentified` and
`underidentified_mappings` retain local indistinguishability for this
representative solver. They do not count globally optimal alignments. A
canonical representative is a reproducibility choice, not evidence that one
occurrence is historically homologous.

Candidate alignment cost is not automatically a metric or a phylogenetic
branch length. Objective-scoped optimality does not change that. It must not be
added to overlapping WL-character differences. The current lineage matrix
continues to use its separately declared graph comparison method.

If the dynamic-programming matrix would exceed `max_matrix_cells`, alignment
raises `AlignmentResourceLimitError`. This is an explicit comparability failure,
not a missing edge with zero cost.

## Matrix diagnostics

For a complete pairwise matrix, report:

- symmetry and diagonal checks;
- triangle violations;
- quartet four-point residuals;
- least-squares tree residuals;
- missing or incompatible graph pairs;
- bootstrap stability over regions and probes.

A metric matrix can still be nonadditive. Use neighbor joining or weighted
least squares only after these diagnostics. Preserve systematic incompatibility
as network or contact signal.

## Paper annotations

Labels and descriptions can be joined to comparison output for explanation.
They cannot change alignment eligibility, feature counts, edit cost, or a
structural character ID.

## Related notes

- [[Python API]]
- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Homology and Convergence]]
- [[Validation Strategy]]
