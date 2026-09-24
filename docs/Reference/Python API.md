---
title: Python API
tags:
  - phylodigy/reference
  - python
---

# Python API

The public structural API is graph-first and open-vocabulary. No public API
loads a trait ontology, exports a feature catalog, or maps class/config/paper
strings to architecture states.

## Computation graph

```python
from phylodigy import (
    ComputationGraph,
    GraphNode,
    GraphEdge,
    GraphProfile,
    build_graph_profile,
    computation_graph_from_fx,
)
```

### `GraphNode`

```python
GraphNode(node_id, operation, attributes={}, observations={})
```

`operation` is an open-vocabulary frontend identity. `attributes` contains
canonical execution-used static arguments and declared resource structure.
`observations` retains probe-dependent tensor metadata for audit but excludes
it from characters and structural distance. Unknown operations are valid. The
core does not infer algebra from names; a frontend may explicitly declare a
generic property such as
`attributes={"operator_properties": {"commutative": True}}`.

### `GraphEdge`

```python
GraphEdge(source, target, position)
```

The position distinguishes repeated inputs and preserves argument order for
noncommutative operations.

### `ComputationGraph`

```python
ComputationGraph(nodes, edges)
```

Construction validates unique node IDs, endpoint references, and acyclicity.
`to_dict()`, `from_dict()`, `canonical_json()`, and `digest` provide canonical,
content-addressed serialization.

### `computation_graph_from_fx`

```python
computation_graph_from_fx(node_records)
```

This dependency-free lowering boundary accepts FX-like records. Each record
contains its source name, frontend operation kind, target schema, dynamic input
edges, and optional observed attributes. Incidental source names are not graph
identity.

### `build_graph_profile`

```python
build_graph_profile(
    node_records,
    *,
    frontend="torch.fx",
    metadata=None,
    radii=(0, 1, 2, 3),
    include_tensor_metadata=True,
)
```

The result contains the canonical graph plus recomputable fingerprints,
regions, and graph-derived characters. Deserialization verifies every derived
field against the graph.

## Generic discovery

```python
from phylodigy import (
    GraphRegion,
    GraphCharacter,
    GraphCharacterOccurrence,
    discover_graph_regions,
    discover_graph_characters,
    structural_fingerprint,
)
```

`discover_graph_regions(graph)` applies topology-only region discovery to all
operators. It does not select join operators or assign technique names.

`discover_graph_characters(graph, ...)` emits anonymous, content-derived
signatures and exact occurrences from generic neighborhoods and regions. Its
algorithm version and parameters are serialized.

`structural_fingerprint(graph, radii=(0, 1, 2, 3))` counts bidirectional
operator/attribute neighborhoods. It is deterministic but not a complete graph
isomorphism invariant.

Region and character objects expose `to_dict()` and `from_dict()`. IDs derive
from canonical content and cannot be supplied as human architecture names.

## Graph comparison

```python
from phylodigy import compare_graphs

result = compare_graphs(
    left,
    right,
    radii=(0, 1, 2, 3),
    radius_weights=None,
    region_weight=1.0,
)
```

`left` and `right` can be `ComputationGraph` or `GraphProfile`. The result is a
`GraphDistance` with total and per-radius distance, endpoint graph digests,
weights, algorithm scheme, and a reproducible distance signature.

The current value is raw weighted L1 distance in fingerprint space. It is a
metric on the selected feature vectors and a pseudometric on graphs. It is for
screening and alignment, not final phylogenetic branch lengths.

All selected weights must be positive and finite. Pairwise graph-size
normalization is deliberately absent.

## Primitive graph alignment

```python
from phylodigy import (
    AlignmentParameters,
    GraphAlignment,
    align_graphs,
    apply_graph_alignment,
    validate_graph_alignment,
)

parameters = AlignmentParameters(
    node_indel_cost=1.0,
    edge_indel_cost=1.0,
    max_matrix_cells=1_000_000,
)
alignment = align_graphs(left, right, parameters=parameters)
replayed = apply_graph_alignment(left, alignment)
assert replayed.structural_digest == right.graph.structural_digest
validation = validate_graph_alignment(left, right, alignment)
```

`align_graphs(left, right, *, parameters=None)` accepts a `ComputationGraph`
or `GraphProfile` at each endpoint and rejects noncanonical direct graphs. It
aligns exact structural node labels with a deterministic, bounded
canonical-sequence dynamic program. Node identity uses operation and structural
attributes, excluding node IDs and probe observations. Edges retain direction
and canonicalized input position, including declared commutative ports.

Schema version 1 has only four replay-operation kinds in JSON: `add_node`,
`remove_node`, `add_edge`, and `remove_edge`. Incident edges are explicit
operations rather than an implicit side effect of a node operation. Unequal
nodes require node and affected-edge operations; there is no substitution,
rewiring, duplication, or region macro-operation.

`GraphAlignment` pins the two structural endpoint digests, exact-label
`NodeMatch` and `EdgeMatch` records, solver-scoped underidentification records,
replay operations, and exact integer lower-bound components. Its serialized
`phylodigy.graph_alignment` also exposes neutral `unmatched_left_*` and
`unmatched_right_*` lists, `replay_direction`, and
`ancestral_interpretation: none`. Cost-model identity is separate from the
algorithm and matrix-budget identity through `cost_model_digest` and
`solver_configuration_digest`.

`GraphAlignment.from_dict()` verifies the artifact's internal derived values,
but does not trust it as an endpoint-derived proof.
`apply_graph_alignment()` rejects the wrong source or an alignment that cannot
replay exactly to the pinned target structure. `validate_graph_alignment()`
recomputes bound components from both endpoints, validates replay, and returns
the only `objective_optimality_proven` field.

`total_cost` is the candidate upper bound selected by this solver. A closed
bound becomes an optimality proof only after endpoint validation, and only for
the declared `exact_label_primitive_node_edge_edit.v1` objective. It is not a
claim of globally minimum general graph edit distance, a metric, or a
phylogenetic branch length.

Reversing the endpoint order returns `alignment.invert()` at equal cost. Add
and remove operations mean only mechanical left-to-right replay; they are not
ancestral gains or losses. `solver_mapping_underidentified`,
`underidentified_mappings`, and `representative_traceback_tie_count` describe
this canonical-sequence representative only, not the number of globally
optimal graph alignments or historical homology.

`AlignmentParameters` requires finite, strictly positive node and edge costs
and a positive integer `max_matrix_cells`. Exceeding that bound raises
`AlignmentResourceLimitError` instead of silently changing solvers.

## Live model extraction

```python
from phylodigy import extract_architectural_genome

genome = extract_architectural_genome(
    module,
    model_id="model:example",
    example_inputs=(...),
)
```

The FX frontend records target schemas, nested dynamic input positions, generic
static arguments, generic parameter/buffer resource relationships and
signatures, probe tensor metadata, frontend version, probe signature, and
trace/shape status. Arbitrary public Python fields do not become structure. It
never reads parameter values.

One trace has an explicit capture scope and may contain opaque layer boundaries
or uncovered control flow. Callers must version multi-probe studies and keep
failed or opaque coverage explicit.

Configuration-only and static-source functions may preserve artifact and
source provenance. They do not infer graph characters from recognized keys,
symbols, class names, or module-family names.

## Static source provenance

```python
from phylodigy import (
    SourceManifest,
    SourceScanPolicy,
    StaticSourceGraph,
    extract_code_profile,
    extract_source_profile,
)
```

These APIs parse generic syntax and data structure into a content-addressed
`SourceManifest`. The manifest has no `graph_profile` field and is rejected by
graph comparison, contact inference, and lineage inference. It can document
trace inputs or repository provenance without acting as an architecture
detector.

## Paper documents and graph annotations

```python
from phylodigy import (
    GraphAnnotationTarget,
    GraphAnnotation,
    PaperDocument,
    PaperGraphAnnotations,
    UnverifiedPaperClaim,
    annotate_graph_from_paper,
    normalize_extracted_text,
)
```

### `PaperDocument`

A paper document binds normalized text to its digest and optional bibliographic
identity. Constructing it performs no technique recognition.

### `GraphAnnotationTarget`

A target identifies a supplied graph profile and one concrete graph object:
the whole graph, a node, an edge, a discovered region, or a graph-derived
character. Use `GraphAnnotationTarget.resolve(graph_profile, kind=...,
target_id=...)` so IDs and digests are validated.

### `GraphAnnotation`

An annotation records a target, exact normalized text span, citations, and
confidence. It includes at least one optional verbatim semantic field: `name`,
`description`, `claimed_function`, or `provenance_statement`. The structural
identifier remains unchanged.

### `annotate_graph_from_paper`

```python
annotate_graph_from_paper(
    graph_profile,
    document,
    annotations,
    unverified_claims=(),
)
```

The function validates every graph target, span, verbatim semantic field, and
pinned digest. It does not search a regex vocabulary or infer a structure from
prose. Invalid or stale targets fail explicitly.

`PaperGraphAnnotations` is the canonical, content-addressed result. It permits
synonyms and conflicting descriptions for the same graph target.

`UnverifiedPaperClaim` retains a paper passage that has no validated graph
target. It never appears among structural annotations or graph characters.

## Artifact envelopes

`ArchitecturalGenome` binds immutable artifact identity and dates to exactly
one `GraphProfile` plus extractor provenance. `ArtifactProfile` is a
compatibility alias for this graph-only envelope.

`merge_profiles(*profiles)` accepts only identical complete graph profiles for
one artifact and combines compatible envelope metadata. Different traces,
character radii, paper documents, and annotation overlays remain independent
artifacts.

`read_profile(path)` and `write_profile(profile, path)` verify canonical JSON
and digests. `read_document_text(path)` supports text, Markdown, and optional
PDF extraction.

## Phylogeny boundary

The graph comparison API provides candidate-screening distance, and the first
alignment API provides audited primitive indel candidates. Neither is yet an
event-model branch-length API. Hierarchical region alignment, globally minimum
graph edit search, learned symmetric event costs, and matrix-level additivity
diagnostics remain required before edit costs can be used for neighbor joining
or weighted least squares.

Character-based analysis consumes anonymous graph signatures. Paper labels are
never character states.

## Removed fixed-semantics surface

The following are intentionally not part of the public contract:

- built-in or custom architecture ontologies;
- ontology export commands;
- paper regex lexicons;
- config-key-to-technique bridges;
- AST symbol-to-technique rules;
- module-family classifiers;
- architecture-specific motif detectors;
- manually assigned recurrence classes for named techniques.

## Related notes

- [[Artifact Profile Schema]]
- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Profile Comparison API]]
- [[Validation Strategy]]
