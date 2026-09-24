---
title: Artifact Profile Schema
tags:
  - phylodigy/reference
  - schema
---

# Artifact profile schema

An architectural genome is a canonical envelope for one observed graph profile
and artifact provenance. Paper documents and graph annotations are separate,
content-addressed artifacts. None contains an ontology identifier or a
predeclared trait array.

## Top level

```json
{
  "artifact_type": "phylodigy.architectural_genome",
  "schema_version": "3.0.0",
  "artifact": {
    "id": "model:example",
    "kind": "model",
    "release_date": "2025-01-15"
  },
  "graph_profile": {
    "artifact_type": "phylodigy.computation_graph_profile",
    "profile_version": "2"
  },
  "extractor": {
    "name": "example.frontend",
    "version": "1"
  },
  "digest": "..."
}
```

Collections are canonicalized by their content identities. Reordering input
records does not change the digest.

| Field | Meaning |
| --- | --- |
| `artifact_type` | Constant profile type. |
| `schema_version` | Exact reader/writer contract. |
| `artifact` | Immutable identity, type, date interval, and nonstructural metadata. |
| `graph_profile` | One observed computation graph for a labeled frontend and probe scope. |
| `extractor` | Envelope producer identity. |
| `digest` | SHA-256 digest over every other top-level field. |

## Graph profile

A graph profile uses artifact type
`phylodigy.computation_graph_profile`. It contains:

- `graph`: canonical nodes and directed, position-aware edges;
- `fingerprint`: automatically derived multiscale structural features;
- `regions`: generically discovered regions with exact support;
- `characters`: content-derived structural signatures and occurrences;
- `frontend` and `metadata`: trace, probe, guard, and coverage provenance;
- `digest`: content identity for the complete graph profile.

No field stores an architecture family or a manually assigned technique state.
Operator strings are open vocabulary. Unknown operators remain in `graph`.

### Nodes

```json
{
  "id": "n000004",
  "operation": "function.framework.example_op",
  "attributes": {
    "arguments": {"kwargs['axis']": -1}
  },
  "observations": {
    "tensor_meta": {"dtype": "float32", "shape": [2, 16]}
  }
}
```

Node IDs are canonical within a graph. Incidental source variable and module
path names are not identities. Structural attributes participate in character
generation. Probe-dependent observations are preserved for audit but excluded
from structural IDs and phylogenetic distance.

### Edges

```json
{
  "source": "n000003",
  "target": "n000004",
  "position": "args[0]"
}
```

Positions preserve repeated operands and noncommutative argument order.

### Regions and characters

Every region and character is derived from the graph under a declared generic
algorithm. A record binds:

- its content-derived ID;
- algorithm name, version, and parameters;
- exact supporting node and edge IDs;
- a canonical structural signature;
- occurrences or multiplicity;
- uncertainty and observability metadata when applicable.

The reader recomputes derived records and rejects a mismatch.

## Paper document

A paper document stores normalized text, identity metadata, normalization
method, and a digest. Creating a document performs no technique extraction.

Its artifact type is `phylodigy.paper_document`. It is not embedded in the
architectural genome.

Text spans use zero-based half-open offsets into the exact normalized text
bound by the document digest.

## Graph annotation

```json
{
  "target": {
    "graph_digest": "...",
    "kind": "region",
    "target_id": "...",
    "structure_digest": "..."
  },
  "name": "authors' term",
  "description": "paper-grounded description",
  "excerpt": "exact normalized paper span",
  "source_id": "paper:example",
  "source_digest": "...",
  "start": 420,
  "end": 487,
  "confidence": 0.92,
  "citations": ["doi:10.example/example"]
}
```

Valid target types are `graph`, `node`, `edge`, `region`, and `character`.
Every target resolves to an existing object in the supplied graph profile. A
name is metadata; it never becomes a character ID and never changes the graph
digest. The pinned `graph_digest` is the structural digest, so changing only a
probe observation or frontend metadata does not stale a valid annotation.

Each annotation includes at least one verbatim `name`, `description`,
`claimed_function`, or `provenance_statement` from its excerpt. The reader
rejects missing targets, non-verbatim semantics, spans outside the normalized
document, document-digest mismatches, and graph-digest mismatches.

A passage with no validated graph target is stored separately as an
`UnverifiedPaperClaim`. It is never promoted to a graph annotation or
structural character.

## Dates

Dates use `YYYY-MM-DD`. `date_min` and `date_max` represent uncertainty. When a
single release date is known it may populate both bounds. A possible influence
direction requires the source maximum date to be strictly earlier than the
target minimum date.

## Merge behavior

Architectural genomes can merge only when artifact IDs, pinned identity
metadata, and complete graph-profile digests agree. The merge combines
metadata; it does not merge different traces, character radii, or paper artifacts. `PaperGraphAnnotations`
combines validated annotations and unverified claims for one pinned graph and
paper without mutating either input.

## Canonicalization and validation

Canonical JSON uses sorted object keys, stable collection ordering, UTF-8, and
finite JSON values. Digests are recomputed on read. Derived fingerprints,
regions, characters, and annotation targets are also verified rather than
trusted as opaque payloads.

## Related notes

- [[Artifact Profile]]
- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Python API]]
- [[Validate Artifacts]]
