---
title: Hugging Face Dataset Layout
aliases:
  - Columnar artifact layout
  - Parquet dataset layout
tags:
  - phylodigy/reference
  - parquet
  - graphar
  - hugging-face
status: active
updated: 2026-09-01
---

# Hugging Face Dataset Layout

Phylodigy persists searchable scientific data as typed Apache Parquet tables.
Apache Arrow is the in-memory representation used to build, filter, and read
those tables. Arrow IPC files are caches or transport artifacts, not the
canonical published dataset.

The layout has two simultaneous goals:

1. Each table must work directly in the Hugging Face Dataset Viewer.
2. Graph topology must remain a first-class property graph rather than an
   opaque serialized value.

## Repository layout

```text
phylodigy-data/
├── README.md
├── manifest.toml
├── data/
│   ├── models/
│   ├── papers/
│   ├── model_paper_links/
│   ├── observations/
│   ├── graph_vertices/
│   ├── graph_edges/
│   ├── graph_characters/
│   ├── character_states/
│   ├── phylogenies/
│   ├── tree_edges/
│   ├── distances/
│   └── snapshots/
├── graphar/
│   ├── architectures.graph.yml
│   └── phylogenies.graph.yml
└── exports/
    └── newick/
```

Every directory under `data/` contains one or more Parquet shards. The
dataset card maps each directory to a separate Hugging Face configuration and
uses `models` as the default configuration.

The GraphAr information files describe the graph vertex and edge tables. They
do not create a second copy of the graph data.

## Core tables

| Configuration | Row meaning | Primary identity |
| --- | --- | --- |
| `models` | One current provider-local model, release, artifact, or offering | `model_id` |
| `observations` | One immutable provider observation | `observation_id` |
| `papers` | One resolved publication identity | `paper_id` |
| `model_paper_links` | One evidence-bearing model-to-paper relationship | `model_id`, `paper_id`, `role` |
| `graph_vertices` | One operation vertex in one computation graph | `graph_id`, `node_id` |
| `graph_edges` | One position-aware directed data-flow edge | `graph_id`, `source_id`, `target_id`, `position` |
| `graph_characters` | One discovered anonymous structural character | `graph_id`, `character_id` |
| `character_states` | One nonzero model-by-character count | `lineage_id`, `model_id`, `character_id` |
| `phylogenies` | One inferred tree and its summary diagnostics | `tree_id` |
| `tree_edges` | One branch in one inferred tree | `tree_id`, `parent_id`, `child_id` |
| `distances` | One unordered pairwise structural distance | `lineage_id`, `left_id`, `right_id` |
| `snapshots` | One provider query and its coverage result | `snapshot_id` |

The normalized tables are the searchable interface. Large or irregular source
payloads may be retained in canonical CBOR binary columns, but important
identity, date, access, citation, organization, domain, task, operation, and
topology fields must also exist as typed columns.

## Graph representation

A computation graph is a property graph stored as two normalized tables.

`graph_vertices` contains at least:

- `graph_id`
- `model_id`
- `graph_digest`
- `node_id`
- `operation`
- `structural_attributes_cbor`
- `observations_cbor`
- `search_text`

`graph_edges` contains at least:

- `graph_id`
- `source_id`
- `target_id`
- `position`
- `edge_id`

The source, destination, and input position preserve repeated inputs and
noncommutative argument order. Stable row ordering is part of canonical export,
but row order is never used as graph structure.

The physical layout follows Apache GraphAr's property-graph conventions:
typed vertex and edge properties, Parquet chunks, stable identifiers, and
adjacency-oriented edge ordering. Phylodigy does not require a GraphAr runtime
to read the tables.

## Tree representation

The canonical persisted tree is also normalized rather than embedded as a
recursive value. `tree_edges` contains:

- `tree_id`
- `parent_id`
- `child_id`
- `branch_length`
- `raw_branch_length`
- `length_clamped`
- `child_is_taxon`

The parent/child orientation is a deterministic serialization root. It does
not assert that an unrooted neighbor-joining tree has a biological root.

`phylogenies` retains the inference method, distance scheme, tolerances,
additivity diagnostics, negative-limb diagnostics, and semantic digest.
`character_states` and `distances` retain the inputs needed to audit or rebuild
the result.

Newick is an interoperability export only. It cannot replace the Parquet
tables because it cannot preserve the character matrix, distance audit,
provenance, or most branch diagnostics.

## Parquet conventions

- Use Zstandard compression.
- Use dictionary encoding for repeated categorical strings.
- Write page indexes and bounded row groups for Dataset Viewer filtering.
- Use UTC timestamp logical types and `date32` for calendar dates.
- Sort each shard by its declared primary identity before writing.
- Put schema name, schema version, and semantic digest in Parquet metadata.
- Partition provider observations by provider and observation date.
- Keep large binary payloads out of the first/default `models` table.

Semantic digests cover normalized typed content. They do not depend on bytewise
Parquet output, compression level, row-group boundaries, or PyArrow version.

## Hugging Face configurations

The dataset card declares one configuration per table. A typical entry is:

```yaml
configs:
  - config_name: models
    default: true
    data_files:
      - split: full
        path: data/models/*.parquet
  - config_name: graph_edges
    data_files:
      - split: full
        path: data/graph_edges/*.parquet
```

The built-in Dataset Viewer supplies the first searchable interface. A
Phylodigy Space can use its search and filter endpoints to find models, then
retrieve the selected model's vertex, edge, paper, character, distance, and
tree rows for interactive rendering.

## Structural evidence boundary

Registry popularity, citation counts, names, paper text, and provider metadata
may select records and help users search. They cannot create graph characters,
change structural distances, or change inferred topology.
