---
title: Modelome API
tags:
  - phylodigy/reference
  - modelome
---

# Modelome API

The Modelome API keeps catalog records, observed architectural genome
profiles, and declared catalog relations as separate inputs. Profile identity
is established by exact artifact IDs or an explicit binding. Names, URLs, and
other descriptive fields are never used to guess identity.

```python
from phylodigy import (
    read_modelome_entries,
    load_modelome_profiles,
    bind_modelome_profiles,
    build_modelome_relations,
    plan_modelome_tree,
    build_modelome_tree,
    lineage_to_newick,
)
```

## Read entry corpora

```python
corpus = read_modelome_entries(path)
```

`path` can be a JSON or JSONL file, or a portable bundle directory containing
`entries.jsonl` and `manifest.json`. JSON may be an array, an object containing
an `entries` array, or one entry object. Each entry must have a unique
non-empty `id` and `canonical_name`; known collection fields must be arrays.

The result is a mapping with `entries`, `manifest`, and `source_digest`.
`entries` preserves the input records. For a bundle, `manifest` is the parsed
manifest after checking its format, file SHA-256, entry count, and parsed
content digest. For a standalone file, `manifest` is `None`. In both cases,
`source_digest` is the SHA-256 digest of the input entry-file bytes.
Malformed or unverifiable input raises `ModelomeInputError`, a `ValueError`.

## Load and bind profiles

```python
profiles = load_modelome_profiles(path)
```

`path` can name one profile JSON file or a directory searched recursively for
JSON files. Every profile must be an `ArchitecturalGenome` of artifact kind
`model`. Profiles are keyed by their immutable `artifact_id`; duplicate IDs
and invalid profiles raise an error.

```python
bound = bind_modelome_profiles(entries, profiles, bindings=bindings)
```

Entries supplied here use `id` as their catalog identifier. Profiles
must be a mapping from artifact IDs to matching `ArchitecturalGenome` objects.
An optional `bindings` mapping explicitly maps entry IDs to profile artifact
IDs. Without an explicit mapping, only exact equality between an entry ID and
a profile artifact ID binds automatically.

The result contains:

- `genomes`: bound genomes, with each returned genome's artifact ID set to its
  catalog entry ID for tree use. The source profiles are not changed.
- `coverage`: one record per entry, with `entry_id`, `status` (`bound` or
  `missing_profile`), `profile_id`, and `reason`.
- `unused_profile_ids`: sorted profile IDs that were not bound.

Bindings must refer to known entries and supplied profiles, and a profile may
be assigned to only one entry. Unmatched entries remain in coverage; they are
not treated as evidence of a model architecture.

## Report declared relations

```python
relations = build_modelome_relations(entries)
```

This reports the explicit `model_relations` fields as a graph. A relation is
resolved only when `target.entry_id` exactly matches an input entry `id`;
unmatched targets are retained in `unresolved`. Original relation records are
preserved in each record's `evidence` field. The result has sorted `nodes`,
`edges`, and `unresolved` arrays. These declared links do not by themselves
establish ancestry or determine the inferred tree.

## Plan and build a tree

```python
plan = plan_modelome_tree(entries_path)
result = build_modelome_tree(
    entries_path,
    profiles_dir=None,
    bindings=None,
    max_taxa=100,
    collapse_identical=False,
)
```

Both functions accept the entry path described above. The builder loads and
binds local profiles when `profiles_dir` is supplied, retains entry coverage
and declared relations, and infers from the observed bound profiles.
`max_taxa` is the maximum number of bound profiles accepted for ordinary
inference and must be at least 2. With `collapse_identical=True`, it limits
distinct structural graphs instead.

The build result has top-level `artifact_type`, `status`, `tree`,
`comparisons`, `config`, and `digest` fields. `status` is `inferred` when a
tree can be built and `insufficient_profiles` when fewer than two observed
profiles are available. In the latter case `tree` is `None`. The nested
`modelome` report contains `schema_version`, `source_digest`, `manifest`,
`integrity`, the sorted source `entries`, profile `coverage` and `counts`,
`complete_profile_coverage`, `unused_profile_ids`, `declared_relations`, and
`max_taxa`. Coverage records distinguish bound and unprofiled entries;
unprofiled entries remain unknown and are not interpreted as missing
architecture.

`plan_modelome_tree` returns the same report shape after building the coverage
inventory, with top-level `artifact_type` set to
`phylodigy.modelome_plan` and `status` set to `planned`. It does not load
profiles unless the builder is called with a profile directory; the plan API
accepts only the entry path.

## Serialize a lineage as Newick

```python
newick = lineage_to_newick(lineage)
```

`lineage` is an `infer_lineage_network` result containing a non-empty directed
`tree.edges` sequence. Each edge has `parent`, `child`, and a finite,
non-negative numeric `branch_length`. The serializer walks from the unique
root, emits terminal IDs as labels, sorts siblings deterministically, and
returns a Newick string ending in `;`. Labels that contain characters outside
letters, digits, underscore, period, and hyphen are single-quoted, with
embedded apostrophes doubled. Invalid or disconnected tree structures raise
`ValueError`.

## Compact structural inference

Set `collapse_identical=True` to group profiles with the same exact structural
graph digest before inference. In this mode, `max_taxa` limits the number of
unique graphs. Every bound artifact remains a leaf. The result records each
group in `structural_groups` and maps artifact IDs to representative IDs in
`character_matrix.artifact_representatives`.

This mode uses `neighbor_joining_unique_graphs`, which gives each unique graph
one vote. It is a different estimator from neighbor joining over all artifacts
when distances are non-additive. Tree diagnostics describe representatives.
Four-point diagnostics are `not_assessed` with fewer than four unique graphs.
Exact diagnostics still scale quartically in the unique graph count.

## Infer a compact lineage directly

```python
from phylodigy import infer_compact_lineage

compact = infer_compact_lineage(profiles.values(), max_graphs=100)
```

`profiles` must contain at least two `ArchitecturalGenome` objects with unique
artifact IDs. The method groups only equal structural graph digests. Names,
metadata, and approximate fingerprint similarity do not define groups.
`max_graphs` limits unique graph representatives, not the input artifact
count. The result retains all artifact leaves and compact representative
comparisons and characters.

## Look up original pairwise distances

```python
from phylodigy import lineage_distance_lookup

distance = lineage_distance_lookup(lineage)
raw_distance = distance("artifact:a", "artifact:b")
```

The returned callable expands representative graph distances lazily. It does
not build a full artifact-level distance matrix. Artifacts in one structural
group have distance zero. These are original graph-character distances, not
patristic distances measured along the fitted tree. Unknown artifact IDs raise
`KeyError`.
