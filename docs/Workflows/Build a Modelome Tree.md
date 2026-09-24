---
title: Build a Modelome tree
tags:
  - phylodigy/workflow
  - modelome
status: implemented
---

# Build a Modelome tree

Build a graph-derived tree from a portable Modelome entry corpus and local
architectural genome profiles. The entry corpus defines coverage. Only entries
with observed, bound computation graphs contribute to structural inference.

## Prepare the inputs

In the Modelome repository, export entry seeds and build the portable corpus.
See [Entry construction](https://github.com/modelomics/modelome/blob/main/docs/entries.md) for
the upstream commands and bundle format.

Keep extracted genome JSON files in a local directory. Build each profile from
an observed graph, for example:

```console
phylodigy build-genome trace-records.json \
  --artifact-id model:example \
  --frontend frontend.example \
  -o profiles/model-example.genome.json
```

The artifact ID identifies the observed profile. If it differs from the
Modelome entry ID, provide an explicit binding. Do not infer identity from a
name, paper title, URL, or other descriptive field.

## Review coverage

Plan the operation before building the tree:

```console
phylodigy modelome-plan entries.jsonl -o plan.json
```

`ENTRIES` accepts a portable bundle directory, its `entries.jsonl` file, or a
JSON/JSONL entry input. For a bundle directory, the reader verifies the
manifest format, the entries file SHA-256, the entry count, and the parsed
content digest. A standalone JSON or JSONL file has no bundle manifest, so the
reader computes a digest of its input bytes.

The plan inventories the full entry corpus and declared resources. It does not
read local profiles or extract graphs. Every entry is therefore listed as
unprofiled in this planning report. That means its graph is unobserved; it does
not mean the model lacks a structure.

## Bind observed graphs

Matching entry and profile IDs bind automatically by exact string equality.
Use `--bindings` when IDs differ. The JSON file maps each entry ID to one
profile artifact ID:

```json
{
  "modelome-entry:42": "hub:org/model@revision"
}
```

Pass the mapping to tree construction:

```console
phylodigy modelome-tree entries.jsonl \
  --profiles-dir profiles/ --bindings bindings.json -o tree.json
```

Every binding must name an entry in the corpus and a profile in the supplied
directory. A profile can bind to at most one entry. Unbound entries remain in
coverage without a tree tip. Names and paper evidence describe entries; they
do not contribute graph characters or distances.

## Build the tree

```console
phylodigy modelome-tree entries.jsonl --profiles-dir profiles/ -o tree.json
```

Write a Newick tree alongside the JSON report with `--newick`:

```console
phylodigy modelome-tree entries.jsonl --profiles-dir profiles/ \
  --newick tree.nwk -o tree.json
```

Newick output requires an inferred tree, so at least two entries must have
bound profiles. JSON and Newick outputs must use different paths. Use `-` for
either output to write it to standard output, but not both at once.

The builder derives structural characters from the bound computation graphs
and infers a tree from those characters. Entry metadata and Modelome
relationships remain supporting evidence and do not change graph distances.
The tree artifact retains coverage, profile digests, bindings, and inference
diagnostics so readers can distinguish observed graph evidence from unknown
entries.

Exact tree inference accepts at most 100 bound profiles by default. Set
`--max-taxa` to another integer of at least 2 when the corpus requires it. The
limit applies to observed, bound profiles, not to the number of Modelome
entries. Inference uses the full eligible set up to that limit; it does not
silently sample or truncate taxa. Exact tree diagnostics scale quartically in
the number of profiles, which is why the default limit is 100. Increase it
only when the required computation is acceptable.

The Python API exposes the same operation:

```python
from phylodigy import build_modelome_tree, plan_modelome_tree

plan = plan_modelome_tree("entries.jsonl")
tree = build_modelome_tree(
    "entries.jsonl",
    profiles_dir="profiles",
    bindings={"modelome-entry:42": "hub:org/model@revision"},
    max_taxa=100,
)
```

Return to [[Home]].
