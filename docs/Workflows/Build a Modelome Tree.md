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

## Plan and extract public profiles

Phylodigy can plan and run bounded extraction from public Hugging Face
Transformers repositories. Install the optional `corpus` extra first. Planning
is offline and checks the exact repository and revision evidence already
recorded in each entry:

```console
python -m pip install -e '.[corpus]'
phylodigy modelome-extract-plan entries.jsonl -o extraction-plan.json
```

The resolver accepts only exact `repo_id` plus 40-character commit revision
identifiers. If an entry has multiple evidenced candidates, select one with a
JSON `--pins` mapping. A pin can select only a candidate already present in
that entry's evidence:

```json
{
  "modelome-entry:42": {
    "repo_id": "org/model",
    "revision": "0123456789abcdef0123456789abcdef01234567"
  }
}
```

The plan accepts the same optional `--pins JSON_FILE` mapping used by
extraction. It records entries without an exact target as missing, unsupported,
or ambiguous instead of guessing from names or URLs.

Run a bounded batch with explicit network access:

```console
phylodigy modelome-extract entries.jsonl \
  --output-dir traces --max-models 10 \
  -o extraction-run.json
```

Pass `--pins pins.json` when extraction needs those selections. Add
`--retry-failures` to retry prior failures, or
`--per-model-timeout SECONDS` to change the timeout. Repeating the same command
uses the same entry snapshot, pins, extraction policy, and runtime-specific run
directory. It reuses validated cached results and tries the next ten uncached
ready jobs. The JSON report's `profiles_dir` points to the profile directory
for that snapshot, policy, and runtime. Pass that reported path to
`modelome-tree`; do not substitute the general `traces` parent directory.

Extraction constructs supported models on the meta device. It does not
download weights or run repository supplied custom code. It refuses gated or
private repositories and applies a 45-second per-model timeout by default.
Change it with `--per-model-timeout`. Missing or unsupported references,
ambiguous candidates, extraction failures, and unattempted entries remain in
the report. A `finished` run means there are no pending ready jobs; it does not
mean every Modelome entry has graph coverage.

The Python API composes extraction with tree construction using the returned
profile directory. Run this code from a Python script with the main guard
because extraction starts worker processes with the `spawn` method. Do not run
the extraction call directly in a REPL.

```python
from phylodigy import build_modelome_tree, extract_modelome_profiles

if __name__ == "__main__":
    run = extract_modelome_profiles(
        "entries.jsonl", "traces", max_models=10
    )
    tree = build_modelome_tree(
        "entries.jsonl", profiles_dir=run["profiles_dir"]
    )
```

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
  --profiles-dir profiles/ --bindings bindings.json \
  --collapse-identical -o tree.json
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

Use `--collapse-identical` when many entries have exactly identical observed
graphs. It groups only equal structural graph digests. It does not merge by
name, metadata, or approximate fingerprint. Every bound artifact remains a
separate tree leaf, while graph comparisons and characters use one
representative per structural group. The `structural_groups` report and
`artifact_representatives` map preserve the reversible grouping. With this
option, `--max-taxa` limits unique graphs, not the artifact count.

The compact method is `neighbor_joining_unique_graphs`: each unique graph has
one vote. This is a different estimator from neighbor joining over every
artifact when distances are non-additive. Its exact diagnostics still scale
quartically in the number of unique graphs. Diagnostics apply to structural
representatives; four-point assessment is `not_assessed` with fewer than four
unique graphs. The viewer shows the representative distance matrix and all
artifact tree tips.

The Python API exposes the same operation:

```python
from phylodigy import build_modelome_tree, plan_modelome_tree

plan = plan_modelome_tree("entries.jsonl")
tree = build_modelome_tree(
    "entries.jsonl",
    profiles_dir="profiles",
    bindings={"modelome-entry:42": "hub:org/model@revision"},
    collapse_identical=True,
    max_taxa=100,
)
```

See [[Modelome API]] for compact lineage and pairwise distance functions.

Return to [[Home]].
