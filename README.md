# Phylodigy

A **phylodigital tree** is a tree-like lineage hypothesis for digital artifacts,
derived from observable structure and evidence. **Phylodigital** is the
adjective; **Phylodigy** is the project and the noun for this line of analysis.

Phylodigy is a Python library and CLI for building graph-derived lineage trees
from observed model artifacts and a shared modelome entry bundle. It compares
executable models as computation graphs to study lineage, convergence, and
contact; entries without observed profiles remain visible as uncovered.

The graph is the genome. There is no built-in architecture ontology, trait
catalog, feature list, family classifier, or regex-to-technique detector.
Operators, tensor flow, structural call/resource attributes, and corpus-derived graph
structures are the primary evidence. Papers may name and describe those
structures after extraction, with exact provenance.

Open the [documentation vault](docs/Start%20Here/Home.md) in Obsidian or read
the Markdown files directly.

## Structural contract

For every executable model, Phylodigy builds a canonical graph profile with:

- open-vocabulary operator nodes;
- directed, position-aware data-flow edges;
- observed static arguments plus nonstructural probe tensor metadata;
- extraction frontend, version, probe, and coverage provenance;
- generic multiscale structural fingerprints;
- graph regions and characters discovered without architecture names.

Unknown operations remain first-class nodes. A frontend does not discard an
operation because the project has never seen it before.

Only executable connectivity, operator identity, static call arguments, and
declared tensor-resource structure form characters. Probe-dependent tensor
metadata and arbitrary public Python object fields are retained only as audit
observations. Neither is phylogenetic evidence.

Generic discovery operates over all graphs in the same way. It may enumerate
rooted neighborhoods, graphlets, paths, single-entry/single-exit regions,
repeated regions, and edit candidates. It does not contain a special detector
for ResNets, Transformers, attention, mixture-of-experts, or any other named
technique.

As a result, a bypass and later addition is discoverable from connectivity,
just like any other branch and reconvergence. “Residual connection” is a paper
annotation on the supporting subgraph, not an extractor-defined trait.

## Paper evidence

Paper processing is annotation, not feature extraction. A structured
annotation targets a concrete graph, node, edge, discovered region, or
graph-derived character and records:

- the paper-supplied label and description;
- the exact normalized text span and document digest;
- citation identities and relation;
- annotator or extraction provenance and confidence.

Annotations never alter graph identity. Several papers may use different
names for one structure, and a structure may remain unnamed. A text passage
without a validated graph target remains an unverified paper claim and cannot
become structural evidence.

Closed-model papers can still be retained as document evidence, but an
unobserved graph remains unobserved. Paper claims do not fabricate a graph.

## Install

```console
python -m pip install -e .
# Optional PDF input:
python -m pip install -e '.[paper]'
# Optional torch.nn.Module / FX inspection:
python -m pip install -e '.[model]'
```

The graph core uses only the Python standard library. Framework frontends are
optional.

## Build a modelome tree

Use a modelome entry bundle as the coverage set, then supply observed graph
profiles for structural tree inference:

```console
phylodigy modelome-tree entries.jsonl --profiles-dir profiles/ -o tree.json
phylodigy modelome-plan entries.jsonl -o plan.json
```

`ENTRIES` can be a bundle directory, JSONL file, or JSON file. Every entry is
retained in coverage, including entries without a profile. Profile artifact
IDs must match entry IDs, or pass `--bindings bindings.json` with an
`entry_id` to `artifact_id` mapping. Names and papers help identify and describe
entries; they are not graph evidence. Exact inference is limited to 100
profiles by default; set `--max-taxa` explicitly to change that limit. The
input bundle verifies its recorded hashes before use.

The Python API exposes the same tree builder:

```python
from phylodigy import build_modelome_tree

tree = build_modelome_tree(
    "entries.jsonl", profiles_dir="profiles", bindings=None, max_taxa=100
)
```

To prepare local upstream data, use the modelome CLI's
`export-entry-seeds` and `build-entry-corpus` commands before running the tree
workflow.

## Build a graph profile

The dependency-free graph API accepts records from any frontend:

```python
from phylodigy import build_graph_profile

profile = build_graph_profile(
    [
        {"index": 0, "name": "x", "op": "placeholder", "target": "x"},
        {
            "index": 1,
            "name": "layer",
            "op": "call_function",
            "target": "framework.example_op",
            "inputs": ["x"],
        },
        {
            "index": 2,
            "name": "out",
            "op": "output",
            "target": "output",
            "inputs": ["layer"],
        },
    ]
)
```

For PyTorch, `extract_architectural_genome(...)` lowers an observed FX trace
into the same graph representation. A single trace is input-specific;
comparative studies should use a versioned probe suite and preserve each
probe's guards and coverage.

## Compare graphs

Fast multiscale fingerprint distance is intended for corpus search, candidate
screening, and alignment anchors:

```python
from phylodigy import compare_graphs

distance = compare_graphs(left_profile, right_profile)
```

Finite-radius fingerprints are graph pseudometrics: distinct graphs can
collide. They are not automatically evolutionary branch lengths.

Primitive indel candidates are a separate artifact:

```python
from phylodigy import align_graphs

alignment = align_graphs(left_profile, right_profile)
validation = alignment.validate_against(left_profile, right_profile)
```

The first indel layer aligns exact structural node labels and emits explicit,
replayable node and edge add/remove operations. It pins both graph digests,
retains unmatched structure, normalizes frontend-declared commutative ports,
and reports separate objective bounds and solver underidentification. The
left-to-right replay direction is not an ancestral gain/loss claim.

This bounded sequence alignment is an auditable candidate, not a general
minimum graph-edit distance or a phylogenetic branch length. Endpoint
validation can prove optimality for the declared primitive objective when its
exact component-count bounds close. Operator substitution, rewiring as a
single event, shape-change events, and region duplication/deletion remain
future hierarchical layers. Pairwise size normalization is not used because it
would erase accumulated operation cost.

Before tree inference, Phylodigy checks triangle inequalities and the
four-point condition. Neighbor joining or weighted least squares is suitable
only when the matrix is approximately additive. Systematic violations remain
visible as signals to investigate convergence, component reuse, or contact
rather than being forced into a bifurcating tree. They do not create a contact
edge: that requires explicit endpoint-grounded provenance, awareness, or
character-transfer evidence.

## Evidence and reproducibility

Every persisted graph object is canonical and content-addressed. Discovery,
comparison, and annotation records bind their input digests and algorithm
parameters. Missing trace coverage is unknown, never absence.

Repository text, configuration files, citations, and papers can remain useful
provenance channels. They do not define structural characters. Architecture
names found in a class, config key, or prose are not accepted as graph facts.

## Scope

This is an auditable research baseline, not proof that one artifact caused
another and not an assertion that software evolves biologically. The intended
result is a tree-like vertical backbone accompanied by separately recorded,
explicit contact evidence and unresolved non-tree-like graph signal.

Start with:

- [Computation Graph](docs/Concepts/Computation%20Graph.md)
- [Graph-Derived Characters](docs/Concepts/Graph-Derived%20Characters.md)
- [Artifact Profile](docs/Concepts/Artifact%20Profile.md)
- [Build a Modelome Tree](docs/Workflows/Build%20a%20Modelome%20Tree.md)
- [Python API](docs/Reference/Python%20API.md)
- [Validation Strategy](docs/Research/Validation%20Strategy.md)

## Try the toy tree

Build a five-artifact example from executable PyTorch models:

```console
python -m pip install -e '.[model]'
phylodigy toy-tree --summary
phylodigy toy-tree -o toy-lineage.json
```

The corpus contains five small `torch.nn.Module` networks with learned linear
layers and nonlinear activations. The command traces each model with the normal
`torch.fx` frontend, then runs the observed graphs through the character-distance,
diagnostic, and neighbor-joining pipeline. Names, dates, and illustrative branch
labels are metadata only and cannot affect the inferred tree.

## Build a popularity corpus

This specialized workflow uses public Hugging Face Transformers artifacts.
For broader source coverage, use the modelome entry-bundle workflow below.

Freeze up to 100 current, public Transformers artifacts ranked by Hugging Face
likes, retain exact arXiv citation evidence separately, then trace the
compatible models without downloading weights:

```console
python -m pip install -e '.[corpus]'
phylodigy rank-models --limit 100 --without-citations -o popular-models.json
phylodigy enrich-model-citations popular-models.json -o citation-snapshot.json
phylodigy infer-popularity-tree popular-models.json \
  --profiles-dir popular-profiles \
  -o popular-tree.json
```

Every candidate is pinned to its Hub revision. Gated repositories, required
custom code, unsupported architectures, trace failures, and oversized graphs
remain explicit exclusions in the run report. Likes and citations select and
describe the corpus but never enter the graph-character matrix or tree distance.

## View a phylodigital tree

A dependency-free web viewer lives in `viewer/`:

```console
phylodigy toy-tree -o viewer/toy-lineage.json   # bundled sample
python3 -m http.server --directory viewer 8457
# open http://localhost:8457/
```

It renders the neighbor-joining phylogram with tree-likeness diagnostics kept
visible, plus the pairwise distance matrix, genome digests, retained external
evidence, and the raw artifact. Drop any `phylodigy.architecture_lineage` JSON
onto the page to inspect it. See
[View a phylodigital tree](docs/Workflows/View%20a%20Phylodigital%20Tree.md).
