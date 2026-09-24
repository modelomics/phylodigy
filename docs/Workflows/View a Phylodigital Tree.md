---
title: View a phylodigital tree
aliases:
  - Tree viewer
tags:
  - phylodigy/workflow
  - phylodigy/viewer
status: implemented
---

# View a phylodigital tree

The `viewer/` directory contains a dependency-free web viewer for
`phylodigy.architecture_lineage` artifacts. It renders the inferred
neighbor-joining backbone as a phylogram and keeps diagnostics, pairwise
distances, genome digests, and retained external evidence visible beside it.

## Open the viewer

No build step and no packages are required. Any static server works:

```console
python3 -m http.server --directory viewer 8457
# then visit http://localhost:8457/
```

Opening `viewer/index.html` directly from disk also works; the bundled toy
tree loads from an embedded copy when `fetch` is unavailable over `file://`.

## Load an artifact

The viewer opens the bundled toy demonstration on start
(`phylodigy toy-tree -o viewer/toy-lineage.json`). To inspect another lineage:

- drop a `phylodigy.architecture_lineage` JSON record anywhere on the page, or
- use **Open lineage JSON…** to pick a file.

Produce a record with [[Infer Lineage]]:

```console
phylodigy infer-lineage *.profile.json -o lineage.json
```

## Read the panels

- **Tree** renders a phylogram. Horizontal distance to the display root is the
  cumulative graph-character branch length. The root marker is the final
  neighbor-joining join; the inferred backbone itself is unrooted. Internal
  nodes are unnamed hypothetical ancestors. Click a node (or focus it and
  press Enter) for its provenance: parent edge, raw and clamped branch
  length, dates, and content digests.
- The sidebar shows the tree-likeness diagnostics from the artifact. Metric or
  four-point violations stay listed under *Signals to investigate*; they are
  possible convergence, component reuse, or contact signals and are not
  forced into a bifurcating tree. Negative branch lengths clamped during
  neighbor joining are drawn dashed and labeled.
- **Distances** shows the full pairwise weighted-L<sub>1</sub> matrix over
  dynamic graph characters.
- **Genomes** lists artifact ids, date ranges, and content-addressed digests.
- **Evidence** shows the structural evidence boundary and any external
  chronology, citation, or provenance records. These records are retained
  beside the tree and never inside its distances.
- **Raw JSON** is the unmodified artifact, for auditing.

## Regenerate the sample

```console
phylodigy toy-tree -o viewer/toy-lineage.json
```

## Test the layout logic

The viewer's DOM-free layout logic (`viewer/layout.js`) is pinned by Node
tests:

```console
node --test tests/
```

Return to [[Home]].
