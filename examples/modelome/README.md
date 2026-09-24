# Synthetic Modelome fixture

This portable bundle contains three toy entries and two synthetic architectural
profiles. `toy:model:unobserved` intentionally has no profile so the tree output
shows partial coverage. `toy:model:alpha` has an explicit `derived_from`
relation targeting `toy:model:beta`. All data is synthetic and describes no
real model or registry record.

From the repository root, run:

```console
phylodigy modelome-tree examples/modelome/entries --profiles-dir examples/modelome/profiles -o modelome-tree.json
```
