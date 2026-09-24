---
title: Infer Network
tags:
  - phylodigy/workflow
  - network
---

# Infer network

Use the contact view to retain independently evidenced relationships alongside
a separately inferred graph-character phylogeny.

## Inputs

Start from dated architectural genomes and independently collected provenance,
awareness, or graph-grounded character-transfer records. The graph-derived
lineage, quartet/tree-fit residuals, and graph alignments can guide
investigation, but cannot become contact evidence by themselves.

The implemented baseline runs directly from dated architectural genomes:

```console
phylodigy infer-network model-a.genome.json model-b.genome.json \
  --edge-evidence contact-evidence.json \
  -o contact-network.json
```

Omit `--edge-evidence` to generate graph pair diagnostics only. The resulting
edge list and parent forest will be empty.

## Evidence-grounded edges

Every record must identify represented source and target genomes. Provenance or
awareness evidence grounds the endpoints; character-transfer evidence must also
identify an anonymous character derived from both graphs. The source's latest
possible date must be strictly earlier than the target's earliest possible
date.

Graph-region similarity, shared characters, edit alignment, paper wording, or
a shared architecture name alone never creates an edge. Do not assign a manual
recurrence prior to decide between transfer and convergence.

## Output

Retain explicit contact edges, rejected directions, pairwise graph diagnostics,
and character-origin hypotheses. Characters default to `first_observed`; only
admitted character-transfer evidence changes an occurrence to `transfer`.
Vertical history remains in the separate lineage artifact. The contact network
is an evidence record, not a causal verdict.

## Related notes

- [[Infer Lineage]]
- [[Time-aware Contact DAG]]
- [[Contact Network Schema]]
- [[Citations as Evidence]]
- [[Homology and Convergence]]
