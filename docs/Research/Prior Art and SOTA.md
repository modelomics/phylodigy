---
title: Prior Art and State of the Art
aliases:
  - Prior Art and SOTA
  - Model lineage literature
tags:
  - phylodigy/research
  - literature
status: current-as-of-2026-08-29
updated: 2026-08-29
---

# Prior Art and State of the Art

This note summarizes public research available on 2026-08-29. It separates functional similarity from implementation provenance.

No single reviewed system reconstructs the complete lineage history proposed by Phylodigy. Several systems solve important adjacent problems.

## Leading adjacent systems

| Work | Main signal | Result | Provenance scope |
| --- | --- | --- | --- |
| [Unsupervised MoTHer, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/774164b966cc277c82a960934445140d-Abstract-Conference.html) | Model weights | Directed fine-tune tree | Weight heritage among accessible models |
| [PhyloLM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a2e28663712d5a3429a98918c3058f7b-Abstract-Conference.html) | Sampled outputs | Distance and dendrogram | Functional relation across open and closed models |
| [LLM DNA, ICLR 2026 Oral](https://iclr.cc/virtual/2026/oral/10009244) | Functional representation | Similarity tasks and evolutionary tree | Functional inheritance, not verified implementation descent |
| [Neural Lineage, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Neural_Lineage_CVPR_2024_paper.html) | Network representations | Parent prediction | Candidate-parent fine-tune lineage for vision models |

## Weight heritage

Unsupervised MoTHer reconstructs directed fine-tuning trees from weights alone. It formulates recovery as a directed minimum spanning tree problem.

Among the reviewed systems, MoTHer most directly targets parent-child heritage for open-weight models. Its tree model does not represent method transfer or independent reimplementation.

Neural Lineage predicts which candidate parent produced a fine-tuned child. Its evaluated task concerns vision networks and known candidate parent sets.

## Functional phylogenies

PhyloLM converts output similarity into a distance and dendrograms. Its study includes 111 open models and 45 closed models.

This signal supports behavioral proximity and performance prediction. It cannot establish implementation provenance without another evidence channel.

LLM DNA defines a compact representation of functional behavior. Its study evaluates 305 LLMs and constructs trees with phylogenetic algorithms.

The ICLR 2026 program lists LLM DNA as an Oral. Its functional inheritance properties do not prove code reuse or method transfer.

Functional trees remain valuable for closed models. They should remain separate from provenance claims unless independent evidence supports the same edges.

## Architecture and source graphs

Architecture equality provides weak ancestry evidence. The MoTHer paper notes that unrelated foundation models can share one architecture.

[Graph edit distance](https://en.wikipedia.org/wiki/Graph_edit_distance) supplies a general family for structural comparison. Model studies must still define graph normalization, node types, and edit costs.

Source history can provide stronger evidence for exact reuse. The [Software Heritage graph](https://doi.org/10.1145/3379597.3387510) records public source artifacts and development events in a Merkle DAG.

[Clone genealogy](https://doi.org/10.1145/1083142.1083146) traces copied fragments across repository revisions. It cannot observe private code or paper-based reimplementation.

These methods motivate graph profiles and direct provenance. They do not supply a complete method lineage model.

## Contact and transfer inspiration

Biological network methods motivate reticulate structure when one tree cannot
explain all observed characters.

[contacTrees](https://www.nature.com/articles/s41599-022-01211-7) adds directed
contact edges to a dated backbone. Phylodigy borrows only the reticulate-history
analogy; its character matrix is generated from computation graphs.

[Huson and Bryant](https://doi.org/10.1093/molbev/msj030) review phylogenetic networks. [Doolittle](https://doi.org/10.1126/science.284.5423.2124) explains how horizontal transfer can weaken a universal tree.

These sources provide structural analogies. They do not validate digital lineage inference.

## Open research gap

No reviewed system jointly provides these capabilities:

- Canonical executable operator/layer graphs
- Open-vocabulary, graph-derived structural characters
- Paper names bound to validated graph targets without changing structure
- Uncertain dates and temporal admissibility
- A graph-derived vertical phylogeny and separate explicit contact records
- Explicit convergence and unresolved classifications
- Citation and source provenance retained outside structural distance

The current graph-character phylogeny and dated contact network are research
baselines, not established state-of-the-art methods. [[Infer Lineage]] uses a
raw multiscale graph-character pseudometric, four-point diagnostics, and
neighbor joining. [[Infer Network]] applies an explicit time gate to grounded
provenance, awareness, or character-transfer records. Anonymous character
overlap remains diagnostic and cannot create a contact edge.

The [[Profile Comparison API]] does not yet implement general graph edit
alignment, learned event costs, or a probabilistic gain/loss process. Those are
open research steps and must stay generic across the full operator vocabulary.

## Related notes

- [[Research Design]]
- [[Target Corpus]]
- [[Homology and Convergence]]
- [[Time-aware Contact DAG]]
- [[Artifact Profile]]
