# Top-100 popularity corpus run

Snapshot: `2026-08-31T01:03:48+00:00`

Selection population: public Hugging Face model artifacts tagged for
Transformers. Candidates were sorted by likes descending, with repository ID
as the deterministic tie-breaker, and pinned to exact Hub revisions. Likes and
arXiv identifiers are external evidence only; neither enters graph distance.
This is a reproducible popularity snapshot, not a claim that one universal
"most important models of all time" ranking exists.

## Result

- candidates: 100
- executable graph profiles: 3
- explicit exclusions: 97
- character radii: `[0]`
- tree method: neighbor joining
- tree metric within tolerance: true
- manifest digest: `689888c6934b4edc246386c174093061dc35919d95cc651aa10b727f57595814`
- run digest: `aa910dcda60270c51f8f0b20142fdc26c419bd48e7ea157033db373b4f25f814`

```text
                         /-- GPT-J-6B (rank 97)                : 3109
             /----------+
            /            \-- BERT-base-uncased (rank 35)      : 420
-----------+
            \--------------- all-MiniLM-L6-v2 (rank 8)        : 171
```

The internal branch between the BERT/GPT-J join and the final join has length
171. Branch lengths are unnormalized graph-character distances, not elapsed
time or causal claims.

## Citation evidence

The separate Semantic Scholar snapshot resolves the exact arXiv tags published
on the selected Hub repositories:

- tagged model artifacts: 72
- unique arXiv identifiers: 117
- resolved papers: 114
- unresolved identifiers: 3
- citation snapshot: `2026-08-31T01:29:26+00:00`
- citation digest: `a87b7b4faaedae7f3e08111aaf5f14182275d4889db9fc0c9861ddd12d2a5f8d`

An arXiv tag can identify a model paper, dataset paper, benchmark, or supporting
method. Consequently these counts are retained as source evidence and are not
silently treated as model-level citation scores.

## Exclusions

| Category | Count |
| --- | ---: |
| Graph extraction incompatibility | 30 |
| Required custom code | 21 |
| Gated artifact | 19 |
| Declared layer limit | 17 |
| Per-model timeout | 6 |
| Other configuration/dependency failure | 4 |

The low extraction yield is retained as a result, not repaired with synthetic
graphs or configuration-derived features. See `top-100-tree-run.json` for every
ranked exclusion and `profiles/` for successful canonical genomes.
