---
title: Closed Model Workflow
tags:
  - phylodigy/workflow
  - evidence
---

# Closed model workflow

A paper can document a closed model, but it cannot substitute for an
unobservable computation graph.

## 1. Pin the document and artifact

Record the model identity, public date interval, paper version, external
identifiers, normalized text, and content digest.

```console
phylodigy prepare-paper paper.pdf \
  --artifact-id paper:example \
  -o paper-example.document.json
```

## 2. Retain document claims

Store passages, descriptions, and citations as document evidence. If no graph
is available, do not create graph targets, structural characters, or graph
distances from those phrases.

## 3. Add a graph only when observable

An API endpoint, exported graph, independently archived model, or compatible
runtime trace may later provide direct structural evidence. Pin its frontend,
probe, and artifact identity before linking paper annotations.

## 4. Report the limitation

Paper-only models can participate in document and contact analyses. They must
remain excluded from graph-only distance trees until a compatible graph exists.
This is preferable to assigning a graph from a named architecture family.

## Related notes

- [[Artifact Profile]]
- [[Evidence Channels]]
- [[Citations as Evidence]]
- [[Assumptions and Limits]]
