---
title: Open Model Workflow
tags:
  - phylodigy/workflow
  - graph
---

# Open model workflow

Use an executable framework graph as the structural source for an open model.

## 1. Pin the artifact

Record the immutable release, repository commit, model digest, public date or
date interval, framework version, and environment required to trace it.

## 2. Define probes

Create a versioned probe suite that exercises relevant input shapes and known
control-flow guards. Do not infer absence from a single successful trace.

## 3. Extract graphs

Lower each observed framework graph into the canonical computation graph. Keep
unknown and opaque operations, nested argument positions, observed static
arguments, tensor metadata, trace failures, and coverage provenance.

Source files and configuration may be captured for provenance and reproducible
probe construction. They are not scanned for architecture names or converted
into graph characters.

For generic frontend records:

```console
phylodigy build-genome trace-records.json \
  --artifact-id model:example \
  --frontend frontend.example \
  -o model-example.genome.json
```

For a live PyTorch module, use `extract_architectural_genome` from the Python
API. `extract-code` can create a separate `SourceManifest`; its result cannot
be passed to graph comparison.

## 4. Discover characters

Run the same generic region, graphlet, path, neighborhood, and repetition
algorithms used for the whole corpus. Freeze their versions and parameters.
Review supporting nodes and edges, not human labels.

## 5. Add paper meaning

Store the model paper as a normalized document. Attach labels and descriptions
only to graph targets that validate against the profile. Keep exact text spans,
citations, confidence, synonyms, and conflicts.

## 6. Validate

Recompute digests and all derived graph fields. Check probe compatibility and
disclose opaque or uncovered regions before comparison.

```console
phylodigy validate model-example.genome.json
```

## Related notes

- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Artifact Profile]]
- [[Citation Workflow]]
- [[Validate Artifacts]]
