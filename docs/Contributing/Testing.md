---
title: Testing
tags:
  - phylodigy
  - contributing
  - testing
---

# Testing

Run these checks from the repository root.

## Unit tests

```console
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The environment may skip tests that require optional PyTorch support.

The graph suite must include name-blind discovery tests, unknown-operator
tests, canonicalization invariants, fingerprint collision cases, and paper
annotation target validation. A test must not establish a fixed architecture
feature or accepted-state catalog as a public contract.

## Syntax checks

```console
python3 -m compileall -q src tests
```

## Vault checks

1. Open `docs` as an Obsidian vault.
2. Open [[Home]].
3. Inspect the graph view for unresolved note links.
4. Test every changed command against [[CLI Reference]].
5. Confirm that new notes follow [[Documentation Style]].

Return to [[Home]].
