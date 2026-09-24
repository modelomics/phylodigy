---
title: Citations as Evidence
aliases:
  - Citation evidence
tags:
  - phylodigy/concept
  - citations
  - evidence
status: experimental
updated: 2026-08-29
---

# Citations as Evidence

A citation can support awareness, attribution, or a claimed source. It cannot
prove that a model implements a structure and cannot create a graph character.

## Resolution

Bibliography resolution prefers exact DOI, arXiv, and URL identity, then
conservative title or author/year matches. Ambiguous entries remain unresolved.
Corpus profile IDs are preserved as citation endpoints.

## Local graph binding

Contact evidence is strongest when a citation occurs inside a paper annotation
whose exact passage is bound to a concrete graph target. The record retains the
document digest, normalized offsets, citation marker, bibliography resolution,
graph digest, and target ID.

If a passage has no validated graph target, the citation remains document-level
awareness or provenance. A phrase match alone is never an implementation or
transfer observation.

## Historical role

An authors' statement that they adopted or extended a cited component can
support a contact hypothesis for the annotated graph region. Comparison,
background, negation, and generic context record awareness only.

Even an adoption claim must remain independent from structural distance. The
graph establishes what is similar; the citation is evidence about why.

## Limits

Authors cite for many reasons, omit influential sources, and place markers
ambiguously. Code, people, releases, and shared infrastructure can transmit
components without a citation. Treat citations as one auditable channel, not a
causal verdict.

## Related notes

- [[Graph-Derived Characters]]
- [[Evidence Channels]]
- [[Time-aware Contact DAG]]
- [[Citation Workflow]]
- [[Assumptions and Limits]]
