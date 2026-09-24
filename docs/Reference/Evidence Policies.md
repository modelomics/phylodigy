---
title: Evidence Policies
tags:
  - phylodigy/reference
  - evidence
---

# Evidence policies

Structural evidence and explanatory evidence have different roles. A policy
must not combine them into a manually weighted architecture feature vector.

## Structural comparison policy

Only compatible graph profiles contribute to structural characters,
fingerprints, graph alignments, and phylogenetic distance.

Compatibility checks frontend, lowering scheme, operator schema, probe and
guard coverage, observed-attribute policy, and discovery version. A failed
check yields an explicit missing or incompatible result.

Paper labels, class names, configuration keys, and citations have zero weight
inside structural distance.

Primitive alignment likewise uses only operation identity, structural
attributes, directed input-position edges, and declared alignment parameters.
It excludes paper annotations, artifact names, source symbols, and probe
observations. Its bounded candidate cost remains separate from both the
graph-character distance and historical gain/loss interpretation.

## Multiple graph observations

When several traces exist for an artifact, a study must choose and record a
combination rule. Valid choices include comparing matching probes separately,
using a guarded trace union, or reporting a range across compatible probes.

Do not silently choose whichever trace makes a pair most similar. Do not
renormalize away unmatched coverage.

## Paper annotation policy

An annotation is accepted only when:

- its document digest and normalized span validate;
- its graph digest matches the target profile;
- its target ID and type resolve;
- its label and description are retained as evidence, not identity.

Conflicting labels and descriptions coexist. Non-mention is unknown.

## Citation and contact policy

A citation may support awareness, attribution, or a contact hypothesis. A local
adoption or extension claim bound to a graph target is stronger than generic
context, but neither changes graph distance.

Comparison, background, negated, ambiguous, and unresolved citations remain
diagnostic. Duplicate provenance must be deduplicated before combining contact
evidence.

## Recurrence and convergence

Do not assign `exact`, `rare`, or `recurrent` to named techniques by hand.
Estimate recurrence from discovered-character prevalence, fitted gain/loss
models, and independently documented histories. Store the estimator and frozen
corpus digest.

## Sensitivity analysis

Vary graph discovery parameters, probe subsets, frontend lowering, edit-cost
fits, annotation inclusion, dates, and contact evidence independently. Report
which tree splits and network signals survive.

## Related notes

- [[Evidence Channels]]
- [[Computation Graph]]
- [[Graph-Derived Characters]]
- [[Citations as Evidence]]
- [[Validation Strategy]]
