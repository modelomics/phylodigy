---
title: Documentation Style
tags:
  - phylodigy
  - contributing
  - style
---

# Documentation style

This vault uses controlled English inspired by ASD-STE100.

The project does not claim certified ASD-STE100 compliance. This repository does not bundle the official ASD-STE100 dictionary.

The [ASD-STE100 writing skill](https://github.com/danyuchn/asd-ste100-skill/blob/master/SKILL.md) defines the local rules.

## Core rules

- Use active voice.
- Give one instruction in each sentence.
- Limit each procedure sentence to 20 words.
- Limit each descriptive sentence to 25 words.
- Keep each paragraph focused on one topic.
- Limit each paragraph to six sentences.
- Replace multiword verbs with precise single-word verbs.
- Keep noun groups short.
- Do not use semicolons.
- Preserve all technical qualifications and uncertainty.

## Modal verbs

- Use `must` for an obligation.
- Use `can` for capability.
- Use `may` for permission or possibility.
- Use `should` only for a recommendation.

Do not replace a qualified claim with a certain claim.

## Project terms

| Preferred term | Meaning |
| --- | --- |
| artifact | A model, paper, repository, or configuration in the corpus |
| artifact profile | Graph profiles, documents, annotations, and provenance for one artifact |
| graph profile | Canonical observed operator/layer graph plus derived structural data |
| structural character | Anonymous content-derived graph structure |
| graph annotation | Paper label or description bound to an existing graph target |
| evidence channel | The origin of an observation |
| phylodigital | The adjective for digital-artifact lineage analysis, as in “phylodigital tree” |
| Phylodigy | The project name and noun for this line of analysis |
| contact DAG | The directed graph of time-admissible hypotheses |
| source | An earlier artifact that can explain a target structure |
| target | The artifact or graph object under analysis |
| awareness evidence | A citation that records awareness without lineage weight |

Use one preferred term for each concept. Define a new term before its first use.

## Procedures

1. Verify every technical claim against source code or tests.
2. Select the preferred terms for the note.
3. Write the shortest complete procedure.
4. Retain warnings, conditions, and limitations.
5. Check every wiki link.
6. Run the project tests after command changes.

## Obsidian conventions

- Add YAML properties to every note.
- Use a unique filename for every note.
- Use Obsidian wiki links for internal references.
- Use callouts for warnings and important limits.
- Use Mermaid only when a diagram improves understanding.
- Avoid community plug-in requirements.

See [[Testing]] for the required project checks.
