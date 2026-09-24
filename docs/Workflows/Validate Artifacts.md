---
title: Validate Artifacts
aliases:
  - Validate profiles
tags:
  - phylodigy/workflow
  - phylodigy/validation
status: implemented
---

# Validate Artifacts

The `validate` command checks artifact profile schemas. It verifies a supplied content digest.

It does not validate contact networks or edge evidence files.

## Print a validation report

Validate one or more profiles.

```console
phylodigy validate \
  model-a.profile.json \
  model-b.profile.json
```

The command writes a JSON report to standard output. Each file record contains its status and verified identity fields.

## Save the report

```console
phylodigy validate \
  model-a.profile.json \
  model-b.profile.json \
  -o validation-report.json
```

## Use status-only validation

Use quiet mode in automated checks.

```console
phylodigy validate --quiet model-a.profile.json
```

Quiet mode emits no report. Inspect the process exit status instead.

| Exit status | Meaning |
|---|---|
| `0` | Every profile is valid |
| `1` | At least one profile failed validation |
| `2` | The command or input caused an expected error |

## Resolve a digest failure

1. Do not replace the stored digest manually.
2. Identify the changed source or extraction input.
3. Regenerate the affected profile.
4. Repeat the validation command.

Any content edit can change the canonical digest. [[Artifact Profile Schema]] describes the protected content.

## Validate before inference

Validate every merged profile before [[Infer Lineage]] or [[Infer Network]].
Retain reports when reproducibility or review requires them.
