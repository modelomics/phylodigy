---
title: Install
aliases:
  - Installation
tags:
  - phylodigy/workflow
  - phylodigy/install
status: implemented
---

# Install

Phylodigy requires Python 3.10 or later. The core package uses only the Python standard library.

## Install the core package

1. Open a terminal in the repository root.
2. Create an isolated environment.
3. Activate the environment.
4. Install the editable package.

```console
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Install optional packages

Install only the packages that your workflow requires.

| Need | Command | Package |
|---|---|---|
| PDF text extraction | `python -m pip install -e '.[paper]'` | `pypdf>=4` |
| Live PyTorch inspection | `python -m pip install -e '.[model]'` | `torch>=2.1` |
| Pytest development tests | `python -m pip install -e '.[dev]'` | `pytest>=8` |

The command interface does not require PyTorch. The [[Python API]] provides the live module extractor.

## Verify the installation

1. Print the installed version.
2. Display the available commands.

```console
phylodigy --version
phylodigy --help
```

Continue with [[Open Model Workflow]] or [[Closed Model Workflow]]. See [[Testing]] for repository checks.
