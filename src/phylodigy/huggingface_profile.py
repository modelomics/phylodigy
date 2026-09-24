"""Extract graph profiles from public, revision-pinned Hugging Face models."""

from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from typing import Any

from .model_profile import extract_architectural_genome
from .schema import ArchitecturalGenome


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_REPO_ID_RE = re.compile(r"^(?:\b[\w.-]+\b/)?\b[\w.-]{1,96}\b$")


def _valid_huggingface_repo_id(value: Any) -> bool:
    """Apply the dependency-free equivalent of Hub ``validate_repo_id``."""

    if not isinstance(value, str) or not _REPO_ID_RE.fullmatch(value):
        return False
    return "--" not in value and ".." not in value and not value.endswith(".git")


class HuggingFaceProfileError(RuntimeError):
    """The Hub artifact cannot be safely resolved or graph-profiled."""


def _hub_model_info(repo_id: str, revision: str) -> Any:
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise HuggingFaceProfileError(
            "Hugging Face extraction requires the 'phylodigy[corpus]' extra"
        ) from exc
    return hub.model_info(repo_id, revision=revision, token=False)


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "huggingface_hub"):
        module = importlib.import_module(name)
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def extract_huggingface_genome(
    repo_id: str,
    revision: str,
    *,
    artifact_id: str,
    max_graph_nodes: int = 12000,
    max_hidden_layers: int = 36,
    radii: Sequence[int] = (0, 1, 2, 3),
) -> ArchitecturalGenome:
    """Trace a public HF Transformers model at an exact Git commit.

    Verification and configuration lookup use the pinned repository revision.
    The existing popularity frontend constructs weights on the PyTorch meta
    device and disables remote code, so model weight files are not loaded.
    """

    if not _valid_huggingface_repo_id(repo_id):
        raise ValueError("repo_id must be a valid Hugging Face repository ID")
    if not isinstance(revision, str) or not _COMMIT_RE.fullmatch(revision):
        raise ValueError("revision must be an exact 40-character Git commit hash")
    revision = revision.lower()
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise ValueError("artifact_id must be a non-empty string")
    for field, value in (
        ("max_graph_nodes", max_graph_nodes),
        ("max_hidden_layers", max_hidden_layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    from .lineage import LineageAnalysisConfig

    selected_radii = LineageAnalysisConfig(radii=tuple(radii)).radii

    try:
        info = _hub_model_info(repo_id, revision)
    except HuggingFaceProfileError:
        raise
    except Exception as exc:
        raise HuggingFaceProfileError(
            f"could not verify public Hugging Face model {repo_id}@{revision}: {exc}"
        ) from exc
    actual_sha = getattr(info, "sha", None)
    if actual_sha != revision:
        raise HuggingFaceProfileError(
            f"Hub returned revision {actual_sha!r} for requested commit {revision!r}"
        )
    if getattr(info, "gated", False):
        raise HuggingFaceProfileError(
            "gated Hugging Face repositories are not supported"
        )
    if getattr(info, "private", False):
        raise HuggingFaceProfileError(
            "private Hugging Face repositories are not supported"
        )

    # This helper deliberately accepts only the safe, verified entry policy.
    from .popularity_corpus import _model_from_entry

    entry = {
        "artifact_id": artifact_id,
        "gated": False,
        "hub_id": repo_id,
        "hub_revision": revision,
        "requires_custom_code": False,
    }
    try:
        model, dummy_inputs, architecture = _model_from_entry(
            entry, max_hidden_layers=max_hidden_layers
        )
        genome = extract_architectural_genome(
            model,
            artifact_id=artifact_id,
            name=repo_id,
            example_kwargs=dummy_inputs,
            propagate_shapes=False,
            radii=selected_radii,
        )
    except Exception as exc:
        raise HuggingFaceProfileError(
            f"could not extract graph for {repo_id}@{revision}: {exc}"
        ) from exc

    node_count = len(genome.graph_profile.graph.nodes)
    if node_count > max_graph_nodes:
        raise HuggingFaceProfileError(
            f"observed graph has {node_count} nodes; limit is {max_graph_nodes}"
        )
    metadata = dict(genome.metadata)
    metadata["huggingface"] = {
        "architecture": architecture,
        "repo_id": repo_id,
        "revision": revision,
        "runtime_versions": _runtime_versions(),
        "weights_loaded": False,
        "trust_remote_code": False,
    }
    return ArchitecturalGenome(
        artifact_id=genome.artifact_id,
        graph_profile=genome.graph_profile,
        artifact_kind=genome.artifact_kind,
        name=genome.name,
        release_date=genome.release_date,
        date_min=genome.date_min,
        date_max=genome.date_max,
        extractor_name=genome.extractor_name,
        extractor_version=genome.extractor_version,
        metadata=metadata,
        schema_version=genome.schema_version,
    )


__all__ = ["HuggingFaceProfileError", "extract_huggingface_genome"]
