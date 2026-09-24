"""A small phylodigital tree built from executable PyTorch models.

Each artifact is an actual ``torch.nn.Module`` with learnable parameters. The
normal model frontend traces its executed structure through ``torch.fx`` and
derives the graph characters used for lineage inference. Artifact names,
release dates, and the explanatory branch labels never enter the distance.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from .lineage import infer_lineage_network
from .model_profile import TorchUnavailableError, extract_architectural_genome
from .schema import ArchitecturalGenome


TOY_FRONTEND = "torch.fx"

_TOY_ARTIFACTS = (
    ("toy:seed", "Seed MLP", "2020-01-01", "shared core"),
    ("toy:amber-1", "Amber MLP 1", "2021-01-01", "amber"),
    ("toy:amber-2", "Amber MLP 2", "2022-01-01", "amber"),
    ("toy:blue-1", "Blue MLP 1", "2021-06-01", "blue"),
    ("toy:blue-2", "Blue MLP 2", "2022-06-01", "blue"),
)


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise TorchUnavailableError(
            "the toy tree uses executable PyTorch models; install "
            "'phylodigy[model]' first"
        ) from exc


def build_toy_models() -> dict[str, Any]:
    """Instantiate five small, executable neural-network artifacts.

    All models share a ``Linear(8, 8)`` and ``ReLU`` stem. The amber models add
    a narrowing ReLU branch; the blue models add a width-preserving Tanh
    branch. The second model in each branch adds one more learned projection.
    """

    torch = _torch()
    nn = torch.nn
    return {
        "toy:seed": nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
        ),
        "toy:amber-1": nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        ),
        "toy:amber-2": nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        ),
        "toy:blue-1": nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.Tanh(),
        ),
        "toy:blue-2": nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
        ),
    }


def build_toy_genomes() -> tuple[ArchitecturalGenome, ...]:
    """Trace the toy models into graph-derived architectural genomes."""

    models = build_toy_models()
    genomes = []
    for artifact_id, name, release_date, story_branch in _TOY_ARTIFACTS:
        extracted = extract_architectural_genome(
            models[artifact_id],
            artifact_id=artifact_id,
            name=name,
            release_date=release_date,
            propagate_shapes=False,
        )
        metadata = dict(extracted.metadata)
        metadata["toy_demonstration"] = {
            "demonstration_only": True,
            "story_branch": story_branch,
        }
        genomes.append(
            ArchitecturalGenome(
                artifact_id=extracted.artifact_id,
                graph_profile=extracted.graph_profile,
                artifact_kind=extracted.artifact_kind,
                name=extracted.name,
                release_date=extracted.release_date,
                extractor_name=extracted.extractor_name,
                extractor_version=extracted.extractor_version,
                metadata=metadata,
            )
        )
    return tuple(genomes)


def build_toy_phylodigital_tree(
    genomes: Iterable[ArchitecturalGenome] | None = None,
) -> dict[str, Any]:
    """Infer a graph-character neighbor-joining tree from the toy models.

    Supplying already traced ``genomes`` avoids repeating extraction in a
    notebook or test that also needs to inspect the individual artifacts.
    """

    selected = build_toy_genomes() if genomes is None else genomes
    return infer_lineage_network(selected)


def toy_tree_summary(result: dict[str, Any] | None = None) -> str:
    """Return a concise, human-readable summary of a toy lineage result."""

    lineage = build_toy_phylodigital_tree() if result is None else result
    tree = lineage["tree"]
    diagnostics = tree["tree_likeness"]
    comparisons = lineage["comparisons"]
    distances = {
        tuple(sorted((item["left"]["id"], item["right"]["id"]))): item[
            "graph_distance"
        ]["distance"]
        for item in comparisons
    }

    def distance(left: str, right: str) -> float:
        return distances[tuple(sorted((left, right)))]

    return "\n".join(
        (
            "Toy phylodigital tree (executable PyTorch models)",
            "  shared core: toy:seed",
            "  amber branch: toy:amber-1 -> toy:amber-2",
            "  blue branch:  toy:blue-1  -> toy:blue-2",
            "",
            "Graph-character distances",
            f"  amber siblings: {distance('toy:amber-1', 'toy:amber-2'):g}",
            f"  blue siblings:  {distance('toy:blue-1', 'toy:blue-2'):g}",
            f"  cross-branch:   {distance('toy:amber-1', 'toy:blue-1'):g}",
            "",
            f"Neighbor joining: {len(tree['taxa'])} taxa, {len(tree['edges'])} edges",
            "Tree-like within tolerance: "
            f"{str(diagnostics['tree_metric_within_tolerance']).lower()}",
            f"Lineage digest: {lineage['digest']}",
        )
    )


__all__ = [
    "TOY_FRONTEND",
    "build_toy_models",
    "build_toy_genomes",
    "build_toy_phylodigital_tree",
    "toy_tree_summary",
]
