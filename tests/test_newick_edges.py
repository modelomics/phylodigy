from __future__ import annotations

import re
import unittest

from phylodigy.computation_graph import build_graph_profile
from phylodigy.lineage import infer_lineage_network
from phylodigy.newick import lineage_to_newick
from phylodigy.schema import ArchitecturalGenome


def _genome(artifact_id: str, operation: str) -> ArchitecturalGenome:
    records = [
        {"index": 0, "name": "input", "op": "placeholder", "target": "input"},
        {
            "index": 1,
            "name": "layer",
            "op": "call_function",
            "target": operation,
            "inputs": ["input"],
        },
        {
            "index": 2,
            "name": "output",
            "op": "output",
            "target": "output",
            "inputs": ["layer"],
        },
    ]
    return ArchitecturalGenome(artifact_id, build_graph_profile(records))


def _leaf_labels(newick: str) -> set[str]:
    """Read leaf labels from the subset of Newick emitted by this serializer."""
    tokens = re.findall(r"'(?:(?:'')|[^'])*'|[(),:;]|[^(),:;\s]+", newick)
    leaves: set[str] = set()
    for index, token in enumerate(tokens[:-1]):
        if token in {"(", ",", ")"}:
            continue
        # A label immediately before a branch length is a child leaf; internal
        # subtrees end in ')' and labels never have support values.
        if tokens[index + 1] == ":":
            if token.startswith("'"):
                token = token[1:-1].replace("''", "'")
            leaves.add(token)
    return leaves


class NewickEdgeIntegrationTests(unittest.TestCase):
    def test_real_inference_round_trips_terminal_ids_and_is_deterministic(self):
        genomes = (
            _genome("model: O'Brien", "vendor.alpha"),
            _genome("模型/β", "vendor.alpha"),
            _genome("plain-model", "vendor.beta"),
        )
        inferred = infer_lineage_network(genomes)
        serialized = lineage_to_newick(inferred)
        self.assertEqual(_leaf_labels(serialized), {item.artifact_id for item in genomes})
        self.assertTrue(serialized.endswith(";"))
        self.assertEqual(serialized, lineage_to_newick(inferred))

        reordered = {**inferred, "tree": {**inferred["tree"], "edges": list(reversed(inferred["tree"]["edges"]))}}
        self.assertEqual(serialized, lineage_to_newick(reordered))

    def test_invalid_tree_shapes_and_lengths_are_rejected(self):
        cases = (
            {"tree": {"edges": []}},
            {"tree": {"edges": [{"parent": "r", "child": "x", "branch_length": float("nan")}] }},
            {"tree": {"edges": [{"parent": "r", "child": "x", "branch_length": -1}]}},
            {"tree": {"edges": [
                {"parent": "r", "child": "x", "branch_length": 1},
                {"parent": "s", "child": "x", "branch_length": 1},
            ]}},
            {"tree": {"edges": [
                {"parent": "r", "child": "x", "branch_length": 1},
                {"parent": "x", "child": "r", "branch_length": 1},
            ]}},
            {"tree": {"edges": [
                {"parent": "r", "child": "x", "branch_length": 1},
                {"parent": "s", "child": "t", "branch_length": 1},
            ]}},
        )
        for lineage in cases:
            with self.subTest(lineage=lineage), self.assertRaises(ValueError):
                lineage_to_newick(lineage)


if __name__ == "__main__":
    unittest.main()
