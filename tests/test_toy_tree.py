from __future__ import annotations

import contextlib
import io
import json
import unittest
from unittest import mock

from phylodigy.cli import main
from phylodigy.model_profile import TorchUnavailableError, torch_available
from phylodigy.toy_tree import (
    TOY_FRONTEND,
    build_toy_genomes,
    build_toy_models,
    build_toy_phylodigital_tree,
    toy_tree_summary,
)


class ToyTreeDependencyTests(unittest.TestCase):
    @unittest.skipIf(torch_available(), "PyTorch is installed")
    def test_missing_model_dependency_has_an_install_hint(self):
        with self.assertRaisesRegex(TorchUnavailableError, "phylodigy\\[model\\]"):
            build_toy_models()

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main(["toy-tree"])
        self.assertEqual(code, 2)
        self.assertIn("phylodigy[model]", stderr.getvalue())


@unittest.skipUnless(torch_available(), "PyTorch is not installed")
class ToyTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = build_toy_models()
        cls.genomes = build_toy_genomes()
        cls.result = build_toy_phylodigital_tree(cls.genomes)

    def test_builds_executable_models_and_graph_genomes(self):
        import torch

        models = self.models
        genomes = self.genomes
        result = self.result

        self.assertEqual(len(models), 5)
        self.assertTrue(
            all(isinstance(model, torch.nn.Module) for model in models.values())
        )
        with torch.no_grad():
            self.assertEqual(
                tuple(models["toy:seed"](torch.zeros(1, 8)).shape), (1, 8)
            )
            self.assertEqual(
                tuple(models["toy:amber-2"](torch.zeros(1, 8)).shape), (1, 2)
            )
            self.assertEqual(
                tuple(models["toy:blue-2"](torch.zeros(1, 8)).shape), (1, 2)
            )

        self.assertEqual(len(genomes), 5)
        self.assertEqual(len({item.structural_digest for item in genomes}), 5)
        self.assertTrue(
            all(item.graph_profile.frontend == TOY_FRONTEND for item in genomes)
        )
        self.assertEqual(result["artifact_type"], "phylodigy.architecture_lineage")
        self.assertEqual(result["tree"]["method"], "neighbor_joining")
        self.assertEqual(
            result["character_matrix"]["source"], "operator_layer_graph_only"
        )

    def test_sibling_models_are_closer_than_cross_branch_models(self):
        result = self.result
        distances = {
            frozenset((item["left"]["id"], item["right"]["id"])): item[
                "graph_distance"
            ]["distance"]
            for item in result["comparisons"]
        }

        amber = distances[frozenset(("toy:amber-1", "toy:amber-2"))]
        blue = distances[frozenset(("toy:blue-1", "toy:blue-2"))]
        cross = distances[frozenset(("toy:amber-1", "toy:blue-1"))]
        self.assertLess(amber, cross)
        self.assertLess(blue, cross)

    def test_summary_and_cli_offer_human_and_machine_readable_views(self):
        summary = toy_tree_summary(self.result)
        self.assertIn("executable PyTorch models", summary)
        self.assertIn("amber branch", summary)
        self.assertIn("blue branch", summary)

        stdout = io.StringIO()
        with mock.patch(
            "phylodigy.cli.build_toy_phylodigital_tree", return_value=self.result
        ), contextlib.redirect_stdout(stdout):
            code = main(["toy-tree"])
        result = json.loads(stdout.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["tree"]["method"], "neighbor_joining")

        stdout = io.StringIO()
        with mock.patch(
            "phylodigy.cli.build_toy_phylodigital_tree", return_value=self.result
        ), contextlib.redirect_stdout(stdout):
            code = main(["toy-tree", "--summary"])
        self.assertEqual(code, 0)
        self.assertIn("Toy phylodigital tree", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
