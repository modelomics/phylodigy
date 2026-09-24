from __future__ import annotations

import json
import unittest

from phylodigy.model_profile import (
    GraphExtractionError,
    canonical_profile_json,
    extract_model_profile,
    profile_digest,
    torch_available,
)
from phylodigy.computation_graph import build_graph_profile
from phylodigy.schema import ArchitecturalGenome


class GraphOnlyContractTests(unittest.TestCase):
    def test_graph_extraction_cannot_be_disabled(self):
        with self.assertRaisesRegex(ValueError, "graph extraction cannot be disabled"):
            extract_model_profile(object(), model_id="example", use_fx=False)

    def test_digest_helper_is_order_invariant_for_json_mappings(self):
        self.assertEqual(
            profile_digest({"z": 1, "a": [2, 3]}),
            profile_digest({"a": [2, 3], "z": 1}),
        )

    def test_digest_helper_hashes_artifact_content_not_its_digest_envelope(self):
        graph_profile = build_graph_profile(
            [
                {"name": "x", "op": "placeholder", "target": "x", "inputs": []},
                {"name": "out", "op": "output", "target": "out", "inputs": ["x"]},
            ],
            radii=(0,),
        )
        genome = ArchitecturalGenome("model:test", graph_profile)

        self.assertEqual(profile_digest(graph_profile), graph_profile.digest)
        self.assertEqual(profile_digest(graph_profile.to_dict()), graph_profile.digest)
        self.assertEqual(
            profile_digest(graph_profile.to_dict(include_digest=False)),
            graph_profile.digest,
        )
        self.assertEqual(profile_digest(genome), genome.digest)
        self.assertEqual(profile_digest(genome.to_dict()), genome.digest)
        self.assertEqual(
            profile_digest(genome.to_dict(include_digest=False)),
            genome.digest,
        )

        stale = genome.to_dict()
        stale["artifact"]["kind"] = "changed"
        self.assertNotEqual(profile_digest(stale), stale["digest"])


@unittest.skipUnless(torch_available(), "PyTorch is not installed")
class TorchGraphFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        cls.nn = torch.nn

    def test_same_graph_is_invariant_to_architecture_class_name(self):
        torch = self.torch
        nn = self.nn

        class TransformerAttentionResidual(nn.Module):
            def forward(self, value):
                return torch.sin(value)

        class CompletelyUnrelatedName(nn.Module):
            def forward(self, value):
                return torch.sin(value)

        left = extract_model_profile(
            TransformerAttentionResidual(),
            model_id="left",
            propagate_shapes=False,
        )
        right = extract_model_profile(
            CompletelyUnrelatedName(),
            model_id="right",
            propagate_shapes=False,
        )

        self.assertEqual(left.structural_digest, right.structural_digest)
        self.assertEqual(left.character_counts, right.character_counts)
        structural_json = left.graph_profile.canonical_json().lower()
        for forbidden in ("transformer", "attention", "residual"):
            self.assertNotIn(forbidden, structural_json)

    def test_parameter_values_are_ignored_but_shapes_are_structural(self):
        nn = self.nn

        same_shape_a = nn.Linear(4, 3)
        same_shape_b = nn.Linear(4, 3)
        different_shape = nn.Linear(4, 5)

        left = extract_model_profile(
            same_shape_a,
            model_id="a",
            propagate_shapes=False,
        )
        right = extract_model_profile(
            same_shape_b,
            model_id="b",
            propagate_shapes=False,
        )
        changed = extract_model_profile(
            different_shape,
            model_id="c",
            propagate_shapes=False,
        )

        self.assertEqual(left.structural_digest, right.structural_digest)
        self.assertNotEqual(left.structural_digest, changed.structural_digest)
        payload = left.graph_profile.to_dict()
        text = json.dumps(payload)
        self.assertIn("operation.resource.parameter", text)
        self.assertIn("tensor_signature", text)
        self.assertNotIn("parameter_values", text)

    def test_arbitrary_public_leaf_fields_are_observations_not_structure(self):
        nn = self.nn

        class Wrapper(nn.Module):
            def __init__(self, marker):
                super().__init__()
                self.layer = nn.Linear(4, 4, bias=False)
                self.layer.research_label = marker

            def forward(self, value):
                return self.layer(value)

        left = extract_model_profile(
            Wrapper("observation-only-left"),
            model_id="left",
            propagate_shapes=False,
        )
        right = extract_model_profile(
            Wrapper("observation-only-right"),
            model_id="right",
            propagate_shapes=False,
        )

        self.assertEqual(left.structural_digest, right.structural_digest)
        self.assertEqual(left.character_counts, right.character_counts)
        self.assertNotEqual(
            left.graph_profile.graph.digest,
            right.graph_profile.graph.digest,
        )
        left_layer = next(
            node
            for node in left.graph_profile.graph.nodes
            if node.operation.endswith(".linear")
        )
        self.assertIn("observation-only-left", json.dumps(left_layer.to_dict()))
        self.assertNotIn(
            "observation-only-left",
            json.dumps(left_layer.structural_dict()),
        )

    def test_resource_aliasing_distinguishes_tied_from_untied_parameters(self):
        nn = self.nn

        class Pair(nn.Module):
            def __init__(self, tied, first_name="first", second_name="second"):
                super().__init__()
                first = nn.Linear(4, 4, bias=False)
                second = nn.Linear(4, 4, bias=False)
                if tied:
                    second.weight = first.weight
                setattr(self, first_name, first)
                setattr(self, second_name, second)
                self._first_name = first_name
                self._second_name = second_name

            def forward(self, value):
                first = getattr(self, self._first_name)(value)
                second = getattr(self, self._second_name)(value)
                return first + second

        tied = extract_model_profile(
            Pair(True), model_id="tied", propagate_shapes=False
        )
        tied_renamed = extract_model_profile(
            Pair(True, "alpha", "omega"),
            model_id="tied-renamed",
            propagate_shapes=False,
        )
        untied = extract_model_profile(
            Pair(False), model_id="untied", propagate_shapes=False
        )
        untied_renamed = extract_model_profile(
            Pair(False, "alpha", "omega"),
            model_id="untied-renamed",
            propagate_shapes=False,
        )

        self.assertEqual(tied.structural_digest, tied_renamed.structural_digest)
        self.assertEqual(untied.structural_digest, untied_renamed.structural_digest)
        self.assertNotEqual(tied.structural_digest, untied.structural_digest)

        def parameter_resource_count(genome):
            return sum(
                node.operation == "operation.resource.parameter"
                for node in genome.graph_profile.graph.nodes
            )

        self.assertEqual(parameter_resource_count(tied), 1)
        self.assertEqual(parameter_resource_count(untied), 2)
        structural_text = json.dumps(
            [
                node.structural_dict()
                for node in tied.graph_profile.graph.nodes
            ],
            sort_keys=True,
        )
        for forbidden in ('"first"', '"second"', '"alpha"', '"omega"', '"weight"'):
            self.assertNotIn(forbidden, structural_text)

    def test_resource_aliasing_distinguishes_shared_from_repeated_modules(self):
        nn = self.nn

        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 4)

            def forward(self, value):
                return self.layer(self.layer(value))

        class Repeated(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(4, 4)
                self.second = nn.Linear(4, 4)

            def forward(self, value):
                return self.second(self.first(value))

        shared = extract_model_profile(
            Shared(), model_id="shared", propagate_shapes=False
        )
        repeated = extract_model_profile(
            Repeated(), model_id="repeated", propagate_shapes=False
        )

        self.assertNotEqual(shared.structural_digest, repeated.structural_digest)
        shared_modules = sum(
            node.operation == "operation.resource.module"
            for node in shared.graph_profile.graph.nodes
        )
        repeated_modules = sum(
            node.operation == "operation.resource.module"
            for node in repeated.graph_profile.graph.nodes
        )
        self.assertEqual(shared_modules, 1)
        self.assertEqual(repeated_modules, 2)
        self.assertEqual(shared.graph_profile.regions, ())
        self.assertEqual(repeated.graph_profile.regions, ())

    def test_resource_coverage_reports_uncaptured_unused_parameters(self):
        nn = self.nn

        class PartiallyUsed(nn.Module):
            def __init__(self):
                super().__init__()
                self.used = nn.Linear(4, 4, bias=False)
                self.unused = nn.Linear(4, 4, bias=False)

            def forward(self, value):
                return self.used(value)

        genome = extract_model_profile(
            PartiallyUsed(), model_id="partial", propagate_shapes=False
        )
        coverage = genome.metadata["coverage"]
        resources = genome.metadata["trace"]["resource_capture"]

        self.assertFalse(coverage["weight_shapes_captured"])
        self.assertEqual(
            coverage["module_boundary_policy"],
            "torch_fx_default_leaf_modules_are_atomic",
        )
        self.assertFalse(coverage["example_inputs_specialize_trace"])
        self.assertEqual(resources["declared_parameter_count"], 2)
        self.assertEqual(resources["captured_declared_parameter_count"], 1)
        self.assertEqual(resources["atomic_call_module_node_count"], 1)

    def test_opaque_nested_module_resources_are_captured_recursively(self):
        nn = self.nn

        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.TransformerEncoderLayer(
                    d_model=4,
                    nhead=2,
                    dim_feedforward=8,
                    dropout=0.0,
                    batch_first=True,
                )

            def forward(self, value):
                return self.block(value)

        genome = extract_model_profile(
            Wrapper(), model_id="opaque", propagate_shapes=False
        )
        coverage = genome.metadata["coverage"]
        resources = genome.metadata["trace"]["resource_capture"]

        self.assertTrue(coverage["weight_shapes_captured"])
        self.assertGreater(resources["declared_parameter_count"], 0)
        self.assertEqual(
            resources["captured_declared_parameter_count"],
            resources["declared_parameter_count"],
        )
        self.assertEqual(resources["atomic_call_module_node_count"], 1)

    def test_default_artifact_id_is_content_derived_after_extraction(self):
        torch = self.torch
        nn = self.nn

        class FirstName(nn.Module):
            def forward(self, value):
                return torch.sin(value)

        class RelocatedOrRenamed(nn.Module):
            def forward(self, value):
                return torch.sin(value)

        left = extract_model_profile(FirstName(), propagate_shapes=False)
        right = extract_model_profile(RelocatedOrRenamed(), propagate_shapes=False)

        self.assertEqual(left.artifact_id, right.artifact_id)
        self.assertEqual(
            left.artifact_id,
            f"model:graph:{left.structural_digest}",
        )
        explicit = extract_model_profile(
            FirstName(),
            artifact_id="checkpoint:explicit",
            propagate_shapes=False,
        )
        self.assertEqual(explicit.artifact_id, "checkpoint:explicit")

    def test_nested_dynamic_inputs_and_static_arguments_are_retained(self):
        torch = self.torch
        nn = self.nn

        class GenericProgram(nn.Module):
            def forward(self, left, right):
                joined = torch.cat((left, right), dim=-1)
                return joined.transpose(0, 1)

        genome = extract_model_profile(
            GenericProgram(),
            model_id="generic",
            propagate_shapes=False,
        )
        graph = genome.graph_profile.graph

        cat_nodes = [node for node in graph.nodes if "cat" in node.operation]
        self.assertEqual(len(cat_nodes), 1)
        cat_node = cat_nodes[0]
        incoming = [edge for edge in graph.edges if edge.target == cat_node.node_id]
        self.assertEqual(len(incoming), 2)
        self.assertEqual(
            {edge.position for edge in incoming},
            {"args[0][0]", "args[0][1]"},
        )
        self.assertEqual(
            cat_node.attributes["arguments"]["kwargs['dim']"],
            -1,
        )

    def test_every_fork_join_is_anonymous_even_when_class_name_suggests_a_feature(self):
        nn = self.nn

        class ResNetResidualSkipAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(4, 4)

            def forward(self, value):
                return value + self.layer(value)

        genome = extract_model_profile(
            ResNetResidualSkipAttention(),
            model_id="anonymous",
            propagate_shapes=False,
        )
        profile = genome.graph_profile

        self.assertEqual(len(profile.regions), 1)
        region = profile.regions[0].to_dict()
        self.assertNotIn("kind", region)
        self.assertNotIn("label", region)
        self.assertNotIn("description", region)
        self.assertEqual(region["discovery"], "fork_join.v1")
        text = profile.canonical_json().lower()
        for forbidden in ("resnet", "residual", "skip", "attention"):
            self.assertNotIn(forbidden, text)

    def test_trace_failure_is_an_error_and_never_falls_back_to_names_or_config(self):
        nn = self.nn

        class NamedLikeAKnownArchitecture(nn.Module):
            model_type = "transformer"
            num_attention_heads = 32

            def forward(self, value):
                if value.sum() > 0:
                    return value
                return -value

        with self.assertRaisesRegex(GraphExtractionError, "could not produce"):
            extract_model_profile(
                NamedLikeAKnownArchitecture(),
                model_id="data-dependent",
                propagate_shapes=False,
            )

    def test_torch_export_fallback_profiles_models_with_example_inputs(self):
        torch = self.torch
        nn = self.nn

        class InPlaceProgram(nn.Module):
            def forward(self, value):
                result = value.clone()
                result[:, 0] = 0
                return result

        genome = extract_model_profile(
            InPlaceProgram(),
            model_id="export-only",
            example_inputs=torch.ones(2, 3),
            propagate_shapes=False,
        )

        self.assertEqual(genome.graph_profile.frontend, "torch.export")
        self.assertEqual(genome.metadata["trace"]["frontend"], "torch.export")
        self.assertEqual(
            genome.metadata["coverage"]["module_boundary_policy"],
            "torch_export_lowers_modules_to_aten_operators",
        )

    def test_canonical_genome_json_contains_graph_not_traits_or_ontology(self):
        nn = self.nn
        genome = extract_model_profile(
            nn.Linear(3, 2),
            model_id="linear",
            release_date="2026-01-02",
            propagate_shapes=False,
        )
        payload = json.loads(canonical_profile_json(genome))

        self.assertEqual(payload["artifact_type"], "phylodigy.architectural_genome")
        self.assertIn("graph_profile", payload)
        self.assertNotIn("traits", payload)
        self.assertNotIn("ontology", payload)
        self.assertEqual(payload["digest"], genome.digest)


if __name__ == "__main__":
    unittest.main()
