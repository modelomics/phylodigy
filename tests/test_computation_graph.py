from __future__ import annotations

import unittest

from phylodigy.computation_graph import (
    ComputationGraph,
    FingerprintScale,
    GraphEdge,
    GraphProfile,
    GraphNode,
    StructuralFingerprint,
    build_graph_profile,
    compare_graphs,
    computation_graph_from_fx,
    discover_graph_characters,
    discover_graph_regions,
    normalize_operator,
)


def node(index, name, op, target, inputs=(), **extra):
    return {
        "index": index,
        "inputs": list(inputs),
        "name": name,
        "op": op,
        "target": target,
        **extra,
    }


COMMUTATIVE_ATTRIBUTES = {
    "operator_properties": {"commutative": True}
}


def chain(*operations):
    records = [node(0, "input", "placeholder", "input")]
    previous = "input"
    for index, operation in enumerate(operations, start=1):
        name = f"operation_{index}"
        records.append(node(index, name, "call_function", operation, (previous,)))
        previous = name
    records.append(node(len(records), "output", "output", "output", (previous,)))
    return build_graph_profile(records)


class OperatorNormalizationTests(unittest.TestCase):
    def test_operator_spellings_are_not_mapped_to_a_semantic_alias(self):
        aliases = (
            ("call_function", "_operator.add"),
            ("call_function", "torch.add"),
            ("call_function", "aten.add.Tensor"),
            ("call_method", "add"),
            ("call_method", "__add__"),
            ("call_function", "<built-in function add>"),
        )
        normalized = {normalize_operator(op, target) for op, target in aliases}
        self.assertEqual(len(normalized), len(aliases))
        self.assertNotIn("merge.add", normalized)

    def test_unknown_operators_are_retained_in_open_vocabulary(self):
        self.assertEqual(
            normalize_operator("call_function", "acme.experimental.quantum_gate"),
            "function.acme.experimental.quantum_gate",
        )
        self.assertEqual(
            normalize_operator(
                "call_module", "renameable.path", "vendor.layers.NovelMixer"
            ),
            "module.vendor.layers.novelmixer",
        )
        self.assertEqual(
            normalize_operator(
                "call_module", "renameable.path", "torch.nn.modules.linear.Linear"
            ),
            "module.torch.nn.linear",
        )
        self.assertEqual(
            normalize_operator("call_function", "vendor.add.forward"),
            "function.vendor.add.forward",
        )


class ComputationGraphProfileTests(unittest.TestCase):
    def test_record_and_mapping_order_do_not_change_the_canonical_graph(self):
        records = (
            node(0, "tokens", "placeholder", "tokens"),
            node(1, "hidden", "call_function", "torch.relu", ("tokens",)),
            node(2, "result", "output", "output", ("hidden",)),
        )
        reordered = []
        for record in reversed(records):
            reordered.append(dict(reversed(list(record.items()))))

        first = build_graph_profile(records)
        second = build_graph_profile(reordered)

        self.assertEqual(first.graph.canonical_json(), second.graph.canonical_json())
        self.assertEqual(first.digest, second.digest)

    def test_variable_module_path_and_commutative_argument_renames_are_invariant(self):
        left = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(
                    1,
                    "projection",
                    "call_module",
                    "encoder.old_name",
                    ("x",),
                    target_type="torch.nn.Linear",
                ),
                node(
                    2,
                    "sum",
                    "call_function",
                    "vendor.combine",
                    ("x", "projection"),
                    attributes=COMMUTATIVE_ATTRIBUTES,
                ),
                node(3, "result", "output", "output", ("sum",)),
            )
        )
        right = build_graph_profile(
            (
                node(0, "tokens", "placeholder", "tokens"),
                node(
                    1,
                    "renamed_block",
                    "call_module",
                    "totally.different.path",
                    ("tokens",),
                    target_type="torch.nn.Linear",
                ),
                node(
                    2,
                    "combined",
                    "call_function",
                    "vendor.combine",
                    ("renamed_block", "tokens"),
                    attributes=COMMUTATIVE_ATTRIBUTES,
                ),
                node(3, "return_value", "output", "output", ("combined",)),
            )
        )

        self.assertEqual(left.graph.digest, right.graph.digest)
        self.assertEqual(left.digest, right.digest)
        self.assertEqual(compare_graphs(left, right).distance, 0.0)

    def test_parallel_emission_order_does_not_change_graph_identity(self):
        left = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(1, "branch_a", "call_function", "torch.relu", ("x",)),
                node(2, "branch_b", "call_function", "torch.relu", ("x",)),
                node(3, "tanh", "call_function", "torch.tanh", ("branch_a",)),
                node(4, "sigmoid", "call_function", "torch.sigmoid", ("branch_b",)),
                node(
                    5,
                    "sum",
                    "call_function",
                    "vendor.combine",
                    ("tanh", "sigmoid"),
                    attributes=COMMUTATIVE_ATTRIBUTES,
                ),
                node(6, "out", "output", "output", ("sum",)),
            )
        )
        right = build_graph_profile(
            (
                node(0, "input", "placeholder", "input"),
                node(1, "emitted_first", "call_function", "torch.relu", ("input",)),
                node(2, "emitted_second", "call_function", "torch.relu", ("input",)),
                node(
                    3,
                    "sigmoid_first",
                    "call_function",
                    "torch.sigmoid",
                    ("emitted_first",),
                ),
                node(
                    4,
                    "tanh_second",
                    "call_function",
                    "torch.tanh",
                    ("emitted_second",),
                ),
                node(
                    5,
                    "combined",
                    "call_function",
                    "vendor.combine",
                    ("sigmoid_first", "tanh_second"),
                    attributes=COMMUTATIVE_ATTRIBUTES,
                ),
                node(6, "return_value", "output", "output", ("combined",)),
            )
        )

        self.assertEqual(left.graph.digest, right.graph.digest)
        self.assertEqual(compare_graphs(left, right).distance, 0.0)

    def test_parallel_branch_emission_order_is_canonicalized_from_future_context(self):
        left = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(1, "a0", "call_function", "torch.relu", ("x",)),
                node(2, "b0", "call_function", "torch.relu", ("x",)),
                node(3, "a1", "call_function", "torch.tanh", ("a0",)),
                node(4, "b1", "call_function", "torch.sigmoid", ("b0",)),
                node(
                    5,
                    "join",
                    "call_function",
                    "vendor.combine",
                    ("a1", "b1"),
                    attributes=COMMUTATIVE_ATTRIBUTES,
                ),
                node(6, "out", "output", "output", ("join",)),
            )
        )
        right = build_graph_profile(
            (
                node(0, "input", "placeholder", "input"),
                node(1, "branch_b", "call_function", "torch.relu", ("input",)),
                node(2, "branch_a", "call_function", "torch.relu", ("input",)),
                node(
                    3,
                    "finish_b",
                    "call_function",
                    "torch.sigmoid",
                    ("branch_b",),
                ),
                node(
                    4,
                    "finish_a",
                    "call_function",
                    "torch.tanh",
                    ("branch_a",),
                ),
                node(
                    5,
                    "sum",
                    "call_function",
                    "vendor.combine",
                    ("finish_b", "finish_a"),
                    attributes=COMMUTATIVE_ATTRIBUTES,
                ),
                node(6, "result", "output", "output", ("sum",)),
            )
        )

        self.assertEqual(left.graph.digest, right.graph.digest)
        self.assertEqual(left.fingerprint, right.fingerprint)

    def test_unresolved_color_ties_never_fall_back_to_emission_order(self):
        edges = (
            (0, 1), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4),
            (1, 6), (1, 7), (2, 4), (3, 5), (4, 6), (5, 6), (6, 7),
        )

        def records(order):
            result = []
            for identifier in order:
                inputs = tuple(str(source) for source, target in edges if target == identifier)
                if identifier == 0:
                    result.append(node(99, str(identifier), "placeholder", "input"))
                elif identifier == 7:
                    result.append(node(99, str(identifier), "output", "output", inputs))
                else:
                    target = "operator.add" if len(inputs) > 1 else "vendor.foo"
                    result.append(node(99, str(identifier), "call_function", target, inputs))
            return result

        left = build_graph_profile(records((0, 1, 2, 4, 3, 5, 6, 7)))
        right = build_graph_profile(records((0, 1, 3, 2, 5, 4, 6, 7)))

        self.assertEqual(left.graph, right.graph)
        self.assertEqual(left.graph.structural_digest, right.graph.structural_digest)

    def test_content_addressed_payloads_are_recursively_immutable(self):
        profile = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(
                    1,
                    "op",
                    "call_function",
                    "vendor.op",
                    ("x",),
                    attributes={"nested": {"sizes": [1, 2]}},
                    observations={"probe": {"batch": [8]}},
                ),
                node(2, "out", "output", "output", ("op",)),
            ),
            metadata={"frontend": {"options": ["a"]}},
        )
        digest = profile.digest
        operation = next(item for item in profile.graph.nodes if item.operation == "function.vendor.op")

        with self.assertRaises(TypeError):
            operation.attributes["new"] = 1
        with self.assertRaises(TypeError):
            operation.attributes["nested"]["sizes"] = (3,)
        with self.assertRaises(TypeError):
            profile.metadata["new"] = True
        self.assertEqual(profile.digest, digest)
        self.assertEqual(GraphProfile.from_dict(profile.to_dict()), profile)

    def test_noncommutative_argument_order_is_preserved(self):
        def difference(inputs):
            return build_graph_profile(
                (
                    node(0, "x", "placeholder", "x"),
                    node(1, "relu", "call_function", "torch.relu", ("x",)),
                    node(2, "tanh", "call_function", "torch.tanh", ("x",)),
                    node(3, "subtract", "call_function", "vendor.subtract", inputs),
                    node(4, "out", "output", "output", ("subtract",)),
                )
            )

        left = difference(("relu", "tanh"))
        right = difference(("tanh", "relu"))

        self.assertNotEqual(left.graph.digest, right.graph.digest)
        self.assertGreater(compare_graphs(left, right).distance, 0.0)

    def test_static_operator_attributes_are_dynamic_graph_characters(self):
        def convolution(kernel_size):
            return build_graph_profile(
                (
                    node(0, "x", "placeholder", "x"),
                    node(
                        1,
                        "convolution",
                        "call_function",
                        "vendor.convolution",
                        ("x",),
                        attributes={
                            "arguments": {
                                "kwargs['groups']": 1,
                                "kwargs['kernel_size']": kernel_size,
                            }
                        },
                    ),
                    node(2, "out", "output", "output", ("convolution",)),
                )
            )

        three = convolution([3, 3])
        five = convolution([5, 5])

        self.assertNotEqual(three.graph.digest, five.graph.digest)
        self.assertGreater(compare_graphs(three, five).distance, 0.0)

    def test_probe_tensor_metadata_is_not_a_structural_character(self):
        def traced(batch_size):
            return build_graph_profile(
                (
                    node(
                        0,
                        "x",
                        "placeholder",
                        "x",
                        tensor_meta={"shape": [batch_size, 8]},
                    ),
                    node(
                        1,
                        "layer",
                        "call_function",
                        "vendor.layer",
                        ("x",),
                        tensor_meta={"shape": [batch_size, 8]},
                    ),
                    node(2, "out", "output", "output", ("layer",)),
                )
            )

        small = traced(1)
        large = traced(64)

        self.assertNotEqual(small.graph.digest, large.graph.digest)
        self.assertEqual(small.graph.structural_digest, large.graph.structural_digest)
        self.assertEqual(small.characters, large.characters)
        self.assertEqual(compare_graphs(small, large).distance, 0.0)

    def test_fork_join_is_discovered_as_anonymous_graph_region(self):
        profile = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(
                    1,
                    "layer",
                    "call_module",
                    "layer",
                    ("x",),
                    target_type="torch.nn.Linear",
                ),
                node(2, "sum", "call_function", "operator.add", ("x", "layer")),
                node(3, "out", "output", "output", ("sum",)),
            )
        )

        self.assertEqual(len(profile.regions), 1)
        region = profile.regions[0]
        self.assertEqual(len(region.branch_paths), 2)
        self.assertEqual(set(region.support_nodes), {"n000000", "n000001", "n000002"})
        self.assertTrue(all(edge in region.support_edges for edge in (
            "n000000->n000001@arg:000000",
            "n000001->n000002@arg:000001",
        )))
        self.assertRegex(region.structural_signature, r"^[0-9a-f]{64}$")
        character = next(
            item
            for item in profile.characters
            if item.generator == "fork_join_region.v1"
        )
        self.assertEqual(region.character_id, character.character_id)
        self.assertEqual(
            region.occurrence_id,
            character.occurrences[0].occurrence_id,
        )
        region_characters = [
            item
            for item in profile.characters
            if item.generator == "fork_join_region.v1"
        ]
        self.assertEqual(len(region_characters), 1)
        self.assertEqual(region_characters[0].structural_signature, region.structural_signature)
        serialized = profile.canonical_json().lower()
        for forbidden in ("residual", "skip", "projection", "attention"):
            self.assertNotIn(forbidden, serialized)

    def test_region_discovery_is_operator_agnostic(self):
        def fork_join(join_operation):
            return build_graph_profile(
                (
                    node(0, "origin", "placeholder", "origin"),
                    node(1, "left", "call_function", "vendor.alpha", ("origin",)),
                    node(2, "right", "call_function", "vendor.beta", ("origin",)),
                    node(3, "join", "call_function", join_operation, ("left", "right")),
                    node(4, "out", "output", "output", ("join",)),
                )
            )

        first = fork_join("vendor.binary_a")
        second = fork_join("vendor.binary_b")

        self.assertEqual(len(first.regions), 1)
        self.assertEqual(len(second.regions), 1)
        self.assertNotEqual(
            first.regions[0].structural_signature,
            second.regions[0].structural_signature,
        )
        self.assertEqual(discover_graph_regions(first.graph), first.regions)
        self.assertEqual(
            discover_graph_characters(first.graph, regions=first.regions),
            first.characters,
        )

    def test_unequal_branch_lengths_and_shapes_do_not_create_named_subtypes(self):
        profile = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(
                    1,
                    "short",
                    "call_function",
                    "vendor.alpha",
                    ("x",),
                    tensor_meta={"shape": [2, 8]},
                ),
                node(
                    2,
                    "long_a",
                    "call_function",
                    "vendor.beta",
                    ("x",),
                    tensor_meta={"shape": [2, 8]},
                ),
                node(
                    3,
                    "long_b",
                    "call_function",
                    "vendor.gamma",
                    ("long_a",),
                    tensor_meta={"shape": [2, 8]},
                ),
                node(
                    4,
                    "join",
                    "call_function",
                    "vendor.combine",
                    ("short", "long_b"),
                    tensor_meta={"shape": [2, 8]},
                ),
                node(5, "out", "output", "output", ("join",)),
            )
        )

        self.assertEqual(len(profile.regions), 1)
        region = profile.regions[0].to_dict()
        self.assertNotIn("kind", region)
        self.assertNotIn("label", region)
        self.assertNotIn("description", region)
        self.assertEqual(region["discovery"], "fork_join.v1")

    def test_join_of_independent_inputs_is_not_a_fork_join_region(self):
        profile = build_graph_profile(
            (
                node(0, "left", "placeholder", "left"),
                node(1, "right", "placeholder", "right"),
                node(2, "join", "call_function", "vendor.combine", ("left", "right")),
                node(3, "out", "output", "output", ("join",)),
            )
        )

        self.assertEqual(profile.regions, ())

    def test_resource_incidence_does_not_fabricate_a_data_flow_region(self):
        profile = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(1, "resource", "operation", "resource.parameter"),
                node(
                    2,
                    "first",
                    "call_function",
                    "vendor.layer",
                    input_edges=(
                        {"source": "x", "position": "args[0]"},
                        {"source": "resource", "position": "resource:parameter"},
                    ),
                ),
                node(
                    3,
                    "second",
                    "call_function",
                    "vendor.layer",
                    input_edges=(
                        {"source": "first", "position": "args[0]"},
                        {"source": "resource", "position": "resource:parameter"},
                    ),
                ),
                node(4, "out", "output", "output", ("second",)),
            )
        )

        self.assertEqual(profile.regions, ())
        self.assertTrue(
            any(edge.position == "resource:parameter" for edge in profile.graph.edges)
        )

    def test_repeated_anonymous_regions_become_one_character_with_two_occurrences(self):
        profile = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(1, "a_left", "call_function", "vendor.alpha", ("x",)),
                node(2, "a_right", "call_function", "vendor.beta", ("x",)),
                node(3, "a_join", "call_function", "vendor.combine", ("a_left", "a_right")),
                node(4, "b_left", "call_function", "vendor.alpha", ("x",)),
                node(5, "b_right", "call_function", "vendor.beta", ("x",)),
                node(6, "b_join", "call_function", "vendor.combine", ("b_left", "b_right")),
                node(7, "out", "output", "output", ("a_join", "b_join")),
            )
        )

        self.assertEqual(len(profile.regions), 2)
        self.assertEqual(
            {item.structural_signature for item in profile.regions},
            {profile.regions[0].structural_signature},
        )
        region_characters = [
            item
            for item in profile.characters
            if item.generator == "fork_join_region.v1"
        ]
        self.assertEqual(len(region_characters), 1)
        self.assertEqual(region_characters[0].count, 2)

    def test_multi_value_output_is_not_a_discovered_region(self):
        profile = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(1, "left", "call_function", "torch.relu", ("x",)),
                node(2, "right", "call_function", "torch.sigmoid", ("x",)),
                node(3, "out", "output", "output", ("left", "right")),
            )
        )

        self.assertEqual(profile.regions, ())

    def test_position_aware_edges_retain_repeated_inputs(self):
        graph = computation_graph_from_fx(
            (
                node(0, "x", "placeholder", "x"),
                {
                    "index": 1,
                    "input_edges": [
                        {"position": "args[0]", "source": "x"},
                        {"position": "args[1]", "source": "x"},
                    ],
                    "inputs": ["x", "x"],
                    "name": "pair",
                    "op": "call_function",
                    "target": "vendor.pair",
                },
                node(2, "out", "output", "output", ("pair",)),
            )
        )

        pair_edges = [edge for edge in graph.edges if edge.target == "n000001"]
        self.assertEqual(len(pair_edges), 2)
        self.assertEqual(
            {edge.position for edge in pair_edges}, {"args[0]", "args[1]"}
        )

    def test_round_trip_verifies_content_digest(self):
        profile = chain("torch.relu", "vendor.unknown")
        serialized = profile.to_dict()
        self.assertEqual(
            serialized["artifact_type"],
            "phylodigy.computation_graph_profile",
        )
        self.assertEqual(serialized["profile_version"], "2")
        restored = GraphProfile.from_dict(serialized)

        self.assertEqual(restored, profile)
        self.assertEqual(restored.digest, profile.digest)
        tampered = profile.to_dict()
        tampered["frontend"] = "tampered"
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            GraphProfile.from_dict(tampered)

    def test_derived_fingerprint_is_checked_against_graph(self):
        profile = chain("torch.relu")
        raw = profile.to_dict(include_digest=False)
        raw["fingerprint"] = StructuralFingerprint(
            (
                FingerprintScale(0, {"not-the-real-feature": 2}),
                *profile.fingerprint.scales[1:],
            )
        ).to_dict()

        with self.assertRaisesRegex(ValueError, "fingerprint does not match"):
            GraphProfile.from_dict(raw)

    def test_weighted_l1_fingerprint_distance_is_symmetric_and_triangular(self):
        a = chain()
        b = chain("torch.relu")
        c = chain("torch.relu", "torch.sigmoid")

        ab = compare_graphs(a, b).distance
        ba = compare_graphs(b, a).distance
        bc = compare_graphs(b, c).distance
        ac = compare_graphs(a, c).distance

        self.assertEqual(ab, ba)
        self.assertEqual(compare_graphs(a, a).distance, 0.0)
        self.assertLessEqual(ac, ab + bc)

        description = compare_graphs(a, b).to_dict()
        self.assertEqual(
            description["space"], "dynamic_character_metric_graph_pseudometric"
        )
        self.assertEqual(description["left_digest"], a.graph.digest)
        self.assertEqual(description["right_digest"], b.graph.digest)
        self.assertEqual(len(description["distance_signature"]), 64)
        self.assertEqual(
            compare_graphs(a, b).distance_signature,
            compare_graphs(b, a).distance_signature,
        )
        self.assertEqual(
            description["radius_weights"],
            {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0},
        )
        self.assertNotEqual(
            compare_graphs(a, b, radii=(0,)).distance_signature,
            compare_graphs(a, b, radii=(0, 1)).distance_signature,
        )

    def test_commutative_edge_port_spelling_does_not_change_fingerprint(self):
        nodes = (
            GraphNode("input", "graph.input"),
            GraphNode("left", "function.left"),
            GraphNode("right", "function.right"),
            GraphNode(
                "sum",
                "function.vendor.combine",
                attributes=COMMUTATIVE_ATTRIBUTES,
            ),
        )
        fixed = (
            GraphEdge("input", "left", "args[0]"),
            GraphEdge("input", "right", "args[0]"),
        )
        left = ComputationGraph(
            nodes,
            fixed
            + (
                GraphEdge("left", "sum", "operand:000000"),
                GraphEdge("right", "sum", "operand:000001"),
            ),
        )
        right = ComputationGraph(
            nodes,
            fixed
            + (
                GraphEdge("left", "sum", "operand:000001"),
                GraphEdge("right", "sum", "operand:000000"),
            ),
        )

        self.assertEqual(compare_graphs(left, right).distance, 0.0)

    def test_distance_requires_positive_finite_coordinate_weights(self):
        a = chain()
        b = chain("torch.relu")
        for weight in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, "strictly positive"):
                    compare_graphs(a, b, radii=(0,), radius_weights={0: weight})

    def test_finite_radius_fingerprint_is_explicitly_a_graph_pseudometric(self):
        serial = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(1, "a", "call_function", "vendor.step", ("x",)),
                node(2, "b", "call_function", "vendor.step", ("a",)),
                node(3, "out", "output", "output", ("b",)),
            )
        )
        parallel = build_graph_profile(
            (
                node(0, "x", "placeholder", "x"),
                node(1, "a", "call_function", "vendor.step", ("x",)),
                node(2, "b", "call_function", "vendor.step", ("x",)),
                node(3, "out", "output", "output", ("a", "b")),
            )
        )

        self.assertNotEqual(serial.graph.digest, parallel.graph.digest)
        # Radius zero only counts operator labels, so these distinct graphs collide.
        self.assertEqual(compare_graphs(serial, parallel, radii=(0,)).distance, 0.0)

    def test_invalid_reference_and_cycle_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown input"):
            computation_graph_from_fx((node(0, "out", "output", "output", ("missing",)),))
        with self.assertRaisesRegex(ValueError, "acyclic"):
            computation_graph_from_fx(
                (
                    node(0, "a", "call_function", "a", ("b",)),
                    node(1, "b", "call_function", "b", ("a",)),
                )
            )


if __name__ == "__main__":
    unittest.main()
