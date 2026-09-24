from __future__ import annotations

import copy
import dataclasses
import unittest

from phylodigy.computation_graph import (
    ComputationGraph,
    GraphEdge,
    GraphNode,
    build_graph_profile,
)
from phylodigy.graph_alignment import (
    AddEdgeOperation,
    AddNodeOperation,
    AlignmentParameters,
    AlignmentResourceLimitError,
    EdgeMatch,
    GraphAlignment,
    NodeMatch,
    RemoveEdgeOperation,
    RemoveNodeOperation,
    StructuralEdge,
    StructuralNode,
    align_graphs,
    apply_graph_alignment,
    validate_graph_alignment,
)


def chain(*operations, observations=None, attributes=None):
    records = [{"name": "input", "op": "placeholder", "target": "input"}]
    previous = "input"
    for index, operation in enumerate(operations, 1):
        name = f"node_{index}"
        record = {
            "inputs": [previous],
            "name": name,
            "op": "call_function",
            "target": operation,
        }
        if observations and index in observations:
            record["observations"] = observations[index]
        if attributes and index in attributes:
            record["attributes"] = attributes[index]
        records.append(record)
        previous = name
    records.append(
        {"name": "output", "op": "output", "target": "output", "inputs": [previous]}
    )
    return build_graph_profile(records)


def fork(*, renamed=False):
    names = (
        ("source", "right", "left", "merge", "result")
        if renamed
        else ("x", "a", "b", "join", "out")
    )
    source, left, right, join, output = names
    records = [
        {"name": source, "op": "placeholder", "target": "input"},
        {
            "name": left,
            "op": "call_function",
            "target": "vendor.alpha",
            "inputs": [source],
        },
        {
            "name": right,
            "op": "call_function",
            "target": "vendor.alpha",
            "inputs": [source],
        },
        {
            "attributes": {"operator_properties": {"commutative": True}},
            "name": join,
            "op": "call_function",
            "target": "vendor.join",
            "inputs": [left, right],
        },
        {"name": output, "op": "output", "target": "output", "inputs": [join]},
    ]
    if renamed:
        records[1], records[2] = records[2], records[1]
    return build_graph_profile(records)


class PrimitiveGraphIndelTests(unittest.TestCase):
    def test_identity_is_empty_replayable_and_round_trips(self):
        graph = chain("vendor.alpha", "vendor.beta")

        script = align_graphs(graph, graph)

        self.assertEqual(script.operations, ())
        self.assertEqual(script.total_cost, 0.0)
        self.assertEqual(script.lower_bound, 0.0)
        self.assertTrue(script.bound_closed)
        self.assertTrue(
            validate_graph_alignment(graph, graph, script)[
                "objective_optimality_proven"
            ]
        )
        self.assertEqual(
            apply_graph_alignment(graph, script).structural_digest,
            graph.graph.structural_digest,
        )
        self.assertEqual(GraphAlignment.from_dict(script.to_dict()), script)

    def test_chain_splice_tracks_one_node_and_three_edge_indels(self):
        left = chain("vendor.alpha")
        right = chain("vendor.alpha", "vendor.inserted")

        script = align_graphs(left, right)

        kinds = [operation.kind for operation in script.operations]
        self.assertEqual(kinds.count("add_node"), 1)
        self.assertEqual(kinds.count("remove_node"), 0)
        self.assertEqual(kinds.count("remove_edge"), 1)
        self.assertEqual(kinds.count("add_edge"), 2)
        self.assertEqual(script.total_cost, 4.0)
        self.assertEqual(script.lower_bound, 4.0)
        self.assertTrue(script.bound_closed)
        self.assertEqual(
            script.to_dict()["replay_direction"]["ancestral_interpretation"],
            "none",
        )
        self.assertEqual(
            apply_graph_alignment(left, script).structural_digest,
            right.graph.structural_digest,
        )

    def test_reverse_comparison_is_the_exact_equal_cost_inverse(self):
        left = chain("vendor.alpha")
        right = chain("vendor.alpha", "vendor.inserted")

        forward = align_graphs(left, right)
        reverse = align_graphs(right, left)

        self.assertEqual(reverse, forward.invert())
        self.assertEqual(reverse.invert(), forward)
        self.assertEqual(reverse.total_cost, forward.total_cost)
        self.assertEqual(
            apply_graph_alignment(right, reverse).structural_digest,
            left.graph.structural_digest,
        )
        self.assertEqual(
            {operation.kind for operation in reverse.operations},
            {"remove_node", "remove_edge", "add_edge"},
        )

    def test_edge_only_change_is_one_primitive_edge_indel(self):
        left_records = [
            {"name": "x", "op": "placeholder", "target": "x"},
            {
                "name": "a",
                "op": "call_function",
                "target": "vendor.alpha",
                "inputs": ["x"],
            },
            {
                "name": "b",
                "op": "call_function",
                "target": "vendor.beta",
                "inputs": ["a"],
            },
            {"name": "out", "op": "output", "target": "out", "inputs": ["b"]},
        ]
        right_records = copy.deepcopy(left_records)
        right_records[2]["inputs"] = ["a", "x"]
        left = build_graph_profile(left_records)
        right = build_graph_profile(right_records)

        script = align_graphs(left, right)

        self.assertFalse(any(isinstance(operation, (AddNodeOperation, RemoveNodeOperation)) for operation in script.operations))
        self.assertEqual(sum(isinstance(operation, AddEdgeOperation) for operation in script.operations), 1)
        self.assertEqual(sum(isinstance(operation, RemoveEdgeOperation) for operation in script.operations), 0)
        self.assertEqual(script.total_cost, 1.0)

    def test_changed_operator_is_delete_insert_not_a_substitution(self):
        left = chain("vendor.alpha")
        right = chain("vendor.beta")

        script = align_graphs(left, right)
        kinds = [operation.kind for operation in script.operations]

        self.assertEqual(kinds.count("remove_node"), 1)
        self.assertEqual(kinds.count("add_node"), 1)
        self.assertEqual(kinds.count("remove_edge"), 2)
        self.assertEqual(kinds.count("add_edge"), 2)
        self.assertNotIn("operator_substitution", kinds)
        self.assertEqual(script.total_cost, 6.0)

    def test_changed_static_attributes_are_also_explicit_indels(self):
        left = chain("vendor.alpha", attributes={1: {"axis": 1}})
        right = chain("vendor.alpha", attributes={1: {"axis": 2}})

        kinds = [operation.kind for operation in align_graphs(left, right).operations]

        self.assertEqual(kinds.count("remove_node"), 1)
        self.assertEqual(kinds.count("add_node"), 1)

    def test_observations_names_and_emission_order_do_not_create_indels(self):
        observed_left = chain(
            "vendor.alpha",
            observations={1: {"tensor_meta": {"shape": [1, 4]}}},
        )
        observed_right = chain(
            "vendor.alpha",
            observations={1: {"tensor_meta": {"shape": [8, 4]}, "name": "semantic"}},
        )
        first_fork = fork()
        renamed_fork = fork(renamed=True)

        self.assertEqual(align_graphs(observed_left, observed_right).operations, ())
        self.assertEqual(first_fork.graph.structural_digest, renamed_fork.graph.structural_digest)
        self.assertEqual(align_graphs(first_fork, renamed_fork).operations, ())

    def test_repeated_labels_are_reported_as_correspondence_ambiguity(self):
        graph = fork()

        script = align_graphs(graph, graph)

        self.assertEqual(script.operations, ())
        self.assertTrue(script.solver_mapping_underidentified)
        self.assertTrue(script.underidentified_mappings)
        self.assertTrue(any(len(item.left_refs) == 2 for item in script.underidentified_mappings))
        self.assertEqual(align_graphs(graph, graph), script.invert())

    def test_custom_costs_are_symmetric_and_pinned(self):
        parameters = AlignmentParameters(node_indel_cost=2, edge_indel_cost=0.5)
        left = chain("vendor.alpha")
        right = chain("vendor.alpha", "vendor.inserted")

        script = align_graphs(left, right, parameters=parameters)

        self.assertEqual(script.total_cost, 3.5)
        self.assertEqual(script.parameters.cost_model_digest, script.to_dict()["cost_model_digest"])
        self.assertEqual(align_graphs(right, left, parameters=parameters), script.invert())

    def test_bounds_never_overstate_solver_guarantees(self):
        left = chain("vendor.alpha", "vendor.beta")
        right = chain("vendor.beta", "vendor.alpha")

        script = align_graphs(left, right)

        self.assertLessEqual(script.lower_bound, script.total_cost)
        self.assertEqual(script.upper_bound, script.total_cost)
        self.assertFalse(script.bound_closed)
        self.assertFalse(
            validate_graph_alignment(left, right, script)[
                "objective_optimality_proven"
            ]
        )
        self.assertGreater(script.representative_traceback_tie_count, 0)
        self.assertNotIn("distance", script.to_dict())
        self.assertNotIn("ancestry", script.to_dict())

    def test_matrix_budget_fails_explicitly(self):
        with self.assertRaisesRegex(AlignmentResourceLimitError, "budget"):
            align_graphs(
                chain("vendor.alpha"),
                chain("vendor.beta"),
                parameters=AlignmentParameters(max_matrix_cells=1),
            )

    def test_identity_bypasses_the_quadratic_matrix_budget(self):
        graph = chain("vendor.alpha", "vendor.beta")

        script = align_graphs(
            graph,
            graph,
            parameters=AlignmentParameters(max_matrix_cells=1),
        )

        self.assertEqual(script.operations, ())
        self.assertTrue(script.bound_closed)

    def test_noncanonical_direct_graphs_fail_instead_of_using_node_ids(self):
        canonical = chain("vendor.alpha").graph
        renamed = ComputationGraph(
            nodes=tuple(
                GraphNode(
                    {"n000000": "z", "n000001": "x", "n000002": "y"}[node.node_id],
                    node.operation,
                    node.attributes,
                )
                for node in canonical.nodes
            ),
            edges=tuple(
                GraphEdge(
                    {"n000000": "z", "n000001": "x", "n000002": "y"}[edge.source],
                    {"n000000": "z", "n000001": "x", "n000002": "y"}[edge.target],
                    edge.position,
                )
                for edge in canonical.edges
            ),
        )

        with self.assertRaisesRegex(ValueError, "canonical node labeling"):
            align_graphs(canonical, renamed)

    def test_serial_repetition_is_not_called_structurally_underidentified(self):
        graph = chain("vendor.alpha", "vendor.alpha")

        script = align_graphs(graph, graph)

        self.assertFalse(script.solver_mapping_underidentified)
        self.assertEqual(script.underidentified_mappings, ())

    def test_exact_component_counts_prevent_float_false_proofs(self):
        left = chain("vendor.alpha", "vendor.beta")
        right = chain("vendor.beta", "vendor.alpha")
        parameters = AlignmentParameters(
            node_indel_cost=1e-13,
            edge_indel_cost=1e-13,
        )

        script = align_graphs(left, right, parameters=parameters)

        self.assertFalse(script.bound_closed)
        self.assertFalse(
            validate_graph_alignment(left, right, script)[
                "objective_optimality_proven"
            ]
        )
        self.assertNotIn("optimality_proven", script.to_dict())

    def test_aggregate_cost_overflow_is_rejected_before_serialization(self):
        with self.assertRaisesRegex(ValueError, "total cost must be finite"):
            align_graphs(
                chain("vendor.alpha"),
                chain("vendor.beta"),
                parameters=AlignmentParameters(
                    node_indel_cost=1e308,
                    edge_indel_cost=1e308,
                ),
            )

    def test_duplicate_operations_are_rejected_not_collapsed_during_replay(self):
        script = align_graphs(
            chain("vendor.alpha"),
            chain("vendor.alpha", "vendor.inserted"),
        )

        with self.assertRaisesRegex(ValueError, "operations must be unique"):
            dataclasses.replace(
                script,
                operations=script.operations + (script.operations[0],),
            )

    def test_matched_objects_cannot_also_be_removed(self):
        graph = chain("vendor.alpha")
        alignment = align_graphs(graph, graph)
        node_by_ref = {node.node_id: node for node in graph.graph.nodes}
        edge_by_ref = {edge.edge_id: edge for edge in graph.graph.edges}

        with self.assertRaisesRegex(ValueError, "both matched and removed"):
            dataclasses.replace(
                alignment,
                operations=(
                    RemoveNodeOperation(
                        StructuralNode.from_graph_node(
                            node_by_ref[alignment.node_matches[0].left_ref]
                        ),
                        alignment.parameters.node_indel_cost,
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "both matched and removed"):
            dataclasses.replace(
                alignment,
                operations=(
                    RemoveEdgeOperation(
                        StructuralEdge.from_graph_edge(
                            edge_by_ref[alignment.edge_matches[0].left.ref]
                        ),
                        alignment.parameters.edge_indel_cost,
                    ),
                ),
            )

    def test_replay_rejects_noncanonical_source_and_result_graphs(self):
        canonical = chain("vendor.alpha").graph
        renamed_refs = {
            node.node_id: f"source_{index}"
            for index, node in enumerate(reversed(canonical.nodes))
        }
        renamed = ComputationGraph(
            nodes=tuple(
                GraphNode(
                    renamed_refs[node.node_id],
                    node.operation,
                    node.attributes,
                )
                for node in canonical.nodes
            ),
            edges=tuple(
                GraphEdge(
                    renamed_refs[edge.source],
                    renamed_refs[edge.target],
                    edge.position,
                )
                for edge in canonical.edges
            ),
        )
        base = align_graphs(canonical, canonical)

        renamed_matches = tuple(
            NodeMatch(
                renamed_refs[match.left_ref],
                renamed_refs[match.right_ref],
                match.structural_signature,
            )
            for match in base.node_matches
        )
        renamed_edge_matches = tuple(
            EdgeMatch(
                StructuralEdge(
                    renamed_refs[match.left.source],
                    renamed_refs[match.left.target],
                    match.left.position,
                ),
                StructuralEdge(
                    renamed_refs[match.right.source],
                    renamed_refs[match.right.target],
                    match.right.position,
                ),
                match.canonical_port_label,
            )
            for match in base.edge_matches
        )
        noncanonical_identity = dataclasses.replace(
            base,
            left_structural_digest=renamed.structural_digest,
            right_structural_digest=renamed.structural_digest,
            node_matches=renamed_matches,
            edge_matches=renamed_edge_matches,
        )
        with self.assertRaisesRegex(ValueError, "canonical node labeling"):
            apply_graph_alignment(renamed, noncanonical_identity)

        canonical_to_noncanonical = dataclasses.replace(
            base,
            right_structural_digest=renamed.structural_digest,
            node_matches=tuple(
                NodeMatch(
                    match.left_ref,
                    renamed_refs[match.right_ref],
                    match.structural_signature,
                )
                for match in base.node_matches
            ),
            edge_matches=tuple(
                EdgeMatch(
                    match.left,
                    StructuralEdge(
                        renamed_refs[match.right.source],
                        renamed_refs[match.right.target],
                        match.right.position,
                    ),
                    match.canonical_port_label,
                )
                for match in base.edge_matches
            ),
        )
        with self.assertRaisesRegex(ValueError, "canonical node labeling"):
            apply_graph_alignment(canonical, canonical_to_noncanonical)

    def test_deserialization_requires_the_exact_serialized_shape(self):
        alignment = align_graphs(
            chain("vendor.alpha"),
            chain("vendor.alpha", "vendor.inserted"),
        )
        missing_collection = alignment.to_dict()
        missing_collection.pop("edge_matches")
        unknown_member = alignment.to_dict()
        unknown_member["future_guess"] = True
        missing_nested_ref = alignment.to_dict()
        edge_operation = next(
            operation
            for operation in missing_nested_ref["operations"]
            if "edge" in operation
        )
        edge_operation["edge"].pop("ref")

        for raw in (missing_collection, unknown_member, missing_nested_ref):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    GraphAlignment.from_dict(raw)

    def test_deserialization_rejects_bool_number_type_confusion(self):
        alignment = align_graphs(chain("vendor.alpha"), chain("vendor.alpha"))
        raw = alignment.to_dict()
        raw["bounds"]["closed"] = 1
        raw["bounds"]["lower"] = False
        raw["bounds"]["upper"] = False
        raw["total_cost"] = False

        with self.assertRaisesRegex(ValueError, "bounds mismatch"):
            GraphAlignment.from_dict(raw)

    def test_malformed_collections_raise_type_errors_before_sorting(self):
        alignment = align_graphs(chain("vendor.alpha"), chain("vendor.alpha"))

        for field in (
            "node_matches",
            "edge_matches",
            "underidentified_mappings",
            "operations",
        ):
            with self.subTest(field=field):
                with self.assertRaises(TypeError):
                    dataclasses.replace(alignment, **{field: (object(),)})

    def test_deserialized_bound_claim_requires_endpoint_validation(self):
        left = chain("vendor.alpha", "vendor.beta")
        right = chain("vendor.beta", "vendor.alpha")
        script = align_graphs(left, right)
        manufactured = dataclasses.replace(
            script,
            lower_bound_node_operations=script.node_operation_count,
            lower_bound_edge_operations=script.edge_operation_count,
        )

        self.assertTrue(manufactured.bound_closed)
        with self.assertRaisesRegex(ValueError, "lower-bound components"):
            validate_graph_alignment(left, right, manufactured)

    def test_cost_and_solver_configuration_have_separate_identities(self):
        first = AlignmentParameters(max_matrix_cells=100)
        second = AlignmentParameters(max_matrix_cells=200)

        self.assertEqual(first.cost_model_digest, second.cost_model_digest)
        self.assertNotEqual(
            first.solver_configuration_digest,
            second.solver_configuration_digest,
        )

    def test_replay_and_deserialization_reject_tampering(self):
        left = chain("vendor.alpha")
        right = chain("vendor.alpha", "vendor.inserted")
        script = align_graphs(left, right)
        raw = script.to_dict()
        raw["total_cost"] += 1

        with self.assertRaisesRegex(ValueError, "total_cost mismatch"):
            GraphAlignment.from_dict(raw)
        with self.assertRaisesRegex(ValueError, "left endpoint"):
            apply_graph_alignment(right, script)

    def test_cost_model_validation_is_strict(self):
        for kwargs in (
            {"node_indel_cost": 0},
            {"edge_indel_cost": float("nan")},
            {"max_matrix_cells": 0},
            {"max_matrix_cells": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    AlignmentParameters(**kwargs)


if __name__ == "__main__":
    unittest.main()
