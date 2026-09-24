from __future__ import annotations

import json
import unittest

from phylodigy.computation_graph import build_graph_profile
from phylodigy.contact_network import (
    ContactEvidence,
    ContactInferenceConfig,
    infer_contact_network,
)
from phylodigy.schema import ArchitecturalGenome


def genome(artifact_id, *operations, date=None):
    records = [
        {"index": 0, "name": "input", "op": "placeholder", "target": "input"}
    ]
    previous = "input"
    for index, operation in enumerate(operations, 1):
        name = f"operation_{index}"
        records.append(
            {
                "index": index,
                "inputs": [previous],
                "name": name,
                "op": "call_function",
                "target": operation,
            }
        )
        previous = name
    records.append(
        {
            "index": len(records),
            "inputs": [previous],
            "name": "output",
            "op": "output",
            "target": "output",
        }
    )
    return ArchitecturalGenome(
        artifact_id,
        build_graph_profile(records),
        release_date=date,
    )


class ContactEvidenceTests(unittest.TestCase):
    def test_character_transfer_requires_a_graph_character_id(self):
        with self.assertRaisesRegex(ValueError, "requires character_id"):
            ContactEvidence("a", "b", causal_role="character_transfer")

        evidence = ContactEvidence(
            "a",
            "b",
            character_id="a" * 64,
            character_value=2,
            causal_role="character_transfer",
        )
        raw = evidence.to_dict()
        self.assertEqual(raw["character_id"], "a" * 64)
        self.assertNotIn("trait_id", raw)
        self.assertNotIn("trait_value", raw)

    def test_fixed_trait_keys_and_roles_are_not_accepted_as_aliases(self):
        with self.assertRaisesRegex(ValueError, "requires character_id"):
            ContactEvidence.from_mapping(
                {
                    "source_id": "a",
                    "target_id": "b",
                    "trait_id": "architecture.named_feature",
                    "causal_role": "character_transfer",
                }
            )
        with self.assertRaisesRegex(ValueError, "causal_role"):
            ContactEvidence.from_mapping(
                {
                    "source_id": "a",
                    "target_id": "b",
                    "causal_role": "trait_transfer",
                }
            )

    def test_config_cannot_turn_graph_similarity_into_contact(self):
        config = ContactInferenceConfig(include_unlinked_pair_diagnostics=False)
        self.assertFalse(config.include_unlinked_pair_diagnostics)
        self.assertFalse(hasattr(config, "trait_priors"))
        self.assertFalse(hasattr(config, "graph_similarity_weight"))
        with self.assertRaises(ValueError):
            ContactInferenceConfig(include_unlinked_pair_diagnostics=1)


class GraphContactNetworkTests(unittest.TestCase):
    def setUp(self):
        self.a = genome("model:a", "vendor.alpha", date="2020-01-01")
        self.b = genome(
            "model:b", "vendor.alpha", "vendor.beta", date="2021-01-01"
        )
        self.c = genome(
            "model:c",
            "vendor.alpha",
            "vendor.beta",
            "vendor.gamma",
            date="2022-01-01",
        )

    def test_graph_similarity_alone_creates_no_contact_or_parent_edge(self):
        result = infer_contact_network((self.c, self.a, self.b))

        self.assertEqual(result["artifact_type"], "phylodigy.graph_contact_network")
        self.assertEqual(result["primary_parent_forest"], [])
        self.assertEqual(result["edges"], [])
        self.assertTrue(result["diagnostics"]["pair_diagnostics"])
        self.assertTrue(
            all(
                item["origin_hypothesis"] == "first_observed"
                for item in result["character_sources"]
            )
        )
        self.assertEqual(result["method"]["contact_rule"], "explicit grounded evidence only")
        text = json.dumps(result)
        self.assertNotIn('"traits"', text)
        self.assertNotIn('"ontology"', text)

        compact = infer_contact_network(
            (self.a, self.b),
            config=ContactInferenceConfig(include_unlinked_pair_diagnostics=False),
        )
        self.assertEqual(compact["diagnostics"]["pair_diagnostics"], [])
        self.assertEqual(compact["edges"], [])

    def test_missing_or_overlapping_dates_are_excluded(self):
        undated = genome("model:undated", "vendor.alpha")
        same_day = genome("model:same", "vendor.alpha", date="2020-01-01")
        result = infer_contact_network(
            (self.a, undated, same_day),
            edge_evidence=(
                ContactEvidence("model:a", "model:same"),
                ContactEvidence("model:a", "model:undated"),
            ),
        )
        reasons = {
            (item["source_id"], item["target_id"], item["reason"])
            for item in result["diagnostics"]["temporal_exclusions"]
        }

        self.assertIn(("model:a", "model:same", "not_strictly_earlier"), reasons)
        self.assertTrue(any(reason == "missing_date" for _, _, reason in reasons))

    def test_provenance_creates_contact_but_does_not_change_graph_diagnostic(self):
        evidence = ContactEvidence(
            "model:a", "model:c", strength=1, causal_role="provenance"
        )
        without_evidence = infer_contact_network((self.a, self.c))
        with_evidence = infer_contact_network(
            (self.a, self.c),
            edge_evidence=(evidence,),
        )

        self.assertEqual(without_evidence["edges"], [])
        self.assertEqual(
            [(item["edge_type"], item["source_id"], item["target_id"]) for item in with_evidence["edges"]],
            [("provenance", "model:a", "model:c")],
        )
        graph_candidate = next(
            item
            for item in without_evidence["diagnostics"]["pair_diagnostics"]
            if item["source_id"] == "model:a" and item["target_id"] == "model:c"
        )
        provenance_candidate = next(
            item
            for item in with_evidence["diagnostics"]["pair_diagnostics"]
            if item["source_id"] == "model:a" and item["target_id"] == "model:c"
        )
        self.assertEqual(
            graph_candidate["graph_distance"],
            provenance_candidate["graph_distance"],
        )

    def test_transfer_evidence_must_resolve_to_both_endpoint_graphs(self):
        arbitrary = ContactEvidence(
            "model:a",
            "model:c",
            character_id="f" * 64,
            causal_role="character_transfer",
        )
        with self.assertRaisesRegex(ValueError, "not grounded"):
            infer_contact_network((self.a, self.c), edge_evidence=(arbitrary,))

        shared_id = next(
            iter(sorted(set(self.a.character_counts) & set(self.c.character_counts)))
        )
        wrong_value = ContactEvidence(
            "model:a",
            "model:c",
            character_id=shared_id,
            character_value=self.c.character_counts[shared_id] + 1,
            causal_role="character_transfer",
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            infer_contact_network((self.a, self.c), edge_evidence=(wrong_value,))

    def test_grounded_nonparent_character_transfer_creates_lateral_edge(self):
        shared_id = next(
            iter(sorted(set(self.a.character_counts) & set(self.c.character_counts)))
        )
        evidence = ContactEvidence(
            "model:a",
            "model:c",
            character_id=shared_id,
            character_value=self.c.character_counts[shared_id],
            causal_role="character_transfer",
        )
        result = infer_contact_network(
            (self.a, self.b, self.c),
            edge_evidence=(evidence,),
        )
        transfer = [edge for edge in result["edges"] if edge["edge_type"] == "transfer"]

        self.assertEqual(len(transfer), 1)
        self.assertEqual(transfer[0]["source_id"], "model:a")
        self.assertEqual(transfer[0]["target_id"], "model:c")
        self.assertEqual(transfer[0]["character_ids"], [shared_id])

    def test_awareness_is_an_explicit_contact_edge_not_a_parent(self):
        result = infer_contact_network(
            (self.a, self.b),
            edge_evidence=(
                ContactEvidence("model:a", "model:b", causal_role="awareness"),
            ),
        )

        self.assertEqual(result["primary_parent_forest"], [])
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["edges"][0]["edge_type"], "awareness")
        self.assertTrue(
            all(
                item["origin_hypothesis"] == "first_observed"
                for item in result["character_sources"]
            )
        )

    def test_temporally_inadmissible_evidence_creates_no_edge(self):
        result = infer_contact_network(
            (self.a, self.b),
            edge_evidence=(
                ContactEvidence("model:b", "model:a", causal_role="provenance"),
            ),
        )

        self.assertEqual(result["edges"], [])
        self.assertEqual(result["primary_parent_forest"], [])
        self.assertEqual(
            result["diagnostics"]["temporal_exclusions"][0]["reason"],
            "not_strictly_earlier",
        )

    def test_input_and_evidence_order_are_deterministic(self):
        evidence = (
            ContactEvidence("model:a", "model:b", causal_role="provenance"),
            ContactEvidence("model:b", "model:c", causal_role="provenance"),
        )
        first = infer_contact_network((self.a, self.b, self.c), edge_evidence=evidence)
        second = infer_contact_network(
            (self.c, self.a, self.b), edge_evidence=reversed(evidence)
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
