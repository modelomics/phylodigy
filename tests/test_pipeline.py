from __future__ import annotations

import json
import unittest

from phylodigy.citations import CitationTarget, extract_citation_evidence
from phylodigy.code_profile import extract_source_profile
from phylodigy.computation_graph import build_graph_profile
from phylodigy.contact_network import ContactEvidence, infer_contact_network
from phylodigy.lineage import infer_lineage_network
from phylodigy.paper_profile import (
    GraphAnnotation,
    GraphAnnotationTarget,
    annotate_graph_from_paper,
    extract_paper_profile,
)
from phylodigy.schema import ArchitecturalGenome


def chain_genome(artifact_id, *operations, date=None):
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


def fork_genome(artifact_id, date=None):
    profile = build_graph_profile(
        (
            {"index": 0, "name": "x", "op": "placeholder", "target": "x"},
            {
                "index": 1,
                "inputs": ["x"],
                "name": "left",
                "op": "call_function",
                "target": "vendor.alpha",
            },
            {
                "index": 2,
                "inputs": ["x"],
                "name": "right",
                "op": "call_function",
                "target": "vendor.beta",
            },
            {
                "index": 3,
                "inputs": ["left", "right"],
                "name": "join",
                "op": "call_function",
                "target": "vendor.combine",
            },
            {
                "index": 4,
                "inputs": ["join"],
                "name": "out",
                "op": "output",
                "target": "output",
            },
        )
    )
    return ArchitecturalGenome(artifact_id, profile, release_date=date)


class EndToEndGraphPipelineTests(unittest.TestCase):
    def test_graph_discovery_then_paper_naming_then_phylogeny(self):
        ancestor = chain_genome("model:ancestor", "vendor.alpha", date="2020-01-01")
        descendant = fork_genome("model:descendant", date="2021-01-01")
        self.assertEqual(len(descendant.graph_profile.regions), 1)

        text = """Methods
We call this structure the Lantern Bridge [1].

References
[1] Example, A. Earlier Graph. arXiv:2305.13245.
"""
        paper = extract_paper_profile(text, artifact_id="paper:descendant")
        sentence = "We call this structure the Lantern Bridge [1]."
        start = paper.normalized_text.index(sentence)
        end = start + len(sentence)
        target = GraphAnnotationTarget.resolve(
            descendant.graph_profile,
            kind="region",
            target_id=descendant.graph_profile.regions[0].occurrence_id,
        )
        annotation = GraphAnnotation(
            target=target,
            start=start,
            end=end,
            excerpt=sentence,
            source_id=paper.source_id,
            source_digest=paper.source_digest,
            name="Lantern Bridge",
            citations=("marker:1",),
        )
        overlay = annotate_graph_from_paper(
            descendant.graph_profile,
            paper,
            (annotation,),
        )

        self.assertEqual(overlay.annotations[0].name, "Lantern Bridge")
        self.assertEqual(
            overlay.annotations[0].target.structure_digest,
            target.structure_digest,
        )
        self.assertNotIn("Lantern Bridge", descendant.graph_profile.canonical_json())

        plain = infer_lineage_network((ancestor, descendant))
        explained = infer_lineage_network(
            (ancestor, descendant),
            evidence=(overlay.to_dict(),),
        )
        self.assertEqual(plain["comparisons"], explained["comparisons"])
        self.assertEqual(plain["tree"], explained["tree"])
        self.assertNotEqual(plain["digest"], explained["digest"])

    def test_citation_resolution_stays_semantics_free_until_graph_annotation(self):
        text = """Methods
We use an earlier component [1].

References
[1] Example, A. Earlier Graph. arXiv:2305.13245.
"""
        mentions, report = extract_citation_evidence(
            text,
            (CitationTarget("paper:earlier", arxiv_id="2305.13245"),),
        )

        self.assertEqual(mentions[0].target_ids, ("paper:earlier",))
        self.assertEqual(report["structural_claims_created"], 0)
        self.assertNotIn("causal_role", mentions[0].to_dict())

    def test_grounded_character_contact_is_separate_from_structural_distance(self):
        source = chain_genome("model:source", "vendor.alpha", date="2020-01-01")
        target = chain_genome(
            "model:target", "vendor.alpha", "vendor.beta", date="2021-01-01"
        )
        shared_id = next(
            iter(sorted(set(source.character_counts) & set(target.character_counts)))
        )
        evidence = ContactEvidence(
            source.artifact_id,
            target.artifact_id,
            character_id=shared_id,
            character_value=target.character_counts[shared_id],
            causal_role="character_transfer",
            evidence_tier="paper",
        )
        without = infer_contact_network((source, target))
        with_evidence = infer_contact_network(
            (source, target),
            edge_evidence=(evidence,),
        )

        without_diagnostic = next(
            item
            for item in without["diagnostics"]["pair_diagnostics"]
            if item["source_id"] == source.artifact_id
            and item["target_id"] == target.artifact_id
        )
        with_diagnostic = next(
            item
            for item in with_evidence["diagnostics"]["pair_diagnostics"]
            if item["source_id"] == source.artifact_id
            and item["target_id"] == target.artifact_id
        )
        self.assertEqual(
            without_diagnostic["graph_distance"],
            with_diagnostic["graph_distance"],
        )
        self.assertEqual(without["edges"], [])
        self.assertEqual(with_evidence["edges"][0]["edge_type"], "transfer")

    def test_static_source_manifest_cannot_masquerade_as_a_genome(self):
        manifest = extract_source_profile(
            {"model.py": "class TransformerAttentionResidual: pass\n"},
            artifact_id="model:source-only",
        )
        genome = chain_genome("model:graph", "vendor.alpha", date="2020-01-01")

        self.assertNotIn("graph_profile", manifest.to_dict())
        with self.assertRaises((TypeError, AttributeError)):
            infer_lineage_network((manifest, genome))

    def test_named_semantics_are_absent_from_every_structural_artifact(self):
        traced = fork_genome("model:named", date="2021-01-01")
        model = ArchitecturalGenome(
            traced.artifact_id,
            traced.graph_profile,
            name="Transformer ResNet Attention",
            release_date=traced.release_date,
        )
        lineage = infer_lineage_network(
            (chain_genome("model:base", "vendor.alpha", date="2020-01-01"), model)
        )

        structural = json.dumps(
            {
                "graph_profile": model.graph_profile.to_dict(),
                "character_matrix": lineage["character_matrix"],
                "comparisons": lineage["comparisons"],
            }
        ).lower()
        for forbidden in ("transformer", "resnet", "attention", "ontology", "traits"):
            self.assertNotIn(forbidden, structural)


if __name__ == "__main__":
    unittest.main()
