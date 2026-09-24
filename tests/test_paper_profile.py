from __future__ import annotations

import copy
import unittest

from phylodigy.canonical import content_digest
from phylodigy.computation_graph import build_graph_profile
from phylodigy.paper_profile import (
    GraphAnnotation,
    GraphAnnotationTarget,
    PaperDocument,
    PaperGraphAnnotations,
    annotate_graph_from_paper,
    extract_paper_profile,
    normalize_extracted_text,
    split_sections,
    split_sentences,
)


def _fork_join_profile(
    *,
    renamed: bool = False,
    batch_size: int | None = None,
    radii=(0, 1, 2, 3),
    frontend: str = "test",
    metadata=None,
    second_target: str = "vendor.transform_b",
):
    names = {
        "input": "source" if renamed else "x",
        "first": "alpha" if renamed else "first",
        "second": "beta" if renamed else "second",
        "join": "combine" if renamed else "join",
        "output": "result" if renamed else "output",
    }
    records = [
            {
                "name": names["input"],
                "op": "placeholder",
                "target": names["input"],
                "inputs": [],
            },
            {
                "name": names["first"],
                "op": "call_function",
                "target": "vendor.transform_a",
                "inputs": [names["input"]],
            },
            {
                "name": names["second"],
                "op": "call_function",
                "target": second_target,
                "inputs": [names["first"]],
            },
            {
                "name": names["join"],
                "op": "call_function",
                "target": "operator.add",
                "inputs": [names["input"], names["second"]],
            },
            {
                "name": names["output"],
                "op": "output",
                "target": "output",
                "inputs": [names["join"]],
            },
        ]
    if batch_size is not None:
        for index, record in enumerate(records):
            record["observations"] = {"probe_run": f"batch-{batch_size}-{index}"}
            record["tensor_meta"] = {"shape": [batch_size, 8]}
    return build_graph_profile(
        records,
        frontend=frontend,
        metadata=metadata or {},
        radii=radii,
    )


PAPER_TEXT = """Architecture
We call this a braided bridge. We derive the braided bridge from the earlier split-path design [12]. A braided bridge passes one stream unchanged while another passes through two transformations before their sum.

The optimizer schedule is inherited from prior work [13].
"""


def _annotation_claim(paper: PaperDocument, target: GraphAnnotationTarget):
    start = paper.normalized_text.index("We call")
    end = len(paper.normalized_text)
    return {
        "target": target,
        "name": "braided bridge",
        "description": (
            "A braided bridge passes one stream unchanged while another passes "
            "through two transformations before their sum."
        ),
        "claimed_function": "passes one stream unchanged",
        "provenance_statement": (
            "We derive the braided bridge from the earlier split-path design [12]."
        ),
        "start": start,
        "end": end,
        "confidence": 0.93,
        "citations": ["[12]"],
        "metadata": {"annotator": "semantic-model:test"},
    }


class PaperDocumentTests(unittest.TestCase):
    def test_normalization_is_idempotent_and_offsets_are_exact(self):
        raw = (
            "\ufeffAbstract\r\n"
            "A trans-\r\nformer description.\f"
            "\n1 Methods\n\n"
            "A second  sentence.\n"
        )
        normalized = normalize_extracted_text(raw)
        self.assertEqual(
            normalized,
            "Abstract\nA transformer description.\n\n"
            "1 Methods\nA second sentence.",
        )
        self.assertEqual(normalize_extracted_text(normalized), normalized)
        sections = split_sections(normalized)
        sentences = split_sentences(normalized, sections)
        self.assertEqual([item.name for item in sections], ["Abstract", "Methods"])
        for sentence in sentences:
            self.assertEqual(
                normalized[sentence.start : sentence.end], sentence.text
            )

    def test_paper_preparation_infers_no_architecture_semantics(self):
        paper = extract_paper_profile(
            "Methods\nThis text says ResNet, attention, RMSNorm, and SwiGLU.",
            artifact_id="paper:any-terms",
            identifiers={"doi": "10.0/example"},
        )
        self.assertIsInstance(paper, PaperDocument)
        self.assertFalse(hasattr(paper, "traits"))
        self.assertFalse(hasattr(paper, "ontology_id"))
        self.assertEqual(paper.metadata["semantic_annotations"], 0)
        self.assertEqual(paper.identifiers, {"doi": "10.0/example"})

    def test_document_round_trip_verifies_source_and_artifact_digests(self):
        paper = extract_paper_profile(PAPER_TEXT, artifact_id="paper:bridge")
        raw = paper.to_dict()
        self.assertEqual(PaperDocument.from_dict(raw), paper)
        tampered = copy.deepcopy(raw)
        tampered["normalized_text"] += " altered"
        with self.assertRaisesRegex(ValueError, "source digest mismatch"):
            PaperDocument.from_dict(tampered)


class GraphAnnotationTargetTests(unittest.TestCase):
    def setUp(self):
        self.profile = _fork_join_profile()

    def test_every_dynamic_graph_object_kind_can_be_resolved(self):
        graph = self.profile.graph
        region = self.profile.regions[0]
        character = self.profile.characters[0]
        cases = (
            ("graph", graph.structural_digest),
            ("node", graph.nodes[0].node_id),
            ("edge", graph.edges[0].edge_id),
            ("region", region.occurrence_id),
            ("character", character.character_id),
        )
        for kind, target_id in cases:
            with self.subTest(kind=kind):
                target = GraphAnnotationTarget.resolve(
                    self.profile, kind=kind, target_id=target_id
                )
                self.assertEqual(target.graph_digest, graph.structural_digest)
                self.assertTrue(target.structure_digest)

    def test_invented_or_wrongly_typed_targets_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            GraphAnnotationTarget.resolve(
                self.profile, kind="region", target_id="invented"
            )
        region_id = self.profile.regions[0].occurrence_id
        with self.assertRaisesRegex(ValueError, "unknown"):
            GraphAnnotationTarget.resolve(
                self.profile, kind="node", target_id=region_id
            )

    def test_targets_are_stable_under_program_symbol_renaming(self):
        renamed = _fork_join_profile(renamed=True)
        self.assertEqual(
            self.profile.graph.structural_digest,
            renamed.graph.structural_digest,
        )
        left = GraphAnnotationTarget.resolve(
            self.profile,
            kind="region",
            target_id=self.profile.regions[0].occurrence_id,
        )
        right = GraphAnnotationTarget.resolve(
            renamed,
            kind="region",
            target_id=renamed.regions[0].occurrence_id,
        )
        self.assertEqual(left, right)

    def test_targets_exclude_tensor_and_probe_observations(self):
        first = _fork_join_profile(
            batch_size=2,
            frontend="torch.fx",
            metadata={"probe": "small"},
        )
        second = _fork_join_profile(
            batch_size=64,
            frontend="torch.export",
            metadata={"probe": "large"},
        )
        self.assertNotEqual(first.graph.digest, second.graph.digest)
        self.assertNotEqual(first.digest, second.digest)
        self.assertEqual(
            first.graph.structural_digest, second.graph.structural_digest
        )

        first_node = first.graph.nodes[1]
        second_node = second.graph.nodes[1]
        self.assertNotEqual(first_node.to_dict(), second_node.to_dict())
        self.assertEqual(first_node.structural_dict(), second_node.structural_dict())
        first_target = GraphAnnotationTarget.resolve(
            first, kind="node", target_id=first_node.node_id
        )
        second_target = GraphAnnotationTarget.resolve(
            second, kind="node", target_id=second_node.node_id
        )
        self.assertEqual(first_target, second_target)
        self.assertEqual(
            first_target.structure_digest,
            content_digest(
                {
                    "graph_digest": first.graph.structural_digest,
                    "kind": "node",
                    "node": first_node.structural_dict(),
                }
            ),
        )


class PaperGraphAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.profile = _fork_join_profile()
        self.paper = extract_paper_profile(
            PAPER_TEXT,
            artifact_id="paper:bridge",
            source_id="sha256:paper-source",
        )
        self.target = GraphAnnotationTarget.resolve(
            self.profile,
            kind="region",
            target_id=self.profile.regions[0].occurrence_id,
        )

    def test_paper_names_an_existing_anonymous_region(self):
        result = annotate_graph_from_paper(
            self.profile,
            self.paper,
            [_annotation_claim(self.paper, self.target)],
        )
        self.assertIsInstance(result, PaperGraphAnnotations)
        self.assertEqual(len(result.annotations), 1)
        annotation = result.annotations[0]
        self.assertIsInstance(annotation, GraphAnnotation)
        self.assertEqual(annotation.target, self.target)
        self.assertEqual(annotation.name, "braided bridge")
        self.assertIn("one stream unchanged", annotation.description)
        self.assertEqual(annotation.claimed_function, "passes one stream unchanged")
        self.assertIn("derive the braided bridge", annotation.provenance_statement)
        self.assertEqual(annotation.citations, ("[12]",))
        self.assertEqual(annotation.confidence, 0.93)
        self.assertEqual(
            self.paper.normalized_text[annotation.start : annotation.end],
            annotation.excerpt,
        )
        self.assertEqual(annotation.source_digest, self.paper.source_digest)
        self.assertEqual(
            result.graph_digest, self.profile.graph.structural_digest
        )

    def test_empty_layer_is_valid_and_does_not_search_paper_terms(self):
        result = annotate_graph_from_paper(self.profile, self.paper, [])
        self.assertEqual(result.annotations, ())

    def test_direct_annotation_artifact_construction_still_validates_targets(self):
        start = self.paper.normalized_text.index("braided bridge")
        end = start + len("braided bridge")
        invented = GraphAnnotation(
            target=GraphAnnotationTarget(
                kind="node",
                target_id="invented",
                structure_digest="0" * 64,
                graph_digest=self.profile.graph.structural_digest,
            ),
            start=start,
            end=end,
            excerpt="braided bridge",
            source_id=self.paper.source_id,
            source_digest=self.paper.source_digest,
            name="braided bridge",
        )

        with self.assertRaisesRegex(ValueError, "unknown"):
            PaperGraphAnnotations(
                paper=self.paper,
                graph_profile=self.profile,
                graph_digest=self.profile.graph.structural_digest,
                annotations=(invented,),
            )

    def test_semantic_fields_can_be_recorded_individually(self):
        start = self.paper.normalized_text.index("A braided bridge passes")
        end = self.paper.normalized_text.index(".\n\n", start) + 1
        result = annotate_graph_from_paper(
            self.profile,
            self.paper,
            [
                {
                    "target": self.target,
                    "claimed_function": "passes one stream unchanged",
                    "start": start,
                    "end": end,
                    "confidence": 0.8,
                }
            ],
        )
        annotation = result.annotations[0]
        self.assertEqual(annotation.name, "")
        self.assertEqual(annotation.description, "")
        self.assertEqual(annotation.claimed_function, "passes one stream unchanged")

    def test_ungrounded_paper_claims_cannot_become_structural_annotations(self):
        statement = "The optimizer schedule is inherited from prior work [13]."
        start = self.paper.normalized_text.index(statement)
        result = annotate_graph_from_paper(
            self.profile,
            self.paper,
            [],
            unverified_claims=[
                {
                    "statement": statement,
                    "start": start,
                    "end": start + len(statement),
                    "citations": ["[13]"],
                    "confidence": 0.7,
                }
            ],
        )
        self.assertEqual(result.annotations, ())
        self.assertEqual(len(result.unverified_claims), 1)
        self.assertEqual(result.unverified_claims[0].statement, statement)
        self.assertFalse(hasattr(result.unverified_claims[0], "target"))

    def test_name_and_description_must_be_verbatim_paper_evidence(self):
        claim = _annotation_claim(self.paper, self.target)
        claim["name"] = "residual block"
        with self.assertRaisesRegex(ValueError, "name must be a verbatim"):
            annotate_graph_from_paper(self.profile, self.paper, [claim])

        claim = _annotation_claim(self.paper, self.target)
        claim["description"] = "A generated architectural interpretation."
        with self.assertRaisesRegex(ValueError, "description must be a verbatim"):
            annotate_graph_from_paper(self.profile, self.paper, [claim])

    def test_spans_and_confidence_are_validated(self):
        claim = _annotation_claim(self.paper, self.target)
        claim["end"] = len(self.paper.normalized_text) + 1
        with self.assertRaisesRegex(ValueError, "outside paper text"):
            annotate_graph_from_paper(self.profile, self.paper, [claim])

        claim = _annotation_claim(self.paper, self.target)
        claim["confidence"] = 1.1
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            annotate_graph_from_paper(self.profile, self.paper, [claim])

    def test_supplied_target_digests_are_never_silently_repaired(self):
        claim = _annotation_claim(self.paper, self.target)
        raw_target = self.target.to_dict()
        raw_target["structure_digest"] = "0" * 64
        claim["target"] = raw_target
        with self.assertRaisesRegex(ValueError, "structure_digest does not match"):
            annotate_graph_from_paper(self.profile, self.paper, [claim])

    def test_annotation_round_trip_requires_the_exact_graph_profile(self):
        result = annotate_graph_from_paper(
            self.profile,
            self.paper,
            [_annotation_claim(self.paper, self.target)],
        )
        raw = result.to_dict()
        self.assertEqual(
            PaperGraphAnnotations.from_dict(raw, graph_profile=self.profile),
            result,
        )

        other = build_graph_profile(
            [
                {"name": "x", "op": "placeholder", "target": "x", "inputs": []},
                {"name": "y", "op": "output", "target": "output", "inputs": ["x"]},
            ],
            frontend="test",
        )
        with self.assertRaisesRegex(ValueError, "different graph"):
            PaperGraphAnnotations.from_dict(raw, graph_profile=other)

    def test_annotations_survive_observation_frontend_and_radius_changes(self):
        observed = _fork_join_profile(
            batch_size=32,
            radii=(0,),
            frontend="torch.export",
            metadata={"probe_signature": "different"},
        )
        self.assertNotEqual(self.profile.digest, observed.digest)
        self.assertEqual(
            self.profile.graph.structural_digest,
            observed.graph.structural_digest,
        )
        result = annotate_graph_from_paper(
            self.profile,
            self.paper,
            [_annotation_claim(self.paper, self.target)],
        )
        raw = result.to_dict()
        self.assertNotIn("graph_profile_digest", raw)
        restored = PaperGraphAnnotations.from_dict(raw, graph_profile=observed)
        self.assertEqual(restored, result)

    def test_character_existence_is_authoritative_across_radius_changes(self):
        radius_three = next(
            character
            for character in self.profile.characters
            if character.generator == "directed_wl_neighborhood.v1"
            and character.parameters.get("radius") == 3
        )
        target = GraphAnnotationTarget.resolve(
            self.profile,
            kind="character",
            target_id=radius_three.character_id,
        )
        result = annotate_graph_from_paper(
            self.profile,
            self.paper,
            [_annotation_claim(self.paper, target)],
        )
        reduced = _fork_join_profile(radii=(0,))
        self.assertEqual(
            self.profile.graph.structural_digest,
            reduced.graph.structural_digest,
        )
        with self.assertRaisesRegex(ValueError, "unknown 'character' target"):
            PaperGraphAnnotations.from_dict(
                result.to_dict(), graph_profile=reduced
            )

    def test_structural_operator_change_stales_annotations(self):
        result = annotate_graph_from_paper(
            self.profile,
            self.paper,
            [_annotation_claim(self.paper, self.target)],
        )
        changed = _fork_join_profile(second_target="vendor.transform_c")
        self.assertNotEqual(
            self.profile.graph.structural_digest,
            changed.graph.structural_digest,
        )
        with self.assertRaisesRegex(ValueError, "different graph"):
            PaperGraphAnnotations.from_dict(
                result.to_dict(), graph_profile=changed
            )


if __name__ == "__main__":
    unittest.main()
