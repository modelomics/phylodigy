from __future__ import annotations

import hashlib
import unittest
import unicodedata

from phylodigy.citations import (
    CitationTarget,
    citation_marker_keys,
    extract_bibliography,
    extract_citation_evidence,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    resolve_bibliography,
)


PAPER = """Methods
We build on earlier work [1] and compare several baselines [2–3].

References
[1] Example, A. A Graph Method. arXiv:2305.13245v2.
[2] Other, B. Another Method. https://doi.org/10.1000/ABC.
[3] Third, C. Unresolved Work. 2020.
"""


class CitationParsingTests(unittest.TestCase):
    def test_identifier_and_title_normalization(self):
        self.assertEqual(normalize_arxiv_id("arXiv:2305.13245v3"), "2305.13245")
        self.assertEqual(
            normalize_arxiv_id("https://arxiv.org/pdf/2305.13245v2.pdf"),
            "2305.13245",
        )
        self.assertEqual(normalize_doi("https://doi.org/10.1000/ABC."), "10.1000/abc")
        self.assertEqual(normalize_title("A_Graph: Method!"), "a graph method")

    def test_numeric_ranges_and_author_year_markers(self):
        self.assertEqual(
            citation_marker_keys("[2–4]"),
            ("marker:2", "marker:3", "marker:4"),
        )
        self.assertEqual(
            citation_marker_keys("Example et al. (2023)"),
            ("author_year:example:2023",),
        )

    def test_bibliography_spans_and_digest_are_exact(self):
        entries = extract_bibliography(PAPER)
        normalized = unicodedata.normalize("NFKC", PAPER)
        expected_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        self.assertEqual([entry.marker for entry in entries], ["1", "2", "3"])
        for entry in entries:
            self.assertEqual(entry.source_digest, expected_digest)
            self.assertEqual(entry.to_dict()["offset_basis"], "citation_normalized_text_codepoints")
            self.assertIn("[" + entry.marker + "]", normalized[entry.start : entry.end])

    def test_no_reference_heading_produces_no_bibliography(self):
        self.assertEqual(extract_bibliography("We cite [1], but provide no list."), ())


class BibliographyResolutionTests(unittest.TestCase):
    def test_exact_identifiers_resolve_before_fallbacks(self):
        entries = extract_bibliography(PAPER)
        resolved, report = resolve_bibliography(
            entries,
            (
                CitationTarget("paper:one", arxiv_id="2305.13245"),
                CitationTarget("paper:two", doi="10.1000/abc"),
            ),
        )

        self.assertEqual(resolved["marker:1"]["artifact_id"], "paper:one")
        self.assertEqual(resolved["marker:1"]["match_basis"], "arxiv_id")
        self.assertEqual(resolved["marker:2"]["artifact_id"], "paper:two")
        self.assertEqual(resolved["marker:2"]["match_basis"], "doi")
        self.assertEqual(report["unresolved_markers"], ["3"])

    def test_ambiguous_title_is_not_forced(self):
        text = """References
[1] A Very Specific Graph Paper Title. 2020.
"""
        entries = extract_bibliography(text)
        resolved, report = resolve_bibliography(
            entries,
            (
                CitationTarget("paper:a", title="A Very Specific Graph Paper Title"),
                CitationTarget("paper:b", title="A Very Specific Graph Paper Title"),
            ),
        )

        self.assertNotIn("marker:1", resolved)
        self.assertEqual(
            report["ambiguous"][0]["candidate_artifact_ids"],
            ["paper:a", "paper:b"],
        )


class CitationEvidenceTests(unittest.TestCase):
    def test_mentions_resolve_exact_spans_without_semantic_roles(self):
        mentions, report = extract_citation_evidence(
            PAPER,
            (
                CitationTarget("paper:one", arxiv_id="2305.13245"),
                CitationTarget("paper:two", doi="10.1000/abc"),
            ),
        )

        self.assertEqual(len(mentions), 2)
        normalized = unicodedata.normalize("NFKC", PAPER)
        for mention in mentions:
            self.assertEqual(normalized[mention.start : mention.end], mention.marker)
            raw = mention.to_dict()
            self.assertNotIn("trait_id", raw)
            self.assertNotIn("character_id", raw)
            self.assertNotIn("causal_role", raw)
        self.assertEqual(mentions[0].target_ids, ("paper:one",))
        self.assertEqual(mentions[1].target_ids, ("paper:two",))
        self.assertEqual(report["structural_claims_created"], 0)

    def test_wording_does_not_change_citation_semantics(self):
        catalog = (CitationTarget("paper:one", arxiv_id="2305.13245"),)
        adopted, adopted_report = extract_citation_evidence(
            PAPER.replace("build on", "adopt"), catalog
        )
        rejected, rejected_report = extract_citation_evidence(
            PAPER.replace("build on", "do not use"), catalog
        )

        self.assertEqual(adopted[0].target_ids, rejected[0].target_ids)
        self.assertEqual(adopted[0].citation_keys, rejected[0].citation_keys)
        self.assertEqual(adopted_report["structural_claims_created"], 0)
        self.assertEqual(rejected_report["structural_claims_created"], 0)

    def test_output_is_deterministic_across_line_endings(self):
        first = extract_citation_evidence(PAPER)
        second = extract_citation_evidence(PAPER.replace("\n", "\r\n"))

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
