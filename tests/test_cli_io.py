from __future__ import annotations

import contextlib
import io as stdlib_io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from phylodigy import GraphAlignment, validate_graph_alignment
from phylodigy.cli import build_parser, main
from phylodigy.code_profile import SourceManifest
from phylodigy.computation_graph import build_graph_profile
from phylodigy.io import (
    DocumentReadError,
    ProfileReadError,
    read_document_text,
    read_json_data,
    read_profile,
    write_json_data,
    write_profile,
)
from phylodigy.paper_profile import PaperDocument
from phylodigy.schema import ArchitecturalGenome


def genome(artifact_id, *operations, date=None):
    records = [
        {"index": 0, "name": "input", "op": "placeholder", "target": "input"}
    ]
    previous = "input"
    for index, operation in enumerate(operations, 1):
        name = f"node_{index}"
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


class IOTests(unittest.TestCase):
    def test_reads_utf8_text_markdown_and_stdin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = root / "paper.txt"
            markdown_path = root / "paper.md"
            text_path.write_text("Unicode — text", encoding="utf-8")
            markdown_path.write_text("# Methods\nDescription.", encoding="utf-8")
            self.assertEqual(read_document_text(text_path), "Unicode — text")
            self.assertEqual(read_document_text(markdown_path), "# Methods\nDescription.")
        self.assertEqual(
            read_document_text("-", stdin=stdlib_io.StringIO("stdin paper")),
            "stdin paper",
        )

    def test_pdf_dependency_is_lazy_and_pages_are_joined(self):
        class Page:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        fake_pypdf = types.SimpleNamespace(
            PdfReader=lambda _path: types.SimpleNamespace(
                pages=[Page("page one"), Page(None), Page("page three")]
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            pdf = Path(directory) / "paper.pdf"
            pdf.write_bytes(b"%PDF-placeholder")
            with mock.patch("phylodigy.io.importlib.import_module", return_value=fake_pypdf):
                self.assertEqual(read_document_text(pdf), "page one\f\fpage three")
            with mock.patch(
                "phylodigy.io.importlib.import_module",
                side_effect=ImportError("no pypdf"),
            ):
                with self.assertRaisesRegex(DocumentReadError, "phylodigy\\[paper\\]"):
                    read_document_text(pdf)

    def test_profile_and_generic_json_round_trip(self):
        profile = genome("model:a", "vendor.alpha", date="2026-01-02")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "nested" / "genome.json"
            write_profile(profile, profile_path)
            self.assertEqual(read_profile(profile_path), profile)
            self.assertTrue(profile_path.read_text(encoding="utf-8").endswith("\n"))

            generic_path = root / "generic.json"
            write_json_data({"z": 1, "a": [2]}, generic_path)
            self.assertEqual(read_json_data(generic_path), {"a": [2], "z": 1})

            raw = profile.to_dict()
            raw["digest"] = "0" * 64
            generic_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ProfileReadError, "digest"):
                read_profile(generic_path)


class CLITests(unittest.TestCase):
    @staticmethod
    def run_cli(arguments):
        stdout = stdlib_io.StringIO()
        stderr = stdlib_io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_public_help_has_no_fixed_semantics_commands_or_options(self):
        help_text = build_parser().format_help()

        self.assertIn("prepare-paper", help_text)
        self.assertIn("build-genome", help_text)
        self.assertIn("extract-code", help_text)
        self.assertIn("align", help_text)
        self.assertIn("infer-lineage", help_text)
        for forbidden in (
            "export-ontology",
            "extract-config",
            "trait-priors",
            "bundle-groups",
            "paper regex",
        ):
            self.assertNotIn(forbidden, help_text.lower())

    def test_prepare_paper_normalizes_without_creating_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper_path = root / "paper.md"
            output = root / "paper.json"
            paper_path.write_text(
                "Methods\nWe call this region the Example Loop.", encoding="utf-8"
            )
            code, stdout, stderr = self.run_cli(
                [
                    "prepare-paper",
                    str(paper_path),
                    "--artifact-id",
                    "paper:example",
                    "--identifier",
                    "doi=10.1000/example",
                    "-o",
                    str(output),
                ]
            )

            self.assertEqual((code, stdout, stderr), (0, "", ""))
            document = PaperDocument.from_dict(json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual(document.artifact_id, "paper:example")
            raw = document.to_dict()
            self.assertNotIn("annotations", raw)
            self.assertNotIn("traits", raw)
            self.assertNotIn("ontology", raw)

    def test_extract_code_returns_source_manifest_not_genome(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "model.py").write_text(
                "class TransformerAttentionResidual: pass\n", encoding="utf-8"
            )
            output = root / "source-manifest.json"
            code, stdout, stderr = self.run_cli(
                [
                    "extract-code",
                    str(source),
                    "--artifact-id",
                    "model:source",
                    "-o",
                    str(output),
                ]
            )

            self.assertEqual((code, stdout, stderr), (0, "", ""))
            manifest = SourceManifest.from_dict(json.loads(output.read_text(encoding="utf-8")))
            raw = manifest.to_dict()
            self.assertEqual(raw["artifact_type"], "phylodigy.source_manifest")
            self.assertNotIn("graph_profile", raw)
            self.assertNotIn("traits", raw)

    def test_build_genome_round_trips_open_graph_records(self):
        records = [
            {"index": 0, "name": "x", "op": "placeholder", "target": "x"},
            {
                "index": 1,
                "inputs": ["x"],
                "name": "novel",
                "op": "call_function",
                "target": "vendor.never_seen_before",
            },
            {
                "index": 2,
                "inputs": ["novel"],
                "name": "out",
                "op": "output",
                "target": "output",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records_path = root / "records.json"
            output = root / "genome.json"
            records_path.write_text(json.dumps(records), encoding="utf-8")
            code, stdout, stderr = self.run_cli(
                [
                    "build-genome",
                    str(records_path),
                    "--artifact-id",
                    "model:novel",
                    "--frontend",
                    "test.ir",
                    "-o",
                    str(output),
                ]
            )

            self.assertEqual((code, stdout, stderr), (0, "", ""))
            result = read_profile(output)
            self.assertEqual(result.graph_profile.frontend, "test.ir")
            self.assertIn(
                "function.vendor.never_seen_before",
                {node.operation for node in result.graph_profile.graph.nodes},
            )
            self.assertNotIn("traits", result.to_dict())

    def test_merge_and_validate_identical_graph_genomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = root / "left.json"
            right_path = root / "right.json"
            merged_path = root / "merged.json"
            profile = genome("model:a", "vendor.alpha")
            write_profile(profile, left_path)
            write_profile(profile, right_path)

            code, stdout, stderr = self.run_cli(
                ["merge", str(left_path), str(right_path), "-o", str(merged_path)]
            )
            self.assertEqual((code, stdout, stderr), (0, "", ""))
            self.assertEqual(read_profile(merged_path).structural_digest, profile.structural_digest)

            code, stdout, stderr = self.run_cli(["validate", str(merged_path)])
            self.assertEqual((code, stderr), (0, ""))
            self.assertTrue(json.loads(stdout)["valid"])

    def test_align_emits_a_reversible_bounded_graph_only_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = genome("model:left", "vendor.alpha")
            right = genome("model:right", "vendor.alpha", "vendor.inserted")
            left_path = root / "left.json"
            right_path = root / "right.json"
            config_path = root / "alignment.json"
            output_path = root / "forward.json"
            write_profile(left, left_path)
            write_profile(right, right_path)
            config_path.write_text(
                json.dumps(
                    {
                        "edge_indel_cost": 0.5,
                        "max_matrix_cells": 100,
                        "node_indel_cost": 2.0,
                    }
                ),
                encoding="utf-8",
            )

            code, stdout, stderr = self.run_cli(
                [
                    "align",
                    str(left_path),
                    str(right_path),
                    "--config",
                    str(config_path),
                    "-o",
                    str(output_path),
                ]
            )

            self.assertEqual((code, stdout, stderr), (0, "", ""))
            raw = json.loads(output_path.read_text(encoding="utf-8"))
            forward = GraphAlignment.from_dict(raw)
            self.assertEqual(raw["artifact_type"], "phylodigy.graph_alignment")
            self.assertEqual(forward.left_structural_digest, left.structural_digest)
            self.assertEqual(forward.right_structural_digest, right.structural_digest)
            self.assertEqual(
                raw["replay_direction"],
                {
                    "ancestral_interpretation": "none",
                    "from_endpoint": "left",
                    "to_endpoint": "right",
                },
            )
            self.assertEqual(forward.total_cost, 3.5)
            self.assertEqual(
                {operation.kind for operation in forward.operations},
                {"add_edge", "add_node", "remove_edge"},
            )
            self.assertEqual(len(raw["unmatched_left_edges"]), 1)
            self.assertEqual(len(raw["unmatched_right_edges"]), 2)
            self.assertEqual(len(raw["unmatched_right_nodes"]), 1)
            self.assertNotEqual(
                raw["cost_model_digest"], raw["solver_configuration_digest"]
            )
            self.assertNotIn("objective_optimality_proven", raw)
            validation = validate_graph_alignment(
                left.graph_profile, right.graph_profile, forward
            )
            self.assertTrue(validation["objective_optimality_proven"])
            serialized = json.dumps(raw).lower()
            for forbidden in (
                "branch_length",
                "historical_gain",
                "historical_loss",
                "ontology",
                "operator_substitution",
                "trait",
            ):
                self.assertNotIn(forbidden, serialized)

            code, stdout, stderr = self.run_cli(
                [
                    "align",
                    str(right_path),
                    str(left_path),
                    "--config",
                    str(config_path),
                ]
            )
            self.assertEqual((code, stderr), (0, ""))
            reverse = GraphAlignment.from_dict(json.loads(stdout))
            self.assertEqual(reverse, forward.invert())

    def test_infer_lineage_and_network_consume_only_genomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for item in (
                genome("model:a", "vendor.alpha", date="2020-01-01"),
                genome("model:b", "vendor.alpha", "vendor.beta", date="2021-01-01"),
            ):
                path = root / (item.artifact_id.replace(":", "-") + ".json")
                write_profile(item, path)
                paths.append(path)

            code, stdout, stderr = self.run_cli(
                ["infer-lineage", *(str(path) for path in paths)]
            )
            self.assertEqual((code, stderr), (0, ""))
            lineage = json.loads(stdout)
            self.assertEqual(lineage["character_matrix"]["source"], "operator_layer_graph_only")

            code, stdout, stderr = self.run_cli(
                ["infer-network", *(str(path) for path in paths)]
            )
            self.assertEqual((code, stderr), (0, ""))
            network = json.loads(stdout)
            self.assertEqual(network["artifact_type"], "phylodigy.graph_contact_network")
            self.assertEqual(network["edges"], [])
            self.assertEqual(network["primary_parent_forest"], [])
            self.assertNotIn("traits", network)

    def test_legacy_commands_are_rejected_by_parser(self):
        for command in ("export-ontology", "extract-config"):
            with self.subTest(command=command):
                with self.assertRaises(SystemExit):
                    with contextlib.redirect_stderr(stdlib_io.StringIO()):
                        main([command])

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(stdlib_io.StringIO()):
                main(["infer-network", "genome.json", "--evidence-policy", "graph_only"])


if __name__ == "__main__":
    unittest.main()
