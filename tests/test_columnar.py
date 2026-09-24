from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from phylodigy.canonical import normalize_json
from phylodigy.columnar import (
    read_graph_tables,
    read_lineage_tables,
    write_graph_tables,
    write_lineage_tables,
)
from phylodigy.computation_graph import ComputationGraph, GraphEdge, GraphNode


try:
    import cbor2  # noqa: F401
    import pyarrow  # noqa: F401
except ImportError:  # pragma: no cover - environment dependent
    COLUMNAR_AVAILABLE = False
else:
    COLUMNAR_AVAILABLE = True


@unittest.skipUnless(COLUMNAR_AVAILABLE, "columnar dependencies are not installed")
class ColumnarGraphTests(unittest.TestCase):
    def test_graphs_round_trip_as_typed_parquet_vertex_and_edge_tables(self):
        graph = ComputationGraph(
            nodes=(
                GraphNode(
                    "n0",
                    "graph.input",
                    attributes={"shape": [1, 32], "dtype": "float32"},
                    observations={"probe": {"device": "cpu"}},
                ),
                GraphNode(
                    "n1",
                    "module.torch.nn.linear",
                    attributes={"bias": True, "out_features": 64},
                ),
            ),
            edges=(GraphEdge("n0", "n1", "arg:000000"),),
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_graph_tables({"model/example": graph}, directory)
            restored = read_graph_tables(directory)
            files = sorted(Path(directory).iterdir())

            self.assertEqual(set(paths), {"graphs", "graph_vertices", "graph_edges"})
            self.assertTrue(all(path.suffix == ".parquet" for path in files))
            self.assertTrue(all(path.read_bytes()[:4] == b"PAR1" for path in files))
            self.assertEqual(restored["model/example"].to_dict(), graph.to_dict())
            self.assertEqual(
                restored["model/example"].structural_digest, graph.structural_digest
            )


@unittest.skipUnless(COLUMNAR_AVAILABLE, "columnar dependencies are not installed")
class ColumnarLineageTests(unittest.TestCase):
    def test_lineage_round_trip_retains_tree_matrix_distances_and_audit_payload(self):
        lineage = {
            "analysis_version": "2",
            "artifact_type": "phylodigy.architecture_lineage",
            "character_matrix": {
                "character_ids": ["char:a", "char:b"],
                "counts_by_artifact": {
                    "model:a": [1, 0],
                    "model:b": [1, 2],
                },
                "source": "operator_layer_graph_only",
            },
            "comparisons": [
                {
                    "artifact_type": "phylodigy.graph_comparison",
                    "digest": "comparison:ab",
                    "graph_distance": {
                        "distance": 2.0,
                        "scheme": "weighted-l1-on-dynamic-graph-characters.v1",
                    },
                    "left": {"id": "model:a"},
                    "right": {"id": "model:b"},
                }
            ],
            "config": {"four_point_tolerance": 1e-9, "radii": [0, 1]},
            "external_evidence": [{"kind": "citation", "paper_id": "doi:example"}],
            "genomes": [
                {
                    "date_max": "2020-01-01",
                    "date_min": "2020-01-01",
                    "digest": "genome:a",
                    "graph_digest": "graph:a",
                    "id": "model:a",
                },
                {
                    "date_max": "2021-01-01",
                    "date_min": "2021-01-01",
                    "digest": "genome:b",
                    "graph_digest": "graph:b",
                    "id": "model:b",
                },
            ],
            "structural_evidence_boundary": {
                "paper_can_create_characters": False,
                "structural_source": "operator_layer_graph",
            },
            "tree": {
                "artifact_type": "phylodigy.architecture_phylogeny",
                "distance_model": "weighted_l1_dynamic_graph_characters",
                "edges": [
                    {
                        "branch_length": 1.0,
                        "child": "model:a",
                        "length_clamped": False,
                        "parent": "internal:0",
                        "raw_branch_length": 1.0,
                    },
                    {
                        "branch_length": 1.0,
                        "child": "model:b",
                        "length_clamped": False,
                        "parent": "internal:0",
                        "raw_branch_length": 1.0,
                    },
                ],
                "method": "neighbor_joining",
                "normalization": "none",
                "taxa": ["model:a", "model:b"],
                "tree_likeness": {
                    "additive_within_tolerance": True,
                    "distance_domain_valid": True,
                    "maximum_four_point_violation": 0.0,
                    "metric_within_tolerance": True,
                    "tree_metric_within_tolerance": True,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = write_lineage_tables(lineage, directory)
            restored = read_lineage_tables(directory)
            files = sorted(Path(directory).iterdir())

            self.assertEqual(
                set(paths),
                {
                    "phylogenies",
                    "tree_edges",
                    "taxa",
                    "character_states",
                    "distances",
                },
            )
            self.assertTrue(all(path.suffix == ".parquet" for path in files))
            self.assertEqual(restored, normalize_json(lineage))


if __name__ == "__main__":
    unittest.main()
