from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

from phylodigy.huggingface_dataset import export_huggingface_dataset


try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - environment dependent
    COLUMNAR_AVAILABLE = False
else:
    COLUMNAR_AVAILABLE = True


def _write_parquet(path: Path, **columns: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="zstd")


@unittest.skipUnless(COLUMNAR_AVAILABLE, "pyarrow is not installed")
class HuggingFaceDatasetExportTests(unittest.TestCase):
    def test_exports_present_tables_as_repo_ready_parquet_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            graphs = root / "graphs"
            lineage = root / "lineage"
            destination = root / "dataset"

            _write_parquet(
                registry / "current_records.parquet",
                catalog_id=["hf:example/model"],
                name=["Example Model"],
                search_text=["Example Model transformer"],
                payload_cbor=[b"\xa1ax\x01"],
            )
            _write_parquet(
                registry / "observations.parquet",
                observation_id=["observation:1"],
                payload_cbor=[b"\xa1ax\x01"],
            )
            _write_parquet(registry / "papers.parquet", paper_id=["doi:example"])
            _write_parquet(
                registry / "current_model_papers.parquet",
                catalog_id=["hf:example/model"],
                paper_id=["doi:example"],
            )
            _write_parquet(registry / "snapshots.parquet", digest=["snapshot:1"])
            _write_parquet(
                registry / "current_metrics.parquet",
                catalog_id=["hf:example/model"],
                metric=["downloads"],
                value=[100.0],
            )
            _write_parquet(
                registry / "paper_links.parquet",
                observation_id=["observation:1"],
                paper_id=["doi:example"],
            )
            _write_parquet(
                registry / "paper_metrics.parquet",
                paper_id=["doi:example"],
                metric=["citations"],
                value=[12.0],
            )
            _write_parquet(
                registry / "provider_checkpoints.parquet",
                provider=["huggingface"],
                complete=[True],
            )
            _write_parquet(
                graphs / "graph_vertices.parquet",
                graph_id=["graph:1"],
                node_id=["node:1"],
            )
            _write_parquet(
                graphs / "graph_edges.parquet",
                graph_id=["graph:1"],
                edge_id=["edge:1"],
            )
            for table_name in (
                "phylogenies",
                "tree_edges",
                "taxa",
                "character_states",
                "distances",
            ):
                _write_parquet(
                    lineage / f"{table_name}.parquet",
                    row_id=[f"{table_name}:1"],
                )

            written = export_huggingface_dataset(
                registry,
                destination,
                graph_tables=graphs,
                lineage_tables=lineage,
            )

            expected_configs = (
                "models",
                "observations",
                "papers",
                "model_paper_links",
                "snapshots",
                "graph_vertices",
                "graph_edges",
                "phylogenies",
                "tree_edges",
                "taxa",
                "character_states",
                "distances",
                "model_metrics",
                "paper_links",
                "paper_metrics",
                "coverage",
            )
            self.assertEqual(tuple(written), expected_configs)
            for config_name in expected_configs:
                self.assertEqual(
                    written[config_name],
                    (destination / "data" / config_name / "part-00000.parquet",),
                )

            card = (destination / "README.md").read_text(encoding="utf-8")
            card_configs = tuple(re.findall(r"^  - config_name: (\w+)$", card, re.M))
            self.assertEqual(card_configs, expected_configs)
            self.assertIn("  - config_name: models\n    default: true", card)
            for config_name in expected_configs:
                self.assertIn(f"path: data/{config_name}/*.parquet", card)

            data_files = tuple((destination / "data").rglob("*"))
            self.assertTrue(data_files)
            self.assertTrue(
                all(path.is_dir() or path.suffix == ".parquet" for path in data_files)
            )
            parquet_files = tuple(
                path for path in data_files if path.suffix == ".parquet"
            )
            self.assertTrue(
                all(path.read_bytes()[:4] == b"PAR1" for path in parquet_files)
            )
            for path in parquet_files:
                pq.ParquetFile(path)

            model_columns = pq.read_schema(written["models"][0]).names
            observation_columns = pq.read_schema(written["observations"][0]).names
            self.assertNotIn("payload_cbor", model_columns)
            self.assertIn("payload_cbor", observation_columns)
            self.assertFalse(tuple(destination.rglob("*.json")))

    def test_card_omits_optional_tables_that_are_not_present(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry"
            destination = root / "dataset"
            _write_parquet(
                registry / "current_records.parquet",
                catalog_id=["model:1"],
                search_text=["model one"],
            )
            _write_parquet(registry / "papers.parquet", paper_id=["paper:1"])

            export_huggingface_dataset(registry, destination)

            card = (destination / "README.md").read_text(encoding="utf-8")
            self.assertEqual(
                re.findall(r"^  - config_name: (\w+)$", card, re.M),
                ["models", "papers"],
            )
            self.assertEqual(
                sorted(
                    path.name
                    for path in (destination / "data").iterdir()
                    if path.is_dir()
                ),
                ["models", "papers"],
            )


if __name__ == "__main__":
    unittest.main()
