from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from phylodigy.io import write_json_data


class AtomicJsonOutputTests(unittest.TestCase):
    def test_file_output_is_canonical_and_pretty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "data.json"

            write_json_data({"z": 1, "a": [2]}, destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), '{\n  "a": [\n    2\n  ],\n  "z": 1\n}\n')

    def test_serialization_failure_preserves_destination_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "data.json"
            destination.write_text('{"old": true}\n', encoding="utf-8")

            with self.assertRaisesRegex(TypeError, "unsupported canonical JSON"):
                write_json_data({"bad": object()}, destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_replace_failure_cleans_temp_and_preserves_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "data.json"
            destination.write_text('{"old": true}\n', encoding="utf-8")

            with mock.patch("phylodigy.io.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_json_data({"new": True}, destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_stream_failure_cleans_temp_and_preserves_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "data.json"
            destination.write_text('{"old": true}\n', encoding="utf-8")

            def broken_stream(_encoder, _value):
                yield "{"
                raise OSError("stream failed")

            with mock.patch("phylodigy.io.json.JSONEncoder.iterencode", broken_stream):
                with self.assertRaisesRegex(OSError, "stream failed"):
                    write_json_data({"new": True}, destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), '{"old": true}\n')
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_stdout_output_keeps_canonical_format(self) -> None:
        from io import StringIO

        output = StringIO()
        write_json_data({"z": 1, "a": 2}, "-", stdout=output)

        self.assertEqual(output.getvalue(), json.dumps(
            {"a": 2, "z": 1}, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
        ) + "\n")
