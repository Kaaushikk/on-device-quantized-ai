import json
import tempfile
import unittest
from pathlib import Path
from quantbench.audit import validate_run
from quantbench.common import VARIANTS, write_json


class AuditTests(unittest.TestCase):
    def fixture(self, root):
        settings = {"batch_sizes": [1], "sequence_lengths": [32], "iterations": 2, "repetitions": 1}
        write_json(root / "protocol.json", settings)
        for variant in VARIANTS:
            folder = root / f"0-{variant}"
            write_json(folder / "manifest.json", {"settings": settings, "variant": variant,
                       "repetition": 0, "data_sha256": "shared-data"})
            rows = [{"variant": variant, "repetition": 0, "scope": scope, "batch_size": 1,
                     "sequence_length": 32, "iteration": i, "input_ids": [i], "repeat_counts": [1],
                     "unpadded_token_lengths": [10], "duration_ns": 100}
                    for scope in ("inference_only", "tokenization_plus_inference") for i in range(2)]
            (folder / "timings.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_matching_experiment_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            self.assertEqual(validate_run(root)["total_records"], 12)

    def test_changed_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "0-onnx_int8" / "timings.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["input_ids"] = [999]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Input IDs"):
                validate_run(root)

    def test_missing_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            path = root / "0-onnx_int8" / "timings.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "record count"):
                validate_run(root)
