"""Check output plumbing with a synthetic runner; these are not model results."""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantbench.metrics import summarize


class SyntheticRunner:
    def load(self):
        pass

    def preprocess(self, texts):
        return [len(text) for text in texts]

    def predict(self, inputs):
        return [[float(length), -float(length)] for length in inputs]

    def metadata(self):
        return {"variant": "synthetic", "real_model": False, "provider": "Python"}


def main():
    runner = SyntheticRunner()
    runner.load()
    inputs = runner.preprocess(["fixture"])
    for _ in range(10):
        runner.predict(inputs)
    records = []
    for iteration in range(200):
        start = perf_counter_ns()
        runner.predict(inputs)
        duration = perf_counter_ns() - start
        records.append({"iteration": iteration, "input_id": "synthetic-0", "duration_ns": duration})
    summary = summarize([row["duration_ns"] for row in records], 1)
    root = Path(__file__).resolve().parents[1]
    destination = root / "results" / "local" / ("smoke-" + uuid4().hex)
    destination.mkdir(parents=True)
    manifest = {**runner.metadata(), "created_at": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "timing_scope": "inference_only", "warmup": 10, "iterations": 200}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (destination / "timings.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    print(f"Synthetic smoke check passed: {destination}")


if __name__ == "__main__":
    main()
