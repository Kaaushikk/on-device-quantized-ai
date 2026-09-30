import csv
import json
import os
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from uuid import uuid4
import numpy as np
import psutil
from quantbench.common import ARTIFACTS, ROOT, VARIANTS, config, read_json, sha256, versions, write_json
from quantbench.metrics import summarize
from quantbench.runners.local import LocalRunner


def data():
    split = read_json(ROOT / "configs" / "split.json")
    if sha256(ARTIFACTS / "data.json") != split["local_data_sha256"]:
        raise ValueError("Local data does not match frozen split")
    return read_json(ARTIFACTS / "data.json")


def quality_metrics(labels, predictions):
    matrix = np.zeros((2, 2), dtype=int)
    for expected, actual in zip(labels, predictions, strict=True):
        matrix[expected, actual] += 1
    f1 = []
    for label in range(2):
        denominator = matrix[label].sum() + matrix[:, label].sum()
        f1.append(2 * matrix[label, label] / denominator if denominator else 0)
    return {"n": len(labels), "accuracy": float(np.trace(matrix) / len(labels)),
            "macro_f1": float(np.mean(f1)), "confusion_matrix": matrix.tolist()}


def evaluate(variant, subset):
    dataset = data()
    ids = dataset[f"{subset}_ids"]
    rows = [dataset["rows"][i] for i in ids]
    runner = LocalRunner(variant, config()["threads"])
    runner.load()
    logits = []
    for start in range(0, len(rows), 8):
        inputs = runner.prepare(runner.preprocess([row["text"] for row in rows[start:start + 8]], 512, pad_to_max=False))
        logits.extend(runner.predict(inputs).tolist())
    predictions = np.asarray(logits).argmax(1).tolist()
    result = {"variant": variant, "subset": subset, "row_ids": ids,
              "labels": [row["label"] for row in rows], "predictions": predictions,
              "logits": logits, "metadata": runner.metadata(), "max_sequence_length": 512,
              "padding": "longest in batch", "batch_size": 8,
              "data_sha256": sha256(ARTIFACTS / "data.json")}
    result.update(quality_metrics(result["labels"], predictions))
    write_json(ROOT / "results" / f"quality-{subset}-{variant}.json", result)
    print(f"{variant} {subset}: accuracy={result['accuracy']:.4f}, macro F1={result['macro_f1']:.4f}, N={len(ids)}")


def paired_quality(subset):
    baseline = read_json(ROOT / "results" / f"quality-{subset}-onnx_fp32.json")
    candidate = read_json(ROOT / "results" / f"quality-{subset}-onnx_int8.json")
    if baseline["row_ids"] != candidate["row_ids"] or baseline["labels"] != candidate["labels"]:
        raise ValueError("Quality inputs differ")
    labels = np.array(baseline["labels"])
    differences = (np.array(candidate["predictions"]) == labels).astype(float) - (np.array(baseline["predictions"]) == labels)
    rng = np.random.default_rng(config()["seed"])
    bootstrap = np.mean(rng.choice(differences, (10000, len(labels)), replace=True), axis=1)
    result = {"subset": subset, "n": len(labels), "accuracy_difference_int8_minus_fp32": float(differences.mean()),
              "paired_bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
              "gate": config()["accuracy_loss_gate"],
              "point_estimate_gate_passed": bool(differences.mean() >= -config()["accuracy_loss_gate"]),
              "changed_row_ids": [row for row, a, b in zip(baseline["row_ids"], baseline["predictions"], candidate["predictions"], strict=True) if a != b]}
    write_json(ROOT / "results" / f"quality-{subset}-comparison.json", result)
    print(json.dumps(result))


def hardware():
    def windows_output(command):
        try:
            return subprocess.check_output(command, text=True, timeout=15).strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
    return {"platform": platform.platform(), "processor": platform.processor(),
            "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
            "cpu_name": windows_output(["powershell", "-NoProfile", "-Command", "Get-ItemPropertyValue -LiteralPath 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0' -Name ProcessorNameString"]),
            "power_scheme": windows_output(["powercfg", "/getactivescheme"]),
            "battery": str(psutil.sensors_battery()),
            "thermal_state": "not measured", "energy": "not measured"}


def worker(variant, repetition, destination):
    settings = config()
    process = psutil.Process()
    runner = LocalRunner(variant, settings["threads"])
    start = perf_counter_ns()
    runner.load()
    load_ns = perf_counter_ns() - start
    idle_rss = process.memory_info().rss
    peak = [idle_rss]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    dataset = data()
    rows = [dataset["rows"][i] for i in dataset["development_ids"]]
    token_lengths = [len(runner.tokenizer(row["text"], truncation=False)["input_ids"]) for row in rows]
    records, summaries = [], []
    first_ns = None
    try:
        for batch in settings["batch_sizes"]:
            previous = 0
            for length in settings["sequence_lengths"]:
                pool = [(row, size) for row, size in zip(rows, token_lengths, strict=True) if previous < size <= length]
                previous = length
                if not pool:
                    # SST-2 has no natural >128-token examples in this development
                    # split. A separately labeled repeated-text stress fixture
                    # exercises full-length computation, without a quality label.
                    pool = []
                    for row, size in zip(rows[:16], token_lengths[:16], strict=True):
                        repeats = max(2, (length - 2 + max(1, size - 2) - 1) // max(1, size - 2))
                        text = " ".join([row["text"]] * repeats)
                        count = min(length, len(runner.tokenizer(text)["input_ids"]))
                        pool.append(({**row, "text": text, "repeat_count": repeats}, count))
                # Freeze a bounded representative pool; rotate it identically across variants.
                pool = pool[:min(16, len(pool))]
                batches = [[pool[(offset + j) % len(pool)] for j in range(batch)] for offset in range(len(pool))]
                prepared = [runner.prepare(runner.preprocess([row["text"] for row, _ in group], length)) for group in batches]
                if first_ns is None:
                    start = perf_counter_ns()
                    runner.predict(prepared[0])
                    first_ns = perf_counter_ns() - start
                for scope in ("inference_only", "tokenization_plus_inference"):
                    def predict(index):
                        if scope == "inference_only":
                            return runner.predict(prepared[index])
                        return runner.predict(runner.prepare(runner.preprocess([row["text"] for row, _ in batches[index]], length)))
                    for iteration in range(settings["warmup"]):
                        predict(iteration % len(batches))
                    durations = []
                    for iteration in range(settings["iterations"]):
                        index = iteration % len(batches)
                        start = perf_counter_ns()
                        logits = predict(index)
                        duration = perf_counter_ns() - start
                        if logits.shape != (batch, 2) or not np.isfinite(logits).all():
                            raise ValueError("Invalid model output")
                        durations.append(duration)
                        records.append({"variant": variant, "repetition": repetition, "scope": scope,
                                        "batch_size": batch, "sequence_length": length, "iteration": iteration,
                                        "input_ids": [row["row_id"] for row, _ in batches[index]],
                                        "repeat_counts": [row.get("repeat_count", 1) for row, _ in batches[index]],
                                        "unpadded_token_lengths": [size for _, size in batches[index]], "duration_ns": duration})
                    summaries.append({"variant": variant, "repetition": repetition, "scope": scope,
                                      "sequence_length": length, **summarize(durations, batch)})
    finally:
        stop.set()
        sampler.join()
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False)
    weight_paths = ([ARTIFACTS / "model" / "model.safetensors"] if variant == "pytorch_fp32"
                    else [ARTIFACTS / runner.manifest["graph"]])
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "variant": variant,
                "repetition": repetition, "settings": settings, "hardware": hardware(), "versions": versions(),
                "runner": runner.metadata(), "data_sha256": sha256(ARTIFACTS / "data.json"),
                "load_ns": load_ns, "first_inference_ns": first_ns, "idle_rss_bytes": idle_rss,
                "sampled_peak_rss_bytes": peak[0], "rss_sampling_seconds": 0.01,
                "model_weight_bytes": sum(p.stat().st_size for p in weight_paths),
                "disk_cache_state": "setup artifacts already cached; not cold disk"}
    manifest["long_workload"] = "Repeated development text, truncated at 256; synthetic stress fixture, no quality labels"
    write_json(path / "manifest.json", manifest)
    (path / "timings.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    with (path / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Completed {variant} repetition {repetition}: {len(records)} timing records")


def benchmark():
    run_root = ROOT / "results" / ("run-" + uuid4().hex[:12])
    run_root.mkdir()
    write_json(run_root / "protocol.json", config())
    failures = []
    for repetition in range(config()["repetitions"]):
        order = VARIANTS[repetition % 3:] + VARIANTS[:repetition % 3]
        for variant in order:
            destination = run_root / f"{repetition}-{variant}"
            command = [sys.executable, "-m", "quantbench.cli", "worker", "--variant", variant,
                       "--repetition", str(repetition), "--destination", str(destination)]
            completed = subprocess.run(command, capture_output=True, text=True)
            print(completed.stdout, flush=True)
            if completed.returncode:
                failures.append({"variant": variant, "repetition": repetition, "error": completed.stderr})
                print(completed.stderr, file=sys.stderr)
    write_json(run_root / "failures.json", failures)
    write_json(ROOT / "results" / "latest.json", {"run": run_root.name})
    if failures:
        raise RuntimeError(f"Some scenarios failed; see {run_root / 'failures.json'}")
    print(f"Benchmark complete: {run_root}")
