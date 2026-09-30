import json
from pathlib import Path
from quantbench.common import VARIANTS, read_json


def validate_run(run):
    run = Path(run)
    settings = read_json(run / "protocol.json")
    expected = 2 * len(settings["batch_sizes"]) * len(settings["sequence_lengths"]) * settings["iterations"]
    input_signature = None
    data_hash = None
    observed = []
    for repetition in range(settings["repetitions"]):
        for variant in VARIANTS:
            folder = run / f"{repetition}-{variant}"
            manifest = read_json(folder / "manifest.json")
            if manifest["settings"] != settings or manifest["variant"] != variant or manifest["repetition"] != repetition:
                raise ValueError("Run settings or identity mismatch")
            if data_hash is None:
                data_hash = manifest["data_sha256"]
            if manifest["data_sha256"] != data_hash:
                raise ValueError("Data hashes differ")
            rows = [json.loads(line) for line in (folder / "timings.jsonl").read_text(encoding="utf-8").splitlines()]
            if len(rows) != expected:
                raise ValueError("Timing record count differs from protocol")
            signature = []
            counts = {}
            for row in rows:
                if row["variant"] != variant or row["repetition"] != repetition or row["duration_ns"] <= 0:
                    raise ValueError("Invalid timing identity or duration")
                key = (row["scope"], row["batch_size"], row["sequence_length"])
                counts.setdefault(key, set()).add(row["iteration"])
                signature.append((key, row["iteration"], row["input_ids"], row["repeat_counts"], row["unpadded_token_lengths"]))
            expected_scenarios = {(scope, batch, length) for scope in ("inference_only", "tokenization_plus_inference")
                                  for batch in settings["batch_sizes"] for length in settings["sequence_lengths"]}
            if set(counts) != expected_scenarios or any(indices != set(range(settings["iterations"])) for indices in counts.values()):
                raise ValueError("Missing scenario or duplicate iteration")
            if input_signature is None:
                input_signature = signature
            if signature != input_signature:
                raise ValueError("Input IDs, repeat counts, or lengths differ between variants/repetitions")
            observed.append({"variant": variant, "repetition": repetition, "records": len(rows)})
    return {"passed": True, "processes": observed, "total_records": expected * len(observed),
            "checks": ["positive durations", "exact scenario/iteration counts", "matching inputs", "matching settings and data"]}
