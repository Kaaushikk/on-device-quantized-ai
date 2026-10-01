"""Save shared tensors and CPU references; accelerator workers need no tokenizer."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quantbench.common import ARTIFACTS, read_json, sha256, write_json
from quantbench.experiment import data
from quantbench.runners.local import LocalRunner


def main():
    runner = LocalRunner("onnx_fp32")
    runner.load()
    dataset = data()
    ids = dataset["final_ids"]
    texts = [dataset["rows"][i]["text"] for i in ids]
    lengths = [len(runner.tokenizer(text)["input_ids"]) for text in texts]
    if max(lengths) > 128:
        raise ValueError("Fixed 128-token quality shape would truncate final examples")
    # Quality tensor generation is shared by every accelerator; no device tuning.
    inputs = runner.tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="np")
    reference = []
    for index in range(len(ids)):
        reference.append(runner.predict({key: np.asarray(inputs[key][index:index+1], dtype=np.int64)
                                         for key in ("input_ids", "attention_mask")})[0])
    path = ARTIFACTS / "accelerator-inputs.npz"
    np.savez_compressed(path, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        labels=np.array([dataset["rows"][i]["label"] for i in ids]), reference_logits=np.array(reference))
    write_json(ROOT / "configs" / "accelerator.json", {
        "schema_version": 1, "row_ids": ids, "quality_sequence_length": 128, "quality_batch": 1,
        "max_actual_tokens": max(lengths), "input_sha256": sha256(path),
        "graph_sha256": sha256(ARTIFACTS / "model.fp32.onnx"),
        "reference": "ONNX Runtime CPU FP32 on the same fixed tensors", "model_revision": runner.manifest["model_revision"],
        "performance_lengths": [32, 128, 256], "warmup": 20, "iterations": 200, "repetitions": 3,
        "performance_source": "Original core-run repetition 0 prepared input IDs and repeat counts",
        "core_run": read_json(ROOT / "results" / "latest.json")["run"]})
    # Rebuild the original bounded pools from its raw timing records once.
    import json
    timings = ROOT / "results" / read_json(ROOT / "results" / "latest.json")["run"] / "0-onnx_fp32" / "timings.jsonl"
    seen = set()
    arrays = {}
    trace = {}
    for line in timings.read_text().splitlines():
        record = json.loads(line)
        if record["scope"] != "inference_only":
            continue
        length = record["sequence_length"]
        if (length, record["input_ids"][0]) in seen:
            continue
        seen.add((length, record["input_ids"][0]))
        text = " ".join([dataset["rows"][record["input_ids"][0]]["text"]] * record["repeat_counts"][0])
        encoded = runner.preprocess([text], length)
        for key, value in encoded.items():
            arrays.setdefault(f"{key}_{length}", []).append(value)
        trace.setdefault(str(length), []).append({"row_id": record["input_ids"][0], "repeat_count": record["repeat_counts"][0]})
    pool_path = ARTIFACTS / "accelerator-performance.npz"
    np.savez_compressed(pool_path, **{key: np.concatenate(values, axis=0) for key, values in arrays.items()})
    write_json(ROOT / "results" / "accelerator-inputs.json", {
        "final_ids": ids, "quality_sha256": sha256(path), "performance_sha256": sha256(pool_path),
        "performance_pools": trace, "no_raw_text_published": True})
    print(f"Shared tensors saved; {len(ids)} final examples, max actual length {max(lengths)}.")


if __name__ == "__main__":
    main()
