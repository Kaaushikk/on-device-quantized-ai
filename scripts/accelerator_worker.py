"""Explicit-device OpenVINO study; no AUTO/HETERO fallback and no network use."""
import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter_ns


def block_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        raise RuntimeError("Accelerator experiment must remain offline")


sys.addaudithook(block_network)
import numpy as np
import openvino as ov
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quantbench.common import ARTIFACTS, read_json, sha256, write_json
from quantbench.metrics import summarize


def model_for(core, device, length, profiling=True):
    model = core.read_model(ARTIFACTS / "model.fp32.onnx")
    model.reshape({"input_ids": [1, length], "attention_mask": [1, length]})
    options = {"PERFORMANCE_HINT": "LATENCY", "PERF_COUNT": profiling}
    if device == "CPU":
        options.update({"INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": 1, "NUM_STREAMS": 1})
    else:
        options["INFERENCE_PRECISION_HINT"] = "f16"
    compiled = core.compile_model(model, device, options)
    execution_devices = compiled.get_property("EXECUTION_DEVICES")
    actual = [execution_devices] if isinstance(execution_devices, str) else list(execution_devices)
    if not actual or any(not name.startswith(device) for name in actual):
        raise ValueError(f"Unexpected fallback devices: {actual}")
    return compiled, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["CPU", "GPU", "NPU"], required=True)
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args()
    folder = ROOT / "results" / "accelerators"
    folder.mkdir(exist_ok=True)
    settings = read_json(ROOT / "configs" / "accelerator.json")
    if sha256(ARTIFACTS / "model.fp32.onnx") != settings["graph_sha256"]:
        raise ValueError("FP32 graph hash differs")
    if sha256(ARTIFACTS / "accelerator-inputs.npz") != settings["input_sha256"]:
        raise ValueError("Quality input hash differs")
    core = ov.Core()
    if args.device not in core.available_devices:
        raise ValueError(f"Device unavailable: {args.device}")
    manifest = {"device": args.device, "full_device_name": core.get_property(args.device, "FULL_DEVICE_NAME"),
                "openvino": ov.__version__, "numpy": np.__version__, "model_sha256": settings["graph_sha256"],
                "repetition": args.repetition, "fallback_policy": "explicit single device; AUTO and HETERO forbidden",
                "requested_precision": "f32" if args.device == "CPU" else "f16",
                "quality_input_sha256": settings["input_sha256"]}
    if args.quality_only:
        start = perf_counter_ns()
        compiled, actual = model_for(core, args.device, 128)
        manifest.update(compile_ns=perf_counter_ns() - start, execution_devices=actual)
        request = compiled.create_infer_request()
        dataset = np.load(ARTIFACTS / "accelerator-inputs.npz")
        logits = []
        for i in range(len(dataset["labels"])):
            inputs = {key: dataset[key][i:i+1] for key in ("input_ids", "attention_mask")}
            result = request.infer(inputs)
            logits.append(np.array(result[compiled.output(0)])[0])
        logits = np.array(logits)
        if logits.shape != (len(dataset["labels"]), 2) or not np.isfinite(logits).all():
            raise ValueError("Invalid accelerator logits")
        predictions = logits.argmax(1)
        labels = dataset["labels"]
        matrix = np.zeros((2, 2), dtype=int)
        for truth, prediction in zip(labels, predictions, strict=True):
            matrix[truth, prediction] += 1
        f1 = [2*matrix[i, i]/(matrix[i].sum()+matrix[:, i].sum()) for i in range(2)]
        profiling = [{"node": item.node_name, "type": item.node_type, "execution_type": item.exec_type,
                      "status": str(item.status), "real_time_us": item.real_time.total_seconds()*1e6}
                     for item in request.get_profiling_info()]
        reference = dataset["reference_logits"].argmax(1)
        write_json(folder / f"quality-{args.device}.json", {"manifest": manifest, "row_ids": settings["row_ids"],
                   "n": len(labels), "accuracy": float((predictions == labels).mean()), "macro_f1": float(np.mean(f1)),
                   "confusion_matrix": matrix.tolist(), "predictions": predictions.tolist(), "labels": labels.tolist(),
                   "logits": logits.tolist(), "changed_predictions_vs_ort_fp32": int((predictions != reference).sum()),
                   "max_absolute_logit_error_vs_ort_fp32": float(np.max(np.abs(logits-dataset["reference_logits"]))),
                   "profiling": profiling})
        print(f"{args.device}: quality accuracy={float((predictions==labels).mean()):.4f}, actual devices={actual}", flush=True)
    else:
        trace = read_json(ROOT / "results" / "accelerator-inputs.json")
        path = ARTIFACTS / "accelerator-performance.npz"
        if sha256(path) != trace["performance_sha256"]:
            raise ValueError("Performance input hash differs")
        pools = np.load(path)
        records, summaries, compiles = [], [], []
        for length in settings["performance_lengths"]:
            start = perf_counter_ns()
            compiled, actual = model_for(core, args.device, length, profiling=False)
            compiles.append({"length": length, "compile_ns": perf_counter_ns()-start, "execution_devices": actual})
            request = compiled.create_infer_request()
            size = len(pools[f"input_ids_{length}"])
            inputs = [{key: pools[f"{key}_{length}"][i:i+1] for key in ("input_ids", "attention_mask")} for i in range(size)]
            for i in range(settings["warmup"]):
                request.infer(inputs[i % size])
            durations = []
            for i in range(settings["iterations"]):
                start = perf_counter_ns()
                result = request.infer(inputs[i % size])
                duration = perf_counter_ns() - start
                if not np.isfinite(result[compiled.output(0)]).all():
                    raise ValueError("Invalid timing output")
                durations.append(duration)
                records.append({"device": args.device, "repetition": args.repetition, "length": length,
                                "iteration": i, "duration_ns": duration, **trace["performance_pools"][str(length)][i % size]})
            summaries.append({"device": args.device, "repetition": args.repetition, "length": length, **summarize(durations, 1)})
        write_json(folder / f"manifest-{args.device}-{args.repetition}.json", {**manifest,
                   "compiles": compiles, "profiling_during_timing": False, "performance_input_sha256": trace["performance_sha256"],
                   "scope": "synchronous inference including host/device transfer; tokenizer excluded"})
        (folder / f"timings-{args.device}-{args.repetition}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records), encoding="utf-8")
        with (folder / f"summary-{args.device}-{args.repetition}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"{args.device} repetition {args.repetition}: {len(records)} timing records", flush=True)


if __name__ == "__main__":
    main()
