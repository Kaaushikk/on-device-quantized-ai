import csv
import statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from quantbench.common import ARTIFACTS, ROOT, VARIANTS, read_json, write_json


def generate():
    from quantbench.audit import validate_run
    run = ROOT / "results" / read_json(ROOT / "results" / "latest.json")["run"]
    if read_json(run / "failures.json"):
        raise ValueError("Run has failed scenarios; inspect failures before reporting")
    write_json(run / "audit.json", validate_run(run))
    groups = defaultdict(list)
    manifests = defaultdict(list)
    for path in sorted(run.glob("*/summary.csv")):
        with path.open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["variant"], row["scope"], int(row["batch_size"]), int(row["sequence_length"]))
                groups[key].append(row)
        manifest = read_json(path.parent / "manifest.json")
        manifests[manifest["variant"]].append(manifest)
    protocol = read_json(run / "protocol.json")
    expected_groups = len(VARIANTS) * 2 * len(protocol["batch_sizes"]) * len(protocol["sequence_lengths"])
    if len(groups) != expected_groups or any(len(rows) != protocol["repetitions"] for rows in groups.values()):
        raise ValueError("Incomplete benchmark scenarios or repetitions")
    quality = {variant: read_json(ROOT / "results" / f"quality-final-{variant}.json") for variant in VARIANTS}
    for variant in VARIANTS:
        if quality[variant]["row_ids"] != quality[VARIANTS[0]]["row_ids"]:
            raise ValueError("Final quality row IDs differ")
        for manifest in manifests[variant]:
            if manifest["runner"]["artifact"] != quality[variant]["metadata"]["artifact"]:
                raise ValueError("Timing and quality artifacts differ")
            if manifest["data_sha256"] != quality[variant]["data_sha256"]:
                raise ValueError("Timing and quality data differ")
    comparison = read_json(ROOT / "results" / "quality-final-comparison.json")
    table = []
    for (variant, scope, batch, length), rows in groups.items():
        p50s = [float(row["p50_batch_ms"]) for row in rows]
        table.append({"variant": variant, "scope": scope, "batch": batch, "sequence": length,
                      "median_p50_ms": statistics.median(p50s), "min_p50_ms": min(p50s), "max_p50_ms": max(p50s),
                      "median_p95_ms": statistics.median(float(row["p95_batch_ms"]) for row in rows),
                      "median_examples_per_second": statistics.median(float(row["examples_per_second"]) for row in rows)})
    destination = ROOT / "reports"
    write_json(destination / "comparison.json", {"source_run": run.name, "scenarios": table, "quality_comparison": comparison})
    with (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = ["# Local CPU quantization study", "", f"Source run: `results/{run.name}`.", "",
             "Measured on this Windows laptop. Results apply to this model, runtime, and workload only.", "",
             f"CPU: {next((m['hardware']['cpu_name'] for items in manifests.values() for m in items if m['hardware']['cpu_name']), manifests[VARIANTS[0]][0]['hardware']['processor'])}",
             f"OS: {manifests[VARIANTS[0]][0]['hardware']['platform']}",
             f"Power: {manifests[VARIANTS[0]][0]['hardware']['power_scheme']}", "",
             "## Final held-out validation quality", "",
             "| Variant | N | Accuracy | Macro F1 | Model weight MiB | Median idle RSS MiB | Median sampled peak RSS MiB |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        q = quality[variant]
        meta = manifests[variant]
        lines.append(f"| {variant} | {q['n']} | {q['accuracy']:.4f} | {q['macro_f1']:.4f} | {meta[0]['model_weight_bytes']/2**20:.2f} | {statistics.median(m['idle_rss_bytes'] for m in meta)/2**20:.2f} | {statistics.median(m['sampled_peak_rss_bytes'] for m in meta)/2**20:.2f} |")
    lines.extend(["", "## Cached startup", "", "Medians across fresh-process repetitions. Hash validation is outside load timing; tokenizer and model loading are inside it.", "",
                  "| Variant | Load ms | First 32-token inference ms |", "| --- | ---: | ---: |"])
    for variant in VARIANTS:
        meta = manifests[variant]
        lines.append(f"| {variant} | {statistics.median(m['load_ns'] for m in meta)/1e6:.2f} | {statistics.median(m['first_inference_ns'] for m in meta)/1e6:.2f} |")
    difference = comparison["accuracy_difference_int8_minus_fp32"] * 100
    interval = [value * 100 for value in comparison["paired_bootstrap_95_interval"]]
    lines.extend(["", f"INT8 minus ONNX FP32 accuracy: {difference:.3f} percentage points; paired bootstrap 95% interval [{interval[0]:.3f}, {interval[1]:.3f}].",
                  f"Changed predictions: {len(comparison['changed_row_ids'])}. Point-estimate one-point gate passed: {comparison['point_estimate_gate_passed']}.",
                  "This gate is a project criterion; a point estimate alone does not establish equivalence.", "",
                  "## Repeated device measurements", "",
                  "Values are medians of three per-process summaries, not pooled request percentiles.", "",
                  "| Variant | Scope | Batch | Padded length | p50 ms (min–max) | p95 ms | Examples/s |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    baseline_quality = quality["onnx_fp32"]
    int8_quality = quality["onnx_int8"]
    error_lines = ["# Changed-prediction analysis", "", "Final subset only. No raw dataset text is published.", "",
                   "| Row ID | True label | FP32 prediction | INT8 prediction | FP32 absolute logit margin | INT8 absolute logit margin |",
                   "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    harmed = helped = unchanged_wrong = 0
    for i, row_id in enumerate(baseline_quality["row_ids"]):
        original = baseline_quality["predictions"][i]
        integer = int8_quality["predictions"][i]
        expected = baseline_quality["labels"][i]
        if original != integer:
            harmed += int(original == expected)
            helped += int(integer == expected)
            margin_fp = abs(baseline_quality["logits"][i][0] - baseline_quality["logits"][i][1])
            margin_int = abs(int8_quality["logits"][i][0] - int8_quality["logits"][i][1])
            error_lines.append(f"| {row_id} | {expected} | {original} | {integer} | {margin_fp:.4f} | {margin_int:.4f} |")
        elif original != expected:
            unchanged_wrong += 1
    error_lines.extend(["", f"INT8 turns {harmed} FP32-correct predictions into errors and fixes {helped} FP32 errors; {unchanged_wrong} shared errors remain.",
                        "These are paired outcomes, not evidence that quantization causes a particular semantic bias. Small logit margins indicate boundary sensitivity; scores are not calibrated probabilities."])
    (destination / "ERROR_ANALYSIS.md").write_text("\n".join(error_lines) + "\n", encoding="utf-8")
    lines.insert(lines.index("## Repeated device measurements") - 1, f"[Changed-prediction analysis](ERROR_ANALYSIS.md): {harmed} harmed and {helped} helped predictions.")
    for row in table:
        lines.append(f"| {row['variant']} | {row['scope']} | {row['batch']} | {row['sequence']} | {row['median_p50_ms']:.3f} ({row['min_p50_ms']:.3f}–{row['max_p50_ms']:.3f}) | {row['median_p95_ms']:.3f} | {row['median_examples_per_second']:.2f} |")
    lines.extend(["", "## Quantization effects", ""])
    for length in protocol["sequence_lengths"]:
        fp = next(row for row in table if row['variant'] == 'onnx_fp32' and row['scope'] == 'inference_only' and row['sequence'] == length and row['batch'] == 1)
        integer = next(row for row in table if row['variant'] == 'onnx_int8' and row['scope'] == 'inference_only' and row['sequence'] == length and row['batch'] == 1)
        lines.append(f"- Length {length}, batch 1: ONNX FP32 / INT8 p50 = {fp['median_p50_ms']/integer['median_p50_ms']:.2f}×.")
    reduction = 1 - manifests['onnx_int8'][0]['model_weight_bytes'] / manifests['onnx_fp32'][0]['model_weight_bytes']
    lines.extend(["", f"ONNX weight-file size reduction: {reduction:.1%}.", "", "## Limitations and evidence", "",
                  "No thermal or energy measurement; background activity and OS scheduling can affect timings. Power scheme is recorded, not controlled programmatically. Warmup uses a fixed 20 calls; no statistical stationarity test is applied. First inference uses the 32-token scenario. Load time includes tokenizer and model loading after hash validation; artifacts were already cached.", "",
                  "The 10 ms sampler measures whole-process RSS and can miss brief peaks. Short/medium performance inputs rotate a bounded development pool, padded to scenario lengths. The 256-token workload repeats development text and is a synthetic stress fixture, not naturally long SST-2 text. This is not a representative production traffic study. No mobile, accelerator, container, or remote API performance is claimed.", "",
                  "Final accuracy uses a frozen validation-derived subset, not public test labels. The model and sentiment dataset have task-specific biases. INT8 leaves unsupported operators and embeddings in floating point. Input text and downloaded weights are excluded from Git.", "",
                  f"Raw timings and hardware/artifact manifests: [run directory](../results/{run.name}/). Quality records: `results/quality-final-*.json`. FP32 parity: [parity evidence](../results/parity.json). Setup and commands: [reproduction guide](../docs/REPRODUCE.md).", "",
                  "![Warm latency](latency.png)", "", "![Size, memory, and quality](resources.png)"])
    (destination / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for variant in VARIANTS:
        rows = sorted([row for row in table if row['variant'] == variant and row['scope'] == 'inference_only' and row['batch'] == 1], key=lambda row: row['sequence'])
        axis.plot([row['sequence'] for row in rows], [row['median_p50_ms'] for row in rows], marker='o', label=variant)
    axis.set(xlabel="Padded tensor sequence length", ylabel="Median of repetition p50 batch latency (ms)", title="Native laptop CPU inference — batch size 1")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination / "latency.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = ["PyTorch FP32", "ONNX FP32", "ONNX INT8"]
    axes[0].bar(names, [manifests[v][0]['model_weight_bytes']/2**20 for v in VARIANTS])
    axes[0].set(title="Model weight files", ylabel="MiB")
    axes[1].bar(names, [statistics.median(m['sampled_peak_rss_bytes'] for m in manifests[v])/2**20 for v in VARIANTS])
    axes[1].set(title="Sampled peak process RSS", ylabel="MiB")
    axes[2].bar(names, [quality[v]['accuracy']*100 for v in VARIANTS])
    axes[2].set(title=f"Held-out accuracy (N={comparison['n']})", ylabel="Percent", ylim=(0, 100))
    for axis in axes:
        axis.tick_params(axis='x', labelrotation=30)
        axis.grid(axis='y', alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination / "resources.png", dpi=160)
    plt.close(figure)
    print(f"Report generated: {destination / 'COMPARISON.md'}")
