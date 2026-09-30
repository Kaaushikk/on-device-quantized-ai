import random
from collections import Counter
import numpy as np
from quantbench.common import ARTIFACTS, ROOT, config, read_json, sha256, versions, write_json


def setup():
    from huggingface_hub import HfApi, snapshot_download
    from datasets import load_dataset
    settings = config()
    api = HfApi()
    # Freeze resolved revisions once. Further setup runs reuse the same commits.
    pin_path = ROOT / "configs" / "revisions.json"
    if pin_path.exists():
        pins = read_json(pin_path)
    else:
        pins = {"model_revision": api.model_info(settings["model_id"]).sha,
                "dataset_revision": api.dataset_info(settings["dataset_id"]).sha}
        write_json(pin_path, pins)
    snapshot_download(settings["model_id"], revision=pins["model_revision"],
                      local_dir=ARTIFACTS / "model",
                      allow_patterns=["config.json", "model.safetensors", "tokenizer*", "vocab.txt", "README.md"])
    dataset = load_dataset(settings["dataset_id"], revision=pins["dataset_revision"], split="validation")
    ids = list(range(len(dataset)))
    random.Random(settings["seed"]).shuffle(ids)
    dev_ids = ids[:settings["development_examples"]]
    final_ids = ids[settings["development_examples"]:]
    rows = [{"row_id": i, "text": dataset[i]["sentence"], "label": dataset[i]["label"]} for i in range(len(dataset))]
    split_path = ROOT / "configs" / "split.json"
    if split_path.exists():
        existing = read_json(split_path)
        if (existing["development_ids"] != dev_ids or existing["final_ids"] != final_ids
                or existing["seed"] != settings["seed"] or existing["model_revision"] != pins["model_revision"]
                or existing["dataset_revision"] != pins["dataset_revision"]):
            raise ValueError("Setup would change the frozen split; no data file was overwritten")
    write_json(ARTIFACTS / "data.json", {"rows": rows, "development_ids": dev_ids, "final_ids": final_ids})
    split_manifest = {**pins, "seed": settings["seed"],
               "dataset_id": settings["dataset_id"], "split": "validation",
               "development_ids": dev_ids, "final_ids": final_ids,
               "local_data_sha256": sha256(ARTIFACTS / "data.json")}
    if split_path.exists() and read_json(split_path) != split_manifest:
        raise ValueError("Setup differs from frozen split; restore original configuration/data")
    write_json(split_path, split_manifest)
    files = {str(path.relative_to(ARTIFACTS)).replace("\\", "/"): sha256(path)
             for path in (ARTIFACTS / "model").iterdir() if path.is_file()}
    write_json(ARTIFACTS / "pytorch_fp32.manifest.json", {
        "schema_version": 1, "precision": "FP32", "model_id": settings["model_id"],
        **pins, "versions": versions(), "files": files, "labels": {"0": "NEGATIVE", "1": "POSITIVE"}})
    print("Pinned local model and 872 validation rows; frozen development/final split.")


def export():
    import torch
    import onnx
    from quantbench.runners.local import LocalRunner
    runner = LocalRunner("pytorch_fp32", config()["threads"])
    runner.load()

    class Logits(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    sample = runner.prepare(runner.preprocess(["Export fixture"], 32))
    path = ARTIFACTS / "model.fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(Logits(runner.model).eval(), tuple(sample.values()), str(path),
                          input_names=["input_ids", "attention_mask"], output_names=["logits"],
                          dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                                        "attention_mask": {0: "batch", 1: "sequence"}, "logits": {0: "batch"}},
                          opset_version=config()["opset"], dynamo=False)
    onnx.checker.check_model(str(path))
    manifest = {**runner.manifest, "precision": "FP32", "graph": path.name,
                "opset": config()["opset"], "exporter": "torch.onnx legacy, eager attention",
                "files": {**runner.manifest["files"], path.name: sha256(path)}}
    write_json(ARTIFACTS / "onnx_fp32.manifest.json", manifest)
    print("Exported and checked dynamic ONNX FP32 graph.")


def parity():
    from quantbench.runners.local import LocalRunner
    settings = config()
    reference = LocalRunner("pytorch_fp32", settings["threads"])
    candidate = LocalRunner("onnx_fp32", settings["threads"])
    reference.load()
    candidate.load()
    checks = []
    for batch in (1, 4):
        for length in settings["sequence_lengths"]:
            texts = ["I enjoyed this film.", "This was awful.", "Unicode café 世界", "long " * 600][:batch]
            inputs = reference.preprocess(texts, length)
            actual = candidate.predict(inputs)
            expected = reference.predict(reference.prepare(inputs))
            np.testing.assert_allclose(actual, expected, atol=settings["parity_atol"], rtol=settings["parity_rtol"])
            np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
            checks.append({"batch": batch, "sequence": length, "max_absolute_error": float(np.max(np.abs(actual - expected)))})
    write_json(ROOT / "results" / "parity.json", {"passed": True, "atol": settings["parity_atol"],
               "rtol": settings["parity_rtol"], "checks": checks,
               "pytorch_manifest": reference.manifest, "onnx_manifest": candidate.manifest})
    print("FP32 parity passed across six shapes including Unicode and truncation.")


def quantize():
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from quantbench.common import verify_files
    manifest = read_json(ARTIFACTS / "onnx_fp32.manifest.json")
    verify_files(manifest)
    source = ARTIFACTS / manifest["graph"]
    prepared = ARTIFACTS / "model.preprocessed.onnx"
    destination = ARTIFACTS / "model.int8.onnx"
    # Keep graph rewriting separate, and skip fusion so both runtime paths
    # retain comparable optimizer behavior at session creation.
    quant_pre_process(str(source), str(prepared), skip_optimization=True, auto_merge=True)
    quantize_dynamic(str(prepared), str(destination), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"], per_channel=False,
                     extra_options={"MatMulConstBOnly": True})
    onnx.checker.check_model(str(destination))
    counts = dict(Counter(node.op_type for node in onnx.load(str(destination)).graph.node))
    if not counts.get("MatMulInteger"):
        raise ValueError("Conversion produced no quantized matrix multiplication")
    write_json(ARTIFACTS / "onnx_int8.manifest.json", {
        **manifest, "precision": "dynamic INT8 weights / UINT8 activations", "graph": destination.name,
        "quantization": {"operators": ["MatMul"], "per_channel": False, "constant_weights_only": True,
                         "symbolic_shape_auto_merge": True, "preprocess_graph_optimization": False},
        "operator_counts": counts, "preprocessed_sha256": sha256(prepared),
        "files": {**{key: value for key, value in manifest["files"].items() if key != manifest["graph"]},
                  destination.name: sha256(destination)}})
    print(f"Quantized graph checked: {counts.get('MatMulInteger')} integer MatMul operators.")
