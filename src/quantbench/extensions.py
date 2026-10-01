"""Static calibration is a separate experiment from the original study."""
from collections import Counter
from quantbench.common import ARTIFACTS, ROOT, config, read_json, sha256, verify_files, versions, write_json


def quantize_static_model():
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from quantbench.experiment import data
    from quantbench.runners.local import LocalRunner
    dataset = data()
    # Development-only, fixed before static quality results are inspected.
    ids = dataset["development_ids"][:100]
    if set(ids) & set(dataset["final_ids"]):
        raise ValueError("Calibration overlaps final quality data")
    manifest = read_json(ARTIFACTS / "onnx_fp32.manifest.json")
    verify_files(manifest)
    runner = LocalRunner("onnx_fp32", config()["threads"])
    runner.load()

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rows = iter(ids)

        def get_next(self):
            row = next(self.rows, None)
            if row is None:
                return None
            return runner.preprocess([dataset["rows"][row]["text"]], 128, pad_to_max=False)

    source = ARTIFACTS / "model.preprocessed.onnx"
    if sha256(source) != read_json(ARTIFACTS / "onnx_int8.manifest.json")["preprocessed_sha256"]:
        raise ValueError("Preprocessed source differs from the validated dynamic experiment")
    destination = ARTIFACTS / "model.static.int8.onnx"
    # Preserve the initial all-MatMul candidate and its measured development
    # failure before the single planned sensitivity experiment.
    current_manifest = ARTIFACTS / "onnx_static_int8.manifest.json"
    archive_graph = ARTIFACTS / "model.static.all-matmul.int8.onnx"
    if (current_manifest.exists() and not archive_graph.exists()
            and not read_json(current_manifest).get("quantization", {}).get("selection")):
        previous = read_json(current_manifest)
        verify_files(previous)
        destination.rename(archive_graph)
        previous["files"][archive_graph.name] = previous["files"].pop(destination.name)
        previous["graph"] = archive_graph.name
        write_json(ARTIFACTS / "static-all-matmul.manifest.json", previous)
        initial_result = ROOT / "results" / "quality-development-onnx_static_int8.json"
        if initial_result.exists():
            write_json(ROOT / "results" / "quality-development-static-all-matmul.json", read_json(initial_result))
    graph = onnx.load(str(source))
    constant_names = {value.name for value in graph.graph.initializer}
    selected = [node.name for node in graph.graph.node if node.op_type == "MatMul" and node.input[1] in constant_names]
    if not selected:
        raise ValueError("No constant-weight MatMul nodes selected")
    quantize_static(str(source), str(destination), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, op_types_to_quantize=["MatMul"],
                    nodes_to_quantize=selected, per_channel=False,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    onnx.checker.check_model(str(destination))
    counts = dict(Counter(node.op_type for node in onnx.load(str(destination)).graph.node))
    if not counts.get("QuantizeLinear"):
        raise ValueError("No quantized operators in static graph")
    calibration = {"row_ids": ids, "subset": "development", "samples": len(ids),
                   "max_sequence_length": 128, "padding": "longest in batch", "method": "MinMax",
                   "data_sha256": sha256(ARTIFACTS / "data.json"), "final_overlap": 0}
    write_json(ROOT / "results" / "static-calibration.json", calibration)
    write_json(ARTIFACTS / "onnx_static_int8.manifest.json", {
        **manifest, "versions": versions(), "precision": "static INT8 QDQ activations and weights",
        "graph": destination.name, "calibration": calibration, "operator_counts": counts,
        "quantization": {"method": "QDQ S8S8 MinMax", "operators": ["MatMul"], "per_channel": False,
                         "selection": "constant-weight MatMul only; attention score/value MatMul excluded",
                         "selected_nodes": selected},
        "files": {**{key: value for key, value in manifest["files"].items() if key != manifest["graph"]},
                  destination.name: sha256(destination)}})
    print(f"Static QDQ graph checked; calibrated on {len(ids)} development examples, zero final overlap.")
