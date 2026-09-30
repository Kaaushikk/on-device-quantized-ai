import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
VARIANTS = ("pytorch_fp32", "onnx_fp32", "onnx_int8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "onnx", "onnxruntime", "datasets", "numpy", "psutil")}


def config():
    settings = read_json(ROOT / "configs" / "experiment.json")
    if settings.get("schema_version") != 1:
        raise ValueError("Unsupported experiment schema")
    for key in ("threads", "iterations", "repetitions"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["warmup"] < 1 or any(batch < 1 or batch > 32 for batch in settings["batch_sizes"]):
        raise ValueError("Invalid warmup or batch sizes")
    if not settings["sequence_lengths"] or any(length < 2 or length > 512 for length in settings["sequence_lengths"]):
        raise ValueError("Invalid sequence lengths")
    if settings["sequence_lengths"] != sorted(set(settings["sequence_lengths"])):
        raise ValueError("Sequence lengths must be unique and increasing")
    return settings


def verify_files(manifest):
    for relative, expected in manifest["files"].items():
        path = (ARTIFACTS / relative).resolve()
        if not path.is_relative_to(ARTIFACTS.resolve()):
            raise ValueError("Artifact path escapes artifact directory")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact missing or hash mismatch: {relative}")
