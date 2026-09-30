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
    return read_json(ROOT / "configs" / "experiment.json")


def verify_files(manifest):
    for relative, expected in manifest["files"].items():
        path = (ARTIFACTS / relative).resolve()
        if not path.is_relative_to(ARTIFACTS.resolve()):
            raise ValueError("Artifact path escapes artifact directory")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact missing or hash mismatch: {relative}")
