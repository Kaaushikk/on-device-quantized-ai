"""Demonstrate all variants while rejecting Python socket network operations."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from quantbench.common import VARIANTS, write_json

CODE = '''
import sys
def block_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
        raise RuntimeError("Network operation blocked by offline verification")
sys.addaudithook(block_network)
import json
import numpy as np
from quantbench.runners.local import LocalRunner
runner = LocalRunner(sys.argv[1])
runner.load()
try:
    runner.preprocess([""])
except ValueError:
    pass
else:
    raise AssertionError("Blank text was accepted")
inputs = runner.preprocess(["This film was excellent.", "Unicode café 世界 " + "long " * 1000], 32)
assert inputs["input_ids"].shape == (2, 32)
assert inputs["input_ids"].dtype == np.int64
assert inputs["attention_mask"].dtype == np.int64
logits = runner.predict(runner.prepare(inputs))
assert logits.shape == (2, 2) and np.isfinite(logits).all()
assert int(logits[0].argmax()) == 1
print(json.dumps({"labels": logits.argmax(1).tolist(), "model_revision": runner.manifest["model_revision"],
                  "checks": ["blank rejection", "Unicode batch", "truncation", "int64 tensors", "finite logits", "positive fixture"]}))
'''


def main():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    checks = []
    for variant in VARIANTS:
        command = [sys.executable, "-c", CODE, variant]
        completed = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        checks.append({"variant": variant, "prediction": json.loads(completed.stdout), "passed": True})
    write_json(ROOT / "results" / "offline.json", {
        "passed": True, "method": "Python audit hook rejects socket connect, DNS lookup, and sendto; HF offline flags enabled",
        "limitation": "This verifies Python socket use; it is not an OS firewall isolation test", "checks": checks})
    print("All three local runners and input-policy checks passed with Python network operations blocked.")


if __name__ == "__main__":
    main()
