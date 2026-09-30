# On-device quantized AI benchmark

Project 17 compares a sentiment classifier running locally in PyTorch FP32,
ONNX FP32, and ONNX INT8. The goal is an honest comparison of speed, size,
memory, and accuracy on a Windows laptop. A slowdown is a valid result.

## Current status

The pinned local model, FP32 export, and dynamic INT8 conversion work. Six-shape
FP32 parity and seven unit tests pass locally. On the 200 development examples,
both ONNX variants score 91.5% accuracy; this is preliminary quality evidence.
Final quality and native device timing are in progress.

Use [the reproduction guide](docs/REPRODUCE.md) for environment installation,
real model commands, offline inference, and measurement limitations.

After installing the locked dependencies with Python 3.12, run the checks:

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python scripts/smoke.py
```

The smoke script uses a synthetic fixture and writes ignored local outputs.
It verifies plumbing; it is not a real model benchmark.

## Where things belong

| Folder | Purpose |
| --- | --- |
| `src/quantbench/runners` | Shared interface and inference implementations |
| `scripts` | Setup and reproducible experiment commands |
| `configs` | Versioned experiment settings |
| `tests` | Correctness checks |
| `artifacts` | Local downloaded models and converted graphs, excluded from Git |
| `results` | Raw timing records and run metadata |
| `reports` | Measured comparisons and charts |
| `docs` | Protocol, progress, decisions, and debugging explanations |

Read [the work log](docs/WORK_LOG.md) for completed actions and problems,
and [the protocol](docs/PROTOCOL.md) for measurement rules.

## Planned milestones

1. Tested harness and locked environment.
2. Pinned model, tokenizer, dataset, and PyTorch baseline.
3. Validated ONNX FP32 export.
4. Dynamic INT8 conversion and development quality check.
5. Repeated native laptop benchmarks in isolated processes.
6. Comparison report including failures and limitations.
7. Offline demo and clean-environment reproduction.

GitHub updates will contain meaningful, verified steps and documentation.
Large model weights, caches, secrets, and environments stay local.
