# On-device quantized AI benchmark

Project 17 compares a sentiment classifier running locally in PyTorch FP32,
ONNX FP32, and ONNX INT8. The goal is an honest comparison of speed, size,
memory, and accuracy on a Windows laptop. A slowdown is a valid result.

## Current status

The real model pipeline, held-out quality evaluation, and native laptop study
are complete. Six-shape FP32 parity, ten unit tests, a 10,800-record timing
audit, and blocked-network local inference checks pass. A clean GitHub checkout
with freshly installed locked dependencies also passes the tests, offline
inference checks, and FP32 parity. It reuses verified local artifacts; the full
timing study and fresh downloads were not repeated in that verification.

Measured on an Intel Core Ultra 9 185H laptop CPU with one thread and batch 1:

- ONNX INT8 p50 inference speedup over ONNX FP32: **2.43–2.91×** across the three tested lengths.
- ONNX weight-file size reduction: **48.2%** (255.53 MiB to 132.43 MiB).
- Held-out accuracy, N=672: **90.92% FP32 vs 90.18% INT8**.
- INT8 accuracy loss: **0.744 percentage points**; the paired interval does not establish equivalence within the chosen one-point margin.

The 256-token scenario uses repeated-text stress fixtures. Tail latency varies,
and RSS includes full-process overhead. Read the [comparison report](reports/COMPARISON.md)
for all scenarios, uncertainty, regressions, and links to raw evidence.

Use [the reproduction guide](docs/REPRODUCE.md) for environment installation,
real model commands, offline inference, and measurement limitations.

Try the local demo from this folder in PowerShell:

```powershell
.\scripts\run.ps1 demo --text 'I really enjoyed this movie.'
```

After installing the locked dependencies with Python 3.12, run the checks:

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
python scripts/smoke.py
```

The smoke script uses a synthetic fixture and writes ignored local outputs.
It verifies plumbing; it is not a real model benchmark.

## Where things belong

Optional extensions include a local HTTP API, static INT8 calibration, and
an explicit Intel GPU/NPU study. Static INT8 failed the quality gate; the API
keeps dynamic INT8 as its default. See [extension instructions](docs/EXTENSIONS.md)
and [extension results](reports/EXTENSIONS.md). Energy and phone measurements
require hardware that is not available in this run.

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
The [plain-language glossary](docs/GLOSSARY.md) explains technical terms.

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
