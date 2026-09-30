# Four-minute walkthrough

## First minute: what the project measures

This study runs one pretrained sentiment classifier on a Windows laptop CPU.
The three variants separate PyTorch-to-ONNX runtime effects from FP32-to-INT8
quantization effects. Read `reports/COMPARISON.md` for actual measurements.
The experiment does not claim phone or accelerator performance.

## Second minute: correctness and provenance

Show `configs/revisions.json` and `configs/split.json`: immutable model/data
commits and frozen development/final row IDs. Show `results/parity.json`: the
ONNX FP32 graph matches PyTorch across six shapes at declared tolerances.
INT8 changes arithmetic, so it uses task accuracy and paired uncertainty
instead of demanding equal floating-point logits.

## Third minute: local demo

Run from the repository directory in PowerShell:

```powershell
.\scripts\run.ps1 demo --variant onnx_int8 --text 'I enjoyed this movie.'
.\scripts\run.ps1 demo --variant onnx_fp32 --text 'This movie was disappointing.'
```

The demo loads local artifacts, verifies hashes, and reports label, scores,
variant, and source revision. Scores are model outputs, not calibrated truth.
No remote inference request is made. Blank texts are rejected and long texts
are truncated. Sentiment labels do not generalize to other tasks.

## Fourth minute: results and limits

Show the report and latency chart, then a raw run manifest and timing file.
Point out the baseline named for every speedup, three process repetitions,
separate tokenization timing, full-process RSS, and cached startup timing.
The long scenario is explicitly repeated text, because natural SST-2 examples
are short. Discuss accuracy loss with its interval, rather than only its point
estimate. Explain that model size reduction alone does not prove lower latency.

For debugging history, show `docs/WORK_LOG.md`: symbolic shape inference failed
initially, and the supported auto-merge setting resolved it before quality and
performance checks. Use `docs/REPRODUCE.md` to repeat the complete study.
