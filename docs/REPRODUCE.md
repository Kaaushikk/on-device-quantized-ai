# Reproduce the study on Windows

Use Python 3.12, a CPU laptop, sufficient free memory, and several GB of disk
space. The setup step downloads a pretrained model and SST-2 validation data.
It does not train a model or call a remote inference API. Model and data stay
in ignored `artifacts`. `configs/revisions.json` and `configs/split.json`
record exact source commits and frozen row IDs after setup.

## Install and verify

From the repository directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The requirements lock records the resolved Windows/Python 3.12 environment.
It is a version freeze, not a hash-verified multi-platform lock.

## Prepare artifacts and check correctness

```powershell
.\scripts\run.ps1 setup
.\scripts\run.ps1 export
.\scripts\run.ps1 parity
.\scripts\run.ps1 quantize
.\scripts\run.ps1 evaluate --variant pytorch_fp32 --subset development
.\scripts\run.ps1 evaluate --variant onnx_fp32 --subset development
.\scripts\run.ps1 evaluate --variant onnx_int8 --subset development
.\scripts\run.ps1 compare --subset development
```

Only setup uses the network. Other commands enforce Hugging Face offline
mode and load local files with verified hashes. FP32 parity checks six shapes
at the unchanged declared tolerances. INT8 conversion separately preprocesses
shapes, skips preprocessing graph fusion, and quantizes constant-weight
MatMul operators using dynamic unsigned activations and signed INT8 weights.
Symbolic shape preprocessing enables `auto_merge=True`, needed for this graph.
Embeddings and other operators may remain FP32; the manifest records counts.

## Final quality and device timing

Once development decisions are fixed, evaluate all three variants with
`--subset final`, then run `compare --subset final`. This uses 672 held-out
validation rows if the original 872-row source and 200-row development split
are unchanged. No public test labels are used.
Quality evaluation pads to the longest example in each batch of eight,
with truncation at 512 tokens; this differs from fixed-shape performance timing.

```powershell
.\scripts\run.ps1 benchmark
.\scripts\run.ps1 report
.\scripts\run.ps1 demo --variant onnx_int8 --text 'This movie was excellent.'
```

The benchmark runs each variant in a new process and rotates order across
three repetitions. It times 200 iterations per length/scenario after 20 warmup
calls. Inputs come only from the development split and rotate identically.
Lengths 32/128/256 describe padded tensor sizes; the raw records include actual
unpadded lengths, which may be much shorter, especially in the longest bucket.
Padding work must not be described as a naturally long-text workload.

Outputs include per-process manifests, raw records, summaries, and any errors.
Do not compare different run directories as if power/background conditions
were identical. Close heavy applications and keep power settings consistent.
Thermal state and energy are not measured. Process RSS includes Python and
runtime overhead, and a 10 ms sampler can miss brief peaks.

## Source and licensing notes

The [model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
declares Apache 2.0 and documents sentiment-specific biases. SST-2 derives from
the Stanford Sentiment Treebank. This repository publishes row IDs and results,
not dataset sentences or model weights; users download sources during setup.
Review upstream terms before redistributing either source. Export strategy and
quantization follow the [ONNX Runtime guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
