# Optional extension results

These results supplement the completed core study. They preserve negative findings.

## Static INT8

Development accuracy fell from 91.5% to 87.0% for both static candidates. The selected constant-weight candidate reached 87.95% final accuracy versus 90.92% FP32: a 2.98 percentage-point loss (paired bootstrap 95% interval [-5.06, -1.04] points). It fails the one-point gate and is not the default. Calibration used 100 development examples and zero final examples.

## Laptop GPU/NPU

Intel Arc and Intel AI Boost were selected explicitly in OpenVINO 2025.3.0; no AUTO/HETERO CPU fallback was used. Each scored 90.92% on all 672 final examples. GPU/NPU request FP16; the CPU control uses ONNX Runtime 1.22.1 FP32 with one thread. Comparisons combine runtime, precision and hardware changes.

| Length | ORT CPU p50/p95 ms | GPU p50/p95 ms | NPU p50/p95 ms | GPU / NPU p50 speedup |
| --- | --- | --- | --- | --- |
| 32 | 38.05 / 46.88 | 10.72 / 11.44 | 3.45 / 4.56 | 3.55x / 11.02x |
| 128 | 123.79 / 145.90 | 11.69 / 12.65 | 7.54 / 8.60 | 10.58x / 16.42x |
| 256 | 249.30 / 290.82 | 19.89 / 21.03 | 15.16 / 17.89 | 12.54x / 16.44x |

Audited 5,400 timing records: 200 calls × 3 lengths × 3 repetitions × 3 configurations, after 20 warmups per shape. Table entries are medians of three process-level p50/p95 values; synchronous inference includes host/device transfer and excludes tokenization. Frozen fixture IDs and graph/input hashes match across configurations. Long inputs are synthetic repeated text, as in the core study.

GPU/NPU repetition order was rotated. CPU controls ran afterward, so time, temperature and device-specific threading remain confounders. No confidence interval or energy-efficiency claim is attached to these speedups. The OpenVINO CPU process exceeded a 300-second timeout after saving a passing quality artifact (90.92%, zero changed labels, 0.83-second compilation). Its exit failure remains unresolved; it supplies no timing baseline. Compilation time is recorded separately from inference.

GPU: 0 predicted labels changed versus ORT FP32; maximum absolute logit difference 0.04544. Equal accuracy does not imply equal logits.

NPU: 0 predicted labels changed versus ORT FP32; maximum absolute logit difference 0.04190. Equal accuracy does not imply equal logits.

## Local API

The real dynamic INT8 service passed readiness, positive/negative labels, blank-input rejection and wrong-variant rejection. The unit suite separately checks lifecycle, bounds, metadata and concurrent busy responses.

For two short texts per request, 200 localhost HTTP roundtrips after 20 warmups gave p50 10.11 ms and p95 11.86 ms. This is a single-process service check with HTTP overhead; it is not comparable to the core fixed-length timings.

## Energy and mobile

No energy or phone measurement is available. The exposed Windows power sensor returns unusable zero readings on AC. An external measured-power CSV importer is implemented and arithmetic/invalid-input tests pass; synthetic tests are not energy evidence.

See `docs/EXTENSIONS.md` for commands and `docs/WORK_LOG.md` for debugging. Raw traces and manifests are in `results/accelerators`; capability evidence is in `results/hardware-extensions.json`.
