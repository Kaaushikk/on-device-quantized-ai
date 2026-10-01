# Optional extensions

The completed core study remains in `reports/COMPARISON.md`. These additions
answer separate questions and preserve its original measurements.

## Local HTTP service

Install the updated `requirements.lock.txt` into the main environment. Start:

```powershell
.\scripts\run.ps1 serve --variant onnx_int8 --port 8000
```

The server binds to `127.0.0.1`. Readiness is `GET /health/ready`.
Send `POST /v1/predict` with JSON:

```json
{"variant":"onnx_int8","texts":["I love this movie."]}
```

The response contains labels, scores in negative/positive order, the pinned
model revision, graph hash, and request ID. One model loads at startup. The
service accepts up to eight texts, each at most 4,096 characters; tokenization
truncates at 256 tokens. Blank texts and unexpected fields are rejected.
Concurrent prediction requests receive HTTP 503 while inference is busy.
Use this as a local demonstration, with no authentication or public deployment.
`scripts/check_api.py` verifies a real model through HTTP and records separate
localhost timings. Unit tests exercise lifecycle and concurrent requests.

## Static INT8

```powershell
.\scripts\run.ps1 static
.\scripts\run.ps1 evaluate --variant onnx_static_int8 --subset development
.\scripts\run.ps1 compare --variant onnx_static_int8 --subset development
```

Calibration uses the first 100 frozen development examples; no final examples
enter calibration. The first candidate quantized all matrix multiplications.
The second excludes attention multiplications with nonconstant weights. Both
lost 4.5 accuracy points on development data. The second candidate lost 2.98
points on the final set: 87.95% accuracy versus FP32's 90.92%. Its paired 95%
bootstrap interval for the difference is [-5.06, -1.04] percentage points.
It fails the one-point quality gate. Dynamic INT8 remains the default. We stop
tuning here; final-set results must not guide calibration changes.

## Intel GPU and NPU

A separate environment isolates OpenVINO's NumPy requirement:

```powershell
.\.venv\Scripts\python.exe -m venv .accelerator-venv
.\.accelerator-venv\Scripts\python.exe -m pip install -r requirements.accelerator.lock.txt
.\.venv\Scripts\python.exe scripts/prepare_accelerator.py
.\.venv\Scripts\python.exe scripts/run_accelerators.py
```

The worker explicitly selects GPU or NPU; AUTO and HETERO fallback are not
used. Quality uses all 672 final examples at fixed length 128, with no text
truncated. Timing uses the original frozen input pools at lengths 32/128/256,
20 warmups and 200 calls in each of three fresh processes. Profiling is enabled
for quality verification and disabled for timing. Synchronous timings include
host/device transfer and exclude tokenization. GPU/NPU request FP16; the CPU
control uses ONNX Runtime FP32, one thread. These compare complete
runtime/precision/device combinations, not isolated quantization gains.
The OpenVINO CPU process exceeded a 300-second timeout. It saved a passing
quality artifact with subsecond compilation, but did not exit successfully.
No OpenVINO CPU timing result is used as the baseline.

For the CPU control, run `scripts/accelerator_cpu_control.py 0`, then repeat
with 1 and 2 using the main environment, with other experiments stopped.
Raw traces, quality, compilation metadata and failures live in
`results/accelerators`. Downloaded tensor pools and graphs stay in `artifacts`.

## Energy and mobile scope

The Windows power-meter interface exposes zero readings without usable unit
or sampling information. The laptop is on AC and its battery discharge rate
is zero. Neither proves zero model energy consumption. No energy result
is claimed. `results/hardware-extensions.json` records this capability check.
No phone is connected or evaluated; GPU/NPU results describe this laptop only.

An external calibrated meter can supply CSV headers `time_seconds,watts`:

```powershell
.\.venv\Scripts\python.exe scripts/import_power.py trace.csv --meter "Meter model and calibration" --scope "Whole laptop, named workload and interval" --output results/power.json
```

The importer rejects unusable readings and integrates measured watts over
seconds into joules. It reports gross energy, without idle subtraction. Start
and end the trace at workload boundaries; record AC/battery state, meter
scope, sampling cadence and idle baseline separately before comparing runs.
The integration tests use synthetic inputs solely to check arithmetic.
