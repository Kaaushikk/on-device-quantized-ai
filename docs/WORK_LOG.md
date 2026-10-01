# Project work log

This document records completed work, why it matters, checks, debugging,
and remaining steps in plain language. Update it with every meaningful
implementation or investigation and push it with the corresponding code.

## September 30, 2026 — project setup

- Read `C:\projects\Project_17_On_Device_Quantized_AI_Build_Plan.docx`
  directly from its document contents. It specifies a local classifier
  benchmark, not a cloud model service.
- Checked existing folders to avoid overwriting another project.
- Created `C:\projects\on-device-quantized-ai` with a documented folder layout.
- Wrote the initial benchmark protocol before measurements. This prevents
  adjusting the experiment just to obtain a favorable result.
- Added Git exclusions for environments, secrets, model weights, and caches.
- Verified Git is installed and the configured identity is Kaaushikk.
- Checked GitHub access: the connector can identify the account but lists
  no accessible repositories. GitHub CLI reports an invalid saved token.
  Publication is pending a fresh browser login using `gh auth login -h github.com`.
- Investigated Python availability: `python` is absent from PATH. Located
  the bundled Python runtime instead; no system installation was changed.

## September 30, 2026 — publishing and initial harness

- Completed GitHub CLI browser authentication as Kaaushikk and verified access.
- Created the public repository at https://github.com/Kaaushikk/on-device-quantized-ai
  and pushed the initial structure and protocol.
- Resolved Git's ownership check for this specific folder: sandbox-created files
  and the laptop account have different owners. Added only this repository to
  Git's trusted directory list so authenticated publishing can access it.
- Defined the shared runner interface and kept preprocessing separate from
  prediction so tokenization cannot accidentally enter inference-only timing.
- Added percentile and throughput calculations. Throughput counts examples;
  reported latency remains batch latency.
- Added three correctness tests, all passing on bundled Python 3.12.14.
- Ran a 200-iteration synthetic smoke check after ten warmup calls. It wrote
  a manifest, raw timing records, and a summary under ignored `results/local`.
  These fixture timings do not measure a neural model and are not published
  as performance evidence.
- Added Windows CI to run the same checks after pushes. Remote CI status must
  be checked separately; local success does not establish remote success.

## September 30, 2026 — real pipeline implementation

- Created a project-local `.venv` on Python 3.12.14. Installed pinned PyTorch
  2.8.0, Transformers 4.56.2, ONNX 1.19.0, ONNX Runtime 1.22.1, Datasets
  4.1.1, and supporting libraries. `pip check` found no broken requirements.
- Froze all resolved package versions in `requirements.lock.txt`; documented
  that this is a version freeze for Windows/Python 3.12, not a package-hash lock.
- Added versioned experiment settings and commands for setup, export, parity,
  quantization, evaluation, paired quality comparison, benchmark, demo, and report.
- Setup resolves and saves immutable source commits before downloading, freezes
  200 development row IDs and the remaining validation rows for final quality,
  and hashes local artifacts/data. Downloaded text/weights remain excluded.
- Real runners verify hashes, load offline, validate label/provider choices,
  reject blank input, limit batch size, and truncate at documented sequence lengths.
- Implemented graph export, six-shape FP32 parity, separate shape preprocessing,
  and dynamic constant-weight MatMul quantization. These require real-run validation
  before their results can be called successful.
- Implemented isolated-process timing with rotated variant order, input ID tracking,
  10 ms RSS sampling, startup timing, and hardware/power metadata.
- Added final-quality comparison with paired bootstrap uncertainty and report
  generation from raw results. No final-quality or performance result exists yet.
- Five unit tests pass, including rejection of modified artifact hashes and paths
  outside the artifact folder. Python source compilation also passes.
- Verified the previous Windows CI run completed successfully.
- Started the source downloads. Hugging Face uses ordinary HTTP because its
  optional Xet helper is absent; no extra helper is needed for correctness.

## September 30, 2026 — export validation and preprocessing debug

- Downloads completed: 872 SST-2 validation rows, with 200 frozen development
  examples and 672 final examples. Exact source commits and data hash are saved.
- Exported the classifier using eager attention and PyTorch's legacy ONNX
  exporter at opset 17. ONNX graph checking passed. A tracer warning concerned
  an attention-mask constant; tested parity rather than assuming it was harmless.
- FP32 parity passed all six batch/sequence combinations at the original
  atol=1e-4, rtol=1e-3. Largest observed absolute logit difference was about
  3.58e-6, and all tested predicted labels matched.
- First INT8 preprocessing attempt failed with `Incomplete symbolic shape
  inference`. Inspected the installed runtime's API and enabled its supported
  `auto_merge=True` option. Kept graph fusion disabled. Preprocessing and graph
  checking then passed; the converted graph has 38 MatMulInteger operators.
- Added exclusions for the runtime's diagnostic ONNX/external-data files so
  accidental model weights cannot be published from the repository root.
- Added two quality-metric tests; all seven local unit tests pass.
- Quality evaluation now pads to the longest example within each batch (up to
  512 tokens), avoiding needless fixed-length work. Performance still uses the
  protocol's fixed padded lengths. All variants use the same quality policy.
- Development ONNX FP32 accuracy is 0.915 on 200 examples. INT8 comparison and
  the remaining evaluations are still in progress; this is not a final result.

## September 30, 2026 — quality gates and performance workload

- Re-evaluated the PyTorch development baseline using the same longest-in-batch
  padding policy as ONNX. All variants score 0.915 development accuracy. Four
  predictions change between ONNX precisions; the paired accuracy interval is
  [-2, +2] percentage points, so identical scores do not prove equivalence.
- Kept conversion settings fixed and evaluated all 672 held-out validation rows.
  PyTorch and ONNX FP32 accuracy are 0.909226; INT8 accuracy is 0.901786.
  INT8 loses 0.744 percentage points. Seventeen predictions change. The paired
  bootstrap 95% interval is approximately [-1.935, +0.446] percentage points.
  The point-estimate one-point gate passes, but the interval does not establish
  equivalence within that margin. No tuning was performed on final results.
- Inspected performance source lengths before timing: development examples
  top out at 54 tokens (146 at <=32 and 54 at 33–54). No natural long bucket
  exists. Added an explicitly synthetic repeated-development-text stress case
  for length 256, with repeat counts recorded; it is excluded from quality.
- Started three repetitions per variant in isolated processes with rotated
  variant order, 200 iterations per scenario, and 20 warmup calls. No other
  model experiment is being run alongside these timing processes.
- Added a four-minute walkthrough and expanded reproduction instructions.

## September 30, 2026 — measurement audits and hardware metadata

- First repetition completed for all three variants, with 1,200 timing records
  per process. The second ONNX FP32 repetition is complete. Timing variation,
  particularly p95, is visible; conclusions will retain that limitation.
- Recorded Intel Core Ultra 9 185H, 16 physical/22 logical cores, roughly
  31.4 GiB RAM, Windows 11, Balanced power mode, and connected AC power.
- WMI CPU-name lookup yielded no name inside the sandbox. Replaced it with
  a read-only registry lookup; later process manifests include the full name.
  Earlier manifests retain the original blank field plus processor identifier.
- Added a run auditor that checks positive durations, exact scenario/sample
  counts, and identical row IDs, repeat counts, token lengths, settings, and
  data hashes across all variants/repetitions.
- Added three auditor tests; all ten local unit tests pass.
- Report generation rejects mismatched quality/timing artifacts and includes
  startup, latency, throughput, memory, file size, and quality figures. It
  explicitly labels medians across repetitions rather than pooled percentiles.
- Added changed-prediction analysis: final INT8 harms 11 FP32-correct predictions
  and fixes 6 FP32 errors. Logit margins are recorded without publishing text.
- Prepared an offline verifier using Python socket audit hooks, plus blank-input,
  Unicode, truncation, dtype, and finite-output checks. It will run after timing
  finishes so extra model work cannot compete with measurements.
- Checked the SST-2 dataset card: license is listed as unknown. Dataset text and
  downloaded model weights remain local. Source links are in the reproduction guide.

## September 30, 2026 — final repetition and reproduction preparation

- Completed two repetitions for all variants and all three for INT8. Each
  completed process produced its manifest, six scenario summaries, and 1,200
  raw timing records. The run directory is intentionally incomplete until the
  final PyTorch and ONNX FP32 processes finish; no full report is claimed yet.
- Prepared a clean committed-checkout verification script with a new locked
  environment. It reuses immutable local artifact hardlinks and checks unit
  tests, blocked-network inference, and FP32 parity. It does not repeat device
  timing or fresh artifact downloads; those remain separately reproducible commands.
- Added experiment-schema and setting validation, UTF-8 BOM tolerance for
  PowerShell-written JSON, and a guard against changing frozen split IDs before
  overwriting local data.

## September 30, 2026 — complete device study and offline verification

- All nine timing processes completed: three repetitions for each variant,
  six scenarios per process, 200 timed calls per scenario, 10,800 total records.
  No scenario failed. The full-run auditor passed exact sample counts and
  matching inputs/settings/data across all variants and repetitions.
- Generated and visually inspected latency and resource/quality figures.
  Added p50/p95 repetition ranges, explicit quality uncertainty, startup timing,
  changed-prediction analysis, and links to raw records in the report.
- Median warm inference p50 speedup for ONNX INT8 versus ONNX FP32 is 2.61x
  at 32 tokens, 2.91x at 128, and 2.43x at the synthetic 256-token workload.
- ONNX weight size decreases from 255.53 MiB to 132.43 MiB (48.2%). Sampled
  peak process RSS falls by 33.1% versus ONNX FP32 but remains above PyTorch's.
- Recorded a runtime regression: ONNX FP32 has worse median p95 than PyTorch
  at 128 tokens in this run, despite improved median latency. Tail variation
  limits conclusions; the report does not hide this outcome.
- Explained why separately timed tokenization-inclusive runs can appear faster
  due to variation. Subtracting their summaries cannot isolate tokenizer cost.
- All three runners passed Python-network-blocked inference with blank rejection,
  Unicode batches, truncation, int64 tensors, finite logits, and a positive fixture.
  This is not an OS firewall test; the exact method is saved in `results/offline.json`.
- All ten local unit tests passed after the report changes. Added a glossary so
  technical terms and limitations are easier to read.

## September 30, 2026 — fresh-checkout ownership debug

- Published the complete report, charts, audited raw records, and offline evidence.
- The first clean-checkout attempt failed before dependency installation: Git's
  local clone checks ownership of the sandbox-owned source `.git` directory
  separately from the trusted working folder when running as the laptop account.
- Changed verification to clone the published public GitHub repository instead.
  This avoids another local trust exception and also verifies the published code.
  No inference code, model, split, or benchmark results were changed.

## September 30, 2026 — final verification and delivery

- Fresh locked dependency installation succeeded in a clean public GitHub clone
  of commit `43274dd41c892bebd3e53b0a4ec29596b9740305`. `pip check` passed.
- All ten tests, all three Python-network-blocked runner/input-policy checks,
  and all six FP32 parity shapes passed in that new environment. Saved the exact
  verification scope and commit in `results/reproduction.json`. It reused local
  hash-verified artifact hardlinks and did not repeat the full timing study or
  downloads; those limitations are explicit in the evidence and guide.
- Ran the actual PowerShell CLI demo with an INT8 positive-sentiment fixture;
  it returned a positive label, scores, variant, and the pinned model revision.
- Confirmed PyTorch and ONNX FP32 predicted labels match on all 672 held-out rows.
- Moved failed symbolic-inference diagnostic model/data files into ignored
  `artifacts/debug` so generated weights do not clutter the repository root.
- Verified GitHub's contributor listing contains only Kaaushikk. Commits use the
  configured user identity and contain no assistant co-author trailer.
- CI passed for the report and reproduction-fix commits. The final documentation
  and verification record will be pushed and its CI checked before handoff.

## Remaining scope

The single-laptop classifier study is complete. The user subsequently authorized
optional extensions; the work below records them. Broad thread tuning and phone
measurements remain outside the measured study. No recurring run is configured.

## Results and debugging status

M1–M7 are delivered for the single-device study, with the clean-environment
verification scope described above. The core report includes measured gains,
regressions, uncertainty, and limitations. No numerical equivalence claim,
mobile or energy claim is made in the core report. Accelerator results are
reported separately below.

## Optional extensions: implementation and debugging

- Installed pinned FastAPI/Uvicorn/HTTPX dependencies in the main environment,
  checked dependency compatibility and regenerated the lock file.
- Added a localhost service with one loaded model, readiness, limited requests,
  explicit variant selection, labels, artifact metadata and a busy response.
  Three API tests verify lifecycle, request boundaries and concurrent inference.
  A real HTTP check passed positive/negative labels, readiness, blank rejection,
  wrong-variant rejection and 200 measured roundtrips after 20 warmups.
- Calibrated static signed INT8 QDQ on 100 development examples. All-MatMul
  quantization lost 4.5 accuracy points. Excluding attention MatMuls was a single
  bounded sensitivity check and also lost 4.5 points. Archived the first graph
  locally and retained its development result. The second candidate lost 2.98
  points on the untouched final set and fails the quality gate. No further
  tuning or timing claim is made for this failed candidate.
- Detected Intel Arc graphics and Intel AI Boost NPU, with driver metadata.
  Created an isolated OpenVINO environment because it requires NumPy below 2.3;
  the core environment stays on its locked version. Saved a separate lock file.
- Prepared hash-verified fixed input tensors, preserving all 672 final examples
  and the original timing fixture identities. Explicit GPU/NPU selection
  forbids AUTO/HETERO fallback. Each device matched 90.92% final accuracy.
- Debugged OpenVINO profiling: the property key is PERF_COUNT, not
  ENABLE_PROFILING. Also normalized EXECUTION_DEVICES because the NPU returns
  a string whereas the GPU returns a sequence. The first check accidentally
  split the NPU string into letters and correctly rejected it; normalization
  fixed the check. Profiling is disabled during timed calls.
- The OpenVINO CPU process exceeded the 300-second limit. Inspection afterward
  found a saved 90.92% quality result, zero changed labels, and 0.83-second
  compilation. Thus compilation succeeded; the timeout applies to overall
  process completion, with its exact cause unresolved. Recorded the timeout
  and used an explicit ONNX Runtime FP32 CPU control for timings instead.
- OpenVINO telemetry could not create its state directory under the sandbox
  and reported that no data would be sent. Worker Python socket operations are
  blocked; this is an application-level guard, not a system firewall guarantee.
- Checked power-meter and battery interfaces: zero readings on AC lack usable
  measurement metadata. Added a tested importer for external measured watts;
  no energy numbers are fabricated. No mobile-device measurement is available.

## Extension results and final checks

- Completed three fresh-process timing repetitions each for GPU, NPU and the
  ONNX Runtime FP32 CPU control. Audited all 5,400 records for counts, iteration
  coverage, fixture identities, graph/input hashes and explicit device use.
- GPU and NPU each changed zero predicted labels versus the same-tensor FP32
  reference. Their logits differ slightly; this is prediction agreement, not
  numerical equivalence. NPU p50 speedups were 11.02x, 16.42x and 16.44x at
  lengths 32, 128 and 256. Runtime, precision and device change together;
  CPU controls ran afterward and thermal/scheduling effects remain limitations.
- Generated the separate extension report and commands. Kept the core report
  and original measurement files intact. Downloaded graphs, input tensors and
  environments remain ignored; code, manifests and raw measured results are
  versioned.
- All 15 unit tests passed, including API concurrency and power integration.
  The synthetic output check passed, dependency checking found no conflicts,
  and the live model HTTP check passed.
- Pushed implementation and measured results as `fc85e9b`. GitHub correctness
  checks passed in run `36805168148` (15 tests and synthetic pipeline). Verified
  the working tree was clean and GitHub lists Kaaushikk as the sole contributor.
  This final documentation update records that verification.
- Available-hardware extensions are delivered. Actual phone and calibrated
  energy experiments remain hardware-dependent; no such result is claimed.

## Live demo requested by the user

- Ran the dynamic INT8 CLI on a positive movie review: POSITIVE, with a
  positive-class score of 0.99987.
- Tried the localhost API on port 8000. Windows rejected the bind with
  WinError 10013. Selected an available port, 58228, and the service started
  successfully without changing the model or requiring elevated access.
- Verified readiness and sent three reviews together through HTTP: an
  enthusiastic review returned POSITIVE, a boring/waste-of-time review returned
  NEGATIVE, and a mixed review praising the ending returned POSITIVE. These
  examples demonstrate execution, not a new accuracy study. Scores are model
  outputs, not calibrated probabilities of correctness.
- Requested the interactive API documentation in the app at
  `http://127.0.0.1:58228/docs`. Left the service running for user interaction.
  The raw demo response is saved locally in ignored `artifacts/live-demo.json`;
  historical benchmark results remain unchanged. This port is session-specific.
