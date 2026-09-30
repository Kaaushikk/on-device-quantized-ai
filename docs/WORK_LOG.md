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

## Next steps

Finish repeated timing, validate record counts and matching inputs/artifacts,
generate and inspect the report, verify the offline demo and clean-environment
setup, and publish the final evidence.

## Results and debugging status

M1–M4 are validated locally: locked foundation, PyTorch baseline, FP32 parity,
and executable INT8 with development quality comparison. Final quality is
recorded. Repeated timing and fresh-environment reproduction are in progress;
the final report has not been generated yet.
