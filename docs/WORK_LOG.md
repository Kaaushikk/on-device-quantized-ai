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

## Next steps

Complete source downloads, run the real baseline/export/parity/quantization,
inspect development quality, then freeze decisions and collect final quality
and repeated device measurements. Publish the evidence-backed report and demo.

## Results and debugging status

No real model inference results exist yet. GitHub authentication is resolved.
No export or quantization failures have occurred because those stages have
not run. The locked environment and tested foundation are now in place (M1).
Real pipeline correctness and device measurements remain unverified.
