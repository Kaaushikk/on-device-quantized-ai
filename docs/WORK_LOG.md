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

## Next steps

Create a tested synthetic harness and project-local environment, pin compatible
dependencies, download allowed model/data snapshots, implement real runners,
and validate conversion before collecting any device measurements.

## Results and debugging status

No real model inference results exist yet. GitHub authentication is resolved.
No export or quantization failures have occurred because those stages have
not run. Dependency locking and real runners remain unfinished; milestone M1
is not complete yet.
