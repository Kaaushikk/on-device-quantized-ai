# Benchmark protocol — initial design

## What we compare

Use the same pinned DistilBERT SST-2 classifier and tokenizer for PyTorch
FP32, ONNX FP32, and dynamic ONNX INT8. Start with the CPU provider on this
laptop. Record actual hardware, Windows version, RAM, power settings,
package versions, artifact hashes, and model/data commit revisions.

## Fair inputs and quality

Freeze dataset row IDs before evaluating. Divide labeled SST-2 validation
into a development subset and a final held-out subset. Record the split
seed and IDs. Do not call this public test accuracy. Any tuning uses only
development inputs. Report accuracy, macro F1, confusion matrices, changed
predictions, and a paired uncertainty interval on the final subset.

Compare ONNX FP32 logits against PyTorch with initial tolerances atol=1e-4
and rtol=1e-3 across planned batch and sequence shapes. Investigate failures
before changing tolerances. The proposed INT8 quality gate is no more than
one percentage point of accuracy loss relative to ONNX FP32; this is a
chosen acceptance threshold, not a measured result.

## Performance

Start with batch size 1, lengths 32, 128, and 256, and one CPU thread.
Use identical prepared tensors for inference-only timing. Keep tokenization
plus inference as a separate measurement. Use warmup, at least 200 timed
iterations per scenario, and three repetitions. Rotate variant order and
use a fresh process for each variant. Record every planned scenario and
any failure. Extend warmup or repetitions when measurements are unstable.

Source inspection found no development examples longer than 54 tokens.
Short and medium scenarios use natural inputs in their length buckets.
The 256-token scenario repeats development text to fill the context and is
explicitly a synthetic long-text stress workload; it is never used for quality.
Raw records include repeat counts and source row IDs for reproduction.

Measure load time, first inference, warm p50/p95, examples per second,
weight bytes, idle process RSS, and sampled peak RSS separately. RSS includes
Python/runtime overhead and sampling may miss short peaks. Cached disk load
is not a true cold disk test. Do not claim energy or mobile performance.

## Output and publication

Each real run writes manifest.json, timings.jsonl, quality.json, and
summary.csv as applicable. Every chart must link to raw evidence. Quantization
speedup compares ONNX INT8 with ONNX FP32; runtime effects compare ONNX with
PyTorch. Verify licenses before publishing weights or dataset text. Keep raw
text out of logs. Synthetic fixtures must be labeled and never presented as
model benchmarks. No final result exists yet.
