# Terms used in the project

| Term | Plain-language meaning |
| --- | --- |
| PyTorch | The original framework used to load and run the pretrained model. |
| ONNX | A portable model graph format; ONNX Runtime executes that graph locally. |
| FP32 | Numbers stored using 32-bit floating-point precision. |
| INT8 | Selected model calculations use 8-bit integers. Other parts can still use FP32. |
| Dynamic quantization | Some conversion scales are calculated during each prediction. Smaller numbers can reduce work, but this calculation adds overhead. |
| Token | A piece of text understood by the tokenizer; it is not always a full word. |
| Padding | Extra positions added to make input tensors the same size. It can add computation even when the actual text is short. |
| Logit | A raw model score before conversion into probabilities. |
| Parity | Checking that two implementations give sufficiently similar outputs on identical inputs. |
| Warmup | Predictions performed before timing to reduce startup effects. |
| p50 latency | The median request duration: a typical observed time. |
| p95 latency | A high percentile that helps describe slow requests. It can vary on a busy laptop. |
| Throughput | How many examples run per second of timed work. Batch latency and per-example latency are different quantities. |
| Accuracy | The fraction of examples with the correct predicted label. |
| Macro F1 | A quality score that gives both labels equal weight and considers missed predictions and incorrect positive predictions. |
| RSS | Memory currently resident for the entire process, including Python and the runtime. It is not just model memory. |
| MiB | 1,048,576 bytes. |
| Revision | An exact source commit, used to avoid silently changing model or dataset versions. |
| Hash | A content fingerprint used to detect missing or modified files. |
| Paired bootstrap | Resampling the same evaluated examples for both variants to estimate uncertainty in their quality difference. |
| Held-out validation subset | Rows kept out of this project's development decisions; these are not hidden public test labels. |

A useful result can include a slowdown or an accuracy regression. The project
records those outcomes instead of requiring quantization to win every metric.
