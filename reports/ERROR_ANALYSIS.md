# Changed-prediction analysis

Final subset only. No raw dataset text is published.

| Row ID | True label | FP32 prediction | INT8 prediction | FP32 absolute logit margin | INT8 absolute logit margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 862 | 0 | 0 | 1 | 1.2557 | 2.9441 |
| 871 | 1 | 0 | 1 | 1.2553 | 0.0023 |
| 643 | 1 | 1 | 0 | 0.3728 | 2.2495 |
| 20 | 0 | 0 | 1 | 3.7644 | 2.1421 |
| 135 | 0 | 1 | 0 | 2.4459 | 0.7241 |
| 395 | 0 | 1 | 0 | 1.2718 | 0.4010 |
| 697 | 1 | 1 | 0 | 1.4758 | 1.8128 |
| 813 | 0 | 0 | 1 | 0.5086 | 2.1030 |
| 527 | 0 | 0 | 1 | 0.1417 | 0.5023 |
| 172 | 1 | 1 | 0 | 1.0676 | 3.6410 |
| 448 | 1 | 1 | 0 | 1.1739 | 1.3258 |
| 249 | 1 | 1 | 0 | 1.7329 | 1.3436 |
| 301 | 0 | 0 | 1 | 0.0642 | 0.0850 |
| 683 | 1 | 0 | 1 | 0.8060 | 0.3271 |
| 770 | 0 | 1 | 0 | 1.7545 | 0.5772 |
| 145 | 0 | 1 | 0 | 1.7839 | 0.1085 |
| 178 | 0 | 0 | 1 | 0.6897 | 2.8440 |

INT8 turns 11 FP32-correct predictions into errors and fixes 6 FP32 errors; 55 shared errors remain.
These are paired outcomes, not evidence that quantization causes a particular semantic bias. Small logit margins indicate boundary sensitivity; scores are not calibrated probabilities.
