# live_collect

Run `20261029T042250Z-6ecd7f2`.

**Labels:** latency and token counts are measured on this machine through RoutePilot (decision log); prices are an ASSUMPTION (Ruling V1-R42): proportional to the parameter count, 0.02 USD per million input tokens and 0.06 per million output tokens per billion parameters; answers are graded by the last standalone option letter after an answer cue, else the last standalone option letter, against the ground truth of the release; no prompt text and no model output is in this report.

## Setup

| setting | value |
| --- | --- |
| models | qwen2.5-0.5b (qwen2.5:0.5b), qwen2.5-1.5b (qwen2.5:1.5b), qwen2.5-3b (qwen2.5:3b) |
| measure run | 20261029T015819Z-00deb40 |
| protocol | answer-only-user |
| system message | none |
| max_tokens | 16 |
| user message suffix | "\n\nReply with only the letter of the correct option." |
| time budget (minutes) | 15 |
| mmlupro items planned / used (cap) | 600 / 600 (600) |
| gpqa items | 198 |
| stopped by the time budget | False |
| request time (s) / wall time (s) | 288.5 / 309.0 |
| items every model answered | 798 |
| ... by dataset | {"gpqa": 198, "mmlupro": 600} |
| decision records missing | 0 |

## Per model (latency from the decision log, cost assumed)

| model | requests | failed | accuracy | hit max tokens | graded by cue / last letter / none | completion tokens mean | TTFT ms p50 / p95 | e2e ms p50 / p95 | mean cost USD (assumed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b | 798 | 0 | 0.122807 | 191 | 498 / 136 / 164 | 6.72556 | 35 / 52 | 58 / 133 | 3.05936e-06 |
| qwen2.5-1.5b | 798 | 0 | 0.184211 | 140 | 122 / 550 / 126 | 4.98997 | 64 / 103 | 87 / 219 | 9.02188e-06 |
| qwen2.5-3b | 798 | 0 | 0.22807 | 144 | 99 / 561 / 138 | 5.09273 | 81 / 146 | 114 / 256 | 1.80623e-05 |

## Accuracy per model and dataset

| model | dataset | n | accuracy |
| --- | --- | --- | --- |
| qwen2.5-0.5b | gpqa | 198 | 0.040404 |
| qwen2.5-0.5b | mmlupro | 600 | 0.15 |
| qwen2.5-1.5b | gpqa | 198 | 0.106061 |
| qwen2.5-1.5b | mmlupro | 600 | 0.21 |
| qwen2.5-3b | gpqa | 198 | 0.0909091 |
| qwen2.5-3b | mmlupro | 600 | 0.273333 |
