# live_collect

Run `20261029T033743Z-766a3b8`.

**Labels:** latency and token counts are measured on this machine through RoutePilot (decision log); prices are an ASSUMPTION (Ruling V1-R42): proportional to the parameter count, 0.02 USD per million input tokens and 0.06 per million output tokens per billion parameters; answers are graded by the last standalone option letter after an answer cue, else the last standalone option letter, against the ground truth of the release; no prompt text and no model output is in this report.

## Setup

| setting | value |
| --- | --- |
| models | qwen2.5-0.5b (qwen2.5:0.5b), qwen2.5-1.5b (qwen2.5:1.5b), qwen2.5-3b (qwen2.5:3b) |
| measure run | 20261029T015819Z-00deb40 |
| protocol | answer-only |
| system message | Reply with only the letter of the correct option. |
| max_tokens | 16 |
| time budget (minutes) | 30 |
| mmlupro items planned / used (cap) | 600 / 600 (600) |
| gpqa items | 198 |
| stopped by the time budget | False |
| request time (s) / wall time (s) | 329.9 / 351.9 |
| items every model answered | 798 |
| ... by dataset | {"gpqa": 198, "mmlupro": 600} |
| decision records missing | 0 |

## Per model (latency from the decision log, cost assumed)

| model | requests | failed | accuracy | hit max tokens | graded by cue / last letter / none | completion tokens mean | TTFT ms p50 / p95 | e2e ms p50 / p95 | mean cost USD (assumed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b | 798 | 0 | 0.106516 | 223 | 541 / 51 / 206 | 7.42231 | 34 / 51 | 60 / 135 | 2.91823e-06 |
| qwen2.5-1.5b | 798 | 0 | 0.166667 | 202 | 577 / 19 / 202 | 7.07268 | 61 / 100 | 100 / 221 | 8.72323e-06 |
| qwen2.5-3b | 798 | 0 | 0.196742 | 336 | 413 / 90 / 295 | 9.43734 | 77 / 141 | 166 / 277 | 1.78721e-05 |

## Accuracy per model and dataset

| model | dataset | n | accuracy |
| --- | --- | --- | --- |
| qwen2.5-0.5b | gpqa | 198 | 0.030303 |
| qwen2.5-0.5b | mmlupro | 600 | 0.131667 |
| qwen2.5-1.5b | gpqa | 198 | 0.0858586 |
| qwen2.5-1.5b | mmlupro | 600 | 0.193333 |
| qwen2.5-3b | gpqa | 198 | 0.0454545 |
| qwen2.5-3b | mmlupro | 600 | 0.246667 |
