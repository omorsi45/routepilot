# live_collect

Run `20261029T015906Z-00deb40`.

**Labels:** latency and token counts are measured on this machine through RoutePilot (decision log); prices are an ASSUMPTION (Ruling V1-R42): proportional to the parameter count, 0.02 USD per million input tokens and 0.06 per million output tokens per billion parameters; answers are graded by the last standalone option letter after an answer cue, else the last standalone option letter, against the ground truth of the release; no prompt text and no model output is in this report.

## Setup

| setting | value |
| --- | --- |
| models | qwen2.5-0.5b (qwen2.5:0.5b), qwen2.5-1.5b (qwen2.5:1.5b), qwen2.5-3b (qwen2.5:3b) |
| measure run | 20261029T015819Z-00deb40 |
| time budget (minutes) | 90 |
| mmlupro items planned / used (cap) | 281 / 281 (600) |
| gpqa items | 198 |
| max_tokens | 384 |
| stopped by the time budget | False |
| request time (s) / wall time (s) | 4583.3 / 4618.6 |
| items every model answered | 478 |
| ... by dataset | {"gpqa": 198, "mmlupro": 280} |
| decision records missing | 0 |

## Per model (latency from the decision log, cost assumed)

| model | requests | failed | accuracy | hit max tokens | graded by cue / last letter / none | completion tokens mean | TTFT ms p50 / p95 | e2e ms p50 / p95 | mean cost USD (assumed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b | 479 | 1 | 0.10251 | 386 | 77 / 240 / 161 | 367.157 | 35 / 54 | 2168 / 2618 | 1.37657e-05 |
| qwen2.5-1.5b | 479 | 0 | 0.179541 | 276 | 199 / 157 / 123 | 333.388 | 57 / 90 | 2540 / 3808 | 3.82546e-05 |
| qwen2.5-3b | 479 | 0 | 0.192067 | 353 | 124 / 220 / 135 | 361.969 | 96 / 185 | 4089 / 8198 | 8.16537e-05 |

## Accuracy per model and dataset

| model | dataset | n | accuracy |
| --- | --- | --- | --- |
| qwen2.5-0.5b | gpqa | 198 | 0.0959596 |
| qwen2.5-0.5b | mmlupro | 280 | 0.107143 |
| qwen2.5-1.5b | gpqa | 198 | 0.161616 |
| qwen2.5-1.5b | mmlupro | 281 | 0.192171 |
| qwen2.5-3b | gpqa | 198 | 0.166667 |
| qwen2.5-3b | mmlupro | 281 | 0.209964 |
