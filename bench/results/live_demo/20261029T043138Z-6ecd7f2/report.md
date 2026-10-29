# live_demo

Run `20261029T043138Z-6ecd7f2`.

**Labels:** latency is measured through RoutePilot on this machine (the decision log), never simulated; prices are an ASSUMPTION (Ruling V1-R42); the policy chooses on its own predictions; the tight SLO is the fastest model's measured p50 end-to-end time and the loose SLO the slowest model's measured p95, both over the train and calibration requests; the quality floors are 0, the median and the 90th percentile of the best lower bound over the calibration items (rounded down to 3 decimals), the same floors the live replay used; the gateway's online latency estimator starts from the measured profiles and learns from the demo's own traffic, so later rounds are not independent of earlier ones; realized success is graded against the release's ground truth; a round has a few dozen items, so its intervals are wide; no prompt text and no model output is in this report.

## Setup

| setting | value |
| --- | --- |
| models | qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b |
| artifact version | 21183f8ab3fe |
| protocol | answer-only-user |
| system message | none |
| max_tokens | 16 |
| user message suffix | "\n\nReply with only the letter of the correct option." |
| min_quality floors | 0, 0.166, 0.221 (from 150 calibration items) |
| policy config | {"mode": "min_cost", "default_floor": 0, "epsilon": 0, "ood_action": "conservative", "on_infeasible": "best_effort"} |
| tight SLO (ms) | 58 (p50 of qwen2.5-0.5b) |
| loose SLO (ms) | 258 (p95 of qwen2.5-3b) |
| items per round / test items available | 120 / 120 |
| requests | 1080 |
| served model differs from the policy's pick (failover) | 0 |
| decision records missing | 0 |
| wall time (s) | 106.2 |

## Rounds (min_quality x slo_e2e_ms), measured through RoutePilot

| round | min_quality | slo_e2e_ms | n (failed) | model mix | relaxed any / latency / quality | success [Wilson 95%] | success when not relaxed (n) | TTFT ms p50 | e2e ms p50 / p95 | SLO met (share of requests) | p95 within SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q0-none | 0 | none | 120 (0) | {"qwen2.5-0.5b": 120} | 0 / 0 / 0 | 0.067 [0.034, 0.126] | 0.067 (120) | 35.5 | 58 / 132 | no SLO | no SLO |
| q0-tight | 0 | 58 | 120 (0) | {"qwen2.5-0.5b": 120} | 120 / 120 / 0 | 0.058 [0.029, 0.116] | n/a | 29 | 50 / 126 | 74.2% | False |
| q0-loose | 0 | 258 | 120 (0) | {"qwen2.5-0.5b": 120} | 0 / 0 / 0 | 0.067 [0.034, 0.126] | 0.067 (120) | 29 | 50 / 124 | 100.0% | True |
| q0.166-none | 0.166 | none | 120 (0) | {"qwen2.5-0.5b": 42, "qwen2.5-1.5b": 46, "qwen2.5-3b": 32} | 60 / 0 / 60 | 0.150 [0.097, 0.225] | 0.217 (60) | 66 | 98 / 189 | no SLO | no SLO |
| q0.166-tight | 0.166 | 58 | 120 (0) | {"qwen2.5-0.5b": 42, "qwen2.5-1.5b": 46, "qwen2.5-3b": 32} | 120 / 120 / 60 | 0.158 [0.104, 0.234] | n/a | 36 | 62 / 155 | 44.2% | False |
| q0.166-loose | 0.166 | 258 | 120 (0) | {"qwen2.5-0.5b": 42, "qwen2.5-1.5b": 46, "qwen2.5-3b": 32} | 75 / 75 / 60 | 0.158 [0.104, 0.234] | 0.244 (45) | 33 | 54 / 154 | 100.0% | True |
| q0.221-none | 0.221 | none | 120 (0) | {"qwen2.5-0.5b": 42, "qwen2.5-1.5b": 30, "qwen2.5-3b": 48} | 99 / 0 / 99 | 0.142 [0.090, 0.215] | 0.095 (21) | 34 | 66 / 172 | no SLO | no SLO |
| q0.221-tight | 0.221 | 58 | 120 (0) | {"qwen2.5-0.5b": 42, "qwen2.5-1.5b": 30, "qwen2.5-3b": 48} | 120 / 120 / 99 | 0.142 [0.090, 0.215] | n/a | 35.5 | 62 / 127 | 42.5% | False |
| q0.221-loose | 0.221 | 258 | 120 (0) | {"qwen2.5-0.5b": 42, "qwen2.5-1.5b": 30, "qwen2.5-3b": 48} | 118 / 118 / 99 | 0.142 [0.090, 0.215] | 0.000 (2) | 36 | 61 / 123 | 100.0% | True |

## Median end-to-end bound the policy used (q_ms), per model and round

| round | qwen2.5-0.5b | qwen2.5-1.5b | qwen2.5-3b |
| --- | --- | --- | --- |
| q0-tight | 162.205 | 199.36 | 223.802 |
| q0-loose | 158.473 | 199.36 | 223.802 |
| q0.166-tight | 155.419 | 205.846 | 315.778 |
| q0.166-loose | 155.419 | 204.46 | 342.364 |
| q0.221-tight | 149.303 | 193.095 | 307.79 |
| q0.221-loose | 149.303 | 191.877 | 290.701 |
