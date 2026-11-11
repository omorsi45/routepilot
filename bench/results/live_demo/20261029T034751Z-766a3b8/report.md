# live_demo

Run `20261029T034751Z-766a3b8`.

**Labels:** latency is measured through RoutePilot on this machine (the decision log), never simulated; prices are an ASSUMPTION (Ruling V1-R42); the policy chooses on its own predictions; the tight SLO is the fastest model's measured p50 end-to-end time and the loose SLO the slowest model's measured p95, both over the train and calibration requests; the gateway's online latency estimator starts from the measured profiles and learns from the demo's own traffic, so later rounds are not independent of earlier ones; realized success is graded against the release's ground truth; a round has a few dozen items, so its intervals are wide; no prompt text and no model output is in this report.

## Setup

| setting | value |
| --- | --- |
| models | qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b |
| artifact version | 901e144b65b4 |
| protocol | answer-only |
| system message | Reply with only the letter of the correct option. |
| max_tokens | 16 |
| policy config | {"mode": "min_cost", "default_floor": 0, "epsilon": 0, "ood_action": "conservative", "on_infeasible": "best_effort"} |
| tight SLO (ms) | 59 (p50 of qwen2.5-0.5b) |
| loose SLO (ms) | 279 (p95 of qwen2.5-3b) |
| items per round / test items available | 120 / 120 |
| requests | 1080 |
| served model differs from the policy's pick (failover) | 0 |
| decision records missing | 0 |
| wall time (s) | 163.7 |

## Rounds (min_quality x slo_e2e_ms), measured through RoutePilot

| round | min_quality | slo_e2e_ms | n (failed) | model mix | relaxed any / latency / quality | success [Wilson 95%] | success when not relaxed (n) | TTFT ms p50 | e2e ms p50 / p95 | SLO met (share of requests) | p95 within SLO |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q0-none | 0 | none | 120 (0) | {"qwen2.5-0.5b": 120} | 0 / 0 / 0 | 0.083 [0.046, 0.147] | 0.083 (120) | 34 | 64 / 133 | no SLO | no SLO |
| q0-tight | 0 | 59 | 120 (0) | {"qwen2.5-0.5b": 120} | 120 / 120 / 0 | 0.100 [0.058, 0.167] | n/a | 27 | 52 / 124 | 67.5% | False |
| q0-loose | 0 | 279 | 120 (0) | {"qwen2.5-0.5b": 120} | 0 / 0 / 0 | 0.100 [0.058, 0.167] | 0.100 (120) | 28 | 51 / 123 | 100.0% | True |
| q0.4-none | 0.4 | none | 120 (0) | {"qwen2.5-0.5b": 11, "qwen2.5-1.5b": 15, "qwen2.5-3b": 94} | 120 / 0 / 120 | 0.117 [0.071, 0.186] | n/a | 83.5 | 158 / 303 | no SLO | no SLO |
| q0.4-tight | 0.4 | 59 | 120 (0) | {"qwen2.5-0.5b": 11, "qwen2.5-1.5b": 15, "qwen2.5-3b": 94} | 120 / 120 / 120 | 0.117 [0.071, 0.186] | n/a | 50 | 125 / 299 | 5.0% | False |
| q0.4-loose | 0.4 | 279 | 120 (0) | {"qwen2.5-0.5b": 11, "qwen2.5-1.5b": 15, "qwen2.5-3b": 94} | 120 / 120 / 120 | 0.117 [0.071, 0.186] | n/a | 47 | 124 / 298 | 90.0% | False |
| q0.6-none | 0.6 | none | 120 (0) | {"qwen2.5-0.5b": 11, "qwen2.5-1.5b": 15, "qwen2.5-3b": 94} | 120 / 0 / 120 | 0.117 [0.071, 0.186] | n/a | 45 | 120 / 297 | no SLO | no SLO |
| q0.6-tight | 0.6 | 59 | 120 (0) | {"qwen2.5-0.5b": 11, "qwen2.5-1.5b": 15, "qwen2.5-3b": 94} | 120 / 120 / 120 | 0.117 [0.071, 0.186] | n/a | 48 | 123 / 295 | 5.8% | False |
| q0.6-loose | 0.6 | 279 | 120 (0) | {"qwen2.5-0.5b": 11, "qwen2.5-1.5b": 15, "qwen2.5-3b": 94} | 120 / 120 / 120 | 0.117 [0.071, 0.186] | n/a | 46 | 123 / 296 | 92.5% | False |

## Median end-to-end bound the policy used (q_ms), per model and round

| round | qwen2.5-0.5b | qwen2.5-1.5b | qwen2.5-3b |
| --- | --- | --- | --- |
| q0-tight | 157.187 | 268.718 | 354.493 |
| q0-loose | 157.187 | 268.718 | 354.493 |
| q0.4-tight | 157.187 | 268.718 | 425.532 |
| q0.4-loose | 157.187 | 268.718 | 403.834 |
| q0.6-tight | 157.187 | 230.795 | 367.26 |
| q0.6-loose | 157.187 | 230.795 | 367.26 |
