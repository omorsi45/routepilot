# live_train

Run `20261029T043015Z-6ecd7f2`.

**Labels:** prices are an ASSUMPTION (Ruling V1-R42): proportional to the parameter count, 0.02 USD per million input tokens and 0.06 per million output tokens per billion parameters; latency is measured through RoutePilot on this machine (the decision log), never simulated or assumed; the learned rows' floors are 0, the median and the 90th percentile of the best lower bound over the calibration items (chosen from the trained artifact before the test split is replayed); the test split is replayed once per row and nothing is chosen on it; the test split is small, so the intervals are wide; no prompt text and no model output is in this report.

## Setup

| setting | value |
| --- | --- |
| models | qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b |
| collection protocol | answer-only-user |
| system message | none |
| max_tokens | 16 |
| user message suffix | "\n\nReply with only the letter of the correct option." |
| Best Single (train+calib) / cheapest | qwen2.5-3b / qwen2.5-0.5b |
| items every model answered | 798 |
| items per split | {"calib": 150, "test": 148, "train": 500} |
| artifact version / dim / bytes | 21183f8ab3fe / 16384 / 515941 |
| calibrator | switch |
| OOD k / threshold | 32 / 0.525308 |
| policy config | {"mode": "min_cost", "default_floor": 0, "epsilon": 0, "ood_action": "conservative", "on_infeasible": "best_effort"} |
| min_quality floors | 0, 0.166, 0.221 (from 150 calibration items) |
| best lower bound over calibration items: min / median / 90th pct / max | 0.0681 / 0.1663 / 0.2211 / 0.2211 |
| latency profiles measured on | AMD Ryzen 9 7900X 12-Core Processor (24 threads) and AMD Radeon RX 9060 XT |
| wall time (s) | 13.6 |

## Assumed prices

| model | input USD per Mtok (ASSUMPTION) | output USD per Mtok (ASSUMPTION) | prompt overhead tokens |
| --- | --- | --- | --- |
| qwen2.5-0.5b | 0.01 | 0.03 | 0 |
| qwen2.5-1.5b | 0.03 | 0.09 | 0 |
| qwen2.5-3b | 0.06 | 0.18 | 0 |

## Per-model fit (calibration columns are in-sample for the calibrator)

| model | C | cv_log_loss | alpha | calibrator | calib_ece | calib_brier | calib_success_rate | lcb_bins | length_smear | in_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b | 1 | 0.382436 | 0.1 | platt | 0.0931517 | 0.115549 | 0.133333 | 5 | 1.04625 | 0.309145 |
| qwen2.5-1.5b | 0.1 | 0.477203 | 0.1 | platt | 0.0912008 | 0.140998 | 0.18 | 5 | 1.04188 | 0.309145 |
| qwen2.5-3b | 10 | 0.516553 | 0.1 | platt | 0.0765506 | 0.170149 | 0.233333 | 5 | 1.03302 | 0.309145 |

## Measured latency profiles (train and calibration requests)

| model | train+calib requests | TTFT median ms | TTFT sigma | TPOT median ms | TPOT sigma |
| --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b | 650 | 35 | 0.219859 | 5 | 0.174061 |
| qwen2.5-1.5b | 650 | 64 | 0.264745 | 5.5 | 0.254962 |
| qwen2.5-3b | 650 | 81 | 0.301759 | 6 | 0.277181 |

## Live test split: offline replay on recorded outcomes, measured latency

| policy | n | success [95% CI] | mean cost USD (assumed) [95% CI] | e2e ms p50 (measured) | e2e ms p95 (measured) | relaxed share | OOD share | model mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_strongest | 148 | 0.1892 [0.1284, 0.2500] | 1.83564e-05 [1.71477e-05, 1.95003e-05] | 115 | 249.65 | 0 | 0 | {"qwen2.5-3b": 148} |
| always_cheapest | 148 | 0.0878 [0.0405, 0.1351] | 3.11047e-06 [2.91026e-06, 3.30592e-06] | 57 | 133 | 0 | 0 | {"qwen2.5-0.5b": 148} |
| random | 148 | 0.1622 [0.1081, 0.2297] | 8.11818e-06 [7.16789e-06, 9.12085e-06] | 81.5 | 217.6 | 0 | 0 | {"qwen2.5-0.5b": 65, "qwen2.5-1.5b": 54, "qwen2.5-3b": 29} |
| static_rules | 148 | 0.1892 [0.1284, 0.2500] | 1.83564e-05 [1.71477e-05, 1.95003e-05] | 115 | 249.65 | 0 | 0 | {"qwen2.5-3b": 148} |
| oracle | 148 | 0.3108 [0.2432, 0.3851] | 5.10453e-06 [4.27985e-06, 6.13713e-06] | 65.5 | 134.65 | 0 | 0 | {"qwen2.5-0.5b": 115, "qwen2.5-1.5b": 24, "qwen2.5-3b": 9} |
| static_strongest_rank | 148 | 0.1892 [0.1284, 0.2500] | 1.83564e-05 [1.71477e-05, 1.95003e-05] | 115 | 249.65 | 0 | 0 | {"qwen2.5-3b": 148} |
| static_cheapest_worstcase | 148 | 0.0878 [0.0405, 0.1351] | 3.11047e-06 [2.91026e-06, 3.30592e-06] | 57 | 133 | 0 | 0 | {"qwen2.5-0.5b": 148} |
| learned_min_cost_floor_0 | 148 | 0.0878 [0.0405, 0.1351] | 3.11047e-06 [2.91026e-06, 3.30592e-06] | 57 | 133 | 0 | 0.0135135 | {"qwen2.5-0.5b": 148} |
| learned_min_cost_floor_0.166 | 148 | 0.1486 [0.0946, 0.2027] | 9.80764e-06 [8.61467e-06, 1.10775e-05] | 86 | 175.5 | 0.459459 | 0.0135135 | {"qwen2.5-0.5b": 47, "qwen2.5-1.5b": 61, "qwen2.5-3b": 40} |
| learned_min_cost_floor_0.221 | 148 | 0.1419 [0.0878, 0.1959] | 1.14538e-05 [1.00431e-05, 1.28794e-05] | 95 | 187.85 | 0.797297 | 0.0135135 | {"qwen2.5-0.5b": 47, "qwen2.5-1.5b": 37, "qwen2.5-3b": 64} |
