# live_train

Run `20261029T034651Z-766a3b8`.

**Labels:** prices are an ASSUMPTION (Ruling V1-R42): proportional to the parameter count, 0.02 USD per million input tokens and 0.06 per million output tokens per billion parameters; latency is measured through RoutePilot on this machine (the decision log), never simulated or assumed; the learned rows' floors (0, 0.4, 0.6) are fixed in advance; the test split is replayed once per row and nothing is chosen on it; the test split is small, so the intervals are wide; no prompt text and no model output is in this report.

## Setup

| setting | value |
| --- | --- |
| models | qwen2.5-0.5b, qwen2.5-1.5b, qwen2.5-3b |
| collection protocol | answer-only |
| system message | Reply with only the letter of the correct option. |
| max_tokens | 16 |
| Best Single (train+calib) / cheapest | qwen2.5-3b / qwen2.5-0.5b |
| items every model answered | 798 |
| items per split | {"calib": 150, "test": 148, "train": 500} |
| artifact version / dim / bytes | 901e144b65b4 / 16384 / 504318 |
| calibrator | switch |
| OOD k / threshold | 32 / 0.550401 |
| policy config | {"mode": "min_cost", "default_floor": 0, "epsilon": 0, "ood_action": "conservative", "on_infeasible": "best_effort"} |
| latency profiles measured on | AMD Ryzen 9 7900X 12-Core Processor (24 threads) and AMD Radeon RX 9060 XT |
| wall time (s) | 6.7 |

## Assumed prices

| model | input USD per Mtok (ASSUMPTION) | output USD per Mtok (ASSUMPTION) | prompt overhead tokens |
| --- | --- | --- | --- |
| qwen2.5-0.5b | 0.01 | 0.03 | 0 |
| qwen2.5-1.5b | 0.03 | 0.09 | 0 |
| qwen2.5-3b | 0.06 | 0.18 | 0 |

## Per-model fit (calibration columns are in-sample for the calibrator)

| model | C | cv_log_loss | alpha | calibrator | calib_ece | calib_brier | calib_success_rate | lcb_bins | length_smear | in_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b | 1 | 0.354386 | 0.1 | platt | 0.0850261 | 0.084054 | 0.0933333 | 5 | 1.09682 | 0.308316 |
| qwen2.5-1.5b | 0.1 | 0.450188 | 0.1 | platt | 0.0744439 | 0.120709 | 0.16 | 5 | 1.03208 | 0.308316 |
| qwen2.5-3b | 1 | 0.48886 | 0.1 | platt | 0.0746034 | 0.156355 | 0.22 | 5 | 1.11564 | 0.308316 |

## Measured latency profiles (train and calibration requests)

| model | train+calib requests | TTFT median ms | TTFT sigma | TPOT median ms | TPOT sigma |
| --- | --- | --- | --- | --- | --- |
| qwen2.5-0.5b | 650 | 34 | 0.219828 | 5 | 0.0963283 |
| qwen2.5-1.5b | 650 | 61 | 0.287687 | 7.0625 | 0.152144 |
| qwen2.5-3b | 650 | 77 | 0.305956 | 9.25 | 0.157287 |

## Live test split: offline replay on recorded outcomes, measured latency

| policy | n | success [95% CI] | mean cost USD (assumed) [95% CI] | e2e ms p50 (measured) | e2e ms p95 (measured) | relaxed share | OOD share | model mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_strongest | 148 | 0.1351 [0.0811, 0.1959] | 1.83093e-05 [1.71028e-05, 1.94715e-05] | 166.5 | 267.15 | 0 | 0 | {"qwen2.5-3b": 148} |
| always_cheapest | 148 | 0.0811 [0.0405, 0.1284] | 2.98628e-06 [2.78695e-06, 3.18239e-06] | 62 | 135.65 | 0 | 0 | {"qwen2.5-0.5b": 148} |
| random | 148 | 0.1081 [0.0608, 0.1554] | 7.97311e-06 [7.02924e-06, 8.96365e-06] | 107 | 226.65 | 0 | 0 | {"qwen2.5-0.5b": 65, "qwen2.5-1.5b": 54, "qwen2.5-3b": 29} |
| static_rules | 148 | 0.1351 [0.0811, 0.1959] | 1.83093e-05 [1.71028e-05, 1.94715e-05] | 166.5 | 267.15 | 0 | 0 | {"qwen2.5-3b": 148} |
| oracle | 148 | 0.2838 [0.2095, 0.3581] | 4.41264e-06 [3.8749e-06, 4.99617e-06] | 71 | 135.65 | 0 | 0 | {"qwen2.5-0.5b": 118, "qwen2.5-1.5b": 23, "qwen2.5-3b": 7} |
| static_strongest_rank | 148 | 0.1351 [0.0811, 0.1959] | 1.83093e-05 [1.71028e-05, 1.94715e-05] | 166.5 | 267.15 | 0 | 0 | {"qwen2.5-3b": 148} |
| static_cheapest_worstcase | 148 | 0.0811 [0.0405, 0.1284] | 2.98628e-06 [2.78695e-06, 3.18239e-06] | 62 | 135.65 | 0 | 0 | {"qwen2.5-0.5b": 148} |
| learned_min_cost_floor_0 | 148 | 0.0811 [0.0405, 0.1284] | 2.98628e-06 [2.78695e-06, 3.18239e-06] | 62 | 135.65 | 0 | 0.0202703 | {"qwen2.5-0.5b": 148} |
| learned_min_cost_floor_0.4 | 148 | 0.1081 [0.0608, 0.1689] | 1.55057e-05 [1.38817e-05, 1.70966e-05] | 144.5 | 268.55 | 1 | 0.0202703 | {"qwen2.5-0.5b": 23, "qwen2.5-1.5b": 20, "qwen2.5-3b": 105} |
| learned_min_cost_floor_0.6 | 148 | 0.1081 [0.0608, 0.1689] | 1.55057e-05 [1.38817e-05, 1.70966e-05] | 144.5 | 268.55 | 1 | 0.0202703 | {"qwen2.5-0.5b": 23, "qwen2.5-1.5b": 20, "qwen2.5-3b": 105} |
