# mf_train

Run `20261028T033419Z-5e4dbcc`.

**Labels:** trained on the train split only: no calibration, test or held-out row was read; no price and no latency is used or simulated here; the exported weights are derived data and are not committed; only their SHA-256 is recorded.

## Setup

| key | value |
| --- | --- |
| pool | flagship |
| strong (Best Single on train+calib) | gemini-2.5-pro |
| weak (cheapest on train+calib) | deepseek-v3-0324 |
| hashed dimension | 65536 |
| train items with a pair / pairs | 3616 / 87375 |
| hyperparameters | {"dim_h": 128, "epochs": 10, "lr": 0.001, "l2": 1e-05, "seed": 0, "batch_items": 256} |
| training wall time (s) | 10.6 |
| total wall time with features (s) | 12.7 |
| weights SHA-256 | 8216c85e5965401a4735493d04f70863982fc5475a207bc851cd399eb466accc |

## Epochs (loss: mean pairwise loss on the train items)

| epoch | loss | seconds |
| --- | --- | --- |
| 1 | 0.688384 | 1.02257 |
| 2 | 0.668679 | 1.03739 |
| 3 | 0.634182 | 1.02557 |
| 4 | 0.584719 | 1.0553 |
| 5 | 0.536795 | 1.03077 |
| 6 | 0.505209 | 1.10512 |
| 7 | 0.487708 | 1.09784 |
| 8 | 0.474665 | 1.037 |
| 9 | 0.46335 | 1.04512 |
| 10 | 0.452111 | 1.04383 |
