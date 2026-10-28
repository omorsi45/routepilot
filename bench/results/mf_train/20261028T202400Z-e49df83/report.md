# mf_train

Run `20261028T202400Z-e49df83`.

**Labels:** trained on the train split only: no calibration, test or held-out row was read; no price and no latency is used or simulated here; the exported weights are derived data and are not committed; only their SHA-256 is recorded.

## Setup

| key | value |
| --- | --- |
| pool | mixed |
| strong (Best Single on train+calib) | gpt-5 |
| weak (cheapest on train+calib) | Llama-3.1-Nemotron-Nano-8B-v1 |
| hashed dimension | 65536 |
| train items with a pair / pairs | 814 / 35157 |
| hyperparameters | {"dim_h": 128, "epochs": 10, "lr": 0.001, "l2": 1e-05, "seed": 0, "batch_items": 256} |
| training wall time (s) | 3.2 |
| total wall time with features (s) | 4 |
| weights SHA-256 | d3fba3d30ed140c8af2e92afe4de18a097507ddf7cecbd8b0d52837f9c8ec5e5 |

## Epochs (loss: mean pairwise loss on the train items)

| epoch | loss | seconds |
| --- | --- | --- |
| 1 | 0.691875 | 0.290889 |
| 2 | 0.684454 | 0.305033 |
| 3 | 0.676267 | 0.312051 |
| 4 | 0.666659 | 0.315285 |
| 5 | 0.655186 | 0.306128 |
| 6 | 0.641861 | 0.308357 |
| 7 | 0.626317 | 0.30994 |
| 8 | 0.608683 | 0.301331 |
| 9 | 0.589037 | 0.313938 |
| 10 | 0.567763 | 0.322563 |
