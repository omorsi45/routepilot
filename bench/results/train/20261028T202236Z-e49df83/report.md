# train

Run `20261028T202236Z-e49df83`.

**Labels:** calib_ece and calib_brier are measured on the calibration split, which is in-sample for the calibrator and the lower-bound tables; no test or held-out row was read; prices are not used here; nothing in this run is simulated or assumed.

## Per-model fit (calib_ece and calib_brier: calibration split, in-sample for the calibrator)

| model | C | cv_log_loss | alpha | calibrator | calib_ece | calib_brier | calib_success_rate | lcb_bins | length_smear | in_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-0528-Qwen3-8B | 10 | 0.610619 | 1 | platt | 0.0818185 | 0.207478 | 0.531414 | 7 | 1.39699 | 0.274336 |
| Llama-3.1-Nemotron-Nano-8B-v1 | 10 | 0.479994 | 1 | platt | 0.0502754 | 0.148011 | 0.340314 | 7 | 1.67367 | 0.28363 |
| NVIDIA-Nemotron-Nano-9B-v2 | 10 | 0.560308 | 1 | platt | 0.0475639 | 0.19531 | 0.497382 | 7 | 1.65268 | 0.290469 |
| Qwen3-8B | 10 | 0.621937 | 0.1 | platt | 0.106536 | 0.215998 | 0.557592 | 7 | 1.42958 | 0.280587 |
| claude-sonnet-4 | 10 | 0.592234 | 0.1 | platt | 0.0605308 | 0.191304 | 0.594241 | 7 | 1.17661 | 0.305265 |
| deepseek-r1-0528 | 10 | 0.464252 | 0.1 | platt | 0.0629863 | 0.158738 | 0.772251 | 7 | 1.4417 | 0.266735 |
| deepseek-v3-0324 | 10 | 0.610041 | 0.1 | platt | 0.0685828 | 0.210851 | 0.662304 | 7 | 1.6384 | 0.266735 |
| gemini-2.5-flash | 10 | 0.622267 | 1 | platt | 0.0705991 | 0.18845 | 0.657068 | 7 | 2.8264 | 0.282822 |
| gemini-2.5-pro | 1 | 0.428493 | 10 | platt | 0.0501873 | 0.145565 | 0.803665 | 7 | 2.14549 | 0.281635 |
| glm-4.6 | 10 | 0.559945 | 1 | platt | 0.0727222 | 0.192547 | 0.688482 | 7 | 1.46905 | 0.27451 |
| gpt-5 | 1 | 0.425722 | 1 | platt | 0.0622385 | 0.113927 | 0.866492 | 7 | 1.84125 | 0.273002 |
| gpt-5-chat | 10 | 0.559061 | 0.1 | platt | 0.0780372 | 0.191273 | 0.670157 | 7 | 1.33842 | 0.274397 |
| intern-s1 | 1 | 0.609862 | 1 | platt | 0.0595078 | 0.188107 | 0.596859 | 7 | 1.38315 | 0.369102 |
| kimi-k2-0905 | 10 | 0.555829 | 1 | platt | 0.0576807 | 0.184928 | 0.709424 | 7 | 1.89717 | 0.295181 |
| qwen3-235b-a22b-2507 | 1 | 0.551628 | 1 | platt | 0.0694894 | 0.177294 | 0.688482 | 7 | 1.51178 | 0.284053 |
| qwen3-235b-a22b-thinking-2507 | 10 | 0.455971 | 1 | platt | 0.050905 | 0.139845 | 0.795812 | 7 | 4.89061 | 0.283329 |

## Artifact

| key | value |
| --- | --- |
| pool | mixed |
| artifact version | 15201d82e792 |
| dim | 65536 |
| train items | 1067 |
| calibration items | 382 |
| ood k | 32 |
| ood threshold (0.99 quantile of calibration scores) | 1 |
| wall time (s) | 75.6 |
