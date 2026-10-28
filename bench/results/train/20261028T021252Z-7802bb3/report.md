# train

Run `20261028T021252Z-7802bb3`.

**Labels:** calib_ece and calib_brier are measured on the calibration split, which is in-sample for the calibrator and the lower-bound tables; no test or held-out row was read; prices are not used here; nothing in this run is simulated or assumed.

## Per-model fit (calib_ece and calib_brier: calibration split, in-sample for the calibrator)

| model | C | cv_log_loss | alpha | calibrator | calib_ece | calib_brier | calib_success_rate | lcb_bins | length_smear | in_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-sonnet-4 | 1 | 0.414875 | 1 | isotonic | 0.0117593 | 0.121806 | 0.218274 | 15 | 1.13259 | 0.284211 |
| deepseek-r1-0528 | 1 | 0.524043 | 1 | isotonic | 0.0192146 | 0.170972 | 0.344614 | 15 | 1.20424 | 0.227194 |
| deepseek-v3-0324 | 1 | 0.484995 | 1 | isotonic | 0.0310376 | 0.149613 | 0.280316 | 15 | 1.32437 | 0.227194 |
| gemini-2.5-flash | 1 | 0.527003 | 1 | isotonic | 0.0138757 | 0.158164 | 0.299492 | 15 | 2.21795 | 0.222222 |
| gemini-2.5-pro | 1 | 0.597098 | 10 | isotonic | 0.0147303 | 0.201972 | 0.50141 | 15 | 1.49242 | 0.22108 |
| glm-4.6 | 1 | 0.530877 | 1 | isotonic | 0.0201117 | 0.171869 | 0.319797 | 15 | 1.71231 | 0.243697 |
| gpt-5 | 1 | 0.598186 | 1 | isotonic | 0.020295 | 0.196314 | 0.514382 | 15 | 1.41093 | 0.250588 |
| gpt-5-chat | 1 | 0.525106 | 1 | isotonic | 0.0208152 | 0.174469 | 0.368866 | 15 | 1.36331 | 0.25772 |
| intern-s1 | 1 | 0.426353 | 1 | isotonic | 0.0298098 | 0.133686 | 0.227862 | 15 | 1.2918 | 0.655797 |
| kimi-k2-0905 | 1 | 0.490714 | 1 | isotonic | 0.00915893 | 0.158674 | 0.319797 | 15 | 1.54228 | 0.380165 |
| qwen3-235b-a22b-2507 | 1 | 0.545565 | 1 | isotonic | 0.0148549 | 0.174826 | 0.442752 | 15 | 1.90986 | 0.271829 |
| qwen3-235b-a22b-thinking-2507 | 1 | 0.522376 | 1 | isotonic | 0.0174095 | 0.166871 | 0.440496 | 15 | 18.1597 | 0.276596 |

## Artifact

| key | value |
| --- | --- |
| pool | flagship |
| artifact version | dea4486d7575 |
| dim | 65536 |
| train items | 5185 |
| calibration items | 1773 |
| ood k | 32 |
| ood threshold (0.99 quantile of calibration scores) | 0.959316 |
| wall time (s) | 134.1 |
