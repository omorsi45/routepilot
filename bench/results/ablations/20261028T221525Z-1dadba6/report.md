# ablations

Run `20261028T221525Z-1dadba6`.

**Labels:** costs are the recorded costs of each model on each item; no latency is measured or simulated in this experiment; lambda of the dual mode is retuned on the calibration split for each artifact with the same rule as the headline (the largest value whose calibration success reaches Best Single's minus 0.01); the test split is replayed once per artifact; calibration success is in-sample for the calibrators and lower-bound tables of every artifact (fitted on the same split lambda is tuned on); because every operating point is retuned to the same calibration target, test success differs little by construction; the cost, the paired intervals and the pooled test ECE carry the comparison; better or worse means the paired 95% bootstrap interval against the shipped configuration lies wholly above or below zero; ECE is compared by its point estimate; pooled test ECE: 15 equal-mass bins over all models' test forecasts, recomputed in Python from the exported weights; training wall time is one run per artifact on this machine, measured around the whole fit and export; the shipped artifact was not retrained, its time is that of the committed train run; ablation artifacts are written under bench/scratch/ablations and are not committed.

## Does each ablation help? (against the shipped configuration, test split)

| statement |
| --- |
| both features, dim 2^14: test success +0.00 points against the shipped one (no clear difference), mean cost 14.8% dearer (worse); pooled test ECE 0.0223 against 0.0160 for the shipped one (higher). |
| both features, dim 2^18: test success +0.12 points against the shipped one (no clear difference), mean cost 8.1% dearer (no clear difference); pooled test ECE 0.0158 against 0.0160 for the shipped one (lower). |
| hashed features only, dim 2^16: test success -0.35 points against the shipped one (no clear difference), mean cost 33.3% dearer (worse); pooled test ECE 0.0146 against 0.0160 for the shipped one (lower). |
| dense features only, dim 2^16: test success +0.52 points against the shipped one (no clear difference), mean cost 123.3% dearer (worse); pooled test ECE 0.0186 against 0.0160 for the shipped one (higher). |
| every model Platt, dim 2^16: test success +2.42 points against the shipped one (better), mean cost 64.2% dearer (worse); pooled test ECE 0.0115 against 0.0160 for the shipped one (lower). |
| every model isotonic, dim 2^16: test success +0.00 points against the shipped one (no clear difference), mean cost 0.0% cheaper (no clear difference); pooled test ECE 0.0160 against 0.0160 for the shipped one (equal). |

## Setup

| setting | value |
| --- | --- |
| pool | flagship |
| shipped artifact | dea4486d7575 |
| items (calibration / test) | 1773 / 1735 |
| Best Single (always_strongest) | gemini-2.5-pro |
| Best Single calibration success | 0.50141 |
| target Q_bar (that minus the margin) | 0.49141 (0.01) |
| always_strongest test success [95% CI] | 0.5274 [0.5049, 0.5516] |
| always_strongest test cost USD [95% CI] | 0.0670642 [0.0629193, 0.0712717] |
| policy config (other than lambda) | {"mode": "dual", "default_floor": 0, "epsilon": 0, "ood_action": "conservative", "on_infeasible": "best_effort"} |
| bootstrap | 1000 resamples, percentile 95% intervals, seed 0 |
| replays run (calibration / test) | 295 / 8 |
| wall time (s) | 741.1 |

## Ablations: test split, dual operating point retuned on calibration (paired against always_strongest)

| variant | lambda | calib success | test success [95% CI] | test cost USD [95% CI] | success delta vs always_strongest [95% CI] | cost reduction vs always_strongest [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| always_strongest (reference) |  |  | 0.5274 [0.5049, 0.5516] | 0.0670642 [0.0629193, 0.0712717] |  |  |
| both features, dim 2^14 | 6.2657136368216015 | 0.491822 | 0.4818 [0.4599, 0.5066] | 0.00498091 [0.00431349, 0.00568911] | -0.0455 [-0.0686, -0.0225] | 0.9257 [0.9150, 0.9353] |
| both features, dim 2^16 (shipped) | 6.719579687342048 | 0.491822 | 0.4818 [0.4599, 0.5055] | 0.00433716 [0.00377782, 0.00490005] | -0.0455 [-0.0697, -0.0225] | 0.9353 [0.9259, 0.9439] |
| both features, dim 2^18 | 6.040280823071953 | 0.491822 | 0.4830 [0.4599, 0.5078] | 0.00469028 [0.00415361, 0.00521368] | -0.0444 [-0.0680, -0.0225] | 0.9301 [0.9210, 0.9383] |
| hashed features only, dim 2^16 | 5.178721039555967 | 0.491822 | 0.4784 [0.4559, 0.5026] | 0.00578091 [0.00504018, 0.00649225] | -0.0490 [-0.0709, -0.0259] | 0.9138 [0.9017, 0.9250] |
| dense features only, dim 2^16 | 3.3371288736816496 | 0.491822 | 0.4870 [0.4646, 0.5101] | 0.00968689 [0.0082803, 0.0112356] | -0.0403 [-0.0640, -0.0161] | 0.8556 [0.8328, 0.8755] |
| every model Platt, dim 2^16 | 4.714735950983595 | 0.491822 | 0.5061 [0.4836, 0.5303] | 0.00712356 [0.00628155, 0.00814385] | -0.0213 [-0.0444, 0.0000] | 0.8938 [0.8776, 0.9062] |
| every model isotonic, dim 2^16 | 6.719579687342048 | 0.491822 | 0.4818 [0.4599, 0.5055] | 0.00433716 [0.00377782, 0.00490005] | -0.0455 [-0.0697, -0.0225] | 0.9353 [0.9259, 0.9439] |

## Against the shipped configuration (paired by test item)

| variant | success delta [95% CI] | cost reduction [95% CI] | pooled test ECE | ECE minus the shipped one |
| --- | --- | --- | --- | --- |
| both features, dim 2^14 | 0.0000 [-0.0115, 0.0115] | -0.1484 [-0.3027, -0.0214] | 0.0222609 | 0.00622536 |
| both features, dim 2^16 (shipped) | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0160356 | 0 |
| both features, dim 2^18 | 0.0012 [-0.0092, 0.0110] | -0.0814 [-0.2026, 0.0305] | 0.0157965 | -0.000239051 |
| hashed features only, dim 2^16 | -0.0035 [-0.0179, 0.0098] | -0.3329 [-0.5471, -0.1487] | 0.014624 | -0.00141159 |
| dense features only, dim 2^16 | 0.0052 [-0.0110, 0.0219] | -1.2335 [-1.6495, -0.8655] | 0.0185538 | 0.00251817 |
| every model Platt, dim 2^16 | 0.0242 [0.0127, 0.0363] | -0.6424 [-0.8737, -0.4583] | 0.0115479 | -0.00448764 |
| every model isotonic, dim 2^16 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0160356 | 0 |

## Artifacts: calibration quality, training wall time and weights.bin size

| variant | artifact version | calibrators | pooled test ECE (15 equal-mass bins) | pooled test ECE (equal-width) | Brier | log-loss | OOD threshold | training wall time (s) | wall time from | weights.bin size (bytes) | largest \|p Go - p Python\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| both features, dim 2^14 | ddd10a23e170 | {"isotonic": 12} | 0.0222609 | 0.015772 | 0.173619 | 0.534115 | 0.950261 | 124.7 | this run | 1697280 | 4.30211e-15 |
| both features, dim 2^16 (shipped) | dea4486d7575 | {"isotonic": 12} | 0.0160356 | 0.0150998 | 0.172634 | 0.528971 | 0.959316 | 134.1 | bench/results/train/20261028T021252Z-7802bb3 | 6407616 | 5.60663e-15 |
| both features, dim 2^18 | c23573da8a94 | {"isotonic": 12} | 0.0157965 | 0.0146347 | 0.172184 | 0.528934 | 0.963418 | 228 | this run | 25281768 | 3.25573e-14 |
| hashed features only, dim 2^16 | 0918854c06eb | {"isotonic": 12} | 0.014624 | 0.0141542 | 0.172544 | 0.523678 | 0.959316 | 63.8 | this run | 6407616 | 4.13003e-14 |
| dense features only, dim 2^16 | 681f33285c5a | {"isotonic": 12} | 0.0185538 | 0.0206666 | 0.191133 | 0.582393 | 0.959316 | 5.9 | this run | 6407616 | 1.86517e-14 |
| every model Platt, dim 2^16 | b4ac853617ee | {"platt": 12} | 0.0115479 | 0.0107057 | 0.171374 | 0.514304 | 0.959316 | 140.2 | this run | 6407616 | 4.71845e-16 |
| every model isotonic, dim 2^16 | 188f9804ce15 | {"isotonic": 12} | 0.0160356 | 0.0150998 | 0.172634 | 0.528971 | 0.959316 | 139 | this run | 6407616 | 5.60663e-15 |

## Lambda retuned on calibration (target Q_bar = Best Single's minus 0.01)

| variant | status | lambda (USD^-1) | calib success at lambda | target Q_bar | calib success at lambda 0 | calib mean cost USD |
| --- | --- | --- | --- | --- | --- | --- |
| both features, dim 2^14 | tuned | 6.2657136368216015 | 0.491822 | 0.49141 | 0.56176 | 0.00462001 |
| both features, dim 2^16 (shipped) | tuned | 6.719579687342048 | 0.491822 | 0.49141 | 0.560068 | 0.00411058 |
| both features, dim 2^18 | tuned | 6.040280823071953 | 0.491822 | 0.49141 | 0.567964 | 0.00503913 |
| hashed features only, dim 2^16 | tuned | 5.178721039555967 | 0.491822 | 0.49141 | 0.563452 | 0.00576139 |
| dense features only, dim 2^16 | tuned | 3.3371288736816496 | 0.491822 | 0.49141 | 0.538071 | 0.0113032 |
| every model Platt, dim 2^16 | tuned | 4.714735950983595 | 0.491822 | 0.49141 | 0.555556 | 0.00731991 |
| every model isotonic, dim 2^16 | tuned | 6.719579687342048 | 0.491822 | 0.49141 | 0.560068 | 0.00411058 |

## Not run

| note |
| --- |
| A PyTorch MLP on local Ollama embeddings is not run in this plan (Ruling V1-R19: no PyTorch dependency, and the predictor must stay Go-servable without an embedding model on the request path); the ADR of Task 36 records it as a deferred ablation. |
