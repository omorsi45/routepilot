# ablations

Run `20261031T070638Z-f0429c6`.

**Labels:** costs are the recorded costs of each model on each item; no latency is measured or simulated in this experiment; lambda of the dual mode is retuned on the calibration split for each artifact with the same rule as the headline (the largest value whose calibration success reaches Best Single's minus 0.01); the test split is replayed once per artifact; calibration success is in-sample for the calibrators and lower-bound tables of every artifact (fitted on the same split lambda is tuned on); because every operating point is retuned to the same calibration target, test success differs little by construction; the cost, the paired intervals and the pooled test ECE carry the comparison; better or worse means the paired 95% bootstrap interval against the shipped configuration lies wholly above or below zero; ECE is compared by its point estimate; pooled test ECE: 15 equal-mass bins over all models' test forecasts, recomputed in Python from the exported weights; training wall time is one run per artifact on this machine, measured around the whole fit and export; the shipped artifact was not retrained, its time is that of the committed train run; ablation artifacts are written under bench/scratch/ablations and are not committed.

## Does each ablation help? (against the shipped configuration, test split)

| statement |
| --- |
| both features, dim 2^14: test success +0.00 points against the shipped one (no clear difference), mean cost 13.7% dearer (worse); pooled test ECE 0.0223 against 0.0160 for the shipped one (higher). |
| both features, dim 2^18: test success +0.06 points against the shipped one (no clear difference), mean cost 11.6% dearer (no clear difference); pooled test ECE 0.0158 against 0.0160 for the shipped one (lower). |
| hashed features only, dim 2^16: test success -0.35 points against the shipped one (no clear difference), mean cost 35.6% dearer (worse); pooled test ECE 0.0146 against 0.0160 for the shipped one (lower). |
| dense features only, dim 2^16: test success +0.52 points against the shipped one (no clear difference), mean cost 117.7% dearer (worse); pooled test ECE 0.0186 against 0.0160 for the shipped one (higher). |
| every model Platt, dim 2^16: test success +2.25 points against the shipped one (better), mean cost 63.7% dearer (worse); pooled test ECE 0.0115 against 0.0160 for the shipped one (lower). |
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
| wall time (s) | 670.7 |

## Ablations: test split, dual operating point retuned on calibration (paired against always_strongest)

| variant | lambda | calib success | test success [95% CI] | test cost USD [95% CI] | success delta vs always_strongest [95% CI] | cost reduction vs always_strongest [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| always_strongest (reference) |  |  | 0.5274 [0.5049, 0.5516] | 0.0670642 [0.0629193, 0.0712717] |  |  |
| both features, dim 2^14 | 6.266589480219409 | 0.491822 | 0.4818 [0.4599, 0.5066] | 0.00498853 [0.00433001, 0.00569022] | -0.0455 [-0.0686, -0.0225] | 0.9256 [0.9150, 0.9354] |
| both features, dim 2^16 (shipped) | 6.721045792801306 | 0.491822 | 0.4818 [0.4599, 0.5055] | 0.00438797 [0.00383025, 0.00495434] | -0.0455 [-0.0697, -0.0225] | 0.9346 [0.9250, 0.9432] |
| both features, dim 2^18 | 6.031649718352128 | 0.491822 | 0.4824 [0.4594, 0.5072] | 0.00489678 [0.00426437, 0.00556576] | -0.0450 [-0.0686, -0.0225] | 0.9270 [0.9169, 0.9366] |
| hashed features only, dim 2^16 | 5.096882887301035 | 0.491822 | 0.4784 [0.4548, 0.5032] | 0.0059515 [0.00519692, 0.00667514] | -0.0490 [-0.0703, -0.0271] | 0.9113 [0.8993, 0.9227] |
| dense features only, dim 2^16 | 3.3382793844793923 | 0.491822 | 0.4870 [0.4651, 0.5107] | 0.00955472 [0.00818161, 0.0110024] | -0.0403 [-0.0640, -0.0161] | 0.8575 [0.8356, 0.8781] |
| every model Platt, dim 2^16 | 4.704503226093948 | 0.491822 | 0.5043 [0.4824, 0.5291] | 0.00718341 [0.00631242, 0.00821585] | -0.0231 [-0.0461, -0.0029] | 0.8929 [0.8768, 0.9055] |
| every model isotonic, dim 2^16 | 6.721045792801306 | 0.491822 | 0.4818 [0.4599, 0.5055] | 0.00438797 [0.00383025, 0.00495434] | -0.0455 [-0.0697, -0.0225] | 0.9346 [0.9250, 0.9432] |

## Against the shipped configuration (paired by test item)

| variant | success delta [95% CI] | cost reduction [95% CI] | pooled test ECE | ECE minus the shipped one |
| --- | --- | --- | --- | --- |
| both features, dim 2^14 | 0.0000 [-0.0115, 0.0115] | -0.1369 [-0.2916, -0.0097] | 0.0222609 | 0.00622536 |
| both features, dim 2^16 (shipped) | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0160356 | 0 |
| both features, dim 2^18 | 0.0006 [-0.0098, 0.0098] | -0.1160 [-0.2749, 0.0083] | 0.0157965 | -0.000239051 |
| hashed features only, dim 2^16 | -0.0035 [-0.0179, 0.0104] | -0.3563 [-0.5707, -0.1684] | 0.014624 | -0.00141159 |
| dense features only, dim 2^16 | 0.0052 [-0.0110, 0.0213] | -1.1775 [-1.5928, -0.8239] | 0.0185538 | 0.00251817 |
| every model Platt, dim 2^16 | 0.0225 [0.0098, 0.0346] | -0.6371 [-0.8555, -0.4607] | 0.0115479 | -0.00448764 |
| every model isotonic, dim 2^16 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0160356 | 0 |

## Artifacts: calibration quality, training wall time and weights.bin size

| variant | artifact version | calibrators | pooled test ECE (15 equal-mass bins) | pooled test ECE (equal-width) | Brier | log-loss | OOD threshold | training wall time (s) | wall time from | weights.bin size (bytes) | largest \|p Go - p Python\| |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| both features, dim 2^14 | 6b5f6eb3439c | {"isotonic": 12} | 0.0222609 | 0.015772 | 0.173619 | 0.534115 | 0.950261 | 111 | this run | 1697280 | 4.30211e-15 |
| both features, dim 2^16 (shipped) | dea4486d7575 | {"isotonic": 12} | 0.0160356 | 0.0150998 | 0.172634 | 0.528971 | 0.959316 | 134.1 | bench/results/train/20261028T021252Z-7802bb3 | 6407616 | 5.60663e-15 |
| both features, dim 2^18 | 1a6da465914d | {"isotonic": 12} | 0.0157965 | 0.0146347 | 0.172184 | 0.528934 | 0.963418 | 214.4 | this run | 25281768 | 3.25573e-14 |
| hashed features only, dim 2^16 | 195d38b51957 | {"isotonic": 12} | 0.014624 | 0.0141542 | 0.172544 | 0.523678 | 0.959316 | 54.5 | this run | 6407616 | 4.13003e-14 |
| dense features only, dim 2^16 | 8a6750c31dcd | {"isotonic": 12} | 0.0185538 | 0.0206666 | 0.191133 | 0.582393 | 0.959316 | 5.1 | this run | 6407616 | 1.86517e-14 |
| every model Platt, dim 2^16 | b70d7e117412 | {"platt": 12} | 0.0115479 | 0.0107057 | 0.171374 | 0.514304 | 0.959316 | 128 | this run | 6407616 | 4.71845e-16 |
| every model isotonic, dim 2^16 | 5a2cbe3e5af1 | {"isotonic": 12} | 0.0160356 | 0.0150998 | 0.172634 | 0.528971 | 0.959316 | 127.9 | this run | 6407616 | 5.60663e-15 |

## Lambda retuned on calibration (target Q_bar = Best Single's minus 0.01)

| variant | status | lambda (USD^-1) | calib success at lambda | target Q_bar | calib success at lambda 0 | calib mean cost USD |
| --- | --- | --- | --- | --- | --- | --- |
| both features, dim 2^14 | tuned | 6.266589480219409 | 0.491822 | 0.49141 | 0.56176 | 0.00460907 |
| both features, dim 2^16 (shipped) | tuned | 6.721045792801306 | 0.491822 | 0.49141 | 0.560068 | 0.00408349 |
| both features, dim 2^18 | tuned | 6.031649718352128 | 0.491822 | 0.49141 | 0.567964 | 0.0050081 |
| hashed features only, dim 2^16 | tuned | 5.096882887301035 | 0.491822 | 0.49141 | 0.563452 | 0.00596505 |
| dense features only, dim 2^16 | tuned | 3.3382793844793923 | 0.491822 | 0.49141 | 0.538071 | 0.0113217 |
| every model Platt, dim 2^16 | tuned | 4.704503226093948 | 0.491822 | 0.49141 | 0.555556 | 0.00740412 |
| every model isotonic, dim 2^16 | tuned | 6.721045792801306 | 0.491822 | 0.49141 | 0.560068 | 0.00408349 |

## Not run

| note |
| --- |
| A PyTorch MLP on local Ollama embeddings is not run (no PyTorch dependency, and the predictor must stay Go-servable without an embedding model on the request path); ADR 0007 records it as a deferred ablation. |
