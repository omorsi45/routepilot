# Reproducibility guide

This page says what machine produced the committed results, which commands produced them and in what order, how long each took, which seeds they used, how to check the dataset and an artifact, and what will not come out bit for bit on another machine. Every run under `bench/results/<experiment>/<run-id>/` has a `run.json` with the commit, whether the tree was dirty, the policy version, the dataset SHA-256, the seeds, the machine, the tool versions, the start and end times and the command. The times below are the difference between `started_at` and `ended_at` in those files, in whole seconds, so each is accurate to about a second.

## Machine and tools

| Item | Value (from the `run.json` files unless noted) |
|---|---|
| CPU | AMD Ryzen 9 7900X 12-Core Processor (24 threads) |
| Windows host | Windows 11 (10.0.26300), 31.2 GB RAM visible. Every offline experiment, the failover and quota experiments and the local-model experiments ran here. |
| Linux host | Ubuntu under WSL2 on the same machine: kernel 6.18.33.2-microsoft-standard-WSL2, glibc 2.39, 15.2 GB RAM visible. The overhead and LiteLLM experiments ran here, and so did one of the two clock probes. |
| Go | 1.27.2 (the `go.mod` version), `windows/amd64` and `linux/amd64` |
| Python | 3.14.7 on Windows, 3.14.9 under WSL2 |
| Python packages | The direct dependencies are pinned with `==` in `python/pyproject.toml`: numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, pyarrow 26.0.0, scikit-learn 1.9.1, matplotlib 3.11.2, pyyaml 6.0.3; development: pytest 9.2.0, ruff 0.17.3; build: setuptools 82.0.0. Transitive versions were not recorded and there is no lock file, so a later `make py-setup` can resolve different ones |
| Other tools | GNU Make 4.4.1 and Git 2.55.0 for Windows (read from the tools on this machine on 2026-10-29), bash from Git for Windows |
| LiteLLM | 1.106.6 in a separate environment, resolved with uv 0.11.2 and pinned with hashes in `bench/litellm/requirements.txt` ([bench/litellm/README.md](../bench/litellm/README.md)) |
| Ollama | 0.34.2 on the Windows host, with the models qwen2.5:0.5b, qwen2.5:1.5b and qwen2.5:3b (Q4_K_M) |

No experiment calls a commercial API or needs an API key.

## Timing runs need Linux

Gateway timing metrics are only as fine as the host clock. The clock probe reads `time.Now` for 200 ms and reports the smallest step between two reads. On this machine's Windows host the step was 344,800 to 501,400 ns and more than 99.99% of consecutive reads were equal, so a sub-millisecond duration recorded there is a count of clock ticks. Under WSL2 on the same machine the step was 30 ns in all five probes. `overhead.py` and the other performance experiments refuse to run when the step is above 1000 ns. Runs: [Windows](../bench/results/clock/20261031T070310Z-f0429c6/) and [Linux](../bench/results/clock/20261031T070316Z-f0429c6/), shown in [performance.md](performance.md#clock-resolution). Production targets Linux, and so does every performance number here.

Under WSL2, keep the scratch directory (binaries, configs, the gateway's request log) on the Linux file system, for example `--scratch-root ~/rp-scratch`. The first Linux overhead run kept it on the `/mnt/c` mount and showed a tail at 2,000 requests per second without streaming that the primary run, with scratch files on ext4, did not repeat. Both runs are kept and explained in [performance.md](performance.md#gateway-overhead-against-the-local-mock).

## Verify the dataset

The dataset is the LLMRouterBench release `bench-release.tar.gz` ([data/SOURCES.md](../data/SOURCES.md) has the URL and license). It is never committed. Expected size and SHA-256:

```text
1,283,503,080 bytes
b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1
```

`make data` downloads it to `data/raw/bench-release.tar.gz` and `download.verify` refuses a file whose size or SHA-256 differs. To check a copy by hand:

```bash
ls -l data/raw/bench-release.tar.gz          # 1283503080 bytes
sha256sum data/raw/bench-release.tar.gz      # b79f8cde...c8f1
```

```powershell
Get-FileHash -Algorithm SHA256 data\raw\bench-release.tar.gz
```

The copy used for the committed runs has this size and this hash (checked again on 2026-10-29). Every `run.json` records the same hash as `dataset_sha256`.

## Verify an artifact version

An artifact's version is the first 12 hex digits of the SHA-256 of its manifest (with an empty version field) followed by `weights.bin` ([the format](architecture.md#the-artifact-format-and-content-hash)). `routepilot policy verify` recomputes it and runs the same checks as the gateway's loader, including the golden predictions. It prints the version and exits 0, or exits 1 and names the failed check. `inspect` runs the same checks and prints a JSON summary.

```bash
make build
./bin/routepilot policy verify policies/live-ollama/21183f8ab3fe    # prints 21183f8ab3fe
./bin/routepilot policy inspect policies/live-ollama/21183f8ab3fe
```

| Artifact | Version | In the repository |
|---|---|---|
| Flagship pool, behind the headline and most experiments | `dea4486d7575` | No. `make train` writes it under `policies/flagship/`. |
| Mixed pool (sensitivity only) | `15201d82e792` | No. Written under `policies/mixed/` by the mixed training command below. |
| Local-model demo | `21183f8ab3fe` | Yes, under `policies/live-ollama/`. |

On the machine that produced the results, `verify` printed all three versions and exited 0 (2026-10-29). The version the gateway reports in `x-routepilot-policy-version` is `<artifact version>+<8 hex digits of the policy config hash>`, for example `dea4486d7575+ef0a989c` in the overhead runs (their policy config had the earlier lambda, see the note under the commands table). A trained artifact carries its training metadata (commit, creation time) in the hashed manifest, so retraining gives a new version even from identical inputs; see the last section.

## Seeds

| Seed | Value | Used by |
|---|---|---|
| Split salt | `routepilot-split-v1` | data, train |
| Cross-fitting salt | `routepilot-crossfit-v1` | headline |
| Cross-validation folds, k-means | 0 | train, ablations |
| Bootstrap | 0 (OPE validation: the logging seed) | every table with intervals |
| Random policy | 0 (baselines: 0 to 4) | headline, baselines, live_train |
| Matrix factorization | 0 | mf_train, sweeps |
| Calibration subsamples | 0 to 19 | calibration |
| Logging policy | 0 to 49 | ope_validation |
| Bandits | 0 to 9; hyperparameter seed 0; SVD seed 0 | bandits |
| Load generator, mock | 1, 1 | overhead, litellm |
| Failover: generator pre-stream, mid-stream; mocks A, B | 1, 2; 1, 2 | failover |
| Quota clients | 1 | quota_correctness |
| Live sample, demo items | 0, 0 | live_collect, live_demo |

## Commands in the order they were run

Run everything from the repository root with the repository's virtual environment (`make py-setup` creates `.venv`; its interpreter is `.venv/Scripts/python.exe` on Windows and `.venv/bin/python` elsewhere, and the Makefile finds it). Times are wall times from `run.json`. Run IDs link to the run. The superseded runs that are not listed are in `bench/results/SELECTION.yaml` with the reason.

| # | Command | Host | Wall time | Run |
|---|---|---|---|---|
| 1 | `make py-setup` | Windows | not recorded | |
| 2 | `make data` (download, ingest, build splits, pools and replay exports into `data/derived`, about 1.28 GB to download) | Windows | not recorded (no `run.json`) | |
| 3 | `python bench/scripts/data_summary.py` | Windows | under 1 s | [`20261031T075023Z-f03e732`](../bench/results/data/20261031T075023Z-f03e732/) |
| 4 | `python bench/scripts/baselines.py --pool flagship` | Windows | 8 s | [`20261031T064741Z-d02bb26`](../bench/results/baselines/20261031T064741Z-d02bb26/) |
| 5 | `python bench/scripts/baselines.py --pool mixed` | Windows | 2 s | [`20261031T064750Z-d02bb26`](../bench/results/baselines/20261031T064750Z-d02bb26/) |
| 6 | `python bench/scripts/inupper.py` | Windows | 8 s | [`20261028T012056Z-1212012`](../bench/results/inupper/20261028T012056Z-1212012/) |
| 7 | `make train` (`python -m routepilot_train.train.pipeline --pool flagship --derived data/derived --out policies`, run with `--dim 65536`, the default) | Windows | 134 s | [`20261028T021252Z-7802bb3`](../bench/results/train/20261028T021252Z-7802bb3/) |
| 8 | `python -m routepilot_train.baselines.routellm_mf --pool flagship --derived data/derived` | Windows | 13 s | [`20261028T033419Z-5e4dbcc`](../bench/results/mf_train/20261028T033419Z-5e4dbcc/) |
| 9 | `python bench/scripts/sweeps.py --pool flagship` (tunes lambda into `bench/policies/flagship.yaml`) | Windows | 27 s | [`20261031T064753Z-d02bb26`](../bench/results/sweeps/20261031T064753Z-d02bb26/) |
| 10 | `python bench/scripts/overhead.py --scratch-root ~/rp-scratch` | Linux | 1241 s (about 21 min) | [`20261028T193224Z-9ac474d`](../bench/results/overhead/20261028T193224Z-9ac474d/) |
| 11 | `python -m routepilot_train.train.pipeline --pool mixed --derived data/derived --out policies --dim 65536 --calibrator switch` | Windows | 76 s | [`20261028T202236Z-e49df83`](../bench/results/train/20261028T202236Z-e49df83/) |
| 12 | `python -m routepilot_train.baselines.routellm_mf --pool mixed --derived data/derived` | Windows | 4 s | [`20261028T202400Z-e49df83`](../bench/results/mf_train/20261028T202400Z-e49df83/) |
| 13 | `python bench/scripts/headline.py --pool flagship` | Windows | 26 s | [`20261031T070340Z-f0429c6`](../bench/results/headline/20261031T070340Z-f0429c6/) |
| 14 | `python bench/scripts/headline.py --pool mixed` | Windows | 9 s | [`20261031T070408Z-f0429c6`](../bench/results/headline/20261031T070408Z-f0429c6/) |
| 15 | `python bench/scripts/calibration_study.py --pool flagship` | Windows | 28 s | [`20261028T203251Z-028d6b4`](../bench/results/calibration/20261028T203251Z-028d6b4/) |
| 16 | `python bench/scripts/lcb_vs_point.py --pool flagship` | Windows | 4 s | [`20261031T070419Z-f0429c6`](../bench/results/lcb_vs_point/20261031T070419Z-f0429c6/) |
| 17 | `python bench/scripts/shift_ood.py --pool flagship` | Windows | 3 s | [`20261031T070425Z-f0429c6`](../bench/results/shift_ood/20261031T070425Z-f0429c6/) |
| 18 | `python bench/scripts/ope_validation.py --pool flagship` | Windows | 28 s | [`20261031T070430Z-f0429c6`](../bench/results/ope_validation/20261031T070430Z-f0429c6/) |
| 19 | `python bench/scripts/ablations.py --pool flagship` | Windows | 671 s (about 11 min) | [`20261031T070638Z-f0429c6`](../bench/results/ablations/20261031T070638Z-f0429c6/) |
| 20 | `python bench/scripts/bandits.py --pool flagship` | Windows | 95 s | [`20261031T070500Z-f0429c6`](../bench/results/bandits/20261031T070500Z-f0429c6/) |
| 21 | `python bench/scripts/litellm_compare.py --scratch-root ~/rp-scratch` (needs the LiteLLM environment) | Linux | 1901 s (about 32 min) | [`20261028T234116Z-ee7619a`](../bench/results/litellm/20261028T234116Z-ee7619a/) |
| 22 | `python bench/scripts/quota_correctness.py` (needs Docker, or a Redis named by `ROUTEPILOT_TEST_REDIS_URL`) | Windows | 19 s | [`20261029T010202Z-f5ff91d`](../bench/results/quota_correctness/20261029T010202Z-f5ff91d/) |
| 23 | `python bench/scripts/failover.py` | Windows | 160 s | [`20261029T010235Z-f5ff91d`](../bench/results/failover/20261029T010235Z-f5ff91d/) |
| 24 | `python bench/scripts/failure_analysis.py` | Windows | 1 s | [`20261031T071751Z-f0429c6`](../bench/results/failure_analysis/20261031T071751Z-f0429c6/) |
| 25 | `python bench/scripts/live_collect.py measure` (needs Ollama) | Windows | 32 s | [`20261029T015819Z-00deb40`](../bench/results/live_collect/20261029T015819Z-00deb40/) |
| 26 | `python bench/scripts/live_collect.py collect --live-dir data/derived/live-cot` (needs Ollama; first protocol, as written) | Windows | 4619 s (about 77 min) | [`20261029T015906Z-00deb40`](../bench/results/live_collect/20261029T015906Z-00deb40/) |
| 27 | `python bench/scripts/live_collect.py collect --protocol answer-only --budget-minutes 30` (needs Ollama) | Windows | 352 s | [`20261029T033743Z-766a3b8`](../bench/results/live_collect/20261029T033743Z-766a3b8/) |
| 28 | `python bench/scripts/live_collect.py collect --protocol answer-only-user --budget-minutes 15 --live-dir data/derived/live-user` (needs Ollama) | Windows | 309 s | [`20261029T042250Z-6ecd7f2`](../bench/results/live_collect/20261029T042250Z-6ecd7f2/) |
| 29 | `python bench/scripts/live_train.py --live-dir data/derived/live-user` | Windows | 14 s | [`20261029T043015Z-6ecd7f2`](../bench/results/live_train/20261029T043015Z-6ecd7f2/) |
| 30 | `python bench/scripts/live_demo.py --live-dir data/derived/live-user` (needs Ollama) | Windows | 106 s | [`20261029T043138Z-6ecd7f2`](../bench/results/live_demo/20261029T043138Z-6ecd7f2/) |
| 31 | `python bench/scripts/clock_probe.py` | Windows | 2 s | [`20261031T070310Z-f0429c6`](../bench/results/clock/20261031T070310Z-f0429c6/) |
| 32 | `python bench/scripts/clock_probe.py` | Linux | 5 s | [`20261031T070316Z-f0429c6`](../bench/results/clock/20261031T070316Z-f0429c6/) |

Row 27 is the answer-only protocol with the instruction in a system message. The `live_train` and `live_demo` runs made on it (`20261029T034651Z-766a3b8`, `20261029T034751Z-766a3b8`) are superseded and listed in `SELECTION.yaml`, and so are the earlier failover and quota runs. The first Linux overhead run ([`20261028T185609Z-44f7f4d`](../bench/results/overhead/20261028T185609Z-44f7f4d/), 1253 s) is kept next to the primary one.

Rows 3 to 5, 9, 13, 14, 16 to 20 and 24 are the second pass over the experiments that depend on the pool prices or on lambda. The prices in `pools/<pool>.json` and in the pool configs are fitted on the train and calibration rows only, and after that change `python -m routepilot_train.data.build --derived data/derived` (the last step of `make data`, which writes no `run.json`) was run again. Against the earlier build it changed the `prices` of both pools and nothing else in the pool JSON, and the splits and both replay exports are byte for byte the same. Training does not read prices (the run report says so), so rows 7, 8, 11 and 12 were not repeated and the artifacts `dea4486d7575` and `15201d82e792` stand. The rows named above were then run again in the order of the table, each from a clean tree, and the runs they replace stay in the tree, marked `older` in `SELECTION.yaml`. Row 3 ran twice: the first run counted the rows of every split in its rows-fitted column, and the second counts the train and calibration rows the prices were fitted on. Row 9 retuned lambda from 6.719579687342048 (run `20261028T033434Z-5e4dbcc`) to 6.721045792801306, and `bench/gates/baseline.json` was rewritten with `python bench/scripts/gate.py --update` and the flags of `make bench-gate`.

The other rows were not repeated, for these reasons. Rows 6, 7, 8, 11, 12 and 15 use outcomes, token counts, the prompt overhead and the artifacts, not the fitted prices. Rows 10 and 21 to 23 measure the gateway against a mock. The learned-policy part of row 10 loads the flagship catalog, so it ran with the earlier prices and lambda, and its served policy version `dea4486d7575+ef0a989c` names the config with the earlier lambda; the shipped config now gives `dea4486d7575+3d8c470b`. A different price or lambda can change which model the one repeated prompt of that run routes to, not the steps the router runs, so it was not repeated. Rows 25 to 30 use the assumed prices of the local models and their own policy.

Rows 26 to 30 need their live directories apart. The recorded command of run 26 is the bare `collect`, which wrote `data/derived/live`; that directory was renamed to `live-cot` before row 27 ran, a step that left no record. The flag in the table gives the first collection that directory from the start, so row 27 (default directory `data/derived/live`) does not stop on the existing `raw.jsonl`. Runs 29 and 30 recorded the bare `live_train.py` and `live_demo.py` as their commands, because the scripts did not record their arguments then. Both ran with `--live-dir data/derived/live-user`, which the `answer-only-user` protocol in the `measurement` block of each `run.json` shows; without the flag the scripts read the answer-only collection of row 27. A new run 29 also trains a new artifact version under `policies/live-ollama/` next to the committed `21183f8ab3fe`, because the manifest embeds its creation time; `live_demo.py` stops when it finds more than one, so run 30 needs `--policies-root` pointing at a directory that holds only the version it should use.

Three steps have no `run.json`, so they have no recorded time:

- `python bench/scripts/render_docs.py` regenerates `docs/results.md` and `docs/performance.md` from `bench/results` and `SELECTION.yaml`. It fails if a run named in `SELECTION.yaml` does not exist. Do not edit the two generated files by hand.
- `make bench-gate` replays the flagship test split with the learned dual policy and compares it with `bench/gates/baseline.json`. It needs `make data` and `make train` first, and exactly one artifact under `policies/flagship`.
- `make bench-smoke` runs the same gate mechanics on a synthetic fixture and needs no dataset.

The gateway tests, `make demo` and `make bench-smoke` need no dataset, no Docker, no Ollama and no key. `make verify` runs the hermetic checks.

## What needs Docker, Ollama or the internet

- **Internet:** the dataset download (about 1.28 GB), the Python and Go dependency installs, the LiteLLM environment, and pulling the Ollama models. Nothing else, and no API key.
- **Docker:** the real Redis in the quota experiment (unless `ROUTEPILOT_TEST_REDIS_URL` names one), the compose stack, `scripts/compose-smoke.sh`, and the Redis and PostgreSQL test suites, which run when `ROUTEPILOT_TEST_REDIS_URL` and `ROUTEPILOT_TEST_POSTGRES_DSN` are set.
- **Ollama:** the live collection, the live demo and the model speed probe (runs 25 to 28 and 30). Training on the collected data (run 29) needs only the files they wrote.
- **Linux:** the overhead and LiteLLM experiments (runs 10 and 21), for the clock reason above. The artifacts they load are built on the Windows host, and Go verifies their hashes on load.

## What is not reproducible bit for bit

- **Trained artifacts across platforms.** Linear algebra and math library routines (BLAS, libm) differ between Windows and Linux in the last bits of a float. A retrain on another platform can move the last bits of a weight, which changes `weights.bin` and so the content hash. The tests therefore compare predictions within tolerance and not bytes: a refit tiny artifact must reproduce the committed predictions (probability and out-of-distribution score within 1e-3, lower bound within 0.1 because a probability at a bin edge can land one bin over, expected output length within 1%), the committed artifact's own recomputed goldens must agree within 1e-12, and the load check allows 1e-5 on probabilities and 1e-9 on lower bounds (`python/tests/export/test_tiny.py`, `python/tests/train/test_crossfit.py`, [the load checks](architecture.md#the-artifact-format-and-content-hash)). Go still verifies the exact bytes of every artifact it is given.
- **Versions, even on one machine.** The hashed manifest includes the training metadata (commit and creation time), so two training runs on identical data get different versions. The ablation's retrained artifact with isotonic regression for every model has version `5a2cbe3e5af1` against `dea4486d7575` for the shipped one, with identical test results ([ablations](results.md#ablations)).
- **Results from a retrained artifact.** Expect the same figures up to the last digits, and note that a decision sitting on a tie or on a calibration bin edge can flip. The committed results were produced with `dea4486d7575` (flagship) and `15201d82e792` (mixed).
- **Anything timed.** Overhead, throughput and the LiteLLM comparison depend on the machine, its load, WSL2 and the share of the CPU that the load generator, the gateway and the mock take from each other. Each cell is one run. The tables are measurements of one machine on one day, not values to reproduce to the digit.
- **The local-model experiments.** Outputs and latency depend on the Ollama version, the hardware and what else uses the server. Another local application shared the Ollama server during part of the first collection ([limitations](limitations.md#evaluation-and-experiments)).
- **Interpreter and library versions.** The Windows runs used Python 3.14.7 and the Linux runs 3.14.9. The direct dependencies have the same pins in both; transitive versions were not recorded.

Sampling in the experiments is seeded (see Seeds).
