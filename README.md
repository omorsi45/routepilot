# RoutePilot

A quality-constrained, SLO-aware LLM routing gateway.

RoutePilot is an OpenAI-compatible gateway written in Go. A client sends `model: "auto"` and,
if it wants, a quality floor and a latency SLO. For each request the gateway picks the least
expensive model whose calibrated, lower-confidence-bound probability of success clears the
floor, inside the limits of the tenant's budget and rate limits, the capabilities of the
models and the health of the providers. It reserves a request's worst-case cost before it
calls a provider, fails over between deployments only before the first byte reaches the
client, and logs every decision with its probability, so a candidate policy can be evaluated
offline. The prediction models are trained offline in Python on recorded outcomes of 12 models
on 11 benchmark datasets, exported as a content-addressed artifact and evaluated in Go. No
Python runs on the request path, and the offline replay runs the same Go decision code that
serves traffic.

Everything in this repository costs nothing to run: the tests and the demo use loopback fakes,
the training data are recorded outcomes, and the live experiment used small local Ollama
models. No commercial API was called, so nothing here measures the live answer quality of a
commercial model.

The evidence is mixed and this page says so. The cost and quality targets set before the
experiments were not met on the main pool (see [Headline numbers](#headline-numbers)), and
several mechanisms work as designed but gave a weaker result than expected.

## What it does and what the evidence shows

Every result below comes from a committed run under `bench/results/`; the linked sections of
[docs/results.md](docs/results.md) and [docs/performance.md](docs/performance.md) show the
tables, the run IDs and the exact settings. "Recorded" costs and outcomes come from the
benchmark release, "fitted" prices are fitted from recorded costs, "assumed" prices are stated
assumptions, "simulated" values are modeled and never measured, and "measured" means measured
on one machine by a committed script.

| Capability | Mechanism | Backed by | Result |
|---|---|---|---|
| Calibrated success probability with a lower confidence bound | One logistic regression per model on hashed text features and 12 dense features, then Platt or isotonic calibration, then a one-sided 95% Clopper-Pearson bound per calibration bin. The quality floor is compared with the bound, not with the point estimate. | The artifact loader recomputes stored golden predictions and refuses a mismatch (`TestLoadRejectsGoldenMismatch`). Runs: [calibration study](docs/results.md#calibration-study), [bound against point estimate](docs/results.md#lower-confidence-bound-against-point-estimate). | Pooled test ECE 0.0160 over 15 equal-mass bins (target below 0.05; per model 0.026 to 0.054). Floors were met at all 12 tested floors on the test split, but the point-estimate arm met them too, so the run does not show that the bound is needed. The bounds are conservative: at floor 0.5 the router relaxed 59.4% of requests. |
| Constrained selection | `min_cost` takes the cheapest expected cost whose bound clears the floor; `dual` takes the highest probability minus lambda times cost. Both choose inside a safe set (capability, permission, budget, p95 latency). An empty safe set is relaxed (`best_effort`) or rejected with 422. | Optimizer tests in `internal/router/optimize_test.go`. Runs: [headline](docs/results.md#headline-operating-points-and-targets), [sweeps](docs/results.md#operating-point-sweeps). | At the chosen operating point the learned policy is 93.5% cheaper than the best single model and 4.55 points less successful: the targets were not met. A static rules baseline beat the best single model on both axes. |
| Budgets that cannot overshoot by construction | The worst-case cost of the first candidate (input-token bound times price, plus output cap times price) is reserved atomically before any upstream call. Every further attempt, to a pricier model or to another deployment of the same one, re-reserves first and is skipped if the reservation fails. The commit uses the usage the provider reported. | One conformance suite for the memory and Redis ledgers (`ConcurrentNeverOvershoots`), `TestFallbackReReserveFailureSkipsModel`. Run: [quota correctness](docs/results.md#quota-correctness-under-concurrent-clients). | 64 concurrent clients on a 0.05 USD budget: overshoot counter 0, spend 0.049904 USD (memory) and 0.049894 USD (Redis), and the Redis ledger and counter reconcile with a difference of 0. The bound is a proof only for text the gateway sends (see [the input-token bound](docs/architecture.md#the-input-token-bound)). |
| Off-policy evaluation from gateway logs | Exploration inside the safe set with a logged propensity. IPS, SNIPS and doubly robust estimators with bootstrap intervals in `python/routepilot_train/ope.py`. | Run: [OPE validation](docs/results.md#off-policy-evaluation-validation), 50 logging seeds against full-information replay. | The estimators work but are noisy at this sample size: effective sample size 15 to 25 on 1,735 items, RMSE of the success estimate 9.5 to 17.5 points, interval coverage 0.84 to 0.96 for success (nominal 0.95) and 0.70 to 0.92 for cost. |
| Versioned policies with a rollout lifecycle | An artifact's version is a hash of its manifest and weights. A registry holds stable, canary and shadow policies, reloads on SIGHUP or file change, rolls back through guardrails or the admin API, and `routepilot policy gate` decides a promotion with a paired doubly robust check. | Unit tests only: registry reload (`TestReloadKeepsOldOnCorruptArtifact`), guardrails with a fake clock, the gate command. | No experiment and no production traffic. The guardrail thresholds were not tuned on real canary traffic, and the gate has little power at the sample sizes above. |
| Streaming failover only before the first byte | An attempt commits when the first content reaches the client. Before that the gateway fails over to the next deployment or model; after it, an upstream error becomes one SSE error event and there is no second upstream call. | `TestMidstreamNeverRetried`, `TestStreamMidstreamErrorEvent`. Run: [failover](docs/results.md#failover-against-injected-failures). | Of 10,000 requests with injected failures before the stream, 1,749 first attempts failed and all 1,749 were answered by the other deployment. Of 2,000 streams, 411 were cut after the commit and none made a second upstream call. The failures are independent and deployment B never failed. |
| Offline results describe the shipped router | `routepilot replay` runs the same optimizer, catalog and artifact as `routepilot serve` over recorded outcomes. | Python against Go prediction comparison inside the [headline run](docs/results.md#headline-operating-points-and-targets); rules parity (12,067 items, 0 mismatches); `TestReplayOutputHasNoPromptText`. | The two implementations agree to floating-point precision on the test split (largest absolute difference 5.6e-15 over 1,713 items). The headline tables use recorded costs and no latency. |
| Low gateway overhead | A thin proxy path with an in-process decision (0.056 ms on average with the learned policy); the usage ledger and the decision log are written by background writers. | Runs: [overhead](docs/performance.md#gateway-overhead-against-the-local-mock), [LiteLLM comparison](docs/performance.md#routepilot-and-litellm-against-the-same-mock). | Measured on Linux (WSL2) over loopback against a mock with constant delays: p95 overhead of 0.29 to 0.64 ms without streaming. Accounting, the decision log and tracing were off in that run, and there is one run per cell, with no confidence intervals. |

### What the experiments did not show

- The headline targets (at least 25% cost reduction with success within 1 point of the best
  single model) were not met on the main pool, and only nearly met on the mixed sensitivity
  pool (23.9% cheaper at equal success). A static rules mapping chosen on the same train and
  calibration data met the numbers of the targets, so the learned policy has not been shown
  to beat the simplest baseline.
- The lower confidence bound was not shown to be required to meet a floor: a point-estimate
  arm met all 12 floors too. What it shows is that bounds are conservative, so a high floor
  is relaxed for most requests.
- The out-of-distribution detector is close to inert on this data (AUROC 0.557 for held-out
  datasets against the test split; it flagged 2 of 3,374 held-out items).
- Two contextual bandits did not beat the supervised policy, and one planned ablation (a
  PyTorch model on embeddings) was not run.
- The only live-model evidence comes from three small Qwen2.5 models served by Ollama, with
  accuracies of 12% to 23% on a multiple-choice sample and assumed prices. It shows the
  plumbing works, not that routing helps real workloads.
- Latency of API models in offline replay is simulated, and every table that shows it says
  so. Performance numbers are single runs on one machine with a mock provider.
- The Anthropic adapter is contract-tested against documented wire formats and has never run
  against the live API.
- RoutePilot has no prefix-cache-aware routing and no hosted service.

The full list, with the run behind each item, is in [docs/limitations.md](docs/limitations.md).

## Headline numbers

Main pool (flagship): 12 models, 11 datasets, test split of 1,735 items. Success and cost are
**recorded** values from the benchmark release; the policies are replayed through the Go
router. Intervals are 95% percentile bootstrap intervals over items. The operating point
(lambda, the floor, the RouteLLM threshold, the static rules) is chosen on the calibration
split only, and the test split is read once per row. Run
[`20261028T203202Z-028d6b4`](bench/results/headline/20261028T203202Z-028d6b4/), model artifact
`dea4486d7575`; the static rules and oracle rows are also in baselines run
[`20261028T004934Z-bec7b21`](bench/results/baselines/20261028T004934Z-bec7b21/).

| Policy | Success | Mean cost USD per item | Success vs best single, points | Cost reduction vs best single |
|---|---|---|---|---|
| Best single (always gemini-2.5-pro) | 52.7% [50.5%, 55.2%] | 0.06706 [0.06292, 0.07127] | 0.00 | 0.0% |
| Learned, dual mode, lambda 6.72 (the chosen operating point) | 48.2% [46.0%, 50.6%] | 0.004337 [0.003778, 0.0049] | -4.55 [-6.97, -2.25] | 93.5% [92.6%, 94.4%] |
| Learned, min_cost mode, floor 0.2 | 49.7% [47.3%, 52.1%] | 0.021 [0.01839, 0.02338] | -3.00 [-5.36, -0.63] | 68.7% [65.7%, 72.2%] |
| RouteLLM matrix factorization, threshold 0.7547 | 52.0% [49.7%, 54.6%] | 0.06526 [0.06112, 0.06944] | -0.69 [-1.21, -0.23] | 2.7% [2.0%, 3.5%] |
| Static rules (multiple-choice, code, math and long prompts to gpt-5, the rest to gemini-2.5-pro) | 54.6% [52.4%, 56.9%] | 0.04397 [0.04073, 0.04718] | +1.90 [+0.69, +3.17] | 34.4% [30.9%, 38.0%] |
| Oracle (reads the recorded outcomes: an upper bound, not a policy) | 77.5% [75.5%, 79.5%] | 0.009284 [0.007501, 0.01103] | +24.73 [+22.65, +26.74] | 86.2% [83.7%, 88.6%] |

**The targets were not met.** The target was a cost reduction of at least 25% with success
within 1 point of the best single model. The learned dual policy meets the cost half (93.5%)
and misses the quality half (-4.55 points). The learned min_cost policy does the same (68.7%
cheaper, -3.00 points), and the RouteLLM baseline meets the quality half (-0.69 points) and
misses the cost half (2.7%). The run states "targets NOT met" for all three, and for a
cross-fitted variant of the dual policy (97.1% cheaper, -5.07 points). The static rules row,
chosen on the same train and calibration data, meets the numbers of both targets, so on this
data the simple baseline is the better trade and the learned policy has not been shown to
beat it.

Two things the run does show. At lambda 0 (a reference, not an operating point) the learned
dual policy reaches 56.0% [53.7%, 58.3%] success at 44.3% lower cost, which suggests the
predictor carries signal. And the gap appears between calibration and test: on the
calibration split, where lambda is tuned, the tuned policy is 0.9 points below the best
single model (49.2% against 50.1%), on the test split 4.55 points. The run labels the
calibration figure in-sample and optimistic, because the calibrators were fitted on that
split. The losses concentrate in a few datasets: the largest drops against the best single
model are arc-agi (-20.5 points), hle (-12.5), gpqa (-12.1) and livecodebench (-10.7), from
the [failure analysis](docs/results.md#failure-analysis) run
[`20261029T015809Z-00deb40`](bench/results/failure_analysis/20261029T015809Z-00deb40/).

Mixed pool (sensitivity only: the 12 models plus four small models whose prices are
**fitted** from recorded costs; 361 test items). Run
[`20261028T203239Z-028d6b4`](bench/results/headline/20261028T203239Z-028d6b4/). The learned
dual policy (lambda 0.7028) keeps the best single model's success, 83.7% [80.1%, 87.3%]
against 83.7% [79.8%, 87.3%] for gpt-5 (difference 0.00 points [-3.32, +3.05]), at a cost
reduction of 23.9% [17.4%, 30.7%]. That is 1.1 points short of the cost target. The static
rules mapping on this pool is the best single model, so it saves nothing.

Calibration and systems measurements:

| What | Result | Run |
|---|---|---|
| Pooled test ECE of the shipped flagship artifact (15 equal-mass bins) | 0.0160 (Platt scaling for every model gives 0.0115; the fixed rule keeps isotonic) | [`20261028T203251Z-028d6b4`](bench/results/calibration/20261028T203251Z-028d6b4/) |
| Gateway overhead, p95 of the gateway minus the direct path, no streaming, 100 / 500 / 1000 / 2000 requests per second (**measured**, Linux under WSL2, loopback, mock with a constant 20 ms time to first token) | 0.64 / 0.29 / 0.38 / 0.38 ms. The 15 ms target was met in all 8 cells. One cell has a worse tail: streaming at 2000 requests per second, p99 +179 ms. | [`20261028T193224Z-9ac474d`](bench/results/overhead/20261028T193224Z-9ac474d/) |
| Learned policy at 1000 requests per second, p95 over the direct path | 0.32 ms; the router's decision took 0.056 ms on average | same run |
| LiteLLM 1.106.6 (8 workers) against the same mock, p95 over the direct path, no streaming, at 100 / 500 / 1000 / 2000 requests per second | LiteLLM +10.27 / +68.64 / +69.12 / +2284.51 ms (saturated at 2000); RoutePilot +0.29 / +0.35 / +0.29 / +0.58 ms. With streaming, LiteLLM dropped arrivals and returned errors from 500 requests per second. One configuration, no master key set (which can only favor LiteLLM). | [`20261028T234116Z-ee7619a`](bench/results/litellm/20261028T234116Z-ee7619a/) |
| Failover, 10,000 requests with injected failures before the stream | 1,749 first attempts failed, 1,749 recovered (100.00%), 0 requests failed; target 99% | [`20261029T010235Z-f5ff91d`](bench/results/failover/20261029T010235Z-f5ff91d/) |
| Quota, 64 clients, budget 0.05 USD | spend 0.049904 USD (memory) and 0.049894 USD (Redis), overshoot counter 0, 9 of 9 and 10 of 10 checks passed | [`20261029T010202Z-f5ff91d`](bench/results/quota_correctness/20261029T010202Z-f5ff91d/) |
| Clock step of the timing host | 30 ns on Linux, about 0.4 to 0.5 ms on Windows, which is why every timing run is on Linux | [`20261029T074240Z-e12fd9f`](bench/results/clock/20261029T074240Z-e12fd9f/), [`20261029T074230Z-e12fd9f`](bench/results/clock/20261029T074230Z-e12fd9f/) |

Live check on local models (three Qwen2.5 models through Ollama, **measured** latency through
the gateway, **assumed** prices proportional to parameter count; 148 test items, so wide
intervals). With a quality floor of 0.166 the learned policy reached 14.9% [9.5%, 20.3%]
success against 18.9% [12.8%, 25.0%] for always using the largest model, at 9.808e-06 against
1.836e-05 USD per item. It cut cost and median latency and did not improve quality. Runs
[`20261029T043015Z-6ecd7f2`](bench/results/live_train/20261029T043015Z-6ecd7f2/) and
[`20261029T043138Z-6ecd7f2`](bench/results/live_demo/20261029T043138Z-6ecd7f2/).

## Quickstart

You need Go 1.27.2 (the version in `go.mod`), GNU make, bash and curl. On Windows, use Git
Bash. These steps use `examples/config.demo.yaml`: the learned `live-ollama` policy, committed
under `policies/`, routing over `rp-mock`, the deterministic mock provider. They need no
dataset, Docker, Ollama or API key. The routing decisions come from a policy trained on real
local runs; the answers come from the mock, so the demo shows routing and not answer quality.
The prices in the config are an assumption, not the price of anything.

(`examples/config.local.yaml` also routes `auto`, but with the `cheapest` policy, which picks
the free Ollama model first. It needs Ollama running on 127.0.0.1:11434, so it is not used
here.)

1. Build the gateway and the mock into `bin/`:

   ```bash
   make build
   go build -o bin/rp-mock ./cmd/rp-mock
   ```

2. Terminal 1: start the mock on port 18781 (the port the demo config expects):

   ```bash
   ./bin/rp-mock --listen 127.0.0.1:18781 --ttft-median-ms 40 --ttft-sigma 0.3 \
     --tpot-median-ms 5 --tokens 64
   ```

3. Terminal 2: start the gateway on port 18780:

   ```bash
   ./bin/routepilot serve --config examples/config.demo.yaml
   ```

4. Terminal 3: check that it is ready, then send a request with the demo key
   (`demo-key-local`, a fake local key whose SHA-256 is in the config):

   ```bash
   key=demo-key-local
   curl -sS --max-time 5 http://127.0.0.1:18780/readyz

   curl -sS -i --max-time 30 http://127.0.0.1:18780/v1/chat/completions \
     -H "Authorization: Bearer $key" \
     -H 'Content-Type: application/json' \
     -d '{"model":"auto","messages":[{"role":"user","content":"What is 17 times 23?"}]}'
   ```

   The response is a normal OpenAI chat completion. Its headers say how it was routed:
   `x-routepilot-model`, `x-routepilot-deployment`, `x-routepilot-policy-version`,
   `x-routepilot-decision-id`, `x-routepilot-estimated-cost-usd`, and `x-routepilot-relaxed`
   when the policy gave up a constraint.

5. Ask for a quality floor with the `routepilot` extension object, here with the OpenAI
   Python SDK. The snippet was written against `openai` 3.41.0, the PyPI release of
   2026-10-28 (Python 3.10 or newer). Install it in a throwaway virtual environment outside
   the repository:

   ```bash
   python -m venv /tmp/rp-sdk
   . /tmp/rp-sdk/bin/activate        # Git Bash on Windows: . /tmp/rp-sdk/Scripts/activate
   pip install openai==3.41.0
   python - <<'PY'
   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:18780/v1", api_key="demo-key-local")
   raw = client.chat.completions.with_raw_response.create(
       model="auto",
       messages=[{"role": "user", "content": "What is 17 times 23?"}],
       extra_body={"routepilot": {"min_quality": 0.3}},
   )
   completion = raw.parse()
   print(completion.choices[0].message.content)
   for name in ("x-routepilot-model", "x-routepilot-policy-version", "x-routepilot-relaxed"):
       print(name, raw.headers.get(name))
   PY
   ```

   A floor of 0.3 is above the 90th percentile of the best lower bound on this policy's
   calibration items (0.221, see the live replay in [docs/results.md](docs/results.md)), so
   most prompts come back relaxed: look for `x-routepilot-relaxed: quality`. Name a model
   (`qwen2.5-3b`) instead of `auto` to bypass the policy.

6. Stop both processes with Ctrl-C.

Per-request controls go in the body under `routepilot` (`min_quality` between 0 and 1,
`slo_ttft_ms`, `slo_e2e_ms`, `timeout_ms`) or in the headers `x-routepilot-min-quality`,
`x-routepilot-slo-ttft-ms`, `x-routepilot-slo-e2e-ms` and `x-routepilot-timeout-ms`. The body
wins field by field. `GET /v1/models` lists `auto` and the models the tenant may use.
Errors use the OpenAI error shape.

## Demo

```bash
make demo
```

`scripts/demo.sh` builds both binaries, starts the mock and the gateway on loopback with the
same demo config, and shows in order: an `auto` request with its routing headers; a request
with `min_quality` 0.99, which no model clears, so the policy relaxes it; a pinned model; a
failover from a deployment that answers HTTP 500 to its second deployment; a tenant whose
0.00001 USD budget runs out and gets a 429; and the Prometheus series that recorded all of
it. It needs Go and curl, stops both processes on exit, and writes its logs to
`bench/scratch/demo/`.

## Compose stack

`deploy/docker-compose.yml` starts the gateway with the same live-ollama policy over the mock
provider, Redis for the rate-limit counters and budgets, PostgreSQL for the usage ledger,
Prometheus, Grafana with a provisioned RoutePilot dashboard, and an OpenTelemetry collector.

```bash
docker compose -f deploy/docker-compose.yml up -d --build
# gateway http://127.0.0.1:8080, Prometheus :9091, Grafana :3000 (dashboard "RoutePilot")
docker compose -f deploy/docker-compose.yml down -v
```

`bash scripts/compose-smoke.sh` does the whole round trip (up, an `auto` request with the demo
key, a check of `/metrics`, down), and the integration workflow in `.github/workflows` runs
it. Every host port binds `127.0.0.1`. The stack is a local demo and not a deployment recipe:
Grafana allows anonymous viewing, and the Redis and PostgreSQL passwords and the demo key are
throwaway values.

## Security notes

- The gateway speaks plaintext HTTP. The examples bind `127.0.0.1`, and so does a config
  without `server.listen` (the default is `127.0.0.1:8080`); keep it that way, or put TLS in
  front before binding another interface.
- `GET /metrics` is unauthenticated and served on the main listener, like `/healthz` and
  `/readyz`. Its series hold identifiers (tenant IDs, model and deployment names, policy
  versions), counts, durations and costs, and never a prompt, a message, an API key or a key
  hash. Tenant labels are bounded to the configured tenants. It is still an internal-network
  endpoint: scrape it from inside the network, and block the path at any edge that faces the
  internet.
- Tenant API keys are stored only as SHA-256 hashes in the config and compared in constant
  time. Provider keys are read from the environment variables the config names
  (`api_key_env`), never from config values. Compute a hash with
  `printf %s your-key | shasum -a 256`.
- The admin routes (`GET /admin/policies`, `POST /admin/policies/rollback`) exist only when
  `server.admin_key_sha256` is set, and then need that key as a bearer token. Without it they
  answer like any unknown path. A rollback made through them lives in memory and the next
  reload restores what the config file says. The admin routes have no rate limit.
  `/v1/feedback` is charged against the tenant's request rate (`rpm`) like a chat request,
  and needs a valid tenant key.
- No prompt text or key reaches a log, trace, metric or decision log: the decision log holds
  a 16-hex prefix of the prompt hash and the prompt's byte count. A redaction test plants
  sentinel strings and fails if any appears (`TestNoSecretsOrPromptsAnywhere`).
- Redis budgets are exact only below 9,000 USD (Lua numbers are doubles), and larger
  `budget_usd` values are rejected with the Redis backend.
- The compose stack is a local demo (see above).

## Make targets

`make help` prints this list:

```text
  bench-gate             replay the flagship test split with the learned dual policy and compare with bench/gates/baseline.json (needs make data and make train)
  bench-smoke            build the binary and run the replay gate on the synthetic fixture (no dataset)
  build                  build the CLI into bin/routepilot
  check                  gofmt check, vet and build, the way CI runs them
  clean                  remove build output
  data                   download and verify the dataset, ingest it, then build splits, pools and replay exports in data/derived
  demo                   build and run the demo: routing headers, relaxation, a pinned model, a failover, a budget 429 and metrics (no dataset, Docker or keys)
  fmt                    format all Go source
  help                   show this help
  py-check               ruff lint, ruff format check and pytest for the Python project
  py-setup               create .venv and install the pinned Python dependencies
  test-race              run the suite under the race detector
  test                   run the hermetic test suite (loopback only, no network)
  tidy-check             go.mod and go.sum are what go mod tidy writes, the way CI checks it
  train                  train the flagship policy into policies/flagship/<version> and record the run in bench/results/train (needs make data)
  verify                 everything the CI lanes run
```

`make verify` is what CI runs: gofmt, vet, build, the tests, the race detector, the `go mod tidy`
check, ruff and pytest for the Python project, and the benchmark smoke. The default tests are
hermetic: loopback only, no network, no API keys, no Docker. The Redis and PostgreSQL suites run
when `ROUTEPILOT_TEST_REDIS_URL` and `ROUTEPILOT_TEST_POSTGRES_DSN` are set, which the
integration workflow does.

To rebuild the results, run `make py-setup`, `make data` (a 1.28 GB download, checked against
its SHA-256), `make train` and `make bench-gate`, then the scripts under `bench/scripts/`;
[docs/reproducibility.md](docs/reproducibility.md) has the order, the seeds and the run times.

## Repository layout

```text
cmd/routepilot/     the CLI: serve, replay, policy (verify, inspect, gate), reconcile, version
cmd/rp-mock/        deterministic OpenAI-compatible mock provider with failure injection
cmd/rp-load/        open-loop, coordinated-omission-safe load generator
internal/           private packages, one responsibility each:
  openai/             wire types, validation, SSE codec, error shapes
  provider/           provider interface, openaicompat and anthropic adapters
  catalog/            models, deployments, capabilities, prices, the input-token bound
  config/             YAML config loading and validation
  features/           feature spec v1 (hashed and dense), identical to the Python side
  policy/             artifact loading and checks, predictor, registry, guardrails
  router/             feasibility, the constrained optimizer, static policies
  latency/            online latency quantiles and the union-bound end-to-end quantile
  quota/              rate limiters and the budget ledger (memory and Redis)
  accounting/         usage ledger (SQLite, PostgreSQL) with a batching writer
  reliability/        circuit breakers, health, retry backoff
  gateway/            HTTP server, handlers, admission, budget, streaming proxy, admin
  decisionlog/        append-only JSONL decision log
  telemetry/          Prometheus metrics and OpenTelemetry tracing
  replay/             offline replay of recorded outcomes through the router
  loadgen/, mock/, money/, version/
python/             the offline pipeline (routepilot_train): data, features, train, export,
                    evaluate, OPE, bandits; tests in python/tests
bench/              experiment scripts, the configs they start the gateway with, latency
                    profiles, gate baselines and committed results (bench/results)
deploy/             Dockerfile, compose stack, Prometheus, Grafana and collector configs
examples/           config.demo.yaml, config.local.yaml
policies/           the committed live-ollama demo artifact (trained flagship and mixed
                    artifacts stay local)
scripts/            demo.sh, compose-smoke.sh
testdata/           golden feature vectors, parity cases, small configs and fixtures
data/               SOURCES.md; the dataset and derived tables are fetched, never committed
docs/               architecture, results, performance, limitations, comparison, reproducibility
.github/workflows/  ci.yml (hermetic lane) and integration.yml (Redis, PostgreSQL, image build,
                    compose smoke test)
```

## Documentation

- [docs/architecture.md](docs/architecture.md): the request path and the offline loop, the
  request lifecycle as built, the constrained program, the artifact format, the policy
  lifecycle, budgets and accounting, and the input-token bound.
- [docs/results.md](docs/results.md): every experiment, generated from the committed runs.
- [docs/performance.md](docs/performance.md): gateway overhead, the LiteLLM comparison and
  the clock-resolution runs, generated from the committed runs.
- [docs/limitations.md](docs/limitations.md): what RoutePilot does not do and what the
  experiments did not show.
- [docs/comparison.md](docs/comparison.md): how RoutePilot compares with existing gateways
  and routers.
- [docs/reproducibility.md](docs/reproducibility.md): hardware, commands and expected run
  times for every experiment.
- [docs/adr/](docs/adr/): the architecture decision records.

## License

Apache-2.0. See `LICENSE`.
