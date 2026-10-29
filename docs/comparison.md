# Comparison with existing gateways and routers

This page compares RoutePilot with the gateways and research routers it was designed against. It uses the same six axes throughout: routing signals, quality prediction, budget enforcement, off-policy evaluation, rollout lifecycle and streaming failover. It also says where RoutePilot is weaker.

Checked on 2026-10-29. Each public page named below was read again on that date, one page per request, and the statement here is what the page said then. A page that failed to load, or that loaded only in part, is marked "not re-checked". Vendor and author numbers are labeled as claims: they are not reproduced here and their methods are theirs. RoutePilot's own numbers come from [results.md](results.md) and [performance.md](performance.md), and every one of them is a single run on one machine.

## The six axes at a glance

"Not checked" means the page read does not say, which is not the same as "absent".

| System | Routing signals | Quality prediction | Budget enforcement | Off-policy evaluation | Rollout lifecycle | Streaming failover |
|---|---|---|---|---|---|---|
| RoutePilot | Calibrated success lower bound, cost, latency quantile, capability, permission, health, remaining budget | Per request and model, calibrated, with a lower confidence bound ([ADR 0002](adr/0002-lcb-admission.md)) | Worst-case reservation before the call, commit from usage ([ADR 0003](adr/0003-worst-case-budget-reservation.md)) | Logged propensities, IPS, SNIPS and doubly robust estimators; noisy at the sample sizes tested | Artifact registry with stable, canary and shadow slots, guardrails, a promotion gate; unit-tested only | Only before the first byte; after it, one error event and no second upstream call |
| LiteLLM | Shuffle by rpm and tpm weights, latency, least busy, lowest cost; complexity tiers | Adaptive router (beta): a quality mean per request type and model, from observed signals | Budgets exist; concurrent requests reported to bypass two of them | Shadow evaluation with an LLM judge on sampled live traffic | Not checked | Continues a failed stream on a fallback model by re-prompting with the partial text |
| Portkey gateway | Not re-checked | Not re-checked | Not re-checked | Not re-checked | Not re-checked | Not re-checked |
| vLLM Semantic Router | Quality index, observed latency percentile, cost, in-flight load | A versioned quality index with a point `min_score` floor | Cost ceiling per million tokens; no budget ledger on the page read | Not checked | Not checked | Not checked |
| llm-d | Predicted TTFT and TPOT per pod, queue depth, KV cache use, prefix cache match | None (it predicts latency, inside a pool of one model) | Not on the page read | Not on the page read | Not checked | Not checked |
| SGLang Model Gateway | Random, round robin, cache-aware, power of two, bucket | None on the page read | Token-bucket rate limits only | Not on the page read | Not checked | Retries and circuit breakers; mid-stream behavior not checked |
| Envoy AI Gateway | Model name and headers (a provider fallback page exists, not read) | None on the page read | Token rate limits charged after the response completes | Not on the page read | Not checked | Not checked |
| Bifrost | Weights, routing rules, budget and rate limit exclusion | None on the page read | Hierarchical budgets, checked before and deducted after | Not on the page read | Not checked | Not checked |

## Production gateways and serving routers

### LiteLLM

Re-checked: the [routing](https://docs.litellm.ai/docs/routing), [auto routing](https://docs.litellm.ai/docs/proxy/auto_routing), [adaptive router](https://docs.litellm.ai/docs/adaptive_router), [evaluate on your traffic](https://docs.litellm.ai/docs/auto_router/evaluate) and [benchmarks](https://docs.litellm.ai/docs/benchmarks) pages (each loaded only in part, the first 8,000 characters), and four issues.

- Routing strategies are the usual ones: weighted shuffle by `rpm` and `tpm` (the default and the recommended one), rate-limit aware, latency based, least busy, lowest cost and custom, with cooldowns, retries and fallbacks.
- Auto routing (beta, since v1.94.0) maps each request to a tier (SIMPLE, MEDIUM, COMPLEX, REASONING) with heuristics, an LLM classifier, keyword rules or a custom plugin, and a tier may be a single model, a random pool or a Thompson-sampled pool. The page says config keys can still change between releases.
- The adaptive router (beta) needs a Postgres database, classifies a request into one of 7 types, keeps a quality mean per type and model, and moves it with signals such as a satisfaction turn. The operator sets quality and cost weights (default 0.7 and 0.3). It is a point estimate per cell, with no per-request probability and no bound. The page was cut before any statement about decay or latency, so those points are not re-checked.
- Shadow evaluation duplicates a sample of live traffic through the router, hides the shadow response from the client and has an LLM judge compare the two answers. That evaluates a router on the operator's traffic, with a model as the judge. It is not off-policy evaluation from logged propensities. Vendor claim: 88.1% of 143 judged turns matched or beat the current model, at 1.55 USD of judge spend, on the vendor's own traffic.
- Budgets: [issue 34732](https://github.com/BerriAI/litellm/issues/34732) (opened 2026-07-26, open when checked) reports that concurrent requests both pass a session budget before spend is recorded. [Issue 40095](https://github.com/BerriAI/litellm/issues/40095) (opened 2026-09-07, open when checked) reports that 8 concurrent first requests for an unknown end user all reached the model, with persisted spend 264,000 times a budget of 0.000000001 USD, reproduced on LiteLLM 1.95.0. The text of the second issue refers to a budget reservation setting (`disable_budget_reservation`), so LiteLLM has some reservation logic that these pages do not describe and that was not read. This page does not claim that LiteLLM checks only recorded spend.
- Streaming: according to issue 31067, the router continues a stream that failed midway by sending the fallback model a continuation prompt that contains the partial text. [Issue 31067](https://github.com/BerriAI/litellm/issues/31067) reported that this makes a fallback model answer in prose instead of calling tools, and [issue 33273](https://github.com/BerriAI/litellm/issues/33273) reported that the tokens of the partial stream drop out of the combined usage. Both were closed as not planned when checked, so neither is known to be fixed. RoutePilot never restarts a stream: after the first byte there is one error event.
- Performance, vendor claim: the benchmarks page reports gateway request latency p50, p95 and p99 of 30.581, 54.029 and 91.645 ms at 3,000 requests per second on 33 pods (4 workers and 4 vCPUs each), with prompts of 50K to 100K tokens, half of them streaming, an in-process mock model, a public load balancer, PostgreSQL and Redis. It is not comparable with a single process on loopback. Measured here, one configuration (LiteLLM 1.106.6, 8 workers, no master key, same mock): the p95 added over the direct path was 10.27 ms at 100 requests per second and 68.64, 69.12 and 2,284.51 ms at 500, 1,000 and 2,000 (saturated), against 0.29, 0.35, 0.29 and 0.58 ms for RoutePilot, without streaming ([performance](performance.md#routepilot-and-litellm-against-the-same-mock)). That compares a Python proxy with a Go proxy on one shared machine, and says nothing about LiteLLM's multi-pod deployment.

### Portkey gateway

Re-checked: only the repository front page loaded, and it holds the project's own description (a fast gateway with integrated guardrails that routes to more than 1,600 models, a vendor claim). The README body did not load. An earlier reading of the project's documentation listed conditional routing, fallbacks, weighted load balancing, retries and a circuit breaker, no quality prediction, and a latency claim of under 1 ms given without a method. None of that is re-checked.

### vLLM Semantic Router

Re-checked: the [multi-factor selection](https://vllm-sr.ai/docs/tutorials/algorithm/selection/multi-factor) page (docs v0.4). It is the closest design.

- Candidates are compared on quality, latency, cost and load, after hard eligibility rules, with a weighted or lexicographic objective.
- Quality comes from a versioned index (built-in or operator-defined), resolved per reasoning effort. `min_score` is a hard floor on the index's own scale and excludes models below it before the objective runs. It is a point value on a static index, with no calibrated per-request probability or bound.
- Latency is the observed TTFT or TPOT at a configurable percentile (default 95). SLO ceilings (`max_ttft_ms`, `max_tpot_ms`, `max_cost_per_1m`, `max_inflight`) are enforced only when the corresponding observation exists. Load and latency observations are local to each Router process, so replicas may choose differently, as the page states.
- Cost uses the request's pre-routing token estimate and `max_output_tokens` when the caller sets it. The page describes no budget ledger.
- New since an earlier reading: the documentation lists learned selectors (KNN, KMeans, SVM, MLP, Router DC, AutoMix, Hybrid) next to this one. Only the multi-factor page was read, so what those selectors output, and whether any carries a bound, is not checked.

### llm-d

Re-checked: the [latency predictor](https://llm-d.ai/docs/0.8/architecture/advanced/latency-predictor) page of v0.8. The site now shows v1.0 as the latest version and calls v0.8 no longer maintained, and the v1.0 page was not read.

- The endpoint picker asks an online-trained XGBoost model for TTFT and TPOT for every candidate pod, conditioned on pod state (KV cache utilization, queue depth, prefix cache match), and can gate on per-request SLOs. If the predictor is unreachable, scoring falls back to a heuristic. The page reports about 5% mean absolute percentage error across its benchmark runs (a claim).
- The predictor assumes a homogeneous pool: every pod shares the GPU type, model weights and serving configuration. Heterogeneous pools are not modeled. It places a request among replicas of one model and does not choose between models.

### SGLang Model Gateway

Re-checked: the [sglang-router](https://pypi.org/project/sglang-router/) page on PyPI (version 0.3.2, in Rust). Load balancing policies are `random`, `round_robin`, `cache_aware`, `power_of_two` and `bucket`, with prefill and decode disaggregation, retries with jitter, per-worker circuit breakers, token-bucket rate limiting and queuing, and OpenAI-compatible endpoints well beyond chat (responses, embeddings, rerank and others). The page describes no quality prediction.

### Envoy AI Gateway

Re-checked: the [usage-based rate limiting](https://aigateway.envoyproxy.io/docs/capabilities/traffic/usage-based-ratelimiting) page (docs 1.2). The site banner now says the project was formerly called Envoy AI Gateway. Token usage is charged after the response completes. A request is checked against usage already charged, and the page says an admitted stream is not interrupted if it pushes the count over the limit: with a 1,000-token hourly limit, a request admitted on an empty bucket can stream 1,200 tokens, and later requests are limited until the window recovers. A separate quota policy page exists and was not read.

### Bifrost

Re-checked: the [budget and limits](https://docs.getbifrost.ai/features/governance/budget-and-limits) page. Budgets are hierarchical (customer, team, virtual key, provider config). Every applicable budget is checked, and the cost is deducted from all of them afterwards. The page's own example has a provider budget of 4 of 5 USD remaining, a request that costs 2 USD, a provider budget of 6 of 5 USD afterwards, and the next request blocked. The page also says the open-source edition keeps its state in memory, that running several open-source nodes against one database is not supported, and that one instance can handle "around 3,000 to 5,000 RPS" (a claim). The claim of 11 microseconds of overhead at 5,000 requests per second, made on a benchmarks page of the vendor, was not re-checked.

### Not re-checked

Kong AI Gateway, the NVIDIA LLM Router blueprint, OpenRouter Auto Router, Not Diamond and Arch-Router were not re-read. LiteLLM's Rust gateway and its cache-aware routing pages appear in the navigation of the pages above and were not read.

## Research routers and evaluation work

- RouteLLM ([repository](https://github.com/lm-sys/RouteLLM), re-read). A framework for serving and evaluating routers, with a strong model, a weak model and a cost threshold per request. Authors' claims: costs reduced by up to 85% while keeping 95% of GPT-4 performance on MT Bench, and the same performance as commercial routers at more than 40% lower cost. The paper's own figures and the way it measures cost savings were not re-checked. RoutePilot reimplements the matrix-factorization router as a baseline and it saved 2.7% of the cost of Best Single on the flagship pool ([results](results.md#headline-operating-points-and-targets)).
- [LLMRouterBench](https://arxiv.org/abs/2601.07206) (abstract re-read). Over 400K instances, 21 datasets, 33 models and 10 baselines. The authors find that many routing methods perform similarly under unified evaluation, that several recent approaches, commercial routers included, fail to reliably beat a simple baseline, and that a large gap to the Oracle remains. RoutePilot's data come from this release ([ADR 0006](adr/0006-dataset-pools-and-prices.md)), and its result is of the same kind: a static rules mapping beat the learned policy. The percentages quoted in earlier readings of the paper are in its body and were not re-checked.
- [False Floors](https://arxiv.org/abs/2610.01535) (abstract re-read). The paper is about safety routers, not cost routers. It argues that choosing the best-single comparator on the evaluation data biases the result, more so under distribution shift, and recommends evaluating against a baseline chosen without the test labels. RoutePilot chooses every baseline on the train and calibration splits ([ADR 0005](adr/0005-static-rules-baseline.md)). The paper's results concern safety benchmarks and are not evidence about cost routing.
- [PROTEUS](https://arxiv.org/abs/2601.19402) (abstract re-read). A router that takes an accuracy target as a runtime input, using Lagrangian dual control inside a learned policy. Authors' claims, on RouterBench and SPROUT: floor compliance, a correlation of 0.97 to 0.98 between target and response, and up to 89.8% cost savings against the best fixed model. The abstract states no finite-sample guarantee and no latency term, and the body was not read.
- CARROT and SPROUT (arXiv 2502.03261) and the other routing papers (FrugalGPT, AutoMix, Hybrid LLM, RouterDC, GraphRouter, EmbedLLM, MixLLM, PILOT, UniRoute and others) were not re-checked.

## What RoutePilot adds

On the axes above, and only against what was read, RoutePilot combines things the pages describe separately: a floor compared with a lower confidence bound and not a point value, a constrained choice that includes a latency quantile and the remaining budget, a budget reserved before the call, logged propensities for offline evaluation, and a streaming rule with no restart. The evidence for each is in the [README](../README.md#what-it-does-and-what-the-evidence-shows) with its limits, and several are weak: the bound was not shown to be needed to meet a floor, the off-policy estimates are noisy, the lifecycle has unit tests and no production traffic, and the headline cost and quality targets were not met.

## Where RoutePilot is weaker

- No prefix-cache or KV-cache awareness. llm-d and the SGLang gateway route on cache state and load inside a serving pool, and LiteLLM has cache routing pages. RoutePilot sees providers as OpenAI-compatible endpoints and has no view of their internals.
- No hosted service and no high-availability story. It is one binary. Budgets and rate limits can be shared through Redis and the usage ledger through PostgreSQL, but hot reload re-reads only the `policies` section, and an admin rollback lives in memory until the next reload.
- No commercial provider has been called. The Anthropic adapter is contract-tested against documented wire formats only and has never run against the live API. The only real model outputs come from three small local models, with assumed prices.
- A narrow API. It serves `/v1/chat/completions`, `/v1/models` and `/v1/feedback`. There are no embeddings, images, audio, responses or tool-hosting endpoints, no caching and no content guardrails, where LiteLLM and the SGLang gateway serve much more.
- The quality model needs recorded outcomes for every model on representative prompts, and gets no help from the models' public benchmark scores. A gateway with a hand-defined quality index or complexity tiers needs no training data and works on the first day. The predictor here is a linear model on hashed n-grams, trained offline from one benchmark release, with no online learning, whereas LiteLLM's adaptive router updates from live signals.
- The learned policy did not beat a static rules mapping on the flagship pool, two contextual bandits did not beat the supervised policy, the out-of-distribution detector is close to inert, and the lower bounds are conservative: a floor of 0.50 relaxed 59.4% of requests on the test split. See [limitations](limitations.md).
- Latency of API models is simulated offline, and every performance number is a single run on one machine against a mock, with the load generator sharing the machine. The comparison with LiteLLM is one configuration.
- Operations are thin: plaintext HTTP, tenant API keys only, an unauthenticated `/metrics` endpoint, breakers shared across tenants, and guardrail thresholds that were not tuned on live traffic.
