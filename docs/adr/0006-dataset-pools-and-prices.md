# ADR 0006: Dataset pools, splits and prices

Status: Accepted, 2026-10-27.

## Context

The project costs nothing to run, so no commercial API is called. Training and evaluating a router needs the outcome of many models on the same prompts, with token counts and costs. The LLMRouterBench release records exactly that: the score, token counts and cost of each model on each benchmark prompt (`data/SOURCES.md` has the file, its size, its SHA-256 and its license). It records no latency. The release mixes complete flagship models with small models, and some of its dataset groups overlap or are not single-turn.

## Decision

- **Flagship pool (primary).** The 12 models with a non-zero recorded cost on every kept dataset: claude-sonnet-4, deepseek-r1-0528, deepseek-v3-0324, gemini-2.5-flash, gemini-2.5-pro, glm-4.6, gpt-5, gpt-5-chat, intern-s1, kimi-k2-0905, qwen3-235b-a22b-2507 and qwen3-235b-a22b-thinking-2507. A router in the release (openrouter) is not a model and is excluded.
- **Kept datasets (11).** aime, arc-agi, arenahard_coding, arenahard_creative_writing, arenahard_math, gpqa, hle, livecodebench, livemathbench, mmlupro and simpleqa, each on one canonical split. Excluded, with the reason in the ingest report: the union `arenahard` split (it would duplicate the three category subsets), the `subset_500` and `test_1000` splits (subsets of kept splits), swe-bench and tau2 (multi-step or multi-turn tasks), and the datasets that only small models ran.
- **Items.** `item_id` is `<dataset>/<index>`. A prompt whose SHA-256 appears in two kept items is kept once. An item is kept only if every model of the pool has a record for it: 12,067 items.
- **Splits.** Deterministic, from the first 16 hex digits of `SHA-256(item_id + "routepilot-split-v1")`: below 0.6 train, below 0.8 calibration, the rest test. Three whole datasets, at most one per domain, are held out of all fitting and tuning by a rule fixed before any model was trained, and appear only in the shift and out-of-distribution experiments: arenahard_coding, livemathbench and mmlupro. Flagship counts: train 5,185, calibration 1,773, test 1,735, held out 3,374.
- **Mixed pool (sensitivity only).** The flagship models plus four small ones (Qwen3-8B, DeepSeek-R1-0528-Qwen3-8B, NVIDIA-Nemotron-Nano-9B-v2, Llama-3.1-Nemotron-Nano-8B-v1), on the items where all 16 models have records: 3,182 items over 8 datasets, 361 of them in the test split.
- **Costs and prices.** Every reported cost is the recorded cost. The router needs a price per token to estimate cost, and the release gives none, so a price per model is fitted by non-negative least squares of recorded cost on prompt and completion tokens, over the rows with a non-zero recorded cost. The fit's RMSE and R^2 are reported next to each price and a fit below R^2 0.9 is flagged. The small models of the mixed pool are priced the same way, from their own recorded costs. The first design assumed a price table for them, which recorded costs made unnecessary.
- **Odd rows are kept as recorded.** Rows with zero recorded cost (mostly failed calls the benchmark was not billed for) stay at zero, and 10 gpt-5 rows with a negative completion token count stay as recorded, with the length model leaving them out of its fit.
- **Latency is simulated.** The release has no latency, so the offline replay uses profiles built from public pages, and every table that shows them says "simulated".
- The tarball and everything derived from it stay out of version control. `make data` checks the size and SHA-256 before ingesting ([reproducibility](../reproducibility.md#verify-the-dataset)).

## Consequences

- Results hold for this benchmark: single-turn question sets that are hard for the models. Best Single (gemini-2.5-pro) succeeds on 52.7% of the flagship test items and the weakest model of the pool (intern-s1) on 21.7%. They say little about traffic that looks different, and nothing about the live answer quality of a commercial model.
- The held-out rule was fixed in advance and kept even though it holds out mmlupro, which is 3,000 of the 12,067 pool items. That leaves fewer training items than a different rule would.
- Fitted prices are not list prices. Two flagship fits are poor, gpt-5-chat (R^2 0.89) and qwen3-235b-a22b-thinking-2507 (R^2 0.65), probably because billed reasoning tokens are not in the recorded completion tokens. The router's expected cost for these two is less accurate. The mixed pool has the same two weak fits (R^2 0.63 and 0.52). Reported costs are not affected ([data](../results.md#data-pools-splits-and-best-single)).
- Keeping zero-cost rows at zero understates the cost of models with many failed calls, by an amount the data cannot tell.
- The mixed pool is small and its test split has 361 items, so its intervals are wide. It is reported as a sensitivity check and never as the main result.
- Because latency is simulated, no offline number about latency or SLO violation is a measurement. The gateway's overhead is measured separately against a mock ([performance](../performance.md)).
- The only evidence with real outputs and measured latency comes from three small local models, with assumed prices, on two datasets ([live collection](../results.md#live-collection-on-local-models)).
