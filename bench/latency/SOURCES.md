# Latency profile sources

`profiles.yaml` holds simulated latency priors for the 12 flagship pool models. Every number
derived from it is a simulated figure: a public page's median for a model, with an assumed
spread, and not a measurement of RoutePilot or of any provider.

All figures were read from public Artificial Analysis model pages on 2026-10-28 (UTC), exactly as
the page showed them. The "Speed" box of each page is "Output tokens per second"; the time to
first token is the "time to first token (TTFT)" sentence of the page's FAQ ("What is the latency
of ...?"), in seconds. Every page carried the notice "This model is deprecated. We only continue
performance benchmarking for the default 10k input token workload."

| Model | Variant on the page | URL | Output speed as shown | TTFT as shown | Retrieved |
|---|---|---|---|---|---|
| claude-sonnet-4 | Claude 4 Sonnet (Non-reasoning) | https://artificialanalysis.ai/models/claude-4-sonnet | N/A | not shown | 2026-10-28 |
| deepseek-r1-0528 | DeepSeek R1 0528 (May '25) | https://artificialanalysis.ai/models/deepseek-r1 | 38.1 tokens per second | 1.18s | 2026-10-28 |
| deepseek-v3-0324 | DeepSeek V3 0324 | https://artificialanalysis.ai/models/deepseek-v3-0324 | 135.6 tokens per second | 0.75s | 2026-10-28 |
| gemini-2.5-flash | Gemini 2.5 Flash (Non-reasoning) | https://artificialanalysis.ai/models/gemini-2-5-flash | 209.5 tokens per second | 0.37s | 2026-10-28 |
| gemini-2.5-pro | Gemini 2.5 Pro | https://artificialanalysis.ai/models/gemini-2-5-pro | 115.9 tokens per second | 25.36s | 2026-10-28 |
| glm-4.6 | GLM-4.6 (Reasoning) | https://artificialanalysis.ai/models/glm-4-6-reasoning | 31.2 tokens per second | 4.06s | 2026-10-28 |
| gpt-5 | GPT-5 (Medium) | https://artificialanalysis.ai/models/gpt-5-medium | 103.5 tokens per second | 27.95s | 2026-10-28 |
| gpt-5-chat | GPT-5 (ChatGPT) | https://artificialanalysis.ai/models/gpt-5-chatgpt | N/A | not shown | 2026-10-28 |
| intern-s1 | no page found | https://artificialanalysis.ai/leaderboards/models?status=all | no entry | no entry | 2026-10-28 |
| kimi-k2-0905 | Kimi K2 0905 | https://artificialanalysis.ai/models/kimi-k2-0905 | 46.8 tokens per second | 1.38s | 2026-10-28 |
| qwen3-235b-a22b-2507 | Qwen3 235B A22B 2507 Instruct | https://artificialanalysis.ai/models/qwen3-235b-a22b-instruct-2507 | 48.8 tokens per second | 1.26s | 2026-10-28 |
| qwen3-235b-a22b-thinking-2507 | Qwen3 235B A22B 2507 (Reasoning) | https://artificialanalysis.ai/models/qwen3-235b-a22b-instruct-2507-reasoning | 124.5 tokens per second | 0.75s | 2026-10-28 |

"not shown" means the page's FAQ has no TTFT sentence; "N/A" is what the page's Speed box
displays. For Intern-S1 the model listing (716 model pages) was searched for the name and the
slugs `intern-s1` and a few variants returned 404, so there is no page to read.

## How the profile values follow from the table

- `tpot_median_ms` = 1000 / output speed. A model with no displayed speed (claude-sonnet-4,
  gpt-5-chat, intern-s1) gets the median of the nine displayed ones, 9.6618357 ms
  (the 103.5 tokens per second of gpt-5), marked `assumed, no public figure`.
- `ttft_median_ms` = the displayed TTFT in milliseconds for the non-reasoning variants that show
  one (deepseek-v3-0324 750, gemini-2.5-flash 370, kimi-k2-0905 1380, qwen3-235b-a22b-2507 1260).
- The six reasoning variants (deepseek-r1-0528, gemini-2.5-pro, glm-4.6, gpt-5, intern-s1,
  qwen3-235b-a22b-thinking-2507) take the median of those four, 1005 ms. Their displayed TTFT
  includes hidden reasoning time and the recorded `completion_tokens` already include reasoning
  tokens, so using it would count the reasoning twice. The displayed value stays in each note.
- The two non-reasoning models with no displayed TTFT (claude-sonnet-4, gpt-5-chat) also get
  1005 ms, marked `assumed, no public figure`.
- `ttft_sigma` 0.5 and `tpot_sigma` 0.3 for every model. No page gives a spread; the notes say
  `sigma assumed`.

`internal/latency` checks the file against these rules (`TestShippedProfiles`).
