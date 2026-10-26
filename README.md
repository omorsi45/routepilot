# RoutePilot

A quality-constrained, SLO-aware LLM inference routing gateway.

RoutePilot is an OpenAI-compatible gateway written in Go. For every request it
chooses the least expensive model whose calibrated, lower-confidence-bound
probability of success clears the caller's quality floor, while respecting
latency SLOs, tenant budgets, rate limits, and provider health.

Status: under active development. Nothing in this repository claims a
measured result until the corresponding benchmark report exists under
`bench/results/`.

## License

Apache-2.0. See `LICENSE`.
