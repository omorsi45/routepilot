# RoutePilot

A quality-constrained, SLO-aware LLM inference routing gateway.

RoutePilot is an OpenAI-compatible gateway written in Go. For every request it
chooses the least expensive model whose calibrated, lower-confidence-bound
probability of success clears the caller's quality floor, while respecting
latency SLOs, tenant budgets, rate limits, and provider health.

Status: under active development. Nothing in this repository claims a
measured result until the corresponding benchmark report exists under
`bench/results/`.

## Security notes

- `GET /metrics` is unauthenticated and served on the main listener, like `/healthz` and
  `/readyz`. Its series hold identifiers (tenant IDs, model and deployment names, policy
  versions), counts, durations and costs, and never a prompt, a message, an API key or a key
  hash. Tenant labels are bounded to the configured tenants. It is still an internal-network
  endpoint: scrape it from inside the network, and block the path at any edge that faces the
  internet.

## License

Apache-2.0. See `LICENSE`.
