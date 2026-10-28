# Gateway overhead against the local mock (20261028T181907Z-0bb874d)

Labels: real local measurement, nothing simulated. Machine: AMD Ryzen 9 7900X 12-Core Processor (24 threads), 31.2 GB RAM, Windows-11-10.0.26300-SP0, go1.27.2. The load generator (rp-load), the gateway (routepilot serve) and the mock (rp-mock) share one machine and talk over loopback. The mock's delays are configured constants (time to first token 20 ms, 1 ms per output token, 16 output tokens), so absolute latencies are the mock's; only the difference between the gateway path and the direct path measures RoutePilot. Latencies run from each request's scheduled time (open loop, coordinated-omission safe).

Settings: 10 s warmup (sent, not recorded) and 60 s measurement window per cell, one run per cell, request body a fixed invented sentence of about 200 bytes with max_tokens 16, generator max in-flight 4096, a fresh mock and gateway process for every cell.

Target: gateway overhead at p95 below 15 ms. It is a target, not a guarantee; each row below says whether it was met.

## Runs (client side, ms)

| run | scheduled | completed | errors | dropped | throughput rps | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct-100rps-nonstream | 5876 | 5876 | 0 | 0 | 97.9 | 36.64 | 36.93 | 37.02 | 37.25 | 37.66 | 39.10 |
| gateway-100rps-nonstream | 5876 | 5876 | 0 | 0 | 97.9 | 36.70 | 37.02 | 37.12 | 37.34 | 38.40 | 44.83 |
| direct-100rps-stream | 5876 | 5876 | 0 | 0 | 97.9 | 41.50 | 42.98 | 43.39 | 45.66 | 58.53 | 63.65 |
| gateway-100rps-stream | 5876 | 5876 | 0 | 0 | 97.9 | 41.53 | 42.94 | 43.30 | 43.90 | 50.78 | 53.09 |
| direct-500rps-nonstream | 29900 | 29900 | 0 | 0 | 498.3 | 36.61 | 36.93 | 36.99 | 37.25 | 40.41 | 44.16 |
| gateway-500rps-nonstream | 29900 | 29900 | 0 | 0 | 498.3 | 36.67 | 36.99 | 37.09 | 37.31 | 38.21 | 43.81 |
| direct-500rps-stream | 29900 | 29900 | 0 | 0 | 498.3 | 38.91 | 40.51 | 41.02 | 42.05 | 46.02 | 54.34 |
| gateway-500rps-stream | 29900 | 29900 | 0 | 0 | 498.3 | 38.88 | 40.54 | 41.02 | 42.24 | 55.65 | 72.45 |
| direct-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 36.61 | 36.93 | 36.99 | 37.28 | 39.90 | 46.88 |
| gateway-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 36.64 | 36.99 | 37.09 | 37.44 | 40.45 | 48.61 |
| direct-1000rps-stream | 59708 | 59708 | 0 | 0 | 995.1 | 37.60 | 38.81 | 39.23 | 40.19 | 46.88 | 53.28 |
| gateway-1000rps-stream | 59708 | 59708 | 0 | 0 | 995.1 | 37.73 | 38.98 | 39.39 | 40.26 | 42.17 | 48.93 |
| direct-2000rps-nonstream | 119503 | 119503 | 0 | 0 | 1991.7 | 36.58 | 36.90 | 36.96 | 37.12 | 37.53 | 39.90 |
| gateway-2000rps-nonstream | 119503 | 119503 | 0 | 0 | 1991.7 | 36.58 | 36.93 | 37.02 | 37.31 | 39.90 | 45.41 |
| direct-2000rps-stream | 119503 | 119503 | 0 | 0 | 1991.7 | 36.96 | 37.98 | 38.30 | 39.10 | 45.66 | 53.15 |
| gateway-2000rps-stream | 119503 | 119503 | 0 | 0 | 1991.7 | 37.15 | 38.30 | 38.72 | 40.73 | 58.66 | 76.54 |
| learned-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 36.64 | 36.99 | 37.12 | 37.41 | 40.32 | 48.03 |

Time to first token (streaming runs, ms):

| run | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| direct-100rps-stream | 20.62 | 20.94 | 21.04 | 21.47 | 26.98 | 29.66 |
| gateway-100rps-stream | 20.66 | 20.99 | 21.10 | 21.38 | 22.86 | 25.97 |
| direct-500rps-stream | 20.61 | 20.91 | 20.99 | 21.23 | 22.67 | 30.80 |
| gateway-500rps-stream | 20.66 | 21.02 | 21.18 | 21.55 | 27.01 | 35.65 |
| direct-1000rps-stream | 20.61 | 20.89 | 20.99 | 21.28 | 22.32 | 26.99 |
| gateway-1000rps-stream | 20.69 | 21.10 | 21.31 | 21.68 | 22.27 | 29.28 |
| direct-2000rps-stream | 20.59 | 20.93 | 21.02 | 21.39 | 22.53 | 33.25 |
| gateway-2000rps-stream | 20.67 | 21.25 | 21.49 | 22.34 | 32.14 | 50.56 |

Generator launch lag (how late the scheduler started a request, ms; it is part of every latency above):

| run | p50 | p99 | max |
|---|---|---|---|
| direct-100rps-nonstream | 0.30 | 0.67 | 2.39 |
| gateway-100rps-nonstream | 0.33 | 0.77 | 1.40 |
| direct-100rps-stream | 0.31 | 0.71 | 6.91 |
| gateway-100rps-stream | 0.33 | 0.76 | 1.35 |
| direct-500rps-nonstream | 0.28 | 0.68 | 4.77 |
| gateway-500rps-nonstream | 0.32 | 0.76 | 2.52 |
| direct-500rps-stream | 0.29 | 0.64 | 4.99 |
| gateway-500rps-stream | 0.31 | 0.73 | 4.54 |
| direct-1000rps-nonstream | 0.28 | 0.69 | 7.72 |
| gateway-1000rps-nonstream | 0.31 | 0.78 | 6.99 |
| direct-1000rps-stream | 0.28 | 0.62 | 3.29 |
| gateway-1000rps-stream | 0.29 | 0.70 | 5.30 |
| direct-2000rps-nonstream | 0.27 | 0.60 | 3.19 |
| gateway-2000rps-nonstream | 0.29 | 0.74 | 4.04 |
| direct-2000rps-stream | 0.27 | 0.58 | 10.89 |
| gateway-2000rps-stream | 0.27 | 0.66 | 12.86 |
| learned-1000rps-nonstream | 0.31 | 0.78 | 6.45 |

## Overhead, client side: gateway minus direct (ms)

Label: difference of percentiles, not a percentile of differences. Each cell is the percentile seen through the gateway minus the same percentile seen directly at the same rate and mode.

| rate | mode | p50 | p90 | p95 | p99 | p99.9 | p95 < 15 ms |
|---|---|---|---|---|---|---|---|
| 100 | non-streaming | 0.06 | 0.10 | 0.10 | 0.10 | 0.74 | met |
| 100 | streaming | 0.03 | -0.03 | -0.10 | -1.76 | -7.74 | met |
| 500 | non-streaming | 0.06 | 0.06 | 0.10 | 0.06 | -2.21 | met |
| 500 | streaming | -0.03 | 0.03 | 0.00 | 0.19 | 9.63 | met |
| 1000 | non-streaming | 0.03 | 0.06 | 0.10 | 0.16 | 0.54 | met |
| 1000 | streaming | 0.13 | 0.16 | 0.16 | 0.06 | -4.70 | met |
| 2000 | non-streaming | 0.00 | 0.03 | 0.06 | 0.19 | 2.37 | met |
| 2000 | streaming | 0.19 | 0.32 | 0.42 | 1.63 | 12.99 | met |

Time to first token, gateway minus direct (ms, difference of percentiles):

| rate | p50 | p90 | p95 | p99 | p99.9 |
|---|---|---|---|---|---|
| 100 | 0.03 | 0.05 | 0.06 | -0.10 | -4.11 |
| 500 | 0.05 | 0.11 | 0.19 | 0.32 | 4.34 |
| 1000 | 0.08 | 0.21 | 0.32 | 0.40 | -0.05 |
| 2000 | 0.08 | 0.32 | 0.46 | 0.94 | 9.62 |

## Overhead, server side: rp_gateway_overhead_seconds (ms, from histogram buckets)

Label: from histogram buckets. The gateway defines this series as the request duration minus the summed duration of its upstream attempts, so it is the gateway's own accounting and need not equal the client-side difference. The gateway's /metrics was scraped right after the warmup and after the run; the bucket counts between the two exclude the warmup. Quantiles are interpolated inside a bucket and shown as value (bucket lower-upper]; the true value lies inside that bucket. 'server n' is the number of observations in the window and 'completed' is the generator's count; they differ by the requests in flight at the boundary and by failed attempts.

| rate | mode | server n | completed | p50 | p95 | p99 | p95 < 15 ms |
|---|---|---|---|---|---|---|---|
| 100 | non-streaming | 5881 | 5876 | 0.005 (0.000-0.010] | 0.010 (0.000-0.010] | 0.010 (0.000-0.010] | met |
| 100 | streaming | 5883 | 5876 | 0.005 (0.000-0.010] | 0.010 (0.000-0.010] | 0.010 (0.000-0.010] | met |
| 500 | non-streaming | 29915 | 29900 | 0.005 (0.000-0.010] | 0.010 (0.000-0.010] | 0.010 (0.000-0.010] | met |
| 500 | streaming | 29915 | 29900 | 0.005 (0.000-0.010] | 0.010 (0.000-0.010] | 0.867 (0.464-1.000] | met |
| 1000 | non-streaming | 59737 | 59708 | 0.005 (0.000-0.010] | 0.010 (0.000-0.010] | 0.010 (0.000-0.010] | met |
| 1000 | streaming | 59759 | 59708 | 0.005 (0.000-0.010] | 0.530 (0.464-1.000] | 0.915 (0.464-1.000] | met |
| 2000 | non-streaming | 119571 | 119503 | 0.005 (0.000-0.010] | 0.010 (0.000-0.010] | 0.010 (0.000-0.010] | met |
| 2000 | streaming | 119533 | 119503 | 0.005 (0.000-0.010] | 0.650 (0.464-1.000] | 0.948 (0.464-1.000] | met |

## Target summary

- 100 rps, non-streaming: client-side p95 difference 0.10 ms, met; server-side p95 0.010 (0.000-0.010] ms, met.
- 100 rps, streaming: client-side p95 difference -0.10 ms, met; server-side p95 0.010 (0.000-0.010] ms, met.
- 500 rps, non-streaming: client-side p95 difference 0.10 ms, met; server-side p95 0.010 (0.000-0.010] ms, met.
- 500 rps, streaming: client-side p95 difference 0.00 ms, met; server-side p95 0.010 (0.000-0.010] ms, met.
- 1000 rps, non-streaming: client-side p95 difference 0.10 ms, met; server-side p95 0.010 (0.000-0.010] ms, met.
- 1000 rps, streaming: client-side p95 difference 0.16 ms, met; server-side p95 0.530 (0.464-1.000] ms, met.
- 2000 rps, non-streaming: client-side p95 difference 0.06 ms, met; server-side p95 0.010 (0.000-0.010] ms, met.
- 2000 rps, streaming: client-side p95 difference 0.42 ms, met; server-side p95 0.650 (0.464-1.000] ms, met.

## Learned policy run (flagship artifact behind the auto model, gateway only)

The learned run is compared with the direct run (the whole cost of going through RoutePilot with the learned policy) and with the pinned-model gateway run at the same rate (what the policy adds). Every request carries the same prompt, so the decision takes the same path each time. Differences of percentiles, not percentiles of differences.

| comparison | rate | p50 | p90 | p95 | p99 | p99.9 | p95 < 15 ms |
|---|---|---|---|---|---|---|---|
| learned vs direct | 1000 | 0.03 | 0.06 | 0.13 | 0.13 | 0.42 | met |
| learned vs gateway | 1000 | 0.00 | 0.00 | 0.03 | -0.03 | -0.13 | met |

| server side (from histogram buckets, ms) | p50 | p95 | p99 |
|---|---|---|---|
| rp_gateway_overhead_seconds (learned run) | 0.005 (0.000-0.010] | 0.010 (0.000-0.010] | 0.622 (0.464-1.000] |
| rp_decision_duration_seconds (learned run) | 0.005 (0.000-0.010] | 0.010 (0.000-0.010] | 0.010 (0.000-0.010] |

Policy version served: dea4486d7575+ef0a989c.

## Limitations

- One run per cell: there are no repetitions and no confidence intervals. A difference of a fraction of a millisecond is within the run-to-run noise of a shared machine.
- The generator, the gateway and the mock compete for the same CPU, and other programs on this machine could not be controlled. The launch lag table shows how much the generator itself added.
- The gateway writes one log line per request to a file; accounting, the decision log and tracing are off, and the tenant has no limits (see bench/configs/overhead.yaml).
- The server-side quantiles have the resolution of the histogram buckets (about a factor of 2.15 per bucket), so a p95 can be undecided against the target.
- The mock answers with constant delays and tiny bodies; real providers add variable latency, and larger bodies would cost the gateway more to copy.
- All requests carry one fixed short prompt.
