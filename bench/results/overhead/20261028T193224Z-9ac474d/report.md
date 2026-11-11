# Gateway overhead against the local mock (20261028T193224Z-9ac474d)

Labels: real local measurement, nothing simulated. Machine: AMD Ryzen 9 7900X 12-Core Processor (24 threads), 15.2 GB RAM, Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39, go1.27.2. The load generator (rp-load), the gateway (routepilot serve) and the mock (rp-mock) share one machine and talk over loopback. The mock's delays are configured constants (time to first token 20 ms, 1 ms per output token, 16 output tokens), so absolute latencies are the mock's; only the difference between the gateway path and the direct path measures RoutePilot. Latencies run from each request's scheduled time (open loop, coordinated-omission safe). Measured with rp-load --clock-probe before the run, the clock step is 30 ns (the smallest non-zero difference between consecutive time.Now reads; 0.0000 of the consecutive reads were equal).

Settings: 10 s warmup (sent, not recorded) and 60 s measurement window per cell, one run per cell, request body a fixed invented sentence of about 200 bytes with max_tokens 16, generator max in-flight 4096, a fresh mock and gateway process for every cell.

Target: gateway overhead at p95 below 15 ms. It is a target, not a guarantee; each row below says whether it was met.

## Runs (client side, ms)

| run | scheduled | completed | errors | dropped | throughput rps | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct-100rps-nonstream | 5876 | 5876 | 0 | 0 | 97.9 | 37.53 | 38.14 | 38.27 | 38.46 | 38.81 | 40.19 |
| gateway-100rps-nonstream | 5876 | 5876 | 0 | 0 | 97.9 | 38.05 | 38.72 | 38.91 | 40.80 | 47.65 | 63.36 |
| direct-100rps-stream | 5876 | 5876 | 0 | 0 | 97.9 | 39.33 | 40.22 | 40.51 | 41.70 | 51.65 | 64.06 |
| gateway-100rps-stream | 5876 | 5876 | 0 | 0 | 97.9 | 39.81 | 40.93 | 41.28 | 41.92 | 44.35 | 47.52 |
| direct-500rps-nonstream | 29900 | 29900 | 0 | 0 | 498.3 | 37.34 | 38.02 | 38.21 | 39.52 | 49.22 | 78.33 |
| gateway-500rps-nonstream | 29900 | 29900 | 0 | 0 | 498.3 | 37.63 | 38.30 | 38.49 | 39.20 | 45.73 | 57.66 |
| direct-500rps-stream | 29900 | 29900 | 0 | 0 | 498.3 | 41.60 | 43.26 | 43.90 | 53.34 | 82.17 | 118.72 |
| gateway-500rps-stream | 29900 | 29900 | 0 | 0 | 498.3 | 43.10 | 45.02 | 45.57 | 46.75 | 49.34 | 54.02 |
| direct-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 37.28 | 37.92 | 38.08 | 38.34 | 39.81 | 46.94 |
| gateway-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 37.57 | 38.27 | 38.46 | 38.81 | 43.97 | 63.90 |
| direct-1000rps-stream | 59708 | 59708 | 0 | 0 | 995.1 | 43.45 | 46.05 | 48.61 | 58.14 | 72.83 | 94.85 |
| gateway-1000rps-stream | 59708 | 59708 | 0 | 0 | 995.1 | 43.10 | 45.31 | 45.98 | 47.42 | 58.37 | 74.05 |
| direct-2000rps-nonstream | 119503 | 119503 | 0 | 0 | 1991.7 | 37.15 | 37.79 | 37.95 | 38.27 | 39.97 | 49.63 |
| gateway-2000rps-nonstream | 119503 | 119503 | 0 | 0 | 1991.7 | 37.44 | 38.14 | 38.34 | 38.66 | 40.83 | 55.65 |
| direct-2000rps-stream | 119503 | 119503 | 0 | 0 | 1991.7 | 43.23 | 45.18 | 45.79 | 47.20 | 64.16 | 89.22 |
| gateway-2000rps-stream | 119503 | 119503 | 0 | 0 | 1991.7 | 41.85 | 44.16 | 45.44 | 226.30 | 480.77 | 606.21 |
| learned-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 37.57 | 38.24 | 38.40 | 38.69 | 40.03 | 47.84 |

Time to first token (streaming runs, ms):

| run | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| direct-100rps-stream | 21.39 | 22.08 | 22.24 | 22.46 | 24.24 | 26.64 |
| gateway-100rps-stream | 21.71 | 22.37 | 22.53 | 22.78 | 23.60 | 26.14 |
| direct-500rps-stream | 21.30 | 22.03 | 22.24 | 23.81 | 43.62 | 70.08 |
| gateway-500rps-stream | 21.74 | 22.54 | 22.80 | 23.42 | 25.01 | 31.42 |
| direct-1000rps-stream | 21.21 | 22.16 | 22.66 | 26.46 | 36.19 | 58.46 |
| gateway-1000rps-stream | 21.65 | 22.59 | 22.94 | 23.93 | 32.70 | 53.02 |
| direct-2000rps-stream | 21.04 | 21.92 | 22.21 | 22.89 | 35.17 | 68.93 |
| gateway-2000rps-stream | 21.58 | 22.82 | 23.66 | 149.89 | 369.41 | 574.46 |

Generator launch lag (how late the scheduler started a request, ms; it is part of every latency above):

| run | p50 | p99 | max |
|---|---|---|---|
| direct-100rps-nonstream | 0.57 | 1.09 | 3.39 |
| gateway-100rps-nonstream | 0.58 | 1.11 | 6.77 |
| direct-100rps-stream | 0.52 | 1.11 | 3.33 |
| gateway-100rps-stream | 0.49 | 1.09 | 1.73 |
| direct-500rps-nonstream | 0.48 | 1.10 | 13.76 |
| gateway-500rps-nonstream | 0.47 | 1.09 | 7.90 |
| direct-500rps-stream | 0.36 | 1.18 | 12.32 |
| gateway-500rps-stream | 0.22 | 1.03 | 4.05 |
| direct-1000rps-nonstream | 0.43 | 1.08 | 8.12 |
| gateway-1000rps-nonstream | 0.41 | 1.08 | 9.23 |
| direct-1000rps-stream | 0.19 | 1.54 | 11.13 |
| gateway-1000rps-stream | 0.11 | 1.02 | 4.74 |
| direct-2000rps-nonstream | 0.40 | 1.07 | 6.43 |
| gateway-2000rps-nonstream | 0.33 | 1.06 | 8.94 |
| direct-2000rps-stream | 0.12 | 1.21 | 9.56 |
| gateway-2000rps-stream | 0.07 | 1.87 | 51.62 |
| learned-1000rps-nonstream | 0.41 | 1.07 | 5.42 |

## Overhead, client side: gateway minus direct (ms)

Label: difference of percentiles, not a percentile of differences. Each cell is the percentile seen through the gateway minus the same percentile seen directly at the same rate and mode.

| rate | mode | p50 | p90 | p95 | p99 | p99.9 | p95 < 15 ms |
|---|---|---|---|---|---|---|---|
| 100 | non-streaming | 0.51 | 0.58 | 0.64 | 2.34 | 8.83 | met |
| 100 | streaming | 0.48 | 0.70 | 0.77 | 0.22 | -7.30 | met |
| 500 | non-streaming | 0.29 | 0.29 | 0.29 | -0.32 | -3.49 | met |
| 500 | streaming | 1.50 | 1.76 | 1.66 | -6.59 | -32.83 | met |
| 1000 | non-streaming | 0.29 | 0.35 | 0.38 | 0.48 | 4.16 | met |
| 1000 | streaming | -0.35 | -0.74 | -2.62 | -10.72 | -14.46 | met |
| 2000 | non-streaming | 0.29 | 0.35 | 0.38 | 0.38 | 0.86 | met |
| 2000 | streaming | -1.38 | -1.02 | -0.35 | 179.10 | 416.61 | met |

Time to first token, gateway minus direct (ms, difference of percentiles):

| rate | p50 | p90 | p95 | p99 | p99.9 |
|---|---|---|---|---|---|
| 100 | 0.32 | 0.29 | 0.29 | 0.32 | -0.64 |
| 500 | 0.45 | 0.51 | 0.56 | -0.38 | -18.61 |
| 1000 | 0.43 | 0.43 | 0.29 | -2.53 | -3.49 |
| 2000 | 0.54 | 0.90 | 1.46 | 126.99 | 334.24 |

## Overhead, server side: rp_gateway_overhead_seconds (ms, from histogram buckets)

Label: from histogram buckets. The gateway defines this series as the request duration minus the summed duration of its upstream attempts, so it is the gateway's own accounting and need not equal the client-side difference. The gateway's /metrics was scraped right after the warmup and after the run; the bucket counts between the two exclude the warmup. Quantiles are interpolated inside a bucket and shown as value (bucket lower-upper]; the true value lies inside that bucket. 'server n' is the number of observations in the window and 'completed' is the generator's count; they differ by the requests in flight at the boundary and by failed attempts.

| rate | mode | server n | completed | p50 | p95 | p99 | p95 < 15 ms |
|---|---|---|---|---|---|---|---|
| 100 | non-streaming | 5882 | 5876 | 0.161 (0.100-0.215] | 0.328 (0.215-0.464] | 0.447 (0.215-0.464] | met |
| 100 | streaming | 5882 | 5876 | 0.096 (0.046-0.100] | 0.207 (0.100-0.215] | 0.340 (0.215-0.464] | met |
| 500 | non-streaming | 29915 | 29900 | 0.121 (0.100-0.215] | 0.209 (0.100-0.215] | 0.317 (0.215-0.464] | met |
| 500 | streaming | 29920 | 29900 | 0.092 (0.046-0.100] | 0.401 (0.215-0.464] | 0.902 (0.464-1.000] | met |
| 1000 | non-streaming | 59758 | 59708 | 0.120 (0.100-0.215] | 0.209 (0.100-0.215] | 0.342 (0.215-0.464] | met |
| 1000 | streaming | 59762 | 59708 | 0.079 (0.046-0.100] | 0.244 (0.215-0.464] | 0.916 (0.464-1.000] | met |
| 2000 | non-streaming | 119596 | 119503 | 0.086 (0.046-0.100] | 0.201 (0.100-0.215] | 0.226 (0.215-0.464] | met |
| 2000 | streaming | 119588 | 119503 | 0.076 (0.046-0.100] | 0.192 (0.100-0.215] | 1.177 (1.000-2.154] | met |

## Target summary

- 100 rps, non-streaming: client-side p95 difference 0.64 ms, met; server-side p95 0.328 (0.215-0.464] ms, met.
- 100 rps, streaming: client-side p95 difference 0.77 ms, met; server-side p95 0.207 (0.100-0.215] ms, met.
- 500 rps, non-streaming: client-side p95 difference 0.29 ms, met; server-side p95 0.209 (0.100-0.215] ms, met.
- 500 rps, streaming: client-side p95 difference 1.66 ms, met; server-side p95 0.401 (0.215-0.464] ms, met.
- 1000 rps, non-streaming: client-side p95 difference 0.38 ms, met; server-side p95 0.209 (0.100-0.215] ms, met.
- 1000 rps, streaming: client-side p95 difference -2.62 ms, met; server-side p95 0.244 (0.215-0.464] ms, met.
- 2000 rps, non-streaming: client-side p95 difference 0.38 ms, met; server-side p95 0.201 (0.100-0.215] ms, met.
- 2000 rps, streaming: client-side p95 difference -0.35 ms, met; server-side p95 0.192 (0.100-0.215] ms, met.

## Learned policy run (flagship artifact behind the auto model, gateway only)

The learned run is compared with the direct run (the whole cost of going through RoutePilot with the learned policy) and with the pinned-model gateway run at the same rate (what the policy adds). Every request carries the same prompt, so the decision takes the same path each time. Differences of percentiles, not percentiles of differences.

| comparison | rate | p50 | p90 | p95 | p99 | p99.9 | p95 < 15 ms |
|---|---|---|---|---|---|---|---|
| learned vs direct | 1000 | 0.29 | 0.32 | 0.32 | 0.35 | 0.22 | met |
| learned vs gateway | 1000 | 0.00 | -0.03 | -0.06 | -0.13 | -3.94 | met |

| server side (from histogram buckets, ms) | p50 | p95 | p99 |
|---|---|---|---|
| rp_gateway_overhead_seconds (learned run) | 0.160 (0.100-0.215] | 0.213 (0.100-0.215] | 0.416 (0.215-0.464] |
| rp_decision_duration_seconds (learned run) | 0.069 (0.046-0.100] | 0.098 (0.046-0.100] | 0.156 (0.100-0.215] |

Policy version served: dea4486d7575+ef0a989c.

## Limitations

- One run per cell: there are no repetitions and no confidence intervals. A difference of a fraction of a millisecond is within the run-to-run noise of a shared machine.
- The generator, the gateway and the mock compete for the same CPU, and other programs on this machine could not be controlled. The launch lag table shows how much the generator itself added.
- The gateway writes one log line per request to a file; accounting, the decision log and tracing are off, and the tenant has no limits (see bench/configs/overhead.yaml).
- The server-side quantiles have the resolution of the histogram buckets (about a factor of 2.15 per bucket), so a p95 can be undecided against the target.
- The mock answers with constant delays and tiny bodies; real providers add variable latency, and larger bodies would cost the gateway more to copy.
- All requests carry one fixed short prompt.
