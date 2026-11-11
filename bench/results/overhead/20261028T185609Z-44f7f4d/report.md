# Gateway overhead against the local mock (20261028T185609Z-44f7f4d)

Labels: real local measurement, nothing simulated. Machine: AMD Ryzen 9 7900X 12-Core Processor (24 threads), 15.2 GB RAM, Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39, go1.27.2. The load generator (rp-load), the gateway (routepilot serve) and the mock (rp-mock) share one machine and talk over loopback. The mock's delays are configured constants (time to first token 20 ms, 1 ms per output token, 16 output tokens), so absolute latencies are the mock's; only the difference between the gateway path and the direct path measures RoutePilot. Latencies run from each request's scheduled time (open loop, coordinated-omission safe). Measured with rp-load --clock-probe before the run, the clock step is 30 ns (the smallest non-zero difference between consecutive time.Now reads; 0.0000 of the consecutive reads were equal).

Settings: 10 s warmup (sent, not recorded) and 60 s measurement window per cell, one run per cell, request body a fixed invented sentence of about 200 bytes with max_tokens 16, generator max in-flight 4096, a fresh mock and gateway process for every cell.

Target: gateway overhead at p95 below 15 ms. It is a target, not a guarantee; each row below says whether it was met.

## Runs (client side, ms)

| run | scheduled | completed | errors | dropped | throughput rps | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct-100rps-nonstream | 5876 | 5876 | 0 | 0 | 97.9 | 37.53 | 38.11 | 38.27 | 38.46 | 38.66 | 38.88 |
| gateway-100rps-nonstream | 5876 | 5876 | 0 | 0 | 97.9 | 37.95 | 38.56 | 38.72 | 39.01 | 40.73 | 43.17 |
| direct-100rps-stream | 5876 | 5876 | 0 | 0 | 97.9 | 39.20 | 40.16 | 40.51 | 47.52 | 61.22 | 77.63 |
| gateway-100rps-stream | 5876 | 5876 | 0 | 0 | 97.9 | 39.62 | 40.70 | 41.09 | 41.70 | 42.49 | 43.10 |
| direct-500rps-nonstream | 29900 | 29900 | 0 | 0 | 498.3 | 37.34 | 37.98 | 38.14 | 38.43 | 42.43 | 47.68 |
| gateway-500rps-nonstream | 29900 | 29900 | 0 | 0 | 498.3 | 37.76 | 38.49 | 38.78 | 41.57 | 48.83 | 61.44 |
| direct-500rps-stream | 29900 | 29900 | 0 | 0 | 498.3 | 41.38 | 42.81 | 43.33 | 44.54 | 56.77 | 69.06 |
| gateway-500rps-stream | 29900 | 29900 | 0 | 0 | 498.3 | 42.56 | 44.38 | 44.96 | 46.21 | 54.11 | 63.33 |
| direct-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 37.25 | 37.92 | 38.11 | 38.59 | 45.47 | 56.86 |
| gateway-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 37.57 | 38.27 | 38.49 | 39.20 | 45.34 | 65.47 |
| direct-1000rps-stream | 59708 | 59708 | 0 | 0 | 995.1 | 43.17 | 44.90 | 45.38 | 46.37 | 47.81 | 55.42 |
| gateway-1000rps-stream | 59708 | 59708 | 0 | 0 | 995.1 | 42.98 | 45.89 | 51.07 | 104.45 | 203.90 | 272.38 |
| direct-2000rps-nonstream | 119503 | 119503 | 0 | 0 | 1991.7 | 37.15 | 37.79 | 37.98 | 38.34 | 43.10 | 62.21 |
| gateway-2000rps-nonstream | 119503 | 119503 | 0 | 0 | 1991.7 | 38.21 | 877.57 | 1065.98 | 1520.64 | 1620.99 | 2729.98 |
| direct-2000rps-stream | 119503 | 119503 | 0 | 0 | 1991.7 | 43.33 | 45.31 | 45.89 | 46.98 | 48.26 | 50.21 |
| gateway-2000rps-stream | 119503 | 119503 | 0 | 0 | 1991.7 | 41.66 | 43.55 | 44.16 | 45.82 | 65.31 | 83.33 |
| learned-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 37.60 | 38.34 | 38.56 | 39.33 | 45.60 | 53.53 |

Time to first token (streaming runs, ms):

| run | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| direct-100rps-stream | 21.39 | 22.08 | 22.24 | 22.70 | 28.72 | 34.40 |
| gateway-100rps-stream | 21.65 | 22.30 | 22.48 | 22.72 | 23.01 | 23.28 |
| direct-500rps-stream | 21.26 | 21.98 | 22.18 | 22.51 | 25.71 | 42.53 |
| gateway-500rps-stream | 21.61 | 22.40 | 22.67 | 23.36 | 31.50 | 41.44 |
| direct-1000rps-stream | 21.15 | 21.92 | 22.16 | 22.61 | 23.23 | 31.30 |
| gateway-1000rps-stream | 21.61 | 22.80 | 24.66 | 56.83 | 163.46 | 237.95 |
| direct-2000rps-stream | 21.05 | 21.90 | 22.18 | 22.72 | 23.49 | 25.36 |
| gateway-2000rps-stream | 21.44 | 22.41 | 22.77 | 23.97 | 34.85 | 58.08 |

Generator launch lag (how late the scheduler started a request, ms; it is part of every latency above):

| run | p50 | p99 | max |
|---|---|---|---|
| direct-100rps-nonstream | 0.57 | 1.09 | 1.19 |
| gateway-100rps-nonstream | 0.57 | 1.09 | 1.36 |
| direct-100rps-stream | 0.52 | 1.10 | 13.47 |
| gateway-100rps-stream | 0.45 | 1.08 | 1.14 |
| direct-500rps-nonstream | 0.48 | 1.08 | 5.55 |
| gateway-500rps-nonstream | 0.47 | 1.10 | 7.47 |
| direct-500rps-stream | 0.37 | 1.13 | 3.56 |
| gateway-500rps-stream | 0.19 | 1.02 | 4.52 |
| direct-1000rps-nonstream | 0.43 | 1.09 | 7.66 |
| gateway-1000rps-nonstream | 0.41 | 1.07 | 5.84 |
| direct-1000rps-stream | 0.20 | 1.22 | 6.04 |
| gateway-1000rps-stream | 0.10 | 2.46 | 80.70 |
| direct-2000rps-nonstream | 0.40 | 1.08 | 10.18 |
| gateway-2000rps-nonstream | 0.28 | 1.09 | 33.22 |
| direct-2000rps-stream | 0.12 | 1.21 | 3.25 |
| gateway-2000rps-stream | 0.06 | 0.74 | 6.16 |
| learned-1000rps-nonstream | 0.42 | 1.07 | 3.81 |

## Overhead, client side: gateway minus direct (ms)

Label: difference of percentiles, not a percentile of differences. Each cell is the percentile seen through the gateway minus the same percentile seen directly at the same rate and mode.

| rate | mode | p50 | p90 | p95 | p99 | p99.9 | p95 < 15 ms |
|---|---|---|---|---|---|---|---|
| 100 | non-streaming | 0.42 | 0.45 | 0.45 | 0.54 | 2.08 | met |
| 100 | streaming | 0.42 | 0.54 | 0.58 | -5.82 | -18.72 | met |
| 500 | non-streaming | 0.42 | 0.51 | 0.64 | 3.14 | 6.40 | met |
| 500 | streaming | 1.18 | 1.57 | 1.63 | 1.66 | -2.66 | met |
| 1000 | non-streaming | 0.32 | 0.35 | 0.38 | 0.61 | -0.13 | met |
| 1000 | streaming | -0.19 | 0.99 | 5.70 | 58.08 | 156.10 | met |
| 2000 | non-streaming | 1.06 | 839.78 | 1028.00 | 1482.30 | 1577.89 | not met |
| 2000 | streaming | -1.66 | -1.76 | -1.73 | -1.15 | 17.06 | met |

Time to first token, gateway minus direct (ms, difference of percentiles):

| rate | p50 | p90 | p95 | p99 | p99.9 |
|---|---|---|---|---|---|
| 100 | 0.26 | 0.22 | 0.24 | 0.02 | -5.71 |
| 500 | 0.35 | 0.42 | 0.50 | 0.85 | 5.79 |
| 1000 | 0.46 | 0.88 | 2.50 | 34.22 | 140.22 |
| 2000 | 0.38 | 0.51 | 0.59 | 1.25 | 11.36 |

## Overhead, server side: rp_gateway_overhead_seconds (ms, from histogram buckets)

Label: from histogram buckets. The gateway defines this series as the request duration minus the summed duration of its upstream attempts, so it is the gateway's own accounting and need not equal the client-side difference. The gateway's /metrics was scraped right after the warmup and after the run; the bucket counts between the two exclude the warmup. Quantiles are interpolated inside a bucket and shown as value (bucket lower-upper]; the true value lies inside that bucket. 'server n' is the number of observations in the window and 'completed' is the generator's count; they differ by the requests in flight at the boundary and by failed attempts.

| rate | mode | server n | completed | p50 | p95 | p99 | p95 < 15 ms |
|---|---|---|---|---|---|---|---|
| 100 | non-streaming | 5883 | 5876 | 0.663 (0.464-1.000] | 0.985 (0.464-1.000] | 1.970 (1.000-2.154] | met |
| 100 | streaming | 5883 | 5876 | 0.428 (0.215-0.464] | 0.947 (0.464-1.000] | 1.028 (1.000-2.154] | met |
| 500 | non-streaming | 29921 | 29900 | 0.724 (0.464-1.000] | 1.942 (1.000-2.154] | 4.320 (2.154-4.642] | met |
| 500 | streaming | 29922 | 29900 | 0.676 (0.464-1.000] | 1.796 (1.000-2.154] | 2.471 (2.154-4.642] | met |
| 1000 | non-streaming | 59761 | 59708 | 0.628 (0.464-1.000] | 1.913 (1.000-2.154] | 3.134 (2.154-4.642] | met |
| 1000 | streaming | 59765 | 59708 | 0.930 (0.464-1.000] | 382.652 (215.400-464.200] | 1000.000 (1000.000-inf] | not met |
| 2000 | non-streaming | 119589 | 119503 | 2.074 (1.000-2.154] | 1000.000 (1000.000-inf] | 1000.000 (1000.000-inf] | not met |
| 2000 | streaming | 119594 | 119503 | 6.765 (4.642-10.000] | 56.786 (46.420-100.000] | 91.625 (46.420-100.000] | not met |

## Target summary

- 100 rps, non-streaming: client-side p95 difference 0.45 ms, met; server-side p95 0.985 (0.464-1.000] ms, met.
- 100 rps, streaming: client-side p95 difference 0.58 ms, met; server-side p95 0.947 (0.464-1.000] ms, met.
- 500 rps, non-streaming: client-side p95 difference 0.64 ms, met; server-side p95 1.942 (1.000-2.154] ms, met.
- 500 rps, streaming: client-side p95 difference 1.63 ms, met; server-side p95 1.796 (1.000-2.154] ms, met.
- 1000 rps, non-streaming: client-side p95 difference 0.38 ms, met; server-side p95 1.913 (1.000-2.154] ms, met.
- 1000 rps, streaming: client-side p95 difference 5.70 ms, met; server-side p95 382.652 (215.400-464.200] ms, not met.
- 2000 rps, non-streaming: client-side p95 difference 1028.00 ms, not met; server-side p95 1000.000 (1000.000-inf] ms, not met.
- 2000 rps, streaming: client-side p95 difference -1.73 ms, met; server-side p95 56.786 (46.420-100.000] ms, not met.

## Learned policy run (flagship artifact behind the auto model, gateway only)

The learned run is compared with the direct run (the whole cost of going through RoutePilot with the learned policy) and with the pinned-model gateway run at the same rate (what the policy adds). Every request carries the same prompt, so the decision takes the same path each time. Differences of percentiles, not percentiles of differences.

| comparison | rate | p50 | p90 | p95 | p99 | p99.9 | p95 < 15 ms |
|---|---|---|---|---|---|---|---|
| learned vs direct | 1000 | 0.35 | 0.42 | 0.45 | 0.74 | 0.13 | met |
| learned vs gateway | 1000 | 0.03 | 0.06 | 0.06 | 0.13 | 0.26 | met |

| server side (from histogram buckets, ms) | p50 | p95 | p99 |
|---|---|---|---|
| rp_gateway_overhead_seconds (learned run) | 0.744 (0.464-1.000] | 1.954 (1.000-2.154] | 3.659 (2.154-4.642] |
| rp_decision_duration_seconds (learned run) | 0.065 (0.046-0.100] | 0.097 (0.046-0.100] | 0.100 (0.046-0.100] |

Policy version served: dea4486d7575+ef0a989c.

## Limitations

- One run per cell: there are no repetitions and no confidence intervals. A difference of a fraction of a millisecond is within the run-to-run noise of a shared machine.
- The generator, the gateway and the mock compete for the same CPU, and other programs on this machine could not be controlled. The launch lag table shows how much the generator itself added.
- The gateway writes one log line per request to a file; accounting, the decision log and tracing are off, and the tenant has no limits (see bench/configs/overhead.yaml).
- The server-side quantiles have the resolution of the histogram buckets (about a factor of 2.15 per bucket), so a p95 can be undecided against the target.
- The mock answers with constant delays and tiny bodies; real providers add variable latency, and larger bodies would cost the gateway more to copy.
- All requests carry one fixed short prompt.
