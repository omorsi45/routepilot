# RoutePilot and LiteLLM against the same mock (20261028T234116Z-ee7619a)

Labels: real local measurement, nothing simulated. Machine: AMD Ryzen 9 7900X 12-Core Processor (24 threads), 15.2 GB RAM, Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39, go1.27.2. The load generator (rp-load), the gateway under test and the mock (rp-mock) share one machine and talk over loopback, in one session so the three paths share conditions. The mock's delays are configured constants (time to first token 20 ms, 1 ms per output token, 16 output tokens), so absolute latencies are the mock's; only the differences between paths measure a gateway. Latencies run from each request's scheduled time (open loop, coordinated-omission safe). Measured with rp-load --clock-probe before the run, the clock step is 30 ns (the smallest non-zero difference between consecutive time.Now reads; 0.0000 of the consecutive reads were equal).

Settings: 10 s warmup (sent, not recorded) and 60 s measurement window per cell, one run per cell, a fresh mock and a fresh proxy or gateway process for every cell, the request body of the overhead experiment.

LiteLLM: version 1.106.6 (read from the x-litellm-version header of the running proxy), Python 3.14.9, started with --num_workers 8 (LiteLLM ran with 8 workers: 8 Python processes against RoutePilot's one Go process), LITELLM_LOCAL_MODEL_COST_MAP=True and LITELLM_LOG=ERROR, telemetry off, drop_params on, no database and no master key. With no master key the proxy does no key lookup per request, while RoutePilot checks the tenant key on every request, so that setting can only favor LiteLLM. RoutePilot runs as in the overhead experiment (one pinned model, no limits, no decision log, accounting off).

Vendor claims (LiteLLM's own benchmark) are not restated as measured here; the numbers below are this machine's.

## Runs (client side, ms)

Latency percentiles are over the completed requests of the measurement window; a cell with errors or drops says so in its own columns. 'gateway' is RoutePilot.

| run | scheduled | completed | errors | dropped | throughput rps | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| direct-100rps-nonstream | 5876 | 5876 | 0 | 0 | 97.9 | 37.60 | 38.17 | 38.37 | 38.75 | 46.08 | 55.87 |
| gateway-100rps-nonstream | 5876 | 5876 | 0 | 0 | 97.9 | 37.92 | 38.53 | 38.66 | 38.88 | 39.13 | 39.58 |
| litellm-100rps-nonstream | 5876 | 5876 | 0 | 0 | 97.9 | 43.62 | 46.81 | 48.64 | 52.00 | 57.34 | 60.29 |
| direct-100rps-stream | 5876 | 5876 | 0 | 0 | 97.9 | 39.13 | 39.97 | 40.22 | 40.77 | 41.60 | 62.08 |
| gateway-100rps-stream | 5876 | 5876 | 0 | 0 | 97.9 | 40.22 | 41.73 | 42.49 | 50.53 | 59.84 | 68.61 |
| litellm-100rps-stream | 5876 | 5876 | 0 | 0 | 97.9 | 48.03 | 54.37 | 58.17 | 69.63 | 84.09 | 96.51 |
| direct-500rps-nonstream | 29900 | 29900 | 0 | 0 | 498.3 | 37.34 | 37.98 | 38.17 | 38.56 | 44.51 | 51.17 |
| gateway-500rps-nonstream | 29900 | 29900 | 0 | 0 | 498.3 | 37.66 | 38.34 | 38.53 | 38.85 | 42.27 | 46.88 |
| litellm-500rps-nonstream | 29900 | 29900 | 0 | 0 | 498.3 | 50.30 | 78.97 | 106.81 | 191.49 | 295.68 | 390.91 |
| direct-500rps-stream | 29900 | 29900 | 0 | 0 | 498.3 | 41.60 | 43.04 | 43.49 | 44.41 | 45.50 | 46.91 |
| gateway-500rps-stream | 29900 | 29900 | 0 | 0 | 498.3 | 43.07 | 44.93 | 45.50 | 46.62 | 47.90 | 49.95 |
| litellm-500rps-stream | 29900 | 25960 | 463 | 3477 | 432.7 | 6881.28 | 10141.69 | 15728.64 | 26820.61 | 29802.49 | 30015.49 |
| direct-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 37.28 | 37.92 | 38.08 | 38.34 | 38.62 | 41.73 |
| gateway-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 37.50 | 38.17 | 38.37 | 38.62 | 38.88 | 40.70 |
| litellm-1000rps-nonstream | 59708 | 59708 | 0 | 0 | 995.1 | 61.25 | 93.18 | 107.20 | 150.91 | 361.21 | 446.98 |
| direct-1000rps-stream | 59708 | 59708 | 0 | 0 | 995.1 | 43.23 | 44.99 | 45.50 | 46.46 | 47.58 | 55.26 |
| gateway-1000rps-stream | 59708 | 59708 | 0 | 0 | 995.1 | 43.26 | 45.41 | 46.05 | 47.20 | 48.54 | 50.59 |
| litellm-1000rps-stream | 59708 | 22858 | 2407 | 34443 | 381.0 | 5656.57 | 6447.10 | 11780.09 | 28459.01 | 29818.88 | 29999.10 |
| direct-2000rps-nonstream | 119503 | 119503 | 0 | 0 | 1991.7 | 37.15 | 37.76 | 37.92 | 38.24 | 38.59 | 46.17 |
| gateway-2000rps-nonstream | 119503 | 119503 | 0 | 0 | 1991.7 | 37.47 | 38.24 | 38.49 | 54.72 | 109.63 | 158.08 |
| litellm-2000rps-nonstream | 119503 | 72889 | 2855 | 43759 | 1214.8 | 1510.40 | 2086.91 | 2322.43 | 16326.66 | 28049.41 | 29949.95 |
| direct-2000rps-stream | 119503 | 119503 | 0 | 0 | 1991.7 | 43.33 | 45.41 | 46.08 | 67.52 | 114.43 | 158.97 |
| gateway-2000rps-stream | 119503 | 119503 | 0 | 0 | 1991.7 | 41.82 | 43.71 | 44.29 | 45.50 | 47.04 | 50.24 |
| litellm-2000rps-stream | 119503 | 24336 | 1903 | 93264 | 405.6 | 5611.52 | 6414.34 | 10280.96 | 26214.40 | 29769.73 | 29999.10 |

Time to first token (streaming runs, ms):

| run | p50 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| direct-100rps-stream | 21.39 | 22.05 | 22.19 | 22.41 | 22.59 | 26.13 |
| gateway-100rps-stream | 21.84 | 22.54 | 22.80 | 24.75 | 34.78 | 39.81 |
| litellm-100rps-stream | 27.36 | 30.86 | 32.99 | 39.55 | 51.01 | 71.30 |
| direct-500rps-stream | 21.28 | 22.00 | 22.19 | 22.50 | 22.86 | 25.54 |
| gateway-500rps-stream | 21.68 | 22.48 | 22.73 | 23.30 | 24.21 | 25.87 |
| litellm-500rps-stream | 1642.49 | 3053.57 | 11419.65 | 19218.43 | 22085.63 | 22478.85 |
| direct-1000rps-stream | 21.17 | 21.93 | 22.18 | 22.62 | 23.21 | 32.70 |
| gateway-1000rps-stream | 21.61 | 22.54 | 22.88 | 23.73 | 25.07 | 27.34 |
| litellm-1000rps-stream | 2193.41 | 3180.54 | 11649.02 | 27836.42 | 29622.27 | 29835.26 |
| direct-2000rps-stream | 21.05 | 21.97 | 22.29 | 27.52 | 59.87 | 75.45 |
| gateway-2000rps-stream | 21.50 | 22.53 | 22.89 | 23.84 | 25.47 | 31.23 |
| litellm-2000rps-stream | 2057.22 | 2639.87 | 10133.50 | 23199.74 | 29573.12 | 29868.03 |

## Latency added by each gateway: difference of percentiles (ms)

Label: difference of percentiles, not a percentile of differences. Each value is the percentile seen through one path minus the same percentile seen through the other at the same rate and mode. Values near the histogram resolution (about 0.03 ms at 40 ms) are not signal.

| rate | mode | comparison | p50 | p90 | p95 | p99 | p99.9 |
|---|---|---|---|---|---|---|---|
| 100 | non-streaming | RoutePilot minus direct | 0.32 | 0.35 | 0.29 | 0.13 | -6.94 |
| 100 | non-streaming | LiteLLM minus direct | 6.02 | 8.64 | 10.27 | 13.25 | 11.26 |
| 100 | non-streaming | LiteLLM minus RoutePilot | 5.70 | 8.29 | 9.98 | 13.12 | 18.21 |
| 100 | streaming | RoutePilot minus direct | 1.09 | 1.76 | 2.27 | 9.76 | 18.24 |
| 100 | streaming | LiteLLM minus direct | 8.90 | 14.40 | 17.95 | 28.86 | 42.50 |
| 100 | streaming | LiteLLM minus RoutePilot | 7.81 | 12.64 | 15.68 | 19.10 | 24.26 |
| 500 | non-streaming | RoutePilot minus direct | 0.32 | 0.35 | 0.35 | 0.29 | -2.24 |
| 500 | non-streaming | LiteLLM minus direct | 12.96 | 40.99 | 68.64 | 152.93 | 251.17 |
| 500 | non-streaming | LiteLLM minus RoutePilot | 12.64 | 40.64 | 68.29 | 152.64 | 253.41 |
| 500 | streaming | RoutePilot minus direct | 1.47 | 1.89 | 2.02 | 2.21 | 2.40 |
| 500 | streaming | LiteLLM minus direct | 6839.68 | 10098.66 | 15685.15 | 26776.19 | 29756.99 |
| 500 | streaming | LiteLLM minus RoutePilot | 6838.21 | 10096.77 | 15683.14 | 26773.98 | 29754.59 |
| 1000 | non-streaming | RoutePilot minus direct | 0.22 | 0.26 | 0.29 | 0.29 | 0.26 |
| 1000 | non-streaming | LiteLLM minus direct | 23.97 | 55.26 | 69.12 | 112.58 | 322.59 |
| 1000 | non-streaming | LiteLLM minus RoutePilot | 23.74 | 55.01 | 68.83 | 112.29 | 322.34 |
| 1000 | streaming | RoutePilot minus direct | 0.03 | 0.42 | 0.54 | 0.74 | 0.96 |
| 1000 | streaming | LiteLLM minus direct | 5613.34 | 6402.11 | 11734.59 | 28412.54 | 29771.30 |
| 1000 | streaming | LiteLLM minus RoutePilot | 5613.31 | 6401.70 | 11734.05 | 28411.81 | 29770.34 |
| 2000 | non-streaming | RoutePilot minus direct | 0.32 | 0.48 | 0.58 | 16.48 | 71.04 |
| 2000 | non-streaming | LiteLLM minus direct | 1473.25 | 2049.15 | 2284.51 | 16288.42 | 28010.82 |
| 2000 | non-streaming | LiteLLM minus RoutePilot | 1472.93 | 2048.67 | 2283.94 | 16271.94 | 27939.78 |
| 2000 | streaming | RoutePilot minus direct | -1.50 | -1.70 | -1.79 | -22.02 | -67.39 |
| 2000 | streaming | LiteLLM minus direct | 5568.19 | 6368.93 | 10234.88 | 26146.88 | 29655.30 |
| 2000 | streaming | LiteLLM minus RoutePilot | 5569.70 | 6370.62 | 10236.67 | 26168.90 | 29722.69 |

Time to first token, difference of percentiles (ms, streaming):

| rate | comparison | p50 | p90 | p95 | p99 | p99.9 |
|---|---|---|---|---|---|---|
| 100 | RoutePilot minus direct | 0.45 | 0.50 | 0.61 | 2.34 | 12.19 |
| 100 | LiteLLM minus direct | 5.97 | 8.82 | 10.80 | 17.14 | 28.42 |
| 100 | LiteLLM minus RoutePilot | 5.52 | 8.32 | 10.19 | 14.80 | 16.22 |
| 500 | RoutePilot minus direct | 0.40 | 0.48 | 0.54 | 0.80 | 1.34 |
| 500 | LiteLLM minus direct | 1621.22 | 3031.57 | 11397.46 | 19195.94 | 22062.77 |
| 500 | LiteLLM minus RoutePilot | 1620.82 | 3031.09 | 11396.91 | 19195.14 | 22061.42 |
| 1000 | RoutePilot minus direct | 0.45 | 0.61 | 0.70 | 1.10 | 1.86 |
| 1000 | LiteLLM minus direct | 2172.24 | 3158.61 | 11626.85 | 27813.79 | 29599.06 |
| 1000 | LiteLLM minus RoutePilot | 2171.79 | 3158.00 | 11626.14 | 27812.69 | 29597.20 |
| 2000 | RoutePilot minus direct | 0.45 | 0.56 | 0.61 | -3.68 | -34.40 |
| 2000 | LiteLLM minus direct | 2036.16 | 2617.90 | 10111.22 | 23172.22 | 29513.25 |
| 2000 | LiteLLM minus RoutePilot | 2035.71 | 2617.34 | 10110.61 | 23175.90 | 29547.65 |

Generator launch lag (how late the scheduler started a request, ms; it is part of every latency above):

| run | p50 | p99 | max |
|---|---|---|---|
| direct-100rps-nonstream | 0.56 | 1.10 | 7.89 |
| gateway-100rps-nonstream | 0.57 | 1.09 | 1.29 |
| litellm-100rps-nonstream | 0.55 | 1.09 | 2.88 |
| direct-100rps-stream | 0.52 | 1.09 | 4.66 |
| gateway-100rps-stream | 0.47 | 1.11 | 11.33 |
| litellm-100rps-stream | 0.34 | 1.08 | 5.00 |
| direct-500rps-nonstream | 0.48 | 1.09 | 8.78 |
| gateway-500rps-nonstream | 0.47 | 1.09 | 5.25 |
| litellm-500rps-nonstream | 0.45 | 2.98 | 40.51 |
| direct-500rps-stream | 0.36 | 1.12 | 3.82 |
| gateway-500rps-stream | 0.23 | 1.04 | 1.61 |
| litellm-500rps-stream | 0.09 | 0.87 | 4.06 |
| direct-1000rps-nonstream | 0.43 | 1.08 | 1.73 |
| gateway-1000rps-nonstream | 0.40 | 1.07 | 3.40 |
| litellm-1000rps-nonstream | 0.34 | 1.07 | 4.52 |
| direct-1000rps-stream | 0.19 | 1.20 | 2.08 |
| gateway-1000rps-stream | 0.11 | 1.02 | 3.13 |
| litellm-1000rps-stream | 0.10 | 0.91 | 5.54 |
| direct-2000rps-nonstream | 0.40 | 1.07 | 4.31 |
| gateway-2000rps-nonstream | 0.34 | 1.10 | 12.62 |
| litellm-2000rps-nonstream | 0.27 | 1.08 | 4.76 |
| direct-2000rps-stream | 0.12 | 1.30 | 12.25 |
| gateway-2000rps-stream | 0.06 | 0.90 | 2.57 |
| litellm-2000rps-stream | 0.09 | 1.00 | 10.34 |

Wall time of the whole run: 1900.1 s.

## Limitations

- One run per cell, no repetitions and no confidence intervals. A difference of a fraction of a millisecond is within the run-to-run noise of a shared machine.
- Linux under WSL2 in a VM that shares the host with Windows; the generator, the mock and the gateway compete for the same CPU. The launch lag table shows how much the generator itself added.
- LiteLLM's numbers depend on its worker count (8 here) and its settings; another count would give other numbers. Only this version and these settings were measured.
- The mock answers with constant delays and tiny bodies; real providers add variable latency, and larger bodies cost a gateway more to copy.
- All requests carry one fixed short prompt.
