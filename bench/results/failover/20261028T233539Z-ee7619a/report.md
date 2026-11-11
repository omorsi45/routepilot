# Failover against injected failures (20261028T233539Z-ee7619a)

Labels: real local measurement, nothing simulated. Machine: AMD Ryzen 9 7900X 12-Core Processor (24 threads), 31.2 GB RAM, Windows-11-10.0.26300-SP0, go1.27.2. The load generator (rp-load), the gateway (routepilot serve) and the two mocks (rp-mock) share one machine and talk over loopback. The mocks' delays are configured constants (time to first token 20 ms, 1 ms per output token, 16 output tokens). The experiment measures counts, not time; the latency columns are the client's view and carry the Retry-After waits.

## Pre-stream arm: failures injected on deployment A

10000 non-streaming requests at 100 requests per second (open loop, Poisson arrivals, seeded). Deployment A injects, per request: 10% HTTP 500, 5% HTTP 429 with Retry-After 1 s and 5% connection resets (20% in all); deployment B injects nothing. max_attempts is 3.

Target: at least 99% of the requests whose first attempt failed with an eligible failure are recovered. It is a target, not a guarantee; this is one run.

| quantity | value |
|---|---|
| requests | 10000 |
| requests started on A | 8679 |
| eligible failures on A (attempts) | 1749 (http_500: 875, connect: 472, http_429: 402) |
| requests whose first attempt failed eligibly | 1749 |
| of those, recovered (answered, on B) | 1749 (1749 on B) |
| per-request recovery rate (recovered / first attempt failed eligibly) | 100.00% |
| target >= 99% | met |
| requests routed straight to B (A's breaker refused them, Ruling M2-R21) | 1321 (13.21% of requests) |
| A's breaker opened (polled every 100 ms) | 1 times |
| requests that failed overall | 0 (none) |
| of the failed-first requests, not recovered | 0 (none) |
| client-side errors (load generator) | 0 |

The recovery rate counts only the requests that tried A first and failed there. A request the breaker sent straight to B never failed on A, so it is not in the numerator or the denominator; the line above says how many there were, because the breaker is the mechanism that keeps the failing deployment out of the path.

Attempts per deployment and outcome (decision log) and the mocks' own request counts:

| deployment | attempts in the decision log | outcomes | requests the mock received |
|---|---|---|---|
| A | 8679 | ok: 6930, http_500: 875, connect: 472, http_429: 402 | 8679 |
| B | 3070 | ok: 3070 | 3070 |

Responses by status: 200: 10000. Attempts per request: 1: 8251, 2: 1749.

Integrity checks:

| check | result | detail |
|---|---|---|
| schedule: the generator scheduled exactly 10000 requests | pass | scheduled 10000 |
| decision log: one record per scheduled request and none dropped | pass | 10000 records for 10000 requests, 0 dropped by the writer |
| drops: the generator dropped no arrival | pass | dropped 0 |
| mock counts: each mock received the attempts the log shows for its deployment | pass | mock A 8679 vs log 8679; mock B 3070 vs log 3070 |
| client errors: the generator's error count equals the requests that failed overall | pass | generator 0, decision log 0 |

## Mid-stream arm: deployment A drops the connection after the first content chunk

2000 streaming requests at 50 requests per second. Deployment A's upstream model carries the mock directive fail=midstream;fail_after=2: the mock sends the role chunk and the first content chunk, then drops the connection. The gateway commits a response at the first chunk with content, so every stream that A served was already committed when it broke. Deployment B is clean.

Rule under test: after the first byte the gateway makes no second upstream call. While A's breaker is closed every request starts on A; once A's breaker opens, later requests start on B, which is allowed (they never touched A).

| quantity | value |
|---|---|
| requests | 2000 |
| attempts in the decision log | 2000 |
| attempts per request | 1: 2000 |
| requests started on A (committed, then the stream broke) | 26 |
| requests started on B (A's breaker refused them) | 1974 |
| requests with attempts on both deployments | 0 |
| requests the mocks received (A, B, sum) | 26, 1974, 2000 |
| A's breaker opened (polled every 100 ms) | 3 times |
| client-side errors (stream ended without the end marker) | 26 |

Integrity checks:

| check | result | detail |
|---|---|---|
| schedule: the generator scheduled exactly 2000 requests | pass | scheduled 2000 |
| decision log: one record per scheduled request and none dropped | pass | 2000 records for 2000 requests, 0 dropped by the writer |
| drops: the generator dropped no arrival | pass | dropped 0 |
| one attempt: every request made exactly one attempt | pass | attempts per request {'1': 2000} |
| both: no request attempted both deployments | pass | 0 requests |
| mock counts: the mocks together received one request per client request, each the ones the log shows | pass | mock A 26 vs log 26; mock B 1974 vs log 1974; sum 2000 vs 2000 records |
| committed: every request that started on A committed and then lost its stream | pass | 26 broken streams, 26 requests started on A |
| client errors: the generator's error count equals the streams that broke after commit | pass | generator 26, decision log 26 |
| breaker: B received requests only if A's breaker opened | pass | B received 1974 requests, A's breaker opened 3 times |

## Client-side latency (ms, real measurement)

| arm | completed | errors | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| pre-stream | 10000 | 0 | 36.80 | 65.25 | 83.84 | 1037.31 | 1038.34 |
| mid-stream | 1974 | 26 | 42.53 | 43.68 | 43.94 | 44.29 | 48.26 |

Wall time: pre-stream 107.0 s, mid-stream 40.8 s; the whole run 157.1 s.

## Limitations

- One run per arm, no repetitions: the failures are drawn from a seeded random number generator, but the order in which concurrent requests reach it varies, so another run has the same rates and different individual failures.
- The mocks are deterministic and fast; real providers fail in correlated ways (a whole region, a deploy) that independent per-request failures do not model.
- Deployment B never fails here, so recovery can only fail through the retry limit, the deadline or a breaker on B; it says nothing about the case where both deployments fail.
- Windows loopback on one shared machine. The numbers are counts and do not depend on the timer resolution.
