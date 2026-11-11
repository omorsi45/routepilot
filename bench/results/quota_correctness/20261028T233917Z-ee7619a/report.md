# Quota correctness under 64 concurrent clients (20261028T233917Z-ee7619a)

Labels: real local measurement, nothing simulated. Machine: AMD Ryzen 9 7900X 12-Core Processor (24 threads), 31.2 GB RAM, Windows-11-10.0.26300-SP0, go1.27.2. The load clients, the gateway (routepilot serve), the mock (rp-mock) and Redis share one machine and talk over loopback. The experiment measures counts and money, not time.

Setup: one tenant with a total budget of 0.05 USD and TPM and concurrency limits far above reach. 64 clients (threads), each sending requests with a seeded random max_tokens in [8, 512] until the gateway has answered it 20 times in a row with 429 insufficient_quota. The model costs 1 USD per million input tokens and 2 USD per million output tokens. The mock answers every request with 8 completion tokens (upstream model bench-model;tokens=8, because the mock does not honor max_tokens and 8 is the smallest cap a client sends), so the worst case reserved for a request is always at least what it then costs. Money is integer picodollars (1e-12 USD).

## Result

- memory: 7 of 9 checks passed, 2 check(s) failed: rows: the ledger has one row per admitted request; cost: the ledger sum equals the cost recomputed from the tokens the clients saw.
- redis: 7 of 10 checks passed, 3 check(s) failed: reconcile: the usage ledger and the Redis budget counter agree (exit 0); rows: the ledger has one row per admitted request; cost: the ledger sum equals the cost recomputed from the tokens the clients saw.

| quantity | memory backend | redis backend |
|---|---|---|
| requests sent by the 64 clients | 2546 | 2535 |
| admitted (200) | 1084 | 1084 |
| rejected (429 insufficient_quota) | 1462 | 1451 |
| other responses | 0 | 0 |
| committed spend recomputed from the tokens the clients saw (independent of the gateway) | 0.049864 USD | 0.049864 USD |
| committed spend per the usage ledger (sum of cost_actual) | 0.044206 USD | 0.044436 USD |
| committed spend per the Redis budget counter (from reconcile) | - | 0.049864 USD |
| budget limit | 0.05 USD | 0.05 USD |
| unused budget at the end (limit minus the recomputed committed spend) | 0.000136 USD (0.27% of the limit) | 0.000136 USD (0.27% of the limit) |
| rp_budget_overshoot_picousd_total (picodollars) | 0 | 0 |
| rate-limit rejections rpm/tpm/concurrency | 0/0/0 | 0/0/0 |
| usage ledger rows | 961 (961 ok), 123 fewer than admitted | 966 (966 ok), 118 fewer than admitted |
| usage ledger write errors (rp_accounting_write_errors_total) | 127 | 121 |
| requests the mock received | 1084 | 1084 |
| worst-case reservation (cost_estimate) of the ledger's rows, min to max | 0.000141 to 0.001149 USD | 0.000141 to 0.001149 USD |
| routepilot reconcile (exit; tenant ledger_usd counter_usd diff_usd) | exit 2 | exit 1; quota-redis-20261028t233917z 0.044436 0.049864 -0.005428 |
| wall time | 6.4 s | 3.3 s |

## Backend: memory

routepilot reconcile cannot read the memory backend's counter, which lives in the gateway process, so it exits 2 by design. For this backend the ledger sum against the limit and the overshoot counter are the evidence.

reconcile output: `routepilot reconcile: quota.backend is memory, and the memory budget counter lives in the gateway process, so it cannot be read from here; use quota.backend redis`

Usage ledger rows missing on the memory backend: the ledger holds 961 rows for 1084 admitted requests, 123 rows fewer. The gateway counted 127 ledger write errors (rp_accounting_write_errors_total) and logged 127 warnings covering 127 rows; the causes in the log are: context deadline exceeded: 127. The gateway writes a request's rows synchronously after the request, with a 250 ms bound per request, through the one connection of the SQLite store, so concurrent writes queue behind each other; a write that does not finish within the bound is dropped and counted instead of failing the request. The budget counter is not affected by a lost row: it is charged before the ledger write.

| check | result | detail |
|---|---|---|
| ledger: the committed spend (sum of cost_actual) is within the limit | pass | 0.044206 USD against a limit of 0.05 USD |
| spend: the spend recomputed from the tokens the clients saw is within the limit | pass | 0.049864 USD against a limit of 0.05 USD |
| overshoot: rp_budget_overshoot_picousd_total is 0 | pass | 0 picodollars |
| reconcile: refuses the memory backend by design (exit 2) | pass | exit 2 |
| unexpected responses: every answer was a 200 or a 429 insufficient_quota | pass | other responses: none |
| rows: the ledger has one row per admitted request | FAIL | 961 rows (961 ok) for 1084 admitted requests |
| mock: the provider received exactly the admitted requests | pass | mock received 1084, admitted 1084 |
| rate limit: no RPM, TPM or concurrency limit fired | pass | rejections by kind: {'concurrency': 0, 'rpm': 0, 'tpm': 0, 'upstream': 0} |
| cost: the ledger sum equals the cost recomputed from the tokens the clients saw | FAIL | ledger 44206000000, recomputed 49864000000 picodollars |

## Backend: redis

Redis: real Redis 7.4 (Docker, loopback port).

reconcile output: `quota-redis-20261028t233917z 0.044436 0.049864 -0.005428`

Usage ledger rows missing on the redis backend: the ledger holds 966 rows for 1084 admitted requests, 118 rows fewer. The gateway counted 121 ledger write errors (rp_accounting_write_errors_total) and logged 121 warnings covering 121 rows; the causes in the log are: context deadline exceeded: 121. The gateway writes a request's rows synchronously after the request, with a 250 ms bound per request, through the one connection of the SQLite store, so concurrent writes queue behind each other; a write that does not finish within the bound is dropped and counted instead of failing the request. The budget counter is not affected by a lost row: it is charged before the ledger write.

| check | result | detail |
|---|---|---|
| ledger: the committed spend (sum of cost_actual) is within the limit | pass | 0.044436 USD against a limit of 0.05 USD |
| spend: the spend recomputed from the tokens the clients saw is within the limit | pass | 0.049864 USD against a limit of 0.05 USD |
| overshoot: rp_budget_overshoot_picousd_total is 0 | pass | 0 picodollars |
| counter: the Redis budget counter equals the spend recomputed from the tokens the clients saw | pass | counter ['0.049864'] USD, recomputed 0.049864 USD |
| reconcile: the usage ledger and the Redis budget counter agree (exit 0) | FAIL | exit 1: quota-redis-20261028t233917z ledger 0.044436 counter 0.049864 diff -0.005428 |
| unexpected responses: every answer was a 200 or a 429 insufficient_quota | pass | other responses: none |
| rows: the ledger has one row per admitted request | FAIL | 966 rows (966 ok) for 1084 admitted requests |
| mock: the provider received exactly the admitted requests | pass | mock received 1084, admitted 1084 |
| rate limit: no RPM, TPM or concurrency limit fired | pass | rejections by kind: {'concurrency': 0, 'rpm': 0, 'tpm': 0, 'upstream': 0} |
| cost: the ledger sum equals the cost recomputed from the tokens the clients saw | FAIL | ledger 44436000000, recomputed 49864000000 picodollars |

Wall time of the whole run: 26.2 s.

## Limitations

- One run per backend and one seed: the clients race, so the interleaving (and with it which requests are admitted) differs between runs. The checks do not depend on it.
- Every answer costs the same 8 completion tokens, so what varies between requests is the reservation (the worst case at the client's max_tokens), not the actual cost. A provider that honors max_tokens would make the actual cost vary too.
- The 20-in-a-row stop rule ends a client while budget may remain that only a smaller max_tokens could use; the unused budget above is what was left when the last client stopped.
- A single Redis instance on the same machine: no network latency, no failover, no cluster.
