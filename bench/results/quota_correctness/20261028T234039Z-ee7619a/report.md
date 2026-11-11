# Quota correctness under 8 concurrent clients (20261028T234039Z-ee7619a)

Control run: 8 clients instead of the briefed 64, to see whether what differs in the briefed run depends on the concurrency. It is not the briefed run.

Labels: real local measurement, nothing simulated. Machine: AMD Ryzen 9 7900X 12-Core Processor (24 threads), 31.2 GB RAM, Windows-11-10.0.26300-SP0, go1.27.2. The load clients, the gateway (routepilot serve), the mock (rp-mock) and Redis share one machine and talk over loopback. The experiment measures counts and money, not time.

Setup: one tenant with a total budget of 0.05 USD and TPM and concurrency limits far above reach. 8 clients (threads), each sending requests with a seeded random max_tokens in [8, 512] until the gateway has answered it 20 times in a row with 429 insufficient_quota. The model costs 1 USD per million input tokens and 2 USD per million output tokens. The mock answers every request with 8 completion tokens (upstream model bench-model;tokens=8, because the mock does not honor max_tokens and 8 is the smallest cap a client sends), so the worst case reserved for a request is always at least what it then costs. Money is integer picodollars (1e-12 USD).

## Result

- memory: 9 of 9 checks passed, every check passed.
- redis: 10 of 10 checks passed, every check passed.

| quantity | memory backend | redis backend |
|---|---|---|
| requests sent by the 8 clients | 1427 | 1491 |
| admitted (200) | 1083 | 1084 |
| rejected (429 insufficient_quota) | 344 | 407 |
| other responses | 0 | 0 |
| committed spend recomputed from the tokens the clients saw (independent of the gateway) | 0.049818 USD | 0.049864 USD |
| committed spend per the usage ledger (sum of cost_actual) | 0.049818 USD | 0.049864 USD |
| committed spend per the Redis budget counter (from reconcile) | - | 0.049864 USD |
| budget limit | 0.05 USD | 0.05 USD |
| unused budget at the end (limit minus the recomputed committed spend) | 0.000182 USD (0.36% of the limit) | 0.000136 USD (0.27% of the limit) |
| rp_budget_overshoot_picousd_total (picodollars) | 0 | 0 |
| rate-limit rejections rpm/tpm/concurrency | 0/0/0 | 0/0/0 |
| usage ledger rows | 1083 (1083 ok), 0 fewer than admitted | 1084 (1084 ok), 0 fewer than admitted |
| usage ledger write errors (rp_accounting_write_errors_total) | 0 | 0 |
| requests the mock received | 1083 | 1084 |
| worst-case reservation (cost_estimate) of the ledger's rows, min to max | 0.000141 to 0.001149 USD | 0.000141 to 0.001149 USD |
| routepilot reconcile (exit; tenant ledger_usd counter_usd diff_usd) | exit 2 | exit 0; quota-redis-20261028t234039z 0.049864 0.049864 0 |
| wall time | 4.4 s | 3.4 s |

## Backend: memory

routepilot reconcile cannot read the memory backend's counter, which lives in the gateway process, so it exits 2 by design. For this backend the ledger sum against the limit and the overshoot counter are the evidence.

reconcile output: `routepilot reconcile: quota.backend is memory, and the memory budget counter lives in the gateway process, so it cannot be read from here; use quota.backend redis`

| check | result | detail |
|---|---|---|
| ledger: the committed spend (sum of cost_actual) is within the limit | pass | 0.049818 USD against a limit of 0.05 USD |
| spend: the spend recomputed from the tokens the clients saw is within the limit | pass | 0.049818 USD against a limit of 0.05 USD |
| overshoot: rp_budget_overshoot_picousd_total is 0 | pass | 0 picodollars |
| reconcile: refuses the memory backend by design (exit 2) | pass | exit 2 |
| unexpected responses: every answer was a 200 or a 429 insufficient_quota | pass | other responses: none |
| rows: the ledger has one row per admitted request | pass | 1083 rows (1083 ok) for 1083 admitted requests |
| mock: the provider received exactly the admitted requests | pass | mock received 1083, admitted 1083 |
| rate limit: no RPM, TPM or concurrency limit fired | pass | rejections by kind: {'concurrency': 0, 'rpm': 0, 'tpm': 0, 'upstream': 0} |
| cost: the ledger sum equals the cost recomputed from the tokens the clients saw | pass | ledger 49818000000, recomputed 49818000000 picodollars |

## Backend: redis

Redis: real Redis 7.4 (Docker, loopback port).

reconcile output: `quota-redis-20261028t234039z 0.049864 0.049864 0`

| check | result | detail |
|---|---|---|
| ledger: the committed spend (sum of cost_actual) is within the limit | pass | 0.049864 USD against a limit of 0.05 USD |
| spend: the spend recomputed from the tokens the clients saw is within the limit | pass | 0.049864 USD against a limit of 0.05 USD |
| overshoot: rp_budget_overshoot_picousd_total is 0 | pass | 0 picodollars |
| counter: the Redis budget counter equals the spend recomputed from the tokens the clients saw | pass | counter ['0.049864'] USD, recomputed 0.049864 USD |
| reconcile: the usage ledger and the Redis budget counter agree (exit 0) | pass | exit 0: quota-redis-20261028t234039z ledger 0.049864 counter 0.049864 diff 0 |
| unexpected responses: every answer was a 200 or a 429 insufficient_quota | pass | other responses: none |
| rows: the ledger has one row per admitted request | pass | 1084 rows (1084 ok) for 1084 admitted requests |
| mock: the provider received exactly the admitted requests | pass | mock received 1084, admitted 1084 |
| rate limit: no RPM, TPM or concurrency limit fired | pass | rejections by kind: {'concurrency': 0, 'rpm': 0, 'tpm': 0, 'upstream': 0} |
| cost: the ledger sum equals the cost recomputed from the tokens the clients saw | pass | ledger 49864000000, recomputed 49864000000 picodollars |

Wall time of the whole run: 17.8 s.

## Limitations

- One run per backend and one seed: the clients race, so the interleaving (and with it which requests are admitted) differs between runs. The checks do not depend on it.
- Every answer costs the same 8 completion tokens, so what varies between requests is the reservation (the worst case at the client's max_tokens), not the actual cost. A provider that honors max_tokens would make the actual cost vary too.
- The 20-in-a-row stop rule ends a client while budget may remain that only a smaller max_tokens could use; the unused budget above is what was left when the last client stopped.
- A single Redis instance on the same machine: no network latency, no failover, no cluster.
