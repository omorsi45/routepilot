# ADR 0008: Circuit breakers with tickets, and a bulkhead for non-streaming attempts

Status: Accepted, 2026-10-26. The ticket design and the bulkhead were settled during the resilience work on 2026-10-27.

## Context

The gateway fails over between deployments, so it needs to stop sending traffic to a deployment that is failing and to find out when it recovers. Three things make a plain counter unsafe here:

- Many requests are in flight on one deployment. An attempt that started before the breaker opened can finish after it, and if that late outcome were applied to the current state it could settle another caller's half-open probe, close a breaker that should stay open, or count in a window it did not belong to.
- A slow deployment holds a goroutine and a connection for every request sent to it. Without a limit, one deployment that stalls can absorb every request slot of the process.
- The client chooses the request deadline. A tenant that sends a very short timeout must not be able to open a breaker that every tenant shares.

## Decision

- **One breaker per deployment**, in `internal/reliability`. It opens after 5 consecutive failures, or when at least half of at least 20 requests in a 30 second window failed. It stays open for 10 seconds, then admits exactly one probe at a time. A failed probe reopens it for twice as long, up to 120 seconds, and a successful probe closes it and clears its history.
- **Tickets.** `Allow` returns a ticket that carries the breaker's generation at claim time and whether it holds the probe. Every state transition increments the generation (closed to open, open to half-open, the probe's outcome or abort, and every forced trip). `Record` and `Abort` do nothing for a ticket from an older generation, so a late outcome from before a transition resolves nothing. The zero ticket is what a refused `Allow` returns, and resolving it is a no-op.
- **A probe held by a stream** is settled as a success when the stream commits, since the deployment has produced content. A later failure of that stream is an SSE error event for that client and is not recorded against the breaker.
- **Blame for timeouts.** A timeout caused by the request deadline counts against the deployment only if the attempt ran for at least 90% of the deadline the request would have had without a client-chosen `timeout_ms` (the server default, clamped to the tenant maximum). A shorter run was cut off by the client and says nothing about the deployment. Watchdogs on the provider side (time to first token, idle time) always count.
- **Retry-After.** A value of up to 2 seconds is waited out. A longer one trips that deployment for that long, capped at 120 seconds, and the loop moves on. A 401 or 403 from the provider quarantines the deployment for 120 seconds, because it points at a credential problem.
- **Bulkhead.** A deployment may set `max_inflight`. The gateway keeps a counter per limited deployment (a compare-and-swap loop, no lock), takes a slot before a non-streaming attempt and gives it back when the provider call returns, on every path and at most once. A full deployment is skipped as if unavailable, and the rejection is counted. The bulkhead is in `internal/gateway`, not in `reliability`: it is about capacity, not health, and takes no part in the breaker's bookkeeping. Streams do not use it.
- Breaker state is exported as a metric per deployment (`rp_breaker_state`), and so is the error-rate average the health registry keeps, with a 30 second half-life (`rp_deployment_error_rate`).

## Consequences

- Failover: in 10,000 requests with injected failures before the stream (deployment A failing 20% of its requests, deployment B never), 1,749 first attempts failed and all 1,749 were answered by B, with no request failing. A's breaker was seen open once during the run, and 1,321 requests started on B. ([failover](../results.md#failover-against-injected-failures)). The failures are independent and B never failed, so this does not show behavior when both deployments are unhealthy.
- No second upstream call after the first byte: of 2,000 streams, A cut 411 after the commit and none of those made a second upstream call. This is the rule the stream design depends on.
- A breaker per deployment is shared by all tenants, so one tenant's failures can open a deployment for everyone. The deadline rule above limits what a client-chosen timeout can do, not what a tenant's bad requests can do.
- The bulkhead covers non-streaming requests only. A late failure of a probe stream, after it committed, is not recorded. The outcome of a stale ticket is ignored by the window and by the probe, but it still feeds the error-rate average of the health registry. All four are listed in [limitations.md](../limitations.md#reliability).
- The breaker constants and the guardrail thresholds are fixed values from the design and were not tuned on live traffic.
- The Anthropic adapter's error classification is tested against documented wire formats and has never run against the live API.
