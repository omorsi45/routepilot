# ADR 0002: Admit on a lower confidence bound of quality

Status: Accepted, 2026-10-26.

## Context

A caller can ask for a quality floor: "route this request to a model that succeeds on at least this share of requests like it". A router that compares a point estimate of success with the floor treats a probability backed by a handful of calibration samples like one backed by hundreds. The calibration split is finite, so the estimate has sampling error, and an error in the optimistic direction makes the router pick a cheaper model than the floor allows.

Other designs the project looked at: a point `min_score` floor on a static quality index (the closest gateway, see [comparison](../comparison.md)), complexity tiers mapped to models, and a learned policy with a floor given as a training-time input. In the pages and papers read for the comparison, none states a bound that accounts for the size of the sample behind the quality number.

## Decision

- Per model, the predictor gives a calibrated probability `p` (Platt scaling or isotonic regression, chosen by the calibration sample count, see [ADR 0007](0007-predictor-choice.md)).
- On the calibration split, requests are sorted by calibrated `p` and cut into 15 equal-mass bins. Bins with fewer than 30 samples are merged with a neighbour. Each bin stores the one-sided 95% Clopper-Pearson lower bound of its success rate, made non-decreasing across bins by a running maximum. `L` for a request is the value of the bin that holds its `p`.
- Admission compares the floor with `L`, never with `p`. The constrained program is: minimize expected cost over the models whose `L` is at least the floor, whose latency quantile meets the SLO and whose worst-case cost fits the remaining budget (see [the constrained program](../architecture.md#the-constrained-program)). A second mode, `dual`, maximizes `p` minus lambda times expected cost inside the same safe set, for a traffic-level target.
- An empty safe set is handled by the tenant's `on_infeasible` setting: `best_effort` (default) drops the latency constraint and then the floor and reports it in `x-routepilot-relaxed`, and `reject` answers 422 with the binding constraints. Capability, permission and budget are never relaxed.
- Under distribution shift, an out-of-distribution score above the calibrated threshold replaces `p` and `L` with each model's pooled calibration rate and pooled lower bound.
- The point comparison exists only as an experiment switch (`WithFloorOnPoint`) so the two arms can be compared. The gateway never calls it.
- Operating points are chosen on the train and calibration splits only. For `dual`, lambda is found by bisection on the calibration split so that calibration success reaches Best Single's calibration success minus one point.

## Consequences

- The bound is conservative, and the price is paid in relaxed requests. At a floor of 0.50 on the test split the router relaxed 59.4% of requests with the lower bound against 36.3% with the point estimate, and the mean cost was 0.03016 USD against 0.02753 USD ([lower confidence bound against point estimate](../results.md#lower-confidence-bound-against-point-estimate)).
- The experiment did not show that the bound is needed to meet a floor. Both arms met all 12 tested floors (0.05 to 0.60) on the test split, measured as realized success of the requests whose floor was not relaxed. The bound arm has the larger margin at each floor and costs more. A relaxed request made no promise, so the compliance figure says nothing about it.
- At the headline operating point the learned policy is 93.5% cheaper than Best Single and 4.55 points less successful, so the quality half of the target was not met ([headline](../results.md#headline-operating-points-and-targets)). The tuned lambda is optimistic: the calibrators and bound tables were fitted on the same split that lambda is tuned on. A cross-fitted variant (calibrators fitted on half of the calibration items, lambda tuned on the other half) did not do better, at 5.07 points below Best Single.
- A static rules mapping chosen on the same data beat Best Single on both axes, so the learned policy has not been shown to beat the simplest baseline ([ADR 0005](0005-static-rules-baseline.md)).
- The bound describes the calibration distribution. It is not a guarantee for traffic that looks different. The out-of-distribution detector that is meant to catch such traffic is close to inert on this data (AUROC 0.557 for held-out datasets against the test split, [distribution shift](../results.md#distribution-shift-and-out-of-distribution-detection)).
- Callers must read `x-routepilot-relaxed`. A response that carries it was routed without the floor, or without the latency constraint, that the request asked for.
- Every decision logs `p`, `L` and the other terms for each candidate, which is what the off-policy evaluation in [ADR 0007](0007-predictor-choice.md) and the promotion gate use.
