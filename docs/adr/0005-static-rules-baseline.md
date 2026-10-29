# ADR 0005: A static rules mapping is the baseline to beat

Status: Accepted, 2026-10-27.

## Context

A learned router is worth shipping only if it beats something simple that costs nothing to run. Published router comparisons are sensitive to how that simple comparator is chosen: picking the best single model on the evaluation labels inflates the gain of a router, and a recent paper sizes that effect for safety routers under distribution shift ([comparison](../comparison.md)). So the comparators have to be chosen without the test split, and the simplest useful routing rule has to be among them.

## Decision

- Best Single is the model with the highest success on the train and calibration splits. The cheapest model is the one with the lowest mean recorded cost per item on train and calibration, which is the model that is cheapest per request in practice (ordering by price or by worst-case cost picks a different model, and both orderings are reported as separate labeled rows). Both baselines run through the replay as a fixed-model policy.
- Static rules is a serving and replay policy (`rules`) with an ordered rule set that was fixed before any model was trained. The first rule whose conditions all hold wins; the rule matches a lowercased keyword in the first 32 KiB of the message text, or a minimum prompt size:

| Rule | Condition |
|---|---|
| `code` | contains any of a code fence, `def `, `class `, `#include`, `function `, `import `, `public static` |
| `math` | contains any of `\frac`, `$$`, `\(`, `\sqrt`, `\sum`, `integer`, `equation` |
| `choices` | contains any of `(a)`, `a) `, `a. `, `options:` |
| `long` | the prompt has at least 4,000 bytes |

- Only the model each rule maps to is learned: among the train and calibration items whose first matching rule is that rule, the model with the highest success (ties go to the lower mean recorded cost, then the name). A rule that first-matches fewer than 30 such items maps to Best Single, and so does the default.
- The Python mapping and the Go policy are checked against each other on every item of the pool, and against shared cases in `testdata/rules/cases.json`.
- The other baselines are a seeded random policy, RouteLLM matrix factorization with a threshold chosen on calibration, and an oracle that reads the recorded outcomes (an upper bound, not a policy).

## Consequences

- On the flagship test split (1,735 items), static rules reached 54.6% success against 52.7% for Best Single (+1.90 points [+0.69, +3.17]) at a mean cost 34.4% lower [30.9%, 38.0%]. The rules map `code`, `math`, `choices` and `long` to gpt-5 and the default to gemini-2.5-pro ([baselines](../results.md#baselines)).
- That row meets the numbers of the headline targets (a cost reduction of at least 25% with success within one point of Best Single), and the learned policy does not ([headline](../results.md#headline-operating-points-and-targets)). On this data the simple baseline is the better trade, and the learned policy has not been shown to beat it ([ADR 0002](0002-lcb-admission.md)).
- On the mixed pool every rule maps to gpt-5, which is also Best Single, so the rules save nothing there.
- The rule set is a builder's choice, and a different set could do better or worse. The benchmark's datasets differ in format (multiple choice, code, math), so rules that detect a format may partly detect the dataset. The data cannot separate the two, so the result says nothing about how the rules would do on traffic that does not look like these datasets.
- Parity between the Python mapping and the Go policy was 0 mismatches over 12,067 items (flagship pool) and 3,182 items (mixed pool).
- A `rules` policy in a gateway config is only as good as the mapping its operator writes. The gateway ships the mechanism, and the mapping in the replay configs is generated from the data.
