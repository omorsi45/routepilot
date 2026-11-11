# ADR 0007: A linear predictor per model; the neural predictor is deferred

Status: Accepted, 2026-10-28. The embedding-based neural predictor is deferred, not rejected.

## Context

The router needs, for each model and request, a calibrated probability of success that Go can evaluate in tens of microseconds from a plain array file ([ADR 0001](0001-go-data-plane-python-training.md)). The design allowed the production predictor to change only if an ablation showed a material gain that Go could serve, and asked for the decision to be recorded here. The candidates were variations of the shipped design (feature parts, hashed width, calibrator) and a neural network on text embeddings.

## Decision

- One L2-regularized logistic regression per model, on the hashed and dense features of [ADR 0004](0004-hashed-features.md). The penalty is chosen by 5-fold cross-validation inside the train split.
- A calibrator per model, fitted on the calibration split by a rule fixed in advance: isotonic regression when the model has at least 1,000 calibration samples, Platt scaling below that. All 12 flagship models got isotonic regression.
- A lower-confidence-bound table per model ([ADR 0002](0002-lcb-admission.md)), a ridge regression on `log1p(completion tokens)` for the output length, and one out-of-distribution detector per pool (k-means with 32 centroids).
- The ablation outcome did not change the production configuration. The calibration rule stays as fixed, although Platt scaling scored better on one measure (below).
- The PyTorch multilayer perceptron on embeddings is deferred and was not run. The project has no PyTorch dependency, and an embedding model on the request path is not something the Go gateway can serve at the latency target. It would be reconsidered if a Go-servable way to compute the embeddings exists and an offline run shows a material gain over this table.

## Consequences

- Measured, on the flagship test split with each variant retrained and its dual operating point retuned to the same calibration target ([ablations](../results.md#ablations), [calibration study](../results.md#calibration-study)):
  - The shipped configuration (both feature parts, 65,536 buckets, the fixed calibration rule) was not beaten on cost by any variant. The width and the feature parts are in [ADR 0004](0004-hashed-features.md).
  - Platt scaling for every model has the lowest pooled test ECE at every calibration size, including the full split: 0.0115 against 0.0160 for isotonic regression. The shipped artifact has 0.0160. With Platt scaling for every model, test success rises by 2.25 points [+0.98, +3.46] and the mean cost is 63.7% higher. Which of the two is better depends on how the extra success is valued against the extra cost, and the fixed rule was kept.
  - Pooled test ECE of the shipped artifact is 0.0160 (target below 0.05, met); per model it is 0.026 to 0.054.
- Two contextual bandits, LinUCB and linear Thompson sampling, learning from one reward per item, did not beat the supervised policy on the test stream, with or without a shift in the middle ([bandits](../results.md#contextual-bandits)). Mean reward per item was 0.4524 for the supervised policy, 0.4413 for LinUCB and 0.4302 for Thompson sampling. Both chose the lowest value of their hyperparameter grid, so a better setting may lie below the grid, and the supervised policy was trained with the outcome of every model on every item, which the bandits do not see.
- At the tuned lambda, the learned policy has no reward edge over one fixed model: its mean reward is 0.4524 per item against 0.4537 for the single model with the best reward on train and calibration, and it spends about five times as much per item ([limitations](../limitations.md#routing-and-prediction)).
- The predictor is small, inspectable and quick to retrain (134 seconds for the flagship pool in the committed run). Its capacity is limited, and whether a richer predictor would raise the quality of the cheap choices is untested. The one planned experiment that could have said so is the deferred one.
- The artifact stores only arrays, so a future predictor that Go can serve from arrays fits the same format with a new format version.
