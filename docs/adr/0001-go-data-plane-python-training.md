# ADR 0001: Go for the data plane, Python for training

Status: Accepted, 2026-10-26.

## Context

The gateway sits on the request path of every call. It has to be one process with predictable latency, cheap concurrency, streaming I/O and a single deployable binary. Fitting the prediction models needs numerical libraries (a linear-model solver, isotonic regression, k-means, a dataframe library) that Go does not offer in the form the experiments need. The prediction has to run per request and has to be the same computation that the offline experiments evaluate, or the offline results do not describe the shipped router.

## Decision

- The gateway, the CLI (`serve`, `replay`, `policy`, `reconcile`), the mock provider and the load generator are Go (`cmd/`, `internal/`).
- Ingest, features, training, calibration, export, evaluation, off-policy estimators and bandits are Python (`python/routepilot_train`, dependencies pinned with `==`). No Python runs on the request path, and the gateway never calls a model server.
- The two halves meet in three places only:
  - a content-addressed artifact (`manifest.json` and `weights.bin`, see [the artifact format](../architecture.md#the-artifact-format-and-content-hash)) that Python writes and Go loads;
  - feature spec v1, implemented in both languages and pinned by golden vectors in `testdata/features/` (see [ADR 0004](0004-hashed-features.md));
  - `routepilot replay`, which runs the Go router, catalog and optimizer over recorded outcomes and writes JSON lines that Python reads to compute metrics.
- Go only evaluates. A model is eligible for the artifact if Go can score it with sparse dot products, interpolation and table lookups.
- Go refuses an artifact whose hash, feature spec version or golden predictions do not match (`TestLoadRejectsHashMismatch`, `TestLoadRejectsGoldenMismatch`), so a drift between the two implementations stops the load instead of changing a decision.

## Consequences

- The request path has no interpreter and no extra network hop. Measured on Linux (WSL2) over loopback against a mock with constant delays, the router's decision took 0.056 ms on average with the learned policy, and the p95 overhead of the gateway over the direct path was 0.29 to 0.64 ms without streaming. These are single runs on one machine ([performance](../performance.md#gateway-overhead-against-the-local-mock)).
- Offline results describe the shipped decision code. In the headline run, the Python and Go predictions agree to floating-point precision on the test split (largest absolute difference 5.60663e-15 over 1,713 items, [results](../results.md#headline-operating-points-and-targets)). The replay covers the decision and not the whole request; see [the architecture](../architecture.md#two-halves-and-one-artifact).
- Two implementations of the features and the scoring have to stay equal. The guards are the golden vectors, the golden predictions stored in every artifact and re-checked at load, and the parity checks inside the experiments. They do not remove the cost: a change to the feature spec is a change in both languages plus a retrain.
- The model family is limited to what Go evaluates cheaply. This is the reason a neural predictor on embeddings was not built ([ADR 0007](0007-predictor-choice.md)).
- Trained artifacts are not bit-for-bit reproducible across platforms. Linear algebra and math library routines differ in the last bits between Windows and Linux, which moves weights slightly and changes the content hash, while predictions stay equal within the tolerances the tests state. Go still verifies the exact bytes of any artifact it is given. [reproducibility.md](../reproducibility.md) has the details.
- Contributors need two toolchains (Go 1.27.2 and Python 3.14 for the pipeline). `make verify` runs both.
