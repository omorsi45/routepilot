"""Tests for the off-policy evaluation validation script."""

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import fixture_pool
import numpy as np
import pandas as pd
import pytest

from routepilot_train import ope, replaytool
from routepilot_train.runinfo import repo_root

ROOT = repo_root()
SCRIPT = ROOT / "bench" / "scripts" / "ope_validation.py"
LEARNED = ROOT / "testdata" / "replay" / "learned"
TINY_ARTIFACT = fixture_pool.TINY_ARTIFACT
LAMBDA = 100000.0
PICO = 1e12


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ope_validation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- pure helpers ---------------------------------------------------------------------------------


def test_the_design_is_fixed_in_advance():
    script = load_script()
    assert tuple(script.SEEDS) == tuple(range(50))
    assert script.EPSILON == 0.1
    assert script.FLOOR == 0.3  # Ruling V1-R36: the plan's 0.8 relaxes nearly every decision
    assert script.BOOTSTRAP == 500
    assert script.ESTIMATORS == ("ips", "snips", "dr")
    assert script.METRICS == ("success", "cost_usd")


def test_target_specs_are_the_four_policies_of_ruling_r36():
    script = load_script()
    pool = {"best_single": "gemini-2.5-pro", "cheapest": "deepseek-v3-0324"}
    learned = ["--artifact", "A", "--policy-config", "P"]

    specs = script.target_specs(pool, learned, 0.3)

    assert [s[0] for s in specs] == [
        "always_strongest",
        "always_cheapest",
        "static_rules",
        "learned_min_cost",
    ]
    assert specs[0][1:] == ("fixed", ["--model", "gemini-2.5-pro"])
    assert specs[1][1:] == ("fixed", ["--model", "deepseek-v3-0324"])
    assert specs[2][1:] == ("rules", [])
    # deterministic: epsilon 0 is explicit, whatever the policy config says
    assert specs[3][1:] == (
        "learned",
        [*learned, "--mode", "min_cost", "--floor", "0.3", "--epsilon", "0"],
    )


def test_logging_flags_are_dual_at_the_spec_lambda_with_epsilon_seed_and_candidates():
    script = load_script()
    assert script.logging_flags(6.5, 0.1, 7) == [
        "--mode",
        "dual",
        "--lambda",
        "6.5",
        "--epsilon",
        "0.1",
        "--seed",
        "7",
        "--emit-candidates",
    ]


def test_snips_estimate_is_the_weighted_mean_with_a_bootstrap_interval():
    script = load_script()
    # weights 1/propensity on the matches: 2, 4 and 0 (the third item disagrees)
    est = script.snips_estimate(
        [1.0, 0.0, 1.0], ["a", "a", "b"], [0.5, 0.25, 1.0], ["a", "a", "a"], n_boot=200, seed=0
    )
    assert est["mean"] == pytest.approx(2 / 6)
    assert est["n"] == 3 and est["lo"] <= est["hi"]
    again = script.snips_estimate(
        [1.0, 0.0, 1.0], ["a", "a", "b"], [0.5, 0.25, 1.0], ["a", "a", "a"], n_boot=200, seed=0
    )
    assert again == est


def test_snips_interval_equals_the_ips_interval_when_every_weight_is_equal():
    script = load_script()
    rng = np.random.default_rng(2)
    r = rng.integers(0, 2, 400).astype(float)
    same = ["a"] * 400
    est = script.snips_estimate(r, same, [0.5] * 400, same, n_boot=300, seed=4)
    reference = ope.estimate(r, n_boot=300, seed=4)  # the mean of r, same resampling stream
    assert est["mean"] == pytest.approx(reference["mean"], abs=1e-12)
    assert est["lo"] == pytest.approx(reference["lo"], abs=1e-12)
    assert est["hi"] == pytest.approx(reference["hi"], abs=1e-12)


def test_snips_interval_resamples_the_weighted_ratio():
    script = load_script()
    rng = np.random.default_rng(8)
    n = 60
    r = rng.integers(0, 2, n).astype(float)
    propensity = rng.choice([0.2, 0.5, 1.0], n)  # unequal weights
    same = ["a"] * n  # every item agrees, so no resample is empty
    est = script.snips_estimate(r, same, propensity, same, n_boot=80, seed=3)

    idx = np.random.default_rng(3).integers(0, n, size=(80, n))
    w = 1.0 / propensity
    ratios = (w * r)[idx].sum(axis=1) / w[idx].sum(axis=1)
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    assert est["mean"] == pytest.approx((w * r).sum() / w.sum())
    assert est["lo"] == pytest.approx(lo, abs=1e-12) and est["hi"] == pytest.approx(hi, abs=1e-12)


def test_snips_estimate_is_nan_when_the_policies_never_agree():
    script = load_script()
    est = script.snips_estimate([1.0, 0.0], ["a", "b"], [0.5, 0.5], ["c", "c"], n_boot=50, seed=0)
    assert math.isnan(est["mean"]) and math.isnan(est["lo"]) and math.isnan(est["hi"])


def test_aggregate_estimates_gives_bias_rmse_coverage_and_width_against_the_truth():
    script = load_script()
    estimates = [
        {"mean": 0.5, "lo": 0.4, "hi": 0.6},  # covers 0.6 (inclusive)
        {"mean": 0.7, "lo": 0.65, "hi": 0.75},  # does not
        {"mean": math.nan, "lo": math.nan, "hi": math.nan},  # the target never matched
    ]
    agg = script.aggregate_estimates(estimates, truth=0.6)

    assert agg["n"] == 2 and agg["n_missing"] == 1
    assert agg["mean_estimate"] == pytest.approx(0.6)
    assert agg["bias"] == pytest.approx(0.0, abs=1e-12)
    assert agg["rmse"] == pytest.approx(0.1)
    assert agg["sd"] == pytest.approx(math.sqrt(0.02))  # sample standard deviation of 0.5, 0.7
    assert agg["coverage"] == 0.5
    assert agg["ci_width"] == pytest.approx(0.15)
    assert agg["bias_rel"] == pytest.approx(0.0, abs=1e-12)
    assert agg["rmse_rel"] == pytest.approx(0.1 / 0.6)


def test_aggregate_estimates_of_a_biased_estimator():
    script = load_script()
    estimates = [{"mean": 0.45, "lo": 0.4, "hi": 0.5}] * 4
    agg = script.aggregate_estimates(estimates, truth=0.6)
    assert agg["bias"] == pytest.approx(-0.15)
    assert agg["rmse"] == pytest.approx(0.15)  # no spread, so RMSE is the absolute bias
    assert agg["sd"] == 0.0 and agg["coverage"] == 0.0


def test_aggregate_estimates_with_no_valid_estimate_has_nothing_to_report():
    script = load_script()
    nan = {"mean": math.nan, "lo": math.nan, "hi": math.nan}
    agg = script.aggregate_estimates([nan, nan], truth=0.6)
    assert agg["n"] == 0 and agg["n_missing"] == 2
    assert agg["bias"] is None and agg["rmse"] is None and agg["coverage"] is None


def write_logged(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def logged_row(item_id: str, model, relaxed, ood, safe, error="") -> dict:
    models = ["m1", "m2", "m3"]
    return {
        "item_id": item_id,
        "model": model,
        "relaxed": relaxed,
        "ood": ood,
        "error": error,
        "candidates": [
            {"model": m, "p": 0.5, "l": 0.4, "cost_exp_picousd": 1, "feasible": m in safe}
            for m in models
        ],
    }


def test_support_is_the_safe_set_of_an_item_that_could_explore_and_the_choice_otherwise(
    tmp_path: Path,
):
    script = load_script()
    path = tmp_path / "logged.jsonl"
    write_logged(
        path,
        [
            logged_row("i1", "m1", [], False, ["m1", "m2"]),  # explores in {m1, m2}
            logged_row("i2", "m3", ["quality"], False, []),  # relaxed: only the choice
            logged_row("i3", "m2", [], True, ["m1", "m2"]),  # out of distribution
            logged_row("i4", None, [], False, [], error="no_feasible_model"),
        ],
    )
    target = pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3", "i4"],
            "model": ["m2", "m1", "m1", "m1"],
            "error": ["", "", "", ""],
        }
    )

    conservative = script.support_coverage(
        script.read_logging_support(path, "conservative"), target
    )
    assert conservative["n"] == 3  # i4 has no logged action
    assert conservative["covered"] == 2  # i1 (m2 is in the safe set) and i3 (it explores too)
    assert conservative["share"] == pytest.approx(2 / 3)
    assert conservative["logged_errors"] == 1 and conservative["target_errors"] == 0

    strongest = script.support_coverage(script.read_logging_support(path, "strongest"), target)
    assert strongest["covered"] == 1  # i3 takes the highest rank and never explores
    assert strongest["share"] == pytest.approx(1 / 3)


def test_support_counts_the_logged_choice_as_covered_even_when_it_did_not_explore(tmp_path: Path):
    script = load_script()
    path = tmp_path / "logged.jsonl"
    write_logged(path, [logged_row("i1", "m3", ["quality"], False, [])])
    target = pd.DataFrame({"item_id": ["i1"], "model": ["m3"], "error": [""]})
    cover = script.support_coverage(script.read_logging_support(path, "conservative"), target)
    assert cover["covered"] == 1 and cover["share"] == 1.0


def test_support_reports_target_errors_apart(tmp_path: Path):
    script = load_script()
    path = tmp_path / "logged.jsonl"
    write_logged(
        path,
        [logged_row("i1", "m1", [], False, ["m1"]), logged_row("i2", "m1", [], False, ["m1"])],
    )
    target = pd.DataFrame({"item_id": ["i1", "i2"], "model": ["m1", None], "error": ["", "boom"]})
    cover = script.support_coverage(script.read_logging_support(path, "conservative"), target)
    assert cover["n"] == 1 and cover["covered"] == 1 and cover["target_errors"] == 1


def test_a_logged_choice_outside_the_safe_set_of_an_exploring_item_is_counted(tmp_path: Path):
    script = load_script()
    path = tmp_path / "logged.jsonl"
    # an item that could explore whose logged model is not feasible would break the support rule
    write_logged(path, [logged_row("i1", "m3", [], False, ["m1", "m2"])])
    support = script.read_logging_support(path, "conservative")
    target = pd.DataFrame({"item_id": ["i1"], "model": ["m1"], "error": [""]})
    assert script.support_coverage(support, target)["logged_outside_safe"] == 1


# --- the whole script on the synthetic pool -------------------------------------------------------


@pytest.fixture
def policy_config(tmp_path: Path) -> Path:
    """The fixture policy config as a dual policy with a lambda that trades quality for cost."""
    text = (LEARNED / "policy.yaml").read_text(encoding="utf-8")
    text = text.replace("mode: min_cost", "mode: dual").replace("lambda: 0", f"lambda: {LAMBDA}")
    path = tmp_path / "policy.yaml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Every replay the script makes: (split, policy, flags)."""
    calls: list[tuple] = []
    real = replaytool.replay

    def spy(binary, config, input, split, policy, out, extra=()):  # noqa: A002
        calls.append((split, policy, tuple(extra)))
        return real(binary, config, input, split, policy, out, extra)

    monkeypatch.setattr(replaytool, "replay", spy)
    return calls


SEEDS = (0, 1, 2)


def run(derived: Path, binary: Path, out: Path, policy_config: Path, seeds=SEEDS) -> Path:
    script = load_script()
    return script.run_ope_validation(
        derived,
        "tiny",
        TINY_ARTIFACT,
        policy_config,
        out,
        ["python", "bench/scripts/ope_validation.py", "--pool", "tiny"],
        binary,
        seeds=seeds,
        bootstrap=60,
        workers=2,
    )


def results_of(out: Path) -> dict:
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


@pytest.fixture
def derived(tmp_path: Path) -> Path:
    return fixture_pool.make_pool(tmp_path / "derived")


def test_outputs_and_the_replays_that_are_made(
    derived: Path, binary: Path, recorded: list[tuple], policy_config: Path, tmp_path: Path
):
    out = run(derived, binary, tmp_path / "out", policy_config)

    assert sorted(p.name for p in out.iterdir()) == [
        "estimator_error.png",
        "report.md",
        "results.json",
        "run.json",
    ]
    assert len(recorded) == len(set(recorded)) == len(SEEDS) + 4
    assert {c[0] for c in recorded} == {"test"}  # nothing but the test split is replayed
    logging = [c for c in recorded if c[1] == "learned" and "--emit-candidates" in c[2]]
    assert len(logging) == len(SEEDS)
    for call in logging:
        flags = list(call[2])
        assert flags[flags.index("--epsilon") + 1] == "0.1"
        assert flags[flags.index("--mode") + 1] == "dual"
        assert flags[flags.index("--lambda") + 1] == repr(LAMBDA)
    assert sorted(int(c[2][c[2].index("--seed") + 1]) for c in logging) == list(SEEDS)
    others = [c for c in recorded if c not in logging]
    assert sorted(c[1] for c in others) == ["fixed", "fixed", "learned", "rules"]
    assert all("--emit-candidates" not in c[2] for c in others)

    run_json = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_json["experiment"] == "ope_validation"
    assert run_json["seeds"]["logging"] == list(SEEDS)
    assert run_json["policy_version"] == TINY_ARTIFACT.name


def test_true_values_are_the_full_information_replays_of_the_targets(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    results = results_of(run(derived, binary, tmp_path / "out", policy_config))

    items = [
        json.loads(line)
        for line in (derived / "replay" / "tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test_items = [i for i in items if i["split"] == "test"]
    n = len(test_items)
    assert results["n_items"] == n == 80
    assert set(results["targets"]) == {
        "always_strongest",
        "always_cheapest",
        "static_rules",
        "learned_min_cost",
    }
    for name, model in (("always_strongest", "tiny-strong"), ("always_cheapest", "tiny-weak")):
        truth = results["targets"][name]["truth"]
        assert truth["success"] == pytest.approx(
            sum(i["outcomes"][model]["success"] for i in test_items) / n
        )
        assert truth["cost_usd"] == pytest.approx(
            sum(i["outcomes"][model]["cost_picousd"] for i in test_items) / n / PICO
        )


def test_estimates_are_recomputed_by_hand_from_the_logged_replays(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    """IPS, SNIPS and DR of always_strongest, from the raw logged file of each seed."""
    out = run(derived, binary, tmp_path / "out", policy_config)
    results = results_of(out)
    script = load_script()
    config = derived / "pools" / "tiny.config.yaml"
    replay_input = derived / "replay" / "tiny.jsonl"

    per_seed = results["targets"]["always_strongest"]["per_seed"]
    assert [row["seed"] for row in per_seed] == list(SEEDS)
    for row in per_seed:
        path = tmp_path / f"by-hand-{row['seed']}.jsonl"
        replaytool.replay(
            binary,
            config,
            replay_input,
            "test",
            "learned",
            path,
            [
                "--artifact",
                str(TINY_ARTIFACT),
                "--policy-config",
                str(policy_config),
                *script.logging_flags(LAMBDA, 0.1, row["seed"]),
            ],
        )
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        n = len(records)
        ips_s = ips_c = dr_s = dr_c = w_sum = w_r = w_c = 0.0
        terms: dict[str, list[float]] = {"ips_s": [], "ips_c": [], "dr_s": [], "dr_c": []}
        for rec in records:
            cands = {c["model"]: c for c in rec["candidates"]}
            match = rec["model"] == "tiny-strong"
            weight = 1.0 / rec["propensity"] if match else 0.0
            success, cost = float(rec["success"]), rec["cost_picousd"] / PICO
            q_log, q_tgt = cands[rec["model"]], cands["tiny-strong"]
            terms["ips_s"].append(weight * success)
            terms["ips_c"].append(weight * cost)
            terms["dr_s"].append(q_tgt["p"] + weight * (success - q_log["p"]))
            terms["dr_c"].append(
                q_tgt["cost_exp_picousd"] / PICO
                + weight * (cost - q_log["cost_exp_picousd"] / PICO)
            )
            w_sum += weight
            w_r += weight * success
            w_c += weight * cost
        ips_s, ips_c = sum(terms["ips_s"]) / n, sum(terms["ips_c"]) / n
        dr_s, dr_c = sum(terms["dr_s"]) / n, sum(terms["dr_c"]) / n
        assert row["success"]["ips"]["mean"] == pytest.approx(ips_s, abs=1e-9)
        assert row["cost_usd"]["ips"]["mean"] == pytest.approx(ips_c, abs=1e-12)
        assert row["success"]["dr"]["mean"] == pytest.approx(dr_s, abs=1e-9)
        assert row["cost_usd"]["dr"]["mean"] == pytest.approx(dr_c, abs=1e-12)
        # the intervals: ope.estimate of the same terms, bootstrap size and seed
        for metric, est, key in (
            ("success", "ips", "ips_s"),
            ("success", "dr", "dr_s"),
            ("cost_usd", "ips", "ips_c"),
            ("cost_usd", "dr", "dr_c"),
        ):
            reference = ope.estimate(np.array(terms[key]), n_boot=60, seed=row["seed"])
            assert row[metric][est]["lo"] == pytest.approx(reference["lo"], abs=1e-9)
            assert row[metric][est]["hi"] == pytest.approx(reference["hi"], abs=1e-9)
        if w_sum > 0:
            assert row["success"]["snips"]["mean"] == pytest.approx(w_r / w_sum, abs=1e-9)
            assert row["cost_usd"]["snips"]["mean"] == pytest.approx(w_c / w_sum, abs=1e-12)
        for metric in ("success", "cost_usd"):
            for est in ("ips", "snips", "dr"):
                cell = row[metric][est]
                if cell["mean"] is not None:
                    assert cell["lo"] <= cell["hi"]


def test_aggregates_are_bias_rmse_and_coverage_of_the_per_seed_estimates(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    results = results_of(run(derived, binary, tmp_path / "out", policy_config))
    for name, target in results["targets"].items():
        truth = target["truth"]
        for metric in ("success", "cost_usd"):
            for est in ("ips", "snips", "dr"):
                cells = [
                    row[metric][est]
                    for row in target["per_seed"]
                    if row[metric][est]["mean"] is not None
                ]
                agg = target["estimators"][metric][est]
                assert agg["n"] == len(cells), (name, metric, est)
                if not cells:
                    continue
                means = np.array([c["mean"] for c in cells])
                assert agg["bias"] == pytest.approx(means.mean() - truth[metric], abs=1e-12)
                assert agg["rmse"] == pytest.approx(
                    math.sqrt(np.mean((means - truth[metric]) ** 2)), abs=1e-12
                )
                covered = [c["lo"] <= truth[metric] <= c["hi"] for c in cells]
                assert agg["coverage"] == pytest.approx(sum(covered) / len(cells))
                assert agg["ci_width"] == pytest.approx(
                    np.mean([c["hi"] - c["lo"] for c in cells]), abs=1e-12
                )


def test_every_item_is_used_and_support_and_weights_are_reported(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    results = results_of(run(derived, binary, tmp_path / "out", policy_config))
    for name, target in results["targets"].items():
        assert target["n_used_min"] == 80, name
        assert target["excluded_total"] == 0, name
        support = target["support"]
        assert support["n"] == 80 and 0 <= support["covered"] <= 80
        assert support["share"] == pytest.approx(support["covered"] / 80)
        # the floor is 0 in the fixture policy, so every model is in the safe set
        assert support["share"] == 1.0, name
        assert support["logged_outside_safe"] == 0
        ess = target["ess"]
        assert 0 <= ess["min"] <= ess["mean"] <= ess["max"] <= 80
        assert 0 <= target["matches_mean"] <= 80


def test_the_report_states_the_design_and_the_caveats(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    out = run(derived, binary, tmp_path / "out", policy_config)
    report = (out / "report.md").read_text(encoding="utf-8")

    assert "Ruling V1-R36" in report
    assert "60 bootstrap resamples" in report  # the number used, stated
    assert "full-information replay" in report
    assert "support coverage" in report.lower()
    assert "no latency is measured or simulated" in report
    assert "Success: bias, RMSE and CI coverage" in report
    assert "Cost (USD per item): bias, RMSE and CI coverage" in report
    assert "the same test items" in report  # the truth is on the items the CIs resample


def test_no_prompt_text_in_the_outputs(
    derived: Path, binary: Path, policy_config: Path, tmp_path: Path
):
    out = run(derived, binary, tmp_path / "out", policy_config)
    texts = [
        json.loads(line)["messages"][0]["content"]
        for line in (derived / "replay" / "tiny.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for name in ("results.json", "report.md", "run.json"):
        body = (out / name).read_text(encoding="utf-8")
        assert not any(text in body for text in texts if len(text) > 12), name


def test_main_uses_the_pool_artifact_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = load_script()
    seen = {}

    def fake(derived, pool, artifact_dir, policy_config, out_root, command, **kwargs):
        seen.update(pool=pool, artifact=artifact_dir, policy=policy_config, **kwargs)
        return tmp_path

    monkeypatch.setattr(script, "run_ope_validation", fake)
    monkeypatch.setattr(script.sweeps, "find_artifact", lambda root, pool: tmp_path / "v1")

    assert script.main(["--pool", "flagship", "--workers", "3"]) == 0
    assert seen["pool"] == "flagship" and seen["artifact"] == tmp_path / "v1"
    assert seen["policy"].name == "flagship.yaml" and seen["workers"] == 3
