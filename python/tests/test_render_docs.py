"""Tests of bench/scripts/render_docs.py: run selection, labels, header and output hygiene.

The result trees here are invented and tiny. They contain the files a real run holds
(run.json, results.json, report.md) and nothing else.
"""

import json
import re
import sys
from pathlib import Path

import pytest

from routepilot_train.runinfo import repo_root

SCRIPTS = repo_root() / "bench" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import render_docs as rd  # noqa: E402

EM_DASH = "\u2014"
EN_DASH = "\u2013"

OLD = "20261001T000000Z-aaaaaaa"
NEW = "20261002T000000Z-bbbbbbb"


def make_run(
    results_dir: Path,
    experiment: str,
    run_id: str,
    value: str,
    labels: str = "invented labels",
    limitation: str | None = None,
    figure: str | None = None,
) -> Path:
    """Write one fake run directory and return it."""
    run_dir = results_dir / experiment / run_id
    run_dir.mkdir(parents=True)
    info = {"run_id": run_id, "experiment": experiment, "git_sha": "a" * 40, "git_dirty": False}
    (run_dir / "run.json").write_text(json.dumps(info), encoding="utf-8", newline="\n")
    (run_dir / "results.json").write_text(
        json.dumps({"value": value}), encoding="utf-8", newline="\n"
    )
    lines = [f"# {experiment}", "", f"Run `{run_id}`.", "", f"**Labels:** {labels}.", ""]
    if limitation is not None:
        lines += ["## Limitations", "", f"- {limitation}", ""]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    if figure is not None:
        (run_dir / figure).write_bytes(b"\x89PNG\r\n")
    return run_dir


def alpha_section(doc: str = "results.md") -> rd.Section:
    def render(run: rd.Run) -> list[str]:
        return [f"value: {run.results['value']}"]

    return rd.Section(doc=doc, experiment="alpha", title="Alpha", render=render)


def write_selection(results_dir: Path, text: str) -> None:
    (results_dir / "SELECTION.yaml").write_text(text, encoding="utf-8", newline="\n")


def build(results_dir: Path, *sections: rd.Section) -> str:
    return rd.build_docs(results_dir, list(sections) or [alpha_section()])["results.md"]


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    path = tmp_path / "bench" / "results"
    path.mkdir(parents=True)
    return path


def test_render_uses_newest_run_and_labels(results_dir: Path) -> None:
    make_run(results_dir, "alpha", OLD, "OLD-VALUE", labels="old labels")
    make_run(results_dir, "alpha", NEW, "NEW-VALUE", labels="new labels, recorded and simulated")

    text = build(results_dir)

    assert "value: NEW-VALUE" in text
    assert "value: OLD-VALUE" not in text
    assert f"[`{NEW}`](../bench/results/alpha/{NEW}/)" in text
    # The Labels line is the run's own line, character for character.
    assert "**Labels:** new labels, recorded and simulated." in text.splitlines()
    assert "old labels" not in text
    # The run left out of the tables is still named, so nothing disappears silently.
    assert f"`{OLD}`" in text


def test_selection_entry_wins_over_the_newest_run(results_dir: Path) -> None:
    make_run(results_dir, "alpha", OLD, "OLD-VALUE")
    make_run(results_dir, "alpha", NEW, "NEW-VALUE")
    write_selection(
        results_dir, f'alpha:\n  run: "{OLD}"\n  reason: the older run is the one to read\n'
    )

    text = build(results_dir)

    assert "value: OLD-VALUE" in text
    assert "value: NEW-VALUE" not in text
    assert "the older run is the one to read" in text


def test_selection_shows_two_runs_and_lists_the_superseded_one(results_dir: Path) -> None:
    third = "20261003T000000Z-ccccccc"
    make_run(results_dir, "alpha", OLD, "OLD-VALUE")
    make_run(results_dir, "alpha", NEW, "NEW-VALUE")
    make_run(results_dir, "alpha", third, "THIRD-VALUE")
    write_selection(
        results_dir,
        "alpha:\n"
        f'  - run: "{third}"\n'
        "    status: current\n"
        "    reason: current run\n"
        f'  - run: "{NEW}"\n'
        "    status: also\n"
        "    reason: also shown for comparison\n"
        f'  - run: "{OLD}"\n'
        "    status: older\n"
        "    reason: superseded because the setup changed\n",
    )

    text = build(results_dir)

    assert "value: THIRD-VALUE" in text
    assert "value: NEW-VALUE" in text
    assert "value: OLD-VALUE" not in text
    assert text.index("THIRD-VALUE") < text.index("NEW-VALUE")
    assert "also shown for comparison" in text
    assert f"[`{OLD}`](../bench/results/alpha/{OLD}/): superseded because the setup changed" in text


def test_missing_selected_run_fails(results_dir: Path) -> None:
    make_run(results_dir, "alpha", NEW, "NEW-VALUE")
    write_selection(
        results_dir, 'alpha:\n  run: "20250101T000000Z-0000000"\n  reason: not in the tree\n'
    )

    with pytest.raises(rd.RenderError, match="20250101T000000Z-0000000"):
        build(results_dir)


def test_selection_of_an_unknown_experiment_fails(results_dir: Path) -> None:
    make_run(results_dir, "alpha", NEW, "NEW-VALUE")
    write_selection(results_dir, f'beta:\n  run: "{NEW}"\n  reason: no such experiment\n')

    with pytest.raises(rd.RenderError, match="beta"):
        build(results_dir)


def test_selection_entry_needs_a_one_line_reason(results_dir: Path) -> None:
    make_run(results_dir, "alpha", NEW, "NEW-VALUE")
    write_selection(results_dir, f'alpha:\n  run: "{NEW}"\n')
    with pytest.raises(rd.RenderError, match="reason"):
        build(results_dir)

    write_selection(results_dir, f'alpha:\n  run: "{NEW}"\n  reason: "two\\nlines"\n')
    with pytest.raises(rd.RenderError, match="reason"):
        build(results_dir)


def test_experiment_without_a_section_fails(results_dir: Path) -> None:
    make_run(results_dir, "alpha", NEW, "NEW-VALUE")
    make_run(results_dir, "gamma", NEW, "GAMMA-VALUE")

    with pytest.raises(rd.RenderError, match="gamma"):
        build(results_dir)


def test_generated_header_is_first_and_says_so(results_dir: Path) -> None:
    make_run(results_dir, "alpha", NEW, "NEW-VALUE")

    text = build(results_dir)
    lines = text.splitlines()

    assert (
        lines[0] == "<!-- Generated by bench/scripts/render_docs.py from bench/results/. "
        "Do not edit by hand. -->"
    )
    assert lines[0] == rd.GENERATED_COMMENT
    head = "\n".join(lines[:6])
    assert "generated by `bench/scripts/render_docs.py` from the committed runs" in head
    assert "Do not edit" in head


def test_output_is_lf_and_has_no_dash_characters(results_dir: Path, tmp_path: Path) -> None:
    make_run(results_dir, "alpha", NEW, "NEW-VALUE", limitation="one run per cell")
    out_dir = tmp_path / "docs"

    written = rd.write_docs(results_dir, out_dir, [alpha_section()])

    assert [p.name for p in written] == ["results.md"]
    raw = (out_dir / "results.md").read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    text = raw.decode("utf-8")
    assert EM_DASH not in text
    assert EN_DASH not in text


@pytest.mark.parametrize("dash", [EM_DASH, EN_DASH])
def test_a_dash_in_a_run_text_fails_instead_of_shipping(results_dir: Path, dash: str) -> None:
    make_run(results_dir, "alpha", NEW, "NEW-VALUE", labels=f"before {dash} after")

    with pytest.raises(rd.RenderError, match="dash"):
        build(results_dir)


def test_figures_of_a_shown_run_are_linked(results_dir: Path) -> None:
    make_run(results_dir, "alpha", NEW, "NEW-VALUE", figure="curve.png")

    text = build(results_dir)

    assert f"![curve](../bench/results/alpha/{NEW}/curve.png)" in text


def test_the_limitations_of_a_report_are_copied(results_dir: Path) -> None:
    make_run(results_dir, "alpha", NEW, "NEW-VALUE", limitation="one run per cell")

    text = build(results_dir)

    assert "- one run per cell" in text.splitlines()


def test_a_run_without_a_labels_line_uses_the_results_labels_or_says_nothing(
    results_dir: Path,
) -> None:
    run_dir = make_run(results_dir, "alpha", NEW, "NEW-VALUE")
    (run_dir / "report.md").write_text("# alpha\n", encoding="utf-8", newline="\n")
    (run_dir / "results.json").write_text(
        json.dumps({"value": "V", "labels": ["first label", "second label"]}),
        encoding="utf-8",
        newline="\n",
    )

    text = build(results_dir)

    assert "**Labels:** first label; second label." in text.splitlines()


# --- internal references in copied text ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "held-out datasets were fixed before training (Task 3) and appear only in shift",
            "held-out datasets were fixed before training and appear only in shift",
        ),
        (
            "the share of decisions whose floor was not relaxed (Ruling V1-R5).",
            "the share of decisions whose floor was not relaxed.",
        ),
        (
            "prices are an ASSUMPTION (Ruling V1-R42): proportional to the parameter count",
            "prices are an ASSUMPTION: proportional to the parameter count",
        ),
        (
            "against the true value; Ruling V1-R36: the learned target is min_cost",
            "against the true value; the learned target is min_cost",
        ),
        (
            "the logging policy's own predictions (Ruling V1-R32), so its bias includes this",
            "the logging policy's own predictions, so its bias includes this",
        ),
        (
            "prices are fitted (Rulings V1-R2 and V1-R3) from the recorded costs",
            "prices are fitted from the recorded costs",
        ),
        (
            "the bound counts every byte (Ruling M2-R11) and is an allowance",
            "the bound counts every byte and is an allowance",
        ),
        (
            "the bound is decided per Ruling M2-R11 and kept",
            "the bound is decided and kept",
        ),
        ("The bound holds, see Ruling M2-R11.", "The bound holds."),
        ("It is fixed per Task 12 and kept", "It is fixed and kept"),
        ("Splits come from Tasks 3 and 4 and are kept", "Splits come and are kept"),
        ("Ruling V1-R5: the floor was relaxed.", "The floor was relaxed."),
        ("Ruling V1-R5, V1-R6 and V1-R7: three rules", "Three rules"),
        (
            "a plain sentence about a task force and 3 tasks",
            "a plain sentence about a task force and 3 tasks",
        ),
    ],
)
def test_internal_references_are_dropped_from_copied_text(text: str, expected: str) -> None:
    assert rd.scrub_internal_references(text) == expected


def test_labels_and_limitations_lose_their_internal_references(results_dir: Path) -> None:
    make_run(
        results_dir,
        "alpha",
        NEW,
        "NEW-VALUE",
        labels="fixed before training (Task 3); prices are an ASSUMPTION (Ruling V1-R42): fitted",
        limitation="the floor was not relaxed (Ruling V1-R5)",
    )

    text = build(results_dir)

    assert (
        "**Labels:** fixed before training; prices are an ASSUMPTION: fitted." in text.splitlines()
    )
    assert "- the floor was not relaxed" in text.splitlines()
    assert "Ruling" not in text
    assert "Task 3" not in text


def test_an_internal_reference_in_any_other_text_fails_instead_of_shipping(
    results_dir: Path,
) -> None:
    make_run(results_dir, "alpha", NEW, "see Ruling V1-R9 for why")

    with pytest.raises(rd.RenderError, match="internal reference"):
        build(results_dir)


# --- renderers, on small invented results ---------------------------------------------------


def run_of(experiment: str, results: dict, status: str = "current", report: str = "") -> rd.Run:
    return rd.Run(
        experiment=experiment,
        run_id="20261001T000000Z-aaaaaaa",
        path=Path("does-not-exist"),
        status=status,
        reason="",
        results=results,
        info={},
        report=report,
    )


def hist(value: float, lower: float, upper: float | None) -> dict:
    return {"value_ms": value, "lower_ms": lower, "upper_ms": upper}


def server_block(count: float) -> dict:
    return {
        "count": count,
        "mean_ms": 0.1,
        "p50": hist(0.16, 0.1, 0.2154),
        "p95": hist(0.33, 0.2154, 0.4642),
        "p99": hist(1000.0, 1000.0, None),
    }


def load_block(p50: float, lag_p99: float = 1.1) -> dict:
    lat = {"p50": p50, "p90": p50, "p95": p50, "p99": p50, "p99.9": p50, "max": p50}
    return {
        "scheduled": 100,
        "completed": 100,
        "errors": 0,
        "dropped": 0,
        "throughput_rps": 97.9,
        "latency_ms": lat,
        "launch_lag_ms": {"p50": 0.57, "p99": lag_p99, "max": 3.39},
    }


def overhead_results() -> dict:
    pcts = {"p50": 0.5, "p90": 0.6, "p95": 0.7, "p99": 0.8, "p99.9": 0.9}
    return {
        "target_p95_ms": 15.0,
        "settings": {
            "levels": [100],
            "warmup_s": 10,
            "duration_s": 60,
            "max_tokens": 16,
            "learned_rps": 100,
        },
        "runs": [
            {
                "name": "direct-100rps-nonstream",
                "rps": 100,
                "stream": False,
                "model": "bench-model",
                "load": load_block(37.5),
                "server": None,
            },
            {
                "name": "gateway-100rps-nonstream",
                "rps": 100,
                "stream": False,
                "model": "bench-model",
                "load": load_block(38.0),
                "server": {"overhead": server_block(5882.0), "decision": server_block(5876.0)},
            },
        ],
        "comparisons": [
            {
                "kind": "gateway_vs_direct",
                "rps": 100,
                "stream": False,
                "client_diff_ms": pcts,
                "ttft_diff_ms": None,
                "client_verdict": "met",
                "server": server_block(59758.0),
                "server_verdict": "met",
            }
        ],
        "clock": {"clock_step_ns": 30, "zero_fraction": 0.0, "coarse": False},
    }


def test_overhead_counts_print_as_whole_numbers_and_buckets_as_ranges() -> None:
    text = "\n".join(rd.render_overhead(run_of("overhead", overhead_results())))

    assert "| 100 | non-streaming | 59758 | 0.160 (0.100-0.215] | 0.330 (0.215-0.464] |" in text
    assert "1000.000 (1000.000-inf]" in text
    assert "e+" not in text


def test_overhead_shows_the_launch_lag_the_limitations_refer_to() -> None:
    text = "\n".join(rd.render_overhead(run_of("overhead", overhead_results())))

    assert "launch lag" in text.lower()
    assert "| direct-100rps-nonstream | 0.57 | 1.10 | 3.39 |" in text


def test_overhead_decision_cost_names_the_model_each_run_requested() -> None:
    text = "\n".join(rd.render_overhead(run_of("overhead", overhead_results())))

    assert "| gateway-100rps-nonstream | bench-model | 5876 |" in text


def test_overhead_second_run_is_shown_as_a_tail_only() -> None:
    results = overhead_results()
    results["settings"]["levels"] = [100, 2000]
    results["runs"].append(
        {
            "name": "gateway-2000rps-nonstream",
            "rps": 2000,
            "stream": False,
            "model": "bench-model",
            "load": load_block(38.0),
            "server": None,
        }
    )

    text = "\n".join(rd.render_overhead(run_of("overhead", results, status="also")))

    assert "gateway-2000rps-nonstream" in text
    assert "gateway-100rps-nonstream" not in text
    assert "Decision cost" not in text


def headline_results() -> dict:
    row = {
        "n": 100,
        "success_rate": 0.48,
        "success_ci": [0.46, 0.5],
        "mean_cost_usd": 0.004,
        "mean_cost_ci": [0.003, 0.005],
        "success_delta_vs_strongest": -0.0455,
        "success_delta_ci": [-0.0697, -0.0225],
        "cost_reduction_vs_strongest": 0.935,
        "cost_reduction_ci": [0.926, 0.944],
    }
    target = {
        "cost_target_met": True,
        "success_target_met": False,
        "both_met": False,
    }
    return {
        "pool": "flagship",
        "artifact_version": "abc",
        "n_items": {"calib": 10, "test": 100},
        "best_single": "m1",
        "cheapest": "m2",
        "lambda": {
            "source": "config",
            "value": 6.7,
            "calib_success": 0.49,
            "calib_success_at_zero": 0.56,
        },
        "target": {"best_single_calib_success": 0.5, "margin": 0.01, "target": 0.49},
        "targets": {"cost_reduction": 0.25, "success_margin": 0.01},
        "baselines": {"always_strongest": {**row, "success_rate": 0.527}},
        "operating_points": {
            "dual": {"param_name": "lambda", "param": 6.7, "test_row": row, "targets": target},
            "dual_lambda0_reference": {
                "param_name": "lambda",
                "param": 0.0,
                "test_row": row,
                "targets": {"cost_target_met": True, "success_target_met": True, "both_met": True},
            },
        },
        "target_statements": ["learned dual: targets NOT met."],
    }


def test_headline_verdicts_come_from_the_run_and_the_reference_row_has_none() -> None:
    text = "\n".join(rd.render_headline(run_of("headline", headline_results())))

    dual = next(line for line in text.splitlines() if line.startswith("| learned dual |"))
    assert "-4.55 [-6.97, -2.25]" in dual
    assert "93.5% [92.6%, 94.4%]" in dual
    assert dual.endswith("| met | not met | not met |")
    reference = next(line for line in text.splitlines() if "lambda 0 (reference)" in line)
    assert reference.endswith("| reference | reference | reference |")
    assert "- learned dual: targets NOT met." in text.splitlines()
    assert "cost reduction >= 25%" in text
    assert "success within 1 point" in text


def test_quota_shows_dollars_converted_from_picodollars() -> None:
    backend = {
        "clients": {"requests": 10, "admitted": 4, "rejected": 6},
        "ledger": {"rows": 4, "ok_rows": 4, "sum_pico": 49894000000},
        "metrics": {"overshoot_pico": 0, "accounting_write_errors": 0},
        "expected_cost_pico": 49894000000,
        "limit_pico": 50000000000,
        "unused_pico": 106000000,
        "mock_requests": 4,
        "reconcile": {"exit_code": 0},
        "wall_time_s": 1.5,
        "checks": [{"name": "ledger", "passed": True, "detail": "ok"}],
    }
    results = {"settings": {"clients": 64, "budget_usd": 0.05}, "backends": {"redis": backend}}

    text = "\n".join(rd.render_quota(run_of("quota_correctness", results)))

    assert "| spend per the usage ledger (USD) | 0.049894 |" in text
    assert "| unused budget at the end (USD) | 0.000106 |" in text
    assert "1 of 1 checks passed." in text


def test_failover_says_yes_when_every_check_passed() -> None:
    arm = {"scheduled": 10, "rps": 5, "classification": {}, "checks": [], "load": {}, "mocks": {}}
    results = {
        "all_checks_passed": True,
        "settings": {},
        "arms": {"prestream": arm, "midstream": arm},
    }

    text = "\n".join(rd.render_failover(run_of("failover", results)))

    assert "All integrity checks passed: yes." in text


def test_inupper_lists_the_records_above_the_adjusted_bound() -> None:
    adjusted = {
        "prompt_overhead_tokens": {"intern-s1": 124, "gpt-5": 0},
        "per_split": {},
        "worst": [
            {
                "item_id": "arenahard_math/365",
                "split": "test",
                "model": "intern-s1",
                "prompt_tokens": 186,
                "bound": 178,
                "ratio": 1.0449,
                "anomaly": False,
            }
        ],
    }
    pool = {"overall": {"n": 5, "max": 2.0}, "adjusted": adjusted, "worst": []}
    results = {"max_ratio": 2.0, "anomaly_rule": "rule", "pools": {"flagship": pool}}

    text = "\n".join(rd.render_inupper(run_of("inupper", results)))

    assert "records still above 1 after the overhead" in text
    assert "| arenahard_math/365 | test | intern-s1 | 186 | 178 | 1.045 | no |" in text
    assert "intern-s1 124" in text


def test_live_train_prints_the_split_sizes_without_python_syntax() -> None:
    results = {
        "pool": {"models": ["a"], "datasets": ["d"], "prices": {}, "n_items": 3},
        "protocol": {"name": "p", "max_tokens": 16},
        "artifact": {"version": "v", "size_bytes": 5},
        "items_per_split": {"train": 2, "calib": 1, "test": 0},
        "floors": {"values": [0.0]},
        "replay": {"rows": [], "n_items": 0},
        "fit": {"models": {}},
    }

    text = "\n".join(rd.render_live_train(run_of("live_train", results)))

    assert "{'" not in text
    assert "train: 2, calib: 1, test: 0" in text


def test_train_does_not_print_a_calibrator_rule_the_run_did_not_record() -> None:
    results = {
        "pool": "flagship",
        "artifact_version": "abc",
        "dim": 65536,
        "n_train": 10,
        "n_calib": 4,
        "ood": {"k": 32, "threshold": 0.9},
        "wall_time_s": 1.5,
        "models": {},
    }

    text = "\n".join(rd.render_train(run_of("train", results)))

    assert "None" not in text
    assert "calibrator rule" not in text
    with_rule = "\n".join(rd.render_train(run_of("train", {**results, "calibrator": "platt"})))
    assert "calibrator rule `platt`" in with_rule


def clock_results(os_name: str, steps: list[int], zeros: list[float]) -> dict:
    probes = [{"clock_step_ns": s, "zero_fraction": z} for s, z in zip(steps, zeros, strict=True)]
    return {
        "os": os_name,
        "system": os_name.split("-")[0],
        "go_version": "go1.27.2",
        "probe_ms": 200,
        "probes": probes,
        "summary": {
            "probes": len(steps),
            "min_step_ns": min(steps),
            "median_step_ns": sorted(steps)[len(steps) // 2],
            "max_step_ns": max(steps),
            "max_zero_fraction": max(zeros),
            "threshold_ns": 1000,
            "coarse": max(steps) > 1000,
        },
    }


def test_clock_lists_every_probe_and_says_whether_the_step_is_fine_enough() -> None:
    fine = "\n".join(
        rd.render_clock(run_of("clock", clock_results("Linux-6.1-test", [30] * 5, [0.0] * 5)))
    )
    coarse = "\n".join(
        rd.render_clock(
            run_of(
                "clock",
                clock_results(
                    "Windows-11-test", [500_000, 380_000, 501_000, 502_000, 500_500], [0.99999] * 5
                ),
            )
        )
    )

    assert "Linux-6.1-test" in fine
    assert fine.count("| 30 | 0.0000% |") == 5
    assert "fine enough" in fine
    assert "Windows-11-test" in coarse
    assert "| 2 | 380000 | 99.9990% |" in coarse
    assert "too coarse" in coarse
    assert "380000, 500500 and 502000 ns" in coarse


# --- the committed tree ----------------------------------------------------------------------

REPO = repo_root()


def test_every_run_of_the_committed_tree_renders() -> None:
    docs = rd.build_docs(REPO / "bench" / "results")

    assert set(docs) == {"results.md", "performance.md"}
    for text in docs.values():
        assert text.startswith(rd.GENERATED_COMMENT + "\n")
        assert EM_DASH not in text and EN_DASH not in text
    selection = rd.load_selection(REPO / "bench" / "results")
    for experiment, entries in selection.items():
        for entry in entries:
            if entry.status != "older":
                assert entry.run_id in docs["results.md"] + docs["performance.md"], experiment


def test_committed_docs_are_the_generator_output_byte_for_byte() -> None:
    docs = rd.build_docs(REPO / "bench" / "results")

    for name, text in docs.items():
        committed = (REPO / "docs" / name).read_bytes()
        assert committed == text.encode("utf-8"), (
            f"docs/{name} is stale: run bench/scripts/render_docs.py"
        )


def test_committed_docs_carry_no_internal_references() -> None:
    for name in ("results.md", "performance.md", "limitations.md"):
        text = (REPO / "docs" / name).read_text(encoding="utf-8")
        assert not re.search(r"Rulings?\b|\bTasks? \d|\b[VM]\d-R\d+", text), name


def test_the_training_runs_of_both_pools_are_shown() -> None:
    selection = rd.load_selection(REPO / "bench" / "results")

    for experiment in ("train", "mf_train"):
        assert [e.status for e in selection[experiment]] == ["current", "also"], experiment


def test_performance_doc_shows_the_clock_runs_of_both_hosts() -> None:
    text = (REPO / "docs" / "performance.md").read_text(encoding="utf-8")

    assert "## Clock resolution" in text
    assert "Windows-" in text and "Linux-" in text
    selection = rd.load_selection(REPO / "bench" / "results")
    assert [e.status for e in selection["clock"]] == ["current", "also"]
