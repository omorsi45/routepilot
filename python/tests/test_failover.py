import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from routepilot_train.runinfo import repo_root

SCRIPT = repo_root() / "bench" / "scripts" / "failover.py"
EM_DASH = "\u2014"
A = "deployment-a"
B = "deployment-b"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("failover", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fo() -> ModuleType:
    return load_script()


def rec(status: int, *attempts: tuple[str, str], **extra: object) -> dict[str, object]:
    """A decision record with only the fields the classification reads."""
    return {
        "type": "decision",
        "status": status,
        "attempts": [{"deployment": d, "outcome": o} for d, o in attempts],
        **extra,
    }


# --- which outcomes are eligible failures -----------------------------------------------------


@pytest.mark.parametrize(
    "outcome",
    ["http_500", "http_429", "http_502", "http_503", "http_504", "http_408", "connect", "protocol"],
)
def test_eligible_failures(fo: ModuleType, outcome: str) -> None:
    assert fo.is_eligible_failure(outcome)


@pytest.mark.parametrize(
    "outcome", ["ok", "http_400", "http_404", "http_422", "canceled", "aborted", "error", ""]
)
def test_not_eligible_failures(fo: ModuleType, outcome: str) -> None:
    assert not fo.is_eligible_failure(outcome)


# --- test_classify_attempts: a synthetic decision log -----------------------------------------


def synthetic_log() -> list[dict[str, object]]:
    return [
        *[rec(200, (A, "ok")) for _ in range(5)],  # answered by A at once
        rec(200, (A, "http_500"), (B, "ok")),  # recovered on B
        rec(200, (A, "http_500"), (B, "ok")),
        rec(200, (A, "http_429"), (B, "ok")),
        rec(200, (A, "connect"), (B, "ok")),
        *[rec(200, (B, "ok")) for _ in range(3)],  # the breaker sent these straight to B
        rec(502, (A, "http_500"), (B, "http_500")),  # not recovered
        rec(400, (A, "http_400")),  # failed, but not an eligible failure
        rec(503),  # nothing was sent upstream
    ]


def test_classify_attempts(fo: ModuleType) -> None:
    got = fo.classify_attempts(synthetic_log())

    assert got["requests"] == 15
    assert got["responses_by_status"] == {"200": 12, "400": 1, "502": 1, "503": 1}
    assert got["attempts"] == 5 + 8 + 3 + 2 + 1
    assert got["attempts_per_request"] == {"0": 1, "1": 9, "2": 5}
    assert got["attempts_by_deployment"] == {
        A: {"ok": 5, "http_500": 3, "http_429": 1, "connect": 1, "http_400": 1},
        B: {"ok": 7, "http_500": 1},
    }
    assert got["started_on_a"] == 11
    assert got["started_on_b"] == 3
    assert got["eligible_failures_on_a"] == 5
    assert got["first_attempt_failed_eligibly"] == 5
    assert got["recovered"] == 4
    assert got["recovered_on_b"] == 4
    assert got["recovery_rate"] == pytest.approx(4 / 5)
    assert got["not_recovered"] == {"502:http_500": 1}
    assert got["failed_requests"] == 3
    assert got["failed_by_reason"] == {"502:http_500": 1, "400:http_400": 1, "503:no_attempt": 1}
    assert got["requests_with_attempts_on_both"] == 5
    assert got["committed_then_failed"] == 0


def test_classify_attempts_without_failures_has_no_rate(fo: ModuleType) -> None:
    got = fo.classify_attempts([rec(200, (A, "ok")), rec(200, (B, "ok"))])
    assert got["first_attempt_failed_eligibly"] == 0
    assert got["recovery_rate"] is None
    assert got["not_recovered"] == {}


def test_classify_attempts_empty_log(fo: ModuleType) -> None:
    got = fo.classify_attempts([])
    assert got["requests"] == 0
    assert got["recovery_rate"] is None
    assert got["attempts_per_request"] == {}


def test_classify_counts_a_stream_that_failed_after_commit(fo: ModuleType) -> None:
    # The client got 200 and part of a stream, then the stream broke: the one attempt is not ok.
    log = [rec(200, (A, "protocol")) for _ in range(3)] + [rec(200, (B, "ok")) for _ in range(5)]
    got = fo.classify_attempts(log)
    assert got["committed_then_failed"] == 3
    assert got["attempts_per_request"] == {"1": 8}
    assert got["requests_with_attempts_on_both"] == 0
    assert got["started_on_a"] == 3
    assert got["started_on_b"] == 5
    # A mid-stream failure is not a pre-stream failure that failover could recover: the response
    # was already committed, so it is neither an eligible failure nor a request to recover.
    assert got["failed_requests"] == 0
    assert got["eligible_failures_on_a"] == 0
    assert got["first_attempt_failed_eligibly"] == 0
    assert got["recovery_rate"] is None
    assert got["not_recovered"] == {}


def test_classify_flags_a_second_call_after_a_commit(fo: ModuleType) -> None:
    # A committed (200, a broken stream) and the gateway still called B: the rule this arm exists
    # to check. The request shows two attempts, one on each deployment.
    log = [rec(200, (A, "protocol"), (B, "ok"))]
    got = fo.classify_attempts(log)
    assert got["attempts_per_request"] == {"2": 1}
    assert got["requests_with_attempts_on_both"] == 1


def test_classify_ignores_shadow_and_other_record_types(fo: ModuleType) -> None:
    log = [
        rec(200, (A, "ok")),
        rec(200, (A, "ok"), policy_role="shadow"),
        {"type": "feedback", "decision_id": "dec_x", "success": True},
    ]
    assert fo.classify_attempts(log)["requests"] == 1


# --- reading the decision log -----------------------------------------------------------------


def test_read_decision_records(fo: ModuleType, tmp_path: Path) -> None:
    first = tmp_path / "decisions.20261028T000000Z.jsonl"  # a rotated file sorts first
    active = tmp_path / "decisions.jsonl"
    first.write_text(json.dumps(rec(200, (A, "ok"))) + "\n", encoding="utf-8")
    active.write_text(
        json.dumps(rec(500, (A, "http_500")))
        + "\n"
        + json.dumps({"type": "feedback", "decision_id": "dec_1"})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("not a log", encoding="utf-8")

    got = fo.read_decision_records(tmp_path)
    assert [r["status"] for r in got] == [200, 500]


def test_read_decision_records_rejects_a_torn_line(fo: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "decisions.jsonl").write_text('{"type":"decision","sta', encoding="utf-8")
    with pytest.raises(ValueError, match="decisions.jsonl"):
        fo.read_decision_records(tmp_path)


# --- the breaker, seen through /metrics -------------------------------------------------------

METRICS = """\
# TYPE rp_breaker_state gauge
rp_breaker_state{deployment="deployment-a"} 2
rp_breaker_state{deployment="deployment-b"} 0
rp_requests_total{model="failover-model",policy_role="pinned",status="200",tenant="failover"} 7
"""


def test_parse_breaker_state(fo: ModuleType) -> None:
    assert fo.parse_breaker_state(METRICS, A) == 2
    assert fo.parse_breaker_state(METRICS, B) == 0
    assert fo.parse_breaker_state(METRICS, "deployment-c") is None


def test_summarize_breaker(fo: ModuleType) -> None:
    # closed, closed, open, open, half-open, closed, open, closed
    states = [0, 0, 2, 2, 1, 0, 2, 0]
    samples = [(1000.0 + i, s) for i, s in enumerate(states)]
    got = fo.summarize_breaker(samples)
    assert got["polls"] == 8
    assert got["opens"] == 2
    assert got["half_open_seen"] is True
    assert got["polls_not_closed"] == 4


def test_summarize_breaker_never_opened(fo: ModuleType) -> None:
    got = fo.summarize_breaker([(1.0, 0), (2.0, 0)])
    assert got["opens"] == 0
    assert got["half_open_seen"] is False
    assert fo.summarize_breaker([])["polls"] == 0


# --- the gateway config -----------------------------------------------------------------------


def test_failover_config(fo: ModuleType, tmp_path: Path) -> None:
    cfg = fo.failover_config(
        gateway_port=18080,
        mock_a_port=18081,
        mock_b_port=18082,
        a_upstream_model="bench-model;fail=midstream;fail_after=2",
        decision_log=tmp_path / "decisions.jsonl",
        key="test-key-failover",
    )
    deployments = cfg["models"][0]["deployments"]
    assert [d["name"] for d in deployments] == [A, B]  # A first: every request starts on A
    assert [d["provider"] for d in deployments] == ["mock-a", "mock-b"]
    assert deployments[0]["upstream_model"] == "bench-model;fail=midstream;fail_after=2"
    assert deployments[1]["upstream_model"] == "bench-model"
    providers = {p["name"]: p["base_url"] for p in cfg["providers"]}
    assert providers == {
        "mock-a": "http://127.0.0.1:18081/v1",
        "mock-b": "http://127.0.0.1:18082/v1",
    }
    assert cfg["server"]["listen"] == "127.0.0.1:18080"
    assert cfg["reliability"]["max_attempts"] == 3
    assert cfg["decision_log"]["path"] == (tmp_path / "decisions.jsonl").as_posix()
    # The key never reaches the file; only its hash does.
    tenant = cfg["tenants"][0]
    assert tenant["api_key_sha256"] == fo.key_sha256("test-key-failover")
    assert "test-key-failover" not in yaml.safe_dump(cfg)


def test_the_template_loads_as_yaml_with_every_placeholder_filled(fo: ModuleType) -> None:
    text = fo.TEMPLATE.read_text(encoding="utf-8")
    for token in (
        "{{GATEWAY_PORT}}",
        "{{MOCK_A_PORT}}",
        "{{MOCK_B_PORT}}",
        "{{A_UPSTREAM_MODEL}}",
        "{{KEY_SHA256}}",
        "{{DECISION_LOG}}",
    ):
        assert token in text, token
    assert EM_DASH not in text


# --- the mock command lines -------------------------------------------------------------------


def test_mock_a_injects_the_briefed_mix(fo: ModuleType) -> None:
    args = fo.mock_args("prestream", "a")
    assert args[args.index("--fail-rate-http500") + 1] == "0.1"
    assert args[args.index("--fail-rate-http429") + 1] == "0.05"
    assert args[args.index("--fail-rate-reset") + 1] == "0.05"
    assert args[args.index("--seed") + 1] == "1"


def test_mock_b_and_the_midstream_arm_inject_no_rates(fo: ModuleType) -> None:
    for arm, side in (("prestream", "b"), ("midstream", "a"), ("midstream", "b")):
        assert not any(a.startswith("--fail-rate") for a in fo.mock_args(arm, side)), (arm, side)


def test_midstream_directive_commits_before_it_cuts(fo: ModuleType) -> None:
    # The gateway commits at the first chunk with content. The mock's first event is the empty
    # role chunk, so fail_after=1 would cut before any content and the gateway would fail over;
    # fail_after=2 lets the role chunk and the first content chunk through, then cuts.
    assert fo.A_UPSTREAM["midstream"] == "bench-model;fail=midstream;fail_after=2"
    assert fo.A_UPSTREAM["prestream"] == "bench-model"


# --- the checks -------------------------------------------------------------------------------


def good_prestream() -> dict[str, object]:
    classification = {
        "requests": 100,
        "attempts": 125,
        "started_on_a": 90,
        "started_on_b": 10,
        "failed_requests": 0,
        "attempts_by_deployment": {A: {"ok": 70, "http_500": 20}, B: {"ok": 35}},
    }
    return {
        "scheduled": 100,
        "load": {"scheduled": 100, "completed": 100, "errors": 0, "dropped": 0},
        "mocks": {"a": {"requests": 90}, "b": {"requests": 35}},
        "decision_log": {"records": 100, "dropped": 0},
        "classification": classification,
    }


def failed_names(checks: list[dict[str, object]]) -> list[str]:
    return [str(c["name"]) for c in checks if not c["passed"]]


def test_prestream_checks_pass_for_a_consistent_run(fo: ModuleType) -> None:
    checks = fo.prestream_checks(**good_prestream())
    assert checks and failed_names(checks) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d["load"].update(scheduled=99), "schedule"),
        (lambda d: d["decision_log"].update(records=99), "decision log"),
        (lambda d: d["decision_log"].update(dropped=1), "decision log"),
        (lambda d: d["mocks"]["b"].update(requests=36), "mock counts"),
        (lambda d: d["load"].update(errors=2), "client errors"),
        (lambda d: d["load"].update(dropped=1), "drops"),
    ],
)
def test_prestream_checks_catch_each_inconsistency(
    fo: ModuleType, mutate: object, expected: str
) -> None:
    data = good_prestream()
    mutate(data)  # type: ignore[operator]
    names = failed_names(fo.prestream_checks(**data))
    assert any(expected in n for n in names), names


def good_midstream() -> dict[str, object]:
    classification = {
        "requests": 100,
        "attempts": 100,
        "started_on_a": 20,
        "started_on_b": 80,
        "committed_then_failed": 20,
        "requests_with_attempts_on_both": 0,
        "attempts_per_request": {"1": 100},
        "failed_requests": 0,
    }
    return {
        "scheduled": 100,
        "load": {"scheduled": 100, "completed": 80, "errors": 20, "dropped": 0},
        "mocks": {"a": {"requests": 20}, "b": {"requests": 80}},
        "decision_log": {"records": 100, "dropped": 0},
        "classification": classification,
        "breaker": {"opens": 2},
    }


def test_midstream_checks_pass_for_a_consistent_run(fo: ModuleType) -> None:
    checks = fo.midstream_checks(**good_midstream())
    assert checks and failed_names(checks) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda d: d["classification"].update(attempts_per_request={"1": 99, "2": 1}),
            "one attempt",
        ),
        (lambda d: d["classification"].update(requests_with_attempts_on_both=1), "both"),
        (lambda d: d["mocks"]["b"].update(requests=81), "mock counts"),
        (lambda d: d["classification"].update(committed_then_failed=19), "committed"),
        (lambda d: d["load"].update(errors=19), "client errors"),
        (lambda d: d.update(breaker={"opens": 0}), "breaker"),
    ],
)
def test_midstream_checks_catch_each_violation(
    fo: ModuleType, mutate: object, expected: str
) -> None:
    data = good_midstream()
    mutate(data)  # type: ignore[operator]
    names = failed_names(fo.midstream_checks(**data))
    assert any(expected in n for n in names), names


def test_midstream_b_count_zero_needs_no_breaker_trip(fo: ModuleType) -> None:
    data = good_midstream()
    data["classification"].update(started_on_a=100, started_on_b=0, committed_then_failed=100)  # type: ignore[attr-defined]
    data["mocks"] = {"a": {"requests": 100}, "b": {"requests": 0}}
    data["load"].update(completed=0, errors=100)  # type: ignore[attr-defined]
    data["breaker"] = {"opens": 0}
    assert failed_names(fo.midstream_checks(**data)) == []


# --- the report -------------------------------------------------------------------------------


def results_for_report() -> dict[str, object]:
    pre = good_prestream()
    pre_class = dict(pre["classification"])  # type: ignore[arg-type]
    pre_class.update(
        eligible_failures_on_a=20,
        first_attempt_failed_eligibly=20,
        recovered=20,
        recovered_on_b=20,
        recovery_rate=1.0,
        not_recovered={},
        failed_by_reason={},
        responses_by_status={"200": 100},
        attempts_per_request={"1": 80, "2": 20},
    )
    mid = good_midstream()
    return {
        "experiment": "failover",
        "target_recovery": 0.99,
        "settings": {
            "prestream": {
                "rps": 100,
                "requests": 100,
                "mix": {"http500": 0.1, "http429": 0.05, "reset": 0.05},
            },
            "midstream": {"rps": 50, "requests": 100},
            "max_attempts": 3,
        },
        "arms": {
            "prestream": {
                **pre,
                "classification": pre_class,
                "breaker": {"opens": 1, "polls": 10, "half_open_seen": True, "polls_not_closed": 3},
                "checks": fo_checks_ok(),
                "wall_time_s": 120.0,
            },
            "midstream": {
                **mid,
                "breaker": {"opens": 2, "polls": 10, "half_open_seen": True, "polls_not_closed": 5},
                "checks": fo_checks_ok(),
                "wall_time_s": 50.0,
            },
        },
        "wall_time_s": 175.0,
    }


def fo_checks_ok() -> list[dict[str, object]]:
    return [{"name": "mock counts match the decision log", "passed": True, "detail": "ok"}]


def test_render_report(fo: ModuleType) -> None:
    machine = {"cpu": "Test CPU (8 threads)", "ram_gb": 16.0, "os": "TestOS"}
    text = fo.render_report(results_for_report(), machine, "20261029T000000Z-abc1234", "go1.27.2")
    assert EM_DASH not in text
    assert "Labels:" in text
    assert "nothing simulated" in text
    assert "target" in text.lower()
    assert "99%" in text
    assert "recovered" in text
    assert "per-request recovery" in text
    assert "routed straight to" in text
    assert "20261029T000000Z-abc1234" in text
    assert "TestOS" in text
