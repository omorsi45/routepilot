import importlib.util
import random
import sqlite3
import sys
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from routepilot_train.runinfo import repo_root

SCRIPT = repo_root() / "bench" / "scripts" / "quota_correctness.py"
EM_DASH = "\u2014"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("quota_correctness", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QC = load_script()


@pytest.fixture(scope="module")
def qc() -> ModuleType:
    return QC


# --- money ------------------------------------------------------------------------------------


def test_usd_to_pico_is_exact(qc: ModuleType) -> None:
    assert qc.usd_to_pico("0.05") == 50_000_000_000
    assert qc.usd_to_pico("0") == 0
    assert qc.usd_to_pico("0.049962") == 49_962_000_000
    assert qc.usd_to_pico("1.000000000001") == 1_000_000_000_001


def test_pico_to_usd_round_trips(qc: ModuleType) -> None:
    for pico in (0, 1, 50_000_000_000, 49_962_123_456):
        assert qc.usd_to_pico(qc.pico_to_usd(pico)) == pico
    assert qc.pico_to_usd(50_000_000_000) == "0.05"


# --- the clients ------------------------------------------------------------------------------


def test_max_tokens_are_seeded_and_in_range(qc: ModuleType) -> None:
    first = qc.max_tokens_stream(seed=1, client=3)
    again = qc.max_tokens_stream(seed=1, client=3)
    other_client = qc.max_tokens_stream(seed=1, client=4)
    other_seed = qc.max_tokens_stream(seed=2, client=3)
    a = [next(first) for _ in range(200)]
    assert a == [next(again) for _ in range(200)]
    assert a != [next(other_client) for _ in range(200)]
    assert a != [next(other_seed) for _ in range(200)]
    assert all(8 <= k <= 512 for k in a)
    assert min(a) < 60 and max(a) > 460  # it really spreads over the range


def reply(status: int, code: str = "", prompt: int = 0, completion: int = 0) -> object:
    return QC.Reply(status=status, code=code, prompt_tokens=prompt, completion_tokens=completion)


def test_client_stops_after_twenty_rejections_in_a_row(qc: ModuleType) -> None:
    seen: list[int] = []

    def send(max_tokens: int) -> object:
        seen.append(max_tokens)
        return reply(429, "insufficient_quota")

    tally = qc.run_client(send, iter(range(8, 10_000)), stop_after=20, max_requests=1000)
    assert len(seen) == 20
    assert tally.admitted == 0
    assert tally.rejected == 20
    assert tally.stopped == "rejected_in_a_row"


def test_an_admitted_request_resets_the_count(qc: ModuleType) -> None:
    # 19 rejections, one admission, then rejections for good: 19 + 1 + 20 requests.
    script = [reply(429, "insufficient_quota")] * 19 + [reply(200, "", 20, 8)]
    sent = iter(script)

    def send(max_tokens: int) -> object:
        return next(sent, None) or reply(429, "insufficient_quota")

    tally = qc.run_client(send, iter(range(8, 10_000)), stop_after=20, max_requests=1000)
    assert tally.admitted == 1
    assert tally.rejected == 19 + 20
    assert tally.prompt_tokens == 20 and tally.completion_tokens == 8
    assert tally.stopped == "rejected_in_a_row"


def test_other_responses_are_counted_and_do_not_count_as_rejections(qc: ModuleType) -> None:
    script = [reply(500), reply(429, "rate_limit_exceeded"), reply(200, "", 1, 1)]
    sent = iter(script)

    def send(max_tokens: int) -> object:
        return next(sent, None) or reply(429, "insufficient_quota")

    tally = qc.run_client(send, iter(range(8, 10_000)), stop_after=20, max_requests=1000)
    assert tally.other == {"500": 1, "429:rate_limit_exceeded": 1}
    assert tally.admitted == 1
    assert tally.rejected == 20


def test_client_has_a_request_cap(qc: ModuleType) -> None:
    tally = qc.run_client(
        lambda k: reply(200, "", 1, 1), iter(range(8, 10_000)), 20, max_requests=50
    )
    assert tally.admitted == 50
    assert tally.stopped == "max_requests"


def test_merge_tallies(qc: ModuleType) -> None:
    a = qc.Tally(admitted=2, rejected=3, prompt_tokens=10, completion_tokens=4, other={"500": 1})
    b = qc.Tally(admitted=1, rejected=1, prompt_tokens=5, completion_tokens=2, other={"500": 2})
    got = qc.merge_tallies([a, b])
    assert (got.admitted, got.rejected, got.prompt_tokens, got.completion_tokens) == (3, 4, 15, 6)
    assert got.other == {"500": 3}


# --- reading the results back -----------------------------------------------------------------


def test_parse_samples(qc: ModuleType) -> None:
    text = (
        'rp_ratelimit_rejections_total{kind="rpm"} 0\n'
        'rp_ratelimit_rejections_total{kind="tpm"} 3\n'
        'rp_requests_total{model="m",policy_role="pinned",status="200",tenant="t"} 7\n'
        "rp_budget_overshoot_picousd_total 0\n"
    )
    assert qc.parse_samples(text, "rp_ratelimit_rejections_total") == [
        ({"kind": "rpm"}, 0.0),
        ({"kind": "tpm"}, 3.0),
    ]
    assert qc.parse_samples(text, "rp_budget_overshoot_picousd_total") == [({}, 0.0)]
    assert qc.parse_samples(text, "rp_missing") == []


def make_ledger(path: Path, rows: list[tuple[str, int, str, int, int, int, int, str]]) -> None:
    """The usage table of the accounting store (internal/accounting/sqlite.go)."""
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE usage (request_id TEXT NOT NULL, attempt INTEGER NOT NULL, "
            "tenant TEXT NOT NULL, model TEXT NOT NULL, deployment TEXT NOT NULL, "
            "prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL, "
            "usage_estimated INTEGER NOT NULL, cost_estimate INTEGER NOT NULL, "
            "cost_actual INTEGER NOT NULL, policy_version TEXT NOT NULL, status TEXT NOT NULL, "
            "started_at TEXT NOT NULL, ended_at TEXT NOT NULL, PRIMARY KEY (request_id, attempt))"
        )
    make_ledger_rows(path, rows)


def make_ledger_rows(path: Path, rows: list[tuple[str, int, str, int, int, int, int, str]]) -> None:
    """Add rows to an existing usage table."""
    with sqlite3.connect(path) as db:
        for request_id, attempt, tenant, prompt, completion, estimate, actual, status in rows:
            db.execute(
                "INSERT INTO usage VALUES (?, ?, ?, 'm', 'd', ?, ?, 0, ?, ?, 'pinned', ?, "
                "'2026-10-28T00:00:00.000000000Z', '2026-10-28T00:00:01.000000000Z')",
                (request_id, attempt, tenant, prompt, completion, estimate, actual, status),
            )


def test_ledger_summary(qc: ModuleType, tmp_path: Path) -> None:
    db = tmp_path / "usage.db"
    make_ledger(
        db,
        [
            ("r1", 1, "t", 20, 8, 900, 40, "ok"),
            ("r2", 1, "t", 20, 8, 500, 40, "ok"),
            ("r3", 1, "other", 20, 8, 500, 40, "ok"),  # another tenant is not counted
            ("r4", 1, "t", 0, 0, 100, 0, "http_500"),
        ],
    )
    got = qc.ledger_summary(db, "t")
    assert got == {
        "rows": 3,
        "ok_rows": 2,
        "sum_pico": 80,
        "estimate_min": 100,
        "estimate_max": 900,
        "prompt_tokens": 40,
        "completion_tokens": 16,
    }


def test_ledger_summary_of_an_empty_ledger(qc: ModuleType, tmp_path: Path) -> None:
    db = tmp_path / "usage.db"
    make_ledger(db, [])
    got = qc.ledger_summary(db, "t")
    assert got["rows"] == 0 and got["sum_pico"] == 0 and got["estimate_min"] is None


GATEWAY_LOG = "\n".join(
    [
        '{"time":"t","level":"INFO","msg":"listening","addr":"127.0.0.1:1"}',
        '{"time":"t","level":"WARN","msg":"usage ledger write failed","request_id":"a",'
        '"rows_lost":1,"error":"accounting: record attempt a/1: context deadline exceeded"}',
        '{"time":"t","level":"WARN","msg":"usage ledger write failed","request_id":"b",'
        '"rows_lost":2,"error":"accounting: record attempt b/1: context deadline exceeded"}',
        '{"time":"t","level":"WARN","msg":"usage ledger write failed","request_id":"c",'
        '"rows_lost":1,"error":"accounting: record attempt c/1: database is locked"}',
        "not json at all",
        "",
    ]
)


def test_parse_ledger_failures(qc: ModuleType) -> None:
    assert qc.parse_ledger_failures(GATEWAY_LOG) == {
        "log_lines": 3,
        "rows_lost": 4,
        "errors": {"context deadline exceeded": 2, "database is locked": 1},
    }
    assert qc.parse_ledger_failures("") == {"log_lines": 0, "rows_lost": 0, "errors": {}}


def test_parse_ledger_failures_counts_a_batch_and_a_full_queue(qc: ModuleType) -> None:
    # The gateway's batcher warns once per failed batch (no request id) and, at most once a
    # second, when its queue was full and rows were dropped.
    log = "\n".join(
        [
            '{"level":"WARN","msg":"usage ledger write failed","rows_lost":512,'
            '"error":"accounting: record attempt r/1: database is locked"}',
            '{"level":"WARN","msg":"usage ledger queue full, rows dropped","rows_dropped":40,'
            '"queue_rows":65536}',
            '{"level":"WARN","msg":"usage ledger queue full, rows dropped","rows_dropped":2,'
            '"queue_rows":65536}',
        ]
    )
    assert qc.parse_ledger_failures(log) == {
        "log_lines": 3,
        "rows_lost": 554,
        "errors": {"database is locked": 1, "queue full": 2},
    }


def test_wait_for_ledger_rows(qc: ModuleType, tmp_path: Path) -> None:
    # The gateway writes a request's rows in the background, so the last rows reach the ledger a
    # moment after the last response. The wait returns the count as soon as it is complete, and
    # when it never is, the count it reached: a missing row stays visible.
    db = tmp_path / "usage.db"
    make_ledger(
        db, [("r1", 1, "t", 20, 8, 900, 40, "ok"), ("r2", 1, "other", 20, 8, 900, 40, "ok")]
    )
    assert qc.wait_for_ledger_rows(db, "t", 1, timeout_s=5.0, poll_s=0.01) == 1

    start = time.monotonic()
    assert qc.wait_for_ledger_rows(db, "t", 2, timeout_s=0.3, poll_s=0.01) == 1
    assert 0.25 <= time.monotonic() - start < 3.0

    late = threading.Timer(
        0.2,
        lambda: make_ledger_rows(db, [("r3", 1, "t", 20, 8, 900, 40, "ok")]),
    )
    late.start()
    try:
        assert qc.wait_for_ledger_rows(db, "t", 2, timeout_s=5.0, poll_s=0.01) == 2
    finally:
        late.join()


def test_the_mock_answers_with_the_length_each_request_allows(qc: ModuleType) -> None:
    # The mock caps its answer at max_tokens, so with a default length above every
    # max_tokens and no tokens= directive each answer has exactly max_tokens completion tokens
    # and what a request costs is what was reserved for it.
    assert qc.MOCK_TOKENS > qc.MAX_TOKENS_RANGE[1]
    assert qc.MOCK_ARGS[qc.MOCK_ARGS.index("--tokens") + 1] == str(qc.MOCK_TOKENS)
    assert qc.UPSTREAM_MODEL == "bench-model"
    template = yaml.safe_load(
        qc.TEMPLATE.read_text(encoding="utf-8").replace("{{", "").replace("}}", "")
    )
    assert template["models"][0]["deployments"][0]["upstream_model"] == qc.UPSTREAM_MODEL


def test_parse_reconcile_output(qc: ModuleType) -> None:
    assert qc.parse_reconcile_output("quota-redis 0.049962 0.049962 0\n") == [
        {
            "tenant": "quota-redis",
            "ledger_usd": "0.049962",
            "counter_usd": "0.049962",
            "diff_usd": "0",
        }
    ]
    assert qc.parse_reconcile_output("") == []
    with pytest.raises(ValueError, match="reconcile"):
        qc.parse_reconcile_output("only three fields\n")


# --- the gateway config -----------------------------------------------------------------------


def test_quota_config_memory(qc: ModuleType, tmp_path: Path) -> None:
    cfg = qc.quota_config(
        gateway_port=18080,
        mock_port=18081,
        key="test-key-quota",
        tenant="quota-memory",
        backend="memory",
        sqlite_path=tmp_path / "usage.db",
    )
    tenant = cfg["tenants"][0]
    assert tenant["id"] == "quota-memory"
    assert tenant["budget_usd"] == "0.05" and tenant["budget_period"] == "total"
    assert tenant["tpm"] > 0 and tenant["max_concurrency"] > 64
    assert tenant["api_key_sha256"] == qc.key_sha256("test-key-quota")
    assert "test-key-quota" not in yaml.safe_dump(cfg)
    assert cfg["quota"] == {"backend": "memory"}
    assert cfg["accounting"] == {
        "driver": "sqlite",
        "sqlite_path": (tmp_path / "usage.db").as_posix(),
    }
    assert cfg["models"][0]["deployments"][0]["upstream_model"] == "bench-model"
    assert cfg["providers"][0]["base_url"] == "http://127.0.0.1:18081/v1"
    assert cfg["server"]["listen"] == "127.0.0.1:18080"


def test_quota_config_redis_names_the_environment_variable(qc: ModuleType, tmp_path: Path) -> None:
    cfg = qc.quota_config(18080, 18081, "test-key-quota", "quota-redis", "redis", tmp_path / "u.db")
    assert cfg["quota"] == {"backend": "redis", "redis_url_env": "ROUTEPILOT_TEST_REDIS_URL"}


def test_quota_config_rejects_an_unknown_backend(qc: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend"):
        qc.quota_config(1, 2, "test-key-quota", "t", "memcached", tmp_path / "u.db")


def test_the_budget_is_far_above_one_request_and_priced_as_briefed(qc: ModuleType) -> None:
    # One request's worst case at max_tokens 512 must fit the 0.05 USD budget many times over, and
    # a request at max_tokens 8 must cost far less than one at 512.
    limit = qc.usd_to_pico(qc.BUDGET_USD)
    worst = qc.PROMPT_BOUND_TOKENS * qc.PRICE_IN_PICO + 512 * qc.PRICE_OUT_PICO
    best = qc.PROMPT_BOUND_TOKENS * qc.PRICE_IN_PICO + 8 * qc.PRICE_OUT_PICO
    assert limit // worst >= 40
    assert worst > 10 * best / 2


# --- the checks -------------------------------------------------------------------------------


def good_backend() -> dict[str, object]:
    return {
        "backend": "redis",
        "limit_pico": 50_000_000_000,
        "clients": {
            "admitted": 100,
            "rejected": 500,
            "other": {},
            "prompt_tokens": 2000,
            "completion_tokens": 800,
        },
        "ledger": {"rows": 100, "ok_rows": 100, "sum_pico": 49_000_000_000},
        "metrics": {
            "overshoot_pico": 0,
            "ratelimit": {"rpm": 0, "tpm": 0, "concurrency": 0, "upstream": 0},
        },
        "mock_requests": 100,
        "reconcile": {
            "exit_code": 0,
            "lines": [
                {
                    "tenant": "quota-redis",
                    "ledger_usd": "0.049",
                    "counter_usd": "0.049",
                    "diff_usd": "0",
                }
            ],
        },
        "expected_cost_pico": 49_000_000_000,
    }


def failed_names(checks: list[dict[str, object]]) -> list[str]:
    return [str(c["name"]) for c in checks if not c["passed"]]


def test_quota_checks_pass_for_a_consistent_run(qc: ModuleType) -> None:
    checks = qc.quota_checks(**good_backend())
    assert checks and failed_names(checks) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d["ledger"].update(sum_pico=50_000_000_001), "ledger"),
        (lambda d: d.update(expected_cost_pico=50_000_000_001), "spend"),
        (lambda d: d["metrics"].update(overshoot_pico=1), "overshoot"),
        (lambda d: d["reconcile"].update(exit_code=1), "reconcile"),
        (lambda d: d["reconcile"]["lines"][0].update(diff_usd="0.000001"), "reconcile"),
        (lambda d: d["reconcile"]["lines"][0].update(counter_usd="0.048"), "counter"),
        (lambda d: d["clients"].update(other={"500": 1}), "unexpected"),
        (lambda d: d["ledger"].update(rows=99, ok_rows=99), "rows"),
        (lambda d: d.update(mock_requests=101), "mock"),
        (lambda d: d["metrics"]["ratelimit"].update(tpm=2), "rate limit"),
        (lambda d: d.update(expected_cost_pico=48_000_000_000), "cost"),
    ],
)
def test_quota_checks_catch_each_failure(qc: ModuleType, mutate: object, expected: str) -> None:
    data = good_backend()
    mutate(data)  # type: ignore[operator]
    names = failed_names(qc.quota_checks(**data))
    assert any(expected in n for n in names), names


def test_memory_backend_expects_reconcile_to_refuse(qc: ModuleType) -> None:
    data = good_backend()
    data["backend"] = "memory"
    data["reconcile"] = {"exit_code": 2, "lines": []}
    assert failed_names(qc.quota_checks(**data)) == []
    # Exit 0 for the memory backend would mean reconcile read a counter that lives in the gateway.
    data["reconcile"] = {"exit_code": 0, "lines": []}
    assert any("reconcile" in n for n in failed_names(qc.quota_checks(**data)))


# --- the Redis container ----------------------------------------------------------------------


class FakeDocker:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        return "container-id\n"


def test_redis_container_starts_on_loopback_and_stops_on_error(qc: ModuleType) -> None:
    docker = FakeDocker()
    with pytest.raises(RuntimeError, match="boom"):
        with qc.redis_container(docker, port=16379, wait=lambda port: None):
            raise RuntimeError("boom")
    run, stop = docker.commands
    assert run[:3] == ["docker", "run", "--rm"]
    assert "-d" in run
    assert run[run.index("--name") + 1] == "rp-quota-redis"
    assert run[run.index("-p") + 1] == "127.0.0.1:16379:6379"  # loopback only
    assert run[-1] == "redis:7.4-alpine"
    assert stop == ["docker", "stop", "rp-quota-redis"]


def test_redis_container_yields_a_loopback_url(qc: ModuleType) -> None:
    with qc.redis_container(FakeDocker(), port=16379, wait=lambda port: None) as url:
        assert url == "redis://127.0.0.1:16379/0"


def test_redis_container_is_stopped_when_it_never_answers(qc: ModuleType) -> None:
    docker = FakeDocker()

    def never_ready(port: int) -> None:
        raise TimeoutError("no PING")

    with pytest.raises(TimeoutError):
        with qc.redis_container(docker, port=16379, wait=never_ready):
            pytest.fail("the block must not run when Redis never answers")
    assert docker.commands[-1] == ["docker", "stop", "rp-quota-redis"]


def test_a_failed_stop_does_not_hide_the_error(
    qc: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    def docker(command: list[str]) -> str:
        if command[1] == "stop":
            raise RuntimeError("stop failed")
        return ""

    with pytest.raises(ValueError, match="the real error"):
        with qc.redis_container(docker, port=16379, wait=lambda port: None):
            raise ValueError("the real error")
    assert "could not stop rp-quota-redis" in capsys.readouterr().err


def test_miniredis_fallback_runs_the_gateway_test_and_says_so(qc: ModuleType) -> None:
    seen: list[list[str]] = []

    def run(command: list[str]) -> tuple[int, str]:
        seen.append(command)
        return 0, "=== RUN\n--- PASS: TestBudgetNeverOvershootsUnderConcurrency/miniredis\nok\n"

    got = qc.miniredis_go_test(run)
    assert got["redis_mode"] == "miniredis-go-test"
    assert got["checks"][0]["passed"]
    assert "TestBudgetNeverOvershootsUnderConcurrency/miniredis" in seen[0]
    assert qc.miniredis_go_test(lambda command: (1, "FAIL"))["checks"][0]["passed"] is False


# --- the report -------------------------------------------------------------------------------


def results_for_report(qc: ModuleType) -> dict[str, object]:
    redis = good_backend()
    redis.update(
        wall_time_s=12.5,
        redis_mode="docker",
        unused_pico=1_000_000_000,
        checks=[{"name": "ledger: sum within the limit", "passed": True, "detail": "ok"}],
    )
    redis["reconcile"].update(stdout="quota-redis 0.049 0.049 0\n", stderr="")  # type: ignore[attr-defined]
    redis["clients"].update(requests=600)  # type: ignore[attr-defined]
    redis["ledger"].update(estimate_min=100, estimate_max=900)  # type: ignore[attr-defined]
    memory = {
        **redis,
        "backend": "memory",
        "redis_mode": None,
        "reconcile": {"exit_code": 2, "lines": [], "stdout": "", "stderr": "memory"},
    }
    return {
        "experiment": "quota_correctness",
        "settings": {
            "clients": 64,
            "seed": 1,
            "budget_usd": "0.05",
            "limit_pico": 50_000_000_000,
            "upstream_model": "bench-model",
            "mock_tokens": 600,
            "price_in_per_mtok": "1",
            "price_out_per_mtok": "2",
            "max_tokens_range": [8, 512],
            "stop_after": 20,
        },
        "backends": {"memory": memory, "redis": redis},
        "all_checks_passed": True,
        "wall_time_s": 30.0,
    }


def test_render_report(qc: ModuleType) -> None:
    machine = {"cpu": "Test CPU (8 threads)", "ram_gb": 16.0, "os": "TestOS"}
    text = qc.render_report(results_for_report(qc), machine, "20261029T000000Z-abc1234", "go1.27.2")
    assert EM_DASH not in text
    assert "Labels:" in text and "nothing simulated" in text
    assert "memory" in text and "redis" in text
    assert "unused budget" in text.lower()
    assert "reconcile" in text
    # The mock caps its answer at each request's max_tokens: no directive, and a
    # default length above every max_tokens, so what is spent is what was reserved.
    assert "tokens=8" not in text
    assert "equal" in text and "max_tokens" in text and "--tokens 600" in text
    assert "20261029T000000Z-abc1234" in text


def test_render_report_names_the_client_count(qc: ModuleType) -> None:
    results = results_for_report(qc)
    results["settings"]["clients"] = 8  # type: ignore[index]
    machine = {"cpu": "Test CPU (8 threads)", "ram_gb": 16.0, "os": "TestOS"}
    text = qc.render_report(results, machine, "20261029T000000Z-abc1234", "go1.27.2")
    assert "8 clients" in text and "64 clients" not in text
    assert "control run" in text.lower()
    full = qc.render_report(results_for_report(qc), machine, "20261029T000000Z-abc1234", "go1.27.2")
    assert "control run" not in full.lower()


def test_render_report_explains_missing_ledger_rows(qc: ModuleType) -> None:
    results = results_for_report(qc)
    redis = results["backends"]["redis"]  # type: ignore[index]
    redis["ledger"].update(rows=90, ok_rows=90, sum_pico=44_000_000_000)
    redis["metrics"]["accounting_write_errors"] = 10
    redis["ledger_write_failures"] = {
        "log_lines": 10,
        "rows_lost": 10,
        "errors": {"context deadline exceeded": 10},
    }
    redis["checks"] = [{"name": "rows: complete", "passed": False, "detail": "90 of 100"}]
    machine = {"cpu": "Test CPU (8 threads)", "ram_gb": 16.0, "os": "TestOS"}
    text = qc.render_report(results, machine, "20261029T000000Z-abc1234", "go1.27.2")
    assert "FAIL" in text
    assert "context deadline exceeded" in text
    assert "10 rows" in text
    assert EM_DASH not in text


def test_random_streams_do_not_share_state(qc: ModuleType) -> None:
    # The module must not seed the global generator: another consumer of random is unaffected.
    random.seed(7)
    expected = random.random()
    random.seed(7)
    stream = qc.max_tokens_stream(seed=1, client=0)
    next(stream)
    assert random.random() == expected
