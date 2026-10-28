import importlib.util
import json
import math
import sys
from types import ModuleType

import pytest

from routepilot_train.runinfo import repo_root

SCRIPT = repo_root() / "bench" / "scripts" / "overhead.py"
EM_DASH = "\u2014"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("overhead", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ov() -> ModuleType:
    return load_script()


# A scrape of the series the script reads, in the Prometheus text format the gateway serves:
# 10 observations at or below 1 ms, 90 more at or below 2 ms, none above.
SCRAPE = """\
# HELP rp_gateway_overhead_seconds Request duration minus the time spent inside the providers.
# TYPE rp_gateway_overhead_seconds histogram
rp_gateway_overhead_seconds_bucket{le="0.0005"} 0
rp_gateway_overhead_seconds_bucket{le="0.001"} 10
rp_gateway_overhead_seconds_bucket{le="0.002"} 100
rp_gateway_overhead_seconds_bucket{le="+Inf"} 100
rp_gateway_overhead_seconds_sum 0.15
rp_gateway_overhead_seconds_count 100
rp_decision_duration_seconds_bucket{le="+Inf"} 7
rp_decision_duration_seconds_count 7
rp_decision_duration_seconds_sum 0.007
rp_requests_total{model="bench-model",policy_role="none",status="200",tenant="bench"} 100
"""


def test_parse_histogram_reads_one_series(ov: ModuleType) -> None:
    h = ov.parse_histogram(SCRAPE, "rp_gateway_overhead_seconds")
    assert h.buckets == ((0.0005, 0.0), (0.001, 10.0), (0.002, 100.0), (math.inf, 100.0))
    assert h.count == 100.0
    assert h.total == pytest.approx(0.15)


def test_parse_histogram_rejects_a_missing_series(ov: ModuleType) -> None:
    with pytest.raises(ValueError, match="rp_nothing_seconds"):
        ov.parse_histogram(SCRAPE, "rp_nothing_seconds")


def test_subtract_gives_the_window_between_two_scrapes(ov: ModuleType) -> None:
    before = ov.Histogram(((0.001, 5.0), (0.002, 8.0), (math.inf, 8.0)), 8.0, 0.01)
    after = ov.Histogram(((0.001, 15.0), (0.002, 28.0), (math.inf, 29.0)), 29.0, 0.05)
    delta = ov.subtract(after, before)
    assert delta.buckets == ((0.001, 10.0), (0.002, 20.0), (math.inf, 21.0))
    assert delta.count == 21.0
    assert delta.total == pytest.approx(0.04)


def test_subtract_rejects_different_buckets_and_counter_resets(ov: ModuleType) -> None:
    a = ov.Histogram(((0.001, 1.0), (math.inf, 1.0)), 1.0, 0.0)
    b = ov.Histogram(((0.002, 1.0), (math.inf, 1.0)), 1.0, 0.0)
    with pytest.raises(ValueError, match="bounds"):
        ov.subtract(a, b)
    big = ov.Histogram(((0.001, 9.0), (math.inf, 9.0)), 9.0, 0.0)
    with pytest.raises(ValueError, match="decreased"):
        ov.subtract(a, big)


def test_bucket_quantile_interpolates_like_prometheus(ov: ModuleType) -> None:
    h = ov.Histogram(((1.0, 0.0), (2.0, 10.0), (4.0, 20.0), (math.inf, 20.0)), 20.0, 0.0)
    # The median is the 10th of 20 observations, the last one of the (1, 2] bucket.
    q50 = ov.bucket_quantile(h, 0.5)
    assert (q50.value, q50.lower, q50.upper) == (2.0, 1.0, 2.0)
    # The 15th observation is halfway through the (2, 4] bucket.
    q75 = ov.bucket_quantile(h, 0.75)
    assert (q75.value, q75.lower, q75.upper) == (3.0, 2.0, 4.0)


def test_bucket_quantile_starts_the_first_bucket_at_zero(ov: ModuleType) -> None:
    h = ov.Histogram(((0.01, 4.0), (0.02, 4.0), (math.inf, 4.0)), 4.0, 0.0)
    q = ov.bucket_quantile(h, 0.5)
    assert (q.lower, q.upper) == (0.0, 0.01)
    assert q.value == pytest.approx(0.005)


def test_bucket_quantile_above_the_last_bound_reports_that_bound(ov: ModuleType) -> None:
    h = ov.Histogram(((0.5, 1.0), (1.0, 2.0), (math.inf, 10.0)), 10.0, 0.0)
    q = ov.bucket_quantile(h, 0.99)
    assert q.value == 1.0
    assert q.lower == 1.0
    assert math.isinf(q.upper)


def test_bucket_quantile_needs_observations(ov: ModuleType) -> None:
    h = ov.Histogram(((0.5, 0.0), (math.inf, 0.0)), 0.0, 0.0)
    with pytest.raises(ValueError, match="no observations"):
        ov.bucket_quantile(h, 0.5)


def test_difference_of_percentiles_is_gateway_minus_direct(ov: ModuleType) -> None:
    gateway = {"p50": 38.0, "p90": 40.0, "p95": 41.5, "p99": 45.0, "p99.9": 60.0}
    direct = {"p50": 36.5, "p90": 37.0, "p95": 37.5, "p99": 38.0, "p99.9": 41.0}
    diff = ov.difference_of_percentiles(gateway, direct)
    assert diff == {"p50": 1.5, "p90": 3.0, "p95": 4.0, "p99": 7.0, "p99.9": 19.0}


def test_client_verdict_compares_the_p95_difference_with_the_target(ov: ModuleType) -> None:
    assert ov.TARGET_P95_MS == 15.0
    assert ov.client_verdict(14.99) == "met"
    assert ov.client_verdict(15.0) == "not met"
    assert ov.client_verdict(40.0) == "not met"


def test_server_verdict_is_indeterminate_when_the_bucket_spans_the_target(ov: ModuleType) -> None:
    q = ov.Quantile
    assert ov.server_verdict(q(0.003, 0.00215, 0.00464)) == "met"
    assert ov.server_verdict(q(0.03, 0.0215, 0.0464)) == "not met"
    assert ov.server_verdict(q(0.012, 0.0100, 0.0215)) == "undecided (bucket spans 15 ms)"


def test_matrix_orders_runs_and_adds_the_learned_run_last(ov: ModuleType) -> None:
    specs = ov.matrix((100, 500), learned_rps=500)
    names = [s.name for s in specs]
    assert names == [
        "direct-100rps-nonstream",
        "gateway-100rps-nonstream",
        "direct-100rps-stream",
        "gateway-100rps-stream",
        "direct-500rps-nonstream",
        "gateway-500rps-nonstream",
        "direct-500rps-stream",
        "gateway-500rps-stream",
        "learned-500rps-nonstream",
    ]
    assert specs[-1].model == "auto"
    assert {s.model for s in specs[:-1]} == {"bench-model"}
    assert [s.name for s in ov.matrix((100,), learned_rps=None)][-1] == "gateway-100rps-stream"


def test_gateway_config_fills_ports_and_hashes_the_key_at_run_time(ov: ModuleType) -> None:
    cfg = ov.gateway_config(gateway_port=18080, mock_port=19090, key="some-fake-key")
    assert cfg["server"]["listen"] == "127.0.0.1:18080"
    assert cfg["server"]["max_inflight"] == 4096
    assert cfg["providers"][0]["base_url"] == "http://127.0.0.1:19090/v1"
    tenant = cfg["tenants"][0]
    assert len(tenant["api_key_sha256"]) == 64
    assert "some-fake-key" not in str(cfg)
    assert [m["id"] for m in cfg["models"]] == ["bench-model"]
    # No limit that would add work per request.
    assert not {"rpm", "tpm", "max_concurrency", "budget_usd"} & set(tenant)
    assert cfg["accounting"] == {"driver": "none"}
    assert "decision_log" not in cfg and "tracing" not in cfg


def test_learned_config_uses_the_flagship_catalog_on_the_mock(ov: ModuleType) -> None:
    pool = {
        "models": [
            {
                "id": "m-a",
                "rank": 1,
                "context_window": 1048576,
                "max_output_tokens": 65536,
                "capabilities": ["streaming"],
                "input_price_per_mtok": "1.000000",
                "output_price_per_mtok": "2.000000",
                "deployments": [{"name": "m-a-replay", "provider": "replay"}],
            },
            {"id": "m-b", "rank": 2, "deployments": [{"name": "m-b-replay"}]},
        ]
    }
    cfg = ov.learned_config(
        gateway_port=18081,
        mock_port=19091,
        key="some-fake-key",
        pool=pool,
        artifact_version="abc123",
    )
    assert [m["id"] for m in cfg["models"]] == ["m-a", "m-b"]
    for m in cfg["models"]:
        (dep,) = m["deployments"]
        assert dep["provider"] == "mock"
        assert dep["upstream_model"] == m["id"]
        assert ";" not in dep["upstream_model"]  # no mock directives
    assert cfg["models"][0]["input_price_per_mtok"] == "1.000000"
    assert cfg["routing"] == {"policy": "learned"}
    stable = cfg["policies"]["stable"]
    assert (stable["name"], stable["version"]) == ("flagship", "abc123")
    assert stable["config"].endswith("flagship.yaml")
    assert cfg["server"]["latency_profiles"].endswith("profiles.yaml")


def _run(ov: ModuleType, name_path: str, rps: int, stream: bool, p95: float, srv_p95: float | None):
    """One synthetic run record in the shape measure() returns."""
    pct = {
        "count": 100,
        "min": p95 - 1,
        "mean": p95 - 0.5,
        "p50": p95 - 0.7,
        "p90": p95 - 0.2,
        "p95": p95,
        "p99": p95 + 0.5,
        "p99.9": p95 + 1.0,
        "max": p95 + 2.0,
    }
    load = {
        "scheduled": 100,
        "completed": 100,
        "errors": 0,
        "dropped": 0,
        "errors_by_kind": {},
        "throughput_rps": float(rps),
        "offered_rps": float(rps),
        "latency_ms": pct,
        "ttft_ms": dict(pct, p95=p95 / 2) if stream else None,
        "launch_lag_ms": dict(pct),
    }
    server = None
    if srv_p95 is not None:
        q = {"value_ms": srv_p95, "lower_ms": srv_p95 - 1, "upper_ms": srv_p95 + 1}
        server = {
            "overhead": {"count": 100, "p50": q, "p95": q, "p99": q},
            "decision": None,
        }
    return {
        "name": ov.Spec(name_path, rps, stream).name,
        "path": name_path,
        "rps": rps,
        "stream": stream,
        "load": load,
        "server": server,
    }


def test_render_report_states_labels_target_and_verdicts(ov: ModuleType) -> None:
    runs = []
    for stream in (False, True):
        runs.append(_run(ov, "direct", 100, stream, 37.0, None))
        runs.append(_run(ov, "gateway", 100, stream, 39.0, 1.5))
    runs.append(_run(ov, "learned", 100, False, 40.0, 2.5))
    results = {
        "experiment": "overhead",
        "target_p95_ms": 15.0,
        "settings": {"warmup_s": 10, "duration_s": 60, "levels": [100], "learned_rps": 100},
        "runs": runs,
    }
    results["comparisons"] = ov.compare(results["runs"])
    machine = {"cpu": "Test CPU (8 threads)", "ram_gb": 16.0, "os": "TestOS"}

    text = ov.render_report(results, machine, "20261029T000000Z-abc1234", "go1.27.2")

    assert EM_DASH not in text
    assert "difference of percentiles, not a percentile of differences" in text
    assert "from histogram buckets" in text
    assert "share one machine" in text or "share this one machine" in text
    assert "Test CPU (8 threads)" in text
    assert "< 15 ms" in text
    assert "met" in text
    assert "learned" in text.lower()
    # 39.0 - 37.0 at p95 is 2.0 ms: the table carries it.
    assert "2.00" in text


# --- the clock-resolution guard ----------------------------------------------------------------

FINE_CLOCK = {"clock_step_ns": 40, "zero_fraction": 0.0}
COARSE_CLOCK = {"clock_step_ns": 503_000, "zero_fraction": 0.9999}


def test_parse_probe_reads_the_json_rp_load_prints(ov: ModuleType) -> None:
    probe = ov.parse_probe('{"clock_step_ns": 40, "zero_fraction": 0.0123}\n')
    assert probe == {"clock_step_ns": 40, "zero_fraction": 0.0123}
    assert isinstance(probe["clock_step_ns"], int)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json",
        "[1, 2]",
        '{"clock_step_ns": 40}',
        '{"zero_fraction": 0.5}',
        '{"clock_step_ns": 0, "zero_fraction": 0.5}',
        '{"clock_step_ns": 40, "zero_fraction": 1.5}',
        '{"clock_step_ns": 40, "zero_fraction": -0.1}',
    ],
)
def test_parse_probe_rejects_anything_but_a_valid_probe(ov: ModuleType, text: str) -> None:
    with pytest.raises(ValueError, match="clock probe"):
        ov.parse_probe(text)


def test_check_clock_accepts_a_fine_clock(ov: ModuleType) -> None:
    assert ov.check_clock(FINE_CLOCK, allow_coarse=False) is False
    assert ov.check_clock(FINE_CLOCK, allow_coarse=True) is False


def test_check_clock_threshold_is_one_microsecond(ov: ModuleType) -> None:
    assert ov.COARSE_CLOCK_NS == 1000
    assert ov.check_clock({"clock_step_ns": 1000, "zero_fraction": 0.5}, False) is False
    with pytest.raises(ov.CoarseClockError):
        ov.check_clock({"clock_step_ns": 1001, "zero_fraction": 0.5}, False)


def test_check_clock_refuses_a_coarse_clock_and_names_the_way_out(ov: ModuleType) -> None:
    with pytest.raises(ov.CoarseClockError) as raised:
        ov.check_clock(COARSE_CLOCK, allow_coarse=False)
    message = str(raised.value)
    assert "503000 ns" in message
    assert "--allow-coarse-clock" in message


def test_check_clock_lets_a_coarse_clock_through_when_asked(ov: ModuleType) -> None:
    assert ov.check_clock(COARSE_CLOCK, allow_coarse=True) is True


def _synthetic_results(ov: ModuleType, clock: dict | None) -> dict:
    runs = []
    for stream in (False, True):
        runs.append(_run(ov, "direct", 100, stream, 37.0, None))
        runs.append(_run(ov, "gateway", 100, stream, 39.0, 1.5))
    runs.append(_run(ov, "learned", 100, False, 40.0, 2.5))
    results: dict = {
        "experiment": "overhead",
        "target_p95_ms": 15.0,
        "settings": {"warmup_s": 10, "duration_s": 60, "levels": [100], "learned_rps": 100},
        "runs": runs,
    }
    if clock is not None:
        results["clock"] = clock
    results["comparisons"] = ov.compare(results["runs"])
    return results


MACHINE = {"cpu": "Test CPU (8 threads)", "ram_gb": 16.0, "os": "Linux-6.1-microsoft-standard-WSL2"}


def test_report_labels_line_states_the_measured_clock_step(ov: ModuleType) -> None:
    clock = {**FINE_CLOCK, "coarse": False}
    text = ov.render_report(_synthetic_results(ov, clock), MACHINE, "run-1", "go1.27.2")
    labels = next(line for line in text.splitlines() if line.startswith("Labels:"))
    assert "WSL2" in labels
    assert "Test CPU (8 threads)" in labels
    assert "share one machine" in labels
    assert "clock step" in labels
    assert "40 ns" in labels
    assert EM_DASH not in text
    # A fine clock leaves every section in place.
    for heading in ("## Overhead, client side", "## Overhead, server side", "## Target summary"):
        assert heading in text
    assert "omitted" not in text.lower()


def test_report_without_clock_data_still_renders(ov: ModuleType) -> None:
    # Results written before the guard existed have no clock key.
    text = ov.render_report(_synthetic_results(ov, None), MACHINE, "run-1", "go1.27.2")
    assert "## Overhead, client side" in text
    assert "clock step" not in text


def test_coarse_clock_report_omits_every_sub_millisecond_quantity_and_says_why(
    ov: ModuleType,
) -> None:
    clock = {**COARSE_CLOCK, "coarse": True}
    text = ov.render_report(_synthetic_results(ov, clock), MACHINE, "run-1", "go1.27.2")
    labels = next(line for line in text.splitlines() if line.startswith("Labels:"))
    assert "503000 ns" in labels
    for heading in (
        "## Overhead, client side",
        "## Overhead, server side",
        "## Target summary",
        "## Learned policy run",
        "Generator launch lag",
    ):
        assert heading not in text
    assert "rp_gateway_overhead_seconds" not in text
    assert "rp_decision_duration_seconds" not in text
    omitted = next(line for line in text.splitlines() if line.startswith("Omitted:"))
    assert "503000 ns" in omitted
    assert "server-side" in omitted and "client-side" in omitted
    assert "1000 ns" in omitted
    # The absolute client latencies (tens of milliseconds) stay.
    assert "## Runs (client side, ms)" in text
    assert EM_DASH not in text


class _Pipeline:
    """Replaces everything in main() that needs Go, a mock or a gateway."""

    def __init__(
        self, ov: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path, probe: dict
    ) -> None:
        self.ov = ov
        self.measured: list[str] = []
        monkeypatch.setattr(ov, "ROOT", tmp_path)
        monkeypatch.setattr(ov, "COOLDOWN_S", 0.0)
        monkeypatch.setattr(ov, "probe_clock", lambda binary: dict(probe))
        monkeypatch.setattr(
            ov,
            "build_binaries",
            lambda bin_dir: {n: bin_dir / n for n in ("rp-mock", "routepilot", "rp-load")},
        )
        monkeypatch.setattr(ov, "measure", self._measure)

    def _measure(self, spec, bins, scratch, warmup_s, duration_s, learned):
        self.measured.append(spec.name)
        direct = spec.path == "direct"
        p95 = 37.0 if direct else 37.1
        return _run(self.ov, spec.path, spec.rps, spec.stream, p95, None if direct else 0.005)


ARGS = ["--rps", "100", "--skip-learned", "--warmup", "0", "--duration", "1"]


def test_main_refuses_a_coarse_clock_before_measuring(
    ov: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline = _Pipeline(ov, monkeypatch, tmp_path, COARSE_CLOCK)
    out = tmp_path / "out"

    code = ov.main([*ARGS, "--out", str(out)])

    assert code != 0
    assert pipeline.measured == []
    assert not out.exists()
    err = capsys.readouterr().err
    assert "503000 ns" in err and "--allow-coarse-clock" in err


def test_main_records_the_clock_everywhere_on_a_fine_clock(
    ov: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pipeline = _Pipeline(ov, monkeypatch, tmp_path, FINE_CLOCK)
    out = tmp_path / "out"

    assert ov.main([*ARGS, "--out", str(out)]) == 0

    assert pipeline.measured == [
        "direct-100rps-nonstream",
        "gateway-100rps-nonstream",
        "direct-100rps-stream",
        "gateway-100rps-stream",
    ]
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["clock"] == {"clock_step_ns": 40, "zero_fraction": 0.0, "coarse": False}
    info = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert info["measurement"] == FINE_CLOCK
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Overhead, client side" in report
    assert "40 ns" in next(line for line in report.splitlines() if line.startswith("Labels:"))


def test_main_with_allow_coarse_clock_measures_but_omits_the_sub_millisecond_numbers(
    ov: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pipeline = _Pipeline(ov, monkeypatch, tmp_path, COARSE_CLOCK)
    out = tmp_path / "out"

    assert ov.main([*ARGS, "--allow-coarse-clock", "--out", str(out)]) == 0

    assert len(pipeline.measured) == 4
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["clock"] == {"clock_step_ns": 503_000, "zero_fraction": 0.9999, "coarse": True}
    info = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert info["measurement"] == COARSE_CLOCK
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Overhead, client side" not in report
    assert "## Overhead, server side" not in report
    assert any(line.startswith("Omitted:") for line in report.splitlines())
