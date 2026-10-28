import importlib.util
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
