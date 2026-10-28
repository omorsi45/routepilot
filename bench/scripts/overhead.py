"""Gateway overhead: RoutePilot against the mock provider, at fixed request rates.

Run from the repository root with the repo venv (Go must be on PATH, and the learned run needs
``make data`` and ``make train`` first because it reads the flagship pool config and artifact):

    python bench/scripts/overhead.py

Every measured cell starts its own mock (``rp-mock``, constant 20 ms time to first token, constant
1 ms per output token, 16 output tokens) and, for the gateway paths, its own ``routepilot serve``
on loopback ports picked at run time. ``rp-load`` (the open-loop, coordinated-omission-safe
generator) sends the same request body in both paths: directly to the mock, and through the
gateway. The matrix is {100, 500, 1000, 2000} requests per second x {non-streaming, streaming}
x {direct, gateway}, each with a 10 s warmup and a 60 s measurement window, plus one learned
policy run (the flagship artifact behind the ``auto`` model) at 1000 requests per second,
non-streaming, through the gateway only.

Two overhead measures are reported, and both are labeled in the output:

* client side: the percentiles the generator saw through the gateway minus the percentiles it saw
  directly at the same rate and mode (a difference of percentiles, not a percentile of
  differences);
* server side: ``rp_gateway_overhead_seconds`` scraped from the gateway's ``/metrics`` right after
  the warmup and again after the run. The bucket counts between the two scrapes exclude the
  warmup, and p50, p95 and p99 are interpolated inside the buckets (from histogram buckets, so
  their resolution is the bucket width, which the output gives).

Writes bench/results/overhead/<run-id>/{results.json,report.md,run.json}. Output holds numbers
only; the request body is a fixed invented sentence and the key is a fake benchmark key. The
processes bind loopback only and are always stopped, also on an error.

``--render DIR`` rewrites report.md of an earlier run from its results.json and run.json.

Timing below a microsecond needs a clock that ticks that finely. Before any cell runs, the script
calls ``rp-load --clock-probe`` and records the clock step and the share of equal consecutive
reads in results.json, in the report's Labels line and in run.json (under ``measurement``). When
the step is above 1000 ns the script stops with an error before measuring, because the
sub-millisecond quantities it reports would be counts of clock ticks (this is what a Windows host
with a 503 us tick produced). ``--allow-coarse-clock`` measures anyway and leaves every
sub-millisecond quantity out of the report, with a line saying why.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import yaml

from routepilot_train import runinfo

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "overhead"
TEMPLATE = ROOT / "bench" / "configs" / "overhead.yaml"
POOL_CONFIG = ROOT / "data" / "derived" / "pools" / "flagship.config.yaml"
POLICY_ROOT = ROOT / "policies"
POLICY_NAME = "flagship"
POLICY_CONFIG = ROOT / "bench" / "policies" / "flagship.yaml"
LATENCY_PROFILES = ROOT / "bench" / "latency" / "profiles.yaml"

BENCH_KEY = "bench-key-local"  # a fake benchmark key; its SHA-256 is computed at run time
BENCH_MODEL = "bench-model"
LEARNED_MODEL = "auto"
LEVELS = (100, 500, 1000, 2000)
LEARNED_RPS = 1000
WARMUP_S = 10
DURATION_S = 60
MAX_INFLIGHT = 4096
MAX_TOKENS = 16
LOAD_SEED = 1
MOCK_SEED = 1
# Constant (sigma 0) mock delays: 20 ms to the first token, then 1 ms per token, 16 tokens.
MOCK_ARGS = (
    *("--seed", str(MOCK_SEED)),
    *("--ttft-median-ms", "20", "--ttft-sigma", "0"),
    *("--tpot-median-ms", "1", "--tpot-sigma", "0"),
    *("--tokens", "16"),
)
TARGET_P95_MS = 15.0
OVERHEAD_METRIC = "rp_gateway_overhead_seconds"
DECISION_METRIC = "rp_decision_duration_seconds"
QUANTILES = (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
CLIENT_KEYS = ("p50", "p90", "p95", "p99", "p99.9")
POLICY_VERSION_HEADER = "x-routepilot-policy-version"
START_MARKER = "rp-load: started"
COOLDOWN_S = 2.0
# A clock that advances in steps above this cannot time the sub-millisecond quantities below.
COARSE_CLOCK_NS = 1000
LABEL_LE = re.compile(r'le="([^"]+)"')

# Only these fields of a flagship pool model are kept in the learned catalog.
POOL_MODEL_FIELDS = (
    "id",
    "rank",
    "context_window",
    "max_output_tokens",
    "prompt_overhead_tokens",
    "capabilities",
    "input_price_per_mtok",
    "output_price_per_mtok",
)

# Requests never go through a proxy: the processes are on loopback.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# --- the clock-resolution guard ----------------------------------------------------------------


class CoarseClockError(RuntimeError):
    """The host clock is too coarse for the sub-millisecond quantities this script reports."""


def parse_probe(text: str) -> dict[str, Any]:
    """The clock step (ns) and zero fraction from the JSON ``rp-load --clock-probe`` prints."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"clock probe: output is not JSON: {text[:200]!r}") from err
    if not isinstance(data, dict):
        raise ValueError(f"clock probe: output is not a JSON object: {text[:200]!r}")
    step, zero = data.get("clock_step_ns"), data.get("zero_fraction")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError(f"clock probe: clock_step_ns must be a positive integer, got {step!r}")
    if isinstance(zero, bool) or not isinstance(zero, int | float) or not 0 <= zero <= 1:
        raise ValueError(f"clock probe: zero_fraction must be in [0, 1], got {zero!r}")
    return {"clock_step_ns": step, "zero_fraction": float(zero)}


def probe_clock(binary: Path) -> dict[str, Any]:
    """Run ``rp-load --clock-probe`` (about 0.2 s) and return its parsed result."""
    proc = subprocess.run(
        [str(binary), "--clock-probe"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rp-load --clock-probe exited with {proc.returncode}: {proc.stderr}")
    return parse_probe(proc.stdout)


def check_clock(probe: Mapping[str, Any], allow_coarse: bool) -> bool:
    """Whether the clock is too coarse for sub-millisecond quantities.

    Raises CoarseClockError for a coarse clock unless ``allow_coarse``; with it, returns True
    so the caller leaves those quantities out of the report.
    """
    step = probe["clock_step_ns"]
    if step <= COARSE_CLOCK_NS:
        return False
    if not allow_coarse:
        raise CoarseClockError(
            f"the clock step on this host is {step} ns ({probe['zero_fraction']:.4f} of "
            f"consecutive time.Now reads are equal), above {COARSE_CLOCK_NS} ns, so "
            "sub-millisecond overhead cannot be measured here. Run on a host with a finer clock "
            "(for example Linux), or pass --allow-coarse-clock to measure anyway and leave every "
            "sub-millisecond quantity out of the report."
        )
    return True


# --- histograms and quantiles ------------------------------------------------------------------


@dataclass(frozen=True)
class Histogram:
    """A Prometheus histogram: cumulative (upper bound, count) pairs, ending at +Inf."""

    buckets: tuple[tuple[float, float], ...]
    count: float
    total: float


@dataclass(frozen=True)
class Quantile:
    """A quantile estimate in seconds and the bucket (lower, upper] that holds it."""

    value: float
    lower: float
    upper: float


def parse_histogram(text: str, name: str) -> Histogram:
    """Read the histogram ``name`` from Prometheus text exposition.

    Only the three line shapes of one histogram are read (``<name>_bucket{le="x"} n``,
    ``<name>_sum x`` and ``<name>_count n``); the gateway serves this plain format.
    """
    buckets: list[tuple[float, float]] = []
    count: float | None = None
    total: float | None = None
    for line in text.splitlines():
        if line.startswith(f"{name}_bucket{{"):
            labels, _, value = line.rpartition(" ")
            match = LABEL_LE.search(labels)
            if match is None:
                raise ValueError(f"{name}: bucket line without le: {line!r}")
            buckets.append((float(match.group(1)), float(value)))
        elif line.startswith(f"{name}_count "):
            count = float(line.split()[1])
        elif line.startswith(f"{name}_sum "):
            total = float(line.split()[1])
    if not buckets or count is None or total is None:
        raise ValueError(f"histogram {name} not found in the scrape")
    buckets.sort()
    return Histogram(tuple(buckets), count, total)


def subtract(after: Histogram, before: Histogram) -> Histogram:
    """The observations between two scrapes of the same histogram."""
    if [b for b, _ in after.buckets] != [b for b, _ in before.buckets]:
        raise ValueError("the two scrapes have different bucket bounds")
    merged = tuple((b, a - p) for (b, a), (_, p) in zip(after.buckets, before.buckets, strict=True))
    if any(c < 0 for _, c in merged) or after.count < before.count:
        raise ValueError("a bucket count decreased between the scrapes")
    return Histogram(merged, after.count - before.count, after.total - before.total)


def bucket_quantile(h: Histogram, q: float) -> Quantile:
    """The q-quantile by linear interpolation inside the bucket that holds it.

    This is Prometheus's ``histogram_quantile``: the first bucket starts at 0, and a quantile
    above the last finite bound is reported as that bound (its bucket is then (bound, +Inf]).
    """
    total = h.buckets[-1][1]
    if total <= 0:
        raise ValueError("the histogram has no observations")
    rank = q * total
    lower, below = 0.0, 0.0
    for bound, cumulative in h.buckets:
        if cumulative >= rank:
            if math.isinf(bound):
                return Quantile(lower, lower, bound)
            width = cumulative - below
            fraction = (rank - below) / width if width > 0 else 1.0
            return Quantile(lower + (bound - lower) * fraction, lower, bound)
        lower, below = bound, cumulative
    raise ValueError("the histogram is not cumulative")  # unreachable for a sorted scrape


def _quantile_ms(q: Quantile) -> dict[str, float | None]:
    return {
        "value_ms": q.value * 1000,
        "lower_ms": q.lower * 1000,
        "upper_ms": None if math.isinf(q.upper) else q.upper * 1000,
    }


def server_summary(before: Histogram, after: Histogram) -> dict[str, Any] | None:
    """Count, mean and p50, p95, p99 of the observations between two scrapes.

    None when the window holds no observation.
    """
    delta = subtract(after, before)
    if delta.count <= 0:
        return None
    summary: dict[str, Any] = {"count": delta.count, "mean_ms": delta.total / delta.count * 1000}
    for key, q in QUANTILES:
        summary[key] = _quantile_ms(bucket_quantile(delta, q))
    return summary


# --- verdicts and comparisons ------------------------------------------------------------------


def difference_of_percentiles(
    gateway: Mapping[str, float], direct: Mapping[str, float]
) -> dict[str, float]:
    """Gateway minus direct, percentile by percentile (not the percentiles of per-request gaps)."""
    return {
        k: round(gateway[k] - direct[k], 6) for k in CLIENT_KEYS if k in gateway and k in direct
    }


def client_verdict(p95_difference_ms: float) -> str:
    return "met" if p95_difference_ms < TARGET_P95_MS else "not met"


def server_verdict(p95: Quantile) -> str:
    """Whether a bucketed p95 is under the target; it is undecided when a bucket spans it."""
    target = TARGET_P95_MS / 1000
    if p95.upper < target:
        return "met"
    if p95.lower >= target:
        return "not met"
    return f"undecided (bucket spans {TARGET_P95_MS:g} ms)"


def _server_verdict_from_json(q: Mapping[str, float | None]) -> str:
    upper = math.inf if q["upper_ms"] is None else q["upper_ms"] / 1000
    value, lower = (q["value_ms"] or 0.0) / 1000, (q["lower_ms"] or 0.0) / 1000
    return server_verdict(Quantile(value, lower, upper))


@dataclass(frozen=True)
class Spec:
    """One measured cell: where the load goes, at what rate, streaming or not."""

    path: str  # "direct" (to the mock), "gateway" (pinned model) or "learned" (auto model)
    rps: int
    stream: bool

    @property
    def mode(self) -> str:
        return "stream" if self.stream else "nonstream"

    @property
    def name(self) -> str:
        return f"{self.path}-{self.rps}rps-{self.mode}"

    @property
    def model(self) -> str:
        return LEARNED_MODEL if self.path == "learned" else BENCH_MODEL


def matrix(levels: Sequence[int], learned_rps: int | None) -> list[Spec]:
    """Every cell in run order: per rate, non-streaming then streaming, direct then gateway."""
    specs = [
        Spec(path, rps, stream)
        for rps in levels
        for stream in (False, True)
        for path in ("direct", "gateway")
    ]
    if learned_rps is not None:
        specs.append(Spec("learned", learned_rps, False))
    return specs


def _find(
    runs: Sequence[Mapping[str, Any]], path: str, rps: int, stream: bool
) -> Mapping[str, Any]:
    for run in runs:
        if (run["path"], run["rps"], run["stream"]) == (path, rps, stream):
            return run
    raise KeyError(f"no {path} run at {rps} rps, stream={stream}")


def _compare_to(base: Mapping[str, Any], other: Mapping[str, Any], kind: str) -> dict[str, Any]:
    diff = difference_of_percentiles(other["load"]["latency_ms"], base["load"]["latency_ms"])
    ttft = None
    if other["load"].get("ttft_ms") and base["load"].get("ttft_ms"):
        ttft = difference_of_percentiles(other["load"]["ttft_ms"], base["load"]["ttft_ms"])
    server = (other.get("server") or {}).get("overhead")
    return {
        "kind": kind,
        "rps": other["rps"],
        "stream": other["stream"],
        "base": base["name"],
        "other": other["name"],
        "client_diff_ms": diff,
        "ttft_diff_ms": ttft,
        "client_verdict": client_verdict(diff["p95"]),
        "server": server,
        "server_verdict": _server_verdict_from_json(server["p95"]) if server else None,
    }


def compare(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Gateway against direct at every rate and mode; the learned run against both baselines."""
    out: list[dict[str, Any]] = []
    for run in runs:
        if run["path"] == "gateway":
            direct = _find(runs, "direct", run["rps"], run["stream"])
            out.append(_compare_to(direct, run, "gateway_vs_direct"))
    for run in runs:
        if run["path"] == "learned":
            direct = _find(runs, "direct", run["rps"], run["stream"])
            out.append(_compare_to(direct, run, "learned_vs_direct"))
            gateway = _find(runs, "gateway", run["rps"], run["stream"])
            out.append(_compare_to(gateway, run, "learned_vs_gateway"))
    return out


# --- gateway configs ---------------------------------------------------------------------------


def key_sha256(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def gateway_config(gateway_port: int, mock_port: int, key: str) -> dict[str, Any]:
    """The pinned-model gateway config: the template with ports and the key hash filled in."""
    text = TEMPLATE.read_text(encoding="utf-8")
    fills = {
        "{{GATEWAY_PORT}}": str(gateway_port),
        "{{MOCK_PORT}}": str(mock_port),
        "{{KEY_SHA256}}": key_sha256(key),
    }
    for token, value in fills.items():
        text = text.replace(token, value)
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"{TEMPLATE} is not a YAML mapping")
    return cfg


def learned_config(
    gateway_port: int,
    mock_port: int,
    key: str,
    pool: Mapping[str, Any],
    artifact_version: str,
) -> dict[str, Any]:
    """The learned-policy config: the flagship catalog, each model with one mock deployment.

    The catalog (ids, ranks, prices) comes from the flagship pool config the artifact was trained
    for. The upstream model name is the model id, which carries no mock directive.
    """
    cfg = gateway_config(gateway_port, mock_port, key)
    models = []
    for source in pool["models"]:
        model = {k: source[k] for k in POOL_MODEL_FIELDS if k in source}
        model["deployments"] = [
            {"name": f"{source['id']}-mock", "provider": "mock", "upstream_model": source["id"]}
        ]
        models.append(model)
    cfg["models"] = models
    cfg["routing"] = {"policy": "learned"}
    cfg["policies"] = {
        "root": POLICY_ROOT.as_posix(),
        "stable": {
            "name": POLICY_NAME,
            "version": artifact_version,
            "config": POLICY_CONFIG.as_posix(),
        },
    }
    cfg["server"]["latency_profiles"] = LATENCY_PROFILES.as_posix()
    return cfg


def flagship_artifact_version() -> str:
    versions = sorted(p.name for p in (POLICY_ROOT / POLICY_NAME).glob("*") if p.is_dir())
    if len(versions) != 1:
        raise RuntimeError(
            f"want exactly one artifact under policies/{POLICY_NAME} (run make train), "
            f"found {len(versions)}"
        )
    return versions[0]


# --- processes ---------------------------------------------------------------------------------


def free_port() -> int:
    """A loopback port nothing listens on (bind port 0 and read it back)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_binaries(bin_dir: Path) -> dict[str, Path]:
    """Build rp-mock, routepilot and rp-load from this tree into bin_dir."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    env = {**os.environ}
    env.setdefault("GOTOOLCHAIN", "local")
    built = {}
    for name in ("rp-mock", "routepilot", "rp-load"):
        target = bin_dir / f"{name}{suffix}"
        proc = subprocess.run(
            ["go", "build", "-o", str(target), f"./cmd/{name}"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"go build ./cmd/{name} failed: {proc.stderr.strip()}")
        built[name] = target
    return built


def stop(proc: subprocess.Popen[Any]) -> None:
    """Terminate, wait up to 10 s, then kill."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@contextlib.contextmanager
def service(command: Sequence[str], log: Path) -> Iterator[subprocess.Popen[bytes]]:
    """Run a process with its output in ``log`` and always stop it."""
    with log.open("wb") as sink:
        proc = subprocess.Popen(
            list(command), cwd=ROOT, stdin=subprocess.DEVNULL, stdout=sink, stderr=sink
        )
        try:
            yield proc
        finally:
            stop(proc)


def wait_ready(url: str, proc: subprocess.Popen[Any], timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited with code {proc.returncode} before {url} answered")
        try:
            with OPENER.open(url, timeout=1) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"{url} did not answer within {timeout_s:g} s")


def post_chat(url: str, model: str, key: str, stream: bool) -> dict[str, Any]:
    """One request before the load, to fail fast on a wrong config; returns status and headers."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one word."}],
            "max_tokens": MAX_TOKENS,
            "stream": stream,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with OPENER.open(request, timeout=10) as resp:
            resp.read()
            headers = {k.lower(): v for k, v in resp.getheaders()}
            routepilot = {k: v for k, v in headers.items() if k.startswith("x-routepilot-")}
            return {"status": resp.status, "headers": routepilot}
    except urllib.error.HTTPError as err:
        excerpt = err.read(300).decode("utf-8", errors="replace")
        raise RuntimeError(f"preflight request got HTTP {err.code}: {excerpt}") from err


def scrape(port: int) -> str:
    with OPENER.open(f"http://127.0.0.1:{port}/metrics", timeout=10) as resp:
        return str(resp.read().decode("utf-8"))


def _pump(
    stream: TextIO, lines: list[str], marker: threading.Event | None, stamp: list[float]
) -> None:
    """Collect a pipe's lines; the first start marker sets ``marker`` and records its time."""
    for line in stream:
        lines.append(line.rstrip("\n"))
        if marker is not None and line.startswith(START_MARKER) and not marker.is_set():
            stamp.append(time.monotonic())
            marker.set()


def run_load(
    binary: Path,
    url: str,
    spec: Spec,
    key: str,
    metrics_port: int | None,
    warmup_s: int,
    duration_s: int,
) -> tuple[dict[str, Any], tuple[str, str] | None]:
    """Run rp-load; return its JSON result and, for a gateway, two /metrics scrapes.

    The first scrape is taken as the warmup ends and the second after the run, so the bucket
    counts between them are the measurement window.
    """
    command = [
        *(str(binary), "--url", url, "--model", spec.model, "--rps", str(spec.rps)),
        *("--warmup", f"{warmup_s}s", "--duration", f"{duration_s}s", "--seed", str(LOAD_SEED)),
        *("--max-inflight", str(MAX_INFLIGHT), "--max-tokens", str(MAX_TOKENS)),
    ]
    if spec.stream:
        command.append("--stream")
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env={**os.environ, "RP_LOAD_API_KEY": key},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("rp-load pipes are missing")
    started = threading.Event()
    stamp: list[float] = []
    out_lines: list[str] = []
    err_lines: list[str] = []
    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, out_lines, None, stamp), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, err_lines, started, stamp), daemon=True),
    ]
    for thread in threads:
        thread.start()
    scrapes: tuple[str, str] | None = None
    try:
        if not started.wait(30):
            raise RuntimeError("rp-load did not start: " + " | ".join(err_lines))
        before = None
        if metrics_port is not None:
            # The first arrival is scheduled right after the start marker, so the warmup ends
            # warmup_s later.
            time.sleep(max(0.0, stamp[0] + warmup_s - time.monotonic()))
            before = scrape(metrics_port)
        code = proc.wait(timeout=warmup_s + duration_s + 120)
        for thread in threads:
            thread.join(10)
        if code != 0:
            raise RuntimeError(f"rp-load exited with {code}: " + " | ".join(err_lines))
        if metrics_port is not None and before is not None:
            scrapes = (before, scrape(metrics_port))
    finally:
        stop(proc)
    result = json.loads("\n".join(out_lines))
    if not isinstance(result, dict):
        raise RuntimeError("rp-load did not print a JSON object")
    return result, scrapes


# --- one cell ----------------------------------------------------------------------------------


def measure(
    spec: Spec,
    bins: Mapping[str, Path],
    scratch: Path,
    warmup_s: int,
    duration_s: int,
    learned: tuple[Mapping[str, Any], str] | None,
) -> dict[str, Any]:
    """Start the mock (and the gateway), check one request, run the load, stop everything."""
    mock_port = free_port()
    mock_command = [str(bins["rp-mock"]), "--listen", f"127.0.0.1:{mock_port}", *MOCK_ARGS]
    with contextlib.ExitStack() as stack:
        mock = stack.enter_context(service(mock_command, scratch / f"{spec.name}.mock.log"))
        wait_ready(f"http://127.0.0.1:{mock_port}/v1/models", mock)

        gateway_port: int | None = None
        if spec.path == "direct":
            url = f"http://127.0.0.1:{mock_port}/v1/chat/completions"
        else:
            gateway_port = free_port()
            if spec.path == "learned":
                if learned is None:
                    raise RuntimeError("the learned run needs the flagship pool and artifact")
                pool, version = learned
                cfg = learned_config(gateway_port, mock_port, BENCH_KEY, pool, version)
            else:
                cfg = gateway_config(gateway_port, mock_port, BENCH_KEY)
            config_path = scratch / f"{spec.name}.yaml"
            config_path.write_text(
                yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n"
            )
            gateway_command = [str(bins["routepilot"]), "serve", "-config", str(config_path)]
            gateway = stack.enter_context(
                service(gateway_command, scratch / f"{spec.name}.gateway.log")
            )
            wait_ready(f"http://127.0.0.1:{gateway_port}/healthz", gateway)
            url = f"http://127.0.0.1:{gateway_port}/v1/chat/completions"

        preflight = post_chat(url, spec.model, BENCH_KEY, spec.stream)
        if spec.path == "learned":
            served = preflight["headers"].get(POLICY_VERSION_HEADER, "")
            if served in ("", "pinned") or served.startswith("static-"):
                raise RuntimeError(f"the learned run is not using the learned policy: {preflight}")
            preflight["policy_version"] = served

        load, scrapes = run_load(
            bins["rp-load"], url, spec, BENCH_KEY, gateway_port, warmup_s, duration_s
        )

    server = None
    if scrapes is not None:
        before, after = scrapes
        server = {
            metric: server_summary(parse_histogram(before, name), parse_histogram(after, name))
            for metric, name in (("overhead", OVERHEAD_METRIC), ("decision", DECISION_METRIC))
        }
    return {
        "name": spec.name,
        "path": spec.path,
        "rps": spec.rps,
        "stream": spec.stream,
        "model": spec.model,
        "preflight": preflight,
        "load": load,
        "server": server,
    }


# --- report ------------------------------------------------------------------------------------


def _f(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _pcts(load: Mapping[str, Any], block: str, keys: Sequence[str]) -> list[str]:
    """The named percentiles of a block of an rp-load result, or dashes when it is null."""
    data = load.get(block)
    return [_f(data[k]) if data else "-" for k in keys]


def _q(server: Mapping[str, Any] | None, key: str) -> str:
    """A bucketed quantile as ``value (lower-upper]`` in ms, to the microsecond."""
    if not server:
        return "-"
    q = server[key]
    upper = "inf" if q["upper_ms"] is None else _f(q["upper_ms"], 3)
    return f"{_f(q['value_ms'], 3)} ({_f(q['lower_ms'], 3)}-{upper}]"


def _mode(stream: bool) -> str:
    return "streaming" if stream else "non-streaming"


LIMITATIONS = """\
## Limitations

- One run per cell: there are no repetitions and no confidence intervals. A difference of a \
fraction of a millisecond is within the run-to-run noise of a shared machine.
- The generator, the gateway and the mock compete for the same CPU, and other programs on this \
machine could not be controlled. The launch lag table shows how much the generator itself added.
- The gateway writes one log line per request to a file; accounting, the decision log and tracing \
are off, and the tenant has no limits (see bench/configs/overhead.yaml).
- The server-side quantiles have the resolution of the histogram buckets (about a factor of 2.15 \
per bucket), so a p95 can be undecided against the target.
- The mock answers with constant delays and tiny bodies; real providers add variable latency, and \
larger bodies would cost the gateway more to copy.
- All requests carry one fixed short prompt.
"""


def _clock_label(clock: Mapping[str, Any] | None) -> str:
    """The sentence of the Labels line about the measured clock step ('' for older results)."""
    if not clock:
        return ""
    step, zero = clock["clock_step_ns"], clock["zero_fraction"]
    text = (
        f"Measured with rp-load --clock-probe before the run, the clock step is {step} ns "
        f"(the smallest non-zero difference between consecutive time.Now reads; {zero:.4f} of "
        "the consecutive reads were equal)."
    )
    if clock.get("coarse"):
        text += f" That is above {COARSE_CLOCK_NS} ns, so sub-millisecond quantities are omitted."
    return text


def _intro(
    results: Mapping[str, Any], machine: Mapping[str, Any], run_id: str, go_version: str | None
) -> list[str]:
    settings = results["settings"]
    go = f", {go_version}" if go_version else ""
    clock = _clock_label(results.get("clock"))
    coarse = bool((results.get("clock") or {}).get("coarse"))
    return [
        f"# Gateway overhead against the local mock ({run_id})",
        "",
        "Labels: real local measurement, nothing simulated. Machine: "
        f"{machine['cpu']}, {machine['ram_gb']} GB RAM, {machine['os']}{go}. The load generator "
        "(rp-load), the gateway (routepilot serve) and the mock (rp-mock) share one machine and "
        "talk over loopback. The mock's delays are configured constants (time to first token "
        "20 ms, 1 ms per output token, 16 output tokens), so absolute latencies are the mock's; "
        "only the difference between the gateway path and the direct path measures RoutePilot. "
        "Latencies run from each request's scheduled time (open loop, coordinated-omission safe)."
        + (f" {clock}" if clock else ""),
        "",
        f"Settings: {settings['warmup_s']} s warmup (sent, not recorded) and "
        f"{settings['duration_s']} s measurement window per cell, one run per cell, request body "
        f"a fixed invented sentence of about 200 bytes with max_tokens {MAX_TOKENS}, generator "
        f"max in-flight {MAX_INFLIGHT}, a fresh mock and gateway process for every cell.",
        "",
        f"Target: gateway overhead at p95 below {results['target_p95_ms']:g} ms. It is a target, "
        "not a guarantee" + ("." if coarse else "; each row below says whether it was met."),
        "",
    ]


def _runs_section(runs: Sequence[Mapping[str, Any]], with_lag: bool = True) -> list[str]:
    all_keys = ("p50", "p90", "p95", "p99", "p99.9", "max")
    rows = []
    for run in runs:
        load = run["load"]
        counts = [str(load[k]) for k in ("scheduled", "completed", "errors", "dropped")]
        rows.append(
            [run["name"], *counts, _f(load["throughput_rps"], 1)]
            + _pcts(load, "latency_ms", all_keys)
        )
    headers = ["run", "scheduled", "completed", "errors", "dropped", "throughput rps", *all_keys]
    out = ["## Runs (client side, ms)", "", _table(headers, rows), ""]
    streams = [r for r in runs if r["stream"]]
    if streams:
        ttft = [[r["name"], *_pcts(r["load"], "ttft_ms", all_keys)] for r in streams]
        out += ["Time to first token (streaming runs, ms):", ""]
        out += [_table(["run", *all_keys], ttft), ""]
    if not with_lag:
        return out
    lag_keys = ("p50", "p99", "max")
    lag = [[r["name"], *_pcts(r["load"], "launch_lag_ms", lag_keys)] for r in runs]
    out += [
        "Generator launch lag (how late the scheduler started a request, ms; it is part of "
        "every latency above):",
        "",
        _table(["run", *lag_keys], lag),
        "",
    ]
    return out


def _omitted_section(clock: Mapping[str, Any]) -> list[str]:
    return [
        "Omitted: the server-side overhead quantiles, the client-side differences between the "
        "gateway and direct paths, the generator launch lag and the learned-policy comparison, "
        f"because the clock step on this host is {clock['clock_step_ns']} ns, above the "
        f"{COARSE_CLOCK_NS} ns limit: a duration below about a millisecond is then a count of "
        "clock ticks, not a measurement. No verdict against the p95 target is given. Run the "
        "experiment on a host with a finer clock (for example Linux) to measure the overhead.",
        "",
    ]


def _client_section(gateway: Sequence[Mapping[str, Any]], target: float) -> list[str]:
    rows = [
        [str(c["rps"]), _mode(c["stream"])]
        + [_f(c["client_diff_ms"].get(k)) for k in CLIENT_KEYS]
        + [c["client_verdict"]]
        for c in gateway
    ]
    out = [
        "## Overhead, client side: gateway minus direct (ms)",
        "",
        "Label: difference of percentiles, not a percentile of differences. Each cell is the "
        "percentile seen through the gateway minus the same percentile seen directly at the same "
        "rate and mode.",
        "",
        _table(["rate", "mode", *CLIENT_KEYS, f"p95 < {target:g} ms"], rows),
        "",
    ]
    streaming = [c for c in gateway if c["ttft_diff_ms"]]
    if streaming:
        ttft = [
            [str(c["rps"])] + [_f(c["ttft_diff_ms"].get(k)) for k in CLIENT_KEYS] for c in streaming
        ]
        out += ["Time to first token, gateway minus direct (ms, difference of percentiles):", ""]
        out += [_table(["rate", *CLIENT_KEYS], ttft), ""]
    return out


def _server_section(
    gateway: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]], target: float
) -> list[str]:
    by_name = {r["name"]: r for r in runs}
    rows = []
    for c in gateway:
        server = c["server"]
        completed = by_name[c["other"]]["load"]["completed"]
        rows.append(
            [
                str(c["rps"]),
                _mode(c["stream"]),
                _f(server["count"], 0) if server else "-",
                str(completed),
                *(_q(server, k) for k in ("p50", "p95", "p99")),
                c["server_verdict"] or "-",
            ]
        )
    headers = ["rate", "mode", "server n", "completed", "p50", "p95", "p99", f"p95 < {target:g} ms"]
    return [
        "## Overhead, server side: rp_gateway_overhead_seconds (ms, from histogram buckets)",
        "",
        "Label: from histogram buckets. The gateway defines this series as the request duration "
        "minus the summed duration of its upstream attempts, so it is the gateway's own "
        "accounting and need not equal the client-side difference. "
        "The gateway's /metrics was scraped right after the warmup "
        "and after the run; the bucket counts between the two exclude the warmup. Quantiles are "
        "interpolated inside a bucket and shown as value (bucket lower-upper]; the true value lies "
        "inside that bucket. 'server n' is the number of observations in the window and "
        "'completed' is the generator's count; they differ by the requests in flight at the "
        "boundary and by failed attempts.",
        "",
        _table(headers, rows),
        "",
    ]


def _summary_section(gateway: Sequence[Mapping[str, Any]]) -> list[str]:
    out = ["## Target summary", ""]
    for c in gateway:
        out.append(
            f"- {c['rps']} rps, {_mode(c['stream'])}: client-side p95 difference "
            f"{_f(c['client_diff_ms'].get('p95'))} ms, {c['client_verdict']}; server-side p95 "
            f"{_q(c['server'], 'p95')} ms, {c['server_verdict'] or '-'}."
        )
    return [*out, ""]


def _learned_section(
    comparisons: Sequence[Mapping[str, Any]], runs: Sequence[Mapping[str, Any]], target: float
) -> list[str]:
    learned = [c for c in comparisons if c["kind"].startswith("learned")]
    if not learned:
        return []
    rows = [
        [c["kind"].replace("_", " "), str(c["rps"])]
        + [_f(c["client_diff_ms"].get(k)) for k in CLIENT_KEYS]
        + [c["client_verdict"]]
        for c in learned
    ]
    run = next(r for r in runs if r["path"] == "learned")
    server = run["server"] or {}
    keys = ("p50", "p95", "p99")
    overhead = ["rp_gateway_overhead_seconds (learned run)"]
    overhead += [_q(server.get("overhead"), k) for k in keys]
    decision = ["rp_decision_duration_seconds (learned run)"]
    decision += [_q(server.get("decision"), k) for k in keys]
    out = [
        "## Learned policy run (flagship artifact behind the auto model, gateway only)",
        "",
        "The learned run is compared with the direct run (the whole cost of going through "
        "RoutePilot with the learned policy) and with the pinned-model gateway run at the same "
        "rate (what the policy adds). Every request carries the same prompt, so the decision "
        "takes the same path each time. Differences of percentiles, not percentiles of "
        "differences.",
        "",
        _table(["comparison", "rate", *CLIENT_KEYS, f"p95 < {target:g} ms"], rows),
        "",
        _table(["server side (from histogram buckets, ms)", *keys], [overhead, decision]),
        "",
    ]
    version = run.get("preflight", {}).get("policy_version")
    if version:
        out += [f"Policy version served: {version}.", ""]
    return out


def render_report(
    results: Mapping[str, Any], machine: Mapping[str, Any], run_id: str, go_version: str | None
) -> str:
    """The report.md text for a results dict (the content of results.json)."""
    runs = results["runs"]
    target = results["target_p95_ms"]
    clock = results.get("clock")
    coarse = bool(clock and clock.get("coarse"))
    gateway = [c for c in results["comparisons"] if c["kind"] == "gateway_vs_direct"]
    out = _intro(results, machine, run_id, go_version)
    out += _runs_section(runs, with_lag=not coarse)
    if coarse and clock:
        out += _omitted_section(clock)
    else:
        out += _client_section(gateway, target)
        out += _server_section(gateway, runs, target)
        out += _summary_section(gateway)
        out += _learned_section(results["comparisons"], runs, target)
    out.append(LIMITATIONS)
    return "\n".join(out)


# --- main --------------------------------------------------------------------------------------


def _levels(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gateway overhead against the local mock.")
    parser.add_argument("--rps", type=_levels, default=LEVELS, help="comma-separated rates")
    parser.add_argument("--warmup", type=int, default=WARMUP_S, help="warmup seconds per cell")
    parser.add_argument(
        "--duration", type=int, default=DURATION_S, help="measured seconds per cell"
    )
    parser.add_argument("--learned-rps", type=int, default=LEARNED_RPS, help="rate of learned run")
    parser.add_argument("--skip-learned", action="store_true", help="skip the learned-policy run")
    parser.add_argument("--out", type=Path, help="result directory (default: under bench/results)")
    parser.add_argument("--keep-scratch", action="store_true", help="keep logs and configs")
    parser.add_argument(
        "--allow-coarse-clock",
        action="store_true",
        help=f"measure on a clock step above {COARSE_CLOCK_NS} ns, leaving sub-millisecond "
        "quantities out of the report",
    )
    parser.add_argument("--render", type=Path, metavar="DIR", help="rewrite DIR/report.md and exit")
    return parser


def _rewrite_report(directory: Path) -> int:
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    info = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    text = render_report(results, info["machine"], info["run_id"], info["go_version"])
    (directory / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"overhead: wrote {(directory / 'report.md').as_posix()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.render:
        return _rewrite_report(args.render)

    started = datetime.now(UTC)
    clock = time.monotonic()
    run_id = runinfo.new_run_id(started)
    out_dir = args.out or ROOT / "bench" / "results" / EXPERIMENT / run_id
    scratch = ROOT / "bench" / "scratch" / f"{EXPERIMENT}-{run_id}"
    scratch.mkdir(parents=True, exist_ok=True)

    learned_rps = None if args.skip_learned else args.learned_rps
    learned: tuple[Mapping[str, Any], str] | None = None
    if learned_rps is not None:
        pool = yaml.safe_load(POOL_CONFIG.read_text(encoding="utf-8"))
        learned = (pool, flagship_artifact_version())
    specs = matrix(args.rps, learned_rps)

    print(f"overhead: building binaries ({run_id})", flush=True)
    bins = build_binaries(scratch / "bin")
    probe = probe_clock(bins["rp-load"])
    print(
        f"overhead: clock step {probe['clock_step_ns']} ns, "
        f"zero fraction {probe['zero_fraction']:.4f}",
        flush=True,
    )
    try:
        coarse = check_clock(probe, args.allow_coarse_clock)
    except CoarseClockError as err:
        print(f"overhead: {err}", file=sys.stderr, flush=True)
        return 1
    runs: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        run = measure(spec, bins, scratch, args.warmup, args.duration, learned)
        runs.append(run)
        load = run["load"]
        latency = load["latency_ms"] or {}
        print(
            f"overhead: [{index}/{len(specs)}] {spec.name} completed={load['completed']} "
            f"errors={load['errors']} dropped={load['dropped']} p95={_f(latency.get('p95'))} ms",
            flush=True,
        )
        time.sleep(COOLDOWN_S)

    results: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "target_p95_ms": TARGET_P95_MS,
        "settings": {
            "warmup_s": args.warmup,
            "duration_s": args.duration,
            "levels": list(args.rps),
            "learned_rps": learned_rps,
            "max_inflight": MAX_INFLIGHT,
            "max_tokens": MAX_TOKENS,
            "mock_args": list(MOCK_ARGS),
        },
        "runs": runs,
        "comparisons": compare(runs),
        "clock": {**probe, "coarse": coarse},
        "wall_time_s": round(time.monotonic() - clock, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    policy_version = next(
        (r["preflight"].get("policy_version") for r in runs if r["path"] == "learned"), None
    )
    info = runinfo.write_run_json(
        out_dir,
        EXPERIMENT,
        {"loadgen": LOAD_SEED, "mock": MOCK_SEED},
        ["python", "bench/scripts/overhead.py", *(argv if argv is not None else sys.argv[1:])],
        policy_version,
        started,
        extra=probe,
    )
    report = render_report(results, info["machine"], info["run_id"], info["go_version"])
    (out_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"overhead: wrote {out_dir.as_posix()} in {results['wall_time_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
