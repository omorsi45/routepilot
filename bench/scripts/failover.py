"""Failover: RoutePilot against a deployment that fails, with a second deployment to fail over to.

Run from the repository root with the repo venv (Go must be on PATH):

    python bench/scripts/failover.py

One model has two deployments, each on its own ``rp-mock`` process, so the request count of each
deployment is the mock's own (``GET /mock/stats``). Deployment A comes first in the config, so
every request starts on A unless A's circuit breaker refuses it; deployment B is clean. The gateway
runs with ``max_attempts`` 3 and the decision log on; every number below is read back from that log
(the ``attempts`` of each record), the mocks' own counters, the load generator and ``/metrics``.

Two arms, each against its own fresh mocks and gateway:

* pre-stream: 10,000 non-streaming requests at 100 requests per second (open loop, ``rp-load``).
  Deployment A injects, per request and with a fixed seed, 10% HTTP 500, 5% HTTP 429 with
  Retry-After 1 s and 5% connection resets (20% in all). The script reports the eligible failures
  on A, the requests whose first attempt failed that way, how many of those were answered by B (the
  per-request recovery rate, against the target of 99%), the requests that never tried A because
  its breaker was open (Ruling M2-R21: the breaker stops sending to a failing deployment, so these
  requests are not part of the recovery rate), and every request that failed overall with its
  reason.
* mid-stream: 2,000 streaming requests at 50 requests per second. Deployment A's mock is started
  with ``--fail-rate-disconnect 0.2``: for one stream in five it sends the empty role chunk and the
  first content chunk, which is where the gateway commits the response (it commits at the first
  chunk with content), and then drops the connection; the other four are answered in full, so A's
  circuit breaker (it opens after five failures in a row) stays closed and A keeps receiving most
  of the requests. The rule under test: once a response is committed the gateway never makes a
  second upstream call. The script checks, from the decision log and the mocks' counters, that
  every request made exactly one attempt, that no request touched both deployments, that the
  streams A's mock says it cut are the streams the log shows broken after the commit, and that the
  mocks together received exactly one request per client request.

The schedule length is chosen so the seeded Poisson schedule holds exactly 10,000 and 2,000
arrivals: the duration of each arm lies between the offsets of its last wanted arrival and the
next one (see ``ARMS``); the script fails if the generator scheduled any other number.

Writes bench/results/failover/<run-id>/{results.json,report.md,run.json}. Output holds numbers
only; the request body is the load generator's fixed invented sentence and the key is a fake test
key. Every process binds loopback only and is stopped in a ``finally``, also on an error.
``--render DIR`` rewrites report.md of an earlier run from its results.json and run.json.
"""

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from routepilot_train import runinfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overhead  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "failover"
TEMPLATE = ROOT / "bench" / "configs" / "failover.yaml"

KEY = "test-key-failover"  # a fake key; its SHA-256 is computed at run time
MODEL = "failover-model"
DEPLOYMENT_A = "deployment-a"
DEPLOYMENT_B = "deployment-b"
MAX_ATTEMPTS = 3
MAX_TOKENS = 16
MAX_INFLIGHT = 4096
TARGET_RECOVERY = 0.99
MOCK_SEEDS = {"a": 1, "b": 2}
# Constant (sigma 0) mock delays, as in the overhead run: 20 ms to the first token, 1 ms per token.
MOCK_DELAYS = (
    *("--ttft-median-ms", "20", "--ttft-sigma", "0"),
    *("--tpot-median-ms", "1", "--tpot-sigma", "0"),
    *("--tokens", "16"),
)
# Deployment A's injected pre-stream failures in the first arm: flag and probability.
FAIL_MIX = (
    ("--fail-rate-http500", 0.10),
    ("--fail-rate-http429", 0.05),  # the mock answers with Retry-After 1 by default
    ("--fail-rate-reset", 0.05),
)
# The second arm's failure on A: the share of streams the mock cuts after the second chunk event
# (the role chunk and the first content chunk), i.e. just after the commit. A rate, not a
# directive on A's model, so A's breaker stays closed and most streams are served by A.
DISCONNECT_RATE = 0.2
# A's upstream model name per arm: neither arm uses a directive.
A_UPSTREAM = {
    "prestream": "bench-model",
    "midstream": "bench-model",
}
# The gateway's own retry list for upstream HTTP statuses (internal/gateway/attempts.go).
RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529, 401, 403})
RETRYABLE_KINDS = frozenset({"connect", "protocol", "timeout"})
HTTP_OUTCOME = re.compile(r"^http_(\d{3})$")
BREAKER_POLL_S = 0.1
LOG_WAIT_S = 30.0
RECORDS_FILE = "decisions.jsonl"
STATE_NAMES = {0: "closed", 1: "half-open", 2: "open"}


@dataclass(frozen=True)
class Arm:
    """One arm of the experiment: how much load, at what rate, and whether it streams."""

    name: str
    rps: int
    duration: str  # a Go duration for rp-load --duration
    requests: int  # the number of arrivals the seeded schedule holds for this duration
    seed: int  # the load generator's seed
    stream: bool


# The duration lies in (offset of arrival N, offset of arrival N+1] of the seeded schedule:
# seed 1 at 100 rps has its 10,000th arrival at 101.452002 s and the next at 101.466307 s; seed 2
# at 50 rps has its 2,000th at 39.180132 s and the next at 39.238991 s.
ARMS = {
    "prestream": Arm("prestream", 100, "101.46s", 10_000, 1, False),
    "midstream": Arm("midstream", 50, "39.2s", 2_000, 2, True),
}
SMOKE_DURATIONS = {"prestream": "10s", "midstream": "8s"}


# --- classifying the decision log --------------------------------------------------------------


def is_eligible_failure(outcome: str) -> bool:
    """Whether an attempt outcome is a failure the gateway may follow with another attempt.

    The outcome vocabulary is the usage ledger's: ``http_<status>`` for an upstream status, or a
    provider failure kind. The statuses are the gateway's retry list, and a connect or protocol
    error or a timeout is retryable too. ``canceled`` (the client left), ``aborted`` and ``error``
    are not.
    """
    if outcome in RETRYABLE_KINDS:
        return True
    match = HTTP_OUTCOME.match(outcome)
    return bool(match) and int(match.group(1)) in RETRYABLE_STATUSES


def _attempts(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(record.get("attempts") or [])


def classify_attempts(
    records: Sequence[Mapping[str, Any]],
    a: str = DEPLOYMENT_A,
    b: str = DEPLOYMENT_B,
) -> dict[str, Any]:
    """Count what the decision log says happened to each request.

    ``records`` are decision records (anything of another ``type``, and shadow records, is
    ignored). Each carries the HTTP ``status`` the client got and its ``attempts``, in send order,
    each with a ``deployment`` and an ``outcome`` (``ok`` or a failure).

    * An eligible failure is an attempt whose outcome ``is_eligible_failure``.
    * ``recovered`` are the requests whose first attempt failed eligibly and that were then
      answered (status 200 with the last attempt ``ok``); ``recovered_on_b`` those answered by B.
      ``recovery_rate`` is recovered over the requests whose first attempt failed eligibly (None
      when none did).
    * ``started_on_b`` are the requests whose first attempt went to B: A was not tried, which with
      this config only happens when A's breaker refused it (Ruling M2-R21).
    * ``committed_then_failed`` are the requests that got status 200 and whose last attempt did
      not end ``ok``: a stream that broke after the response was committed.
    """
    primary = [
        r
        for r in records
        if r.get("type", "decision") == "decision" and r.get("policy_role") != "shadow"
    ]
    per_request: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    by_deployment: dict[str, Counter[str]] = {}
    failed_reasons: Counter[str] = Counter()
    not_recovered: Counter[str] = Counter()
    started_on_a = started_on_b = 0
    eligible_on_a = first_failed = recovered = recovered_on_b = 0
    committed_then_failed = on_both = total_attempts = 0

    for record in primary:
        attempts = _attempts(record)
        status = int(record.get("status", 0))
        by_status[str(status)] += 1
        per_request[str(len(attempts))] += 1
        total_attempts += len(attempts)
        last_outcome = attempts[-1]["outcome"] if attempts else "no_attempt"
        answered = status == 200 and last_outcome == "ok"
        # A 200 whose last attempt did not end ok was committed before it broke: that attempt's
        # failure came after the first byte, so failover could not act on it.
        broke_after_commit = status == 200 and bool(attempts) and last_outcome != "ok"
        if broke_after_commit:
            committed_then_failed += 1
        retryable = [
            is_eligible_failure(att["outcome"])
            and not (broke_after_commit and i == len(attempts) - 1)
            for i, att in enumerate(attempts)
        ]
        for attempt, eligible in zip(attempts, retryable, strict=True):
            by_deployment.setdefault(attempt["deployment"], Counter())[attempt["outcome"]] += 1
            if attempt["deployment"] == a and eligible:
                eligible_on_a += 1
        deployments = {att["deployment"] for att in attempts}
        if a in deployments and b in deployments:
            on_both += 1
        if attempts and attempts[0]["deployment"] == a:
            started_on_a += 1
        elif attempts and attempts[0]["deployment"] == b:
            started_on_b += 1
        if retryable and retryable[0]:
            first_failed += 1
            if answered:
                recovered += 1
                if attempts[-1]["deployment"] == b:
                    recovered_on_b += 1
            else:
                not_recovered[f"{status}:{last_outcome}"] += 1
        if status != 200:
            failed_reasons[f"{status}:{last_outcome}"] += 1

    return {
        "requests": len(primary),
        "responses_by_status": dict(sorted(by_status.items())),
        "attempts": total_attempts,
        "attempts_per_request": dict(sorted(per_request.items())),
        "attempts_by_deployment": {d: dict(c) for d, c in sorted(by_deployment.items())},
        "started_on_a": started_on_a,
        "started_on_b": started_on_b,
        "eligible_failures_on_a": eligible_on_a,
        "first_attempt_failed_eligibly": first_failed,
        "recovered": recovered,
        "recovered_on_b": recovered_on_b,
        "recovery_rate": recovered / first_failed if first_failed else None,
        "not_recovered": dict(not_recovered),
        "failed_requests": sum(failed_reasons.values()),
        "failed_by_reason": dict(failed_reasons),
        "requests_with_attempts_on_both": on_both,
        "committed_then_failed": committed_then_failed,
    }


def read_decision_records(directory: Path) -> list[dict[str, Any]]:
    """Every decision record in the ``*.jsonl`` files of ``directory``, oldest file first.

    A rotated file (``decisions.<UTC time>.<sequence>.jsonl``) sorts before the active ``decisions.jsonl``.
    Lines of any other record type (feedback) are skipped. A line that is not JSON, such as the
    torn last line of a file still being written, raises ValueError naming the file.
    """
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as err:
                    raise ValueError(f"{path.name} line {number}: not JSON ({err})") from err
                if isinstance(record, dict) and record.get("type") == "decision":
                    records.append(record)
    return records


# --- the breaker and the counters, seen through /metrics ---------------------------------------


def parse_breaker_state(text: str, deployment: str) -> int | None:
    """The ``rp_breaker_state`` of a deployment (0 closed, 1 half-open, 2 open), None if absent."""
    prefix = f'rp_breaker_state{{deployment="{deployment}"}} '
    for line in text.splitlines():
        if line.startswith(prefix):
            return int(float(line[len(prefix) :]))
    return None


def parse_counter(text: str, name: str) -> float | None:
    """The value of an unlabeled counter, None when the scrape does not hold it."""
    prefix = f"{name} "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line[len(prefix) :])
    return None


def summarize_breaker(samples: Sequence[tuple[float, int]]) -> dict[str, Any]:
    """What polling a breaker's state saw: the polls, the trips and the time it was not closed.

    ``opens`` counts the changes from closed (or half-open) into open between two polls, which
    undercounts a trip shorter than the polling interval; a trip lasts at least 10 s, far more.
    """
    opens = 0
    previous: int | None = None
    for _, state in samples:
        if state == 2 and previous != 2:
            opens += 1
        previous = state
    return {
        "polls": len(samples),
        "opens": opens,
        "half_open_seen": any(state == 1 for _, state in samples),
        "polls_not_closed": sum(1 for _, state in samples if state != 0),
        "poll_interval_s": BREAKER_POLL_S,
    }


class BreakerWatcher:
    """Polls a gateway's ``/metrics`` for one deployment's breaker state in a background thread."""

    def __init__(self, port: int, deployment: str, interval: float = BREAKER_POLL_S) -> None:
        self._port = port
        self._deployment = deployment
        self._interval = interval
        self._stop = threading.Event()
        self.samples: list[tuple[float, int]] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                state = parse_breaker_state(overhead.scrape(self._port), self._deployment)
            except (urllib.error.URLError, OSError):
                state = None
            if state is not None:
                self.samples.append((time.time(), state))
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[tuple[float, int]]:
        self._stop.set()
        self._thread.join(10)
        return self.samples


# --- checks ------------------------------------------------------------------------------------


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _common_checks(
    scheduled: int | None,
    load: Mapping[str, Any],
    mocks: Mapping[str, Mapping[str, Any]],
    decision_log: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks = []
    if scheduled is not None:
        checks.append(
            _check(
                f"schedule: the generator scheduled exactly {scheduled} requests",
                load["scheduled"] == scheduled,
                f"scheduled {load['scheduled']}",
            )
        )
    checks.append(
        _check(
            "decision log: one record per scheduled request and none dropped",
            decision_log["records"] == load["scheduled"] and decision_log["dropped"] == 0,
            f"{decision_log['records']} records for {load['scheduled']} requests, "
            f"{decision_log['dropped']} dropped by the writer",
        )
    )
    checks.append(
        _check(
            "drops: the generator dropped no arrival",
            load["dropped"] == 0,
            f"dropped {load['dropped']}",
        )
    )
    return checks


def prestream_checks(
    scheduled: int | None,
    load: Mapping[str, Any],
    mocks: Mapping[str, Mapping[str, Any]],
    decision_log: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Integrity checks of the pre-stream arm: the independent counts must agree."""
    checks = _common_checks(scheduled, load, mocks, decision_log, classification)
    seen = classification["attempts_by_deployment"]
    on_a = sum(seen.get(DEPLOYMENT_A, {}).values())
    on_b = sum(seen.get(DEPLOYMENT_B, {}).values())
    a_seen, b_seen = mocks["a"]["requests"], mocks["b"]["requests"]
    checks.append(
        _check(
            "mock counts: each mock received the attempts the log shows for its deployment",
            a_seen == on_a and b_seen == on_b,
            f"mock A {a_seen} vs log {on_a}; mock B {b_seen} vs log {on_b}",
        )
    )
    checks.append(
        _check(
            "client errors: the generator's error count equals the requests that failed overall",
            load["errors"] == classification["failed_requests"],
            f"generator {load['errors']}, decision log {classification['failed_requests']}",
        )
    )
    return checks


def midstream_checks(
    scheduled: int | None,
    load: Mapping[str, Any],
    mocks: Mapping[str, Mapping[str, Any]],
    decision_log: Mapping[str, Any],
    classification: Mapping[str, Any],
    breaker: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Integrity checks of the mid-stream arm, including the rule: no second call after commit."""
    checks = _common_checks(scheduled, load, mocks, decision_log, classification)
    requests = classification["requests"]
    per = classification["attempts_per_request"]
    checks.append(
        _check(
            "one attempt: every request made exactly one attempt",
            per == {"1": requests},
            f"attempts per request {per}",
        )
    )
    checks.append(
        _check(
            "both: no request attempted both deployments",
            classification["requests_with_attempts_on_both"] == 0,
            f"{classification['requests_with_attempts_on_both']} requests",
        )
    )
    a_count, b_count = mocks["a"]["requests"], mocks["b"]["requests"]
    checks.append(
        _check(
            "mock counts: the mocks together received one request per client request, "
            "each the ones the log shows",
            a_count == classification["started_on_a"]
            and b_count == classification["started_on_b"]
            and a_count + b_count == decision_log["records"],
            f"mock A {a_count} vs log {classification['started_on_a']}; "
            f"mock B {b_count} vs log {classification['started_on_b']}; "
            f"sum {a_count + b_count} vs {decision_log['records']} records",
        )
    )
    cut = classification["committed_then_failed"]
    # A cut stream is neither completed (no end marker) nor canceled (the client did not leave):
    # the mock's own count of what it did is independent of the gateway's decision log.
    mock_cut = a_count - mocks["a"]["completed"] - mocks["a"]["canceled"]
    checks.append(
        _check(
            "cut count: the streams A's mock says it cut are the streams the log shows broken "
            "after the commit",
            mock_cut == cut,
            f"mock A: {a_count} received - {mocks['a']['completed']} completed - "
            f"{mocks['a']['canceled']} canceled = {mock_cut}; decision log: {cut}",
        )
    )
    on_a = classification["attempts_by_deployment"].get(DEPLOYMENT_A, {})
    failed_on_a = sum(n for outcome, n in on_a.items() if outcome != "ok")
    checks.append(
        _check(
            "committed: every failed attempt on A was a stream that broke after the commit",
            failed_on_a == cut,
            f"{failed_on_a} failed attempts on A, {cut} streams broken after the commit",
        )
    )
    checks.append(
        _check(
            "cuts: at least one stream was cut after the commit, so the rule was exercised",
            cut > 0,
            f"{cut} streams cut after the commit",
        )
    )
    checks.append(
        _check(
            "client errors: the generator's error count equals the streams that broke after commit",
            load["errors"] == cut,
            f"generator {load['errors']}, decision log {cut}",
        )
    )
    checks.append(
        _check(
            "breaker: B received requests only if A's breaker opened",
            b_count == 0 or breaker["opens"] >= 1,
            f"B received {b_count} requests, A's breaker opened {breaker['opens']} times",
        )
    )
    return checks


# --- configs and commands ----------------------------------------------------------------------


def failover_config(
    gateway_port: int,
    mock_a_port: int,
    mock_b_port: int,
    a_upstream_model: str,
    decision_log: Path,
    key: str,
) -> dict[str, Any]:
    """The gateway config: the template with ports, A's upstream model, log path and key hash."""
    text = TEMPLATE.read_text(encoding="utf-8")
    fills = {
        "{{GATEWAY_PORT}}": str(gateway_port),
        "{{MOCK_A_PORT}}": str(mock_a_port),
        "{{MOCK_B_PORT}}": str(mock_b_port),
        "{{A_UPSTREAM_MODEL}}": a_upstream_model,
        "{{DECISION_LOG}}": decision_log.as_posix(),
        "{{KEY_SHA256}}": overhead.key_sha256(key),
    }
    for token, value in fills.items():
        text = text.replace(token, value)
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"{TEMPLATE} is not a YAML mapping")
    return cfg


def key_sha256(key: str) -> str:
    return overhead.key_sha256(key)


def mock_args(arm: str, side: str) -> list[str]:
    """The ``rp-mock`` flags (after ``--listen``) for deployment ``side`` ('a' or 'b') of an arm.

    Only deployment A injects failures, through rates: the pre-stream mix in the first arm and
    the disconnect rate in the second. B is always clean.
    """
    args = ["--seed", str(MOCK_SEEDS[side]), *MOCK_DELAYS]
    if arm == "prestream" and side == "a":
        for flag, rate in FAIL_MIX:
            args += [flag, str(rate)]
    if arm == "midstream" and side == "a":
        args += ["--fail-rate-disconnect", str(DISCONNECT_RATE)]
    return args


# --- processes ---------------------------------------------------------------------------------


def mock_stats(port: int) -> dict[str, int]:
    """The mock's own request counters."""
    with overhead.OPENER.open(f"http://127.0.0.1:{port}/mock/stats", timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return {k: int(v) for k, v in data.items()}


def check_key_accepted(port: int) -> None:
    """Ask the gateway for its models with the key: it fails fast on a wrong key or config.

    This sends no chat request, so no mock sees it and no decision is logged.
    """
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": f"Bearer {KEY}"}
    )
    try:
        with overhead.OPENER.open(request, timeout=10) as resp:
            models = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"the gateway refused the test key: HTTP {err.code}") from err
    ids = {m.get("id") for m in models.get("data", [])}
    if MODEL not in ids:
        raise RuntimeError(f"the gateway does not list {MODEL}: {sorted(map(str, ids))}")


def run_rp_load(binary: Path, url: str, arm: Arm, duration: str) -> dict[str, Any]:
    """Run ``rp-load`` for the arm (no warmup: every request counts) and return its JSON result."""
    command = [
        *(str(binary), "--url", url, "--model", MODEL, "--rps", str(arm.rps)),
        *("--warmup", "0s", "--duration", duration, "--seed", str(arm.seed)),
        *("--max-inflight", str(MAX_INFLIGHT), "--max-tokens", str(MAX_TOKENS)),
    ]
    if arm.stream:
        command.append("--stream")
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "RP_LOAD_API_KEY": KEY},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"rp-load exited with {proc.returncode}: {proc.stderr.strip()}")
    result = json.loads(proc.stdout)
    if not isinstance(result, dict):
        raise RuntimeError("rp-load did not print a JSON object")
    return result


def wait_for_records(
    directory: Path, expected: int, timeout_s: float = LOG_WAIT_S
) -> list[dict[str, Any]]:
    """Read the decision log until it holds ``expected`` records, or ``timeout_s`` has passed.

    The writer queues records and flushes in the background, so the log trails the last response
    by a moment. A half-written last line is retried.
    """
    deadline = time.monotonic() + timeout_s
    records: list[dict[str, Any]] = []
    while True:
        try:
            records = read_decision_records(directory)
        except ValueError:
            if time.monotonic() >= deadline:
                raise
        else:
            if len(records) >= expected or time.monotonic() >= deadline:
                return records
        time.sleep(0.25)


# --- one arm -----------------------------------------------------------------------------------


def run_arm(
    arm: Arm,
    bins: Mapping[str, Path],
    scratch: Path,
    duration: str,
    expected_requests: int | None,
) -> dict[str, Any]:
    """Start two mocks and a gateway, run the arm's load, read everything back, stop it all."""
    began = time.monotonic()
    log_dir = scratch / f"{arm.name}-decisions"
    log_dir.mkdir(parents=True, exist_ok=True)
    mock_ports = {"a": overhead.free_port(), "b": overhead.free_port()}
    gateway_port = overhead.free_port()
    cfg = failover_config(
        gateway_port,
        mock_ports["a"],
        mock_ports["b"],
        A_UPSTREAM[arm.name],
        log_dir / RECORDS_FILE,
        KEY,
    )
    config_path = scratch / f"{arm.name}.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")

    with contextlib.ExitStack() as stack:
        for side in ("a", "b"):
            command = [
                str(bins["rp-mock"]),
                *("--listen", f"127.0.0.1:{mock_ports[side]}"),
                *mock_args(arm.name, side),
            ]
            mock = stack.enter_context(
                overhead.service(command, scratch / f"{arm.name}.mock-{side}.log")
            )
            overhead.wait_ready(f"http://127.0.0.1:{mock_ports[side]}/v1/models", mock)
        gateway = stack.enter_context(
            overhead.service(
                [str(bins["routepilot"]), "serve", "-config", str(config_path)],
                scratch / f"{arm.name}.gateway.log",
            )
        )
        overhead.wait_ready(f"http://127.0.0.1:{gateway_port}/healthz", gateway)
        check_key_accepted(gateway_port)

        before = {side: mock_stats(port) for side, port in mock_ports.items()}
        watcher = BreakerWatcher(gateway_port, DEPLOYMENT_A)
        watcher.start()
        try:
            load = run_rp_load(
                bins["rp-load"],
                f"http://127.0.0.1:{gateway_port}/v1/chat/completions",
                arm,
                duration,
            )
        finally:
            samples = watcher.stop()
        records = wait_for_records(log_dir, int(load["scheduled"]))
        after = {side: mock_stats(port) for side, port in mock_ports.items()}
        metrics = overhead.scrape(gateway_port)

    mocks = {
        side: {k: after[side][k] - before[side].get(k, 0) for k in after[side]} for side in after
    }
    classification = classify_attempts(records)
    log_info = {
        "records": len(records),
        "dropped": int(parse_counter(metrics, "rp_decision_log_dropped_total") or 0),
    }
    breaker = summarize_breaker(samples)
    if arm.name == "prestream":
        checks = prestream_checks(expected_requests, load, mocks, log_info, classification)
    else:
        checks = midstream_checks(expected_requests, load, mocks, log_info, classification, breaker)
    return {
        "arm": arm.name,
        "rps": arm.rps,
        "duration": duration,
        "stream": arm.stream,
        "scheduled": expected_requests,
        "load": load,
        "mocks": mocks,
        "decision_log": log_info,
        "classification": classification,
        "breaker": breaker,
        "checks": checks,
        "wall_time_s": round(time.monotonic() - began, 1),
    }


# --- report ------------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.2f}%"


def _verdict(rate: float | None) -> str:
    if rate is None:
        return "no eligible failure, nothing to recover"
    return "met" if rate >= TARGET_RECOVERY else "not met"


def _counts(table: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in table.items()) or "none"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _f(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _lat(load: Mapping[str, Any]) -> list[str]:
    data = load.get("latency_ms")
    return [_f(data[k]) if data else "-" for k in ("p50", "p90", "p95", "p99", "max")]


def _checks_section(arm: Mapping[str, Any]) -> list[str]:
    rows = [[c["name"], "pass" if c["passed"] else "FAIL", c["detail"]] for c in arm["checks"]]
    return ["Integrity checks:", "", _table(["check", "result", "detail"], rows), ""]


def _prestream_section(arm: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    c = arm["classification"]
    load = arm["load"]
    breaker = arm["breaker"]
    total = c["requests"]
    mix = settings["mix"]
    straight = c["started_on_b"]
    out = [
        "## Pre-stream arm: failures injected on deployment A",
        "",
        f"{total} non-streaming requests at {settings['rps']} requests per second (open loop, "
        f"Poisson arrivals, seeded). Deployment A injects, per request: "
        f"{mix['http500'] * 100:g}% HTTP 500, {mix['http429'] * 100:g}% HTTP 429 with Retry-After "
        f"1 s and {mix['reset'] * 100:g}% connection resets "
        f"({(mix['http500'] + mix['http429'] + mix['reset']) * 100:g}% in all); deployment B "
        f"injects nothing. max_attempts is {settings['max_attempts']}.",
        "",
        "Target: at least 99% of the requests whose first attempt failed with an eligible failure "
        "are recovered. It is a target, not a guarantee; this is one run.",
        "",
        _table(
            ["quantity", "value"],
            [
                ["requests", str(total)],
                ["requests started on A", str(c["started_on_a"])],
                [
                    "eligible failures on A (attempts)",
                    f"{c['eligible_failures_on_a']} ({_counts(_failure_outcomes(c))})",
                ],
                [
                    "requests whose first attempt failed eligibly",
                    str(c["first_attempt_failed_eligibly"]),
                ],
                [
                    "of those, recovered (answered, on B)",
                    f"{c['recovered']} ({c['recovered_on_b']} on B)",
                ],
                [
                    "per-request recovery rate (recovered / first attempt failed eligibly)",
                    _pct(c["recovery_rate"]),
                ],
                [f"target >= {TARGET_RECOVERY * 100:g}%", _verdict(c["recovery_rate"])],
                [
                    "requests routed straight to B (A's breaker refused them, Ruling M2-R21)",
                    f"{straight} ({_pct(straight / total if total else None)} of requests)",
                ],
                ["A's breaker opened (polled every 100 ms)", f"{breaker['opens']} times"],
                [
                    "requests that failed overall",
                    f"{c['failed_requests']} ({_counts(c['failed_by_reason'])})",
                ],
                [
                    "of the failed-first requests, not recovered",
                    f"{sum(c['not_recovered'].values())} ({_counts(c['not_recovered'])})",
                ],
                ["client-side errors (load generator)", str(load["errors"])],
            ],
        ),
        "",
        "The recovery rate counts only the requests that tried A first and failed there. A request "
        "the breaker sent straight to B never failed on A, so it is not in the numerator or the "
        "denominator; the line above says how many there were, because the breaker is the "
        "mechanism that keeps the failing deployment out of the path.",
        "",
        "Attempts per deployment and outcome (decision log) and the mocks' own request counts:",
        "",
        _table(
            [
                "deployment",
                "attempts in the decision log",
                "outcomes",
                "requests the mock received",
            ],
            [
                [
                    name,
                    str(sum(c["attempts_by_deployment"].get(dep, {}).values())),
                    _counts(c["attempts_by_deployment"].get(dep, {})),
                    str(arm["mocks"][side]["requests"]),
                ]
                for name, dep, side in (("A", DEPLOYMENT_A, "a"), ("B", DEPLOYMENT_B, "b"))
            ],
        ),
        "",
        f"Responses by status: {_counts(c['responses_by_status'])}. Attempts per request: "
        f"{_counts(c['attempts_per_request'])}.",
        "",
    ]
    return out + _checks_section(arm)


def _failure_outcomes(classification: Mapping[str, Any]) -> dict[str, int]:
    on_a = classification["attempts_by_deployment"].get(DEPLOYMENT_A, {})
    return {k: v for k, v in on_a.items() if is_eligible_failure(k)}


def _midstream_section(arm: Mapping[str, Any], settings: Mapping[str, Any]) -> list[str]:
    c = arm["classification"]
    load = arm["load"]
    breaker = arm["breaker"]
    mock_a = arm["mocks"]["a"]
    cut = c["committed_then_failed"]
    share = f"{cut / c['started_on_a'] * 100:.1f}%" if c["started_on_a"] else "-"
    rate = settings["fail_rate_disconnect"]
    return [
        "## Mid-stream arm: deployment A cuts a share of its streams after the commit",
        "",
        f"{c['requests']} streaming requests at {settings['rps']} requests per second. Deployment "
        f"A's mock runs with --fail-rate-disconnect {rate}: for that share of the streams it "
        "sends the role chunk and the first content chunk, then drops the connection; the rest "
        "it answers in full. The gateway commits a response at the first chunk with content, "
        "so every cut stream was already committed when it broke. Deployment B is clean.",
        "",
        "Rule under test: after the first byte the gateway makes no second upstream call. "
        "While A's breaker is closed every request starts on A; if A's breaker opens (five "
        "failures in a row), later requests start on B, which is allowed (they never touched A).",
        "",
        _table(
            ["quantity", "value"],
            [
                ["requests", str(c["requests"])],
                ["attempts in the decision log", str(c["attempts"])],
                ["attempts per request", _counts(c["attempts_per_request"])],
                ["requests started on A", str(c["started_on_a"])],
                [
                    "streams cut after the commit (decision log)",
                    f"{cut} ({share} of the requests started on A)",
                ],
                [
                    "streams A's mock cut (received - completed - canceled)",
                    str(mock_a["requests"] - mock_a["completed"] - mock_a["canceled"]),
                ],
                ["requests started on B (A's breaker refused them)", str(c["started_on_b"])],
                [
                    "requests with attempts on both deployments",
                    str(c["requests_with_attempts_on_both"]),
                ],
                [
                    "requests the mocks received (A, B, sum)",
                    f"{arm['mocks']['a']['requests']}, {arm['mocks']['b']['requests']}, "
                    f"{arm['mocks']['a']['requests'] + arm['mocks']['b']['requests']}",
                ],
                ["A's breaker opened (polled every 100 ms)", f"{breaker['opens']} times"],
                ["client-side errors (stream ended without the end marker)", str(load["errors"])],
            ],
        ),
        "",
        *_checks_section(arm),
    ]


def render_report(
    results: Mapping[str, Any],
    machine: Mapping[str, Any],
    run_id: str,
    go_version: str | None,
) -> str:
    """The report.md text for a results dict (the content of results.json)."""
    go = f", {go_version}" if go_version else ""
    settings = results["settings"]
    out = [
        f"# Failover against injected failures ({run_id})",
        "",
        "Labels: real local measurement, nothing simulated. Machine: "
        f"{machine['cpu']}, {machine['ram_gb']} GB RAM, {machine['os']}{go}. The load generator "
        "(rp-load), the gateway (routepilot serve) and the two mocks (rp-mock) share one machine "
        "and talk over loopback. The mocks' delays are configured constants (time to first token "
        "20 ms, 1 ms per output token, 16 output tokens). The experiment measures counts, not "
        "time; the latency columns are the client's view and carry the Retry-After waits.",
        "",
    ]
    arms = results["arms"]
    rows = []
    if "prestream" in arms:
        out += _prestream_section(
            arms["prestream"], settings["prestream"] | {"max_attempts": settings["max_attempts"]}
        )
        rows.append(("pre-stream", arms["prestream"]))
    if "midstream" in arms:
        out += _midstream_section(arms["midstream"], settings["midstream"])
        rows.append(("mid-stream", arms["midstream"]))
    out += [
        "## Client-side latency (ms, real measurement)",
        "",
        _table(
            ["arm", "completed", "errors", "p50", "p90", "p95", "p99", "max"],
            [
                [name, str(a["load"]["completed"]), str(a["load"]["errors"]), *_lat(a["load"])]
                for name, a in rows
            ],
        ),
        "",
        "Wall time: "
        + ", ".join(f"{name} {a['wall_time_s']} s" for name, a in rows)
        + f"; the whole run {results['wall_time_s']} s.",
        "",
        "## Limitations",
        "",
        "- One run per arm, no repetitions: the failures are drawn from a seeded random number "
        "generator, but the order in which concurrent requests reach it varies, so another run "
        "has the same rates and different individual failures.",
        "- The mocks are deterministic and fast; real providers fail in correlated ways (a whole "
        "region, a deploy) that independent per-request failures do not model.",
        "- Deployment B never fails here, so recovery can only fail through the retry limit, the "
        "deadline or a breaker on B; it says nothing about the case where both deployments fail.",
        "- Windows loopback on one shared machine. The numbers are counts and do not depend on "
        "the timer resolution.",
        "",
    ]
    return "\n".join(out)


# --- main --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Failover against injected failures.")
    parser.add_argument("--arms", default="prestream,midstream", help="comma-separated arms to run")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run short arms (10 s and 8 s) and skip the exact-schedule check; not for results",
    )
    parser.add_argument("--out", type=Path, help="result directory (default: under bench/results)")
    parser.add_argument("--keep-scratch", action="store_true", help="keep logs and configs")
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=ROOT / "bench" / "scratch",
        help="where the binaries, configs, decision logs and process logs go",
    )
    parser.add_argument("--render", type=Path, metavar="DIR", help="rewrite DIR/report.md and exit")
    return parser


def _rewrite_report(directory: Path) -> int:
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    info = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    text = render_report(results, info["machine"], info["run_id"], info["go_version"])
    (directory / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"failover: wrote {(directory / 'report.md').as_posix()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.render:
        return _rewrite_report(args.render)
    names = [n for n in args.arms.split(",") if n]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        print(f"failover: unknown arm {unknown[0]!r} (want {', '.join(ARMS)})", file=sys.stderr)
        return 2

    started = datetime.now(UTC)
    clock = time.monotonic()
    run_id = runinfo.new_run_id(started)
    out_dir = args.out or ROOT / "bench" / "results" / EXPERIMENT / run_id
    scratch = args.scratch_root / f"{EXPERIMENT}-{run_id}"
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"failover: building binaries ({run_id})", flush=True)
    bins = overhead.build_binaries(scratch / "bin")
    arms: dict[str, Any] = {}
    all_passed = True
    try:
        for name in names:
            arm = ARMS[name]
            duration = SMOKE_DURATIONS[name] if args.smoke else arm.duration
            expected = None if args.smoke else arm.requests
            print(f"failover: running the {name} arm ({duration} at {arm.rps} rps)", flush=True)
            result = run_arm(arm, bins, scratch, duration, expected)
            arms[name] = result
            failed = [c["name"] for c in result["checks"] if not c["passed"]]
            all_passed = all_passed and not failed
            c = result["classification"]
            print(
                f"failover: {name}: requests={c['requests']} recovery={_pct(c['recovery_rate'])} "
                f"started_on_b={c['started_on_b']} failed={c['failed_requests']} "
                f"breaker_opens={result['breaker']['opens']} checks_failed={len(failed)} "
                f"({result['wall_time_s']} s)",
                flush=True,
            )
            for check in failed:
                print(f"failover: CHECK FAILED: {check}", file=sys.stderr, flush=True)
            time.sleep(overhead.COOLDOWN_S)
    finally:
        if not args.keep_scratch and not arms:
            shutil.rmtree(scratch, ignore_errors=True)

    results: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "smoke": bool(args.smoke),
        "target_recovery": TARGET_RECOVERY,
        "settings": {
            "max_attempts": MAX_ATTEMPTS,
            "mock_delays": list(MOCK_DELAYS),
            "prestream": {
                "rps": ARMS["prestream"].rps,
                "requests": ARMS["prestream"].requests,
                "mix": {
                    "http500": FAIL_MIX[0][1],
                    "http429": FAIL_MIX[1][1],
                    "reset": FAIL_MIX[2][1],
                },
                "a_upstream_model": A_UPSTREAM["prestream"],
            },
            "midstream": {
                "rps": ARMS["midstream"].rps,
                "requests": ARMS["midstream"].requests,
                "fail_rate_disconnect": DISCONNECT_RATE,
                "a_upstream_model": A_UPSTREAM["midstream"],
            },
        },
        "arms": arms,
        "all_checks_passed": all_passed,
        "wall_time_s": round(time.monotonic() - clock, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    info = runinfo.write_run_json(
        out_dir,
        EXPERIMENT,
        {
            "loadgen_prestream": ARMS["prestream"].seed,
            "loadgen_midstream": ARMS["midstream"].seed,
            "mock_a": MOCK_SEEDS["a"],
            "mock_b": MOCK_SEEDS["b"],
        },
        ["python", "bench/scripts/failover.py", *(argv if argv is not None else sys.argv[1:])],
        None,
        started,
    )
    report = render_report(results, info["machine"], info["run_id"], info["go_version"])
    (out_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"failover: wrote {out_dir.as_posix()} in {results['wall_time_s']} s", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
