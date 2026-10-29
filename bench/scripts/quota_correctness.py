"""Quota correctness: 64 concurrent clients against one tenant budget, on each quota backend.

Run from the repository root with the repo venv (Go must be on PATH; Docker for real Redis):

    python bench/scripts/quota_correctness.py

One tenant has a total budget of 0.05 USD (bench/configs/quota.yaml). 64 client threads, each with
a seeded random ``max_tokens`` in [8, 512] per request, send chat requests through the gateway
until the gateway has answered that thread 20 times in a row with 429 ``insufficient_quota``. The
gateway reserves each request's worst case before it calls the provider, so 64 clients race for
the same budget and most requests near the end are refused. The mock caps its answer at each
request's ``max_tokens`` (it is started with a default length of 600 tokens, above every
``max_tokens`` a client sends, and the model carries no ``tokens=`` directive), so a request
spends what was reserved for its output and the budget is really contended.

The experiment runs against the memory backend and against Redis: a real Redis 7.4 started with
Docker on a loopback port (``docker run --rm -d --name rp-quota-redis``, stopped in a ``finally``),
or the Redis at ``ROUTEPILOT_TEST_REDIS_URL`` when that variable is set, or, when neither is
available, the gateway test that runs the same race against miniredis (labeled as such).

Per backend the checks are:

* the sum of ``cost_actual`` in the usage ledger (SQLite) is at most the budget;
* ``rp_budget_overshoot_picousd_total`` on ``/metrics`` is 0;
* ``routepilot reconcile`` agrees with the ledger: exit 0 and no difference for Redis; for the
  memory backend it exits 2 by design, because the in-process counter cannot be read from another
  process, so the ledger sum against the limit and the overshoot counter are what is reported;
* the ledger has one row per admitted request, the mock received exactly the admitted requests,
  nothing else answered, no rate limit fired, and the ledger sum equals the cost computed from the
  token counts the clients saw (an independent recomputation).

It also reports the requests admitted and rejected and the unused budget at the end.

Writes bench/results/quota_correctness/<run-id>/{results.json,report.md,run.json}. Output holds
numbers only; the prompt is a fixed invented sentence and the key is a fake test key. Every
process binds loopback only and is stopped in a ``finally``, also on an error.
``--render DIR`` rewrites report.md of an earlier run from its results.json and run.json.
"""

import argparse
import contextlib
import http.client
import json
import os
import random
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from routepilot_train import runinfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overhead  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "quota_correctness"
TEMPLATE = ROOT / "bench" / "configs" / "quota.yaml"

KEY = "test-key-quota"  # a fake key; its SHA-256 is computed at run time
MODEL = "quota-model"
CLIENTS = 64
SEED = 1
MAX_TOKENS_RANGE = (8, 512)
STOP_AFTER = 20  # consecutive 429 insufficient_quota answers that end a client
MAX_REQUESTS_PER_CLIENT = 20_000  # a backstop so a bug cannot keep a client running forever
BUDGET_USD = "0.05"
PRICE_IN_PER_MTOK = "1"
PRICE_OUT_PER_MTOK = "2"
PRICE_IN_PICO = 1_000_000  # picodollars per input token at 1 USD per million tokens
PRICE_OUT_PICO = 2_000_000
UPSTREAM_MODEL = "bench-model"  # no mock directive: the answer is capped by the request
MOCK_TOKENS = 600  # the mock's default answer length, above every max_tokens a client sends
PROMPT = (
    "Summarize in one sentence why regular backups protect a small online shop "
    "from losing its order data."
)
# The gateway's upper bound on prompt tokens: the bytes of the text, 8 per message and 16 fixed
# (internal/catalog/requirements.go).
PROMPT_BOUND_TOKENS = len(PROMPT.encode("utf-8")) + 8 + 16
REDIS_ENV = "ROUTEPILOT_TEST_REDIS_URL"
REDIS_IMAGE = "redis:7.4-alpine"
REDIS_CONTAINER = "rp-quota-redis"
BACKENDS = ("memory", "redis")
MOCK_ARGS = (
    *("--seed", "1"),
    *("--ttft-median-ms", "5", "--ttft-sigma", "0"),
    *("--tpot-median-ms", "0", "--tpot-sigma", "0"),
    *("--tokens", str(MOCK_TOKENS)),
)
MINIREDIS_TEST = "TestBudgetNeverOvershootsUnderConcurrency/miniredis"
LEDGER_WAIT_S = 10.0  # how long to wait for the gateway's background writer to catch up


# --- money -------------------------------------------------------------------------------------


def usd_to_pico(usd: str) -> int:
    """An exact decimal USD amount as integer picodollars (1e-12 USD)."""
    pico = Decimal(usd) * 10**12
    if pico != pico.to_integral_value():
        raise ValueError(f"{usd} USD has more than twelve fractional digits")
    return int(pico)


def pico_to_usd(pico: int) -> str:
    """Integer picodollars as a decimal USD string without trailing zeros."""
    text = f"{Decimal(pico) / 10**12:.12f}".rstrip("0").rstrip(".")
    return text or "0"


# --- the clients -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reply:
    """What a client keeps of one response: the status, the error code and the reported usage."""

    status: int
    code: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Tally:
    """Counts of one client's requests, or of all clients' when merged."""

    admitted: int = 0
    rejected: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    other: dict[str, int] = field(default_factory=dict)
    stopped: str = ""

    @property
    def requests(self) -> int:
        return self.admitted + self.rejected + sum(self.other.values())


def max_tokens_stream(seed: int, client: int) -> Iterator[int]:
    """An endless, seeded stream of ``max_tokens`` values in the briefed range for one client.

    Each client has its own generator, so the run does not touch the global ``random`` state.
    """
    rng = random.Random(seed * 1_000_003 + client)
    low, high = MAX_TOKENS_RANGE
    while True:
        yield rng.randint(low, high)


def run_client(
    send: Callable[[int], Reply],
    max_tokens: Iterator[int],
    stop_after: int,
    max_requests: int,
) -> Tally:
    """Send requests until ``stop_after`` 429 ``insufficient_quota`` answers come in a row.

    An admitted request (200) resets the count, and so does any other answer. A response that is
    neither a 200 nor that rejection is counted under ``other`` (``<status>`` or
    ``<status>:<code>``), never as a rejection. ``max_requests`` is a backstop.
    """
    tally = Tally()
    in_a_row = 0
    while tally.requests < max_requests:
        reply = send(next(max_tokens))
        if reply.status == 200:
            tally.admitted += 1
            tally.prompt_tokens += reply.prompt_tokens
            tally.completion_tokens += reply.completion_tokens
            in_a_row = 0
        elif reply.status == 429 and reply.code == "insufficient_quota":
            tally.rejected += 1
            in_a_row += 1
            if in_a_row >= stop_after:
                tally.stopped = "rejected_in_a_row"
                return tally
        else:
            key = str(reply.status) + (f":{reply.code}" if reply.code else "")
            tally.other[key] = tally.other.get(key, 0) + 1
            in_a_row = 0
    tally.stopped = "max_requests"
    return tally


def merge_tallies(tallies: Sequence[Tally]) -> Tally:
    merged = Tally()
    other: Counter[str] = Counter()
    for t in tallies:
        merged.admitted += t.admitted
        merged.rejected += t.rejected
        merged.prompt_tokens += t.prompt_tokens
        merged.completion_tokens += t.completion_tokens
        other.update(t.other)
    merged.other = dict(other)
    return merged


def make_sender(port: int) -> Callable[[int], Reply]:
    """A function that sends one chat request with the given ``max_tokens`` on its own connection.

    Each client thread builds its own sender, so the keep-alive connection is never shared.
    """
    holder: dict[str, http.client.HTTPConnection] = {}

    def connection() -> http.client.HTTPConnection:
        if "c" not in holder:
            holder["c"] = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        return holder["c"]

    def send(max_tokens: int) -> Reply:
        body = json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": max_tokens,
            }
        )
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
        for attempt in (1, 2):
            try:
                conn = connection()
                conn.request("POST", "/v1/chat/completions", body, headers)
                resp = conn.getresponse()
                data = resp.read()
                break
            except (OSError, http.client.HTTPException):
                holder.pop("c", None)  # a dropped keep-alive connection: reconnect once
                if attempt == 2:
                    return Reply(status=0, code="connection")
        try:
            doc = json.loads(data)
        except json.JSONDecodeError:
            return Reply(status=resp.status, code="not_json")
        if resp.status == 200:
            usage = doc.get("usage") or {}
            return Reply(
                200,
                "",
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
            )
        error = doc.get("error") or {}
        return Reply(status=resp.status, code=str(error.get("code") or ""))

    return send


# --- reading the results back ------------------------------------------------------------------


def parse_samples(text: str, name: str) -> list[tuple[dict[str, str], float]]:
    """The samples of one metric in a Prometheus text scrape: (labels, value) in scrape order."""
    out: list[tuple[dict[str, str], float]] = []
    for line in text.splitlines():
        if not line.startswith(name):
            continue
        rest = line[len(name) :]
        labels: dict[str, str] = {}
        if rest.startswith("{"):
            end = rest.index("}")
            for part in rest[1:end].split(","):
                key, _, value = part.partition("=")
                labels[key] = value.strip('"')
            rest = rest[end + 1 :]
        elif not rest.startswith(" "):
            continue  # another metric whose name starts with this one
        out.append((labels, float(rest.split()[0])))
    return out


def ledger_summary(db_path: Path, tenant: str) -> dict[str, Any]:
    """Totals of a tenant's rows in the usage ledger (read-only; the gateway may still hold it)."""
    uri = f"file:{db_path.as_posix()}?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=30)) as db:
        row = db.execute(
            "SELECT COUNT(*), COALESCE(SUM(status = 'ok'), 0), COALESCE(SUM(cost_actual), 0), "
            "MIN(cost_estimate), MAX(cost_estimate), COALESCE(SUM(prompt_tokens), 0), "
            "COALESCE(SUM(completion_tokens), 0) FROM usage WHERE tenant = ?",
            (tenant,),
        ).fetchone()
    return {
        "rows": int(row[0]),
        "ok_rows": int(row[1]),
        "sum_pico": int(row[2]),
        "estimate_min": row[3],
        "estimate_max": row[4],
        "prompt_tokens": int(row[5]),
        "completion_tokens": int(row[6]),
    }


def wait_for_ledger_rows(
    db_path: Path,
    tenant: str,
    expected: int,
    timeout_s: float = LEDGER_WAIT_S,
    poll_s: float = 0.05,
) -> int:
    """The tenant's row count in the ledger once it reaches ``expected``, else after ``timeout_s``.

    The gateway queues a request's rows and a background writer puts them in the ledger in
    batches, so the last rows arrive a moment after the last response. Waiting for them does not
    hide a loss: when rows are really missing the count stays below ``expected``, the wait ends at
    the timeout, and the caller reads that lower count.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        rows = ledger_summary(db_path, tenant)["rows"]
        if rows >= expected or time.monotonic() >= deadline:
            return rows
        time.sleep(poll_s)


def parse_ledger_failures(log_text: str) -> dict[str, Any]:
    """What the gateway's log says about usage ledger rows it could not write.

    The gateway logs one JSON warning per batch it could not write (``usage ledger write
    failed``, with ``rows_lost`` and the ``error`` text) and, at most once a second, one when its
    queue was full (``usage ledger queue full, rows dropped``, with ``rows_dropped``). The result
    counts those lines, sums the rows, and groups them by the last part of the error text (the
    cause, such as ``context deadline exceeded``; ``queue full`` for the second kind). Lines that
    are not JSON are skipped.
    """
    lines = rows = 0
    errors: Counter[str] = Counter()
    for raw in log_text.splitlines():
        if "usage ledger " not in raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("msg") == "usage ledger write failed":
            lines += 1
            rows += int(entry.get("rows_lost", 0))
            errors[str(entry.get("error", "")).rsplit(": ", 1)[-1]] += 1
        elif entry.get("msg") == "usage ledger queue full, rows dropped":
            lines += 1
            rows += int(entry.get("rows_dropped", 0))
            errors["queue full"] += 1
    return {"log_lines": lines, "rows_lost": rows, "errors": dict(errors)}


def parse_reconcile_output(text: str) -> list[dict[str, str]]:
    """The lines ``routepilot reconcile`` prints: ``tenant ledger_usd counter_usd diff_usd``."""
    names = ("tenant", "ledger_usd", "counter_usd", "diff_usd")
    lines = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != len(names):
            raise ValueError(f"reconcile printed a line of {len(fields)} fields, want 4: {raw!r}")
        lines.append(dict(zip(names, fields, strict=True)))
    return lines


# --- the gateway config ------------------------------------------------------------------------


def key_sha256(key: str) -> str:
    return overhead.key_sha256(key)


def quota_config(
    gateway_port: int,
    mock_port: int,
    key: str,
    tenant: str,
    backend: str,
    sqlite_path: Path,
) -> dict[str, Any]:
    """The gateway config: the template with ports, tenant, key hash, ledger path and backend."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    text = TEMPLATE.read_text(encoding="utf-8")
    fills = {
        "{{GATEWAY_PORT}}": str(gateway_port),
        "{{MOCK_PORT}}": str(mock_port),
        "{{TENANT}}": tenant,
        "{{KEY_SHA256}}": overhead.key_sha256(key),
        "{{SQLITE_PATH}}": sqlite_path.as_posix(),
    }
    for token, value in fills.items():
        text = text.replace(token, value)
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"{TEMPLATE} is not a YAML mapping")
    cfg["quota"] = (
        {"backend": "redis", "redis_url_env": REDIS_ENV}
        if backend == "redis"
        else {"backend": "memory"}
    )
    return cfg


# --- checks ------------------------------------------------------------------------------------


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def quota_checks(
    backend: str,
    limit_pico: int,
    clients: Mapping[str, Any],
    ledger: Mapping[str, Any],
    metrics: Mapping[str, Any],
    mock_requests: int,
    reconcile: Mapping[str, Any],
    expected_cost_pico: int,
) -> list[dict[str, Any]]:
    """The correctness checks of one backend (see the module docstring)."""
    spent = pico_to_usd(ledger["sum_pico"])
    checks = [
        _check(
            "ledger: the committed spend (sum of cost_actual) is within the limit",
            ledger["sum_pico"] <= limit_pico,
            f"{spent} USD against a limit of {pico_to_usd(limit_pico)} USD",
        ),
        _check(
            "spend: the spend recomputed from the tokens the clients saw is within the limit",
            expected_cost_pico <= limit_pico,
            f"{pico_to_usd(expected_cost_pico)} USD against a limit of "
            f"{pico_to_usd(limit_pico)} USD",
        ),
        _check(
            "overshoot: rp_budget_overshoot_picousd_total is 0",
            metrics["overshoot_pico"] == 0,
            f"{metrics['overshoot_pico']} picodollars",
        ),
    ]
    if backend == "redis":
        lines = reconcile["lines"]
        counters = [usd_to_pico(x["counter_usd"]) for x in lines]
        checks.append(
            _check(
                "counter: the Redis budget counter equals the spend recomputed from the tokens "
                "the clients saw",
                bool(counters) and all(c == expected_cost_pico for c in counters),
                f"counter {[pico_to_usd(c) for c in counters]} USD, "
                f"recomputed {pico_to_usd(expected_cost_pico)} USD",
            )
        )
        agrees = (
            reconcile["exit_code"] == 0
            and bool(lines)
            and all(
                line["ledger_usd"] == line["counter_usd"] and usd_to_pico(line["diff_usd"]) == 0
                for line in lines
            )
        )
        detail = "; ".join(
            f"{x['tenant']} ledger {x['ledger_usd']} counter {x['counter_usd']} "
            f"diff {x['diff_usd']}"
            for x in lines
        )
        checks.append(
            _check(
                "reconcile: the usage ledger and the Redis budget counter agree (exit 0)",
                agrees,
                f"exit {reconcile['exit_code']}: {detail}",
            )
        )
    else:
        checks.append(
            _check(
                "reconcile: refuses the memory backend by design (exit 2)",
                reconcile["exit_code"] == 2,
                f"exit {reconcile['exit_code']}",
            )
        )
    admitted = clients["admitted"]
    checks += [
        _check(
            "unexpected responses: every answer was a 200 or a 429 insufficient_quota",
            not clients["other"],
            f"other responses: {clients['other'] or 'none'}",
        ),
        _check(
            "rows: the ledger has one row per admitted request",
            ledger["rows"] == admitted and ledger["ok_rows"] == admitted,
            f"{ledger['rows']} rows ({ledger['ok_rows']} ok) for {admitted} admitted requests",
        ),
        _check(
            "mock: the provider received exactly the admitted requests",
            mock_requests == admitted,
            f"mock received {mock_requests}, admitted {admitted}",
        ),
        _check(
            "rate limit: no RPM, TPM or concurrency limit fired",
            sum(v for k, v in metrics["ratelimit"].items() if k != "upstream") == 0,
            f"rejections by kind: {metrics['ratelimit']}",
        ),
        _check(
            "cost: the ledger sum equals the cost recomputed from the tokens the clients saw",
            ledger["sum_pico"] == expected_cost_pico,
            f"ledger {ledger['sum_pico']}, recomputed {expected_cost_pico} picodollars",
        ),
    ]
    return checks


# --- Redis -------------------------------------------------------------------------------------


def docker_run(command: list[str]) -> str:
    """Run a docker command and return its stdout; a failure raises."""
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(command[:3])} failed: {proc.stderr.strip()}")
    return proc.stdout


def wait_redis(port: int, timeout_s: float = 30.0) -> None:
    """Wait until a Redis on a loopback port answers PING."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1) as sock:
                sock.sendall(b"PING\r\n")
                if sock.recv(16).startswith(b"+PONG"):
                    return
        except OSError:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"Redis on 127.0.0.1:{port} did not answer PING within {timeout_s:g} s")


@contextlib.contextmanager
def redis_container(
    docker: Callable[[list[str]], str],
    port: int,
    wait: Callable[[int], None] = wait_redis,
) -> Iterator[str]:
    """A throwaway Redis 7.4 on a loopback port; yields its URL and always stops it."""
    docker(
        [
            *("docker", "run", "--rm", "-d", "--name", REDIS_CONTAINER),
            *("-p", f"127.0.0.1:{port}:6379", REDIS_IMAGE),
        ]
    )
    try:
        wait(port)
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        try:
            docker(["docker", "stop", REDIS_CONTAINER])
        except RuntimeError as err:  # do not hide the error the block may be raising
            print(f"quota_correctness: could not stop {REDIS_CONTAINER}: {err}", file=sys.stderr)


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and bool(proc.stdout.strip())


@contextlib.contextmanager
def environment(name: str, value: str | None) -> Iterator[None]:
    """Set (or clear) an environment variable of this process for the duration of the block."""
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def miniredis_go_test(run: Callable[[list[str]], tuple[int, str]]) -> dict[str, Any]:
    """The fallback when no real Redis is available: the gateway test that runs the same race.

    It drives the gateway handler in the Go test process against miniredis (an in-process Redis
    stand-in), not the ``routepilot`` binary, and says so. ``run`` returns (exit code, output).
    """
    command = ["go", "test", "-count=1", "-v", "-run", MINIREDIS_TEST, "./internal/gateway/"]
    code, output = run(command)
    tail = [ln for ln in output.splitlines() if ln.strip()][-12:]
    return {
        "backend": "redis",
        "redis_mode": "miniredis-go-test",
        "go_test": {"command": " ".join(command), "exit_code": code, "tail": tail},
        "checks": [
            _check(
                "the miniredis gateway test passes (64 concurrent reservers, no overshoot)",
                code == 0,
                f"exit {code}",
            )
        ],
    }


# --- one backend -------------------------------------------------------------------------------


def mock_request_count(port: int) -> int:
    """The number of chat requests the mock itself has received."""
    with overhead.OPENER.open(f"http://127.0.0.1:{port}/mock/stats", timeout=10) as resp:
        return int(json.loads(resp.read().decode("utf-8"))["requests"])


def run_reconcile(binary: Path, config_path: Path) -> dict[str, Any]:
    """Run ``routepilot reconcile`` on the config and keep its exit code and output."""
    proc = subprocess.run(
        [str(binary), "reconcile", "--config", str(config_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    try:
        lines = parse_reconcile_output(proc.stdout)
    except ValueError:
        lines = []
    return {
        "exit_code": proc.returncode,
        "lines": lines,
        "stdout": proc.stdout,
        "stderr": proc.stderr.strip(),
    }


def run_backend(
    backend: str,
    redis_url: str | None,
    bins: Mapping[str, Path],
    scratch: Path,
    run_suffix: str,
    clients: int = CLIENTS,
) -> dict[str, Any]:
    """Start a mock and a gateway on the backend, run the clients, read everything back."""
    began = time.monotonic()
    # A Redis that outlives this run keeps its budget counters, so its tenant id is unique.
    tenant = f"quota-redis-{run_suffix}" if backend == "redis" else "quota-memory"
    work = scratch / backend
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "usage.db"
    mock_port, gateway_port = overhead.free_port(), overhead.free_port()
    cfg = quota_config(gateway_port, mock_port, KEY, tenant, backend, db_path)
    config_path = work / "gateway.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")

    env_value = redis_url if backend == "redis" else None
    with environment(REDIS_ENV, env_value), contextlib.ExitStack() as stack:
        mock = stack.enter_context(
            overhead.service(
                [str(bins["rp-mock"]), "--listen", f"127.0.0.1:{mock_port}", *MOCK_ARGS],
                work / "mock.log",
            )
        )
        overhead.wait_ready(f"http://127.0.0.1:{mock_port}/v1/models", mock)
        gateway = stack.enter_context(
            overhead.service(
                [str(bins["routepilot"]), "serve", "-config", str(config_path)],
                work / "gateway.log",
            )
        )
        overhead.wait_ready(f"http://127.0.0.1:{gateway_port}/healthz", gateway)

        with ThreadPoolExecutor(max_workers=clients) as pool:
            futures = [
                pool.submit(
                    run_client,
                    make_sender(gateway_port),
                    max_tokens_stream(SEED, i),
                    STOP_AFTER,
                    MAX_REQUESTS_PER_CLIENT,
                )
                for i in range(clients)
            ]
            tallies = [f.result() for f in futures]
        total = merge_tallies(tallies)
        stopped = Counter(t.stopped for t in tallies)

        # The gateway writes the ledger rows in the background: give its writer the time to put
        # the last ones in before the ledger is read. A real loss is not hidden by this; the
        # count then stays short and the checks report it.
        wait_for_ledger_rows(db_path, tenant, total.admitted)
        scrape = overhead.scrape(gateway_port)
        ledger = ledger_summary(db_path, tenant)
        mock_requests = mock_request_count(mock_port)
        reconcile = run_reconcile(bins["routepilot"], config_path)

    # The gateway has stopped, so its log is complete and closed.
    ledger_failures = parse_ledger_failures(
        (work / "gateway.log").read_text(encoding="utf-8", errors="replace")
    )
    overshoot = parse_samples(scrape, "rp_budget_overshoot_picousd_total")
    metrics = {
        "overshoot_pico": int(overshoot[0][1]) if overshoot else 0,
        "ratelimit": {
            labels["kind"]: int(value)
            for labels, value in parse_samples(scrape, "rp_ratelimit_rejections_total")
        },
        "budget_rejections": int(
            sum(v for _, v in parse_samples(scrape, "rp_budget_rejections_total"))
        ),
        "accounting_write_errors": int(
            sum(v for _, v in parse_samples(scrape, "rp_accounting_write_errors_total"))
        ),
        "requests_by_status": {
            labels["status"]: int(value)
            for labels, value in parse_samples(scrape, "rp_requests_total")
        },
    }
    limit_pico = usd_to_pico(BUDGET_USD)
    expected = total.prompt_tokens * PRICE_IN_PICO + total.completion_tokens * PRICE_OUT_PICO
    client_stats = {
        "threads": clients,
        "requests": total.requests,
        "admitted": total.admitted,
        "rejected": total.rejected,
        "other": total.other,
        "prompt_tokens": total.prompt_tokens,
        "completion_tokens": total.completion_tokens,
        "stopped": dict(stopped),
    }
    checks = quota_checks(
        backend, limit_pico, client_stats, ledger, metrics, mock_requests, reconcile, expected
    )
    return {
        "backend": backend,
        "redis_mode": None,
        "tenant": tenant,
        "limit_pico": limit_pico,
        "clients": client_stats,
        "ledger": ledger,
        "ledger_write_failures": ledger_failures,
        "metrics": metrics,
        "mock_requests": mock_requests,
        "reconcile": reconcile,
        "expected_cost_pico": expected,
        "unused_pico": limit_pico - expected,
        "checks": checks,
        "wall_time_s": round(time.monotonic() - began, 1),
    }


# --- report ------------------------------------------------------------------------------------


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _usd(pico: int | None) -> str:
    return "-" if pico is None else pico_to_usd(int(pico))


def _redis_label(mode: str | None) -> str:
    return {
        "docker": "real Redis 7.4 (Docker, loopback port)",
        "url": f"real Redis at ${REDIS_ENV}",
        "miniredis-go-test": "miniredis (in-process Go test, not the gateway binary)",
    }.get(mode or "", "-")


def _reconcile_text(rec: Mapping[str, Any]) -> str:
    lines = "; ".join(
        f"{x['tenant']} {x['ledger_usd']} {x['counter_usd']} {x['diff_usd']}" for x in rec["lines"]
    )
    return f"exit {rec['exit_code']}" + (f"; {lines}" if lines else "")


def _counter_text(rec: Mapping[str, Any]) -> str:
    counters = [x["counter_usd"] for x in rec["lines"]]
    return f"{', '.join(counters)} USD" if counters else "-"


def _backend_rows(b: Mapping[str, Any]) -> list[str]:
    c, led, met = b["clients"], b["ledger"], b["metrics"]
    unused, limit = b["unused_pico"], b["limit_pico"]
    lost = c["admitted"] - led["rows"]
    return [
        str(c["requests"]),
        str(c["admitted"]),
        str(c["rejected"]),
        str(sum(c["other"].values())),
        f"{_usd(b['expected_cost_pico'])} USD",
        f"{_usd(led['sum_pico'])} USD",
        _counter_text(b["reconcile"]),
        f"{_usd(limit)} USD",
        f"{_usd(unused)} USD ({unused / limit * 100:.2f}% of the limit)",
        f"{met['overshoot_pico']}",
        "/".join(str(met["ratelimit"].get(k, 0)) for k in ("rpm", "tpm", "concurrency")),
        f"{led['rows']} ({led['ok_rows']} ok), {lost} fewer than admitted",
        str(met.get("accounting_write_errors", 0)),
        str(b["mock_requests"]),
        f"{_usd(led['estimate_min'])} to {_usd(led['estimate_max'])} USD",
        _reconcile_text(b["reconcile"]),
        f"{b['wall_time_s']} s",
    ]


def _row_labels(clients: int) -> tuple[str, ...]:
    return (
        f"requests sent by the {clients} clients",
        "admitted (200)",
        "rejected (429 insufficient_quota)",
        "other responses",
        "committed spend recomputed from the tokens the clients saw (independent of the gateway)",
        "committed spend per the usage ledger (sum of cost_actual)",
        "committed spend per the Redis budget counter (from reconcile)",
        "budget limit",
        "unused budget at the end (limit minus the recomputed committed spend)",
        "rp_budget_overshoot_picousd_total (picodollars)",
        "rate-limit rejections rpm/tpm/concurrency",
        "usage ledger rows",
        "usage ledger write errors (rp_accounting_write_errors_total)",
        "requests the mock received",
        "worst-case reservation (cost_estimate) of the ledger's rows, min to max",
        "routepilot reconcile (exit; tenant ledger_usd counter_usd diff_usd)",
        "wall time",
    )


def _result_section(backends: Mapping[str, Mapping[str, Any]]) -> list[str]:
    out = ["## Result", ""]
    for name, b in backends.items():
        failed = [c for c in b["checks"] if not c["passed"]]
        total = len(b["checks"])
        verdict = (
            "every check passed"
            if not failed
            else f"{len(failed)} check(s) failed: " + "; ".join(c["name"] for c in failed)
        )
        out.append(f"- {name}: {total - len(failed)} of {total} checks passed, {verdict}.")
    return [*out, ""]


def _ledger_loss_section(name: str, b: Mapping[str, Any]) -> list[str]:
    failures = b.get("ledger_write_failures")
    lost = b["clients"]["admitted"] - b["ledger"]["rows"]
    if lost <= 0 and not (failures and failures["log_lines"]):
        return []
    errors = (failures or {}).get("errors", {})
    causes = ", ".join(f"{k}: {v}" for k, v in errors.items()) or "none logged"
    return [
        f"Usage ledger rows missing on the {name} backend: the ledger holds {b['ledger']['rows']} "
        f"rows for {b['clients']['admitted']} admitted requests, {lost} rows fewer. The gateway "
        f"counted {b['metrics'].get('accounting_write_errors', 0)} ledger write errors "
        "(rp_accounting_write_errors_total) and logged "
        f"{(failures or {}).get('log_lines', 0)} warnings covering "
        f"{(failures or {}).get('rows_lost', 0)} rows; the causes in the log are: {causes}. "
        "The gateway queues a request's rows and a background writer puts them in the ledger in "
        "batches (up to 512 rows, or 50 ms after the oldest), through the one connection of the "
        "SQLite store. A row is lost when the queue is full, or when a batch write fails or "
        "exceeds its 5 s bound; the loss is counted instead of failing the request. The budget "
        "counter is not affected by a lost row: it is charged before the ledger write.",
        "",
    ]


def render_report(
    results: Mapping[str, Any],
    machine: Mapping[str, Any],
    run_id: str,
    go_version: str | None,
) -> str:
    """The report.md text for a results dict (the content of results.json)."""
    go = f", {go_version}" if go_version else ""
    s = results["settings"]
    backends = results["backends"]
    full = {name: b for name, b in backends.items() if "clients" in b}
    low, high = s["max_tokens_range"]
    out = [
        f"# Quota correctness under {s['clients']} concurrent clients ({run_id})",
        "",
        "Labels: real local measurement, nothing simulated. Machine: "
        f"{machine['cpu']}, {machine['ram_gb']} GB RAM, {machine['os']}{go}. The load clients, the "
        "gateway (routepilot serve), the mock (rp-mock) and Redis share one machine and talk over "
        "loopback. The experiment measures counts and money, not time.",
        "",
        f"Setup: one tenant with a total budget of {s['budget_usd']} USD and TPM and concurrency "
        f"limits far above reach. {s['clients']} clients (threads), each sending requests with a "
        f"seeded random max_tokens in [{low}, {high}] until the gateway has answered it "
        f"{s['stop_after']} times in a row with 429 insufficient_quota. The model costs "
        f"{s['price_in_per_mtok']} USD per million input tokens and {s['price_out_per_mtok']} USD "
        "per million output tokens. The mock caps its answer at each request's max_tokens (the "
        f"model has no mock directive, and the mock's default length is {s['mock_tokens']} "
        f"tokens, started with --tokens {s['mock_tokens']}, above every max_tokens a client "
        "sends), so the completion tokens of a request equal its max_tokens and what it costs "
        "is what was reserved for its output. Money is integer picodollars (1e-12 USD).",
        "",
        *_result_section(backends),
    ]
    if s["clients"] != CLIENTS:
        out[2:2] = [
            f"Control run: {s['clients']} clients instead of the briefed {CLIENTS}, to see whether "
            "what differs in the briefed run depends on the concurrency. It is not the briefed "
            "run.",
            "",
        ]
    if full:
        names = list(full)
        columns = [_backend_rows(full[n]) for n in names]
        labels = _row_labels(s["clients"])
        rows = [[label, *(col[i] for col in columns)] for i, label in enumerate(labels)]
        out += [_table(["quantity", *(f"{n} backend" for n in names)], rows), ""]
    for name, b in backends.items():
        out += [f"## Backend: {name}", ""]
        if b.get("redis_mode"):
            out += [f"Redis: {_redis_label(b['redis_mode'])}.", ""]
        if "go_test" in b:
            out += [
                "No real Redis was available, so this row is the gateway test that runs the same "
                "race against miniredis. It is a Go test of the gateway handler, not a run of the "
                "routepilot binary.",
                "",
                "```",
                *b["go_test"]["tail"],
                "```",
                "",
            ]
        if "reconcile" in b:
            rec = b["reconcile"]
            if name == "memory":
                out += [
                    "routepilot reconcile cannot read the memory backend's counter, which lives in "
                    "the gateway process, so it exits 2 by design. For this backend the ledger sum "
                    "against the limit and the overshoot counter are the evidence.",
                    "",
                ]
            out += [f"reconcile output: `{rec['stdout'].strip() or rec['stderr']}`", ""]
        if "ledger" in b:
            out += _ledger_loss_section(name, b)
        rows = [[c["name"], "pass" if c["passed"] else "FAIL", c["detail"]] for c in b["checks"]]
        out += [_table(["check", "result", "detail"], rows), ""]
    out += [
        f"Wall time of the whole run: {results['wall_time_s']} s.",
        "",
        "## Limitations",
        "",
        "- One run per backend and one seed: the clients race, so the interleaving (and with it "
        "which requests are admitted) differs between runs. The checks do not depend on it.",
        "- The mock's answer has exactly max_tokens completion tokens, the most a request allows, "
        "so the output cost is always the full reservation; the prompt part of the reservation is "
        "an upper bound that the actual prompt count stays below. A real provider often stops "
        "short of max_tokens, which would leave more of the budget unused.",
        "- The 20-in-a-row stop rule ends a client while budget may remain that only a smaller "
        "max_tokens could use; the unused budget above is what was left when the last client "
        "stopped.",
        "- A single Redis instance on the same machine: no network latency, no failover, no "
        "cluster.",
        "",
    ]
    return "\n".join(out)


# --- main --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quota correctness under concurrent clients.")
    parser.add_argument("--backends", default=",".join(BACKENDS), help="comma-separated backends")
    parser.add_argument(
        "--clients",
        type=int,
        default=CLIENTS,
        help=f"concurrent client threads (default {CLIENTS}; fewer is a control run)",
    )
    parser.add_argument(
        "--redis",
        choices=("auto", "docker", "url", "miniredis"),
        default="auto",
        help=f"where the redis backend runs: docker (start {REDIS_IMAGE}), url (the Redis at "
        f"${REDIS_ENV}) or miniredis (the Go test); auto picks url, then docker, then miniredis",
    )
    parser.add_argument("--out", type=Path, help="result directory (default: under bench/results)")
    parser.add_argument("--keep-scratch", action="store_true", help="keep logs and configs")
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=ROOT / "bench" / "scratch",
        help="where the binaries, configs, ledgers and process logs go",
    )
    parser.add_argument("--render", type=Path, metavar="DIR", help="rewrite DIR/report.md and exit")
    return parser


def _rewrite_report(directory: Path) -> int:
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    info = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    text = render_report(results, info["machine"], info["run_id"], info["go_version"])
    (directory / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"quota_correctness: wrote {(directory / 'report.md').as_posix()}")
    return 0


def _go_test_runner(command: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=900,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _resolve_redis_mode(choice: str) -> str:
    if choice != "auto":
        return choice
    if os.environ.get(REDIS_ENV):
        return "url"
    return "docker" if docker_available() else "miniredis"


def _run_redis(
    mode: str, bins: Mapping[str, Path], scratch: Path, suffix: str, clients: int
) -> dict[str, Any] | None:
    """Run the redis backend in the mode; None when the mode needs a variable that is not set."""
    if mode == "miniredis":
        return miniredis_go_test(_go_test_runner)
    if mode == "url":
        url = os.environ.get(REDIS_ENV)
        if not url:
            return None
        result = run_backend("redis", url, bins, scratch, suffix, clients)
        result["redis_mode"] = "url"
        return result
    with redis_container(docker_run, overhead.free_port()) as url:
        result = run_backend("redis", url, bins, scratch, suffix, clients)
    result["redis_mode"] = "docker"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.render:
        return _rewrite_report(args.render)
    names = [n for n in args.backends.split(",") if n]
    unknown = [n for n in names if n not in BACKENDS]
    if unknown:
        print(f"quota_correctness: unknown backend {unknown[0]!r}", file=sys.stderr)
        return 2

    started = datetime.now(UTC)
    clock = time.monotonic()
    run_id = runinfo.new_run_id(started)
    out_dir = args.out or ROOT / "bench" / "results" / EXPERIMENT / run_id
    scratch = args.scratch_root / f"{EXPERIMENT}-{run_id}"
    scratch.mkdir(parents=True, exist_ok=True)
    suffix = run_id.split("-")[0].lower()

    print(f"quota_correctness: building binaries ({run_id})", flush=True)
    bins = overhead.build_binaries(scratch / "bin")
    backends: dict[str, Any] = {}
    all_passed = True
    for name in names:
        print(f"quota_correctness: running the {name} backend", flush=True)
        if name == "memory":
            result = run_backend("memory", None, bins, scratch, suffix, args.clients)
        else:
            mode = _resolve_redis_mode(args.redis)
            print(f"quota_correctness: redis mode {mode}", flush=True)
            result = _run_redis(mode, bins, scratch, suffix, args.clients)
            if result is None:
                print(f"quota_correctness: {REDIS_ENV} is not set", file=sys.stderr)
                return 2
        backends[name] = result
        failed = [c["name"] for c in result["checks"] if not c["passed"]]
        all_passed = all_passed and not failed
        if "clients" in result:
            print(
                f"quota_correctness: {name}: admitted={result['clients']['admitted']} "
                f"rejected={result['clients']['rejected']} "
                f"spent={_usd(result['ledger']['sum_pico'])} USD "
                f"overshoot={result['metrics']['overshoot_pico']} checks_failed={len(failed)} "
                f"({result['wall_time_s']} s)",
                flush=True,
            )
        for check in failed:
            print(f"quota_correctness: CHECK FAILED: {check}", file=sys.stderr, flush=True)
        time.sleep(overhead.COOLDOWN_S)

    results: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "settings": {
            "clients": args.clients,
            "seed": SEED,
            "budget_usd": BUDGET_USD,
            "limit_pico": usd_to_pico(BUDGET_USD),
            "upstream_model": UPSTREAM_MODEL,
            "mock_tokens": MOCK_TOKENS,
            "price_in_per_mtok": PRICE_IN_PER_MTOK,
            "price_out_per_mtok": PRICE_OUT_PER_MTOK,
            "max_tokens_range": list(MAX_TOKENS_RANGE),
            "stop_after": STOP_AFTER,
            "prompt_bound_tokens": PROMPT_BOUND_TOKENS,
        },
        "backends": backends,
        "all_checks_passed": all_passed,
        "wall_time_s": round(time.monotonic() - clock, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    command = [
        "python",
        "bench/scripts/quota_correctness.py",
        *(argv if argv is not None else sys.argv[1:]),
    ]
    info = runinfo.write_run_json(out_dir, EXPERIMENT, {"clients": SEED}, command, None, started)
    report = render_report(results, info["machine"], info["run_id"], info["go_version"])
    (out_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    print(
        f"quota_correctness: wrote {out_dir.as_posix()} in {results['wall_time_s']} s", flush=True
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
