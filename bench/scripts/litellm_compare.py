"""RoutePilot and LiteLLM against the same mock, with the direct mock path as the baseline.

Run on Linux (the latency numbers are measured there, Ruling V1-R34) from the repository root with
the repo venv, after installing the pinned LiteLLM into its own venv (bench/litellm/README.md):

    python bench/scripts/litellm_compare.py --scratch-root ~/rp-scratch

It reuses the machinery of ``overhead.py`` (the mock, the open-loop generator, the RoutePilot
cell, the clock guard) and adds a third path: the LiteLLM proxy, which routes one model to the same
``rp-mock``. The matrix is {100, 500, 1000, 2000} requests per second x {non-streaming, streaming}
x {direct to the mock, through RoutePilot, through LiteLLM}, each with a 10 s warmup and a 60 s
measurement window, all in one session so the three paths share conditions. Every cell starts a
fresh mock and a fresh proxy or gateway, and every process binds loopback and is stopped in a
``finally`` (the LiteLLM proxy and its workers as one process group).

The request body, the mock settings (constant 20 ms to the first token, 1 ms per token, 16 tokens)
and the seeds are those of the overhead experiment. For each rate and mode the report gives the
latency percentiles, throughput, errors and drops of every path, and the difference of percentiles
of each gateway against direct (and of LiteLLM against RoutePilot), labeled as such. Vendor claims
(LiteLLM's own benchmark) are not restated as measured here.

Writes bench/results/litellm/<run-id>/{results.json,report.md,run.json}. The clock step is measured
before any cell runs, recorded, and a step above 1000 ns stops the run (the differences are
fractions of a millisecond). The proxy's version is read from the ``x-litellm-version`` header of
the running proxy and must equal the version pinned in bench/litellm/requirements.txt.
"""

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from routepilot_train import runinfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overhead  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "litellm"
LITELLM_DIR = ROOT / "bench" / "litellm"
LITELLM_TEMPLATE = LITELLM_DIR / "config.yaml"
REQUIREMENTS = LITELLM_DIR / "requirements.txt"
DEFAULT_LITELLM_BIN = Path.home() / "rp-tools" / "litellm-venv" / "bin" / "litellm"

LEVELS = overhead.LEVELS
WARMUP_S = overhead.WARMUP_S
DURATION_S = overhead.DURATION_S
COOLDOWN_S = overhead.COOLDOWN_S
# One Python process per worker against RoutePilot's single Go process on 24 threads; a different
# count changes LiteLLM's numbers (bench/litellm/README.md).
WORKERS = 8
LITELLM_ENV = {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_LOG": "ERROR"}
VERSION_HEADER = "x-litellm-version"
READY_TIMEOUT_S = 180.0
PATHS = ("direct", "gateway", "litellm")  # "gateway" is RoutePilot, as in overhead.py
PATH_LABELS = {"direct": "direct", "gateway": "RoutePilot", "litellm": "LiteLLM"}
LITELLM_BODY_KEY = overhead.BENCH_KEY  # sent as the bearer token; the proxy has no master key


# --- the matrix and the comparisons ------------------------------------------------------------


def matrix(levels: Sequence[int]) -> list[overhead.Spec]:
    """Every cell in run order: per rate, non-streaming then streaming, then the three paths."""
    return [
        overhead.Spec(path, rps, stream)
        for rps in levels
        for stream in (False, True)
        for path in PATHS
    ]


def _find(
    runs: Sequence[Mapping[str, Any]], path: str, rps: int, stream: bool
) -> Mapping[str, Any]:
    for run in runs:
        if (run["path"], run["rps"], run["stream"]) == (path, rps, stream):
            return run
    raise KeyError(f"no {path} run at {rps} rps, stream={stream}")


def _diff(other: Mapping[str, Any], base: Mapping[str, Any], block: str) -> dict[str, float]:
    return overhead.difference_of_percentiles(other["load"][block], base["load"][block])


def compare(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per rate and mode: each gateway against direct, and LiteLLM against RoutePilot.

    All are differences of percentiles (the percentile seen through one path minus the same
    percentile seen through the other), not percentiles of per-request differences.
    """
    out: list[dict[str, Any]] = []
    for direct in runs:
        if direct["path"] != "direct":
            continue
        rps, stream = direct["rps"], direct["stream"]
        gateway = _find(runs, "gateway", rps, stream)
        litellm = _find(runs, "litellm", rps, stream)
        ttft = None
        if stream and all(r["load"].get("ttft_ms") for r in (direct, gateway, litellm)):
            ttft = {
                "routepilot_vs_direct": _diff(gateway, direct, "ttft_ms"),
                "litellm_vs_direct": _diff(litellm, direct, "ttft_ms"),
                "litellm_vs_routepilot": _diff(litellm, gateway, "ttft_ms"),
            }
        out.append(
            {
                "rps": rps,
                "stream": stream,
                "routepilot_vs_direct_ms": _diff(gateway, direct, "latency_ms"),
                "litellm_vs_direct_ms": _diff(litellm, direct, "latency_ms"),
                "litellm_vs_routepilot_ms": _diff(litellm, gateway, "latency_ms"),
                "ttft_diff_ms": ttft,
            }
        )
    return out


# --- the LiteLLM proxy -------------------------------------------------------------------------


def litellm_config(mock_port: int) -> dict[str, Any]:
    """The proxy config: the template with the mock's port filled in."""
    text = LITELLM_TEMPLATE.read_text(encoding="utf-8").replace("{{MOCK_PORT}}", str(mock_port))
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"{LITELLM_TEMPLATE} is not a YAML mapping")
    return cfg


def litellm_command(binary: Path, config_path: Path, port: int) -> list[str]:
    """The proxy command line: loopback only, and the fixed number of workers."""
    return [
        str(binary),
        *("--config", str(config_path)),
        *("--host", "127.0.0.1", "--port", str(port)),
        *("--num_workers", str(WORKERS)),
    ]


def litellm_env(base: Mapping[str, str]) -> dict[str, str]:
    """The proxy's environment: the cost map is read from the package, never fetched."""
    return {**base, **LITELLM_ENV}


def version_from_headers(headers: Mapping[str, str]) -> str | None:
    """The proxy's version, from the ``x-litellm-version`` header of one of its responses."""
    for name, value in headers.items():
        if name.lower() == VERSION_HEADER:
            return value
    return None


def pinned_litellm_version(requirements: str) -> str:
    """The version of ``litellm`` pinned in a requirements file."""
    match = re.search(r"^litellm==([0-9][^\s\\]*)", requirements, re.MULTILINE)
    if match is None:
        raise ValueError("the requirements do not pin litellm with ==")
    return match.group(1)


def litellm_python_version(binary: Path) -> str:
    """The Python of the venv the proxy is installed in, such as ``Python 3.14.9``."""
    python = binary.parent / "python"
    proc = subprocess.run(
        [str(python), "--version"], capture_output=True, text=True, timeout=30, check=False
    )
    return (proc.stdout or proc.stderr).strip()


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_group(proc: "subprocess.Popen[Any]", grace_s: float = 10.0) -> None:
    """Stop a process started with ``start_new_session`` and everything in its group.

    SIGTERM first; whatever is still running after ``grace_s`` seconds gets SIGKILL. Returns once
    the process has been reaped and no process of the group is left, so a worker the parent did
    not wait for cannot outlive the cell.
    """
    pgid = proc.pid  # the session leader's pid is its group id
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if proc.poll() is not None and not _group_alive(pgid):
            return
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    proc.wait()
    deadline = time.monotonic() + 10.0
    while _group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.1)


@contextlib.contextmanager
def litellm_service(
    command: Sequence[str], env: Mapping[str, str], log: Path
) -> Iterator["subprocess.Popen[bytes]"]:
    """Run the proxy in a session of its own with its output in ``log``; always stop all of it."""
    with log.open("wb") as sink:
        proc = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=sink,
            start_new_session=True,
        )
        try:
            yield proc
        finally:
            stop_group(proc)


def litellm_preflight(url: str, stream: bool) -> dict[str, Any]:
    """One request before the load: fails fast on a wrong config, and reads the proxy's version."""
    body = json.dumps(
        {
            "model": overhead.BENCH_MODEL,
            "messages": [{"role": "user", "content": "Reply with one word."}],
            "max_tokens": overhead.MAX_TOKENS,
            "stream": stream,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LITELLM_BODY_KEY}",
        },
    )
    try:
        with overhead.OPENER.open(request, timeout=30) as resp:
            resp.read()
            version = version_from_headers(dict(resp.getheaders()))
            return {"status": resp.status, "headers": {VERSION_HEADER: version}}
    except urllib.error.HTTPError as err:
        excerpt = err.read(300).decode("utf-8", errors="replace")
        raise RuntimeError(f"LiteLLM preflight request got HTTP {err.code}: {excerpt}") from err


# --- one cell ----------------------------------------------------------------------------------


def measure_litellm(
    spec: overhead.Spec,
    bins: Mapping[str, Path],
    scratch: Path,
    warmup_s: int,
    duration_s: int,
    litellm_bin: Path,
) -> dict[str, Any]:
    """Start the mock and the LiteLLM proxy, check one request, run the load, stop everything."""
    mock_port = overhead.free_port()
    mock_command = [
        str(bins["rp-mock"]),
        *("--listen", f"127.0.0.1:{mock_port}"),
        *overhead.MOCK_ARGS,
    ]
    with contextlib.ExitStack() as stack:
        mock = stack.enter_context(
            overhead.service(mock_command, scratch / f"{spec.name}.mock.log")
        )
        overhead.wait_ready(f"http://127.0.0.1:{mock_port}/v1/models", mock)

        port = overhead.free_port()
        config_path = scratch / f"{spec.name}.litellm.yaml"
        config_path.write_text(
            yaml.safe_dump(litellm_config(mock_port), sort_keys=False),
            encoding="utf-8",
            newline="\n",
        )
        proxy = stack.enter_context(
            litellm_service(
                litellm_command(litellm_bin, config_path, port),
                litellm_env(os.environ),
                scratch / f"{spec.name}.litellm.log",
            )
        )
        overhead.wait_ready(
            f"http://127.0.0.1:{port}/health/liveliness", proxy, timeout_s=READY_TIMEOUT_S
        )
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        preflight = litellm_preflight(url, spec.stream)
        load, _ = overhead.run_load(
            bins["rp-load"], url, spec, LITELLM_BODY_KEY, None, warmup_s, duration_s
        )
    return {
        "name": spec.name,
        "path": spec.path,
        "rps": spec.rps,
        "stream": spec.stream,
        "model": spec.model,
        "preflight": preflight,
        "load": load,
        "server": None,
    }


def measure_cell(
    spec: overhead.Spec,
    bins: Mapping[str, Path],
    scratch: Path,
    warmup_s: int,
    duration_s: int,
    litellm_bin: Path,
) -> dict[str, Any]:
    """One cell: direct and RoutePilot cells are the overhead experiment's, LiteLLM's is ours."""
    if spec.path == "litellm":
        return measure_litellm(spec, bins, scratch, warmup_s, duration_s, litellm_bin)
    return overhead.measure(spec, bins, scratch, warmup_s, duration_s, None)


# --- report ------------------------------------------------------------------------------------

_f = overhead._f
_table = overhead._table


def _label(path: str) -> str:
    return PATH_LABELS[path]


def _mode(stream: bool) -> str:
    return "streaming" if stream else "non-streaming"


def _runs_section(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = ("p50", "p90", "p95", "p99", "p99.9", "max")
    rows = []
    for run in runs:
        load = run["load"]
        counts = [str(load[k]) for k in ("scheduled", "completed", "errors", "dropped")]
        lat = load.get("latency_ms")
        rows.append(
            [run["name"], *counts, _f(load["throughput_rps"], 1)]
            + [_f(lat[k]) if lat else "-" for k in keys]
        )
    headers = ["run", "scheduled", "completed", "errors", "dropped", "throughput rps", *keys]
    out = [
        "## Runs (client side, ms)",
        "",
        "Latency percentiles are over the completed requests of the measurement window; a cell "
        "with errors or drops says so in its own columns. 'gateway' is RoutePilot.",
        "",
        _table(headers, rows),
        "",
    ]
    streams = [r for r in runs if r["stream"] and r["load"].get("ttft_ms")]
    if streams:
        ttft = [[r["name"], *(_f(r["load"]["ttft_ms"][k]) for k in keys)] for r in streams]
        out += ["Time to first token (streaming runs, ms):", "", _table(["run", *keys], ttft), ""]
    return out


def _diff_section(comparisons: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = overhead.CLIENT_KEYS
    pairs = (
        ("routepilot_vs_direct_ms", "RoutePilot minus direct"),
        ("litellm_vs_direct_ms", "LiteLLM minus direct"),
        ("litellm_vs_routepilot_ms", "LiteLLM minus RoutePilot"),
    )
    rows = [
        [str(c["rps"]), _mode(c["stream"]), label, *(_f(c[field].get(k)) for k in keys)]
        for c in comparisons
        for field, label in pairs
    ]
    out = [
        "## Latency added by each gateway: difference of percentiles (ms)",
        "",
        "Label: difference of percentiles, not a percentile of differences. Each value is the "
        "percentile seen through one path minus the same percentile seen through the other at "
        "the same rate and mode. Values near the histogram resolution (about 0.03 ms at 40 ms) "
        "are not signal.",
        "",
        _table(["rate", "mode", "comparison", *keys], rows),
        "",
    ]
    ttft_rows = [
        [str(c["rps"]), label, *(_f(c["ttft_diff_ms"][field].get(k)) for k in keys)]
        for c in comparisons
        if c.get("ttft_diff_ms")
        for field, label in (
            ("routepilot_vs_direct", "RoutePilot minus direct"),
            ("litellm_vs_direct", "LiteLLM minus direct"),
            ("litellm_vs_routepilot", "LiteLLM minus RoutePilot"),
        )
    ]
    if ttft_rows:
        out += [
            "Time to first token, difference of percentiles (ms, streaming):",
            "",
            _table(["rate", "comparison", *keys], ttft_rows),
            "",
        ]
    return out


def _lag_section(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = ("p50", "p99", "max")
    rows = [
        [
            r["name"],
            *(
                (_f(r["load"]["launch_lag_ms"][k]) if r["load"].get("launch_lag_ms") else "-")
                for k in keys
            ),
        ]
        for r in runs
    ]
    return [
        "Generator launch lag (how late the scheduler started a request, ms; it is part of every "
        "latency above):",
        "",
        _table(["run", *keys], rows),
        "",
    ]


def render_report(
    results: Mapping[str, Any],
    machine: Mapping[str, Any],
    run_id: str,
    go_version: str | None,
) -> str:
    """The report.md text for a results dict (the content of results.json)."""
    s = results["settings"]
    go = f", {go_version}" if go_version else ""
    clock = results["clock"]
    out = [
        f"# RoutePilot and LiteLLM against the same mock ({run_id})",
        "",
        "Labels: real local measurement, nothing simulated. Machine: "
        f"{machine['cpu']}, {machine['ram_gb']} GB RAM, {machine['os']}{go}. The load generator "
        "(rp-load), the gateway under test and the mock (rp-mock) share one machine and talk over "
        "loopback, in one session so the three paths share conditions. The mock's delays are "
        "configured constants (time to first token 20 ms, 1 ms per output token, 16 output "
        "tokens), so absolute latencies are the mock's; only the differences between paths "
        "measure a gateway. Latencies run from each request's scheduled time (open loop, "
        "coordinated-omission safe). Measured with rp-load --clock-probe before the run, the "
        f"clock step is {clock['clock_step_ns']} ns (the smallest non-zero difference between "
        f"consecutive time.Now reads; {clock['zero_fraction']:.4f} of the consecutive reads were "
        "equal).",
        "",
        f"Settings: {s['warmup_s']} s warmup (sent, not recorded) and {s['duration_s']} s "
        "measurement window per cell, one run per cell, a fresh mock and a fresh proxy or "
        "gateway process for every cell, the request body of the overhead experiment.",
        "",
        f"LiteLLM: version {s['litellm_version']} (read from the x-litellm-version header of the "
        f"running proxy), {s['litellm_python']}, started with --num_workers {s['workers']} "
        f"(LiteLLM ran with {s['workers']} workers: {s['workers']} Python processes against "
        "RoutePilot's one Go process), "
        "LITELLM_LOCAL_MODEL_COST_MAP=True and LITELLM_LOG=ERROR, telemetry off, drop_params on, "
        "no database and no master key. With no master key the proxy does no key lookup per "
        "request, while RoutePilot checks the tenant key on every request, so that setting can "
        "only favor LiteLLM. RoutePilot runs as in the overhead experiment (one pinned model, no "
        "limits, no decision log, accounting off).",
        "",
        "Vendor claims (LiteLLM's own benchmark) are not restated as measured here; the numbers "
        "below are this machine's.",
        "",
        *_runs_section(results["runs"]),
        *_diff_section(results["comparisons"]),
        *_lag_section(results["runs"]),
        f"Wall time of the whole run: {results['wall_time_s']} s.",
        "",
        "## Limitations",
        "",
        "- One run per cell, no repetitions and no confidence intervals. A difference of a "
        "fraction of a millisecond is within the run-to-run noise of a shared machine.",
        "- Linux under WSL2 in a VM that shares the host with Windows; the generator, the mock "
        "and the gateway compete for the same CPU. The launch lag table shows how much the "
        "generator itself added.",
        f"- LiteLLM's numbers depend on its worker count ({s['workers']} here) and its settings; "
        "another count would give other numbers. Only this version and these settings were "
        "measured.",
        "- The mock answers with constant delays and tiny bodies; real providers add variable "
        "latency, and larger bodies cost a gateway more to copy.",
        "- All requests carry one fixed short prompt.",
        "",
    ]
    return "\n".join(out)


# --- main --------------------------------------------------------------------------------------


def _levels(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoutePilot and LiteLLM against the same mock.")
    parser.add_argument("--rps", type=_levels, default=LEVELS, help="comma-separated rates")
    parser.add_argument("--warmup", type=int, default=WARMUP_S, help="warmup seconds per cell")
    parser.add_argument(
        "--duration", type=int, default=DURATION_S, help="measured seconds per cell"
    )
    parser.add_argument(
        "--litellm-bin",
        type=Path,
        default=DEFAULT_LITELLM_BIN,
        help="the litellm executable of the venv the pinned requirements were installed into",
    )
    parser.add_argument("--out", type=Path, help="result directory (default: under bench/results)")
    parser.add_argument("--keep-scratch", action="store_true", help="keep logs and configs")
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=ROOT / "bench" / "scratch",
        help="where the binaries, configs and process logs go; under WSL put it on the Linux "
        "filesystem, since a /mnt/c mount makes every log write slow",
    )
    parser.add_argument("--render", type=Path, metavar="DIR", help="rewrite DIR/report.md and exit")
    return parser


def _rewrite_report(directory: Path) -> int:
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    info = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    text = render_report(results, info["machine"], info["run_id"], info["go_version"])
    (directory / "report.md").write_text(text, encoding="utf-8", newline="\n")
    print(f"litellm_compare: wrote {(directory / 'report.md').as_posix()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.render:
        return _rewrite_report(args.render)
    if not args.litellm_bin.exists():
        print(
            f"litellm_compare: the litellm executable {args.litellm_bin} does not exist; install "
            "the pinned requirements first (bench/litellm/README.md)",
            file=sys.stderr,
        )
        return 2
    pinned = pinned_litellm_version(REQUIREMENTS.read_text(encoding="utf-8"))

    started = datetime.now(UTC)
    clock = time.monotonic()
    run_id = runinfo.new_run_id(started)
    out_dir = args.out or ROOT / "bench" / "results" / EXPERIMENT / run_id
    scratch = args.scratch_root / f"{EXPERIMENT}-{run_id}"
    scratch.mkdir(parents=True, exist_ok=True)

    print(f"litellm_compare: building binaries ({run_id})", flush=True)
    bins = overhead.build_binaries(scratch / "bin")
    probe = overhead.probe_clock(bins["rp-load"])
    print(
        f"litellm_compare: clock step {probe['clock_step_ns']} ns, "
        f"zero fraction {probe['zero_fraction']:.4f}",
        flush=True,
    )
    try:
        coarse = overhead.check_clock(probe, False)
    except overhead.CoarseClockError as err:
        print(f"litellm_compare: {err}", file=sys.stderr, flush=True)
        return 1

    specs = matrix(args.rps)
    runs: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        run = measure_cell(spec, bins, scratch, args.warmup, args.duration, args.litellm_bin)
        runs.append(run)
        load = run["load"]
        latency = load["latency_ms"] or {}
        print(
            f"litellm_compare: [{index}/{len(specs)}] {spec.name} completed={load['completed']} "
            f"errors={load['errors']} dropped={load['dropped']} p95={_f(latency.get('p95'))} ms",
            flush=True,
        )
        time.sleep(COOLDOWN_S)

    seen = {version_from_headers(r["preflight"]["headers"]) for r in runs if r["path"] == "litellm"}
    if seen != {pinned}:
        print(
            f"litellm_compare: the proxy reported version {sorted(map(str, seen))}, "
            f"the requirements pin {pinned}; not writing results",
            file=sys.stderr,
            flush=True,
        )
        return 1

    results: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "settings": {
            "warmup_s": args.warmup,
            "duration_s": args.duration,
            "levels": list(args.rps),
            "workers": WORKERS,
            "litellm_version": pinned,
            "litellm_python": litellm_python_version(args.litellm_bin),
            "litellm_env": LITELLM_ENV,
            "max_inflight": overhead.MAX_INFLIGHT,
            "max_tokens": overhead.MAX_TOKENS,
            "mock_args": list(overhead.MOCK_ARGS),
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
    info = runinfo.write_run_json(
        out_dir,
        EXPERIMENT,
        {"loadgen": overhead.LOAD_SEED, "mock": overhead.MOCK_SEED},
        [
            "python",
            "bench/scripts/litellm_compare.py",
            *(argv if argv is not None else sys.argv[1:]),
        ],
        None,
        started,
        extra=probe,
    )
    report = render_report(results, info["machine"], info["run_id"], info["go_version"])
    (out_dir / "report.md").write_text(report, encoding="utf-8", newline="\n")
    if not args.keep_scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"litellm_compare: wrote {out_dir.as_posix()} in {results['wall_time_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
