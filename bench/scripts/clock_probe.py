"""Record the clock resolution of the host that runs the performance experiments.

Run from the repository root with the repo venv (Go must be on PATH):

    python bench/scripts/clock_probe.py

The script builds ``bin/rp-load``, runs ``rp-load --clock-probe`` five times and writes
``bench/results/clock/<run-id>/{results.json,report.md,run.json}``. One probe reads Go's
``time.Now`` in a tight loop for 200 ms and reports the smallest non-zero difference between two
consecutive reads (the clock step) and the share of consecutive reads that were equal. Five probes
show whether the step is stable on the host.

The overhead experiment refuses to run when the step is above 1000 ns, because the
sub-millisecond quantities it reports would then be counts of clock ticks. Running this script on
each host that could run the experiment, Windows and Linux, shows why timing runs need Linux.
Nothing here is simulated, fitted or assumed.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from routepilot_train import runinfo
from routepilot_train.evaluate import report

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overhead  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "clock"
PROBES = 5
PROBE_MS = 200  # the duration of one `rp-load --clock-probe` (clockProbeDuration in cmd/rp-load)


def build_loadgen(bin_dir: Path) -> Path:
    """Build ``rp-load`` from this tree into ``bin_dir`` and return the binary."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    target = bin_dir / f"rp-load{suffix}"
    proc = subprocess.run(
        ["go", "build", "-o", str(target), "./cmd/rp-load"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"go build ./cmd/rp-load failed: {proc.stderr.strip()}")
    return target


def collect(
    binary: Path,
    count: int = PROBES,
    probe: Callable[[Path], dict[str, Any]] = overhead.probe_clock,
) -> list[dict[str, Any]]:
    """Run the probe ``count`` times and keep every result."""
    return [probe(binary) for _ in range(count)]


def summarize(probes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Min, median and max clock step, the worst zero fraction and the verdict on the threshold."""
    if not probes:
        raise ValueError("clock summary: no probes")
    steps = [int(p["clock_step_ns"]) for p in probes]
    return {
        "probes": len(probes),
        "min_step_ns": min(steps),
        "median_step_ns": statistics.median_low(steps),
        "max_step_ns": max(steps),
        "max_zero_fraction": max(float(p["zero_fraction"]) for p in probes),
        "threshold_ns": overhead.COARSE_CLOCK_NS,
        "coarse": max(steps) > overhead.COARSE_CLOCK_NS,
    }


def run_clock_probe(
    out_root: Path,
    command: Sequence[str],
    build: Callable[[Path], Path] = build_loadgen,
    probe: Callable[[Path], dict[str, Any]] = overhead.probe_clock,
    count: int = PROBES,
    bin_dir: Path = ROOT / "bin",
) -> Path:
    """Build, probe, write the run under ``out_root`` and return its directory."""
    started = datetime.now(UTC)
    binary = build(bin_dir)
    found = collect(binary, count, probe)
    summary = summarize(found)
    host = {
        "os": platform.platform(),
        "system": platform.system(),
        "go_version": runinfo.go_version(),
    }
    results: dict[str, Any] = {
        "experiment": EXPERIMENT,
        **host,
        "probe_ms": PROBE_MS,
        "probes": [dict(p) for p in found],
        "summary": summary,
        "labels": [
            "measured on this host with rp-load --clock-probe, which reads time.Now for "
            f"{PROBE_MS} ms and reports the smallest non-zero difference between two consecutive "
            "reads (the clock step) and the share of equal consecutive reads",
            "nothing here is simulated, fitted or assumed",
            f"a step above {overhead.COARSE_CLOCK_NS} ns is too coarse for the sub-millisecond "
            "quantities of the performance experiments, which refuse to run on it",
        ],
    }
    tables = {
        "Probes": pd.DataFrame(
            [{"probe": i, **p} for i, p in enumerate(found, start=1)]
        ),
        "Summary": pd.DataFrame([summary]),
        "Host": pd.DataFrame([host]),
    }
    return report.write_results(
        out_root / runinfo.new_run_id(started),
        EXPERIMENT,
        results,
        tables,
        {},
        {},
        command,
        None,
        started,
        measurement=summary,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record the clock resolution of this host."
    )
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "bench" / "results" / EXPERIMENT
    )
    parser.add_argument("--probes", type=int, default=PROBES)
    args = parser.parse_args(argv)
    given = argv if argv is not None else sys.argv[1:]
    out_dir = run_clock_probe(
        args.out_root,
        ["python", "bench/scripts/clock_probe.py", *given],
        count=args.probes,
    )
    summary = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))[
        "summary"
    ]
    shown = (
        out_dir.relative_to(ROOT).as_posix()
        if out_dir.is_relative_to(ROOT)
        else out_dir
    )
    print(
        f"clock_probe: step {summary['min_step_ns']} to {summary['max_step_ns']} ns over "
        f"{summary['probes']} probes, wrote {shown}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
