"""The live scripts against the real gateway and a mock upstream.

The committed live config is pointed at ``rp-mock`` instead of Ollama, so ``collect`` and the demo
run end to end here: they start ``routepilot serve`` on a loopback port, send streaming requests
through it, read the decision log back and write their tables and reports. Ollama is not
involved (its load and unload calls are replaced), and every prompt and answer is invented.
"""

import json
import subprocess
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_live_train import SENTINEL, catalog, synthetic_live

from routepilot_train import replaytool
from routepilot_train.runinfo import repo_root

SCRIPTS = repo_root() / "bench" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import live_collect as lc  # noqa: E402
import live_demo as ld  # noqa: E402
import live_train as lt  # noqa: E402
import overhead  # noqa: E402

EM_DASH = "\u2014"
ROOT = repo_root()


@pytest.fixture(scope="module")
def binary() -> Path:
    try:
        return replaytool.build_go(ROOT)
    except replaytool.GoToolchainError as err:
        pytest.skip(str(err))


@pytest.fixture(scope="module")
def mock_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """An rp-mock on a free loopback port, stopped when the module is done."""
    exe = tmp_path_factory.mktemp("mock") / (
        "rp-mock.exe" if sys.platform == "win32" else "rp-mock"
    )
    built = subprocess.run(
        ["go", "build", "-o", str(exe), "./cmd/rp-mock"], cwd=ROOT, capture_output=True, text=True
    )
    if built.returncode != 0:
        pytest.skip(f"cannot build rp-mock: {built.stderr.strip()}")
    port = overhead.free_port()
    command = [
        str(exe),
        "--listen",
        f"127.0.0.1:{port}",
        "--tokens",
        "40",  # longer than the answer-only cap, so the cap shows in the completion tokens
        "--ttft-median-ms",
        "20",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        overhead.wait_ready(f"http://127.0.0.1:{port}/v1/models", proc)
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        overhead.stop(proc)


@pytest.fixture
def no_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "ollama_load", lambda tag, keep_alive="60m": None)
    monkeypatch.setattr(lc, "ollama_unload", lambda tag: None)
    monkeypatch.setattr(lc, "ollama_loaded", lambda: [])


def mock_config(tmp_path: Path, mock_url: str) -> Path:
    """The committed live config with its provider pointed at the mock."""
    cfg = yaml.safe_load(lc.CONFIG.read_text(encoding="utf-8"))
    cfg["providers"][0]["base_url"] = mock_url
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def write_measure_run(tmp_path: Path) -> Path:
    run = tmp_path / "measure-run"
    run.mkdir()
    models = {
        c.tag: {
            "ttft_ms_mean": 100.0,
            "tokens_per_s_mean": 960.0,  # 0.5 s for a request that runs to 384 tokens
            "worst_case_request_s": 0.5,
            "load_s": 0.1,
        }
        for c in lc.load_catalog()  # the committed models
    }
    (run / "results.json").write_text(json.dumps({"phase": "measure", "models": models}))
    return run


def build_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A derived dir with invented gpqa and mmlupro prompts and a tarball with their truth."""
    prompts = []
    members: dict[str, list[dict]] = {"gpqa": [], "mmlupro": []}
    for dataset, n, letters in (("gpqa", 6, "ABCD"), ("mmlupro", 14, "ABCDEFGHIJ")):
        for i in range(n):
            options = "\n".join(f"{c}) option {c}" for c in letters)
            text = f"{SENTINEL} {dataset} {i}\n{options}\nThink step by step, then answer."
            prompts.append(
                {
                    "dataset": dataset,
                    "item_id": f"{dataset}/{i}",
                    "prompt": text,
                    "prompt_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(),
                    "prompt_bytes": len(text.encode()),
                }
            )
            members[dataset].append({"index": i, "prompt": text, "ground_truth": letters[i % 4]})
    derived = tmp_path / "derived"
    derived.mkdir()
    pd.DataFrame(prompts).to_parquet(derived / "prompts.parquet", index=False)
    tar_path = tmp_path / "release.tar.gz"
    splits = {"gpqa": "test", "mmlupro": "test_3000"}
    with tarfile.open(tar_path, "w:gz") as tar:
        for dataset, records in members.items():
            payload = {
                "dataset_name": dataset,
                "split": splits[dataset],
                "model_name": "m",
                "records": records,
            }
            data = json.dumps(payload).encode()
            info = tarfile.TarInfo(f"bench-release/{dataset}/m/{dataset}-m-1.json")
            info.size = len(data)
            tar.addfile(info, __import__("io").BytesIO(data))
    return derived, tar_path


def test_collect_runs_through_the_gateway_and_writes_the_tables(
    binary: Path, mock_url: str, no_ollama: None, tmp_path: Path
) -> None:
    derived, tar_path = build_inputs(tmp_path)
    live = tmp_path / "live"
    measure_run = write_measure_run(tmp_path)
    out = lc.collect(
        live,
        tmp_path / "results",
        measure_run,
        budget_minutes=5,
        n_mmlupro=8,
        config=mock_config(tmp_path, mock_url),
        tarball=tar_path,
        derived=derived,
        binary=binary,
    )
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["stopped_by_budget"] is False
    assert results["decisions_missing"] == 0
    assert results["realized_items"] == 6 + 8  # all of gpqa, the sample of mmlupro
    assert results["plan"]["used_mmlupro"] == 8
    assert results["protocol"] == lc.protocol_for("as-written").as_dict()
    assert results["plan"]["max_tokens"] == 384
    assert results["ground_truth"]["items_with_disagreeing_records"] == 0

    outcomes = pd.read_parquet(live / "outcomes.parquet")
    assert len(outcomes) == 3 * 14
    assert set(outcomes["model"]) == {c.id for c in lc.load_catalog()}
    assert (outcomes["prompt_tokens"] > 0).all() and (outcomes["completion_tokens"] > 0).all()
    assert (outcomes["completion_tokens"] == 40).all()  # the mock's answer, under the 384 cap
    assert (outcomes["cost_usd"] > 0).all()
    measured = pd.read_parquet(live / "measured.parquet")
    assert len(measured) == len(outcomes)
    assert (measured["e2e_ms"] >= measured["ttft_ms"]).all() and (measured["ttft_ms"] > 0).all()
    assert pd.read_parquet(live / "prompts.parquet")["item_id"].nunique() == 14
    # the gateway logged one decision per request (warm-ups included), none holding prompt text
    log = (live / "decisions-collect.jsonl").read_text(encoding="utf-8")
    assert log.count('"type":"decision"') == len(outcomes) + 3
    assert SENTINEL not in log

    for name in ("results.json", "report.md", "run.json"):
        text = (out / name).read_text(encoding="utf-8")
        assert SENTINEL not in text and EM_DASH not in text
    # the raw answers stay in the live directory and nowhere else
    assert SENTINEL not in json.dumps(results)

    with pytest.raises(FileExistsError, match="--resume"):
        lc.collect(
            live,
            tmp_path / "results",
            measure_run,
            n_mmlupro=8,
            config=mock_config(tmp_path, mock_url),
            tarball=tar_path,
            derived=derived,
            binary=binary,
        )


def test_collect_stops_at_the_budget_and_reports_the_realized_sample(
    binary: Path, mock_url: str, no_ollama: None, tmp_path: Path
) -> None:
    derived, tar_path = build_inputs(tmp_path)
    live = tmp_path / "live"
    out = lc.collect(
        live,
        tmp_path / "results",
        write_measure_run(tmp_path),
        budget_minutes=0.02,  # 1.2 s: the three warm-ups and a few requests
        n_mmlupro=8,
        config=mock_config(tmp_path, mock_url),
        tarball=tar_path,
        derived=derived,
        binary=binary,
    )
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["stopped_by_budget"] is True
    assert results["realized_items"] < 14
    requests = pd.read_json(live / "raw.jsonl", lines=True)
    assert len(requests) < 3 * 14


def test_collect_with_the_answer_only_protocol_sends_the_system_message_and_the_cap(
    binary: Path, mock_url: str, no_ollama: None, tmp_path: Path
) -> None:
    derived, tar_path = build_inputs(tmp_path)
    live = tmp_path / "live"
    measure_run = write_measure_run(tmp_path)
    only = lc.protocol_for("answer-only")
    command = ["python", "bench/scripts/live_collect.py", "collect", "--protocol", "answer-only"]
    out = lc.collect(
        live,
        tmp_path / "results",
        measure_run,
        budget_minutes=5,
        n_mmlupro=4,
        protocol="answer-only",
        config=mock_config(tmp_path, mock_url),
        tarball=tar_path,
        derived=derived,
        binary=binary,
        command=command,
    )
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["protocol"] == only.as_dict()
    assert results["plan"]["max_tokens"] == 16
    # the plan took the measured speeds at the protocol's cap, not the stored 384-token time
    first = lc.load_catalog()[0].id
    assert results["plan"]["worst_case_request_s"][first] == pytest.approx(0.1 + 16 / 960)
    assert results["decisions_missing"] == 0 and results["realized_items"] == 6 + 4

    outcomes = pd.read_parquet(live / "outcomes.parquet")
    assert len(outcomes) == 3 * 10
    assert (outcomes["completion_tokens"] == 16).all()  # the mock stops at max_tokens
    prompts = pd.read_parquet(live / "prompts.parquet").set_index("item_id")
    for row in outcomes.itertuples():
        size = len(prompts.loc[row.item_id, "prompt"].encode()) + len(only.system.encode())
        assert row.prompt_tokens == (size + 3) // 4 + 8  # the mock counts two messages

    assert lc.load_protocol(live) == only
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["measurement"]["protocol"] == only.as_dict()
    assert run["command"] == command
    report = (out / "report.md").read_text(encoding="utf-8")
    assert only.system in report and "answer-only" in report
    for name in ("results.json", "report.md", "run.json"):
        text = (out / name).read_text(encoding="utf-8")
        assert SENTINEL not in text and EM_DASH not in text

    with pytest.raises(ValueError, match="answer-only"):  # resume keeps the collection's protocol
        lc.collect(
            live,
            tmp_path / "results",
            measure_run,
            n_mmlupro=4,
            protocol="as-written",
            resume=True,
            config=mock_config(tmp_path, mock_url),
            tarball=tar_path,
            derived=derived,
            binary=binary,
        )


def test_a_demo_serves_auto_on_the_learned_policy_in_nine_rounds(
    binary: Path, mock_url: str, no_ollama: None, tmp_path: Path
) -> None:
    prompts, outcomes, measured = synthetic_live()
    live = tmp_path / "live"
    live.mkdir()
    only = lc.protocol_for("answer-only")
    lc.write_protocol(live, only)  # the protocol the (invented) collection ran with
    prompts.to_parquet(live / "prompts.parquet", index=False)
    outcomes.to_parquet(live / "outcomes.parquet", index=False)
    measured.to_parquet(live / "measured.parquet", index=False)
    truth = pd.DataFrame({"item_id": prompts["item_id"], "ground_truth": "A"})
    truth.to_parquet(live / "ground_truth.parquet", index=False)

    config = mock_config(tmp_path, mock_url)
    # the synthetic catalog differs from the committed one, so train on the committed models
    outcomes["model"] = outcomes["model"].map(
        dict(zip(["small", "mid", "big"], [c.id for c in lc.load_catalog()], strict=True))
    )
    measured["model"] = measured["model"].map(
        dict(zip(["small", "mid", "big"], [c.id for c in lc.load_catalog()], strict=True))
    )
    outcomes.to_parquet(live / "outcomes.parquet", index=False)
    measured.to_parquet(live / "measured.parquet", index=False)
    assert catalog()  # the helper still builds
    policy_path = tmp_path / "bench" / "policies" / "live.yaml"
    profiles_path = tmp_path / "bench" / "latency" / "live.yaml"
    train_out = lt.run_live_train(
        live,
        tmp_path / "policies",
        tmp_path / "train-results",
        config=config,
        policy_path=policy_path,
        profiles_path=profiles_path,
        dim=1024,
        bootstrap=20,
        workers=2,
        binary=binary,
    )
    assert train_out.is_dir()
    trained = json.loads((train_out / "results.json").read_text(encoding="utf-8"))
    assert trained["protocol"] == only.as_dict()  # the training report names the data's protocol

    out = ld.run_demo(
        live,
        tmp_path / "demo-results",
        config=config,
        policies_root=tmp_path / "policies",
        policy_path=policy_path,
        profiles_path=profiles_path,
        items_per_round=3,
        binary=binary,
    )
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert [r["round"] for r in results["rounds"]] == [
        "q0-none", "q0-tight", "q0-loose",
        "q0.4-none", "q0.4-tight", "q0.4-loose",
        "q0.6-none", "q0.6-tight", "q0.6-loose",
    ]  # fmt: skip
    assert results["decisions_missing"] == 0
    assert results["requests"] == 27 and results["items_per_round"] == 3
    # the demo sends the protocol the collection used: its system message and its cap
    assert results["protocol"] == only.as_dict()
    sent = [json.loads(line) for line in (live / "demo-raw.jsonl").read_text("utf-8").splitlines()]
    prompt_of = dict(zip(prompts["item_id"], prompts["prompt"], strict=True))
    assert len(sent) == 27
    for record in sent:
        size = len(prompt_of[record["item_id"]].encode()) + len(only.system.encode())
        assert record["prompt_tokens"] == (size + 3) // 4 + 8  # the mock counts two messages
        assert record["completion_tokens"] == 16  # the cap, not the mock's 40
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["measurement"]["protocol"] == only.as_dict()
    served = {m for r in results["rounds"] for m in r["mix"]}
    assert served <= {c.id for c in lc.load_catalog()} and served
    for r in results["rounds"]:
        assert r["n"] == 3 and r["failed"] == 0
        assert r["e2e_ms"]["p50"] > 0 and r["success"]["n"] == 3
    # a tight or loose SLO reports whether it was met; no SLO reports none
    assert results["rounds"][0]["slo"] is None
    assert results["rounds"][2]["slo"]["ms"] == results["slo"]["loose_ms"]
    # the demo log holds only decisions made on the learned policy, and no prompt text
    log = (live / "decisions-demo.jsonl").read_text(encoding="utf-8")
    assert '"policy_role":"stable"' in log and SENTINEL not in log
    for name in ("results.json", "report.md", "run.json"):
        text = (out / name).read_text(encoding="utf-8")
        assert SENTINEL not in text and EM_DASH not in text
