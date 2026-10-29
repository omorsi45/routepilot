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

from routepilot_train import features, replaytool
from routepilot_train.runinfo import repo_root

SCRIPTS = repo_root() / "bench" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import live_collect as lc  # noqa: E402
import live_demo as ld  # noqa: E402
import live_train as lt  # noqa: E402
import overhead  # noqa: E402

EM_DASH = "\u2014"
ROOT = repo_root()
INSTRUCTION = "Reply with only the letter of the correct option."


def expected_messages(name: str, prompt: str) -> list[dict[str, str]]:
    """What each protocol puts to a model, written out here so that a change in the code shows."""
    if name == "as-written":
        return [{"role": "user", "content": prompt}]
    if name == "answer-only":
        return [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": prompt}]
    if name == "answer-only-user":
        return [{"role": "user", "content": prompt + "\n\n" + INSTRUCTION}]
    raise AssertionError(name)


def mock_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """What rp-mock reports: ceil(text bytes / 4) plus 4 per message."""
    size = sum(len(m["content"].encode()) for m in messages)
    return (size + 3) // 4 + 4 * len(messages)


@pytest.fixture
def sent_bodies(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Every request body that ``stream_chat`` sends, collected and then sent on unchanged."""
    bodies: list[dict] = []
    real = lc.stream_chat

    def spy(port, body, **kwargs):
        bodies.append(body)
        return real(port, body, **kwargs)

    monkeypatch.setattr(lc, "stream_chat", spy)
    return bodies


@pytest.fixture(scope="module")
def binary() -> Path:
    try:
        return replaytool.build_go(ROOT)
    except replaytool.GoToolchainError as err:
        pytest.skip(str(err))


def start_mock(tmp_path_factory: pytest.TempPathFactory, ttft_ms: int) -> Iterator[str]:
    """An rp-mock on a free loopback port answering after ``ttft_ms``, stopped afterwards."""
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
        str(ttft_ms),
    ]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        overhead.wait_ready(f"http://127.0.0.1:{port}/v1/models", proc)
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        overhead.stop(proc)


@pytest.fixture(scope="module")
def mock_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A quick rp-mock (20 ms to the first token)."""
    yield from start_mock(tmp_path_factory, 20)


@pytest.fixture(scope="module")
def slow_mock_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """An rp-mock that takes 200 ms per answer, for tests that need requests of a known
    minimum length (with 20 ms a fast Linux host answers a whole small collection inside a
    one-second budget)."""
    yield from start_mock(tmp_path_factory, 200)


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
    binary: Path, slow_mock_url: str, no_ollama: None, tmp_path: Path
) -> None:
    derived, tar_path = build_inputs(tmp_path)
    live = tmp_path / "live"
    out = lc.collect(
        live,
        tmp_path / "results",
        write_measure_run(tmp_path),
        budget_minutes=0.02,  # 1.2 s: the three warm-ups and a few requests
        n_mmlupro=8,
        # 200 ms per answer, so the 3 x 14 requests take well over the budget on any machine
        config=mock_config(tmp_path, slow_mock_url),
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


@pytest.mark.parametrize("name", lc.PROTOCOLS)
def test_collect_sends_exactly_the_messages_and_the_cap_of_each_protocol(
    name: str,
    binary: Path,
    mock_url: str,
    no_ollama: None,
    sent_bodies: list[dict],
    tmp_path: Path,
) -> None:
    derived, tar_path = build_inputs(tmp_path)
    live = tmp_path / "live"
    out = lc.collect(
        live,
        tmp_path / "results",
        write_measure_run(tmp_path),
        budget_minutes=5,
        n_mmlupro=4,
        limit=5,
        protocol=name,
        config=mock_config(tmp_path, mock_url),
        tarball=tar_path,
        derived=derived,
        binary=binary,
    )
    protocol = lc.protocol_for(name)
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["protocol"] == protocol.as_dict() and results["decisions_missing"] == 0
    assert lc.load_protocol(live) == protocol
    prompts = pd.read_parquet(live / "prompts.parquet")
    wanted = {json.dumps(expected_messages(name, p)): p for p in prompts["prompt"]}

    # three warm-ups (one per model), then every item on every model, all under the cap
    warmups = [b for b in sent_bodies if b["messages"][-1]["content"] == lc.WARMUP_PROMPT]
    requests = [b for b in sent_bodies if b not in warmups]
    assert len(warmups) == 3 and len(requests) == 3 * 5
    assert {b["max_tokens"] for b in sent_bodies} == {protocol.max_tokens}
    for body in requests:
        assert json.dumps(body["messages"]) in wanted, "a request was not the protocol's messages"
    # what reached the model: rp-mock counts the text it received
    outcomes = pd.read_parquet(live / "outcomes.parquet")
    prompt_of = dict(zip(prompts["item_id"], prompts["prompt"], strict=True))
    assert len(outcomes) == 3 * 5
    for row in outcomes.itertuples():
        want = mock_prompt_tokens(expected_messages(name, prompt_of[row.item_id]))
        assert row.prompt_tokens == want


@pytest.mark.parametrize("name", lc.PROTOCOLS)
def test_training_replay_and_demo_use_the_messages_the_protocol_sends(
    name: str,
    binary: Path,
    mock_url: str,
    no_ollama: None,
    sent_bodies: list[dict],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompts, outcomes, measured = synthetic_live()
    live = tmp_path / "live"
    live.mkdir()
    protocol = lc.protocol_for(name)
    lc.write_protocol(live, protocol)  # the protocol the (invented) collection ran with
    prompts.to_parquet(live / "prompts.parquet", index=False)
    truth = pd.DataFrame({"item_id": prompts["item_id"], "ground_truth": "A"})
    truth.to_parquet(live / "ground_truth.parquet", index=False)

    config = mock_config(tmp_path, mock_url)
    # the synthetic catalog differs from the committed one, so train on the committed models
    ids = dict(zip(["small", "mid", "big"], [c.id for c in lc.load_catalog()], strict=True))
    outcomes["model"] = outcomes["model"].map(ids)
    measured["model"] = measured["model"].map(ids)
    outcomes.to_parquet(live / "outcomes.parquet", index=False)
    measured.to_parquet(live / "measured.parquet", index=False)
    assert catalog()  # the helper still builds
    policy_path = tmp_path / "bench" / "policies" / "live.yaml"
    profiles_path = tmp_path / "bench" / "latency" / "live.yaml"

    # what the trainer featurizes, recorded at the feature extractor
    featurized: list[list[tuple[str, str]]] = []
    real_rows = features.extract_rows

    def spy_rows(batch, dim=features.DEFAULT_DIM):
        featurized.extend(batch)
        return real_rows(batch, dim)

    monkeypatch.setattr(features, "extract_rows", spy_rows)
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
    monkeypatch.setattr(features, "extract_rows", real_rows)
    trained = json.loads((train_out / "results.json").read_text(encoding="utf-8"))
    assert trained["protocol"] == protocol.as_dict()  # the training report names the protocol

    prompt_of = dict(zip(prompts["item_id"], prompts["prompt"], strict=True))
    split_of = dict(
        pd.read_parquet(live / "splits.parquet")[["item_id", "split"]].itertuples(index=False)
    )
    # the trainer featurized exactly the messages sent, for the train and calibration items
    tuning = [i for i, s in split_of.items() if s in ("train", "calib")]
    want_rows = sorted(json.dumps(expected_messages(name, prompt_of[i])) for i in tuning)
    got_rows = sorted(
        json.dumps([{"role": role, "content": text} for role, text in row]) for row in featurized
    )
    assert got_rows == want_rows

    # the replay export holds the same messages for every item, test items included
    replay = {}
    for line in (live / "replay" / "live-ollama.jsonl").read_text("utf-8").splitlines():
        entry = json.loads(line)
        replay[entry["item_id"]] = entry
    assert set(replay) == set(split_of)
    for item_id, entry in replay.items():
        assert entry["messages"] == expected_messages(name, prompt_of[item_id])

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
    # the floors are 0, the median and the 90th percentile of calibration, as the training found
    floors = trained["floors"]["values"]
    assert results["floors"] == trained["floors"]
    assert floors[0] == 0.0 and len(floors) in (2, 3)
    kinds = ["none", "tight", "loose"]
    assert [r["round"] for r in results["rounds"]] == [f"q{f:g}-{k}" for f in floors for k in kinds]
    assert [r["min_quality"] for r in results["rounds"]] == [f for f in floors for _ in kinds]
    n_rounds = 3 * len(floors)
    assert results["decisions_missing"] == 0
    assert results["requests"] == 3 * n_rounds and results["items_per_round"] == 3
    # the demo sends the protocol the collection used: its messages and its cap
    assert results["protocol"] == protocol.as_dict()
    sent = [json.loads(line) for line in (live / "demo-raw.jsonl").read_text("utf-8").splitlines()]
    assert len(sent) == results["requests"]
    demo_requests = [b for b in sent_bodies if b["messages"][-1]["content"] != lc.WARMUP_PROMPT]
    assert len(demo_requests) == len(sent)
    for record, body in zip(sent, demo_requests, strict=True):
        item_id = record["item_id"]
        assert body["messages"] == expected_messages(name, prompt_of[item_id])
        assert body["messages"] == replay[item_id]["messages"]  # the replay saw what the demo sent
        assert body["max_tokens"] == protocol.max_tokens
        assert record["prompt_tokens"] == mock_prompt_tokens(body["messages"])
        assert record["completion_tokens"] == min(protocol.max_tokens, 40)  # the cap, or the mock
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["measurement"]["protocol"] == protocol.as_dict()
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
    for name_ in ("results.json", "report.md", "run.json"):
        text = (out / name_).read_text(encoding="utf-8")
        assert SENTINEL not in text and EM_DASH not in text
