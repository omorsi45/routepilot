"""Tests of bench/scripts/live_collect.py: planning, sampling, ground truth, the streaming client,
grading into outcomes and the report. Every prompt, answer and score here is invented; nothing
talks to Ollama or starts the gateway."""

import hashlib
import io
import json
import sys
import tarfile
import threading
from collections.abc import Iterator
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from routepilot_train.runinfo import repo_root

SCRIPTS = repo_root() / "bench" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import live_collect as lc  # noqa: E402

SENTINEL_PROMPT = "PROMPT-SENTINEL-7f3a"
SENTINEL_ANSWER = "ANSWER-SENTINEL-91bc"
EM_DASH = "\u2014"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the committed config and the prices -------------------------------------------------------


def test_params_of_reads_the_size_from_the_tag() -> None:
    assert lc.params_of("qwen2.5:0.5b") == 0.5
    assert lc.params_of("qwen2.5:1.5b") == 1.5
    assert lc.params_of("llama3.2:3b") == 3.0
    for bad in ("qwen2.5", "qwen2.5:latest", "x:0b", "x:-1b", "x:abcb"):
        with pytest.raises(ValueError, match="parameter count"):
            lc.params_of(bad)


def test_assumed_prices_are_proportional_to_the_parameter_count() -> None:
    assert lc.assumed_prices(0.5) == (Decimal("0.010"), Decimal("0.030"))
    assert lc.assumed_prices(3.0) == (Decimal("0.060"), Decimal("0.180"))


def test_model_id_has_no_colon() -> None:
    assert lc.model_id("qwen2.5:1.5b") == "qwen2.5-1.5b"


def test_the_committed_config_follows_the_rules_of_the_experiment() -> None:
    catalog = lc.load_catalog()
    assert [c.tag for c in catalog] == ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"]
    for c in catalog:
        assert c.id == lc.model_id(c.tag)
        # Ruling V1-R42: 0.02 and 0.06 USD per Mtok for each billion parameters
        assert (c.in_usd_per_mtok, c.out_usd_per_mtok) == lc.assumed_prices(c.params_b)
    doc = yaml.safe_load(lc.CONFIG.read_text(encoding="utf-8"))
    assert [m["rank"] for m in doc["models"]] == [1, 2, 3]
    assert doc["tenants"][0]["api_key_sha256"] == sha(lc.KEY)
    assert doc["server"]["listen"].startswith("127.0.0.1:")
    assert doc["providers"][0]["base_url"].startswith("http://127.0.0.1:")
    text = lc.CONFIG.read_text(encoding="utf-8")
    assert "ASSUMPTION" in text
    assert EM_DASH not in text


def test_cost_uses_the_assumed_price_per_million_tokens() -> None:
    model = lc.CatalogModel("m", "m:3b", 3.0, Decimal("0.06"), Decimal("0.18"))
    assert model.cost_usd(1_000_000, 0) == pytest.approx(0.06)
    assert model.cost_usd(1000, 2000) == pytest.approx(0.06 * 1e-3 + 0.18 * 2e-3)
    assert model.cost_usd(0, 0) == 0.0


def test_render_config_sets_the_port_and_the_log_without_touching_the_base() -> None:
    base = yaml.safe_load(lc.CONFIG.read_text(encoding="utf-8"))
    before = json.dumps(base, sort_keys=True)
    cfg = lc.render_config(base, 4321, Path("/tmp/x/decisions.jsonl"))
    assert cfg["server"]["listen"] == "127.0.0.1:4321"
    assert cfg["decision_log"] == {"path": "/tmp/x/decisions.jsonl"}
    assert json.dumps(base, sort_keys=True) == before
    merged = lc.render_config(base, 1, Path("/tmp/d.jsonl"), {"server": {"latency_profiles": "p"}})
    assert merged["server"]["latency_profiles"] == "p"
    assert merged["server"]["listen"] == "127.0.0.1:1"


# --- planning and sampling ---------------------------------------------------------------------


def test_worst_case_request_is_the_first_token_plus_the_rest_at_speed() -> None:
    assert lc.worst_case_request_s(50.0, 100.0, 384) == pytest.approx(0.05 + 3.84)


def test_plan_sample_size_fits_the_budget_and_the_cap() -> None:
    worst = {"a": 2.0, "b": 4.0, "c": 5.0}  # 11 s for one item on all three models
    # 5400 s fit 490 items; 198 are gpqa, so 292 are left for mmlupro
    assert lc.plan_sample_size(worst, n_gpqa=198, budget_s=5400) == 292
    assert lc.plan_sample_size(worst, n_gpqa=198, budget_s=5400, cap=100) == 100
    # the fixed cost (loads, warm-ups) comes off the budget first: 5400 - 110 = 5290 -> 480 items
    assert lc.plan_sample_size(worst, n_gpqa=198, budget_s=5400, fixed_s=110) == 282
    # not even gpqa fits
    assert lc.plan_sample_size(worst, n_gpqa=198, budget_s=600) == 0
    assert lc.plan_sample_size(worst, n_gpqa=198, budget_s=0) == 0


def test_plan_sample_size_rejects_a_missing_or_zero_time() -> None:
    with pytest.raises(ValueError, match="positive time"):
        lc.plan_sample_size({}, n_gpqa=1, budget_s=10)
    with pytest.raises(ValueError, match="positive time"):
        lc.plan_sample_size({"a": 0.0}, n_gpqa=1, budget_s=10)


# --- the protocols -------------------------------------------------------------------------------


def test_the_protocols_are_as_written_and_answer_only() -> None:
    assert lc.PROTOCOLS == ("as-written", "answer-only", "answer-only-user")
    written = lc.protocol_for("as-written")
    assert (written.name, written.system, written.max_tokens) == ("as-written", None, 384)
    only = lc.protocol_for("answer-only")
    assert only.name == "answer-only" and only.max_tokens == 16
    assert only.system == "Reply with only the letter of the correct option."
    assert only.as_dict() == {
        "name": "answer-only",
        "system_message": "Reply with only the letter of the correct option.",
        "max_tokens": 16,
    }
    assert written.as_dict() == {"name": "as-written", "system_message": None, "max_tokens": 384}
    with pytest.raises(ValueError, match="protocol"):
        lc.protocol_for("verbose")
    assert lc.LIVE_DIR.parts[-3:] == ("data", "derived", "live")


INSTRUCTION = "Reply with only the letter of the correct option."


def test_the_answer_only_user_protocol_appends_the_instruction_to_the_user_turn() -> None:
    user = lc.protocol_for("answer-only-user")
    assert user.name == "answer-only-user" and user.max_tokens == 16
    assert user.system is None  # no system message: the small models ignored it
    assert user.user_suffix == "\n\n" + INSTRUCTION
    assert user.as_dict() == {
        "name": "answer-only-user",
        "system_message": None,
        "max_tokens": 16,
        "user_suffix": "\n\n" + INSTRUCTION,
    }
    body = user.body("m", SENTINEL_PROMPT, {"min_quality": 0.4})
    assert body["messages"] == [{"role": "user", "content": SENTINEL_PROMPT + "\n\n" + INSTRUCTION}]
    assert body["max_tokens"] == 16 and body["temperature"] == 0 and body["stream"] is True
    assert body["routepilot"] == {"min_quality": 0.4}
    assert user.messages(SENTINEL_PROMPT) == body["messages"]
    # the two earlier protocols have no suffix and keep the form they were recorded in
    for name in ("as-written", "answer-only"):
        old = lc.protocol_for(name)
        assert old.user_suffix is None and "user_suffix" not in old.as_dict()
        assert old.messages(SENTINEL_PROMPT) == old.body("m", SENTINEL_PROMPT)["messages"]
    assert lc.protocol_for("answer-only").messages("p") == [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": "p"},
    ]
    assert lc.protocol_for("as-written").messages("p") == [{"role": "user", "content": "p"}]


def test_the_instruction_in_the_user_turn_reaches_the_server(stub: int) -> None:
    body = lc.protocol_for("answer-only-user").body("m", SENTINEL_PROMPT)
    assert lc.stream_chat(stub, body, key="k", timeout_s=10).ok
    sent = Stub.seen[0]["body"]
    assert sent["messages"] == [{"role": "user", "content": SENTINEL_PROMPT + "\n\n" + INSTRUCTION}]
    assert sent["max_tokens"] == 16


def test_the_cli_offers_every_protocol(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        lc.main(["collect", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert all(name in help_text for name in lc.PROTOCOLS)
    with pytest.raises(SystemExit) as bad:
        lc.main(["collect", "--protocol", "verbose"])
    assert bad.value.code == 2


def test_a_protocol_builds_the_request_body() -> None:
    only = lc.protocol_for("answer-only").body("m", SENTINEL_PROMPT, {"min_quality": 0.4})
    assert only["messages"] == [
        {"role": "system", "content": "Reply with only the letter of the correct option."},
        {"role": "user", "content": SENTINEL_PROMPT},
    ]
    assert only["max_tokens"] == 16 and only["temperature"] == 0 and only["stream"] is True
    assert only["stream_options"] == {"include_usage": True}
    assert only["routepilot"] == {"min_quality": 0.4}
    written = lc.protocol_for("as-written").body("m", SENTINEL_PROMPT)
    assert written["messages"] == [{"role": "user", "content": SENTINEL_PROMPT}]
    assert written["max_tokens"] == 384 and "routepilot" not in written
    assert written == lc.chat_body("m", SENTINEL_PROMPT, 384)  # unchanged from the first collection


def test_the_system_message_reaches_the_server(stub: int) -> None:
    body = lc.protocol_for("answer-only").body("m", SENTINEL_PROMPT)
    assert lc.stream_chat(stub, body, key="k", timeout_s=10).ok
    sent = Stub.seen[0]["body"]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][0]["content"] == lc.ANSWER_ONLY_SYSTEM
    assert sent["max_tokens"] == 16


def test_the_planned_request_time_uses_the_protocols_max_tokens() -> None:
    measured = {"ttft_ms_mean": 50.0, "tokens_per_s_mean": 100.0, "worst_case_request_s": 3.89}
    written = lc.planned_worst_case_s(measured, lc.protocol_for("as-written"))
    only = lc.planned_worst_case_s(measured, lc.protocol_for("answer-only"))
    assert written == pytest.approx(0.05 + 3.84)
    assert only == pytest.approx(0.05 + 0.16)
    # with the short cap, far more items fit the same budget: 1800 / 3.89 = 462 items, 198 of
    # them gpqa, against the 600 cap
    fits_written = lc.plan_sample_size({"a": written}, n_gpqa=198, budget_s=1800, cap=600)
    fits_only = lc.plan_sample_size({"a": only}, n_gpqa=198, budget_s=1800, cap=600)
    assert (fits_written, fits_only) == (264, 600)


def test_a_protocol_file_round_trips_and_a_missing_one_means_as_written(tmp_path: Path) -> None:
    assert lc.load_protocol(tmp_path) == lc.protocol_for("as-written")  # the first collection
    for name in ("answer-only", "answer-only-user"):
        lc.write_protocol(tmp_path, lc.protocol_for(name))
        text = (tmp_path / "protocol.json").read_text(encoding="utf-8")
        assert "\r" not in text and EM_DASH not in text
        assert lc.load_protocol(tmp_path) == lc.protocol_for(name)
    # a file written before the user suffix existed (no such key) still loads
    old = {"name": "answer-only", "system_message": INSTRUCTION, "max_tokens": 16}
    (tmp_path / "protocol.json").write_text(json.dumps(old), encoding="utf-8")
    assert lc.load_protocol(tmp_path) == lc.protocol_for("answer-only")


def test_a_protocol_file_that_disagrees_with_its_name_is_refused(tmp_path: Path) -> None:
    doc = lc.protocol_for("answer-only").as_dict() | {"max_tokens": 99}
    (tmp_path / "protocol.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="answer-only"):
        lc.load_protocol(tmp_path)
    (tmp_path / "protocol.json").write_text(json.dumps({"name": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        lc.load_protocol(tmp_path)
    changed = lc.protocol_for("answer-only-user").as_dict() | {"user_suffix": "\n\nAnswer."}
    (tmp_path / "protocol.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="answer-only-user"):
        lc.load_protocol(tmp_path)


def prompts_frame(n_gpqa: int = 6, n_mmlu: int = 40) -> pd.DataFrame:
    rows = []
    for dataset, n in (("gpqa", n_gpqa), ("mmlupro", n_mmlu)):
        for i in range(n):
            rows.append({"dataset": dataset, "item_id": f"{dataset}/{i}"})
    return pd.DataFrame(rows)


def test_choose_items_takes_all_gpqa_and_a_seeded_mmlupro_sample() -> None:
    import numpy as np

    prompts = prompts_frame()
    items = lc.choose_items(prompts, 10)
    assert len(items) == 16
    assert items["item_id"].is_unique
    assert items.loc[items["dataset"] == "gpqa", "item_id"].nunique() == 6
    sample = items.loc[items["dataset"] == "mmlupro", "item_id"]
    # the sample is default_rng(0).choice over the sorted item IDs
    pool = sorted(prompts.loc[prompts["dataset"] == "mmlupro", "item_id"])
    want = np.random.default_rng(0).choice(pool, size=10, replace=False)
    assert sorted(sample) == sorted(want.tolist())
    assert lc.choose_items(prompts, 10).equals(items)  # deterministic
    assert not lc.choose_items(prompts, 10, seed=1).equals(items)


def test_choose_items_mixes_the_datasets_so_that_a_prefix_has_both() -> None:
    items = lc.choose_items(prompts_frame(60, 300), 200)
    first = items.head(100)
    assert {"gpqa", "mmlupro"} == set(first["dataset"])
    assert first["dataset"].eq("gpqa").mean() == pytest.approx(60 / 260, abs=0.15)


def test_choose_items_rejects_a_sample_larger_than_the_pool() -> None:
    with pytest.raises(ValueError, match="between 0 and 40"):
        lc.choose_items(prompts_frame(), 41)
    assert len(lc.choose_items(prompts_frame(), 0)) == 6


# --- ground truth from the tarball --------------------------------------------------------------


def add_json(tar: tarfile.TarFile, name: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def result(dataset: str, split: str, model: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"dataset_name": dataset, "split": split, "model_name": model, "records": records}


def rec(index: int, prompt: str, truth: str) -> dict[str, Any]:
    return {"index": index, "prompt": prompt, "ground_truth": truth, "raw_output": "o"}


def test_read_ground_truth_votes_over_the_records_with_the_item_prompt(tmp_path: Path) -> None:
    p0, p1, p5 = "prompt zero", "prompt one", "prompt five"
    tar_path = tmp_path / "release.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for model in ("a", "b"):
            add_json(
                tar,
                f"bench-release/gpqa/{model}/gpqa-{model}-1.json",
                result("gpqa", "test", model, [rec(0, p0, "B"), rec(1, p1, "C")]),
            )
        # a model that numbered another prompt as item 1 is left out, and counted
        add_json(
            tar,
            "bench-release/gpqa/c/gpqa-c-1.json",
            result("gpqa", "test", "c", [rec(1, "a different prompt", "D")]),
        )
        # one record disagrees on item 0
        add_json(
            tar,
            "bench-release/gpqa/d/gpqa-d-1.json",
            result("gpqa", "test", "d", [rec(0, p0, " c ")]),
        )
        add_json(
            tar,
            "bench-release/mmlupro/test_3000/a/mmlupro-test_3000-a-1.json",
            result("mmlupro", "test_3000", "a", [rec(5, p5, "J")]),
        )
        # a split that is not the canonical one, and a dataset nobody asked for
        add_json(
            tar,
            "bench-release/mmlupro/test_1000/a/mmlupro-test_1000-a-1.json",
            result("mmlupro", "test_1000", "a", [rec(5, p5, "A")]),
        )
        add_json(
            tar,
            "bench-release/aime/a/aime-a-1.json",
            result("aime", "hybrid", "a", [rec(0, p0, "Z")]),
        )
    items = pd.DataFrame(
        {
            "item_id": ["gpqa/0", "gpqa/1", "mmlupro/5"],
            "prompt_sha256": [sha(p0), sha(p1), sha(p5)],
        }
    )
    truth = lc.read_ground_truth(tar_path, items).set_index("item_id")
    assert truth.loc["gpqa/0", "ground_truth"] == "B"
    assert (truth.loc["gpqa/0", "n_records"], truth.loc["gpqa/0", "n_disagree"]) == (3, 1)
    assert truth.loc["gpqa/1", "ground_truth"] == "C"
    assert (truth.loc["gpqa/1", "n_records"], truth.loc["gpqa/1", "n_other_prompt"]) == (2, 1)
    assert truth.loc["mmlupro/5", "ground_truth"] == "J"
    assert truth.loc["mmlupro/5", "n_records"] == 1


def test_read_ground_truth_reports_an_item_nobody_voted_for(tmp_path: Path) -> None:
    tar_path = tmp_path / "release.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        add_json(
            tar,
            "bench-release/gpqa/a/gpqa-a-1.json",
            result("gpqa", "test", "a", [rec(0, "other", "B")]),
        )
    items = pd.DataFrame({"item_id": ["gpqa/0"], "prompt_sha256": [sha("mine")]})
    row = lc.read_ground_truth(tar_path, items).iloc[0]
    assert (row["ground_truth"], row["n_records"], row["n_other_prompt"]) == ("", 0, 1)


# --- the streaming client ------------------------------------------------------------------------


class Stub(BaseHTTPRequestHandler):
    """A loopback chat endpoint that streams a canned answer; tests set ``mode``."""

    protocol_version = "HTTP/1.1"
    mode = "ok"
    seen: list[dict[str, Any]] = []

    def log_message(self, *args: Any) -> None:  # keep the test output quiet
        pass

    def _chunk(self, text: str) -> None:
        data = text.encode("utf-8")
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(size))
        Stub.seen.append({"body": body, "auth": self.headers.get("Authorization")})
        self.close_connection = True  # one request per connection: nothing to read afterwards
        if Stub.mode == "429":
            payload = json.dumps(
                {"error": {"message": "slow down", "type": "rate_limit", "code": "rate_limited"}}
            ).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.send_header("x-routepilot-model", "m-small")
        self.send_header("x-routepilot-decision-id", "dec_abc")
        self.send_header("x-other", "ignored")
        self.end_headers()
        frames = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"index": 0, "delta": {"content": "Hello"}}]},
            {"choices": [{"index": 0, "delta": {"content": " world"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
        ]
        if Stub.mode == "ok":
            frames.append({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        for frame in frames:
            self._chunk(f"data: {json.dumps(frame)}\n\n")
        self._chunk("data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")


@pytest.fixture
def stub() -> Iterator[int]:
    Stub.mode = "ok"
    Stub.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_stream_chat_collects_text_usage_and_headers(stub: int) -> None:
    body = lc.chat_body("m-small", SENTINEL_PROMPT, 384, {"min_quality": 0.4})
    chat = lc.stream_chat(stub, body, key="k-1", timeout_s=10)
    assert chat.ok and chat.status == 200 and chat.error == ""
    assert chat.text == "Hello world"
    assert chat.finish_reason == "length"
    assert (chat.prompt_tokens, chat.completion_tokens) == (7, 3)
    assert chat.ttft_ms is not None and 0 <= chat.ttft_ms <= chat.wall_ms
    assert dict(chat.headers) == {
        "x-routepilot-model": "m-small",
        "x-routepilot-decision-id": "dec_abc",
    }
    sent = Stub.seen[0]
    assert sent["auth"] == "Bearer k-1"
    assert sent["body"]["stream"] is True
    assert sent["body"]["stream_options"] == {"include_usage": True}
    assert sent["body"]["temperature"] == 0
    assert sent["body"]["max_tokens"] == 384
    assert sent["body"]["routepilot"] == {"min_quality": 0.4}


def test_stream_chat_turns_an_http_error_into_a_result(stub: int) -> None:
    Stub.mode = "429"
    chat = lc.stream_chat(stub, lc.chat_body("m", "p", 8), key="k", timeout_s=10)
    assert not chat.ok and chat.status == 429 and chat.text == ""
    assert "429" in chat.error and "rate_limited" in chat.error


def test_stream_chat_needs_a_usage_report(stub: int) -> None:
    Stub.mode = "no-usage"
    chat = lc.stream_chat(stub, lc.chat_body("m", "p", 8), key="k", timeout_s=10)
    assert chat.status == 200 and chat.text == "Hello world"
    assert not chat.ok and "usage" in chat.error


def test_stream_chat_reports_a_refused_connection() -> None:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    chat = lc.stream_chat(port, lc.chat_body("m", "p", 8), timeout_s=2)
    assert not chat.ok and chat.status == 0 and chat.error


def test_the_error_text_never_holds_the_request(stub: int) -> None:
    Stub.mode = "429"
    chat = lc.stream_chat(stub, lc.chat_body("m", SENTINEL_PROMPT, 8), key="k", timeout_s=10)
    assert SENTINEL_PROMPT not in chat.error


def test_tokens_per_second_counts_the_tokens_after_the_first() -> None:
    chat = lc.Chat(status=200, completion_tokens=101, ttft_ms=100.0, wall_ms=1100.0)
    assert lc.tokens_per_second(chat) == pytest.approx(100.0)
    assert (
        lc.tokens_per_second(lc.Chat(status=200, completion_tokens=1, ttft_ms=1, wall_ms=2)) is None
    )
    assert lc.tokens_per_second(lc.Chat(status=200, completion_tokens=5)) is None


# --- grading, tables, report ---------------------------------------------------------------------


def catalog() -> list[lc.CatalogModel]:
    return [
        lc.CatalogModel("small", "m:0.5b", 0.5, Decimal("0.01"), Decimal("0.03")),
        lc.CatalogModel("big", "m:3b", 3.0, Decimal("0.06"), Decimal("0.18")),
    ]


def synthetic_prompts() -> pd.DataFrame:
    rows = []
    for item_id, letters in (("gpqa/0", "ABCD"), ("gpqa/1", "ABCD"), ("mmlupro/2", "ABCDEFGHIJ")):
        options = "\n".join(f"{c}) option {c}" for c in letters)
        prompt = f"{SENTINEL_PROMPT} {item_id}\n{options}\nThink, then answer."
        rows.append(
            {
                "dataset": item_id.split("/")[0],
                "item_id": item_id,
                "prompt": prompt,
                "prompt_sha256": sha(prompt),
                "prompt_bytes": len(prompt.encode()),
            }
        )
    return pd.DataFrame(rows)


def synthetic_raw() -> pd.DataFrame:
    def row(item: str, model: str, text: str, ok: bool = True, tokens: tuple[int, int] = (50, 80)):
        return {
            "item_id": item,
            "model": model,
            "decision_id": f"dec_{item}_{model}".replace("/", "_"),
            "text": text,
            "finish_reason": "stop",
            "prompt_tokens": tokens[0],
            "completion_tokens": tokens[1],
            "ok": ok,
            "error": "" if ok else "http 500: boom",
            "client_wall_ms": 1000.0,
        }

    return pd.DataFrame(
        [
            row("gpqa/0", "small", f"{SENTINEL_ANSWER} The answer is B."),
            row("gpqa/0", "big", "Answer: B"),
            row("gpqa/1", "small", "I think C.", tokens=(60, 10)),
            row("gpqa/1", "big", "", ok=False),
            row("mmlupro/2", "small", "The answer is J."),
            row("mmlupro/2", "big", "Answer: E"),
        ]
    )


def synthetic_truth() -> pd.DataFrame:
    return pd.DataFrame(
        {"item_id": ["gpqa/0", "gpqa/1", "mmlupro/2"], "ground_truth": ["B", "C", "J"]}
    )


def synthetic_decisions(raw: pd.DataFrame) -> pd.DataFrame:
    ok = raw["decision_id"] != "dec_gpqa_1_big"
    return pd.DataFrame(
        {
            "decision_id": raw.loc[ok, "decision_id"].to_numpy(),
            "status": pd.array([200] * int(ok.sum()), dtype="Int64"),
            "ttft_ms": pd.array([40] * int(ok.sum()), dtype="Int64"),
            "latency_ms": pd.array([1500 + 100 * i for i in range(int(ok.sum()))], dtype="Int64"),
        }
    )


def test_grade_requests_scores_against_the_offered_letters() -> None:
    raw = synthetic_raw()
    graded = lc.grade_requests(raw.loc[raw["ok"]], synthetic_prompts(), synthetic_truth())
    got = dict(
        zip(zip(graded["item_id"], graded["model"], strict=True), graded["score"], strict=True)
    )
    assert got[("gpqa/0", "small")] == 1.0
    assert got[("gpqa/0", "big")] == 1.0
    assert got[("gpqa/1", "small")] == 1.0  # no cue: the last standalone letter
    assert got[("mmlupro/2", "small")] == 1.0  # J is offered by the 10-option prompt
    assert got[("mmlupro/2", "big")] == 0.0
    how = dict(
        zip(zip(graded["item_id"], graded["model"], strict=True), graded["how"], strict=True)
    )
    assert how[("gpqa/0", "big")] == "cue"
    assert how[("gpqa/1", "small")] == "last_letter"


def test_grade_requests_refuses_a_truth_outside_the_offered_letters() -> None:
    raw = synthetic_raw()
    truth = synthetic_truth()
    truth.loc[truth["item_id"] == "gpqa/0", "ground_truth"] = "H"
    with pytest.raises(ValueError, match="gpqa/0"):
        lc.grade_requests(raw.loc[raw["ok"]], synthetic_prompts(), truth)


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = synthetic_raw()
    graded = lc.grade_requests(raw.loc[raw["ok"]], synthetic_prompts(), synthetic_truth())
    raw = raw.merge(
        graded[["item_id", "model", "score", "how"]], on=["item_id", "model"], how="left"
    )
    outcomes, measured = lc.build_tables(
        raw, synthetic_decisions(raw), synthetic_prompts(), catalog()
    )
    return raw, outcomes, measured


def test_build_tables_follows_the_shared_outcomes_contract() -> None:
    _, outcomes, measured = build()
    assert list(outcomes.columns) == [
        "dataset", "item_id", "model", "score", "success",
        "prompt_tokens", "completion_tokens", "cost_usd", "prompt_sha256",
    ]  # fmt: skip
    # the failed request has no outcome, so the item is incomplete for the pool
    assert len(outcomes) == 5
    assert ("gpqa/1", "big") not in set(zip(outcomes["item_id"], outcomes["model"], strict=True))
    one = outcomes.loc[(outcomes["item_id"] == "gpqa/0") & (outcomes["model"] == "small")].iloc[0]
    assert one["dataset"] == "gpqa" and bool(one["success"]) and one["score"] == 1.0
    assert one["cost_usd"] == pytest.approx((0.01 * 50 + 0.03 * 80) / 1e6)
    assert one["prompt_sha256"] == sha(
        synthetic_prompts().set_index("item_id").loc["gpqa/0", "prompt"]
    )
    assert outcomes.dtypes["prompt_tokens"] == "int64"
    # the latency comes from the decision log
    assert set(measured.columns) >= {"item_id", "model", "ttft_ms", "e2e_ms", "decision_id"}
    assert (measured["ttft_ms"] == 40).all()
    assert measured["e2e_ms"].min() >= 1500


def test_a_request_without_a_decision_record_is_not_an_outcome() -> None:
    raw = synthetic_raw()
    graded = lc.grade_requests(raw.loc[raw["ok"]], synthetic_prompts(), synthetic_truth())
    raw = raw.merge(
        graded[["item_id", "model", "score", "how"]], on=["item_id", "model"], how="left"
    )
    decisions = synthetic_decisions(raw)
    decisions = decisions.loc[decisions["decision_id"] != "dec_gpqa_0_small"]
    outcomes, _ = lc.build_tables(raw, decisions, synthetic_prompts(), catalog())
    assert ("gpqa/0", "small") not in set(zip(outcomes["item_id"], outcomes["model"], strict=True))


def test_summarize_collection_counts_the_realized_sample() -> None:
    raw, outcomes, measured = build()
    summary = lc.summarize_collection(raw, outcomes, measured, catalog())
    assert summary["realized_items"] == 2  # gpqa/0 and mmlupro/2; gpqa/1 lacks "big"
    assert summary["realized_by_dataset"] == {"gpqa": 1, "mmlupro": 1}
    small, big = summary["per_model"]["small"], summary["per_model"]["big"]
    assert (small["requests"], small["ok"], small["failed"]) == (3, 3, 0)
    assert (big["requests"], big["ok"], big["failed"]) == (3, 2, 1)
    assert big["failure_kinds"] == {"http 500": 1}
    assert small["accuracy"] == 1.0 and big["accuracy"] == 0.5
    assert small["graded_by"] == {"cue": 2, "last_letter": 1}
    assert small["ttft_ms"]["p50"] == 40


def test_the_report_tables_hold_no_prompt_or_answer_text() -> None:
    raw, outcomes, measured = build()
    summary = lc.summarize_collection(raw, outcomes, measured, catalog())
    results = {
        "plan": {
            "budget_minutes": 90, "planned_mmlupro": 5, "used_mmlupro": 5,
            "cap": 600, "gpqa_items": 2, "max_tokens": 16,
        },
        "protocol": lc.protocol_for("answer-only").as_dict(),
        "measure_run": "run-1",
        "stopped_by_budget": False,
        "request_time_s": 1.0,
        "wall_time_s": 2.0,
        "decisions_missing": 0,
        **summary,
    }  # fmt: skip
    tables = lc._collect_tables(results, catalog())
    text = json.dumps(results, default=str) + "\n".join(t.to_csv() for t in tables.values())
    assert SENTINEL_PROMPT not in text and SENTINEL_ANSWER not in text
    assert EM_DASH not in text
    setup = dict(tables["Setup"].itertuples(index=False, name=None))
    assert setup["protocol"] == "answer-only"
    assert setup["system message"] == "Reply with only the letter of the correct option."
    assert setup["max_tokens"] == 16
    assert setup["user message suffix"] == "none"
    # the appended instruction is shown on one line, with its line breaks escaped
    results["protocol"] = lc.protocol_for("answer-only-user").as_dict()
    user_setup = dict(
        lc._collect_tables(results, catalog())["Setup"].itertuples(index=False, name=None)
    )
    assert user_setup["protocol"] == "answer-only-user"
    assert user_setup["system message"] == "none"
    assert user_setup["user message suffix"] == json.dumps("\n\n" + INSTRUCTION)
    assert "\n" not in user_setup["user message suffix"]


def test_wait_for_decisions_returns_the_ids_still_missing(tmp_path: Path) -> None:
    log = tmp_path / "decisions-collect.jsonl"
    rotated = tmp_path / "decisions-collect.20261029T000000Z.jsonl"
    rotated.write_text(json.dumps({"type": "decision", "decision_id": "dec_1"}) + "\n")
    log.write_text(
        json.dumps({"type": "decision", "decision_id": "dec_2"})
        + "\n"
        + json.dumps({"type": "feedback", "decision_id": "dec_9"})
        + "\n"
        + '{"type": "decision", "decision_id": "dec_3"'  # a torn last line
    )
    missing = lc.wait_for_decisions(log, ["dec_1", "dec_2", "dec_3", "dec_9"], timeout_s=0.3)
    assert missing == {"dec_3", "dec_9"}
    assert lc.wait_for_decisions(log, ["dec_1", "dec_2"], timeout_s=0.3) == set()


def test_latest_measure_picks_the_newest_measure_run(tmp_path: Path) -> None:
    for name, phase in (("20261029T010000Z-a", "measure"), ("20261029T020000Z-b", "collect"),
                        ("20261029T030000Z-c", "measure")):  # fmt: skip
        (tmp_path / name).mkdir()
        (tmp_path / name / "results.json").write_text(json.dumps({"phase": phase}))
    assert lc.latest_measure(tmp_path).name == "20261029T030000Z-c"
    with pytest.raises(ValueError, match="no measure run"):
        lc.latest_measure(tmp_path / "empty")


def test_the_invented_measurement_prompts_have_valid_answers() -> None:
    from routepilot_train import grading

    assert len(lc.MEASURE_PROMPTS) == 3
    for prompt, truth in lc.MEASURE_PROMPTS:
        assert grading.offered_letters(prompt) == "ABCD"
        assert truth in "ABCD"
