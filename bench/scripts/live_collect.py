"""The live local experiment, part 1: measure the candidate Ollama models, then collect outcomes.

    python bench/scripts/live_collect.py measure
    python bench/scripts/live_collect.py collect

``measure`` sends three invented multiple-choice prompts to each candidate model directly (one
request at a time, after a warm-up request that is not counted), and records the output speed
(tokens per second), the time to first token and whether the answers were right. Nothing
else is read from it but the speeds: the models for the experiment are chosen from them and
named in ``bench/configs/live-ollama.yaml``.

``collect`` sends the ``gpqa`` items (all 198) and a seeded sample of ``mmlupro`` to every model
of that config through ``routepilot serve`` (the model pinned per request, streaming with usage,
temperature 0, the decision log on, one request at a time, grouped by model, one warm-up request
per model that is not counted). How an item is put to a model is the protocol (``--protocol``):
``as-written`` sends the dataset prompt alone with ``max_tokens`` 384 (the first collection);
``answer-only`` prepends the system message "Reply with only the letter of the correct option."
and allows 16 tokens; ``answer-only-user`` sends no system message and appends that instruction
to the end of the user message (after a blank line), also with 16 tokens. The protocol is
recorded in ``<live-dir>/protocol.json``, ``results.json``,
the report and ``run.json``; training and the demo read it from the live directory. The size of
the sample is chosen from the measured speeds, at the protocol's ``max_tokens``, so that the
whole collection fits the time budget (90 minutes by default); the script also stops at the
budget and reports the sample it realized (the items every model answered). Answers are graded
by ``routepilot_train.grading`` against the ``ground_truth`` of the
release tarball, which ingest drops. Time to first token and end-to-end time come from the
gateway's decision log, tokens from the usage the model reports.

The prompts, the models' outputs and the decision log stay in the gitignored ``data/derived/live``
(the outcomes there use the shared ``outcomes.parquet`` contract, so the training pipeline reads
them). What is written under ``bench/results/live_collect/<run-id>`` holds counts, rates,
latencies and item IDs only.

Prices are an ASSUMPTION (Ruling V1-R42): per-token prices proportional to the parameter count,
0.02 USD per million input tokens and 0.06 per million output tokens for each billion
parameters. They are not the price of anything.
"""

import argparse
import contextlib
import hashlib
import http.client
import json
import math
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from routepilot_train import grading, replaytool, runinfo
from routepilot_train.data import selection
from routepilot_train.evaluate import report
from routepilot_train.logs import load_decisions

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overhead  # noqa: E402  (a sibling script, not a package)

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "data" / "derived" / "live"
CONFIG = ROOT / "bench" / "configs" / "live-ollama.yaml"
RESULTS_ROOT = ROOT / "bench" / "results" / "live_collect"
TARBALL = ROOT / "data" / "raw" / "bench-release.tar.gz"
EXPERIMENT = "live_collect"
OLLAMA = "http://127.0.0.1:11434"
KEY = "live-ollama-key-local"  # a fake local key; the config holds only its SHA-256
POOL = "live-ollama"

CANDIDATES = ("qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "llama3.2:3b")
GPQA, MMLUPRO = "gpqa", "mmlupro"
DEFAULT_LETTERS = {GPQA: "ABCD", MMLUPRO: "ABCDEFGHIJ"}
MAX_TOKENS = 384
MEASURE_MAX_TOKENS = 256
# The two ways to put an item to a model (see Protocol). The system message is part of the
# protocol and is sent word for word.
AS_WRITTEN, ANSWER_ONLY, ANSWER_ONLY_USER = "as-written", "answer-only", "answer-only-user"
PROTOCOLS = (AS_WRITTEN, ANSWER_ONLY, ANSWER_ONLY_USER)
ANSWER_ONLY_SYSTEM = "Reply with only the letter of the correct option."
# answer-only-user (Ruling V1-R44): the same instruction, appended to the end of the user message
ANSWER_ONLY_USER_SUFFIX = "\n\n" + ANSWER_ONLY_SYSTEM
ANSWER_ONLY_MAX_TOKENS = 16
PROTOCOL_FILE = "protocol.json"
BUDGET_MINUTES = 90.0
MMLUPRO_CAP = 600
SEED = 0
# Input and output price per million tokens for each billion parameters (ASSUMPTION, V1-R42).
PRICE_IN_PER_B, PRICE_OUT_PER_B = Decimal("0.02"), Decimal("0.06")
MAX_CONSECUTIVE_FAILURES = 20
DECISION_WAIT_S = 60.0
WARMUP_PROMPT = "Reply with the single word ready."

# Invented multiple-choice questions with known answers, in the layout of the real prompts.
MEASURE_PROMPTS: tuple[tuple[str, str], ...] = (
    (
        "Which planet is known as the Red Planet?\nA) Venus\nB) Mars\nC) Jupiter\nD) Mercury\n"
        "Think step by step, then finish with a line 'Answer: <letter>'.",
        "B",
    ),
    (
        "What is 12 multiplied by 7?\nA) 74\nB) 82\nC) 84\nD) 96\n"
        "Think step by step, then finish with a line 'Answer: <letter>'.",
        "C",
    ),
    (
        "Which gas do plants absorb from the air for photosynthesis?\nA) Oxygen\nB) Nitrogen\n"
        "C) Carbon dioxide\nD) Helium\n"
        "Think step by step, then finish with a line 'Answer: <letter>'.",
        "C",
    ),
)

LABELS = (
    "latency and token counts are measured on this machine through RoutePilot (decision log)",
    "prices are an ASSUMPTION (Ruling V1-R42): proportional to the parameter count, "
    "0.02 USD per million input tokens and 0.06 per million output tokens per billion parameters",
    "answers are graded by the last standalone option letter after an answer cue, else the last "
    "standalone option letter, against the ground truth of the release",
    "no prompt text and no model output is in this report",
)


MEASURE_LABELS = (
    "speeds and times are measured on this machine by sending requests straight to Ollama, "
    "not through RoutePilot",
    "the three measurement prompts are invented, not dataset items",
    "no prompt text and no model output is in this report",
)


# --- catalog and prices ------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogModel:
    """One model of the live config: its public id, Ollama tag, size and assumed prices."""

    id: str
    tag: str
    params_b: float
    in_usd_per_mtok: Decimal
    out_usd_per_mtok: Decimal

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Assumed cost in USD of a request with the reported token counts."""
        total = (
            self.in_usd_per_mtok * prompt_tokens + self.out_usd_per_mtok * completion_tokens
        ) / Decimal(1_000_000)
        return float(total)


def params_of(tag: str) -> float:
    """Billions of parameters named by an Ollama tag such as ``qwen2.5:1.5b`` (1.5)."""
    size = tag.rsplit(":", 1)[-1].lower()
    if not size.endswith("b"):
        raise ValueError(f"cannot read a parameter count from the tag {tag!r}")
    try:
        value = float(size[:-1])
    except ValueError as err:
        raise ValueError(f"cannot read a parameter count from the tag {tag!r}") from err
    if not value > 0:
        raise ValueError(f"cannot read a parameter count from the tag {tag!r}")
    return value


def assumed_prices(params_b: float) -> tuple[Decimal, Decimal]:
    """The ASSUMPTION prices in USD per million tokens: (input, output) for ``params_b``."""
    scale = Decimal(str(params_b))
    return PRICE_IN_PER_B * scale, PRICE_OUT_PER_B * scale


def model_id(tag: str) -> str:
    """The public model id of an Ollama tag: the colon becomes a dash."""
    return tag.replace(":", "-")


def load_catalog(config: Path = CONFIG) -> list[CatalogModel]:
    """The models of a live config, smallest first. Prices are read from the file."""
    doc = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    out = []
    for m in doc["models"]:
        deployments = m["deployments"]
        if len(deployments) != 1:
            raise ValueError(f"model {m['id']} must have exactly one deployment")
        tag = deployments[0]["upstream_model"]
        out.append(
            CatalogModel(
                id=m["id"],
                tag=tag,
                params_b=params_of(tag),
                in_usd_per_mtok=Decimal(str(m["input_price_per_mtok"])),
                out_usd_per_mtok=Decimal(str(m["output_price_per_mtok"])),
            )
        )
    return sorted(out, key=lambda c: (c.params_b, c.id))


def render_config(
    base: Mapping[str, Any], port: int, decision_log: Path, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """A copy of the committed config that listens on ``port`` and logs to ``decision_log``.

    ``extra`` maps a top-level section to values merged into it (for the demo's policies).
    The decision log path is absolute, so the copy can live anywhere.
    """
    cfg = json.loads(json.dumps(base))
    cfg["server"]["listen"] = f"127.0.0.1:{port}"
    cfg["decision_log"] = {"path": decision_log.as_posix()}
    for section, values in (extra or {}).items():
        if isinstance(values, Mapping) and isinstance(cfg.get(section), dict):
            cfg[section] = {**cfg[section], **values}
        else:
            cfg[section] = json.loads(json.dumps(values))
    return cfg


# --- talking to Ollama and to the gateway -------------------------------------------------------


def ollama_json(path: str, body: Mapping[str, Any] | None = None, timeout: float = 600.0) -> Any:
    """GET (no body) or POST ``body`` to the local Ollama and decode the JSON answer."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA + path, data=data, headers={"Content-Type": "application/json"}
    )
    with overhead.OPENER.open(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_inventory() -> dict[str, Any]:
    """The Ollama version and, per pulled model, its digest and size as Ollama reports them."""
    models = {}
    for m in ollama_json("/api/tags")["models"]:
        details = m.get("details", {})
        models[m["name"]] = {
            "digest": m.get("digest"),
            "size_bytes": m.get("size"),
            "parameter_size": details.get("parameter_size"),
            "family": details.get("family"),
            "quantization": details.get("quantization_level"),
        }
    return {"version": ollama_json("/api/version")["version"], "models": models}


def ollama_loaded() -> list[dict[str, Any]]:
    """The models loaded now, with how much of each sits in video memory."""
    out = []
    for m in ollama_json("/api/ps").get("models", []):
        out.append(
            {
                "name": m.get("name"),
                "size_bytes": m.get("size"),
                "size_vram_bytes": m.get("size_vram"),
                "context_length": m.get("context_length"),
            }
        )
    return out


def ollama_load(tag: str, keep_alive: str | int = "60m") -> None:
    """Load a model into memory (an empty generate request) and keep it for ``keep_alive``."""
    ollama_json("/api/generate", {"model": tag, "keep_alive": keep_alive})


def ollama_unload(tag: str) -> None:
    """Free a model's memory now."""
    ollama_json("/api/generate", {"model": tag, "keep_alive": 0})


@dataclass(frozen=True)
class Chat:
    """One streamed chat request: the answer, the usage and the client-side timings."""

    status: int
    text: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usage_seen: bool = False
    ttft_ms: float | None = None  # client side: request sent to the first content chunk
    wall_ms: float = 0.0  # client side: request sent to the end of the stream
    headers: Mapping[str, str] = field(default_factory=dict)  # the x-routepilot-* headers
    error: str = ""

    @property
    def ok(self) -> bool:
        """The request ended with a complete stream and a usage report."""
        return self.status == 200 and not self.error and self.usage_seen


def stream_chat(
    port: int,
    body: Mapping[str, Any],
    *,
    key: str | None = None,
    timeout_s: float = 300.0,
    path: str = "/v1/chat/completions",
) -> Chat:
    """POST a streaming chat request to ``127.0.0.1:port`` and read it to the end.

    The text, the finish reason and the usage are collected from the chunks. A transport
    problem or an HTTP error ends in a ``Chat`` with ``error`` set instead of an exception;
    the error text never holds the request.
    """
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = f"Bearer {key}"
    started = time.perf_counter()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    pieces: list[str] = []
    finish: str | None = None
    usage: Mapping[str, Any] | None = None
    first: float | None = None
    status = 0
    route_headers: dict[str, str] = {}
    error = ""
    try:
        conn.request("POST", path, body=json.dumps(body).encode("utf-8"), headers=headers)
        resp = conn.getresponse()
        status = resp.status
        route_headers = {
            k.lower(): v for k, v in resp.getheaders() if k.lower().startswith("x-routepilot-")
        }
        if status != 200:
            error = f"http {status}: {_error_text(resp.read(2000))}"
        else:
            while True:
                line = resp.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        pieces.append(content)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    except (OSError, http.client.HTTPException, ValueError) as err:
        error = f"{type(err).__name__}: {str(err)[:200]}"
    finally:
        conn.close()
    ended = time.perf_counter()
    if status == 200 and not error and usage is None:
        error = "the stream ended without a usage report"
    return Chat(
        status=status,
        text="".join(pieces),
        finish_reason=finish,
        prompt_tokens=int((usage or {}).get("prompt_tokens") or 0),
        completion_tokens=int((usage or {}).get("completion_tokens") or 0),
        usage_seen=usage is not None,
        ttft_ms=None if first is None else (first - started) * 1000.0,
        wall_ms=(ended - started) * 1000.0,
        headers=route_headers,
        error=error,
    )


def _error_text(raw: bytes) -> str:
    """The code and message of an OpenAI-style error body, cut short."""
    try:
        err = json.loads(raw.decode("utf-8", errors="replace"))["error"]
        return f"{err.get('code') or err.get('type')}: {str(err.get('message'))[:150]}"
    except (ValueError, KeyError, TypeError, AttributeError):
        return "unreadable error body"


def request_messages(
    prompt: str, system: str | None = None, user_suffix: str | None = None
) -> list[dict[str, str]]:
    """The messages of a request: an optional system message, then the user prompt.

    ``user_suffix``, when given, is appended to the end of the user message. This is the one
    place the messages are built, so the collector, the trainer and the demo agree on them.
    """
    content = prompt if user_suffix is None else prompt + user_suffix
    messages = [{"role": "user", "content": content}]
    if system is not None:
        messages.insert(0, {"role": "system", "content": system})
    return messages


def chat_body(
    model: str,
    prompt: str,
    max_tokens: int,
    routepilot: Mapping[str, Any] | None = None,
    system: str | None = None,
    user_suffix: str | None = None,
) -> dict[str, Any]:
    """The request body: temperature 0, streaming with usage, ``max_tokens`` capped.

    ``system``, when given, is sent as a system message before the prompt; ``user_suffix``,
    when given, is appended to the user message (see ``request_messages``).
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": request_messages(prompt, system, user_suffix),
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if routepilot:
        body["routepilot"] = dict(routepilot)
    return body


# --- the protocol ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Protocol:
    """How an item is put to a model: the messages and the ``max_tokens`` cap.

    ``as-written`` sends the dataset prompt alone with room to reason (the first collection).
    ``answer-only`` prepends a system message asking for the option letter alone and allows
    16 tokens, so the grade measures the answer and not whether the reasoning fit in the cap;
    the small models ignored that system message. ``answer-only-user`` (Ruling V1-R44) sends
    no system message and appends the same instruction to the end of the user message
    (``user_suffix``), with the same 16 tokens.
    """

    name: str
    system: str | None
    max_tokens: int
    user_suffix: str | None = None

    def messages(self, prompt: str) -> list[dict[str, str]]:
        """The messages of ``prompt`` under this protocol: what the models are sent."""
        return request_messages(prompt, self.system, self.user_suffix)

    def body(
        self, model: str, prompt: str, routepilot: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """The request body of ``prompt`` under this protocol."""
        return chat_body(model, prompt, self.max_tokens, routepilot, self.system, self.user_suffix)

    def as_dict(self) -> dict[str, Any]:
        """What is recorded in ``protocol.json``, ``results.json`` and ``run.json``.

        ``user_suffix`` is present only for a protocol that has one, so the records of the
        two earlier protocols keep the form they were written in.
        """
        out: dict[str, Any] = {
            "name": self.name,
            "system_message": self.system,
            "max_tokens": self.max_tokens,
        }
        if self.user_suffix is not None:
            out["user_suffix"] = self.user_suffix
        return out


_PROTOCOLS = {
    AS_WRITTEN: Protocol(AS_WRITTEN, None, MAX_TOKENS),
    ANSWER_ONLY: Protocol(ANSWER_ONLY, ANSWER_ONLY_SYSTEM, ANSWER_ONLY_MAX_TOKENS),
    ANSWER_ONLY_USER: Protocol(
        ANSWER_ONLY_USER, None, ANSWER_ONLY_MAX_TOKENS, ANSWER_ONLY_USER_SUFFIX
    ),
}


def protocol_for(name: str) -> Protocol:
    """The protocol called ``name`` (one of ``PROTOCOLS``)."""
    try:
        return _PROTOCOLS[name]
    except KeyError:
        choices = ", ".join(PROTOCOLS)
        raise ValueError(f"unknown protocol {name!r}; choose one of {choices}") from None


def protocol_suffix_text(protocol: Mapping[str, Any]) -> str:
    """The appended user-message instruction of a recorded protocol, for a one-line table cell.

    JSON-quoted so that its line breaks do not break a markdown table; ``none`` without one.
    """
    suffix = protocol.get("user_suffix")
    return "none" if suffix is None else json.dumps(suffix)


def write_protocol(live_dir: Path, protocol: Protocol) -> None:
    """Record the collection's protocol in ``<live_dir>/protocol.json``."""
    (Path(live_dir) / PROTOCOL_FILE).write_text(
        json.dumps(protocol.as_dict(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def load_protocol(live_dir: Path) -> Protocol:
    """The protocol a collection in ``live_dir`` ran with.

    A directory without ``protocol.json`` holds the first collection, which ran ``as-written``
    before the file existed. A file whose content differs from its named protocol is refused.
    """
    path = Path(live_dir) / PROTOCOL_FILE
    if not path.exists():
        return _PROTOCOLS[AS_WRITTEN]
    doc = json.loads(path.read_text(encoding="utf-8"))
    protocol = protocol_for(str(doc.get("name")))
    if doc != protocol.as_dict():
        raise ValueError(
            f"{path.as_posix()} does not match the protocol {protocol.name!r}: {json.dumps(doc)}"
        )
    return protocol


# --- the measurement ----------------------------------------------------------------------------


def tokens_per_second(chat: Chat) -> float | None:
    """Output tokens per second after the first token, from the client-side timings."""
    if chat.ttft_ms is None or chat.completion_tokens < 2:
        return None
    span = (chat.wall_ms - chat.ttft_ms) / 1000.0
    return (chat.completion_tokens - 1) / span if span > 0 else None


def worst_case_request_s(
    ttft_ms: float, tokens_per_s: float, max_tokens: int = MAX_TOKENS
) -> float:
    """Seconds for a request that runs to ``max_tokens``: the first token plus the rest."""
    return ttft_ms / 1000.0 + max_tokens / tokens_per_s


def planned_worst_case_s(measured_model: Mapping[str, Any], protocol: Protocol) -> float:
    """The worst-case request time of one measured model under ``protocol``.

    Uses the model's measured mean time to first token and output speed with the protocol's
    ``max_tokens``; the 384-token time stored in the measure run is not read.
    """
    return worst_case_request_s(
        float(measured_model["ttft_ms_mean"]),
        float(measured_model["tokens_per_s_mean"]),
        protocol.max_tokens,
    )


def plan_sample_size(
    worst_case_s: Mapping[str, float],
    *,
    n_gpqa: int,
    budget_s: float,
    cap: int = MMLUPRO_CAP,
    fixed_s: float = 0.0,
) -> int:
    """How many ``mmlupro`` items fit with all of ``gpqa`` and every model in the budget.

    One item costs ``sum(worst_case_s.values())`` seconds (every model answers it); ``fixed_s``
    (model loads and warm-ups) comes off the budget first. The result is between 0 and ``cap``.
    """
    if not worst_case_s or any(not (v > 0) for v in worst_case_s.values()):
        raise ValueError("worst_case_s needs a positive time for every model")
    per_item = sum(worst_case_s.values())
    fits = math.floor(max(0.0, budget_s - fixed_s) / per_item)
    return max(0, min(cap, fits - n_gpqa))


def measure(tags: Sequence[str]) -> dict[str, Any]:
    """Time each candidate model on the invented prompts, sent straight to Ollama.

    Per model: a load, a warm-up request (not counted), then every prompt once.
    """
    inventory = ollama_inventory()
    out: dict[str, Any] = {}
    for tag in tags:
        if tag not in inventory["models"]:
            raise ValueError(f"{tag} is not pulled; run: ollama pull {tag}")
        load_started = time.perf_counter()
        ollama_load(tag)
        load_s = time.perf_counter() - load_started
        loaded = [m for m in ollama_loaded() if m["name"] == tag]
        stream_chat(
            11434, chat_body(tag, WARMUP_PROMPT, MEASURE_MAX_TOKENS), timeout_s=300
        )  # warm-up: not counted
        runs = []
        for prompt, truth in MEASURE_PROMPTS:
            chat = stream_chat(11434, chat_body(tag, prompt, MEASURE_MAX_TOKENS), timeout_s=300)
            if not chat.ok:
                raise RuntimeError(f"{tag}: measurement request failed: {chat.error}")
            letter = grading.extract_choice(chat.text, "ABCD")
            runs.append(
                {
                    "ttft_ms": chat.ttft_ms,
                    "wall_ms": chat.wall_ms,
                    "prompt_tokens": chat.prompt_tokens,
                    "completion_tokens": chat.completion_tokens,
                    "tokens_per_s": tokens_per_second(chat),
                    "finish_reason": chat.finish_reason,
                    "correct": letter == truth,
                }
            )
        ollama_unload(tag)
        speeds = [r["tokens_per_s"] for r in runs if r["tokens_per_s"] is not None]
        mean_ttft = float(np.mean([r["ttft_ms"] for r in runs]))
        mean_tps = float(np.mean(speeds))
        out[tag] = {
            "digest": inventory["models"][tag]["digest"],
            "parameter_size": inventory["models"][tag]["parameter_size"],
            "quantization": inventory["models"][tag]["quantization"],
            "params_b_nominal": params_of(tag),
            "load_s": load_s,
            "loaded": loaded,
            "runs": runs,
            "ttft_ms_mean": mean_ttft,
            "tokens_per_s_mean": mean_tps,
            "completion_tokens_mean": float(np.mean([r["completion_tokens"] for r in runs])),
            "correct": int(sum(r["correct"] for r in runs)),
            "worst_case_request_s": worst_case_request_s(mean_ttft, mean_tps),
        }
    return {"ollama_version": inventory["version"], "models": out}


def run_measure(out_root: Path, tags: Sequence[str] = CANDIDATES) -> Path:
    """Measure ``tags`` and write ``results.json``, ``report.md`` and ``run.json``."""
    started = datetime.now(UTC)
    start_git = runinfo.git_state()
    clock = time.perf_counter()
    measured = measure(tags)
    results = {
        "phase": "measure",
        "prompts_sent": len(MEASURE_PROMPTS),
        "max_tokens": MEASURE_MAX_TOKENS,
        **measured,
        "wall_time_s": time.perf_counter() - clock,
        "labels": MEASURE_LABELS,
    }
    rows = []
    for tag, m in measured["models"].items():
        rows.append(
            {
                "model": tag,
                "digest": str(m["digest"])[:12],
                "parameters": m["parameter_size"],
                "TTFT ms (mean of 3)": m["ttft_ms_mean"],
                "output tokens/s (mean of 3)": m["tokens_per_s_mean"],
                "output tokens (mean)": m["completion_tokens_mean"],
                "answers right (of 3)": m["correct"],
                "request s at 384 tokens": m["worst_case_request_s"],
                "load s": m["load_s"],
            }
        )
    out_dir = Path(out_root) / runinfo.new_run_id(started, start_git)
    command = ["python", "bench/scripts/live_collect.py", "measure"]
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        {"Candidate models, measured directly on Ollama": pd.DataFrame(rows)},
        {},
        {},
        command,
        None,
        started,
        git=start_git,
    )
    return out_dir


def latest_measure(out_root: Path = RESULTS_ROOT) -> Path:
    """The newest ``measure`` run under ``out_root``."""
    runs = []
    for results in sorted(Path(out_root).glob("*/results.json")):
        if json.loads(results.read_text(encoding="utf-8")).get("phase") == "measure":
            runs.append(results.parent)
    if not runs:
        raise ValueError(f"no measure run under {Path(out_root).as_posix()}; run 'measure' first")
    return runs[-1]


# --- items, ground truth ------------------------------------------------------------------------


def choose_items(prompts: pd.DataFrame, n_mmlupro: int, seed: int = SEED) -> pd.DataFrame:
    """The items to collect, in the order to send them.

    All ``gpqa`` items plus ``n_mmlupro`` ``mmlupro`` items drawn with
    ``default_rng(seed).choice`` over the sorted item IDs, then one seeded shuffle of both
    together, so that stopping early leaves a random subset of each. Columns ``item_id``,
    ``dataset``.
    """
    gpqa = sorted(prompts.loc[prompts["dataset"] == GPQA, "item_id"])
    pool = sorted(prompts.loc[prompts["dataset"] == MMLUPRO, "item_id"])
    if not 0 <= n_mmlupro <= len(pool):
        raise ValueError(f"n_mmlupro must be between 0 and {len(pool)}, got {n_mmlupro}")
    rng = np.random.default_rng(seed)
    drawn = sorted(rng.choice(pool, size=n_mmlupro, replace=False).tolist())
    ids = np.array([*gpqa, *drawn])
    order = np.random.default_rng([seed, 1]).permutation(len(ids))
    shuffled = ids[order].tolist()
    return pd.DataFrame(
        {"item_id": shuffled, "dataset": [i.split("/", 1)[0] for i in shuffled]},
    )


def read_ground_truth(tar_path: Path, items: pd.DataFrame) -> pd.DataFrame:
    """The ground truth letter of each item, read from the release tarball.

    ``items`` has ``item_id`` and ``prompt_sha256``. The tarball is streamed once and nothing
    is extracted. Only records of a kept dataset's canonical split whose prompt hash equals
    the item's count; every one of them must name the same ``ground_truth``. Returns
    ``item_id``, ``ground_truth`` (the most common value, empty when there is none),
    ``n_records`` (records that vote), ``n_disagree`` (votes against the winner) and
    ``n_other_prompt`` (records at that index with a different prompt, left out).
    """
    wanted = {}
    for row in items.itertuples(index=False):
        dataset, index = str(row.item_id).split("/", 1)
        wanted[(dataset, int(index))] = (str(row.item_id), str(row.prompt_sha256))
    datasets = {d for d, _ in wanted}
    votes: dict[str, dict[str, int]] = {item_id: {} for item_id, _ in wanted.values()}
    other: dict[str, int] = {item_id: 0 for item_id, _ in wanted.values()}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if datasets.isdisjoint(member.name.split("/")):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"{member.name}: cannot read member")
            payload = json.load(handle)
            dataset, split = payload.get("dataset_name"), payload.get("split")
            if selection.KEPT.get(dataset) != split or dataset not in datasets:
                continue
            for rec in payload.get("records", []):
                try:
                    key = (dataset, int(rec["index"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if key not in wanted:
                    continue
                item_id, sha = wanted[key]
                prompt = rec.get("prompt")
                got = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else None
                if got != sha:
                    other[item_id] += 1
                    continue
                truth = str(rec.get("ground_truth", "")).strip().upper()
                votes[item_id][truth] = votes[item_id].get(truth, 0) + 1
    rows = []
    for item_id, _ in wanted.values():
        tally = votes[item_id]
        winner = max(sorted(tally), key=lambda k: tally[k]) if tally else ""
        rows.append(
            {
                "item_id": item_id,
                "ground_truth": winner,
                "n_records": sum(tally.values()),
                "n_disagree": sum(tally.values()) - tally.get(winner, 0),
                "n_other_prompt": other[item_id],
            }
        )
    return pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)


# --- the gateway process ------------------------------------------------------------------------


@contextlib.contextmanager
def gateway(
    binary: Path, config: Mapping[str, Any], live_dir: Path, name: str
) -> Iterator[tuple[int, Path]]:
    """Run ``routepilot serve`` on a free loopback port and always stop it.

    Writes the rendered config as ``<live_dir>/serve-<name>.yaml`` and the process output to
    ``<live_dir>/serve-<name>.log``; yields the port and the decision log path.
    """
    port = overhead.free_port()
    log = live_dir / f"decisions-{name}.jsonl"
    cfg_path = live_dir / f"serve-{name}.yaml"
    cfg = render_config(config, port, log)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    with overhead.service(
        [str(binary), "serve", "-config", str(cfg_path)], live_dir / f"serve-{name}.log"
    ) as proc:
        overhead.wait_ready(f"http://127.0.0.1:{port}/healthz", proc)
        yield port, log


def wait_for_decisions(
    log: Path, ids: Sequence[str], timeout_s: float = DECISION_WAIT_S
) -> set[str]:
    """Poll the decision log until every ID in ``ids`` has a record; return the IDs missing.

    The gateway writes records in the background, so the log trails the last response.
    """
    wanted = set(ids)
    deadline = time.monotonic() + timeout_s
    while True:
        seen: set[str] = set()
        for file in sorted(log.parent.glob(log.stem + "*.jsonl")):
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue  # a line still being written
                    if isinstance(rec, dict) and rec.get("type") == "decision":
                        seen.add(str(rec.get("decision_id")))
        missing = wanted - seen
        if not missing or time.monotonic() >= deadline:
            return missing
        time.sleep(0.25)


# --- grading and the outcome tables -------------------------------------------------------------


def letters_for(dataset: str, prompt: str) -> str:
    """The option letters the prompt offers, or the dataset's usual letters when none parse."""
    return grading.offered_letters(prompt) or DEFAULT_LETTERS[dataset]


def grade_requests(raw: pd.DataFrame, prompts: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Add ``score`` (1.0 or 0.0), ``how`` (cue, last_letter, none), ``letters`` and ``picked``.

    ``raw`` has one row per request with ``item_id`` and ``text``. An item without a ground
    truth letter among the letters its prompt offers raises ``ValueError``.
    """
    prompt_of = dict(zip(prompts["item_id"], prompts["prompt"], strict=True))
    truth_of = dict(zip(truth["item_id"], truth["ground_truth"], strict=True))
    scores, hows, letters, picked = [], [], [], []
    for item_id, text in zip(raw["item_id"], raw["text"], strict=True):
        dataset = item_id.split("/", 1)[0]
        offered = letters_for(dataset, prompt_of[item_id])
        want = truth_of.get(item_id, "")
        if want not in offered:
            raise ValueError(f"{item_id}: ground truth {want!r} is not among the letters {offered}")
        got = grading.extract(text, offered)
        scores.append(1.0 if got.letter == want else 0.0)
        hows.append(got.how)
        letters.append(offered)
        picked.append(got.letter)
    out = raw.copy()
    out["score"], out["how"], out["letters"], out["picked"] = scores, hows, letters, picked
    return out


def build_tables(
    raw: pd.DataFrame,
    decisions: pd.DataFrame,
    prompts: pd.DataFrame,
    catalog: Sequence[CatalogModel],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The shared ``outcomes`` table and the ``measured`` latency table.

    ``raw`` holds the graded requests (``item_id``, ``model``, ``decision_id``, ``score``,
    ``prompt_tokens``, ``completion_tokens``, ``finish_reason``, ``ok``, ``error``,
    ``client_wall_ms``, ``how``); ``decisions`` is the decision log frame. A request is kept in
    ``outcomes`` when it finished with a usage report and its decision record exists. The
    cost is the ASSUMPTION price times the reported tokens. ``measured`` adds the gateway's
    ``ttft_ms`` and ``e2e_ms`` (the record's ``latency_ms``).
    """
    by_model = {c.id: c for c in catalog}
    joined = raw.merge(
        decisions[["decision_id", "status", "ttft_ms", "latency_ms"]],
        on="decision_id",
        how="left",
        validate="m:1",
    )
    joined["has_decision"] = joined["status"].notna()
    kept = joined.loc[joined["ok"] & joined["has_decision"] & (joined["status"] == 200)].copy()
    sha_of = dict(zip(prompts["item_id"], prompts["prompt_sha256"], strict=True))
    outcomes = pd.DataFrame(
        {
            "dataset": kept["item_id"].str.split("/", n=1).str[0],
            "item_id": kept["item_id"],
            "model": kept["model"],
            "score": kept["score"].astype("float64"),
            "success": kept["score"] >= selection.SUCCESS_THRESHOLD,
            "prompt_tokens": kept["prompt_tokens"].astype("int64"),
            "completion_tokens": kept["completion_tokens"].astype("int64"),
            "cost_usd": [
                by_model[m].cost_usd(int(p), int(c))
                for m, p, c in zip(
                    kept["model"], kept["prompt_tokens"], kept["completion_tokens"], strict=True
                )
            ],
            "prompt_sha256": kept["item_id"].map(sha_of),
        }
    )
    outcomes = outcomes.sort_values(["item_id", "model"]).reset_index(drop=True)
    measured = pd.DataFrame(
        {
            "item_id": kept["item_id"],
            "model": kept["model"],
            "decision_id": kept["decision_id"],
            "ttft_ms": kept["ttft_ms"].astype("float64"),
            "e2e_ms": kept["latency_ms"].astype("float64"),
            "prompt_tokens": kept["prompt_tokens"].astype("int64"),
            "completion_tokens": kept["completion_tokens"].astype("int64"),
            "finish_reason": kept["finish_reason"],
            "how": kept["how"],
            "client_wall_ms": kept["client_wall_ms"].astype("float64"),
        }
    )
    measured = measured.sort_values(["item_id", "model"]).reset_index(drop=True)
    return outcomes, measured


def summarize_collection(
    raw: pd.DataFrame,
    outcomes: pd.DataFrame,
    measured: pd.DataFrame,
    catalog: Sequence[CatalogModel],
) -> dict[str, Any]:
    """Counts, accuracy, grading paths, tokens and latency per model; the realized sample."""
    per_model: dict[str, Any] = {}
    for c in catalog:
        req = raw.loc[raw["model"] == c.id]
        out = outcomes.loc[outcomes["model"] == c.id]
        mes = measured.loc[measured["model"] == c.id]
        entry: dict[str, Any] = {
            "tag": c.tag,
            "requests": len(req),
            "ok": int(req["ok"].sum()),
            "failed": int((~req["ok"]).sum()),
            "failure_kinds": req.loc[~req["ok"], "error"]
            .map(lambda e: str(e).split(":")[0])
            .value_counts()
            .to_dict(),
            "in_outcomes": len(out),
            "accuracy": float(out["success"].mean()) if len(out) else None,
            "accuracy_by_dataset": {
                str(d): {"n": len(g), "accuracy": float(g["success"].mean())}
                for d, g in out.groupby("dataset")
            },
            "graded_by": mes["how"].value_counts().to_dict(),
            "hit_max_tokens": int((mes["finish_reason"] == "length").sum()),
            "prompt_tokens_mean": float(out["prompt_tokens"].mean()) if len(out) else None,
            "prompt_tokens_max": int(out["prompt_tokens"].max()) if len(out) else None,
            "completion_tokens_mean": float(out["completion_tokens"].mean()) if len(out) else None,
            "completion_tokens_max": int(out["completion_tokens"].max()) if len(out) else None,
            "total_tokens_max": (
                int((out["prompt_tokens"] + out["completion_tokens"]).max()) if len(out) else None
            ),
            "mean_cost_usd_assumed": float(out["cost_usd"].mean()) if len(out) else None,
        }
        for column in ("ttft_ms", "e2e_ms"):
            values = mes[column].dropna().to_numpy(dtype=float)
            entry[column] = (
                {
                    "mean": float(values.mean()),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                }
                if len(values)
                else None
            )
        per_model[c.id] = entry
    complete = outcomes.groupby("item_id")["model"].nunique()
    full = complete.index[complete == len(catalog)]
    return {
        "per_model": per_model,
        "realized_items": len(full),
        "realized_by_dataset": pd.Series([i.split("/", 1)[0] for i in full])
        .value_counts()
        .sort_index()
        .to_dict(),
    }


# --- the collection -----------------------------------------------------------------------------


def read_raw(path: Path) -> list[dict[str, Any]]:
    """The requests already written to ``raw.jsonl`` (empty without the file)."""
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def collect(
    live_dir: Path,
    out_root: Path,
    measure_run: Path,
    *,
    budget_minutes: float = BUDGET_MINUTES,
    cap: int = MMLUPRO_CAP,
    n_mmlupro: int | None = None,
    limit: int | None = None,
    config: Path = CONFIG,
    tarball: Path = TARBALL,
    derived: Path = ROOT / "data" / "derived",
    binary: Path | None = None,
    resume: bool = False,
    protocol: str = AS_WRITTEN,
    command: Sequence[str] | None = None,
) -> Path:
    """Run the collection and write the tables under ``live_dir`` and the report under ``out_root``.

    ``protocol`` names how the items are put to the models (``as-written`` or ``answer-only``);
    it is recorded in ``<live_dir>/protocol.json``, ``results.json``, the report and ``run.json``
    so that training and the demo use the same one. ``n_mmlupro`` overrides the size planned
    from the measurement (for a short trial). ``resume`` continues an interrupted run: the
    requests in ``raw.jsonl`` are kept and not sent again, and their time counts against the
    budget; the protocol must be the one the interrupted run used. ``command`` is what
    ``run.json`` records (default: the bare ``collect`` command).
    """
    started = datetime.now(UTC)
    start_git = runinfo.git_state()
    clock = time.perf_counter()
    proto = protocol_for(protocol)
    live_dir = Path(live_dir).resolve()
    live_dir.mkdir(parents=True, exist_ok=True)
    if resume and (earlier := load_protocol(live_dir)) != proto:
        raise ValueError(
            f"cannot resume with the protocol {proto.name!r}: {live_dir.as_posix()} was "
            f"collected with {earlier.name!r}"
        )
    base = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    catalog = load_catalog(config)
    measured_run = json.loads((Path(measure_run) / "results.json").read_text(encoding="utf-8"))
    if measured_run.get("phase") != "measure":
        raise ValueError(f"{measure_run} is not a measure run")
    worst = {}
    fixed_s = 0.0
    for c in catalog:
        if c.tag not in measured_run["models"]:
            raise ValueError(f"{c.tag} was not measured in {measure_run}")
        worst[c.id] = planned_worst_case_s(measured_run["models"][c.tag], proto)
        # a model load and a warm-up request that runs to the limit, per model
        fixed_s += measured_run["models"][c.tag]["load_s"] + worst[c.id]

    prompts_all = pd.read_parquet(
        derived / "prompts.parquet", filters=[("dataset", "in", [GPQA, MMLUPRO])]
    )
    n_gpqa = int((prompts_all["dataset"] == GPQA).sum())
    budget_s = budget_minutes * 60.0
    planned = plan_sample_size(worst, n_gpqa=n_gpqa, budget_s=budget_s, cap=cap, fixed_s=fixed_s)
    n_mmlu = planned if n_mmlupro is None else n_mmlupro
    items = choose_items(prompts_all, n_mmlu, SEED)
    if limit is not None:
        items = items.head(limit).reset_index(drop=True)
    prompts = (
        prompts_all.merge(items[["item_id"]], on="item_id")
        .drop_duplicates("item_id")
        .reset_index(drop=True)
    )
    truth = read_ground_truth(tarball, prompts)
    bad = truth.loc[(truth["n_records"] == 0) | (truth["ground_truth"] == "")]
    if len(bad):
        raise ValueError(f"{len(bad)} items have no ground truth, first {bad['item_id'].iloc[0]}")
    truth.to_parquet(live_dir / "ground_truth.parquet", index=False)
    items.to_parquet(live_dir / "items.parquet", index=False)

    binary = binary or replaytool.build_go(ROOT)
    raw_path = live_dir / "raw.jsonl"
    done_records = read_raw(raw_path) if resume else []
    if not resume and raw_path.exists():
        raise FileExistsError(f"{raw_path} exists; pass --resume to continue it or remove it")
    write_protocol(live_dir, proto)
    done = {(r["item_id"], r["model"]) for r in done_records}
    spent_s = sum(float(r["client_wall_ms"]) for r in done_records) / 1000.0
    prompt_of = dict(zip(prompts["item_id"], prompts["prompt"], strict=True))
    warmups: list[dict[str, Any]] = []
    stopped = False
    failures = 0

    with gateway(binary, base, live_dir, "collect") as (port, decision_log):
        ids = [r["decision_id"] for r in done_records if r.get("decision_id")]
        loaded_info = {}
        with raw_path.open("a", encoding="utf-8", newline="\n") as sink:
            for c in catalog:
                ollama_load(c.tag)
                loaded_info[c.id] = [m for m in ollama_loaded() if m["name"] == c.tag]
                warm = stream_chat(
                    port, chat_body(c.id, WARMUP_PROMPT, proto.max_tokens), key=KEY, timeout_s=300
                )
                warmups.append({"model": c.id, "ok": warm.ok, "wall_ms": warm.wall_ms})
                spent_s += warm.wall_ms / 1000.0
                if warm.headers.get("x-routepilot-decision-id"):
                    ids.append(warm.headers["x-routepilot-decision-id"])
                for item_id in items["item_id"]:
                    if (item_id, c.id) in done:
                        continue
                    if spent_s >= budget_s:
                        stopped = True
                        break
                    chat = stream_chat(
                        port,
                        proto.body(c.id, prompt_of[item_id]),
                        key=KEY,
                        timeout_s=300,
                    )
                    spent_s += chat.wall_ms / 1000.0
                    decision_id = chat.headers.get("x-routepilot-decision-id", "")
                    record = {
                        "item_id": item_id,
                        "model": c.id,
                        "decision_id": decision_id,
                        "text": chat.text,
                        "finish_reason": chat.finish_reason,
                        "prompt_tokens": chat.prompt_tokens,
                        "completion_tokens": chat.completion_tokens,
                        "ok": chat.ok,
                        "error": chat.error,
                        "client_wall_ms": chat.wall_ms,
                        "client_ttft_ms": chat.ttft_ms,
                    }
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sink.flush()
                    done_records.append(record)
                    if decision_id:
                        ids.append(decision_id)
                    failures = 0 if chat.ok else failures + 1
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        raise RuntimeError(
                            f"{failures} requests in a row failed, last: {chat.error}"
                        )
                if c is not catalog[-1]:
                    ollama_unload(c.tag)
        missing = wait_for_decisions(decision_log, ids)
    for c in catalog:
        with contextlib.suppress(OSError, urllib.error.URLError):
            ollama_unload(c.tag)

    decisions = load_decisions(live_dir / "decisions-collect*.jsonl")
    raw = pd.DataFrame(done_records)
    graded = grade_requests(raw.loc[raw["ok"]], prompts, truth)
    raw = raw.merge(
        graded[["item_id", "model", "score", "how"]], on=["item_id", "model"], how="left"
    )
    outcomes, measured = build_tables(raw, decisions, prompts, catalog)
    outcomes.to_parquet(live_dir / "outcomes.parquet", index=False)
    measured.to_parquet(live_dir / "measured.parquet", index=False)
    prompts.to_parquet(live_dir / "prompts.parquet", index=False)

    summary = summarize_collection(raw, outcomes, measured, catalog)
    wall = time.perf_counter() - clock
    results = {
        "phase": "collect",
        "measure_run": Path(measure_run).name,
        "protocol": proto.as_dict(),
        "models": [{"id": c.id, "tag": c.tag, "params_b": c.params_b} for c in catalog],
        "assumed_prices_usd_per_mtok": {
            c.id: {"in": float(c.in_usd_per_mtok), "out": float(c.out_usd_per_mtok)}
            for c in catalog
        },
        "plan": {
            "budget_minutes": budget_minutes,
            "cap": cap,
            "worst_case_request_s": worst,
            "planned_mmlupro": planned,
            "used_mmlupro": n_mmlu,
            "gpqa_items": n_gpqa,
            "max_tokens": proto.max_tokens,
            "seed": SEED,
        },
        "stopped_by_budget": stopped,
        "request_time_s": spent_s,
        "wall_time_s": wall,
        "warmups": warmups,
        "decisions_missing": len(missing),
        "ollama_loaded": loaded_info,
        "ground_truth": {
            "items": len(truth),
            "items_with_disagreeing_records": int((truth["n_disagree"] > 0).sum()),
            "records_voting": int(truth["n_records"].sum()),
            "records_with_another_prompt": int(truth["n_other_prompt"].sum()),
        },
        **summary,
        "labels": list(LABELS),
    }
    tables = _collect_tables(results, catalog)
    out_dir = Path(out_root) / runinfo.new_run_id(started, start_git)
    report.write_results(
        out_dir,
        EXPERIMENT,
        results,
        tables,
        {},
        {"sample": SEED},
        list(command or ["python", "bench/scripts/live_collect.py", "collect"]),
        None,
        started,
        measurement={"protocol": proto.as_dict()},
        git=start_git,
    )
    return out_dir


def _collect_tables(results: Mapping[str, Any], catalog: Sequence[CatalogModel]) -> dict[str, Any]:
    plan, protocol = results["plan"], results["protocol"]
    setup = [
        ["models", ", ".join(f"{c.id} ({c.tag})" for c in catalog)],
        ["measure run", results["measure_run"]],
        ["protocol", protocol["name"]],
        ["system message", protocol["system_message"] or "none"],
        ["max_tokens", protocol["max_tokens"]],
        ["user message suffix", protocol_suffix_text(protocol)],
        ["time budget (minutes)", plan["budget_minutes"]],
        [
            "mmlupro items planned / used (cap)",
            f"{plan['planned_mmlupro']} / {plan['used_mmlupro']} ({plan['cap']})",
        ],
        ["gpqa items", plan["gpqa_items"]],
        ["stopped by the time budget", results["stopped_by_budget"]],
        [
            "request time (s) / wall time (s)",
            f"{results['request_time_s']:.1f} / {results['wall_time_s']:.1f}",
        ],
        ["items every model answered", results["realized_items"]],
        ["... by dataset", json.dumps(results["realized_by_dataset"])],
        ["decision records missing", results["decisions_missing"]],
    ]
    rows = []
    for c in catalog:
        m = results["per_model"][c.id]
        rows.append(
            {
                "model": c.id,
                "requests": m["requests"],
                "failed": m["failed"],
                "accuracy": m["accuracy"],
                "hit max tokens": m["hit_max_tokens"],
                "graded by cue / last letter / none": " / ".join(
                    str(m["graded_by"].get(k, 0)) for k in ("cue", "last_letter", "none")
                ),
                "completion tokens mean": m["completion_tokens_mean"],
                "TTFT ms p50 / p95": _pair(m["ttft_ms"]),
                "e2e ms p50 / p95": _pair(m["e2e_ms"]),
                "mean cost USD (assumed)": m["mean_cost_usd_assumed"],
            }
        )
    by_dataset = []
    for c in catalog:
        for dataset, v in sorted(results["per_model"][c.id]["accuracy_by_dataset"].items()):
            by_dataset.append(
                {"model": c.id, "dataset": dataset, "n": v["n"], "accuracy": v["accuracy"]}
            )
    return {
        "Setup": pd.DataFrame(setup, columns=["setting", "value"]),
        "Per model (latency from the decision log, cost assumed)": pd.DataFrame(rows),
        "Accuracy per model and dataset": pd.DataFrame(by_dataset),
    }


def _pair(stats: Mapping[str, float] | None) -> str:
    return "n/a" if not stats else f"{stats['p50']:.0f} / {stats['p95']:.0f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("measure", help="time the candidate models directly on Ollama")
    col = sub.add_parser("collect", help="collect outcomes through routepilot serve")
    col.add_argument("--budget-minutes", type=float, default=BUDGET_MINUTES)
    col.add_argument("--cap", type=int, default=MMLUPRO_CAP, help="largest mmlupro sample")
    col.add_argument("--n-mmlupro", type=int, default=None, help="override the planned sample")
    col.add_argument(
        "--limit", type=int, default=None, help="send only the first N items (a trial)"
    )
    col.add_argument("--measure-run", type=Path, default=None, help="default: the newest")
    col.add_argument("--live-dir", type=Path, default=LIVE_DIR)
    col.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=AS_WRITTEN,
        help=(
            "as-written: the dataset prompt alone, max_tokens 384; answer-only: a system "
            f"message {ANSWER_ONLY_SYSTEM!r} and max_tokens {ANSWER_ONLY_MAX_TOKENS}; "
            "answer-only-user: no system message, the same instruction appended to the user "
            f"message ({ANSWER_ONLY_USER_SUFFIX!r}) and max_tokens {ANSWER_ONLY_MAX_TOKENS}"
        ),
    )
    col.add_argument("--resume", action="store_true")
    for p in (sub.choices["measure"], col):
        p.add_argument("--out-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args(argv)
    if args.command == "measure":
        out = run_measure(args.out_root)
    else:
        out = collect(
            args.live_dir,
            args.out_root,
            args.measure_run or latest_measure(args.out_root),
            budget_minutes=args.budget_minutes,
            cap=args.cap,
            n_mmlupro=args.n_mmlupro,
            limit=args.limit,
            resume=args.resume,
            protocol=args.protocol,
            command=["python", "bench/scripts/live_collect.py", *(argv or sys.argv[1:])],
        )
    print(out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
