"""Streaming ingest of the LLMRouterBench release tarball.

The tarball is read member by member and every result file is parsed straight
from the archive; nothing is extracted to disk. The output is three files in
the output directory:

- ``outcomes.parquet``: one row per (item, model) with success, token counts and
  recorded cost.
- ``prompts.parquet``: one row per kept item with its prompt text. This is the
  only output that holds prompt text and it stays in the gitignored data/derived.
- ``ingest_report.json``: what was read, kept and dropped. It carries counts,
  item ids and 16-hex-digit prompt hash prefixes only, never prompt text.
"""

import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from routepilot_train.data import selection

_OUTCOME_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("item_id", pa.string()),
        ("model", pa.string()),
        ("score", pa.float64()),
        ("success", pa.bool_()),
        ("prompt_tokens", pa.int64()),
        ("completion_tokens", pa.int64()),
        ("cost_usd", pa.float64()),
        ("prompt_sha256", pa.string()),
    ]
)
_PROMPT_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("item_id", pa.string()),
        ("prompt", pa.string()),
        ("prompt_sha256", pa.string()),
        ("prompt_bytes", pa.int64()),
    ]
)


@dataclass(slots=True)
class _Rec:
    """One record with its values still raw: they may be strings or null."""

    index: int
    sha: str
    score: Any
    prompt_tokens: Any
    completion_tokens: Any
    cost: Any


@dataclass(slots=True)
class _Source:
    """The chosen result file of one (dataset, split, model)."""

    dataset: str
    split: str
    model: str
    member: str
    timestamp: str
    records: list[_Rec]
    n_raw: int
    read_drops: Counter[str]


@dataclass(slots=True)
class _Scan:
    members: int = 0
    json_files: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)
    # (dataset, split) -> model -> chosen source
    groups: dict[tuple[str, str], dict[str, _Source]] = field(default_factory=dict)
    # (dataset, split, model) -> every (timestamp, member) seen
    seen: dict[tuple[str, str, str], list[tuple[str, str]]] = field(default_factory=dict)
    # (dataset, split, reason) -> number of files
    excluded: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    # model -> {"files": n, "records": n}
    excluded_models: dict[str, Counter[str]] = field(default_factory=dict)
    # (dataset, split, prompt sha) -> prompt text, canonical splits only
    texts: dict[tuple[str, str, str], str] = field(default_factory=dict)


@dataclass(slots=True)
class _Resolved:
    # (sort key, outcome columns...) for every kept record
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    kept: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    dropped: dict[tuple[str, str, str], Counter[str]] = field(default_factory=dict)
    histogram: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    canonical_items: dict[str, int] = field(default_factory=dict)
    within_split: list[dict[str, Any]] = field(default_factory=list)
    global_dups: list[dict[str, Any]] = field(default_factory=list)
    # dataset -> index -> prompt hash of the items that survive both duplicate checks
    survivors: dict[str, dict[int, str]] = field(default_factory=dict)


def _to_float(value: Any) -> float | None:
    """Parse a number that may be a string; None when it is not a finite number."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _to_int(value: Any) -> int | None:
    """Parse a whole number that may be a string such as "123" or "4.0"."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    number = _to_float(value)
    if number is None or number != int(number):
        return None
    return int(number)


def _prompt_sha256(prompt: Any) -> str | None:
    if not isinstance(prompt, str):
        return None
    try:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:  # a lone surrogate: not valid UTF-8
        return None


def _parse_records(
    raw_records: list[Any],
    dataset: str,
    split: str,
    texts: dict[tuple[str, str, str], str] | None,
) -> tuple[list[_Rec], Counter[str]]:
    records: list[_Rec] = []
    drops: Counter[str] = Counter()
    seen_indexes: set[int] = set()
    for raw in raw_records:
        if not isinstance(raw, dict):
            drops["invalid_record"] += 1
            continue
        index = _to_int(raw.get("index"))
        if index is None:
            drops["unparseable_index"] += 1
            continue
        prompt = raw.get("prompt")
        sha = _prompt_sha256(prompt)
        if sha is None:
            drops["invalid_prompt"] += 1
            continue
        if index in seen_indexes:
            drops["duplicate_record"] += 1
            continue
        seen_indexes.add(index)
        if texts is not None:
            texts.setdefault((dataset, split, sha), prompt)
        records.append(
            _Rec(
                index,
                sha,
                raw.get("score"),
                raw.get("prompt_tokens"),
                raw.get("completion_tokens"),
                raw.get("cost"),
            )
        )
    return records, drops


def _header(member: str, payload: Any) -> tuple[str, str, str, list[Any]]:
    """The authoritative (dataset, split, model, records) of a result file."""
    if isinstance(payload, dict):
        dataset = payload.get("dataset_name")
        split = payload.get("split")
        model = payload.get("model_name")
        records = payload.get("records")
        if (
            isinstance(dataset, str)
            and isinstance(split, str)
            and isinstance(model, str)
            and isinstance(records, list)
        ):
            return dataset, split, model, records
    raise ValueError(f"{member}: missing dataset_name, split, model_name or records")


def _scan_tar(tar_path: Path) -> _Scan:
    scan = _Scan()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            scan.members += 1
            if not member.name.endswith(".json"):
                scan.skipped.append({"member": member.name, "reason": "not a .json file"})
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"{member.name}: cannot read member")
            try:
                payload = json.load(handle)
            except ValueError as err:
                raise ValueError(f"{member.name}: invalid JSON: {err}") from err
            scan.json_files += 1
            dataset, split, model, records = _header(member.name, payload)
            _add_file(scan, member.name, dataset, split, model, records)
    return scan


def _add_file(
    scan: _Scan,
    member: str,
    dataset: str,
    split: str,
    model: str,
    records: list[Any],
) -> None:
    role, reason = selection.classify(dataset, split)
    if role == "excluded":
        scan.excluded[(dataset, split, reason)] += 1
        return
    if model in selection.NON_MODELS:
        counts = scan.excluded_models.setdefault(model, Counter())
        counts["files"] += 1
        counts["records"] += len(records)
        return
    timestamp = member.rsplit("/", 1)[-1].removesuffix(".json").rsplit("-", 1)[-1]
    scan.seen.setdefault((dataset, split, model), []).append((timestamp, member))
    group = scan.groups.setdefault((dataset, split), {})
    current = group.get(model)
    if current is not None and (timestamp, member) < (current.timestamp, current.member):
        return  # older than the file already chosen: not even parsed
    texts = scan.texts if role == "canonical" else None
    parsed, drops = _parse_records(records, dataset, split, texts)
    group[model] = _Source(dataset, split, model, member, timestamp, parsed, len(records), drops)


def _canonical_prompts(sources: dict[str, _Source], models: list[str]) -> dict[int, str]:
    """Index -> prompt hash: the most common one, ties to the first model in sorted order."""
    votes: dict[int, dict[str, int]] = {}
    for model in sorted(models):
        for rec in sources[model].records:
            tally = votes.setdefault(rec.index, {})
            tally[rec.sha] = tally.get(rec.sha, 0) + 1
    # max() keeps the first of equal counts, and tallies are in sorted-model order.
    return {index: max(tally.items(), key=lambda kv: kv[1])[0] for index, tally in votes.items()}


def _split_duplicates(canonical: dict[int, str]) -> dict[int, int]:
    """Indexes whose prompt already appeared at a lower index, mapped to that index."""
    first: dict[str, int] = {}
    duplicates: dict[int, int] = {}
    for index in sorted(canonical):
        sha = canonical[index]
        if sha in first:
            duplicates[index] = first[sha]
        else:
            first[sha] = index
    return duplicates


def _drop(res: _Resolved, source: _Source, reason: str) -> None:
    res.dropped.setdefault((source.dataset, source.split, source.model), Counter())[reason] += 1


def _finish(res: _Resolved, source: _Source, rec: _Rec, item_index: int) -> None:
    """Parse the values of a record that passed every prompt check, then keep or drop it."""
    dataset = source.dataset
    score = _to_float(rec.score)
    if score is None:
        _drop(res, source, "unparseable_score")
        res.histogram[dataset]["unparseable"] += 1
        return
    prompt_tokens = _to_int(rec.prompt_tokens)
    completion_tokens = _to_int(rec.completion_tokens)
    cost = _to_float(rec.cost)
    if prompt_tokens is None or completion_tokens is None or cost is None:
        _drop(res, source, "unparseable_usage")
        return
    res.histogram[dataset]["0" if score == 0.0 else "1" if score == 1.0 else "fractional"] += 1
    res.kept[(source.dataset, source.split, source.model)] += 1
    res.rows.append(
        (
            (dataset, item_index, source.model),
            dataset,
            f"{dataset}/{item_index}",
            source.model,
            score,
            score >= selection.SUCCESS_THRESHOLD,
            prompt_tokens,
            completion_tokens,
            cost,
            rec.sha,
        )
    )


def _resolve(scan: _Scan) -> _Resolved:
    res = _Resolved()
    for (dataset, split), sources in scan.groups.items():
        for model, source in sources.items():
            res.dropped[(dataset, split, model)] = Counter(source.read_drops)
    canonical_sources = {
        dataset: scan.groups[(dataset, split)]
        for dataset, split in selection.KEPT.items()
        if (dataset, split) in scan.groups
    }

    canonical: dict[str, dict[int, str]] = {}
    in_split_dups: dict[str, dict[int, int]] = {}
    for dataset in sorted(canonical_sources):
        sources = canonical_sources[dataset]
        canonical[dataset] = _canonical_prompts(sources, list(sources))
        in_split_dups[dataset] = _split_duplicates(canonical[dataset])
        res.canonical_items[dataset] = len(canonical[dataset])
        dropped_by_kept: dict[int, list[int]] = defaultdict(list)
        for dropped_index, kept_index in in_split_dups[dataset].items():
            dropped_by_kept[kept_index].append(dropped_index)
        for kept_index in sorted(dropped_by_kept):
            res.within_split.append(
                {
                    "dataset": dataset,
                    "split": selection.KEPT[dataset],
                    "prompt_sha256_16": canonical[dataset][kept_index][:16],
                    "kept_item_id": f"{dataset}/{kept_index}",
                    "dropped_item_ids": [
                        f"{dataset}/{i}" for i in sorted(dropped_by_kept[kept_index])
                    ],
                }
            )

    by_sha: dict[str, list[str]] = defaultdict(list)
    for dataset, prompts in canonical.items():
        for index, sha in prompts.items():
            if index not in in_split_dups[dataset]:
                by_sha[sha].append(f"{dataset}/{index}")
    global_dropped: set[str] = set()
    for sha in sorted(by_sha):
        item_ids = sorted(by_sha[sha])
        if len(item_ids) > 1:
            res.global_dups.append(
                {
                    "prompt_sha256_16": sha[:16],
                    "kept_item_id": item_ids[0],
                    "dropped_item_ids": item_ids[1:],
                }
            )
            global_dropped.update(item_ids[1:])

    res.survivors = {
        dataset: {
            index: sha
            for index, sha in prompts.items()
            if index not in in_split_dups[dataset] and f"{dataset}/{index}" not in global_dropped
        }
        for dataset, prompts in canonical.items()
    }

    for dataset, sources in canonical_sources.items():
        for source in sources.values():
            for rec in source.records:
                if rec.sha != canonical[dataset][rec.index]:
                    _drop(res, source, "prompt_mismatch")
                elif rec.index in in_split_dups[dataset]:
                    _drop(res, source, "duplicate_prompt_in_split")
                elif f"{dataset}/{rec.index}" in global_dropped:
                    _drop(res, source, "duplicate_prompt_global")
                else:
                    _finish(res, source, rec, rec.index)

    for dataset, extra_split in selection.MIXED_EXTRA.items():
        extra_sources = scan.groups.get((dataset, extra_split))
        if extra_sources is None:
            continue
        # A model that has a canonical file only ever uses that file.
        in_canonical = set(canonical_sources.get(dataset, {}))
        voters = [model for model in extra_sources if model not in in_canonical]
        extra_canonical = _canonical_prompts(extra_sources, voters)
        extra_dups = _split_duplicates(extra_canonical)
        index_by_sha = {sha: index for index, sha in res.survivors.get(dataset, {}).items()}
        for model, source in extra_sources.items():
            for rec in source.records:
                if model in in_canonical:
                    _drop(res, source, "model_in_canonical_split")
                elif rec.sha != extra_canonical[rec.index]:
                    _drop(res, source, "prompt_mismatch")
                elif rec.index in extra_dups:
                    _drop(res, source, "duplicate_prompt_in_split")
                elif rec.sha not in index_by_sha:
                    _drop(res, source, "no_canonical_item")
                else:
                    _finish(res, source, rec, index_by_sha[rec.sha])
    return res


def _table(schema: pa.Schema, rows: list[tuple[Any, ...]]) -> pa.Table:
    columns = list(zip(*rows, strict=True)) if rows else [()] * len(schema)
    data = {name: list(values) for name, values in zip(schema.names, columns, strict=True)}
    return pa.table(data, schema=schema)


def _write_outputs(scan: _Scan, res: _Resolved, out_dir: Path) -> int:
    """Write the two parquet files and return the number of items in prompts.parquet."""
    rows = sorted(res.rows, key=lambda row: row[0])
    pq.write_table(
        _table(_OUTCOME_SCHEMA, [row[1:] for row in rows]),
        out_dir / "outcomes.parquet",
        compression="zstd",
    )
    with_outcomes = {(row[0][0], row[0][1]) for row in rows}
    prompt_rows = []
    for dataset in sorted(res.survivors):
        split = selection.KEPT[dataset]
        for index in sorted(res.survivors[dataset]):
            if (dataset, index) not in with_outcomes:
                continue
            sha = res.survivors[dataset][index]
            text = scan.texts[(dataset, split, sha)]
            prompt_rows.append(
                (dataset, f"{dataset}/{index}", text, sha, len(text.encode("utf-8")))
            )
    pq.write_table(
        _table(_PROMPT_SCHEMA, prompt_rows), out_dir / "prompts.parquet", compression="zstd"
    )
    return len(prompt_rows)


def _build_report(scan: _Scan, res: _Resolved, n_items: int) -> dict[str, Any]:
    kept: dict[str, dict[str, Any]] = {}
    for (dataset, split), sources in sorted(scan.groups.items()):
        role = selection.classify(dataset, split)[0]
        models: dict[str, Any] = {}
        for model, source in sorted(sources.items()):
            key = (dataset, split, model)
            alternatives = [
                {"member": member, "timestamp": timestamp}
                for timestamp, member in sorted(scan.seen[key], reverse=True)
                if member != source.member
            ]
            models[model] = {
                "files_seen": len(scan.seen[key]),
                "file_chosen": {"member": source.member, "timestamp": source.timestamp},
                "alternatives": alternatives,
                "records_read": source.n_raw,
                "records_kept": res.kept[key],
                "dropped": dict(sorted(res.dropped.get(key, Counter()).items())),
            }
        group: dict[str, Any] = {"role": role, "models": models}
        if role == "canonical":
            group["canonical_items"] = res.canonical_items[dataset]
        kept.setdefault(dataset, {})[split] = group

    excluded = [
        {"dataset": dataset, "split": split, "reason": reason, "files": files}
        for (dataset, split, reason), files in sorted(scan.excluded.items())
    ]
    selected = {f"{d}:{s}" for d, s in (*selection.KEPT.items(), *selection.MIXED_EXTRA.items())}
    present = {f"{d}:{s}" for d, s in scan.groups}
    return {
        "members": scan.members,
        "json_files": scan.json_files,
        "skipped_members": scan.skipped,
        "missing_selected": sorted(selected - present),
        "kept": kept,
        "excluded": excluded,
        "excluded_models": {m: dict(c) for m, c in sorted(scan.excluded_models.items())},
        "score_histogram": {
            dataset: {k: counts[k] for k in ("0", "1", "fractional", "unparseable")}
            for dataset, counts in sorted(res.histogram.items())
        },
        "duplicate_prompts": {"within_split": res.within_split, "global": res.global_dups},
        "totals": {"outcome_rows": len(res.rows), "items": n_items},
    }


def ingest(tar_path: Path, out_dir: Path) -> dict[str, Any]:
    """Read the tarball and write outcomes.parquet, prompts.parquet and ingest_report.json."""
    out_dir = Path(out_dir)
    scan = _scan_tar(Path(tar_path))
    res = _resolve(scan)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_items = _write_outputs(scan, res, out_dir)
    report = _build_report(scan, res, n_items)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (out_dir / "ingest_report.json").write_text(text, encoding="utf-8", newline="\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path, default=Path("data/raw/bench-release.tar.gz"))
    parser.add_argument("--out", type=Path, default=Path("data/derived"))
    args = parser.parse_args(argv)
    report = ingest(args.tar, args.out)
    totals = report["totals"]
    print(f"{totals['outcome_rows']} outcome rows over {totals['items']} items -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
