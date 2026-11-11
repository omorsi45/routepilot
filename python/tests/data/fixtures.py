"""Builds small synthetic LLMRouterBench tarballs for the ingest tests.

Prompts are made-up strings; nothing here is copied from the dataset.
"""

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TIMESTAMP = "20251029_012339"


@dataclass(frozen=True)
class SyntheticFile:
    member: str
    payload: dict[str, Any]


def prompt_for(dataset: str, index: int) -> str:
    return f"synthetic prompt for {dataset} item {index}"


def record(
    index: Any,
    prompt: Any,
    *,
    score: Any = "1.0",
    prompt_tokens: Any = "10",
    completion_tokens: Any = "5",
    cost: Any = "0.001",
) -> dict[str, Any]:
    return {
        "index": index,
        "origin_query": "unused",
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost,
        "score": score,
        "prediction": "p",
        "ground_truth": "g",
        "raw_output": "o",
    }


def result_file(
    dataset: str,
    split: str,
    model: str,
    records: list[Any],
    *,
    timestamp: str = DEFAULT_TIMESTAMP,
    dir_model: str | None = None,
    split_dir: bool = False,
) -> SyntheticFile:
    """One result file. The directory name may differ from the JSON model name.

    Datasets with several splits in the release add a split directory level:
    bench-release/<dataset>/<split>/<model>/<dataset>-<split>-<model>-<timestamp>.json
    """
    if split_dir:
        name = f"{dataset}-{split}-{model}-{timestamp}.json"
        member = f"bench-release/{dataset}/{split}/{dir_model or model}/{name}"
    else:
        name = f"{dataset}-{model}-{timestamp}.json"
        member = f"bench-release/{dataset}/{dir_model or model}/{name}"
    payload = {
        "performance": 0.5,
        "time_taken": 1.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "counts": len(records),
        "model_name": model,
        "dataset_name": dataset,
        "split": split,
        "demo": False,
        "records": records,
    }
    return SyntheticFile(member, payload)


def simple_records(dataset: str, indexes: range | list[int], **fields: Any) -> list[dict[str, Any]]:
    return [record(i, prompt_for(dataset, i), **fields) for i in indexes]


def build_tarball(
    path: Path, files: list[SyntheticFile], extra_members: dict[str, bytes] | None = None
) -> Path:
    """Write a .tar.gz laid out like bench-release.tar.gz, with directory entries."""
    with tarfile.open(path, "w:gz") as tar:
        directories: set[str] = set()
        for f in files:
            parts = f.member.split("/")
            for depth in range(1, len(parts)):
                directories.add("/".join(parts[:depth]))
        for name in sorted(directories):
            info = tarfile.TarInfo(name)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tar.addfile(info)
        members = {}
        for f in files:
            if f.member in members:
                raise ValueError(f"duplicate member {f.member}: give the files distinct timestamps")
            members[f.member] = json.dumps(f.payload).encode("utf-8")
        members.update(extra_members or {})
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return path
