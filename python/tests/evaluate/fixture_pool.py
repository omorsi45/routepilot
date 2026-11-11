"""A synthetic derived directory for the script tests: the tiny artifact's two models.

``make_pool`` writes ``pools/tiny.{json,config.yaml}``, ``replay/tiny.jsonl`` and
``mf/tiny.json`` under a root. The items are invented words (never dataset text): train, calib
and test splits of a few dozen items each over two datasets, with a quality outcome and a cost
per model. ``tiny-strong`` is Best Single and ``tiny-weak`` the cheapest.
"""

import base64
import json
import math
from pathlib import Path

import numpy as np

from routepilot_train import features
from routepilot_train.runinfo import repo_root

ROOT = repo_root()
LEARNED = ROOT / "testdata" / "replay" / "learned"
TINY_ARTIFACT = ROOT / "testdata" / "policy" / "tiny" / "0782d801a5a3"
WORDS = ["ba", "ko", "zin", "pelu", "rav", "tuo", "mik", "sed", "lar", "quo", "fen", "dax"]
MODELS = ("tiny-weak", "tiny-strong")
DIM = 8


def invented_prompt(rng: np.random.Generator, i: int) -> str:
    words = [str(rng.choice(WORDS)) for _ in range(int(rng.integers(3, 60)))]
    return " ".join(words) + (" ?" if i % 3 == 0 else "") + (" 12 + 7" if i % 4 == 0 else "")


def write_mf(path: Path) -> None:
    """strong tiny-strong, weak tiny-weak; the score grows with the text length."""
    u = np.zeros(DIM + features.NUM_DENSE, dtype="<f4")
    u[DIM] = 1.0
    mean = [0.0] * features.NUM_DENSE
    mean[0] = math.log1p(30)
    std = [1.0] * features.NUM_DENSE
    std[0] = 0.5
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "strong": "tiny-strong",
        "weak": "tiny-weak",
        "dim": DIM,
        "dense_mean": mean,
        "dense_std": std,
        "u": base64.b64encode(u.tobytes()).decode("ascii"),
    }
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")


def make_pool(
    root: Path,
    n_calib: int = 80,
    n_test: int = 80,
    n_train: int = 20,
    seed: int = 0,
    flip_test: bool = False,
) -> Path:
    """Write the derived files under ``root`` and return it.

    With ``flip_test`` every test outcome succeeds and costs 1000 picodollars more; the train
    and calibration rows are untouched.
    """
    (root / "pools").mkdir(parents=True)
    (root / "replay").mkdir()
    config = (LEARNED / "config.yaml").read_text(encoding="utf-8")
    # The router prices a request with these configured prices (the replay charges the recorded
    # costs). A hundredth of the fixture's prices puts the lambda that trades quality for cost
    # near 1e5 per USD, which a short bisection from [0, 1e6] resolves to a percent.
    for old, new in (
        ('"0.100000"', '"0.001000"'),
        ('"0.400000"', '"0.004000"'),
        ('"1.000000"', '"0.010000"'),
        ('"4.000000"', '"0.040000"'),
    ):
        config = config.replace(old, new)
    config += (
        "\nrouting:\n  policy: rules\n  default_model: tiny-strong\n  rules:\n"
        '    - name: question\n      model: tiny-weak\n      when:\n        contains_any: ["?"]\n'
    )
    (root / "pools" / "tiny.config.yaml").write_text(config, encoding="utf-8")
    pool = {
        "pool": "tiny",
        "models": list(MODELS),
        "best_single": "tiny-strong",
        "cheapest": "tiny-weak",
    }
    (root / "pools" / "tiny.json").write_text(json.dumps(pool), encoding="utf-8")

    rng = np.random.default_rng(seed)
    lines = []
    for split, count in (("train", n_train), ("calib", n_calib), ("test", n_test)):
        for i in range(count):
            dataset = "syn-a" if i % 2 == 0 else "syn-b"
            prompt = invented_prompt(rng, i)
            outcomes = {}
            for m, model in enumerate(MODELS):
                outcomes[model] = {
                    "success": bool(rng.random() < (0.45, 0.7)[m]),
                    # recorded costs: the replay charges these, the router prices with the config
                    "cost_picousd": int(rng.integers(1, 4)) * (10**6, 5 * 10**6)[m],
                    "prompt_tokens": 10,
                    "completion_tokens": int(rng.integers(5, 80)),
                }
                if flip_test and split == "test":
                    outcomes[model]["success"] = True
                    outcomes[model]["cost_picousd"] += 1000
            lines.append(
                json.dumps(
                    {
                        "item_id": f"{dataset}/{split}-{i}",
                        "dataset": dataset,
                        "split": split,
                        "prompt_sha256": f"{len(lines):064x}",
                        "messages": [{"role": "user", "content": prompt}],
                        "outcomes": outcomes,
                    }
                )
            )
    (root / "replay" / "tiny.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_mf(root / "mf" / "tiny.json")
    return root
