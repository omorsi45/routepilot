"""Which (dataset, split) groups and models of the LLMRouterBench release are used.

The release tarball, measured on 2026-10-27, holds 30
(dataset, split) groups and 40 model names. Everything here is fixed before any
model is trained.
"""

DATASET_URL = (
    "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz"
)
DATASET_BYTES = 1283503080
DATASET_SHA256 = "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"

# A record counts as a success when its score is at least this value.
SUCCESS_THRESHOLD = 0.5

# Kept datasets and the canonical split that defines their items.
KEPT: dict[str, str] = {
    "aime": "hybrid",
    "arc-agi": "v1",
    "arenahard_coding": "test",
    "arenahard_creative_writing": "test",
    "arenahard_math": "test",
    "gpqa": "test",
    "hle": "test",
    "livecodebench": "test",
    "livemathbench": "test",
    "mmlupro": "test_3000",
    "simpleqa": "test",
}

# Splits whose records are joined onto the canonical split's items by prompt
# SHA-256. Only the mixed pool uses them: the 7B models ran test_1000 only.
MIXED_EXTRA: dict[str, str] = {"mmlupro": "test_1000"}

_SEVEN_B_ONLY = "7B-only: no flagship model ran it"

# Excluded datasets (key "dataset") and splits (key "dataset:split") with the reason.
EXCLUDED: dict[str, str] = {
    "arenahard": "union of the three arenahard category subsets, would duplicate items",
    "hle:subset_500": "subset of the kept hle:test",
    "simpleqa:subset_500": "subset of the kept simpleqa:test",
    "mmlupro:test_1000": (
        "subset of the kept mmlupro:test_3000; used only to join the 7B models for the mixed pool"
    ),
    "swe-bench": "agentic task, not a single prompt with a score",
    "tau2": "multi-turn agentic task, not a single prompt with a score",
    "arcc": _SEVEN_B_ONLY,
    "bbh": _SEVEN_B_ONLY,
    "emorynlp": _SEVEN_B_ONLY,
    "finqa": _SEVEN_B_ONLY,
    "humaneval": _SEVEN_B_ONLY,
    "kandk": _SEVEN_B_ONLY,
    "korbench": _SEVEN_B_ONLY,
    "math500": _SEVEN_B_ONLY,
    "mathbench": _SEVEN_B_ONLY,
    "mbpp": _SEVEN_B_ONLY,
    "medqa": _SEVEN_B_ONLY,
    "meld": _SEVEN_B_ONLY,
    "winogrande": _SEVEN_B_ONLY,
}

# Flagship pool: models with non-zero recorded cost on every kept dataset.
FLAGSHIP_MODELS: tuple[str, ...] = (
    "claude-sonnet-4",
    "deepseek-r1-0528",
    "deepseek-v3-0324",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "glm-4.6",
    "gpt-5",
    "gpt-5-chat",
    "intern-s1",
    "kimi-k2-0905",
    "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507",
)

# Names in the release that are routers, not models. Never ingested.
NON_MODELS: tuple[str, ...] = ("openrouter",)

# Domains for the held-out-domain rule, fixed before training.
DOMAINS: dict[str, str] = {
    "aime": "math",
    "arenahard_math": "math",
    "livemathbench": "math",
    "livecodebench": "code",
    "arenahard_coding": "code",
    "gpqa": "knowledge",
    "hle": "knowledge",
    "mmlupro": "knowledge",
    "simpleqa": "knowledge",
    "arc-agi": "reasoning",
    "arenahard_creative_writing": "writing",
}


def classify(dataset: str, split: str) -> tuple[str, str]:
    """Return (role, reason) for a (dataset, split) group of the release.

    The role is "canonical" (defines items), "mixed_extra" (joined onto canonical
    items by prompt hash) or "excluded" (reason says why). The kept and mixed-extra
    rules win over EXCLUDED, which also lists mmlupro:test_1000 for completeness.
    """
    if KEPT.get(dataset) == split:
        return "canonical", ""
    if MIXED_EXTRA.get(dataset) == split:
        return "mixed_extra", ""
    reason = EXCLUDED.get(f"{dataset}:{split}") or EXCLUDED.get(dataset)
    if reason is not None:
        return "excluded", reason
    if dataset in KEPT or dataset in MIXED_EXTRA:
        return "excluded", f"split {split!r} is not the selected split of {dataset}"
    return "excluded", "dataset is not in the selection"
