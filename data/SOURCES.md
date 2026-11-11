# Data sources

## LLMRouterBench release

| Field | Value |
|---|---|
| URL | https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz |
| Size | 1,283,503,080 bytes |
| SHA-256 | `b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1` |
| License | MIT |
| Retrieved | 2026-10-27 |

Used by the offline evaluation pipeline in `python/`: the release records, for
many models and benchmark datasets, the score, token counts and cost of each
model on each prompt. The file is never committed. `data/raw/` and
`data/derived/` are gitignored; `make data` downloads the tarball into
`data/raw/`, checks its size and SHA-256, and writes the ingested tables to
`data/derived/`. `download.verify` refuses any file whose size or SHA-256
differs from the values above.
