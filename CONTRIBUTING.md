# Contributing to RoutePilot

Contributor guide for RoutePilot.

## Project

RoutePilot is an OpenAI-compatible LLM routing gateway written in Go. It sits
between clients and model providers and chooses a provider and model for each
request. The Go data plane is in `cmd/` and `internal/`; the offline training
and evaluation pipeline is Python, in `python/`.

## Layout

The README's "Repository layout" section lists every directory and what it
holds. It is the one place the layout is written down, so add a new directory
there when you add one.

## Working rules

- The Go version is whatever `go.mod` declares. Do not change it casually.
- Run `make verify` before every commit. It runs the checks CI runs: gofmt,
  vet, build, tests and the race detector for Go, and ruff and pytest for
  Python. `make py-setup` creates the repo-root `.venv` the Python targets use.
- Work test-first: write a failing test, watch it fail for the expected
  reason, then write the minimal code to pass.
- Tests are hermetic: loopback `httptest` only, no internet, no API keys, no
  Docker.
- Use fake credentials in tests (for example `test-key-tenant-a`). Compute
  hashes at runtime instead of pasting them.
- Never commit data files, databases, binaries, logs, or secrets.
- Never log prompts or completions.
- Wrap errors with context. Do not panic on input.
- Keep changes scoped to the task. No speculative features or unrelated
  cleanup.
- Write code, comments and docs so they stand on their own. Do not refer to
  documents that are not in the repository (decision IDs, task numbers, design
  document sections); `python/tests/test_repository_text.py` checks for them.
- Commit subjects are short and imperative.
