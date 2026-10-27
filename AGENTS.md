# AGENTS.md

This file guides coding agents contributing to RoutePilot.

## Project

RoutePilot is an OpenAI-compatible LLM routing gateway written in Go. It sits
between clients and model providers and chooses a provider and model for each
request.

## Layout

- `cmd/routepilot/`: the CLI entry point and subcommand dispatch.
- `internal/`: private packages, one responsibility per package:
  `catalog`, `config`, `gateway`, `mock`, `money`, `openai`, `provider` (with
  the `openaicompat`, `anthropic` and `factory` subpackages), `quota` (with the
  `quotatest` conformance suite), `reliability`, `router`, and `version`.
- `python/`: the offline training and evaluation pipeline (package
  `routepilot_train`, tests in `python/tests`). `make py-setup` creates the
  repo-root `.venv`; `make py-check` runs ruff and pytest.

The planned directory layout is described in section 11 of the design spec.
Add a directory only when the task that needs it lands.

## Working rules

- The Go version is whatever `go.mod` declares. Do not change it casually.
- Run `make verify` before every commit. It runs gofmt, vet, build, tests and
  the race detector.
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
- Commit subjects are short and imperative.
