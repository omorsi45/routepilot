# LiteLLM for the comparison

`bench/scripts/litellm_compare.py` measures the LiteLLM proxy and RoutePilot against the same
`rp-mock`, with the direct path to the mock as the baseline. This folder holds what the LiteLLM
side needs. The proxy is not part of RoutePilot and is never installed into the repo venv.

## Version

`litellm[proxy]==1.106.6`, the latest release on PyPI on 2026-10-28. `requirements.txt` is the full
dependency set resolved once with uv 0.11.2 for Python 3.14, every package pinned with `==` and its
hashes, so the install is the same on every machine:

```
echo 'litellm[proxy]==1.106.6' | uv pip compile - --python-version 3.14 --generate-hashes --no-header
```

Every dependency resolved and installed on Python 3.14.9, the same Python as the rest of the Linux
tooling, so no Python 3.12 fallback was needed. The comparison script reads the version from
the `x-litellm-version` header of the running proxy and stops without writing results if it is not
the version pinned in `requirements.txt`.

## Install

On Linux (the latency numbers are measured there), outside the repo:

```
uv venv ~/rp-tools/litellm-venv --python 3.14
uv pip install --python ~/rp-tools/litellm-venv/bin/python --require-hashes -r bench/litellm/requirements.txt
```

A venv created at `bench/litellm/.venv` also works and is ignored by git. The script finds
`~/rp-tools/litellm-venv/bin/litellm` by default; `--litellm-bin` names another executable.

## Run

```
python bench/scripts/litellm_compare.py --scratch-root ~/rp-scratch
```

Under WSL keep `--scratch-root` on the Linux filesystem: a `/mnt/c` mount makes every log write
slow. There are 24 cells (4 rates, 2 modes, 3 paths), each a 10 s warmup and a 60 s window plus the
start and stop of a fresh mock and a fresh proxy or gateway; the report gives the wall time.

## What the proxy runs with

The script starts, once per cell:

```
litellm --config <config.yaml with the mock's port> --host 127.0.0.1 --port <free port> --num_workers 8
```

with `LITELLM_LOCAL_MODEL_COST_MAP=True` (the cost map is read from the package, never fetched from
the internet) and `LITELLM_LOG=ERROR`, in a session of its own. The proxy and its workers are one
process group, stopped with SIGTERM and, if anything is left after 10 s, SIGKILL.

`config.yaml`:

- one model, `bench-model`, sent to `openai/bench-model` at the mock's `/v1` with a fake key the
  mock ignores, the same model name and the same request body the other two paths use;
- `litellm_settings: {telemetry: false, drop_params: true}`;
- no database.

### Workers

8 workers. LiteLLM is a Python program and scales across cores with one process per worker, while
RoutePilot is a single Go process that already uses all 24 threads of the machine. Eight Python
processes give the proxy a fair share of the CPU without starving the load generator and the mock,
which run on the same machine. A different worker count changes LiteLLM's numbers: fewer workers
saturate earlier, more compete harder with the generator. Only 8 was measured.

### No master key

The proxy runs without a master key: it listens on `127.0.0.1` only, so there is
nothing to protect. LiteLLM 1.106.6 refuses to start without one unless it is told the absence is
intended, which `general_settings.dangerously_permit_weak_or_unset_master_key: true` does; that
setting is in `config.yaml` only so the proxy starts.
The proxy logs a critical line about the missing key at startup for the same reason.

Without a key the proxy does no key lookup per request, while RoutePilot checks the tenant key on
every request. That makes the comparison slightly favorable to LiteLLM, and the report says so.

## What is not claimed

LiteLLM's own benchmark numbers are not restated here. Only what `litellm_compare.py` measured on
this machine, in one session, is reported, with the limitations the report lists (one run per cell,
a shared machine, Linux under WSL2).
