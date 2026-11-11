#!/usr/bin/env bash
# RoutePilot demo: builds the gateway and the mock upstream, runs both on loopback with
# examples/config.demo.yaml (the learned live-ollama policy over the mock), and shows
#   1. an "auto" request and the headers that say how it was routed
#   2. a request with a quality floor no model clears, which the policy relaxes
#   3. a pinned model
#   4. a failover to a model's second deployment
#   5. a tenant that runs out of budget and gets a 429
#   6. the Prometheus series that recorded all of it
#
#   bash scripts/demo.sh        (or: make demo)
#
# Runs in Git Bash on Windows and on Linux; needs Go and curl, no Docker, no dataset, no API
# keys. Both processes are stopped on exit. The ports must match examples/config.demo.yaml.
# The answers come from the mock, so the demo shows routing, not answer quality.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

gateway=127.0.0.1:18780
mock=127.0.0.1:18781
base="http://$gateway"
# The fake local keys of the two demo tenants; the config holds only their SHA-256.
key_demo=demo-key-local
key_tight=demo-key-tight
prompt='What is 17 times 23?'

exe=
case "$(go env GOOS)" in windows) exe=.exe ;; esac

scratch=bench/scratch/demo
mkdir -p "$scratch" bin
work="$(mktemp -d)"
pids=()

cleanup() {
  status=$?
  trap - EXIT INT TERM
  for pid in ${pids[@]+"${pids[@]}"}; do
    # In Git Bash a background .exe is a native Windows process, which a POSIX signal does
    # not reach; stop it by its Windows process ID.
    if [ -r "/proc/$pid/winpid" ]; then
      taskkill //F //PID "$(cat "/proc/$pid/winpid")" >/dev/null 2>&1 || true
    fi
    kill "$pid" 2>/dev/null || true
  done
  for pid in ${pids[@]+"${pids[@]}"}; do
    wait "$pid" 2>/dev/null || true
  done
  rm -rf "$work"
  exit "$status"
}
trap cleanup EXIT INT TERM

heading() {
  printf '\n== %s\n' "$*"
}

# chat <key> <model> [header...]: POST one chat completion; prints the status line and the
# x-routepilot-* response headers.
chat() {
  local key=$1 model=$2
  shift 2
  local extra=()
  local h
  for h in "$@"; do
    extra+=(-H "$h")
  done
  curl -sS -o "$work/body" -D "$work/headers" --max-time 30 \
    ${extra[@]+"${extra[@]}"} \
    -H "Authorization: Bearer $key" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}" \
    "$base/v1/chat/completions"
  tr -d '\r' <"$work/headers" | grep -iE '^(HTTP/|x-routepilot-)' || true
}

heading "building bin/routepilot$exe and bin/rp-mock$exe"
go build -o "bin/routepilot$exe" ./cmd/routepilot
go build -o "bin/rp-mock$exe" ./cmd/rp-mock

# Refuse to start on a busy port: the gateway would talk to whatever already listens there.
for addr in "$mock" "$gateway"; do
  if curl -sS -o /dev/null --max-time 1 "http://$addr/" 2>/dev/null; then
    echo "port ${addr##*:} is already in use; stop what listens there or change the ports in" \
      "scripts/demo.sh and examples/config.demo.yaml" >&2
    exit 1
  fi
done

heading "starting rp-mock on $mock and routepilot on $gateway (logs in $scratch)"
rm -f "$scratch"/decisions.jsonl*
"./bin/rp-mock$exe" --listen "$mock" --ttft-median-ms 40 --ttft-sigma 0.3 \
  --tpot-median-ms 5 --tokens 64 >"$scratch/mock.log" 2>&1 &
pids+=($!)
"./bin/routepilot$exe" serve --config examples/config.demo.yaml >"$scratch/serve.log" 2>&1 &
pids+=($!)

ready=0
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null --max-time 2 "$base/readyz" 2>/dev/null; then
    ready=1
    break
  fi
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null || {
      echo "a demo process exited early; see $scratch/serve.log and $scratch/mock.log" >&2
      exit 1
    }
  done
  sleep 0.5
done
[ "$ready" = 1 ] || {
  echo "$base/readyz did not answer within 30 s; see $scratch/serve.log" >&2
  exit 1
}
# The gateway is ready before it ever calls the mock, so check the mock answers too.
for pid in "${pids[@]}"; do
  kill -0 "$pid" 2>/dev/null || {
    echo "a demo process exited early; see $scratch/serve.log and $scratch/mock.log" >&2
    exit 1
  }
done
curl -fsS -o /dev/null --max-time 5 "http://$mock/v1/models" || {
  echo "rp-mock does not answer on $mock; see $scratch/mock.log" >&2
  exit 1
}
echo "gateway and mock ready"

heading "1. auto: the learned policy picks the model; the headers say which, and why"
chat "$key_demo" auto

heading "2. min_quality 0.99: no model clears that floor, so the policy relaxes it (x-routepilot-relaxed)"
chat "$key_demo" auto 'x-routepilot-min-quality: 0.99'

heading "3. a pinned model: the policy is bypassed (policy version 'pinned')"
chat "$key_demo" qwen2.5-3b

heading "4. failover: demo-failover-a answers HTTP 500 (mock directive fail=http500), so demo-failover-b serves"
chat "$key_demo" demo-failover

heading "5. budget: tenant demo-tight has a budget of 0.00001 USD; it is served until the budget cannot cover one more request"
codes=
for _ in $(seq 1 60); do
  code="$(curl -sS -o "$work/body" -w '%{http_code}' --max-time 30 \
    -H "Authorization: Bearer $key_tight" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}]}" \
    "$base/v1/chat/completions")"
  codes="$codes $code"
  [ "$code" = 429 ] && break
done
echo "statuses:$codes"
echo "last response:"
cat "$work/body"
echo

heading "6. /metrics: the series that recorded the requests above"
curl -fsS --max-time 10 -o "$work/metrics" "$base/metrics"
for series in rp_requests_total rp_routing_decisions_total 'rp_upstream_attempts_total{deployment="demo-failover' rp_budget_rejections_total; do
  grep -F "$series" "$work/metrics" | grep -v '^#' | sed -n '1,6p' || true
done

heading "done; stopping rp-mock and routepilot"
