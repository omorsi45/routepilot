#!/usr/bin/env bash
# Smoke test of the compose stack (deploy/docker-compose.yml): build and start it, wait for the
# gateway, send one "auto" chat request with the demo key, check /metrics, tear it down.
#
#   bash scripts/compose-smoke.sh
#
# Runs in Git Bash on Windows and on Linux; needs docker compose v2 and curl. The stack is
# stopped and its volumes removed on exit, a failure included. Set COMPOSE_SMOKE_KEEP=1 to leave
# it running (stop it later with: docker compose -f deploy/docker-compose.yml down -v).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

compose=(docker compose -f deploy/docker-compose.yml)
base=http://127.0.0.1:8080
# The fake local key of the demo tenant; the config holds only its SHA-256.
key=demo-key-local
work="$(mktemp -d)"

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "--- compose smoke failed (exit $status): container state and the gateway log" >&2
    "${compose[@]}" ps -a >&2 || true
    "${compose[@]}" logs --no-color --tail=60 routepilot >&2 || true
  fi
  if [ "${COMPOSE_SMOKE_KEEP:-0}" != "1" ]; then
    "${compose[@]}" down -v >&2 || true
  fi
  rm -rf "$work"
  exit "$status"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

echo "== starting the stack"
"${compose[@]}" up -d --build

# The gateway image is distroless (no shell), so it has no container healthcheck: poll /readyz.
echo "== waiting for $base/readyz (up to 90 s)"
ready=0
for _ in $(seq 1 90); do
  if curl -fsS -o /dev/null --max-time 2 "$base/readyz" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 1
done
[ "$ready" = 1 ] || fail "$base/readyz did not answer within 90 s"
echo "ready"

echo "== one auto chat request"
code="$(curl -sS -o "$work/body" -D "$work/headers.raw" -w '%{http_code}' --max-time 30 \
  -H "Authorization: Bearer $key" \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"What is 17 times 23?"}]}' \
  "$base/v1/chat/completions")"
tr -d '\r' <"$work/headers.raw" >"$work/headers"
echo "status $code"
grep -i '^x-routepilot-' "$work/headers" || true
[ "$code" = 200 ] || fail "chat request answered $code, want 200: $(head -c 300 "$work/body")"
grep -qi '^x-routepilot-policy-version: .' "$work/headers" || fail "no x-routepilot-policy-version header"

echo "== /metrics"
curl -fsS --max-time 10 -o "$work/metrics" "$base/metrics"
grep -q '^rp_requests_total' "$work/metrics" || fail "/metrics has no rp_requests_total"
grep -m3 '^rp_requests_total' "$work/metrics"

echo "compose smoke OK"
