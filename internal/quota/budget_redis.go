package quota

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/omorsi45/routepilot/internal/money"
)

// Lua numbers are IEEE doubles, exact for integers up to 2^53. The budget scripts add,
// subtract and compare picodollar amounts, so they are exact while every amount they handle
// stays below that. Go enforces it, never wrapping:
//   - a limit above redisBudgetMax (9e15 picodollars, 9,000 USD) is treated as redisBudgetMax,
//     which only makes the ledger stricter; config validation refuses a larger budget_usd
//     for the Redis backend, and
//   - a reservation or a committed amount above 2^53-1 is clamped to it. A reservation that
//     large can never fit a limit, and a committed amount that large is already an overshoot
//     of any limit the ledger accepts.
const (
	redisBudgetMax       = int64(9_000_000_000_000_000)
	redisBudgetAmountMax = int64(1)<<53 - 1
	// redisBudgetMaxTTL bounds the lifetime Reserve gives a lease record, in milliseconds.
	redisBudgetMaxTTL = int64(366 * 24 * time.Hour / time.Millisecond)
)

// NewRedisBudget returns a BudgetLedger backed by Redis, sharing client. Each operation is
// one Lua script (EVALSHA, falling back to EVAL when the script is not cached), so it is
// atomic, and the gateway instances that share the Redis share every budget.
//
// For a tenant T and a period P (see PeriodKey) it keeps, under prefix:
//
//	<prefix>:budget:{T}:P:spent    committed spend, an integer string
//	<prefix>:budget:{T}:P:leases   hash of request ID -> reserved amount
//	<prefix>:budget:{T}:P:expiry   sorted set of request ID scored by lease expiry in microseconds
//
// and for a request R:
//
//	<prefix>:budget:lease:{R}      hash of tenant, period, amount; lives until the lease
//	                               expires plus 48 hours, and is removed by Commit or Release
//	<prefix>:budget:commit:{R}     the overshoot Commit returned, for 48 hours
//
// With the prefix "rp" the keys read rp:budget:{tenant-a}:2026-10:spent and so on. A lease
// is live while its score is after now; the scripts ignore the others, and Sweep removes
// them. Commit and Release find a request's period through its lease record, so the
// scripts compute the tenant keys they touch inside Lua: they work on a single Redis (or
// a primary with replicas) but not in Redis Cluster, where the request and tenant keys
// would sit in different slots. The Redis errors are returned wrapped and never include a
// key.
func NewRedisBudget(client redis.UniversalClient, prefix string) BudgetLedger {
	return &redisBudget{client: client, base: prefix + ":budget:"}
}

type redisBudget struct {
	client redis.UniversalClient
	base   string // <prefix>:budget:
}

func (b *redisBudget) tenantKey(tenant, period, suffix string) string {
	return b.base + "{" + tenant + "}:" + period + ":" + suffix
}

func (b *redisBudget) leaseKey(requestID string) string { return b.base + "lease:{" + requestID + "}" }
func (b *redisBudget) commitKey(requestID string) string {
	return b.base + "commit:{" + requestID + "}"
}

// limitOf clamps a limit to the exact range.
func limitOf(l money.PicoUSD) int64 { return min(max(int64(l), 0), redisBudgetMax) }

// amountOf clamps an amount to [0, 2^53-1].
func amountOf(a money.PicoUSD) int64 { return min(max(int64(a), 0), redisBudgetAmountMax) }

func (b *redisBudget) Reserve(ctx context.Context, tenant, requestID string, lim BudgetLimit, amount money.PicoUSD, ttl time.Duration, now time.Time) (bool, money.PicoUSD, error) {
	limit := limitOf(lim.Limit)
	// Nothing above the limit can ever fit, so one picodollar over is as good as any more,
	// and it keeps the script's sums exact.
	amt := min(amountOf(amount), limit+1)
	if ttl <= 0 {
		ttl = time.Second
	}
	nowUS := clampNow(now)
	expiry := nowUS + min(ttl.Microseconds(), redisMaxExact)
	period := PeriodKey(lim.Period, now)
	recTTL := min(ttl.Milliseconds(), redisBudgetMaxTTL) + budgetRetention.Milliseconds()

	keys := []string{
		b.tenantKey(tenant, period, "spent"),
		b.tenantKey(tenant, period, "leases"),
		b.tenantKey(tenant, period, "expiry"),
		b.leaseKey(requestID),
	}
	res, err := reserveBudgetScript.Run(ctx, b.client, keys, nowUS, expiry, amt, limit, requestID, tenant, period, b.base, recTTL).Int64Slice()
	if err != nil {
		return false, 0, fmt.Errorf("quota: redis budget reserve: %w", err)
	}
	if len(res) != 2 {
		return false, 0, fmt.Errorf("quota: redis budget reserve: unexpected script reply of %d values", len(res))
	}
	return res[0] == 1, money.PicoUSD(max(res[1], 0)), nil
}

func (b *redisBudget) Commit(ctx context.Context, tenant, requestID string, actual money.PicoUSD, now time.Time) (money.PicoUSD, error) {
	keys := []string{b.leaseKey(requestID), b.commitKey(requestID)}
	res, err := commitBudgetScript.Run(ctx, b.client, keys,
		clampNow(now), amountOf(actual), requestID, tenant, PeriodKey(PeriodMonthly, now), b.base, budgetRetention.Milliseconds()).Int64Slice()
	if err != nil {
		return 0, fmt.Errorf("quota: redis budget commit: %w", err)
	}
	if len(res) != 2 {
		return 0, fmt.Errorf("quota: redis budget commit: unexpected script reply of %d values", len(res))
	}
	return money.PicoUSD(max(res[0], 0)), nil
}

func (b *redisBudget) Release(ctx context.Context, _, requestID string) error {
	if err := releaseBudgetScript.Run(ctx, b.client, []string{b.leaseKey(requestID)}, requestID, b.base).Err(); err != nil {
		return fmt.Errorf("quota: redis budget release: %w", err)
	}
	return nil
}

func (b *redisBudget) Remaining(ctx context.Context, tenant string, lim BudgetLimit, now time.Time) (money.PicoUSD, error) {
	period := PeriodKey(lim.Period, now)
	keys := []string{
		b.tenantKey(tenant, period, "spent"),
		b.tenantKey(tenant, period, "leases"),
		b.tenantKey(tenant, period, "expiry"),
	}
	got, err := remainingBudgetScript.Run(ctx, b.client, keys, clampNow(now), limitOf(lim.Limit)).Int64()
	if err != nil {
		return 0, fmt.Errorf("quota: redis budget remaining: %w", err)
	}
	return money.PicoUSD(max(got, 0)), nil
}

func (b *redisBudget) Spent(ctx context.Context, tenant string, lim BudgetLimit, now time.Time) (money.PicoUSD, error) {
	got, err := b.client.Get(ctx, b.tenantKey(tenant, PeriodKey(lim.Period, now), "spent")).Int64()
	if errors.Is(err, redis.Nil) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("quota: redis budget spent: %w", err)
	}
	return money.PicoUSD(max(got, 0)), nil
}

func (b *redisBudget) Sweep(ctx context.Context, now time.Time) (int, error) {
	nowUS := clampNow(now)
	swept := 0
	iter := b.client.Scan(ctx, 0, b.base+"*:expiry", 100).Iterator()
	for iter.Next(ctx) {
		expiryKey := iter.Val()
		stem, ok := strings.CutSuffix(expiryKey, ":expiry")
		if !ok {
			continue
		}
		n, err := sweepBudgetScript.Run(ctx, b.client, []string{expiryKey, stem + ":leases"}, nowUS).Int()
		if err != nil {
			return swept, fmt.Errorf("quota: redis budget sweep: %w", err)
		}
		swept += n
	}
	if err := iter.Err(); err != nil {
		return swept, fmt.Errorf("quota: redis budget sweep: %w", err)
	}
	return swept, nil
}

// budgetLua is shared by every budget script. tkey builds the key of a tenant's budget the
// same way tenantKey does, num writes an integer without rounding it to 14 digits, and
// live_sum adds up the amounts of the leases whose expiry is after now.
const budgetLua = `
local function tkey(base, tenant, period, suffix)
  return base .. '{' .. tenant .. '}:' .. period .. ':' .. suffix
end

local function num(x)
  return string.format('%.0f', x)
end

local function live_sum(expiryKey, leasesKey, nowStr)
  local ids = redis.call('ZRANGEBYSCORE', expiryKey, '(' .. nowStr, '+inf')
  local total = 0
  local n = #ids
  local i = 1
  while i <= n do
    local j = math.min(i + 999, n)
    local vals = redis.call('HMGET', leasesKey, unpack(ids, i, j))
    for k = 1, #vals do
      if vals[k] then
        total = total + tonumber(vals[k])
      end
    end
    i = j + 1
  end
  return total
end
`

// reserveBudgetScript reserves or raises a reservation.
//
//	KEYS[1..3] the spent key, the leases hash and the expiry set of the period at now
//	KEYS[4]    the request's lease record
//	ARGV[1]    now in microseconds       ARGV[2] new lease expiry in microseconds
//	ARGV[3]    amount (at most limit+1)  ARGV[4] limit
//	ARGV[5]    request ID                ARGV[6] tenant
//	ARGV[7]    period at now             ARGV[8] key base (<prefix>:budget:)
//	ARGV[9]    lease record ttl in milliseconds
//
// A live reservation of the request stays in the period it was made in, so the keys of the
// period at now are used only for a new reservation. Returns {1 or 0, remaining}.
var reserveBudgetScript = redis.NewScript(budgetLua + `
local nowStr = ARGV[1]
local now = tonumber(nowStr)
local expiry = ARGV[2]
local amount = tonumber(ARGV[3])
local limit = tonumber(ARGV[4])
local id, tenant, period, base = ARGV[5], ARGV[6], ARGV[7], ARGV[8]
local spentKey, leasesKey, expiryKey = KEYS[1], KEYS[2], KEYS[3]

local existing = 0
local rec = redis.call('HMGET', KEYS[4], 'tenant', 'period')
if rec[1] == tenant and rec[2] then
  local ek = tkey(base, tenant, rec[2], 'expiry')
  local score = redis.call('ZSCORE', ek, id)
  if score and tonumber(score) > now then
    period = rec[2]
    spentKey = tkey(base, tenant, period, 'spent')
    leasesKey = tkey(base, tenant, period, 'leases')
    expiryKey = ek
    existing = tonumber(redis.call('HGET', leasesKey, id)) or 0
  end
end

local reserved = live_sum(expiryKey, leasesKey, nowStr)
local spent = tonumber(redis.call('GET', spentKey) or '0')
local others = reserved - existing
local used = spent + others
if used > limit or amount > limit - used then
  local rem = limit - spent - reserved
  if rem < 0 then rem = 0 end
  return {0, rem}
end

local held = math.max(existing, amount)
redis.call('HSET', leasesKey, id, num(held))
redis.call('ZADD', expiryKey, expiry, id)
redis.call('HSET', KEYS[4], 'tenant', tenant, 'period', period, 'amount', num(held))
redis.call('PEXPIRE', KEYS[4], ARGV[9])
local rem = limit - spent - (others + held)
if rem < 0 then rem = 0 end
return {1, rem}
`)

// commitBudgetScript adds the actual cost to the spend of the reservation's period and
// removes the reservation.
//
//	KEYS[1] the request's lease record   KEYS[2] the commit result
//	ARGV[1] now in microseconds          ARGV[2] actual
//	ARGV[3] request ID                   ARGV[4] tenant
//	ARGV[5] period for a request without a record (the monthly one at now)
//	ARGV[6] key base                     ARGV[7] commit result ttl in milliseconds
//
// Returns {overshoot, 1 when the result was remembered from an earlier commit}.
var commitBudgetScript = redis.NewScript(budgetLua + `
local prev = redis.call('GET', KEYS[2])
if prev then
  return {tonumber(prev), 1}
end

local now = tonumber(ARGV[1])
local actual = tonumber(ARGV[2])
local id, tenant, period, base = ARGV[3], ARGV[4], ARGV[5], ARGV[6]

local reserved = 0
local rec = redis.call('HMGET', KEYS[1], 'tenant', 'period')
if rec[1] and rec[2] then
  tenant, period = rec[1], rec[2]
  local ek = tkey(base, tenant, period, 'expiry')
  local lk = tkey(base, tenant, period, 'leases')
  local score = redis.call('ZSCORE', ek, id)
  if score then
    if tonumber(score) > now then
      reserved = tonumber(redis.call('HGET', lk, id)) or 0
    end
    redis.call('ZREM', ek, id)
    redis.call('HDEL', lk, id)
  end
end

local overshoot = actual - reserved
if overshoot < 0 then overshoot = 0 end
redis.call('INCRBY', tkey(base, tenant, period, 'spent'), num(actual))
redis.call('DEL', KEYS[1])
redis.call('SET', KEYS[2], num(overshoot), 'PX', ARGV[7])
return {overshoot, 0}
`)

// releaseBudgetScript removes a reservation without spending anything.
//
//	KEYS[1] the request's lease record
//	ARGV[1] request ID   ARGV[2] key base
//
// Returns 1 when a record was found, else 0.
var releaseBudgetScript = redis.NewScript(budgetLua + `
local rec = redis.call('HMGET', KEYS[1], 'tenant', 'period')
if not (rec[1] and rec[2]) then
  return 0
end
redis.call('ZREM', tkey(ARGV[2], rec[1], rec[2], 'expiry'), ARGV[1])
redis.call('HDEL', tkey(ARGV[2], rec[1], rec[2], 'leases'), ARGV[1])
redis.call('DEL', KEYS[1])
return 1
`)

// remainingBudgetScript is max(0, limit - spent - live reserved) of one period.
//
//	KEYS[1] spent  KEYS[2] leases  KEYS[3] expiry
//	ARGV[1] now in microseconds   ARGV[2] limit
var remainingBudgetScript = redis.NewScript(budgetLua + `
local reserved = live_sum(KEYS[3], KEYS[2], ARGV[1])
local spent = tonumber(redis.call('GET', KEYS[1]) or '0')
local rem = tonumber(ARGV[2]) - spent - reserved
if rem < 0 then rem = 0 end
return rem
`)

// sweepBudgetScript drops the leases of one period that expired at or before now.
//
//	KEYS[1] expiry  KEYS[2] leases
//	ARGV[1] now in microseconds
//
// Returns how many it dropped. The request's lease record is left to its own ttl, so a late
// commit still finds its period.
var sweepBudgetScript = redis.NewScript(`
local ids = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
local n = #ids
local i = 1
while i <= n do
  local j = math.min(i + 999, n)
  redis.call('HDEL', KEYS[2], unpack(ids, i, j))
  i = j + 1
end
if n > 0 then
  redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
end
return n
`)
