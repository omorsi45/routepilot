package quota

import "github.com/redis/go-redis/v9"

// Lua numbers are IEEE doubles, exact for integers up to 2^53. Every value a
// script handles is an integer microsecond count that Go has already clamped
// (see redis.go), so scripts only add, subtract and compare, and stay exact.
// Integers are written back with %.0f: tostring would round to 14 digits.
//
// Each script is one atomic Redis operation. GCRA state lives in a hash with
// two fields: tat (theoretical arrival time) and seen (latest effective now),
// matching the memory backend's gcraEntry.

// reserveScript applies one GCRA decision.
//
//	KEYS[1] GCRA hash
//	ARGV[1] now in microseconds
//	ARGV[2] cost: n*T in microseconds
//	ARGV[3] limit: tau+T in microseconds
//
// Returns {allowed (1 or 0), retry-after in microseconds}. The key's TTL is
// (tat - eff) rounded up to ms, plus one second.
var reserveScript = redis.NewScript(`
local now = tonumber(ARGV[1])
local cost = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

local st = redis.call('HMGET', KEYS[1], 'tat', 'seen')
local tat = tonumber(st[1])
local seen = tonumber(st[2])
local exists = tat ~= nil and seen ~= nil
if not exists then
  tat = now
  seen = now
end

local eff = math.max(now, seen)
local tat0 = math.max(tat, eff)
local span = (tat0 - eff) + cost

local function ttl_ms(t)
  local d = t - eff
  if d < 0 then d = 0 end
  return math.ceil(d / 1000) + 1000
end

if span > limit then
  -- Deny: keep the TAT, record the clock. A deny always has existing state.
  if exists then
    redis.call('HSET', KEYS[1], 'seen', string.format('%.0f', eff))
    redis.call('PEXPIRE', KEYS[1], string.format('%.0f', ttl_ms(tat)))
  end
  return {0, span - limit}
end

local new_tat = tat0 + cost
redis.call('HSET', KEYS[1], 'tat', string.format('%.0f', new_tat), 'seen', string.format('%.0f', eff))
redis.call('PEXPIRE', KEYS[1], string.format('%.0f', ttl_ms(new_tat)))
return {1, 0}
`)

// refundScript moves the TAT back by cost, never to before the effective now.
// A key with no state is left alone.
//
//	KEYS[1] GCRA hash
//	ARGV[1] now in microseconds
//	ARGV[2] cost: n*T in microseconds
//
// Returns 1 when state was updated, 0 when there was none.
var refundScript = redis.NewScript(`
local now = tonumber(ARGV[1])
local cost = tonumber(ARGV[2])

local st = redis.call('HMGET', KEYS[1], 'tat', 'seen')
local tat = tonumber(st[1])
local seen = tonumber(st[2])
if tat == nil or seen == nil then
  return 0
end

local eff = math.max(now, seen)
if tat <= eff or (tat - eff) <= cost then
  tat = eff
else
  tat = tat - cost
end

redis.call('HSET', KEYS[1], 'tat', string.format('%.0f', tat), 'seen', string.format('%.0f', eff))
redis.call('PEXPIRE', KEYS[1], string.format('%.0f', math.ceil((tat - eff) / 1000) + 1000))
return 1
`)

// acquireScript takes or refreshes a lease in a sorted set scored by lease
// expiry in microseconds.
//
//	KEYS[1] lease sorted set
//	ARGV[1] now in microseconds
//	ARGV[2] this lease's new expiry in microseconds
//	ARGV[3] lease id
//	ARGV[4] limit (positive)
//
// Returns 1 when the lease is held afterwards, 0 when the limit was hit. The
// key's TTL is the latest lease expiry minus now, in ms, plus one second.
var acquireScript = redis.NewScript(`
local now = ARGV[1]
local expiry = ARGV[2]
local lease = ARGV[3]
local limit = tonumber(ARGV[4])

-- Leases that expire at or before now are dropped.
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)

local function touch()
  local top = redis.call('ZREVRANGE', KEYS[1], 0, 0, 'WITHSCORES')
  if top[2] then
    local d = tonumber(top[2]) - tonumber(now)
    if d < 0 then d = 0 end
    redis.call('PEXPIRE', KEYS[1], string.format('%.0f', math.ceil(d / 1000) + 1000))
  end
end

if redis.call('ZSCORE', KEYS[1], lease) then
  redis.call('ZADD', KEYS[1], expiry, lease)
  touch()
  return 1
end
if redis.call('ZCARD', KEYS[1]) >= limit then
  touch()
  return 0
end
redis.call('ZADD', KEYS[1], expiry, lease)
touch()
return 1
`)
