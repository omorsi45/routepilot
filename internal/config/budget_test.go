package config

import (
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/money"
	"github.com/omorsi45/routepilot/internal/quota"
)

func TestParseReadsTenantBudget(t *testing.T) {
	tests := []struct {
		name       string
		extra      string
		wantUSD    string
		wantPeriod string
	}{
		{"no budget", "", "", ""},
		{"budget defaults to monthly", "budget_usd: \"25.50\"\n    ", "25.50", quota.PeriodMonthly},
		{"explicit monthly", "budget_usd: \"1\"\n    budget_period: monthly\n    ", "1", quota.PeriodMonthly},
		{"total", "budget_usd: \"0.000000000001\"\n    budget_period: total\n    ", "0.000000000001", quota.PeriodTotal},
		{"zero is a valid budget", "budget_usd: \"0\"\n    ", "0", quota.PeriodMonthly},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := Parse([]byte(baseYAML(t, "allowed_models:", tc.extra+"allowed_models:")))
			if err != nil {
				t.Fatalf("Parse: %v", err)
			}
			got := cfg.Tenants[0]
			if got.BudgetUSD != tc.wantUSD || got.BudgetPeriod != tc.wantPeriod {
				t.Errorf("budget = %q / %q, want %q / %q", got.BudgetUSD, got.BudgetPeriod, tc.wantUSD, tc.wantPeriod)
			}
		})
	}
}

func TestTenantBudgetLimit(t *testing.T) {
	cfg, err := Parse([]byte(baseYAML(t, "allowed_models:", "budget_usd: \"2.5\"\n    budget_period: total\n    allowed_models:")))
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	lim, ok, err := cfg.Tenants[0].BudgetLimit()
	if err != nil || !ok {
		t.Fatalf("BudgetLimit = %+v, %v, %v; want a limit", lim, ok, err)
	}
	if want := (quota.BudgetLimit{Limit: 2_500_000_000_000, Period: quota.PeriodTotal}); lim != want {
		t.Errorf("BudgetLimit = %+v, want %+v", lim, want)
	}

	// A tenant without budget_usd has no limit, and a hand-built period defaults to monthly.
	if _, ok, err := (TenantConfig{}).BudgetLimit(); ok || err != nil {
		t.Errorf("no budget: ok=%v err=%v, want neither", ok, err)
	}
	lim, ok, err = TenantConfig{BudgetUSD: "1"}.BudgetLimit()
	if err != nil || !ok || lim.Period != quota.PeriodMonthly || lim.Limit != money.PicoUSD(1_000_000_000_000) {
		t.Errorf("hand-built budget = %+v, %v, %v; want 1 USD monthly", lim, ok, err)
	}
	if _, _, err := (TenantConfig{BudgetUSD: "abc"}).BudgetLimit(); err == nil {
		t.Error("a malformed budget_usd must be an error")
	}
}

func TestParseRejectsTenantBudget(t *testing.T) {
	tests := []struct {
		name string
		yaml string
		want []string
	}{
		{"negative budget", baseYAML(t, "allowed_models:", "budget_usd: \"-1\"\n    allowed_models:"),
			[]string{"tenants[0].budget_usd"}},
		{"budget with an exponent", baseYAML(t, "allowed_models:", "budget_usd: \"1e3\"\n    allowed_models:"),
			[]string{"tenants[0].budget_usd"}},
		{"budget that is not a number", baseYAML(t, "allowed_models:", "budget_usd: lots\n    allowed_models:"),
			[]string{"tenants[0].budget_usd"}},
		{"budget finer than a picodollar", baseYAML(t, "allowed_models:", "budget_usd: \"0.0000000000001\"\n    allowed_models:"),
			[]string{"tenants[0].budget_usd"}},
		{"budget over the int64 range", baseYAML(t, "allowed_models:", "budget_usd: \"9223373\"\n    allowed_models:"),
			[]string{"tenants[0].budget_usd"}},
		{"unknown period", baseYAML(t, "allowed_models:", "budget_usd: \"1\"\n    budget_period: weekly\n    allowed_models:"),
			[]string{`tenants[0].budget_period: unknown value "weekly"`}},
		{"period without a budget", baseYAML(t, "allowed_models:", "budget_period: total\n    allowed_models:"),
			[]string{"tenants[0].budget_period: budget_period set without budget_usd"}},
		{"redis ledger above 9000 USD", baseYAML(t, "tenants:", "quota:\n  backend: redis\n  redis_url_env: REDIS_URL\ntenants:",
			"allowed_models:", "budget_usd: \"9000.000001\"\n    allowed_models:"),
			[]string{"tenants[0].budget_usd", "9000"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg, err := Parse([]byte(tc.yaml))
			if err == nil {
				t.Fatalf("Parse succeeded, cfg = %+v", cfg.Tenants)
			}
			for _, w := range tc.want {
				if !strings.Contains(err.Error(), w) {
					t.Errorf("error %q does not contain %q", err, w)
				}
			}
		})
	}
}

func TestParseAcceptsBudgetLimits(t *testing.T) {
	redis := func(usd string) string {
		return baseYAML(t, "tenants:", "quota:\n  backend: redis\n  redis_url_env: REDIS_URL\ntenants:",
			"allowed_models:", "budget_usd: \""+usd+"\"\n    allowed_models:")
	}
	for _, usd := range []string{"9000", "0.01", "0"} {
		if _, err := Parse([]byte(redis(usd))); err != nil {
			t.Errorf("redis with budget_usd %s: %v", usd, err)
		}
	}
	// The exactness limit belongs to the Redis ledger only.
	mem := baseYAML(t, "allowed_models:", "budget_usd: \"1000000\"\n    allowed_models:")
	if _, err := Parse([]byte(mem)); err != nil {
		t.Errorf("memory ledger with 1,000,000 USD: %v", err)
	}
}

// TestParseRejectsTPMBurstBelowDefaultMaxTokens pins that a request without
// max_tokens reserves up to default_max_tokens plus its input bound, so a smaller token
// burst could never admit it.
func TestParseRejectsTPMBurstBelowDefaultMaxTokens(t *testing.T) {
	const msg = "below default_max_tokens + 24; requests without max_tokens could never be admitted"
	tenant := func(fields string) string {
		return baseYAML(t, "allowed_models:", fields+"\n    allowed_models:")
	}
	rejected := []struct{ name, fields string }{
		{"default default_max_tokens (4096), burst 4095", "tpm: 60000\n    tpm_burst: 4095"},
		{"default default_max_tokens, burst 4119", "tpm: 60000\n    tpm_burst: 4119"},
		{"burst defaults to tpm, which is too small", "tpm: 1000"},
		{"explicit default_max_tokens 100, burst 123", "tpm: 60000\n    tpm_burst: 123\n    default_max_tokens: 100"},
	}
	for _, tc := range rejected {
		t.Run("rejects "+tc.name, func(t *testing.T) {
			_, err := Parse([]byte(tenant(tc.fields)))
			if err == nil {
				t.Fatal("Parse succeeded, want the burst error")
			}
			if want := "tenants[0].tpm_burst: " + msg; !strings.Contains(err.Error(), want) {
				t.Errorf("error %q does not contain %q", err, want)
			}
		})
	}

	accepted := []struct{ name, fields string }{
		{"burst exactly default_max_tokens + 24", "tpm: 60000\n    tpm_burst: 4120"},
		{"burst well above", "tpm: 60000\n    tpm_burst: 100000"},
		{"burst defaults to a large tpm", "tpm: 200000"},
		{"explicit default_max_tokens 100, burst 124", "tpm: 60000\n    tpm_burst: 124\n    default_max_tokens: 100"},
		{"no tpm means no check", "default_max_tokens: 100000"},
		{"a request rate alone", "rpm: 10"},
	}
	for _, tc := range accepted {
		t.Run("accepts "+tc.name, func(t *testing.T) {
			if _, err := Parse([]byte(tenant(tc.fields))); err != nil {
				t.Fatalf("Parse: %v", err)
			}
		})
	}
}
