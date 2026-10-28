package catalog

import (
	"math"
	"testing"

	"github.com/omorsi45/routepilot/internal/money"
)

func ptrPico(v int64) *money.PicoUSD {
	p := money.PicoUSD(v)
	return &p
}

// TestAffordableK pins the output budget a model can be admitted with under a budget.
// The model costs 150,000 picodollars per input token and 600,000 per output token, the
// request has an input bound of 1,000 tokens (150,000,000 picodollars), and the model
// allows 1,024 output tokens.
func TestAffordableK(t *testing.T) {
	m := &Model{PriceIn: 150_000, PriceOut: 600_000, MaxOutputTokens: 1024}
	implicit := Requirements{InUpper: 1000}
	explicit := Requirements{InUpper: 1000, MaxTokens: 100, HasMaxTokens: true}
	const inCost = 150_000_000

	tests := []struct {
		name    string
		r       Requirements
		tenant  int
		budget  *money.PicoUSD
		wantK   int
		wantOK  bool
		wantMax money.PicoUSD // CostMax at wantK when wantOK
	}{
		{"no budget keeps the effective k", implicit, 4096, nil, 1024, true, 0},
		{"no budget and a tenant default below the model", implicit, 300, nil, 300, true, 0},
		{"a budget above the worst case changes nothing", implicit, 4096, ptrPico(1_000_000_000_000), 1024, true, 0},
		{"exactly the worst case of the full k", implicit, 4096, ptrPico(inCost + 600_000*1024), 1024, true, inCost + 600_000*1024},
		{"one picodollar below the full k shrinks it by one", implicit, 4096, ptrPico(inCost + 600_000*1024 - 1), 1023, true, inCost + 600_000*1023},
		{"floor division of the output budget", implicit, 4096, ptrPico(inCost + 600_000*10 + 599_999), 10, true, inCost + 600_000*10},
		{"exactly one output token", implicit, 4096, ptrPico(inCost + 600_000), 1, true, inCost + 600_000},
		{"one picodollar short of one output token", implicit, 4096, ptrPico(inCost + 599_999), 0, false, 0},
		{"the input alone is the whole budget", implicit, 4096, ptrPico(inCost), 0, false, 0},
		{"the budget does not cover the input", implicit, 4096, ptrPico(inCost - 1), 0, false, 0},
		{"zero budget", implicit, 4096, ptrPico(0), 0, false, 0},
		{"the tenant default caps k below what the budget affords", implicit, 50, ptrPico(1_000_000_000_000), 50, true, inCost + 600_000*50},

		{"explicit cap that fits", explicit, 4096, ptrPico(inCost + 600_000*100), 100, true, inCost + 600_000*100},
		{"explicit cap one picodollar over the budget", explicit, 4096, ptrPico(inCost + 600_000*100 - 1), 100, false, 0},
		{"explicit cap with no budget", explicit, 4096, nil, 100, true, 0},
		{"explicit cap is never shrunk", explicit, 4096, ptrPico(inCost + 600_000*5), 100, false, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			k, ok := m.AffordableK(tc.r, tc.tenant, tc.budget)
			if k != tc.wantK || ok != tc.wantOK {
				t.Fatalf("AffordableK = (%d, %v), want (%d, %v)", k, ok, tc.wantK, tc.wantOK)
			}
			if tc.budget != nil && ok {
				if got := m.CostMax(tc.r, k); got > *tc.budget {
					t.Errorf("CostMax(k=%d) = %d is above the budget %d", k, got, *tc.budget)
				} else if tc.wantMax != 0 && got != tc.wantMax {
					t.Errorf("CostMax(k=%d) = %d, want %d", k, got, tc.wantMax)
				}
			}
		})
	}
}

// TestAffordableKFreeOutput: a model with no output price is not shrunk, so only the
// worst-case check applies.
func TestAffordableKFreeOutput(t *testing.T) {
	m := &Model{PriceIn: 150_000, PriceOut: 0, MaxOutputTokens: 1024}
	r := Requirements{InUpper: 1000}

	if k, ok := m.AffordableK(r, 4096, ptrPico(150_000_000)); k != 1024 || !ok {
		t.Errorf("budget = input cost: AffordableK = (%d, %v), want (1024, true)", k, ok)
	}
	if k, ok := m.AffordableK(r, 4096, ptrPico(149_999_999)); k != 1024 || ok {
		t.Errorf("budget below the input cost: AffordableK = (%d, %v), want (1024, false)", k, ok)
	}

	free := &Model{MaxOutputTokens: 1024}
	if k, ok := free.AffordableK(r, 4096, ptrPico(0)); k != 1024 || !ok {
		t.Errorf("a free model under a zero budget: AffordableK = (%d, %v), want (1024, true)", k, ok)
	}
}

// TestAffordableKSaturates: prices and token counts at the limits never overflow, and the
// result is never negative.
func TestAffordableKSaturates(t *testing.T) {
	huge := &Model{PriceIn: money.PicoUSD(math.MaxInt64 / 2), PriceOut: money.PicoUSD(math.MaxInt64 / 2), MaxOutputTokens: 4096}
	r := Requirements{InUpper: math.MaxInt32}
	if k, ok := huge.AffordableK(r, 4096, ptrPico(math.MaxInt64)); k != 0 || ok {
		t.Errorf("saturated input cost: AffordableK = (%d, %v), want (0, false)", k, ok)
	}

	cheap := &Model{PriceIn: 1, PriceOut: 1, MaxOutputTokens: math.MaxInt}
	if k, ok := cheap.AffordableK(Requirements{InUpper: 10}, math.MaxInt, ptrPico(math.MaxInt64)); k <= 0 || !ok {
		t.Errorf("a vast budget: AffordableK = (%d, %v), want a positive k", k, ok)
	}
}

// TestAffordableKCostMaxNeverExceedsBudget sweeps budgets across the boundaries of one
// model and checks the contract directly: ok implies CostMax(k) <= budget, and the next
// larger k would not fit.
func TestAffordableKCostMaxNeverExceedsBudget(t *testing.T) {
	m := &Model{PriceIn: 7, PriceOut: 13, MaxOutputTokens: 500}
	r := Requirements{InUpper: 20}
	for b := int64(0); b <= 20*7+13*502; b++ {
		budget := ptrPico(b)
		k, ok := m.AffordableK(r, 400, budget)
		if !ok {
			if k >= 1 && m.CostMax(r, k) <= *budget {
				t.Fatalf("budget %d: ok=false with k=%d that fits", b, k)
			}
			continue
		}
		if k < 1 || k > 400 {
			t.Fatalf("budget %d: k = %d outside [1, 400]", b, k)
		}
		if m.CostMax(r, k) > *budget {
			t.Fatalf("budget %d: CostMax(k=%d) = %d does not fit", b, k, m.CostMax(r, k))
		}
		if k < 400 && m.CostMax(r, k+1) <= *budget {
			t.Fatalf("budget %d: k = %d is not the largest that fits", b, k)
		}
	}
}
