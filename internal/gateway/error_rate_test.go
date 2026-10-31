package gateway

import (
	"testing"

	"github.com/omorsi45/routepilot/internal/config"
)

// TestDeploymentErrorRateSeries: every configured deployment has an error rate sample, 0
// before it has been used, and it follows the outcomes of the attempts sent to it.
func TestDeploymentErrorRateSeries(t *testing.T) {
	e := newAttemptEnv(t, attemptOpts{models: failoverModels(), policy: config.PolicyCheapest, upstream: upstream503For("uc")})
	sc := scrapeOf(t, e.gw)
	for _, dep := range []string{"cheap-a", "pricey-a"} {
		if got := sc.value(t, "rp_deployment_error_rate", map[string]string{"deployment": dep}); got != 0 {
			t.Errorf("error rate of %s before any request = %v, want 0", dep, got)
		}
	}

	// The cheap deployment answers 503 and the request fails over to the pricey one.
	if rec := e.post(chatBody("auto", "hi")); rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}

	sc = scrapeOf(t, e.gw)
	if got := sc.value(t, "rp_deployment_error_rate", map[string]string{"deployment": "cheap-a"}); got != 1 {
		t.Errorf("error rate of the failing deployment = %v, want 1 (its only outcome was a failure)", got)
	}
	if got := sc.value(t, "rp_deployment_error_rate", map[string]string{"deployment": "pricey-a"}); got != 0 {
		t.Errorf("error rate of the answering deployment = %v, want 0", got)
	}
}
