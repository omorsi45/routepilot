package telemetry

import "testing"

// TestDeploymentErrorRateSamples: the error rate of each deployment is a gauge read at
// scrape time, and there are no samples until a reader is set.
func TestDeploymentErrorRateSamples(t *testing.T) {
	m, reg := newTestMetrics()
	before, _ := gather(t, reg)
	if len(before["rp_deployment_error_rate"]) != 0 {
		t.Errorf("samples without a source: %+v", before["rp_deployment_error_rate"])
	}

	rates := map[string]float64{"dep-a": 0.25, "dep-b": 0}
	m.SetSources(Sources{DeploymentErrorRates: func() map[string]float64 { return rates }})
	got, types := gather(t, reg)
	if types["rp_deployment_error_rate"] != "GAUGE" {
		t.Errorf("type = %s, want GAUGE", types["rp_deployment_error_rate"])
	}
	if len(got["rp_deployment_error_rate"]) != 2 {
		t.Fatalf("samples = %+v, want one per deployment", got["rp_deployment_error_rate"])
	}
	if v := find(t, got, "rp_deployment_error_rate", map[string]string{"deployment": "dep-a"}).value; v != 0.25 {
		t.Errorf("dep-a = %v, want 0.25", v)
	}
	if v := find(t, got, "rp_deployment_error_rate", map[string]string{"deployment": "dep-b"}).value; v != 0 {
		t.Errorf("dep-b = %v, want 0", v)
	}

	rates["dep-a"] = 0.5 // the next scrape reads the new value
	got, _ = gather(t, reg)
	if v := find(t, got, "rp_deployment_error_rate", map[string]string{"deployment": "dep-a"}).value; v != 0.5 {
		t.Errorf("dep-a after a change = %v, want 0.5", v)
	}
}
