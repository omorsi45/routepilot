package config

import (
	"strings"
	"testing"
)

func TestDecisionLogIsOffByDefault(t *testing.T) {
	cfg, err := Parse([]byte(accountingYAML("")))
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if cfg.DecisionLog != (DecisionLogConfig{}) {
		t.Errorf("DecisionLog = %+v, want the zero value: an absent section disables the log", cfg.DecisionLog)
	}
}

func TestDecisionLogParse(t *testing.T) {
	for _, c := range []struct {
		name, yaml string
		want       DecisionLogConfig
	}{
		{"path only gets the default size", "decision_log:\n  path: logs/decisions.jsonl\n",
			DecisionLogConfig{Path: "logs/decisions.jsonl", MaxBytes: 104857600}},
		{"explicit size", "decision_log:\n  path: d.jsonl\n  max_bytes: 4096\n",
			DecisionLogConfig{Path: "d.jsonl", MaxBytes: 4096}},
	} {
		cfg, err := Parse([]byte(accountingYAML(c.yaml)))
		if err != nil {
			t.Errorf("%s: Parse() error = %v", c.name, err)
			continue
		}
		if cfg.DecisionLog != c.want {
			t.Errorf("%s: DecisionLog = %+v, want %+v", c.name, cfg.DecisionLog, c.want)
		}
	}
}

func TestDecisionLogRejects(t *testing.T) {
	for _, c := range []struct{ name, yaml, want string }{
		{"negative size", "decision_log:\n  path: d.jsonl\n  max_bytes: -1\n", "decision_log.max_bytes"},
		{"size without a path", "decision_log:\n  max_bytes: 100\n", "decision_log.max_bytes"},
		{"unknown key", "decision_log:\n  path: d.jsonl\n  rotate: daily\n", "rotate"},
	} {
		_, err := Parse([]byte(accountingYAML(c.yaml)))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Parse() error = %v, want it to mention %q", c.name, err, c.want)
		}
	}
}
