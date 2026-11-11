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
		{"explicit size", "decision_log:\n  path: d.jsonl\n  max_bytes: 2097152\n",
			DecisionLogConfig{Path: "d.jsonl", MaxBytes: 2097152}},
		{"the smallest size", "decision_log:\n  path: d.jsonl\n  max_bytes: 1048576\n",
			DecisionLogConfig{Path: "d.jsonl", MaxBytes: 1048576}},
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
		{"size below 1 MiB", "decision_log:\n  path: d.jsonl\n  max_bytes: 1048575\n", "1 MiB"},
		{"one byte", "decision_log:\n  path: d.jsonl\n  max_bytes: 1\n", "1 MiB"},
		{"size without a path", "decision_log:\n  max_bytes: 100\n", "decision_log.max_bytes"},
		{"unknown key", "decision_log:\n  path: d.jsonl\n  rotate: daily\n", "rotate"},
	} {
		_, err := Parse([]byte(accountingYAML(c.yaml)))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: Parse() error = %v, want it to mention %q", c.name, err, c.want)
		}
	}
}
