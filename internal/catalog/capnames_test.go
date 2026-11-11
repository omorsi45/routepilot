package catalog

import (
	"fmt"
	"testing"
)

func TestCapSetNames(t *testing.T) {
	var all CapSet
	for c := Capability(0); c < numCaps; c++ {
		all = all.with(c)
	}
	for _, c := range []struct {
		name string
		set  CapSet
		want string
	}{
		{"empty", 0, "[]"},
		{"tools", CapSet(0).with(CapTools), "[tools]"},
		{"in constant order", CapSet(0).with(CapStreaming).with(CapVision).with(CapJSONObject), "[json_object vision streaming]"},
		{"all", all, "[tools json_object json_schema vision streaming]"},
	} {
		got := c.set.Names()
		if got == nil {
			t.Errorf("%s: Names() = nil, want an empty slice that encodes as []", c.name)
		}
		if fmt.Sprint(got) != c.want {
			t.Errorf("%s: Names() = %v, want %s", c.name, got, c.want)
		}
	}
	if got := CapSet(0).String(); got != "none" {
		t.Errorf("empty set String() = %q, want none", got)
	}
}
