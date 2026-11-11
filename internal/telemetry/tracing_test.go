package telemetry

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/omorsi45/routepilot/internal/config"
)

// collector is a loopback OTLP/HTTP trace endpoint that remembers what it was sent.
type collector struct {
	srv *httptest.Server

	mu    sync.Mutex
	posts []post
}

type post struct {
	path, contentType string
	body              []byte
}

func newCollector(t *testing.T) *collector {
	t.Helper()
	c := &collector{}
	c.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		c.mu.Lock()
		c.posts = append(c.posts, post{path: r.URL.Path, contentType: r.Header.Get("Content-Type"), body: b})
		c.mu.Unlock()
		w.Header().Set("Content-Type", "application/x-protobuf")
		w.WriteHeader(http.StatusOK) // an empty ExportTraceServiceResponse
	}))
	t.Cleanup(c.srv.Close)
	return c
}

func (c *collector) got() []post {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]post(nil), c.posts...)
}

func ratio(v float64) *float64 { return &v }

func TestNoopWithoutEndpoint(t *testing.T) {
	ctx := context.Background()
	tp, shutdown, err := NewTracerProvider(ctx, config.TracingConfig{})
	if err != nil {
		t.Fatalf("NewTracerProvider: %v", err)
	}
	if _, ok := tp.(noop.TracerProvider); !ok {
		t.Errorf("provider is %T, want the noop provider when no endpoint is configured", tp)
	}
	_, span := tp.Tracer("test").Start(ctx, "work")
	if span.IsRecording() || span.SpanContext().IsValid() {
		t.Errorf("a noop span records or has a context: recording=%v context=%v", span.IsRecording(), span.SpanContext())
	}
	span.End()
	for i := 0; i < 2; i++ { // shutting down twice is harmless
		if err := shutdown(ctx); err != nil {
			t.Errorf("shutdown #%d = %v, want nil", i+1, err)
		}
	}
	// A sample ratio alone does not turn tracing on (config rejects it; a hand-built value is ignored).
	tp, _, err = NewTracerProvider(ctx, config.TracingConfig{SampleRatio: ratio(1)})
	if err != nil {
		t.Fatalf("NewTracerProvider with only a ratio: %v", err)
	}
	if _, ok := tp.(noop.TracerProvider); !ok {
		t.Errorf("provider is %T with only a sample ratio, want noop", tp)
	}
}

func TestSpansReachTheEndpoint(t *testing.T) {
	col := newCollector(t)
	ctx := context.Background()
	tp, shutdown, err := NewTracerProvider(ctx, config.TracingConfig{OTLPEndpoint: col.srv.URL})
	if err != nil {
		t.Fatalf("NewTracerProvider: %v", err)
	}
	_, span := tp.Tracer("test").Start(ctx, "unit-test-span")
	if !span.IsRecording() {
		t.Fatal("a span is not recorded at the default ratio of 1")
	}
	span.End()

	flush, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if err := shutdown(flush); err != nil {
		t.Fatalf("shutdown: %v", err)
	}
	posts := col.got()
	if len(posts) != 1 {
		t.Fatalf("collector got %d posts, want 1", len(posts))
	}
	p := posts[0]
	if p.path != "/v1/traces" || p.contentType != "application/x-protobuf" {
		t.Errorf("post to %q as %q, want /v1/traces as application/x-protobuf", p.path, p.contentType)
	}
	for _, want := range []string{"unit-test-span", "routepilot"} { // the span name and service.name
		if !strings.Contains(string(p.body), want) {
			t.Errorf("exported body lacks %q", want)
		}
	}
}

func TestEndpointPathIsKept(t *testing.T) {
	col := newCollector(t)
	ctx := context.Background()
	tp, shutdown, err := NewTracerProvider(ctx, config.TracingConfig{OTLPEndpoint: col.srv.URL + "/custom/ingest"})
	if err != nil {
		t.Fatal(err)
	}
	_, span := tp.Tracer("test").Start(ctx, "s")
	span.End()
	if err := shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if posts := col.got(); len(posts) != 1 || posts[0].path != "/custom/ingest" {
		t.Errorf("posts = %+v, want one to /custom/ingest", posts)
	}
}

func TestSampleRatio(t *testing.T) {
	ctx := context.Background()
	col := newCollector(t)
	start := func(r *float64) (trace.TracerProvider, func(context.Context) error) {
		tp, shutdown, err := NewTracerProvider(ctx, config.TracingConfig{OTLPEndpoint: col.srv.URL, SampleRatio: r})
		if err != nil {
			t.Fatal(err)
		}
		return tp, shutdown
	}

	tp, shutdown := start(ratio(0))
	_, span := tp.Tracer("t").Start(ctx, "root")
	if span.IsRecording() || span.SpanContext().IsSampled() {
		t.Error("a root span is sampled at ratio 0")
	}
	// A parent that was sampled upstream is followed, whatever the ratio.
	parent := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID: trace.TraceID{1}, SpanID: trace.SpanID{2}, TraceFlags: trace.FlagsSampled, Remote: true,
	})
	_, child := tp.Tracer("t").Start(trace.ContextWithRemoteSpanContext(ctx, parent), "child")
	if !child.IsRecording() {
		t.Error("a child of a sampled remote parent is not recorded at ratio 0 (ParentBased)")
	}
	unsampled := trace.NewSpanContext(trace.SpanContextConfig{TraceID: trace.TraceID{1}, SpanID: trace.SpanID{2}, Remote: true})
	_, dropped := tp.Tracer("t").Start(trace.ContextWithRemoteSpanContext(ctx, unsampled), "dropped")
	if dropped.IsRecording() {
		t.Error("a child of an unsampled remote parent is recorded")
	}
	_ = shutdown(ctx)

	tp, shutdown = start(ratio(1))
	_, span = tp.Tracer("t").Start(ctx, "root")
	if !span.IsRecording() {
		t.Error("a root span is not sampled at ratio 1")
	}
	_ = shutdown(ctx)

	tp, shutdown = start(nil)
	_, span = tp.Tracer("t").Start(ctx, "root")
	if !span.IsRecording() {
		t.Error("a root span is not sampled when the ratio is not set")
	}
	_ = shutdown(ctx)
}

func TestBadEndpointIsAnErrorWithoutTheSecret(t *testing.T) {
	for _, ep := range []string{"localhost:4318", "grpc://localhost:4317", "http://", "http://user:hunter2@localhost:4318", "://bad"} {
		tp, shutdown, err := NewTracerProvider(context.Background(), config.TracingConfig{OTLPEndpoint: ep})
		if err == nil {
			t.Errorf("endpoint %q was accepted", ep)
			continue
		}
		if tp != nil || shutdown != nil {
			t.Errorf("endpoint %q: got a provider with the error", ep)
		}
		if strings.Contains(err.Error(), "hunter2") {
			t.Errorf("endpoint %q: the error repeats the password: %v", ep, err)
		}
	}
}
