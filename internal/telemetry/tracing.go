package telemetry

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
	"go.opentelemetry.io/otel/trace/noop"

	"github.com/omorsi45/routepilot/internal/config"
)

// ServiceName is the service.name resource attribute of the exported traces.
const ServiceName = "routepilot"

// tracesURL is the URL spans are posted to: the endpoint as given, or with the standard OTLP
// path /v1/traces when it names only a host and port. WithEndpointURL uses the URL as is,
// so without this a bare collector address would be posted to "/".
func tracesURL(endpoint string) string {
	u, err := url.Parse(endpoint)
	if err != nil || (u.Path != "" && u.Path != "/") {
		return endpoint
	}
	u.Path = "/v1/traces"
	return u.String()
}

// NewTracerProvider builds the tracer provider the gateway records its spans with, and the
// func that flushes and stops it. With no otlp_endpoint it is the noop provider: no span is
// recorded, nothing is sent and nothing needs shutting down. With one, spans are batched to
// that OTLP/HTTP endpoint, a new trace is recorded for sample_ratio of the requests, and a
// request that arrives with a sampling decision of its own keeps it (ParentBased). Creating
// the exporter does not connect, so a collector that is down does not stop the server from
// starting; the batcher drops what it cannot deliver.
//
// The package never sets an attribute itself: what a span carries is decided where the span
// is made, and the gateway puts identifiers, numbers and outcome names on its spans, never
// message text or key material.
func NewTracerProvider(ctx context.Context, cfg config.TracingConfig) (trace.TracerProvider, func(context.Context) error, error) {
	if cfg.OTLPEndpoint == "" {
		return noop.NewTracerProvider(), func(context.Context) error { return nil }, nil
	}
	if err := config.CheckOTLPEndpoint(cfg.OTLPEndpoint); err != nil {
		return nil, nil, fmt.Errorf("telemetry: otlp_endpoint %w", err)
	}
	exp, err := otlptracehttp.New(ctx, otlptracehttp.WithEndpointURL(tracesURL(cfg.OTLPEndpoint)))
	if err != nil {
		// The exporter's error can quote the endpoint, whose path may be a secret.
		msg := strings.ReplaceAll(err.Error(), cfg.OTLPEndpoint, "<otlp_endpoint>")
		return nil, nil, fmt.Errorf("telemetry: create the OTLP trace exporter: %s", msg)
	}
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(cfg.Ratio()))),
		sdktrace.WithResource(resource.NewSchemaless(attribute.String("service.name", ServiceName))),
	)
	return tp, tp.Shutdown, nil
}
