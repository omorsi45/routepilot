FROM golang:1.27.2-bookworm AS build

WORKDIR /src
ENV CGO_ENABLED=0

# Dependencies first, so a source-only change reuses the download layer.
COPY go.mod go.sum ./
RUN go mod download

COPY cmd ./cmd
COPY internal ./internal

# VERSION lands in "routepilot version"; it stays "dev" unless the build passes --build-arg.
ARG VERSION=dev
RUN go build -trimpath \
        -ldflags "-s -w -X github.com/omorsi45/routepilot/internal/version.Version=${VERSION}" \
        -o /out/routepilot ./cmd/routepilot \
 && go build -trimpath -ldflags "-s -w" -o /out/rp-mock ./cmd/rp-mock

# Static binaries on a distroless base: no shell, no package manager, runs as uid 65532.
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/routepilot /routepilot
COPY --from=build /out/rp-mock /rp-mock

EXPOSE 8080
ENTRYPOINT ["/routepilot"]
