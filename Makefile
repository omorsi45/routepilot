# RoutePilot developer entry points.
#
# Everything here is a thin wrapper over the `go` commands it runs, so nothing
# is hidden: `make test` is `go test -count=1 ./...`. The targets exist so the
# same commands run locally and in CI.

SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: help
help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## build the CLI into bin/routepilot
	go build -o bin/routepilot ./cmd/routepilot

.PHONY: fmt
fmt: ## format all Go source
	gofmt -w .

.PHONY: check
check: ## gofmt check, vet and build, the way CI runs them
	@unformatted="$$(gofmt -l .)"; \
	if [ -n "$$unformatted" ]; then echo "not gofmt-clean:"; echo "$$unformatted"; exit 1; fi
	go vet ./...
	go build ./...

.PHONY: test
test: ## run the hermetic test suite (loopback only, no network)
	go test -count=1 ./...

.PHONY: test-race
test-race: ## run the suite under the race detector
	go test -race -count=1 ./...

.PHONY: verify
verify: check test test-race ## everything the CI lane runs

.PHONY: clean
clean: ## remove build output
	rm -rf bin
