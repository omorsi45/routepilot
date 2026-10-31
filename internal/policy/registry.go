package policy

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/omorsi45/routepilot/internal/config"
)

// The role of a Slot in a Snapshot.
const (
	// RoleStable is the policy that serves every request the canary does not take.
	RoleStable = "stable"
	// RoleCanary is the policy that serves a configured share of the requests.
	RoleCanary = "canary"
	// RoleShadow is a policy that decides on every request but never acts.
	RoleShadow = "shadow"
)

// DefaultPollEvery is how often Watch checks the policy files when it is given no
// positive interval.
const DefaultPollEvery = 5 * time.Second

// Slot is one loaded policy: a verified artifact with the optimizer configuration it
// runs under.
type Slot struct {
	// Role is RoleStable, RoleCanary or RoleShadow.
	Role     string
	Artifact *Artifact
	Config   Config
	// Version is the artifact version and the first 8 hex digits of the configuration
	// hash, joined with "+". It is what the policy-version response header reports.
	Version string
}

// Snapshot is the set of loaded policies at one moment. It is immutable once published:
// a reload builds a new Snapshot and swaps the registry's pointer, so a request that took
// a Snapshot sees one consistent set of slots from start to finish.
type Snapshot struct {
	Stable Slot
	// Canary is nil when no canary is configured. CanaryPercent is the share of requests,
	// in [0, 100], that Pick sends to it.
	Canary        *Slot
	CanaryPercent float64
	// Shadow lists the policies that decide without acting, in configuration order.
	Shadow []Slot
	// Generation counts the snapshots the registry has published, starting at 1.
	Generation uint64
	LoadedAt   time.Time
}

// canaryBuckets is the resolution of the canary split: percent is in hundredths of a
// percent over this many buckets.
const canaryBuckets = 10000

// Pick returns the slot that serves the request: the canary for a fixed share of the
// (tenant, request ID) space, the stable slot for the rest. The share is
// fnv64a(tenant + requestID) % 10000 < percent * 100, so the same request always gets
// the same slot and a growing percent only adds requests to the canary.
func (s *Snapshot) Pick(tenant, requestID string) Slot {
	if s.Canary != nil && s.CanaryPercent > 0 &&
		fnv64a(tenant, requestID)%canaryBuckets < uint64(s.CanaryPercent*100) {
		return *s.Canary
	}
	return s.Stable
}

// fnv64a is the 64-bit FNV-1a hash of a followed by b, without building the joined string.
func fnv64a(a, b string) uint64 {
	const (
		offset = 14695981039346656037
		prime  = 1099511628211
	)
	h := uint64(offset)
	for i := 0; i < len(a); i++ {
		h ^= uint64(a[i])
		h *= prime
	}
	for i := 0; i < len(b); i++ {
		h ^= uint64(b[i])
		h *= prime
	}
	return h
}

// fileStamp is what the poller compares to notice that a file changed.
type fileStamp struct {
	exists bool
	size   int64
	mod    int64 // modification time in Unix nanoseconds
}

func statFile(path string) fileStamp {
	fi, err := os.Stat(path)
	if err != nil {
		return fileStamp{}
	}
	return fileStamp{exists: true, size: fi.Size(), mod: fi.ModTime().UnixNano()}
}

// Registry holds the policies a learned routing policy serves with and swaps them
// atomically: a reload loads every slot fresh, and only when all of them verify does the
// new Snapshot replace the old one. A reload that fails for any reason (an unreadable or
// corrupt artifact, a bad configuration, a file half written) keeps the old Snapshot
// serving and is counted. It is safe for concurrent use.
type Registry struct {
	load    func() (config.PoliciesConfig, error)
	baseDir string
	now     func() time.Time

	cur          atomic.Pointer[Snapshot]
	reloadErrors atomic.Uint64
	rollbacks    atomic.Uint64
	lastErr      atomic.Pointer[error]

	// mu serializes reloads, polls, rollbacks and canary changes; Snapshot never takes it.
	mu      sync.Mutex
	prev    *Slot                 // the stable slot before the last reload that changed it
	cfgFile string                // the config file the loader reads, polled for changes
	lastCfg config.PoliciesConfig // the latest configuration the loader returned
	stamps  map[string]fileStamp  // the watched files as they were at the latest attempt
	logger  *slog.Logger
	// catalog is the model ids a slot is checked against (WithCatalog); empty checks nothing.
	catalog []string
}

// RegistryOption configures a Registry at construction.
type RegistryOption func(*Registry)

// WithCatalog gives the registry the ids of the models the gateway routes among. A slot whose
// artifact predicts for none of them is refused, as a load error (so a reload to it keeps the
// old snapshot): the router would find no model it can score and answer every automatic
// request with a capability error. A slot that covers only some of them loads, and the
// registry logs which models it leaves out, because those are never chosen automatically.
// Without it, or with no ids, nothing is checked.
func WithCatalog(modelIDs []string) RegistryOption {
	return func(r *Registry) { r.catalog = append([]string(nil), modelIDs...) }
}

// WithLogger sets where the registry logs from its first load on; SetLogger does the same
// later.
func WithLogger(l *slog.Logger) RegistryOption {
	return func(r *Registry) { r.logger = l }
}

// NewRegistry loads and verifies every policy load() names, resolving relative paths
// against baseDir. It fails when the stable slot (or any other) does not load; no registry
// is returned then.
func NewRegistry(load func() (config.PoliciesConfig, error), baseDir string, opts ...RegistryOption) (*Registry, error) {
	if load == nil {
		return nil, errors.New("policy registry: load function is nil")
	}
	r := &Registry{load: load, baseDir: baseDir, now: time.Now}
	for _, o := range opts {
		o(r)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	snap, err := r.build(0)
	if err != nil {
		return nil, err
	}
	r.cur.Store(snap)
	return r, nil
}

// SetLogger sets where reloads and rollbacks are logged. Nil, the default, logs nothing.
func (r *Registry) SetLogger(l *slog.Logger) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.logger = l
}

// SetConfigFile adds the config file the loader reads to the files Watch polls, so an
// edit to the policies section is noticed. Call it once, right after NewRegistry.
func (r *Registry) SetConfigFile(path string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cfgFile = path
	r.stamps = r.stampAll(r.lastCfg, statFile(path))
}

// Snapshot returns the current set of policies. It never returns nil and never blocks.
func (r *Registry) Snapshot() *Snapshot { return r.cur.Load() }

// ReloadErrors counts the reloads that failed and left the old snapshot in place.
func (r *Registry) ReloadErrors() uint64 { return r.reloadErrors.Load() }

// Rollbacks counts the rollbacks that happened: of the stable slot (RollbackStable) and of the
// canary (RollbackCanary). A refused or empty one is not counted.
func (r *Registry) Rollbacks() uint64 { return r.rollbacks.Load() }

// LastError is the error of the latest reload when it failed, nil after a reload that
// succeeded (and before any failed).
func (r *Registry) LastError() error {
	if p := r.lastErr.Load(); p != nil {
		return *p
	}
	return nil
}

// Reload loads every slot fresh. If any step fails the old snapshot keeps serving,
// ReloadErrors goes up by one, LastError records the error and Reload returns it.
func (r *Registry) Reload() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.reloadLocked()
}

func (r *Registry) reloadLocked() error {
	old := r.cur.Load()
	snap, err := r.build(old.Generation)
	if err != nil {
		r.reloadErrors.Add(1)
		r.lastErr.Store(&err)
		if r.logger != nil {
			r.logger.Warn("policy reload failed, keeping the current policies",
				slog.String("error", err.Error()), slog.String("stable", old.Stable.Version),
				slog.Uint64("generation", old.Generation))
		}
		return err
	}
	if snap.Stable.Version != old.Stable.Version {
		prev := old.Stable
		r.prev = &prev
	}
	r.cur.Store(snap)
	r.lastErr.Store(nil)
	if r.logger != nil {
		attrs := []slog.Attr{
			slog.Uint64("generation", snap.Generation),
			slog.String("stable", snap.Stable.Version),
			slog.Int("shadow", len(snap.Shadow)),
		}
		if snap.Canary != nil {
			attrs = append(attrs, slog.String("canary", snap.Canary.Version), slog.Float64("canary_percent", snap.CanaryPercent))
		}
		r.logger.LogAttrs(context.Background(), slog.LevelInfo, "policies reloaded", attrs...)
	}
	return nil
}

// Watch polls the config file and every file the policies refer to (each slot's
// manifest.json, weights.bin and policy configuration) every `every`, and reloads when any
// of them changed in size or modification time or appeared or disappeared. A reload that
// fails is not retried until a file changes again. It returns when ctx is done.
func (r *Registry) Watch(ctx context.Context, every time.Duration) {
	if every <= 0 {
		every = DefaultPollEvery
	}
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			r.poll()
		}
	}
}

func (r *Registry) poll() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for path, was := range r.stamps {
		if statFile(path) != was {
			_ = r.reloadLocked() // logged and counted there
			return
		}
	}
}

// SetCanaryPercent changes the share of requests the canary serves, clamped to [0, 100]
// (not-a-number is 0), and returns the old share. It publishes a new snapshot. A later
// reload takes the percent from the configuration again.
func (r *Registry) SetCanaryPercent(p float64) (old float64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch {
	case math.IsNaN(p) || p < 0:
		p = 0
	case p > 100:
		p = 100
	}
	cur := r.cur.Load()
	next := *cur
	next.CanaryPercent = p
	next.Generation++
	r.cur.Store(&next)
	return cur.CanaryPercent
}

// RollbackCanary stops the canary by taking its share of the requests to 0 (through
// SetCanaryPercent) and returns the share it had. The canary slot stays loaded, so the
// rollout can be resumed by raising the share again. A canary that already had no share, or
// no canary at all, is not a rollback: nothing is published, nothing is counted and 0 is
// returned. Like SetCanaryPercent, it lives in memory until the next reload.
func (r *Registry) RollbackCanary() (old float64) {
	if cur := r.Snapshot(); cur.Canary == nil || !(cur.CanaryPercent > 0) {
		return 0
	}
	old = r.SetCanaryPercent(0)
	if old > 0 {
		r.rollbacks.Add(1)
	}
	return old
}

// RollbackStable replaces the stable slot with the one it had before the last reload that
// changed it, and returns the two slot versions. It fails when there is no such slot: the
// registry keeps exactly one, and rolling back spends it. The canary and shadow slots stay.
//
// The rollback lives in memory. The next reload loads whatever the configuration names, so
// make the rollback durable by editing the configuration.
func (r *Registry) RollbackStable() (from, to string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.prev == nil {
		return "", "", errors.New("policy registry: no previous stable policy to roll back to")
	}
	cur := r.cur.Load()
	next := *cur
	next.Stable = *r.prev
	next.Generation++
	from, to = cur.Stable.Version, r.prev.Version
	r.prev = nil
	r.cur.Store(&next)
	r.rollbacks.Add(1)
	if r.logger != nil {
		r.logger.Warn("stable policy rolled back", slog.String("from", from), slog.String("to", to))
	}
	return from, to, nil
}

// build loads a snapshot from the loader's current answer. It records the files it will
// read and their stamps first, so a change during the load is noticed by the next poll,
// and does so even when the load fails, so a bad file is not retried until it changes.
func (r *Registry) build(prevGeneration uint64) (*Snapshot, error) {
	// The config file is stamped before the loader reads it for the same reason.
	cfgStamp := statFile(r.cfgFile)
	cfg, err := r.load()
	if err != nil {
		r.stamps = r.stampAll(r.lastCfg, cfgStamp)
		return nil, fmt.Errorf("policy registry: load policies config: %w", err)
	}
	r.lastCfg = cfg
	r.stamps = r.stampAll(cfg, cfgStamp)

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("policy registry: %w", err)
	}
	if cfg.Stable == (config.PolicyRef{}) {
		return nil, errors.New("policy registry: policies.stable is required")
	}

	stable, err := r.loadSlot(RoleStable, "stable", cfg.Root, cfg.Stable)
	if err != nil {
		return nil, err
	}
	snap := &Snapshot{Stable: stable, Generation: prevGeneration + 1, LoadedAt: r.now()}
	if c := cfg.Canary; c != nil {
		slot, err := r.loadSlot(RoleCanary, "canary", cfg.Root, c.PolicyRef)
		if err != nil {
			return nil, err
		}
		snap.Canary, snap.CanaryPercent = &slot, c.Percent
	}
	for i, ref := range cfg.Shadow {
		slot, err := r.loadSlot(RoleShadow, fmt.Sprintf("shadow[%d]", i), cfg.Root, ref)
		if err != nil {
			return nil, err
		}
		snap.Shadow = append(snap.Shadow, slot)
	}
	return snap, nil
}

// artifactDir is <root>/<name>/<version>, with root resolved against the base directory.
func (r *Registry) artifactDir(root string, ref config.PolicyRef) string {
	if root == "" {
		root = "policies"
	}
	return filepath.Join(config.ResolvePath(r.baseDir, root), ref.Name, ref.Version)
}

func (r *Registry) loadSlot(role, label, root string, ref config.PolicyRef) (Slot, error) {
	art, err := Load(r.artifactDir(root, ref))
	if err != nil {
		return Slot{}, fmt.Errorf("policy registry: %s policy %s/%s: %w", label, ref.Name, ref.Version, err)
	}
	pc, err := LoadConfig(config.ResolvePath(r.baseDir, ref.Config))
	if err != nil {
		return Slot{}, fmt.Errorf("policy registry: %s policy %s/%s: %w", label, ref.Name, ref.Version, err)
	}
	if err := r.checkCoverage(label, ref, art); err != nil {
		return Slot{}, err
	}
	return Slot{Role: role, Artifact: art, Config: pc, Version: art.Version() + "+" + pc.Hash8()}, nil
}

// checkCoverage compares the artifact's models with the catalog: none in common is an error,
// some missing is logged. It checks nothing without a catalog.
func (r *Registry) checkCoverage(label string, ref config.PolicyRef, art *Artifact) error {
	if len(r.catalog) == 0 {
		return nil
	}
	var missing []string
	for _, id := range r.catalog {
		if !art.HasModel(id) {
			missing = append(missing, id)
		}
	}
	switch {
	case len(missing) == len(r.catalog):
		return fmt.Errorf("policy registry: %s policy %s/%s covers none of the catalog's models (it predicts for %s, the catalog has %s)",
			label, ref.Name, ref.Version, strings.Join(art.Models(), ", "), strings.Join(r.catalog, ", "))
	case len(missing) > 0 && r.logger != nil:
		r.logger.Warn("a policy does not cover every model of the catalog, so those models are never chosen automatically",
			slog.String("slot", label), slog.String("policy", ref.Name+"/"+ref.Version),
			slog.String("uncovered", strings.Join(missing, ", ")))
	}
	return nil
}

// stampAll stamps the config file (with cfgStamp, taken before the loader read it) and
// every file cfg refers to.
func (r *Registry) stampAll(cfg config.PoliciesConfig, cfgStamp fileStamp) map[string]fileStamp {
	stamps := make(map[string]fileStamp)
	if r.cfgFile != "" {
		stamps[r.cfgFile] = cfgStamp
	}
	refs := []config.PolicyRef{cfg.Stable}
	if cfg.Canary != nil {
		refs = append(refs, cfg.Canary.PolicyRef)
	}
	refs = append(refs, cfg.Shadow...)
	for _, ref := range refs {
		if ref == (config.PolicyRef{}) {
			continue
		}
		dir := r.artifactDir(cfg.Root, ref)
		for _, p := range []string{
			filepath.Join(dir, "manifest.json"),
			filepath.Join(dir, "weights.bin"),
			config.ResolvePath(r.baseDir, ref.Config),
		} {
			stamps[p] = statFile(p)
		}
	}
	return stamps
}
