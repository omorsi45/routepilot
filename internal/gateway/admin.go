package gateway

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/omorsi45/routepilot/internal/accounting"
	"github.com/omorsi45/routepilot/internal/openai"
)

// The admin routes. They exist only when server.admin_key_sha256 is configured.
const (
	adminPoliciesPath = "/admin/policies"
	adminRollbackPath = "/admin/policies/rollback"
)

// parseAdminKey reads server.admin_key_sha256 into the digest the admin key is compared with.
// An empty value is no admin key. The value is not repeated in the error: it is the hash of
// a credential.
func parseAdminKey(hexHash string) (digest [sha256.Size]byte, enabled bool, err error) {
	if hexHash == "" {
		return digest, false, nil
	}
	b, decodeErr := hex.DecodeString(hexHash)
	if decodeErr != nil || len(b) != sha256.Size {
		return digest, false, errors.New("server.admin_key_sha256 must be 64 hex characters")
	}
	copy(digest[:], b)
	return digest, true, nil
}

// authenticateAdmin reports whether the request carries the admin key as a bearer token. The
// token is hashed and the digests are compared in constant time, so the time taken does not
// reveal how much of a guess was right. Tenant keys are not looked at: they open nothing here.
func (s *Server) authenticateAdmin(r *http.Request) bool {
	key, ok := bearerToken(r.Header.Get("Authorization"))
	if !ok {
		return false
	}
	sum := sha256.Sum256([]byte(key))
	return subtle.ConstantTimeCompare(sum[:], s.adminKey[:]) == 1
}

// errRouteNotFound is the answer for a path with no route.
func errRouteNotFound(path string) *openai.APIError {
	return &openai.APIError{
		Status: http.StatusNotFound, Type: "invalid_request_error", Code: "not_found",
		Message: fmt.Sprintf("no route for %s", path),
	}
}

// handleAdmin serves GET /admin/policies and POST /admin/policies/rollback.
//
// Without an admin key the routes answer exactly as an unknown path does, to anyone. With one,
// the bearer token must be that key (401 otherwise, before the method is looked at), and the
// gateway must route with the learned policy (404 otherwise).
func (s *Server) handleAdmin(w http.ResponseWriter, rs *reqState, r *http.Request) {
	if !s.adminOn {
		s.fail(w, rs, errRouteNotFound(r.URL.Path))
		return
	}
	if !s.authenticateAdmin(r) {
		s.fail(w, rs, openai.ErrInvalidAPIKey())
		return
	}
	if s.policies == nil {
		e := errRouteNotFound(r.URL.Path)
		e.Message = "the admin policy routes need routing.policy learned"
		s.fail(w, rs, e)
		return
	}
	w.Header().Set("Cache-Control", "no-store")
	switch r.URL.Path {
	case adminPoliciesPath:
		if r.Method != http.MethodGet {
			s.methodNotAllowed(w, rs, http.MethodGet)
			return
		}
		s.writeAdminSnapshot(w)
	case adminRollbackPath:
		if r.Method != http.MethodPost {
			s.methodNotAllowed(w, rs, http.MethodPost)
			return
		}
		s.adminRollback(w, rs)
	}
}

// adminSnapshot is the JSON the admin routes answer with: the versions loaded in each role,
// the canary's share, which snapshot of the registry this is and when it was loaded, and how
// the reloads have gone.
type adminSnapshot struct {
	StableVersion  string    `json:"stable_version"`
	CanaryVersion  *string   `json:"canary_version"`
	ShadowVersions []string  `json:"shadow_versions"`
	CanaryPercent  float64   `json:"canary_percent"`
	Generation     uint64    `json:"generation"`
	LoadedAt       time.Time `json:"loaded_at"`
	ReloadErrors   uint64    `json:"reload_errors"`
	LastError      *string   `json:"last_error"`
}

func (s *Server) adminSnapshot() adminSnapshot {
	snap := s.policies.Snapshot()
	v := adminSnapshot{
		StableVersion:  snap.Stable.Version,
		ShadowVersions: make([]string, 0, len(snap.Shadow)),
		CanaryPercent:  snap.CanaryPercent,
		Generation:     snap.Generation,
		LoadedAt:       snap.LoadedAt.UTC(),
		ReloadErrors:   s.policies.ReloadErrors(),
	}
	if snap.Canary != nil {
		version := snap.Canary.Version
		v.CanaryVersion = &version
	}
	for _, sh := range snap.Shadow {
		v.ShadowVersions = append(v.ShadowVersions, sh.Version)
	}
	if err := s.policies.LastError(); err != nil {
		msg := err.Error()
		v.LastError = &msg
	}
	return v
}

func (s *Server) writeAdminSnapshot(w http.ResponseWriter) {
	writeJSON(w, http.StatusOK, s.adminSnapshot())
}

// adminRollback pulls the canary when it is taking traffic, and otherwise puts the previous
// stable policy back. Either way it writes a rollback event by the admin and answers with the
// new snapshot. With nothing to roll back it answers 409.
func (s *Server) adminRollback(w http.ResponseWriter, rs *reqState) {
	snap := s.policies.Snapshot()
	if canaryActive(snap) {
		if old := s.policies.RollbackCanary(); old > 0 {
			s.disarmGuard()
			s.recordRollout(accounting.RolloutEvent{
				At: s.now(), Action: accounting.ActionRollback,
				From: snap.Canary.Version, To: snap.Stable.Version,
				Percent: 0, Reason: "manual rollback of the canary", Actor: actorAdmin,
			})
			s.logger.LogAttrs(context.Background(), slog.LevelWarn, "canary rolled back through the admin API",
				slog.String("canary", snap.Canary.Version), slog.String("stable", snap.Stable.Version),
				slog.Float64("previous_percent", old))
		}
		s.writeAdminSnapshot(w)
		return
	}

	from, to, err := s.policies.RollbackStable()
	if err != nil {
		s.fail(w, rs, &openai.APIError{
			Status: http.StatusConflict, Type: "invalid_request_error", Code: "no_rollback_target",
			Message: "nothing to roll back: the canary has no share of the traffic and there is no previous stable policy",
		})
		return
	}
	s.recordRollout(accounting.RolloutEvent{
		At: s.now(), Action: accounting.ActionRollback,
		From: from, To: to,
		// The stable policy serves all the traffic the canary does not take, and it is the
		// whole of it here.
		Percent: 100, Reason: "manual rollback of the stable policy", Actor: actorAdmin,
	})
	s.logger.LogAttrs(context.Background(), slog.LevelWarn, "stable policy rolled back through the admin API",
		slog.String("from", from), slog.String("to", to))
	s.writeAdminSnapshot(w)
}
