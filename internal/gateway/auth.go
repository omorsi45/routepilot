package gateway

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"net/http"
	"strings"

	"github.com/omorsi45/routepilot/internal/config"
)

// authenticate resolves the tenant for a request from its Authorization
// header. The key is hashed and compared against every configured hash in
// constant time per comparison, without stopping at the first match, so the
// time taken does not reveal which tenant (if any) matched.
func (s *Server) authenticate(r *http.Request) (*config.TenantConfig, bool) {
	key, ok := bearerToken(r.Header.Get("Authorization"))
	if !ok {
		return nil, false
	}
	sum := sha256.Sum256([]byte(key))
	got := []byte(hex.EncodeToString(sum[:]))

	var found *config.TenantConfig
	for i := range s.cfg.Tenants {
		t := &s.cfg.Tenants[i]
		if subtle.ConstantTimeCompare(got, []byte(t.APIKeySHA256)) == 1 {
			found = t
		}
	}
	return found, found != nil
}

// bearerToken extracts the token from an "Authorization: Bearer <token>"
// header. The scheme is case-insensitive.
func bearerToken(header string) (string, bool) {
	scheme, token, ok := strings.Cut(header, " ")
	if !ok || !strings.EqualFold(scheme, "Bearer") {
		return "", false
	}
	token = strings.TrimSpace(token)
	return token, token != ""
}
