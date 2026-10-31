// Package policy loads a trained routing policy artifact and scores requests with it.
//
// An artifact is a directory <name>/<version>/ holding manifest.json and weights.bin,
// written by python/routepilot_train/export. The version is the first 12 hex digits of
// the SHA-256 of the manifest bytes (with "version" emptied) followed by the weights
// bytes, so the directory name proves the content. Load refuses an artifact whose hash,
// format, feature spec version, array references or golden predictions do not check out,
// so a half-written or edited file never serves traffic.
package policy

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"

	"github.com/omorsi45/routepilot/internal/features"
)

const (
	formatName    = "routepilot-policy-artifact"
	formatVersion = 1
	versionLen    = 12
)

// Each error names the check that refused the artifact; Load wraps them with detail.
var (
	// ErrFormat means the manifest is not a routepilot policy artifact of a known format.
	ErrFormat = errors.New("format check failed")
	// ErrSpecVersion means the artifact was trained on another feature spec.
	ErrSpecVersion = errors.New("feature spec version check failed")
	// ErrHashMismatch means the content does not hash to the version the manifest claims.
	ErrHashMismatch = errors.New("hash check failed")
	// ErrDirName means the directory is not named after the version.
	ErrDirName = errors.New("directory name check failed")
	// ErrArrayRef means a weights.bin array is out of bounds, of the wrong length or type,
	// or holds values the predictor cannot use.
	ErrArrayRef = errors.New("array reference check failed")
	// ErrManifest means the manifest content is malformed.
	ErrManifest = errors.New("manifest check failed")
	// ErrGolden means a golden prediction recorded at training time was not reproduced.
	ErrGolden = errors.New("golden self-check failed")
)

var versionPattern = regexp.MustCompile(`^[0-9a-f]{12}$`)

// residualKeys are the length residual quantile levels, in OutQ order.
var residualKeys = [4]string{"0.5", "0.9", "0.95", "0.99"}

type refJSON struct {
	Offset int64  `json:"offset"`
	Len    int64  `json:"len"`
	DType  string `json:"dtype"`
}

type manifestJSON struct {
	Format             string          `json:"format"`
	FormatVersion      int             `json:"format_version"`
	Version            string          `json:"version"`
	FeatureSpecVersion int             `json:"feature_spec_version"`
	Dim                int             `json:"dim"`
	DenseMean          []float64       `json:"dense_mean"`
	DenseStd           []float64       `json:"dense_std"`
	Models             []modelJSON     `json:"models"`
	OOD                oodJSON         `json:"ood"`
	Golden             []goldenJSON    `json:"golden"`
	Training           json.RawMessage `json:"training"`
}

type modelJSON struct {
	ID      string `json:"id"`
	Quality struct {
		Weights refJSON `json:"weights"`
		Bias    float64 `json:"bias"`
	} `json:"quality"`
	Calibrator calibratorJSON `json:"calibrator"`
	LCB        lcbJSON        `json:"lcb"`
	Length     lengthJSON     `json:"length"`
}

type calibratorJSON struct {
	Type string    `json:"type"`
	X    []float64 `json:"x"`
	Y    []float64 `json:"y"`
	A    float64   `json:"a"`
	B    float64   `json:"b"`
}

type lcbJSON struct {
	Upper      []float64 `json:"upper"`
	Lower      []float64 `json:"lower"`
	N          []int     `json:"n"`
	PooledRate float64   `json:"pooled_rate"`
	PooledLCB  float64   `json:"pooled_lcb"`
}

type lengthJSON struct {
	Weights           refJSON            `json:"weights"`
	Bias              float64            `json:"bias"`
	Smear             float64            `json:"smear"`
	ResidualQuantiles map[string]float64 `json:"residual_quantiles"`
	InRatio           float64            `json:"in_ratio"`
}

type oodJSON struct {
	K         int     `json:"k"`
	Threshold float64 `json:"threshold"`
	Centroids []struct {
		Idx refJSON `json:"idx"`
		Val refJSON `json:"val"`
	} `json:"centroids"`
}

type goldenJSON struct {
	Messages [][]string                 `json:"messages"`
	OODScore float64                    `json:"ood_score"`
	Models   map[string]goldenModelJSON `json:"models"`
}

type goldenModelJSON struct {
	P    float64 `json:"p"`
	L    float64 `json:"l"`
	EOut float64 `json:"e_out"`
	Q99  float64 `json:"q99"`
}

// Artifact is a verified, immutable policy. It is safe for concurrent use.
type Artifact struct {
	version  string
	dim      int
	mean     [features.NumDense]float64
	std      [features.NumDense]float64
	models   []modelParams
	byID     map[string]int
	ood      oodParams
	training json.RawMessage
}

type modelParams struct {
	id string

	qualityW    []float32 // dim hashed weights then NumDense dense weights
	qualityBias float64
	cal         calibrator

	lcbUpper   []float64
	lcbLower   []float64
	pooledRate float64
	pooledLCB  float64

	lengthW    []float32
	lengthBias float64
	smear      float64
	resid      [4]float64
	inRatio    float64
}

type oodParams struct {
	threshold float64
	centroids []centroid
}

// centroid is a sparse unit-ish vector: ascending indices, widened float32 values and the
// L2 norm of those values.
type centroid struct {
	idx  []int32
	val  []float64
	norm float64
}

// Load reads the artifact in dir and verifies it. dir must be named after the artifact's
// version. The returned error wraps one of the Err values.
func Load(dir string) (*Artifact, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("load policy %s: %w", dir, err)
	}
	raw, err := os.ReadFile(filepath.Join(abs, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("load policy %s: read manifest: %w", dir, err)
	}
	weights, err := os.ReadFile(filepath.Join(abs, "weights.bin"))
	if err != nil {
		return nil, fmt.Errorf("load policy %s: read weights: %w", dir, err)
	}
	a, err := parse(raw, weights, filepath.Base(abs))
	if err != nil {
		return nil, fmt.Errorf("load policy %s: %w", dir, err)
	}
	return a, nil
}

// parse verifies and decodes an artifact whose directory is named dirName.
func parse(raw, weights []byte, dirName string) (*Artifact, error) {
	var m manifestJSON
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("%w: decode manifest: %v", ErrManifest, err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("%w: data after the manifest object", ErrManifest)
	}

	if m.Format != formatName || m.FormatVersion != formatVersion {
		return nil, fmt.Errorf("%w: format %q version %d, want %q version %d",
			ErrFormat, m.Format, m.FormatVersion, formatName, formatVersion)
	}
	if m.FeatureSpecVersion != features.SpecVersion {
		return nil, fmt.Errorf("%w: artifact uses feature spec %d, this build implements %d",
			ErrSpecVersion, m.FeatureSpecVersion, features.SpecVersion)
	}

	if !versionPattern.MatchString(m.Version) {
		return nil, fmt.Errorf("%w: version %q is not %d lowercase hex digits", ErrManifest, m.Version, versionLen)
	}
	unversioned, err := unversionedManifest(raw, m.Version)
	if err != nil {
		return nil, err
	}
	if got := contentVersion(unversioned, weights); got != m.Version {
		return nil, fmt.Errorf("%w: content hashes to %s but the manifest says %s", ErrHashMismatch, got, m.Version)
	}
	if dirName != m.Version {
		return nil, fmt.Errorf("%w: directory is %q but the version is %s", ErrDirName, dirName, m.Version)
	}

	a, err := build(&m, weights)
	if err != nil {
		return nil, err
	}
	if err := a.selfCheck(m.Golden); err != nil {
		return nil, err
	}
	return a, nil
}

// unversionedManifest returns raw with its single "version": "<version>" replaced by
// "version": "", which is what the version hash covers.
func unversionedManifest(raw []byte, version string) ([]byte, error) {
	needle := []byte(`"version": "` + version + `"`)
	if n := bytes.Count(raw, needle); n != 1 {
		return nil, fmt.Errorf("%w: found %d occurrences of %s in the manifest, want exactly 1", ErrHashMismatch, n, needle)
	}
	return bytes.Replace(raw, needle, []byte(`"version": ""`), 1), nil
}

// contentVersion is the first 12 hex digits of SHA-256(manifest with empty version || weights).
func contentVersion(unversioned, weights []byte) string {
	h := sha256.New()
	h.Write(unversioned)
	h.Write(weights)
	return hex.EncodeToString(h.Sum(nil))[:versionLen]
}

func build(m *manifestJSON, weights []byte) (*Artifact, error) {
	if m.Dim < 1 || m.Dim > math.MaxInt32 {
		return nil, fmt.Errorf("%w: dim %d is outside [1, %d]", ErrManifest, m.Dim, math.MaxInt32)
	}
	a := &Artifact{version: m.Version, dim: m.Dim, byID: make(map[string]int, len(m.Models)), training: m.Training}
	if len(m.DenseMean) != features.NumDense || len(m.DenseStd) != features.NumDense {
		return nil, fmt.Errorf("%w: dense_mean and dense_std need %d values each", ErrManifest, features.NumDense)
	}
	for i := range features.NumDense {
		if !finite(m.DenseMean[i]) || !finite(m.DenseStd[i]) || m.DenseStd[i] <= 0 {
			return nil, fmt.Errorf("%w: dense_mean[%d]=%v dense_std[%d]=%v need finite values and a positive std",
				ErrManifest, i, m.DenseMean[i], i, m.DenseStd[i])
		}
		a.mean[i], a.std[i] = m.DenseMean[i], m.DenseStd[i]
	}
	if len(m.Models) == 0 {
		return nil, fmt.Errorf("%w: no models", ErrManifest)
	}
	wide := int64(m.Dim) + features.NumDense
	for _, mj := range m.Models {
		if mj.ID == "" {
			return nil, fmt.Errorf("%w: a model has an empty id", ErrManifest)
		}
		if _, dup := a.byID[mj.ID]; dup {
			return nil, fmt.Errorf("%w: duplicate model id %q", ErrManifest, mj.ID)
		}
		mp, err := buildModel(mj, weights, wide)
		if err != nil {
			return nil, fmt.Errorf("model %q: %w", mj.ID, err)
		}
		a.byID[mj.ID] = len(a.models)
		a.models = append(a.models, mp)
	}
	o, err := buildOOD(m.OOD, weights, m.Dim)
	if err != nil {
		return nil, err
	}
	a.ood = o
	return a, nil
}

func buildModel(mj modelJSON, weights []byte, wide int64) (modelParams, error) {
	mp := modelParams{id: mj.ID}
	var err error
	if mp.qualityW, err = readF32(weights, mj.Quality.Weights, wide); err != nil {
		return mp, fmt.Errorf("quality weights: %w", err)
	}
	if mp.lengthW, err = readF32(weights, mj.Length.Weights, wide); err != nil {
		return mp, fmt.Errorf("length weights: %w", err)
	}
	mp.qualityBias = mj.Quality.Bias
	mp.lengthBias, mp.smear, mp.inRatio = mj.Length.Bias, mj.Length.Smear, mj.Length.InRatio
	if mp.cal, err = newCalibrator(mj.Calibrator); err != nil {
		return mp, err
	}

	lcb := mj.LCB
	if len(lcb.Upper) == 0 || len(lcb.Upper) != len(lcb.Lower) || len(lcb.Upper) != len(lcb.N) {
		return mp, fmt.Errorf("%w: lcb upper, lower and n need the same non-zero length", ErrManifest)
	}
	for i, u := range lcb.Upper {
		if !finite(u) || !finite(lcb.Lower[i]) || (i > 0 && u < lcb.Upper[i-1]) {
			return mp, fmt.Errorf("%w: lcb upper must be finite and non-decreasing, lower finite", ErrManifest)
		}
		if lcb.Lower[i] < 0 || lcb.Lower[i] > 1 {
			return mp, fmt.Errorf("%w: lcb lower[%d]=%v is outside [0, 1]", ErrManifest, i, lcb.Lower[i])
		}
	}
	// The last bin must reach 1, or a probability above its edge falls off the table.
	if lcb.Upper[len(lcb.Upper)-1] != 1 {
		return mp, fmt.Errorf("%w: lcb upper must end at 1.0, not %v", ErrManifest, lcb.Upper[len(lcb.Upper)-1])
	}
	mp.lcbUpper, mp.lcbLower = lcb.Upper, lcb.Lower
	mp.pooledRate, mp.pooledLCB = lcb.PooledRate, lcb.PooledLCB

	for i, key := range residualKeys {
		r, ok := mj.Length.ResidualQuantiles[key]
		if !ok || !finite(r) {
			return mp, fmt.Errorf("%w: length residual quantile %q is missing or not finite", ErrManifest, key)
		}
		mp.resid[i] = r
	}
	scalars := []struct {
		name string
		v    float64
	}{
		{"quality bias", mp.qualityBias}, {"length bias", mp.lengthBias}, {"smear", mp.smear},
		{"in_ratio", mp.inRatio}, {"pooled_rate", mp.pooledRate}, {"pooled_lcb", mp.pooledLCB},
	}
	for _, s := range scalars {
		if !finite(s.v) {
			return mp, fmt.Errorf("%w: %s is not finite", ErrManifest, s.name)
		}
	}
	for _, s := range []struct {
		name string
		v    float64
	}{{"pooled_rate", mp.pooledRate}, {"pooled_lcb", mp.pooledLCB}} {
		if s.v < 0 || s.v > 1 {
			return mp, fmt.Errorf("%w: %s %v is outside [0, 1]", ErrManifest, s.name, s.v)
		}
	}
	for _, s := range []struct {
		name string
		v    float64
	}{{"smear", mp.smear}, {"in_ratio", mp.inRatio}} {
		if s.v <= 0 {
			return mp, fmt.Errorf("%w: %s %v must be positive", ErrManifest, s.name, s.v)
		}
	}
	return mp, nil
}

func buildOOD(o oodJSON, weights []byte, dim int) (oodParams, error) {
	if o.K != len(o.Centroids) || o.K < 1 {
		return oodParams{}, fmt.Errorf("%w: ood k=%d but %d centroids", ErrManifest, o.K, len(o.Centroids))
	}
	if !finite(o.Threshold) {
		return oodParams{}, fmt.Errorf("%w: ood threshold is not finite", ErrManifest)
	}
	out := oodParams{threshold: o.Threshold, centroids: make([]centroid, len(o.Centroids))}
	for j, c := range o.Centroids {
		idx, err := readI32(weights, c.Idx, -1)
		if err != nil {
			return out, fmt.Errorf("ood centroid %d indices: %w", j, err)
		}
		val, err := readF32(weights, c.Val, int64(len(idx)))
		if err != nil {
			return out, fmt.Errorf("ood centroid %d values: %w", j, err)
		}
		cen := centroid{idx: idx, val: make([]float64, len(val))}
		var sq float64
		for i, v := range val {
			if i > 0 && idx[i] <= idx[i-1] {
				return out, fmt.Errorf("%w: ood centroid %d indices are not strictly ascending", ErrArrayRef, j)
			}
			if idx[i] < 0 || int(idx[i]) >= dim {
				return out, fmt.Errorf("%w: ood centroid %d index %d is outside [0, %d)", ErrArrayRef, j, idx[i], dim)
			}
			cen.val[i] = float64(v)
			sq += float64(cen.val[i] * cen.val[i])
		}
		cen.norm = math.Sqrt(sq)
		out.centroids[j] = cen
	}
	return out, nil
}

// checkRef validates that r names an array of the given dtype inside weights, with wantLen
// elements when wantLen >= 0, and returns the byte range.
func checkRef(weights []byte, r refJSON, dtype string, wantLen int64) ([]byte, error) {
	if r.DType != dtype {
		return nil, fmt.Errorf("%w: dtype %q, want %q", ErrArrayRef, r.DType, dtype)
	}
	size := int64(len(weights))
	if r.Offset < 0 || r.Len < 0 || r.Offset > size || r.Len > (size-r.Offset)/4 {
		return nil, fmt.Errorf("%w: offset %d len %d does not fit weights.bin (%d bytes)", ErrArrayRef, r.Offset, r.Len, size)
	}
	if wantLen >= 0 && r.Len != wantLen {
		return nil, fmt.Errorf("%w: length %d, want %d", ErrArrayRef, r.Len, wantLen)
	}
	return weights[r.Offset : r.Offset+4*r.Len], nil
}

func readF32(weights []byte, r refJSON, wantLen int64) ([]float32, error) {
	b, err := checkRef(weights, r, "f32", wantLen)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		v := math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		if !finite(float64(v)) {
			return nil, fmt.Errorf("%w: element %d is not finite", ErrArrayRef, i)
		}
		out[i] = v
	}
	return out, nil
}

func readI32(weights []byte, r refJSON, wantLen int64) ([]int32, error) {
	b, err := checkRef(weights, r, "i32", wantLen)
	if err != nil {
		return nil, err
	}
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// selfCheck repeats the predictions recorded at training time and refuses the artifact
// when this build does not reproduce them.
func (a *Artifact) selfCheck(golden []goldenJSON) error {
	if len(golden) == 0 {
		return fmt.Errorf("%w: the manifest holds no golden entries", ErrGolden)
	}
	for n, g := range golden {
		msgs := make([]features.Message, len(g.Messages))
		for i, pair := range g.Messages {
			if len(pair) != 2 {
				return fmt.Errorf("%w: golden entry %d message %d is not [role, text]", ErrManifest, n, i)
			}
			msgs[i] = features.Message{Role: pair[0], Text: pair[1]}
		}
		if len(g.Models) != len(a.models) {
			return fmt.Errorf("%w: golden entry %d covers %d models, the artifact has %d", ErrGolden, n, len(g.Models), len(a.models))
		}
		got := a.Score(msgs)
		if !within(got.OODScore, g.OODScore, 1e-5) {
			return fmt.Errorf("%w: entry %d ood_score %.9g, golden %.9g", ErrGolden, n, got.OODScore, g.OODScore)
		}
		for _, mp := range a.models {
			id := mp.id
			want, ok := g.Models[id]
			if !ok {
				return fmt.Errorf("%w: entry %d has no golden prediction for model %q", ErrGolden, n, id)
			}
			s := got.Models[id]
			switch {
			case !within(s.P, want.P, 1e-5):
				return fmt.Errorf("%w: entry %d model %q p %.9g, golden %.9g", ErrGolden, n, id, s.P, want.P)
			case !within(s.L, want.L, 1e-9):
				return fmt.Errorf("%w: entry %d model %q l %.9g, golden %.9g", ErrGolden, n, id, s.L, want.L)
			case !within(s.EOut, want.EOut, 1e-3*math.Max(1, math.Abs(want.EOut))):
				return fmt.Errorf("%w: entry %d model %q e_out %.9g, golden %.9g", ErrGolden, n, id, s.EOut, want.EOut)
			case !within(s.OutQ[3], want.Q99, 1e-3*math.Max(1, math.Abs(want.Q99))):
				return fmt.Errorf("%w: entry %d model %q q99 %.9g, golden %.9g", ErrGolden, n, id, s.OutQ[3], want.Q99)
			}
		}
	}
	return nil
}

// within reports |got-want| <= tol. It is false when either value is NaN.
func within(got, want, tol float64) bool { return math.Abs(got-want) <= tol }

// Version is the content hash that names the artifact.
func (a *Artifact) Version() string { return a.version }

// Dim is the hashed feature dimension the artifact was trained with.
func (a *Artifact) Dim() int { return a.dim }

// Models lists the model ids in the artifact's fixed order. The slice is a copy.
func (a *Artifact) Models() []string {
	ids := make([]string, len(a.models))
	for i, m := range a.models {
		ids[i] = m.id
	}
	return ids
}

// HasModel reports whether the artifact predicts for the model id.
func (a *Artifact) HasModel(id string) bool {
	_, ok := a.byID[id]
	return ok
}

// OODThreshold is the OOD score above which a request counts as out of distribution.
func (a *Artifact) OODThreshold() float64 { return a.ood.threshold }

// ModelInfo describes one model of an artifact for Info.
type ModelInfo struct {
	ID         string `json:"id"`
	Calibrator string `json:"calibrator"`
}

// Info is a summary of an artifact for humans: no weights and no golden prompts.
type Info struct {
	Version      string          `json:"version"`
	Dim          int             `json:"dim"`
	Models       []ModelInfo     `json:"models"`
	OODThreshold float64         `json:"ood_threshold"`
	Training     json.RawMessage `json:"training"`
}

// Info summarizes the artifact: version, dimension, models with their calibrator types, the
// OOD threshold and the training block of the manifest.
func (a *Artifact) Info() Info {
	models := make([]ModelInfo, len(a.models))
	for i, m := range a.models {
		models[i] = ModelInfo{ID: m.id, Calibrator: m.cal.kind}
	}
	return Info{
		Version:      a.version,
		Dim:          a.dim,
		Models:       models,
		OODThreshold: a.ood.threshold,
		Training:     a.training,
	}
}
