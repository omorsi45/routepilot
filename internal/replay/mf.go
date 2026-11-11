package replay

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/omorsi45/routepilot/internal/catalog"
	"github.com/omorsi45/routepilot/internal/features"
	"github.com/omorsi45/routepilot/internal/openai"
	"github.com/omorsi45/routepilot/internal/router"
)

// PolicyRouteLLMMF is RouteLLM's matrix factorization router as a baseline: a linear score
// of the request's features decides between a strong and a weak model with a threshold.
// For this policy a record's p is the score (the strong model's win probability, not a
// calibrated chance of success), and the learned-policy fields stay null.
const PolicyRouteLLMMF = "routellm_mf"

// MF is the exported matrix factorization scorer for one strong and one weak model
// (python/routepilot_train/baselines/routellm_mf.py). The difference between the two
// models' scores is linear in the features, so a single vector U holds all of it:
// score = sigmoid(U . [hashed features, standardized dense features]).
type MF struct {
	// Strong and Weak are model IDs of the replay config.
	Strong, Weak string
	// Dim is the hashed feature dimension; U has Dim + features.NumDense entries.
	Dim int
	// DenseMean and DenseStd standardize the dense features: z = (dense - mean) / std.
	DenseMean, DenseStd [features.NumDense]float64
	U                   []float32
}

// mfFile is the JSON shape of an exported MF file. U is the base64 of little-endian float32
// values.
type mfFile struct {
	Strong    string    `json:"strong"`
	Weak      string    `json:"weak"`
	Dim       int       `json:"dim"`
	DenseMean []float64 `json:"dense_mean"`
	DenseStd  []float64 `json:"dense_std"`
	U         *string   `json:"u"`
}

// LoadMF reads and validates an exported MF file.
func LoadMF(path string) (*MF, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read mf file: %w", err)
	}
	mf, err := ParseMF(data)
	if err != nil {
		return nil, fmt.Errorf("mf file %s: %w", path, err)
	}
	return mf, nil
}

// ParseMF decodes and validates an exported MF file: two different model IDs, a hashed
// dimension in [1, MaxInt32], twelve finite means and finite positive standard deviations,
// and exactly Dim + 12 finite float32 weights. Unknown keys are an error.
func ParseMF(data []byte) (*MF, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var f mfFile
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("decode: data after the JSON document")
	}
	switch {
	case f.Strong == "" || f.Weak == "":
		return nil, errors.New("strong and weak must name models")
	case f.Strong == f.Weak:
		return nil, fmt.Errorf("strong and weak are both %q", f.Strong)
	case f.Dim < 1 || f.Dim > math.MaxInt32:
		return nil, fmt.Errorf("dim %d is out of range [1, %d]", f.Dim, math.MaxInt32)
	case len(f.DenseMean) != features.NumDense:
		return nil, fmt.Errorf("dense_mean has %d values, want %d", len(f.DenseMean), features.NumDense)
	case len(f.DenseStd) != features.NumDense:
		return nil, fmt.Errorf("dense_std has %d values, want %d", len(f.DenseStd), features.NumDense)
	case f.U == nil:
		return nil, errors.New("u is missing")
	}
	mf := &MF{Strong: f.Strong, Weak: f.Weak, Dim: f.Dim}
	for i := range features.NumDense {
		if !finite(f.DenseMean[i]) {
			return nil, fmt.Errorf("dense_mean[%d] is not finite", i)
		}
		if !finite(f.DenseStd[i]) || f.DenseStd[i] <= 0 {
			return nil, fmt.Errorf("dense_std[%d] = %v, want a finite positive value", i, f.DenseStd[i])
		}
		mf.DenseMean[i], mf.DenseStd[i] = f.DenseMean[i], f.DenseStd[i]
	}
	raw, err := base64.StdEncoding.DecodeString(*f.U)
	if err != nil {
		return nil, fmt.Errorf("u is not base64: %w", err)
	}
	want := 4 * (int64(f.Dim) + features.NumDense)
	if int64(len(raw)) != want {
		return nil, fmt.Errorf("u holds %d bytes, want %d (4 * (dim + %d))", len(raw), want, features.NumDense)
	}
	mf.U = make([]float32, len(raw)/4)
	for i := range mf.U {
		v := math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		if !finite(float64(v)) {
			return nil, fmt.Errorf("u[%d] is not finite", i)
		}
		mf.U[i] = v
	}
	return mf, nil
}

func finite(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }

// Score is sigmoid(U . [v, z]): the probability that the strong model beats the weak one on
// a request made of msgs. v is the hashed feature vector and z the standardized dense
// features. The terms are summed in the trainer's order (hashed terms in ascending index,
// then the dense terms) in float64, with the float32 weights widened first; the explicit
// conversions forbid fused multiply-adds, which would round differently from numpy.
func (m *MF) Score(msgs []features.Message) float64 {
	v := features.Extract(msgs, m.Dim)
	var logit float64
	for k, i := range v.Idx {
		logit += float64(float64(m.U[i]) * v.Val[k])
	}
	for d := range v.Dense {
		z := (v.Dense[d] - m.DenseMean[d]) / m.DenseStd[d]
		logit += float64(float64(m.U[m.Dim+d]) * z)
	}
	return sigmoid(logit)
}

// sigmoid is the logistic function, stable for large positive and negative x.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

// newMFSelector routes an item to the strong model when its MF score reaches the threshold
// and to the weak model otherwise. Each model is asked for as a pinned request, so the router
// applies the tenant's permissions and the model's capability and limit checks; when the
// routed model cannot serve the request the other one is tried, and when neither can the
// item is a routing error (the error of the model tried last). The score is recorded in the
// selection so the record can carry it.
func newMFSelector(cat *catalog.Catalog, opts Options) (selector, error) {
	if opts.MF == nil {
		return nil, usageError("policy %s needs an MF file", PolicyRouteLLMMF)
	}
	mf, threshold := opts.MF, opts.Threshold
	for _, id := range []string{mf.Strong, mf.Weak} {
		if _, ok := cat.Model(id); !ok {
			return nil, fmt.Errorf("build %s policy: model %q of the MF file is not in the config", PolicyRouteLLMMF, id)
		}
	}
	r := router.New(cat, nil)
	return func(item *Item, in router.Input) (selection, *openai.APIError, error) {
		score := mf.Score(features.Messages(&openai.ChatCompletionRequest{Messages: item.Messages}))
		first, second := mf.Weak, mf.Strong
		if score >= threshold {
			first, second = mf.Strong, mf.Weak
		}
		in.Requested = first
		sel, apiErr, err := decideFirst(r, in)
		if apiErr != nil && err == nil {
			in.Requested = second
			sel, apiErr, err = decideFirst(r, in)
		}
		sel.score = &score
		return sel, apiErr, err
	}, nil
}
