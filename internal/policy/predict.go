package policy

import (
	"fmt"
	"math"
	"sort"

	"github.com/omorsi45/routepilot/internal/features"
)

// ModelScore is the policy's prediction of one model for one request.
type ModelScore struct {
	// P is the calibrated probability that the model succeeds; L its lower confidence
	// bound from the calibration table.
	P, L float64
	// PooledRate and PooledLCB are the model's overall calibration success rate and its
	// lower bound, the fallback when a request cannot be scored.
	PooledRate, PooledLCB float64
	// EOut is the expected completion tokens (exp(pred) * smear - 1), unclamped.
	EOut float64
	// OutQ are the completion token quantile estimates for u = 0.5, 0.9, 0.95, 0.99
	// (exp(pred + r_u) - 1), unclamped.
	OutQ [4]float64
	// InRatio converts the request's prompt token upper bound to a token estimate.
	InRatio float64
}

// Scores is the prediction for one request: the OOD verdict and a score per model.
type Scores struct {
	// OODScore is 1 - the best cosine to a k-means centroid; OOD is OODScore above the
	// artifact's threshold. A request with no hashed features scores 1.
	OODScore float64
	OOD      bool
	Models   map[string]ModelScore
}

// calibrator maps a raw linear score to a probability.
type calibrator struct {
	kind   string // "isotonic" or "platt"
	xs, ys []float64
	a, b   float64
}

func newCalibrator(c calibratorJSON) (calibrator, error) {
	switch c.Type {
	case "isotonic":
		if len(c.X) == 0 || len(c.X) != len(c.Y) {
			return calibrator{}, fmt.Errorf("%w: isotonic x and y need the same non-zero length", ErrManifest)
		}
		for i, x := range c.X {
			if !finite(x) || !finite(c.Y[i]) || (i > 0 && x <= c.X[i-1]) {
				return calibrator{}, fmt.Errorf("%w: isotonic knots must be finite and strictly increasing", ErrManifest)
			}
			if c.Y[i] < 0 || c.Y[i] > 1 {
				return calibrator{}, fmt.Errorf("%w: isotonic value y[%d]=%v is outside [0, 1]", ErrManifest, i, c.Y[i])
			}
		}
		return calibrator{kind: c.Type, xs: c.X, ys: c.Y}, nil
	case "platt":
		if !finite(c.A) || !finite(c.B) {
			return calibrator{}, fmt.Errorf("%w: platt a and b must be finite", ErrManifest)
		}
		return calibrator{kind: c.Type, a: c.A, b: c.B}, nil
	default:
		return calibrator{}, fmt.Errorf("%w: unknown calibrator type %q", ErrManifest, c.Type)
	}
}

// sigmoid is the logistic function, stable for large positive and negative x.
func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

// apply returns the calibrated probability of the raw score s. Isotonic follows
// numpy.interp(sigmoid(s), xs, ys): the end values outside the knots, the knot value at a
// knot and linear interpolation between knots.
func (c *calibrator) apply(s float64) float64 {
	if c.kind == "platt" {
		return sigmoid(float64(c.a*s) + c.b)
	}
	x := sigmoid(s)
	last := len(c.xs) - 1
	switch {
	case math.IsNaN(x) || x <= c.xs[0]:
		return c.ys[0]
	case x >= c.xs[last]:
		return c.ys[last]
	}
	j := sort.SearchFloat64s(c.xs, x) // first knot >= x; 0 < j <= last here
	if c.xs[j] == x {
		return c.ys[j]
	}
	slope := (c.ys[j] - c.ys[j-1]) / (c.xs[j] - c.xs[j-1])
	return float64(slope*(x-c.xs[j-1])) + c.ys[j-1]
}

// lookup returns the lower bound of the first bin whose upper edge is at least p, or of the
// last bin. A NaN p gets the lowest bin, the most conservative bound.
func (m *modelParams) lookup(p float64) float64 {
	for i, upper := range m.lcbUpper {
		if p <= upper {
			return m.lcbLower[i]
		}
	}
	if math.IsNaN(p) {
		return m.lcbLower[0]
	}
	return m.lcbLower[len(m.lcbLower)-1]
}

// linear is w . [hashed | z] + bias in float64: the hashed terms in ascending index order,
// then the dense terms, then the bias. The float32 weights are widened first. The explicit
// conversions forbid fused multiply-adds, which would round differently from the trainer.
func linear(w []float32, idx []int32, val []float64, z *[features.NumDense]float64, dim int, bias float64) float64 {
	var acc float64
	for k, i := range idx {
		acc += float64(float64(w[i]) * val[k])
	}
	for d := range z {
		acc += float64(float64(w[dim+d]) * z[d])
	}
	return acc + bias
}

// oodScore is 1 - max_j cos(x, c_j) against the stored centroids. A zero request or a zero
// centroid has cosine 0.
func (o *oodParams) score(idx []int32, val []float64) float64 {
	var sq float64
	for _, v := range val {
		sq += float64(v * v)
	}
	xNorm := math.Sqrt(sq)
	best := math.Inf(-1)
	for i := range o.centroids {
		c := &o.centroids[i]
		var dot float64
		for a, b := 0, 0; a < len(idx) && b < len(c.idx); {
			switch {
			case idx[a] < c.idx[b]:
				a++
			case idx[a] > c.idx[b]:
				b++
			default:
				dot += float64(val[a] * c.val[b])
				a++
				b++
			}
		}
		cos := 0.0
		if denom := xNorm * c.norm; denom > 0 {
			cos = dot / denom
		}
		best = math.Max(best, cos)
	}
	return 1 - best
}

// Score predicts, for every model of the artifact, how a request made of msgs would go.
// It allocates one feature vector and one map per call and is safe for concurrent use.
func (a *Artifact) Score(msgs []features.Message) Scores {
	v := features.Extract(msgs, a.dim)
	var z [features.NumDense]float64
	for i := range z {
		z[i] = (v.Dense[i] - a.mean[i]) / a.std[i]
	}

	out := Scores{
		OODScore: a.ood.score(v.Idx, v.Val),
		Models:   make(map[string]ModelScore, len(a.models)),
	}
	out.OOD = out.OODScore > a.ood.threshold
	for i := range a.models {
		m := &a.models[i]
		p := m.cal.apply(linear(m.qualityW, v.Idx, v.Val, &z, a.dim, m.qualityBias))
		pred := linear(m.lengthW, v.Idx, v.Val, &z, a.dim, m.lengthBias)
		s := ModelScore{
			P:          p,
			L:          m.lookup(p),
			PooledRate: m.pooledRate,
			PooledLCB:  m.pooledLCB,
			EOut:       float64(math.Exp(pred)*m.smear) - 1,
			InRatio:    m.inRatio,
		}
		for q := range s.OutQ {
			s.OutQ[q] = math.Exp(pred+m.resid[q]) - 1
		}
		out.Models[m.id] = s
	}
	return out
}
