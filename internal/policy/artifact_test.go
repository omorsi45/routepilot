package policy

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/omorsi45/routepilot/internal/features"
)

var tinyRoot = filepath.Join("..", "..", "testdata", "policy", "tiny")

// tinyDir is the single committed version directory under testdata/policy/tiny.
func tinyDir(t *testing.T) string {
	t.Helper()
	entries, err := os.ReadDir(tinyRoot)
	if err != nil {
		t.Fatalf("read %s: %v", tinyRoot, err)
	}
	var dirs []string
	for _, e := range entries {
		if e.IsDir() {
			dirs = append(dirs, e.Name())
		}
	}
	if len(dirs) != 1 {
		t.Fatalf("%s holds %d version directories, want exactly 1", tinyRoot, len(dirs))
	}
	return filepath.Join(tinyRoot, dirs[0])
}

func readFile(t *testing.T, path string) []byte {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return b
}

func writeFile(t *testing.T, path string, b []byte) {
	t.Helper()
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// copyTiny copies the tiny artifact into a temp directory named like the original.
func copyTiny(t *testing.T) string {
	t.Helper()
	src := tinyDir(t)
	dst := filepath.Join(t.TempDir(), filepath.Base(src))
	if err := os.Mkdir(dst, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"manifest.json", "weights.bin"} {
		writeFile(t, filepath.Join(dst, name), readFile(t, filepath.Join(src, name)))
	}
	return dst
}

// doc is a manifest decoded into generic maps so a test can edit one field.
type doc = map[string]any

func readDoc(t *testing.T, dir string) doc {
	t.Helper()
	var d doc
	if err := json.Unmarshal(readFile(t, filepath.Join(dir, "manifest.json")), &d); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	return d
}

// objectAt walks nested maps and slices: keys are strings or ints.
func objectAt(t *testing.T, root any, path ...any) any {
	t.Helper()
	cur := root
	for _, step := range path {
		switch s := step.(type) {
		case string:
			m, ok := cur.(map[string]any)
			if !ok {
				t.Fatalf("path %v: %T is not an object at %q", path, cur, s)
			}
			cur = m[s]
		case int:
			a, ok := cur.([]any)
			if !ok {
				t.Fatalf("path %v: %T is not an array at %d", path, cur, s)
			}
			cur = a[s]
		}
	}
	return cur
}

func setAt(t *testing.T, root any, value any, path ...any) {
	t.Helper()
	parent := objectAt(t, root, path[:len(path)-1]...)
	switch s := path[len(path)-1].(type) {
	case string:
		parent.(map[string]any)[s] = value
	case int:
		parent.([]any)[s] = value
	}
}

// marshalManifest writes d the way the exporter does: one space of indent, "key": value.
func marshalManifest(t *testing.T, d doc) []byte {
	t.Helper()
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", " ")
	if err := enc.Encode(d); err != nil {
		t.Fatalf("encode manifest: %v", err)
	}
	return buf.Bytes()
}

// rewrite stores d and the weights of dir in a new directory whose version is recomputed
// independently of the package under test, so only the intended check can fail.
func rewrite(t *testing.T, dir string, d doc) string {
	t.Helper()
	weights := readFile(t, filepath.Join(dir, "weights.bin"))
	d["version"] = ""
	sum := sha256.New()
	sum.Write(marshalManifest(t, d))
	sum.Write(weights)
	version := hex.EncodeToString(sum.Sum(nil))[:12]
	d["version"] = version
	out := filepath.Join(t.TempDir(), version)
	if err := os.Mkdir(out, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(out, "manifest.json"), marshalManifest(t, d))
	writeFile(t, filepath.Join(out, "weights.bin"), weights)
	return out
}

func TestLoadTinyArtifact(t *testing.T) {
	dir := tinyDir(t)

	a, err := Load(dir)
	if err != nil {
		t.Fatalf("Load(%s) = %v", dir, err)
	}

	if a.Version() != filepath.Base(dir) {
		t.Errorf("Version() = %q, want the directory name %q", a.Version(), filepath.Base(dir))
	}
	if a.Dim() != 1024 {
		t.Errorf("Dim() = %d, want 1024", a.Dim())
	}
	models := a.Models()
	if len(models) != 2 || models[0] != "tiny-weak" || models[1] != "tiny-strong" {
		t.Errorf("Models() = %v, want [tiny-weak tiny-strong] in manifest order", models)
	}
	models[0] = "changed"
	if a.Models()[0] != "tiny-weak" {
		t.Error("Models() returned the internal slice")
	}
	if !a.HasModel("tiny-strong") || a.HasModel("ghost") {
		t.Errorf("HasModel: tiny-strong=%v ghost=%v, want true and false", a.HasModel("tiny-strong"), a.HasModel("ghost"))
	}
	if th := a.OODThreshold(); !(th > 0 && th < 1) {
		t.Errorf("OODThreshold() = %v, want in (0, 1)", th)
	}
}

func TestLoadIsRepeatableFromARelativeAndAnAbsolutePath(t *testing.T) {
	rel := tinyDir(t)
	abs, err := filepath.Abs(rel)
	if err != nil {
		t.Fatal(err)
	}
	for _, dir := range []string{rel, abs, rel + string(filepath.Separator)} {
		if _, err := Load(dir); err != nil {
			t.Errorf("Load(%q) = %v", dir, err)
		}
	}
}

func TestLoadRejectsHashMismatch(t *testing.T) {
	dir := copyTiny(t)
	path := filepath.Join(dir, "weights.bin")
	weights := readFile(t, path)
	weights[len(weights)/2] ^= 0x01
	writeFile(t, path, weights)

	_, err := Load(dir)

	if !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("Load = %v, want ErrHashMismatch", err)
	}
}

func TestLoadRejectsAnEditedManifestThatKeepsItsVersion(t *testing.T) {
	dir := copyTiny(t)
	path := filepath.Join(dir, "manifest.json")
	raw := readFile(t, path)
	edited := bytes.Replace(raw, []byte(`"dim": 1024`), []byte(`"dim": 1025`), 1)
	if bytes.Equal(raw, edited) {
		t.Fatal("test setup: the manifest has no dim field to edit")
	}
	writeFile(t, path, edited)

	if _, err := Load(dir); !errors.Is(err, ErrHashMismatch) {
		t.Fatalf("Load = %v, want ErrHashMismatch", err)
	}
}

func TestLoadRejectsSpecVersion(t *testing.T) {
	d := readDoc(t, tinyDir(t))
	setAt(t, d, float64(features.SpecVersion+1), "feature_spec_version")

	_, err := Load(rewrite(t, tinyDir(t), d))

	if !errors.Is(err, ErrSpecVersion) {
		t.Fatalf("Load = %v, want ErrSpecVersion", err)
	}
}

func TestLoadRejectsFormat(t *testing.T) {
	for name, edit := range map[string]func(t *testing.T, d doc){
		"format":         func(t *testing.T, d doc) { setAt(t, d, "other-artifact", "format") },
		"format_version": func(t *testing.T, d doc) { setAt(t, d, float64(2), "format_version") },
	} {
		t.Run(name, func(t *testing.T) {
			d := readDoc(t, tinyDir(t))
			edit(t, d)

			_, err := Load(rewrite(t, tinyDir(t), d))

			if !errors.Is(err, ErrFormat) {
				t.Fatalf("Load = %v, want ErrFormat", err)
			}
		})
	}
}

func TestLoadRejectsOutOfBoundsRef(t *testing.T) {
	tests := map[string]func(t *testing.T, d doc){
		"offset past the end": func(t *testing.T, d doc) {
			setAt(t, d, float64(1<<40), "models", 0, "quality", "weights", "offset")
		},
		"negative offset": func(t *testing.T, d doc) {
			setAt(t, d, float64(-4), "models", 0, "length", "weights", "offset")
		},
		"length past the end": func(t *testing.T, d doc) {
			setAt(t, d, float64(1<<40), "ood", "centroids", 1, "val", "len")
		},
		"wrong length for the quality weights": func(t *testing.T, d doc) {
			setAt(t, d, float64(1024+12-1), "models", 1, "quality", "weights", "len")
		},
		"wrong dtype for the indices": func(t *testing.T, d doc) {
			setAt(t, d, "f32", "ood", "centroids", 0, "idx", "dtype")
		},
		"unknown dtype": func(t *testing.T, d doc) {
			setAt(t, d, "f64", "models", 0, "quality", "weights", "dtype")
		},
		"centroid idx and val differ in length": func(t *testing.T, d doc) {
			n := objectAt(t, d, "ood", "centroids", 0, "val", "len").(float64)
			setAt(t, d, n-1, "ood", "centroids", 0, "val", "len")
		},
	}
	for name, edit := range tests {
		t.Run(name, func(t *testing.T) {
			d := readDoc(t, tinyDir(t))
			edit(t, d)

			_, err := Load(rewrite(t, tinyDir(t), d))

			if !errors.Is(err, ErrArrayRef) {
				t.Fatalf("Load = %v, want ErrArrayRef", err)
			}
		})
	}
}

func TestLoadRejectsTruncatedWeightsEvenWhenRehashed(t *testing.T) {
	src := copyTiny(t)
	path := filepath.Join(src, "weights.bin")
	weights := readFile(t, path)
	writeFile(t, path, weights[:len(weights)-4])

	_, err := Load(rewrite(t, src, readDoc(t, src)))

	if !errors.Is(err, ErrArrayRef) {
		t.Fatalf("Load = %v, want ErrArrayRef (the last array no longer fits)", err)
	}
}

func TestLoadRejectsNonFiniteAndUnsortedArrays(t *testing.T) {
	t.Run("NaN weight", func(t *testing.T) {
		src := copyTiny(t)
		path := filepath.Join(src, "weights.bin")
		weights := readFile(t, path)
		binary.LittleEndian.PutUint32(weights[8:], math.Float32bits(float32(math.NaN())))
		writeFile(t, path, weights)

		if _, err := Load(rewrite(t, src, readDoc(t, src))); !errors.Is(err, ErrArrayRef) {
			t.Fatalf("Load = %v, want ErrArrayRef", err)
		}
	})
	t.Run("centroid indices out of order", func(t *testing.T) {
		src := copyTiny(t)
		d := readDoc(t, src)
		offset := int(objectAt(t, d, "ood", "centroids", 0, "idx", "offset").(float64))
		path := filepath.Join(src, "weights.bin")
		weights := readFile(t, path)
		first := binary.LittleEndian.Uint32(weights[offset:])
		binary.LittleEndian.PutUint32(weights[offset+4:], first) // repeat the first index
		writeFile(t, path, weights)

		if _, err := Load(rewrite(t, src, d)); !errors.Is(err, ErrArrayRef) {
			t.Fatalf("Load = %v, want ErrArrayRef", err)
		}
	})
	t.Run("centroid index outside the dimension", func(t *testing.T) {
		src := copyTiny(t)
		d := readDoc(t, src)
		offset := int(objectAt(t, d, "ood", "centroids", 0, "idx", "offset").(float64))
		length := int(objectAt(t, d, "ood", "centroids", 0, "idx", "len").(float64))
		path := filepath.Join(src, "weights.bin")
		weights := readFile(t, path)
		binary.LittleEndian.PutUint32(weights[offset+4*(length-1):], 1024)
		writeFile(t, path, weights)

		if _, err := Load(rewrite(t, src, d)); !errors.Is(err, ErrArrayRef) {
			t.Fatalf("Load = %v, want ErrArrayRef", err)
		}
	})
}

func TestLoadRejectsGoldenMismatch(t *testing.T) {
	d := readDoc(t, tinyDir(t))
	p := objectAt(t, d, "golden", 0, "models", "tiny-weak", "p").(float64)
	setAt(t, d, p+0.01, "golden", 0, "models", "tiny-weak", "p")

	_, err := Load(rewrite(t, tinyDir(t), d))

	if !errors.Is(err, ErrGolden) {
		t.Fatalf("Load = %v, want ErrGolden", err)
	}
	if !strings.Contains(err.Error(), "tiny-weak") {
		t.Errorf("error %q does not name the model", err)
	}
}

func TestGoldenTolerances(t *testing.T) {
	// field -> (tolerance for the field in a golden entry, is relative to max(1, |want|)).
	tests := []struct {
		name string
		path []any
		tol  float64
		rel  bool
	}{
		{"p", []any{"golden", 0, "models", "tiny-strong", "p"}, 1e-5, false},
		{"l", []any{"golden", 0, "models", "tiny-strong", "l"}, 1e-9, false},
		{"e_out", []any{"golden", 0, "models", "tiny-strong", "e_out"}, 1e-3, true},
		{"q99", []any{"golden", 0, "models", "tiny-strong", "q99"}, 1e-3, true},
		{"ood_score", []any{"golden", 0, "ood_score"}, 1e-5, false},
	}
	for _, tt := range tests {
		for _, factor := range []float64{0.5, 2} {
			name := tt.name + map[float64]string{0.5: " inside", 2: " outside"}[factor]
			t.Run(name, func(t *testing.T) {
				d := readDoc(t, tinyDir(t))
				want := objectAt(t, d, tt.path...).(float64)
				shift := tt.tol * factor
				if tt.rel {
					shift *= math.Max(1, math.Abs(want))
				}
				setAt(t, d, want+shift, tt.path...)

				_, err := Load(rewrite(t, tinyDir(t), d))

				switch {
				case factor < 1 && err != nil:
					t.Fatalf("a shift of %.3g in %s is inside the tolerance but Load = %v", shift, tt.name, err)
				case factor > 1 && !errors.Is(err, ErrGolden):
					t.Fatalf("a shift of %.3g in %s is outside the tolerance but Load = %v", shift, tt.name, err)
				}
			})
		}
	}
}

func TestLoadRejectsGoldenModelSetMismatch(t *testing.T) {
	t.Run("extra model", func(t *testing.T) {
		d := readDoc(t, tinyDir(t))
		models := objectAt(t, d, "golden", 0, "models").(map[string]any)
		models["ghost"] = models["tiny-weak"]
		if _, err := Load(rewrite(t, tinyDir(t), d)); !errors.Is(err, ErrGolden) {
			t.Fatalf("Load = %v, want ErrGolden", err)
		}
	})
	t.Run("missing model", func(t *testing.T) {
		d := readDoc(t, tinyDir(t))
		delete(objectAt(t, d, "golden", 0, "models").(map[string]any), "tiny-strong")
		if _, err := Load(rewrite(t, tinyDir(t), d)); !errors.Is(err, ErrGolden) {
			t.Fatalf("Load = %v, want ErrGolden", err)
		}
	})
	t.Run("no golden entries", func(t *testing.T) {
		d := readDoc(t, tinyDir(t))
		setAt(t, d, []any{}, "golden")
		if _, err := Load(rewrite(t, tinyDir(t), d)); !errors.Is(err, ErrGolden) {
			t.Fatalf("Load = %v, want ErrGolden", err)
		}
	})
}

func TestLoadRejectsDirNameMismatch(t *testing.T) {
	src := tinyDir(t)
	dir := filepath.Join(t.TempDir(), "000000000000")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"manifest.json", "weights.bin"} {
		writeFile(t, filepath.Join(dir, name), readFile(t, filepath.Join(src, name)))
	}

	_, err := Load(dir)

	if !errors.Is(err, ErrDirName) {
		t.Fatalf("Load = %v, want ErrDirName", err)
	}
}

func TestLoadRejectsMalformedManifestContent(t *testing.T) {
	tests := map[string]func(t *testing.T, d doc){
		"isotonic knots not increasing": func(t *testing.T, d doc) {
			xs := objectAt(t, d, "models", 0, "calibrator", "x").([]any)
			xs[1] = xs[0]
		},
		"unknown calibrator": func(t *testing.T, d doc) {
			setAt(t, d, "spline", "models", 0, "calibrator", "type")
		},
		"empty lcb table": func(t *testing.T, d doc) {
			setAt(t, d, []any{}, "models", 0, "lcb", "upper")
			setAt(t, d, []any{}, "models", 0, "lcb", "lower")
			setAt(t, d, []any{}, "models", 0, "lcb", "n")
		},
		"lcb tables of different lengths": func(t *testing.T, d doc) {
			lower := objectAt(t, d, "models", 0, "lcb", "lower").([]any)
			setAt(t, d, lower[1:], "models", 0, "lcb", "lower")
		},
		"zero dense std": func(t *testing.T, d doc) {
			setAt(t, d, float64(0), "dense_std", 3)
		},
		"short dense mean": func(t *testing.T, d doc) {
			mean := objectAt(t, d, "dense_mean").([]any)
			setAt(t, d, mean[:11], "dense_mean")
		},
		"duplicate model id": func(t *testing.T, d doc) {
			setAt(t, d, "tiny-weak", "models", 1, "id")
		},
		"missing residual quantile": func(t *testing.T, d doc) {
			delete(objectAt(t, d, "models", 0, "length", "residual_quantiles").(map[string]any), "0.99")
		},
		"no centroids": func(t *testing.T, d doc) {
			setAt(t, d, []any{}, "ood", "centroids")
			setAt(t, d, float64(0), "ood", "k")
		},
		"k differs from the centroid count": func(t *testing.T, d doc) {
			setAt(t, d, float64(31), "ood", "k")
		},
		"dim zero": func(t *testing.T, d doc) {
			setAt(t, d, float64(0), "dim")
		},
	}
	for name, edit := range tests {
		t.Run(name, func(t *testing.T) {
			d := readDoc(t, tinyDir(t))
			edit(t, d)

			_, err := Load(rewrite(t, tinyDir(t), d))

			if !errors.Is(err, ErrManifest) {
				t.Fatalf("Load = %v, want ErrManifest", err)
			}
		})
	}
}

func TestLoadRejectsUnknownManifestFields(t *testing.T) {
	d := readDoc(t, tinyDir(t))
	setAt(t, d, "surprise", "extra")

	if _, err := Load(rewrite(t, tinyDir(t), d)); !errors.Is(err, ErrManifest) {
		t.Fatalf("Load = %v, want ErrManifest", err)
	}
}

func TestLoadRejectsMissingFiles(t *testing.T) {
	for _, name := range []string{"manifest.json", "weights.bin"} {
		dir := copyTiny(t)
		if err := os.Remove(filepath.Join(dir, name)); err != nil {
			t.Fatal(err)
		}
		if _, err := Load(dir); err == nil {
			t.Errorf("Load without %s succeeded", name)
		}
	}
	if _, err := Load(filepath.Join(t.TempDir(), "absent")); err == nil {
		t.Error("Load of a missing directory succeeded")
	}
}

func TestUnversionedManifestNeedsExactlyOneOccurrence(t *testing.T) {
	one := []byte("{\n \"a\": 1,\n \"version\": \"0123456789ab\"\n}\n")
	got, err := unversionedManifest(one, "0123456789ab")
	if err != nil {
		t.Fatalf("one occurrence: %v", err)
	}
	if want := "{\n \"a\": 1,\n \"version\": \"\"\n}\n"; string(got) != want {
		t.Errorf("unversionedManifest = %q, want %q", got, want)
	}

	two := []byte(`{"version": "0123456789ab", "x": {"version": "0123456789ab"}}`)
	if _, err := unversionedManifest(two, "0123456789ab"); !errors.Is(err, ErrHashMismatch) {
		t.Errorf("two occurrences: err = %v, want ErrHashMismatch", err)
	}
	if _, err := unversionedManifest([]byte(`{"version":"0123456789ab"}`), "0123456789ab"); !errors.Is(err, ErrHashMismatch) {
		t.Errorf("no spaced occurrence: err = %v, want ErrHashMismatch", err)
	}
}

func TestContentVersionIsSHA256PrefixOfManifestThenWeights(t *testing.T) {
	sum := sha256.Sum256([]byte("manifestweights"))
	want := hex.EncodeToString(sum[:])[:12]
	if got := contentVersion([]byte("manifest"), []byte("weights")); got != want {
		t.Errorf("contentVersion = %q, want %q", got, want)
	}
}

func TestInfoSummarizesTheArtifact(t *testing.T) {
	a, err := Load(tinyDir(t))
	if err != nil {
		t.Fatal(err)
	}

	info := a.Info()

	if info.Version != a.Version() || info.Dim != 1024 || info.OODThreshold != a.OODThreshold() {
		t.Errorf("Info = %+v does not match the accessors", info)
	}
	if len(info.Models) != 2 || info.Models[0] != (ModelInfo{ID: "tiny-weak", Calibrator: "isotonic"}) ||
		info.Models[1] != (ModelInfo{ID: "tiny-strong", Calibrator: "platt"}) {
		t.Errorf("Info.Models = %+v, want tiny-weak isotonic then tiny-strong platt", info.Models)
	}
	var training struct {
		Pool      string   `json:"pool"`
		TrainedOn []string `json:"trained_on"`
	}
	if err := json.Unmarshal(info.Training, &training); err != nil {
		t.Fatalf("Info.Training is not JSON: %v", err)
	}
	if training.Pool != "tiny" || len(training.TrainedOn) != 1 || training.TrainedOn[0] != "train" {
		t.Errorf("Info.Training = %+v, want pool tiny trained on train", training)
	}
}
