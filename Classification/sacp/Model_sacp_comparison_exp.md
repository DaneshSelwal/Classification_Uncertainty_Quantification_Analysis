# Spatial Adaptive Conformal Prediction (SACP) for Multispectral Image Classification

## Overview

This notebook evaluates three deep learning models (an AlexNet-style CNN, a GFNet, and a ViT-UNet) on a 6-band multispectral image classification task, and then wraps each model's predictions in a **Spatial Adaptive Conformal Predictor (SACP)**. SACP is a conformal prediction method that produces *prediction sets* (rather than single point predictions) with a statistical coverage guarantee — e.g., "the true class is in this set with at least 95% probability" — while exploiting the fact that neighboring pixels in an image tend to belong to the same class. The notebook sweeps over four neighborhood "window sizes" (3, 5, 7, 9), runs the full SACP pipeline for each of the three models at each window size, and produces per-window and combined cross-window Excel reports, CSV summaries, and comparison plots.

**Who this document is for:** a reader who has this notebook (possibly AI-generated) and wants to understand, line by line and concept by concept, what every function does, why it's there, and how the math works — useful both for learning conformal prediction methods and as a basis for writing up the method in a paper.

---

## Table of Contents

1. [Title & Overview](#spatial-adaptive-conformal-prediction-sacp-for-multispectral-image-classification)
2. [Environment & Dependencies](#environment--dependencies)
3. [Data & Problem Setup](#data--problem-setup)
4. [Methods](#methods)
   - [Custom Keras Layers (Model Architecture Building Blocks)](#method-custom-keras-layers)
   - [Model Loading with Trust Checks](#method-model-loading-with-trust-checks)
   - [Probability Normalisation](#method-probability-normalisation)
   - [Randomised APS Conformity Scores](#method-randomised-aps-conformity-scores)
   - [Spatial Smoothing of Conformity Scores](#method-spatial-smoothing-of-conformity-scores)
   - [SACP Calibration and Prediction Set Construction](#method-sacp-calibration-and-prediction-set-construction)
   - [Full-Scene Inference](#method-full-scene-inference)
   - [Per-Model SACP Pipeline (Orchestration)](#method-per-model-sacp-pipeline)
   - [Window-Size Sweep & Excel Reporting](#method-window-size-sweep--excel-reporting)
5. [Results & Comparisons](#results--comparisons)
6. [Academic Paper Summary](#academic-paper-summary)
7. [References](#references)

---

## Environment & Dependencies

| Library | Purpose |
|---|---|
| `os`, `sys`, `subprocess` | OS interaction, detecting Colab, installing packages |
| `io` | In-memory byte buffers (used to hold PNG plots before embedding in Excel) |
| `json` | Reading/writing JSON (run-config style data) |
| `time` | Timing how long each model's SACP run takes |
| `random` | Python's built-in RNG, seeded for reproducibility |
| `warnings` | Suppressing noisy warning messages |
| `pathlib.Path` | Object-oriented filesystem paths |
| `numpy` | Numerical arrays, the backbone of all score/probability math |
| `pandas` | DataFrames for summaries, per-class tables, CSV/Excel export |
| `matplotlib.pyplot` | Plotting |
| `seaborn` | Statistical plotting style and line plots |
| `sklearn.model_selection.train_test_split` | Stratified train/calibration/evaluation splitting |
| `tqdm.auto` | Progress bars |
| `matplotlib.colors.ListedColormap` | Custom discrete colormaps for class maps |
| `matplotlib.patches.Patch` | Legend swatches for custom colormaps |
| `openpyxl.load_workbook` | Re-opening saved Excel files to validate their sheet structure |
| `tensorflow` / `tensorflow.keras` / `tensorflow.keras.layers` | Loading and running the three trained neural network models |

> **Note:** TensorFlow/Keras version-specific behavior is visible in the model-loading function (Section "Model Loading with Trust Checks"), which has a fallback path for `safe_mode`/Lambda-layer deserialization issues — a known pain point when loading custom Keras models saved with one TF/Keras version and loaded with another.

---

## Data & Problem Setup

**Dataset.** A single multispectral scene of shape `H=330` rows × `W=307` columns × `B=6` spectral bands, stored as two CSV files: `data.csv` (the raw band values, reshaped to `(330, 307, 6)`) and `ref.csv` (integer class labels per pixel, reshaped to `(330, 307)`). Each band is independently min-max normalized to the `[0, 1]` range.

**Problem type.** Pixel-wise multi-class classification (land-cover-style classification, given the "Classification" project naming and multispectral input). Each labeled pixel (label > 0) becomes one training/evaluation sample by extracting a `9×9` spatial patch (`PATCH_SIZE = 9`) around it across all 6 bands, giving input tensors of shape `(9, 9, 6)`. Labels are shifted to be 0-indexed (`label - 1`).

**Preprocessing / splitting pipeline, exactly as done in the notebook:**

1. Load the 6-band image and label map; min-max normalize each band independently.
2. Pad the image by `PATCH_SIZE // 2 = 4` pixels on each side (edge-padding) so every labeled pixel — including those near the border — can have a full `9×9` patch extracted.
3. For every pixel with label > 0, extract its `9×9×6` patch, store its `(row, col)` coordinates, and convert its label to 0-indexed.
4. Split all labeled patches into a **75% train pool** and a **25% test pool** using a stratified split (`TRAIN_PERCENT = 0.75`, `SEED = 42`). The train pool is not actually used further in this notebook (the models are pre-trained and loaded from disk) — only the test pool matters here.
5. Split the 25% test pool again, 50/50 (`CALIB_FRACTION_OF_TEST = 0.5`), into a **calibration set** (`x_cal`, `y_cal`, `coords_cal`) and an **evaluation set** (`x_eval`, `y_eval`, `coords_eval`), again stratified by class where possible (falling back to a non-stratified split if a class has too few members).

This calibration/evaluation split is the foundation of conformal prediction: the calibration set is used to choose a statistical threshold, and the evaluation set is used to check whether that threshold actually delivers the promised coverage on unseen data.

---

## Methods

### Method: Custom Keras Layers

#### a) What it is

> Think of these four classes as specialized "LEGO bricks" that the three pre-trained models (AlexNet-CNN, GFNet, ViT-UNet) were built from. Keras needs the exact blueprint for each brick to reconstruct a saved model, so these classes must be re-declared, with `@register_keras_serializable()`, before any model file is loaded.

#### b) Why it's used here

The three `.keras` model files were trained elsewhere using these custom layers. Without registering identical class definitions in this notebook, `keras.models.load_model(...)` would fail because Keras wouldn't know how to reconstruct layers like `GlobalFilterLayer` or `PatchEncoderWithCLS` from the saved file.

#### c) How it works — Step by step

1. **`PatchExtractor`** — splits an image into non-overlapping `patch_size × patch_size` patches using `tf.image.extract_patches`, then reshapes the result from `(batch, n_patches_h, n_patches_w, patch_dim)` into a flat sequence `(batch, num_patches, patch_dim)`. This is the standard "tokenize the image into patches" step used by Vision Transformers and GFNet.
2. **`PatchPositionEncoder`** — takes the flat patch sequence and (a) linearly projects each patch to `projection_dim` features via a `Dense` layer, and (b) adds a learned positional embedding (one embedding vector per patch index) so the model retains spatial ordering information that would otherwise be lost when patches are flattened into a sequence.
   ```
   encoded_patches = Dense(projection_dim)(patches) + PositionEmbedding(patch_indices)
   ```
3. **`GlobalFilterLayer`** — the core block of GFNet. For each channel, it learns a complex-valued 2D filter (`w_real` + `i * w_imag`) of shape `(token_side, token_side)`. It takes the input token sequence, reshapes it into a square `token_side × token_side` grid, applies a 2D FFT, multiplies element-wise by the learned complex filter in the frequency domain, applies the inverse FFT, keeps only the real part, and reshapes back to a sequence. This implements global (whole-image) frequency-domain filtering instead of local convolution.
   ```
   X_freq = FFT2D(reshape(x, token_side, token_side))
   X_filtered = X_freq * (w_real + i*w_imag)
   x_out = reshape(real(IFFT2D(X_filtered)), num_tokens, channels)
   ```
4. **`PatchEncoderWithCLS`** — like `PatchPositionEncoder`, but additionally prepends a single learnable **CLS token** (a vector of trainable weights, broadcast to every item in the batch) to the front of the patch sequence before adding positional embeddings. This is the classic ViT trick: the CLS token's final-layer representation is used as a summary of the whole image for classification.
5. All four classes are collected into a `CUSTOM_OBJECTS` dictionary, which is passed to `keras.models.load_model(..., custom_objects=CUSTOM_OBJECTS)` so Keras can map the class names stored in the `.keras` file back to these Python class definitions.

#### d) ASCII Flow Diagram

```
Saved .keras file (contains layer class names as strings)
        |
        v
CUSTOM_OBJECTS = { name -> class } dictionary
        |
        v
keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
        |
        v
Fully reconstructed model object, ready for .predict()
```

For `GlobalFilterLayer` specifically, the per-call data flow is:

```
Input tokens (B, N, C)
    |
    v
Reshape -> (B, token_side, token_side, C)
    |
    v
2D FFT  -----------------------\
    |                            \
    v                             v
Complex weight (w_real + i*w_imag)
    |                            /
    v                           /
Elementwise complex multiply  <-
    |
    v
2D Inverse FFT -> take real part
    |
    v
Reshape -> (B, N, C)
```

#### e) Worked Numerical Example

Take a tiny example: suppose `token_side = 2`, one channel, and the input grid (after reshaping) is:

```
x = [[1, 2],
     [3, 4]]
```

Suppose the learned filter weight (for simplicity) is a real-only filter `w_real = [[1, 0], [0, 1]]`, `w_imag = [[0,0],[0,0]]` (i.e., `w = identity`, all 1s on the diagonal in frequency space terms — this is a toy stand-in, not a real FFT pattern, just to show the arithmetic shape).

1. Compute `X_freq = FFT2D(x)` — a complex 2x2 array (the real FFT of `[[1,2],[3,4]]` has DC component `1+2+3+4=10`, and other complex entries encoding row/column frequency content).
2. Multiply elementwise: `X_filtered = X_freq * w` (with `w` as the complex weight grid).
3. Apply `IFFT2D(X_filtered)` and take the real part: `x_out`.
4. If `w` were exactly all-ones (pass-through filter), `x_out` would equal `x` again (FFT followed by IFFT with an identity filter is the identity transform). If `w` instead zeroed out high-frequency components, `x_out` would be a smoothed version of `x` — e.g., something closer to `[[2.5, 2.5],[2.5, 2.5]]` (the average), illustrating how this layer can learn to either preserve or blur spatial patterns depending on what the trained weights look like.

#### f) Code Walkthrough

```python
@tf.keras.utils.register_keras_serializable()
class GlobalFilterLayer(layers.Layer):
    """Applies a learnable 2-D frequency filter via FFT (GFNet core block)."""

    def __init__(self, token_side, **kwargs):
        super().__init__(**kwargs)
        self.token_side = token_side  # side length of the square token grid (e.g. 3 for a 3x3=9 token grid)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        # Real part of the learnable complex filter, one (token_side x token_side) filter per channel
        self.w_real = self.add_weight(
            name='w_real', shape=(self.token_side, self.token_side, channels),
            initializer='glorot_uniform', trainable=True,
        )
        # Imaginary part, initialized to zero so the filter starts as purely real
        self.w_imag = self.add_weight(
            name='w_imag', shape=(self.token_side, self.token_side, channels),
            initializer='zeros', trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        b  = tf.shape(x)[0]
        c  = tf.shape(x)[-1]
        x2 = tf.reshape(x, [b, self.token_side, self.token_side, c])  # back to a square spatial grid
        x_fft = tf.signal.fft2d(tf.cast(x2, tf.complex64))            # 2D FFT over the spatial grid
        w  = tf.complex(self.w_real, self.w_imag)                     # combine real+imag into one complex weight
        x_i = tf.math.real(tf.signal.ifft2d(x_fft * w))               # filter in frequency domain, invert, take real part
        return tf.reshape(x_i, [b, self.token_side * self.token_side, c])  # flatten back to a token sequence

    def get_config(self):
        c = super().get_config()
        c.update({'token_side': self.token_side})  # needed so Keras can re-create this layer when reloading
        return c
```

#### g) Output & Interpretation

These layers don't produce a final result on their own — they're internal building blocks. Their "output" is the reconstructed model objects in `models = load_models(...)`, which can then be called with `.predict(...)` on `9×9×6` patches to produce per-class probability vectors. The smoke test in Section 6.0 checks that each model's output has shape `(N, num_classes)` and contains only finite numbers — that's the practical confirmation that the custom layers were registered correctly and the architecture reconstructed properly.

#### h) Limitations

- These class definitions must match **exactly** (same `get_config`/`__init__` signatures) what was used at training time, or deserialization can silently produce a structurally different model or fail outright.
- `GlobalFilterLayer` assumes the token sequence length is a perfect square (`token_side * token_side`); if the actual sequence length doesn't match, the `reshape` will fail.
- The FFT-based filter is global (whole feature map), so it cannot represent purely local patterns as efficiently as a small convolution kernel could.
- `PatchEncoderWithCLS`'s CLS token is a single shared learned vector broadcast across the batch — it carries no per-sample information until the model's attention layers mix it with patch tokens.

---

### Method: Model Loading with Trust Checks

#### a) What it is

> This is like a security guard checking ID before letting three "expert consultants" (the trained models) into the building, then giving each one a quick verbal quiz (the smoke test) to make sure they can actually answer in the expected format before the real work begins.

#### b) Why it's used here

The notebook needs to load three pre-trained `.keras` model files (`AlexNet_CNN`, `GFNet`, `ViT_UNet`) from disk. Two concerns are addressed: (1) only load model files from a pre-approved directory (`TRUSTED_MODEL_ROOTS`), to avoid deserializing arbitrary/untrusted files, and (2) handle a known Keras deserialization quirk where models containing `Lambda` layers or requiring "unsafe" deserialization fail under default `safe_mode=True` settings.

#### c) How it works — Step by step

1. **Path trust check (`is_trusted_model_path`)**: resolve the candidate model path to an absolute path, and check whether it is equal to, or nested inside, one of the paths in `TRUSTED_MODEL_ROOTS`. If not, raise a `RuntimeError` immediately — the model is never even attempted to load.
2. **Primary load attempt**: for each model, call `keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False, safe_mode=False)`. `compile=False` skips reconstructing the optimizer/loss (not needed for inference). If this succeeds, store the model and move to the next one.
3. **Failure diagnosis**: if the primary load raises an exception, print the error and a hint to check file integrity, custom objects, and TF/Keras version compatibility.
4. **Conditional fallback**: if the error message mentions "lambda" or "safe_mode" (case-insensitive), call `keras.config.enable_unsafe_deserialization()` (a global flag that allows Keras to deserialize layers like `Lambda` that can execute arbitrary code) and retry the load.
5. **Final failure**: if the fallback also fails, or if the original error wasn't lambda/safe_mode-related, raise a `RuntimeError` with both error messages so the user can diagnose the issue.
6. **Smoke test**: after loading, run all three models on the first 8 evaluation patches (`x_eval[:8]`) and assert that (a) the output is 2D, (b) the second dimension equals `num_classes`, and (c) all output values are finite (no `NaN`/`Inf`).

#### d) ASCII Flow Diagram

```
For each model_key, path in MODEL_FILES:
        |
        v
  is_trusted_model_path(path)? --no--> RuntimeError (stop)
        | yes
        v
  Try: load_model(safe_mode=False)
        |
   success? --yes--> store model, continue to next model
        | no
        v
  Error mentions "lambda"/"safe_mode"?
        | yes                          | no
        v                              v
  enable_unsafe_deserialization()   RuntimeError (stop)
  retry load_model(...)
        |
   success? --yes--> store model
        | no
        v
  RuntimeError (stop, with both errors)

After all models loaded:
        |
        v
  Smoke test on x_eval[:8] for each model
        |
        v
  Assert: 2D output, correct class count, all finite
```

#### e) Worked Numerical Example

Suppose `num_classes = 5` and `x_eval[:8]` has shape `(8, 9, 9, 6)`. After `model.predict(x_smoke, verbose=0)`, suppose the output `p` has shape `(8, 5)` — 8 samples, each a 5-element probability-like vector, e.g. for sample 1: `p[0] = [0.05, 0.10, 0.60, 0.20, 0.05]`. The checks then verify:
- `p.ndim == 2` → `True` (it's a 2D array).
- `p.shape[1] == num_classes` → `5 == 5` → `True`.
- `np.isfinite(p).all()` → `True` as long as no entry is `NaN` or `±inf`.

If instead the model accidentally output shape `(8, 5, 1)` (3D), the first assertion would fail with a clear error naming which model and what shape was found.

#### f) Code Walkthrough

```python
def is_trusted_model_path(path: Path) -> bool:
    """Return True only if `path` is inside one of the pre-approved roots."""
    p = path.expanduser().resolve()          # normalize to an absolute, symlink-resolved path
    for root in TRUSTED_MODEL_ROOTS:
        r = root.expanduser().resolve()       # normalize the trusted root the same way
        if p == r or r in p.parents:          # path equals root, or root is an ancestor directory of path
            return True
    return False
```

```python
# Smoke test — verify output shape and numerical validity
x_smoke = x_eval[:8]
for key, model in models.items():
    p = model.predict(x_smoke, verbose=0)
    assert p.ndim == 2,               f'{key}: expected rank-2 output, got {p.shape}'
    assert p.shape[1] == num_classes, f'{key}: expected class dim {num_classes}, got {p.shape[1]}'
    assert np.isfinite(p).all(),      f'{key}: NaN/Inf in prediction output'
    print(key, 'smoke output shape:', p.shape)
```

#### g) Output & Interpretation

The output is the `models` dictionary, mapping each model key (`AlexNet_CNN`, `GFNet`, `ViT_UNet`) to a loaded, ready-to-use Keras model object, plus printed confirmation lines from the smoke test. If everything passes, you see three lines like `AlexNet_CNN smoke output shape: (8, 5)`. If anything fails, the notebook stops with an informative assertion error rather than proceeding to run hours of full-scene inference on a broken model.

#### h) Limitations

- The trust check only validates the *path location*, not the file's actual contents/integrity — a malicious or corrupted file placed inside a trusted directory would still pass this check.
- `enable_unsafe_deserialization()` is a global setting; once enabled, it remains enabled for the rest of the session, which could allow unsafe deserialization of *other* models loaded afterward too.
- The smoke test only checks shape and finiteness, not correctness of predictions (e.g., it wouldn't catch a model that loaded with subtly wrong weights but still produces well-formed numeric output).

---

### Method: Probability Normalisation

#### a) What it is

> This is a "cleanup crew" for raw model outputs — it makes sure every prediction is a proper probability distribution (non-negative numbers that sum to 1) even if the raw network output has tiny negative values, `NaN`s, or doesn't sum exactly to 1 due to floating-point error.

#### b) Why it's used here

Conformal prediction's math (especially the APS score, described next) assumes the input is a valid probability vector. Raw softmax outputs from `model.predict(...)` are *usually* already close to valid probabilities, but floating-point drift or numerical issues (especially with the custom FFT-based `GlobalFilterLayer`) could introduce small negative values or `NaN`s, which would break downstream cumulative-sum and quantile computations.

#### c) How it works — Step by step

1. Cast the input array to 64-bit floats (`np.float64`) for numerical precision.
2. Replace any `NaN`, `+inf`, or `-inf` values with `0.0` using `np.nan_to_num`.
3. Clip all values to the `[0, 1]` range with `np.clip`.
4. Compute the row-sum (`rs`) across the class axis (`axis=-1`).
5. Where the row-sum is at or below a tiny epsilon (`eps`, i.e., effectively zero — meaning the whole row was wiped out by step 2), replace the row-sum with `1.0` to avoid division by zero.
6. Divide each row by its row-sum, so each row sums to (approximately) 1.

```
prob_clean = clip(nan_to_num(prob), 0, 1)
row_sum    = sum(prob_clean, axis=-1)
row_sum    = 1.0 where row_sum <= eps else row_sum
prob_normalized = prob_clean / row_sum
```

#### d) ASCII Flow Diagram

```
Raw model output (N, C)
        |
        v
Cast to float64
        |
        v
Replace NaN/Inf -> 0.0
        |
        v
Clip to [0, 1]
        |
        v
Row-sum across classes ----> if row_sum <= eps, set row_sum = 1.0
        |                                |
        v                                v
        +------------ divide ------------+
        |
        v
Normalized probabilities (rows sum to ~1)
```

#### e) Worked Numerical Example

Suppose for one pixel the raw model output (3 classes) is `[0.5, -0.01, 0.6]` (slightly negative due to floating point noise, and not summing to 1).

1. `nan_to_num`: no `NaN`/`Inf` present, so unchanged: `[0.5, -0.01, 0.6]`.
2. `clip(0, 1)`: the negative value is clipped to 0: `[0.5, 0.0, 0.6]`.
3. `row_sum = 0.5 + 0.0 + 0.6 = 1.1`.
4. `1.1 > eps`, so row_sum stays `1.1`.
5. Normalize: `[0.5/1.1, 0.0/1.1, 0.6/1.1] = [0.4545, 0.0, 0.5455]`.

Now the row sums to exactly 1 and all entries are non-negative.

A degenerate case: if the raw output were `[NaN, NaN, NaN]`:
1. `nan_to_num` → `[0, 0, 0]`.
2. `clip` → `[0, 0, 0]`.
3. `row_sum = 0`, which is `<= eps`, so `row_sum` is set to `1.0`.
4. Normalize: `[0/1, 0/1, 0/1] = [0, 0, 0]` — a degenerate all-zero "probability" vector that at least won't cause a division-by-zero crash downstream.

#### f) Code Walkthrough

```python
def normalize_probs(prob, eps=1e-12):
    """Clip, nan-guard, and L1-normalise a probability array along axis=-1."""
    prob = np.asarray(prob, dtype=np.float64)               # promote to float64 for precision
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)  # remove NaN/Inf
    prob = np.clip(prob, 0.0, 1.0)                            # enforce valid probability range
    rs   = prob.sum(axis=-1, keepdims=True)                   # per-row sum, keep dims for broadcasting
    rs   = np.where(rs <= eps, 1.0, rs)                       # avoid divide-by-zero on degenerate rows
    return prob / rs                                          # L1-normalize each row to sum to 1

def predict_probs(model, x, batch_size=128):
    """Run model inference and return normalised softmax probabilities."""
    return normalize_probs(model.predict(x, batch_size=batch_size, verbose=0), eps=EPS)
```

#### g) Output & Interpretation

The output is an `(N, num_classes)` array where every row is a valid probability distribution: non-negative entries summing to 1 (or all-zero in the pathological all-`NaN` case). This is the array that feeds directly into `compute_aps_scores`. High values in a particular class column mean the model is confident that class is correct for that pixel; the *distribution shape* (peaked vs. flat) is what the APS score in the next section turns into a conformity score.

#### h) Limitations

- The all-zero fallback for degenerate rows (`row_sum <= eps`) produces a vector that doesn't actually sum to 1 — it's a "safe but not meaningful" placeholder, and any downstream computation on such a row should be treated with caution.
- Clipping to `[0, 1]` before normalizing can slightly change the *relative* proportions compared to clipping after normalizing, though for typical near-valid probabilities the effect is negligible.
- This function does not detect or report *how many* rows were problematic — it silently repairs them.

---

### Method: Randomised APS Conformity Scores

#### a) What it is

> Adaptive Prediction Sets (APS) scoring is like asking, "If I sorted all possible answers from most-likely to least-likely according to the model, how far down that ranked list would I have to go before I'd include the *true* answer?" The "randomised" part adds a small random tie-breaking nudge so that, on average across many examples, the resulting prediction sets hit the target coverage level exactly rather than being conservative by a discrete jump.

#### b) Why it's used here

APS is the conformity score used by SACP. It converts each pixel's probability vector into a single number (the "score") that measures how "surprising" or "non-conforming" a particular class label would be given the model's prediction. Lower scores mean the class is well-supported by the model; calibrating a threshold on these scores (on the calibration set) and then including all classes with score ≤ threshold (on the evaluation set) is what produces statistically valid prediction sets.

#### c) How it works — Step by step

1. Sort each row of probabilities in **descending** order, keeping track of which original class index each sorted position corresponds to (`sorted_indices`).
2. Compute the cumulative sum of the sorted probabilities along each row (`cumsum`) — so `cumsum[i, j]` is the sum of the top `j+1` probabilities for sample `i`.
3. Draw one uniform random number `U[i]` in `[0, 1)` per sample, using a fixed-seed random generator (so results are reproducible).
4. **If `labels` is given** (calibration mode — computing one score per sample for its *true* class):
   - Find `rank`, the position of the true label in the sorted (descending) order (0 = most likely class, 1 = second most likely, etc.).
   - If `rank == 0` (the true class is the model's top prediction): `score = U[i] * sorted_probs[i, 0]` — a randomized fraction of just the top probability.
   - Otherwise: `score = cumsum[i, rank-1] + U[i] * sorted_probs[i, rank]` — the sum of probabilities for all classes ranked *above* the true class, plus a randomized fraction of the true class's own probability.
5. **If `labels` is `None`** (inference mode — computing a score for *every* class, for every sample): build the same kind of score for each rank position `0..C-1` (using the same formula pattern: cumulative sum of higher-ranked probabilities plus a randomized fraction of the current rank's probability), then scatter these scores back into the original (unsorted) class order using `sorted_indices`, producing a full `(N, C)` score matrix.

```
For the true-label case (rank = position of true class in sorted order):
  if rank == 0:
      score = U * p_sorted[0]
  else:
      score = sum(p_sorted[0 .. rank-1]) + U * p_sorted[rank]
```

#### d) ASCII Flow Diagram

```
Probabilities (N, C)
        |
        v
Sort each row descending --> sorted_probs, sorted_indices
        |
        v
Cumulative sum along sorted axis --> cumsum
        |
        v
Draw U[i] ~ Uniform(0,1) per sample
        |
        +-------------------------------+
        |                                |
   labels given?                    labels = None
        | yes                            | no
        v                                v
For true class at rank r:        For every rank r (0..C-1):
  rank==0: score = U * p[0]        rank==0: score = U * p[0]
  else:    score = cumsum[r-1]     else:    score = cumsum[r-1] + U*p[r]
                  + U * p[r]               |
        |                                  v
        v                          Scatter scores back to
  1-D array of scores (N,)          original class order
                                            |
                                            v
                                  Full score matrix (N, C)
```

#### e) Worked Numerical Example

Suppose for one pixel there are 3 classes with model probabilities `p = [0.5, 0.3, 0.2]` (already in original class order: class 0, class 1, class 2), and `U = 0.4` for this sample.

1. **Sort descending**: `sorted_probs = [0.5, 0.3, 0.2]`, `sorted_indices = [0, 1, 2]` (already sorted in this example).
2. **Cumulative sum**: `cumsum = [0.5, 0.8, 1.0]`.

**Calibration-mode example** — suppose the *true* label is class 1, which sits at `rank = 1` in the sorted order:
- Since `rank != 0`: `score = cumsum[rank-1] + U * sorted_probs[rank] = cumsum[0] + 0.4 * 0.3 = 0.5 + 0.12 = 0.62`.
- Interpretation: to "reach" class 1 in the ranked list, you must include all of class 0's probability mass (0.5) plus a random 40% slice of class 1's own mass (0.12), totaling 0.62.

**Inference-mode example** — compute a score for *every* class (still `U = 0.4`):
- `rank = 0` (class 0): `score = U * sorted_probs[0] = 0.4 * 0.5 = 0.20`.
- `rank = 1` (class 1): `score = cumsum[0] + U * sorted_probs[1] = 0.5 + 0.4*0.3 = 0.62`.
- `rank = 2` (class 2): `score = cumsum[1] + U * sorted_probs[2] = 0.8 + 0.4*0.2 = 0.88`.
- Since `sorted_indices = [0,1,2]` here, scattering back to original order gives: `scores_matrix = [0.20, 0.62, 0.88]` for classes `[0, 1, 2]` respectively.
- Interpretation: class 0 (the model's top pick) gets the lowest score (easiest to "include"), and class 2 (least likely) gets the highest score (hardest to include) — exactly as desired, since a prediction set built by thresholding scores will tend to include high-probability classes first.

#### f) Code Walkthrough

```python
def compute_aps_scores(self, probabilities, labels=None):
    """
    Compute randomised APS conformity scores.

    If `labels` is provided, returns a 1-D score per sample (calibration).
    Otherwise returns the full (N, C) score matrix (inference).
    """
    n              = probabilities.shape[0]
    sorted_indices = np.argsort(probabilities, axis=1)[:, ::-1]   # descending order indices, per row
    sorted_probs   = np.take_along_axis(probabilities, sorted_indices, axis=1)  # reorder probs to match
    cumsum         = np.cumsum(sorted_probs, axis=1)              # running total of sorted probs

    rng = np.random.default_rng(self.seed)   # reproducible RNG (note: re-seeded every call!)
    U   = rng.random(n)                       # one uniform random number per sample

    if labels is not None:
        scores = np.zeros(n)
        for i in range(n):
            y    = int(labels[i])
            rank = int(np.where(sorted_indices[i] == y)[0][0])  # where does the true class sit in the ranking?
            if rank == 0:
                scores[i] = U[i] * sorted_probs[i, 0]            # top-ranked: only a random slice of its own mass
            else:
                # everything ranked above it, plus a random slice of its own mass
                scores[i] = cumsum[i, rank - 1] + U[i] * sorted_probs[i, rank]
        return scores

    # Inference mode: compute a score for every class, for every sample
    scores_matrix = np.zeros_like(probabilities)
    for i in range(n):
        scores_sorted    = np.zeros(self.num_classes)
        scores_sorted[0] = U[i] * sorted_probs[i, 0]
        scores_sorted[1:] = cumsum[i, :-1] + U[i] * sorted_probs[i, 1:]
        scores_matrix[i, sorted_indices[i]] = scores_sorted   # un-sort back to original class order
    return scores_matrix
```

#### g) Output & Interpretation

- In **calibration mode**, the output is a 1-D array of length `N` — one "true-class APS score" per calibration pixel. These scores form the empirical distribution from which the conformal threshold `q_hat` is later chosen.
- In **inference mode**, the output is an `(N, C)` matrix — every class gets its own score for every pixel. Lower scores correspond to classes the model considers more plausible. A prediction set for a pixel is later formed by keeping all classes whose score is `<= q_hat`.
- A **low** true-class score (close to 0) means the model assigned that pixel's true class a very high probability (it was near the top of the ranking) — this pixel is "easy." A **high** score (close to 1) means the true class was deep in the ranked list — this pixel was "hard" or the model was very wrong/uncertain.

#### h) Limitations

- The randomness `U[i]` is regenerated from `np.random.default_rng(self.seed)` **every time `compute_aps_scores` is called** — since the same `seed` is reused, calls with the same `n` produce the *same* sequence of `U` values, but calls with different `n` (e.g., calibration vs. evaluation set sizes) get different `U` arrays drawn from the same seeded stream starting point. This is a subtle reproducibility detail worth being aware of if comparing scores across calls.
- The randomization assumes `U ~ Uniform(0,1)` independent of the data; this is what gives APS its *exact* (not just conservative) marginal coverage property in the standard (non-spatial) conformal setting.
- The function uses Python-level `for i in range(n)` loops over potentially hundreds of thousands of pixels (full-scene inference has `H*W = 330*307 ≈ 101,310` pixels), which is computationally expensive compared to a fully vectorized implementation.

---

### Method: Spatial Smoothing of Conformity Scores

#### a) What it is

> Imagine each pixel has "voted" on how surprising each class label would be (that's its APS score vector). Spatial smoothing is like each pixel partially adopting the *average opinion of its neighbors* — blending its own vote with the neighborhood's average vote — because in real images, neighboring pixels are very likely to belong to the same class, so their scores should agree.

#### b) Why it's used here

This is the "Spatial" part of SACP. Standard (non-spatial) conformal prediction treats every pixel independently, ignoring the strong spatial correlation in image data (a pixel's class is highly predictable from its neighbors' classes). By smoothing each pixel's score map with its local neighborhood, SACP produces scores — and ultimately prediction sets — that are spatially coherent, which can tighten prediction sets (smaller, more confident sets) in homogeneous regions while still flagging genuinely ambiguous boundary regions as uncertain.

#### c) How it works — Step by step

1. Start with a `score_map` of shape `(H, W, num_classes)` (one APS score vector per pixel) and a boolean `mask_map` of shape `(H, W)` indicating which pixels actually have valid scores (i.e., are part of the calibration or evaluation set, or — in the full-scene case — all pixels).
2. Make a copy of the score map called `smoothed` (so updates don't affect the neighbor-lookups of pixels processed later in the same pass).
3. For every pixel `(r, c)` where `mask_map[r, c]` is `True`:
   - Look at all precomputed neighbor offsets `(dr, dc)` for the configured `window_size` (e.g., for `window_size=3`, this is the 8 surrounding pixels in a 3×3 block, excluding the center).
   - For each neighbor `(r+dr, c+dc)` that is within image bounds **and** also has `mask_map == True`, add its score vector to a running sum `n_sum` and increment `n_count`.
   - If at least one valid neighbor was found (`n_count > 0`), update `smoothed[r, c] = lambda * original_score + lambda * (neighbor_average)`.
   - If no valid neighbors exist (e.g., an isolated masked pixel), `smoothed[r, c]` remains exactly the original `score_map[r, c]` (unchanged copy).
4. Return the `smoothed` map. This whole process can be repeated `k` times (the SACP hyperparameter `SACP_K = 1` in this notebook, so by default it runs once).

```
smoothed[r, c] = lambda * original[r, c] + lambda * mean(original[neighbors])
               (only if at least one valid neighbor exists)
```

#### d) ASCII Flow Diagram

```
score_map (H, W, C), mask_map (H, W)
        |
        v
smoothed = copy(score_map)
        |
        v
For each masked pixel (r, c):
    |
    v
  Gather valid neighbor scores within window
    |
    v
  n_count > 0?  --no--> leave smoothed[r,c] unchanged
    | yes
    v
  smoothed[r,c] = lambda*original + lambda*mean(neighbors)
        |
        v
Repeat k times (SACP_K)
        |
        v
Final smoothed score map
```

For `window_size = 3`, the neighbor pattern (✕ = center pixel, excluded; ● = neighbors used) is:

```
● ● ●
● ✕ ●
● ● ●
```

For `window_size = 5`, the neighbor pattern expands to a 5×5 block (24 neighbors), and similarly for 7 and 9.

#### e) Worked Numerical Example

Suppose `window_size = 3`, `lambda = 0.5`, and `num_classes = 2`. Consider a tiny 3×3 grid of single-class scores (for simplicity, treat each pixel's score as a scalar instead of a 2-vector — the same arithmetic applies independently per class):

```
Original score_map (one class shown):
[ 0.10  0.20  0.30 ]
[ 0.40  0.50  0.60 ]
[ 0.70  0.80  0.90 ]
```

All 9 pixels are masked (`mask_map` all `True`). Compute the smoothed value for the **center pixel** `(1,1)`, whose original score is `0.50`:

1. Its 8 neighbors (3×3 window minus center) are: `0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90`.
2. Sum of neighbors: `0.10+0.20+0.30+0.40+0.60+0.70+0.80+0.90 = 4.00`.
3. Count of neighbors: `8`.
4. Neighbor average: `4.00 / 8 = 0.50`.
5. Smoothed value: `0.5 * 0.50 (original) + 0.5 * 0.50 (neighbor avg) = 0.25 + 0.25 = 0.50`.

In this symmetric example the value happens to stay the same. Now compute for a **corner pixel** `(0,0)`, whose original score is `0.10`:

1. In a 3×3 grid, `(0,0)` has only 3 valid in-bounds neighbors (no pixels exist above or to the left): `(0,1)=0.20`, `(1,0)=0.40`, `(1,1)=0.50`.
2. Sum of neighbors: `0.20 + 0.40 + 0.50 = 1.10`.
3. Count: `3`.
4. Neighbor average: `1.10 / 3 ≈ 0.3667`.
5. Smoothed value: `0.5 * 0.10 + 0.5 * 0.3667 = 0.05 + 0.1833 = 0.2333`.

So the corner pixel's score moved from `0.10` toward `0.2333` — pulled up toward its (higher-scoring) neighbors.

#### f) Code Walkthrough

```python
def spatial_smoothing(self, score_map, mask_map):
    """One pass of spatial score smoothing over the neighbourhood window."""
    smoothed    = np.copy(score_map)            # don't mutate input or affect later lookups in this pass
    H, W, C     = score_map.shape
    rows, cols  = np.where(mask_map)             # coordinates of all masked (valid) pixels

    for r, c in zip(rows, cols):
        ori     = score_map[r, c]                # this pixel's original score vector (length C)
        n_sum   = np.zeros(C)
        n_count = 0
        for dr, dc in self.neighbors:            # precomputed offsets for this window_size
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask_map[nr, nc]:  # in-bounds and valid
                n_sum   += score_map[nr, nc]
                n_count += 1
        if n_count > 0:
            # blend original with neighborhood average, weighted by lambda on each side
            smoothed[r, c] = self.lmd * ori + self.lmd * (n_sum / n_count)
    return smoothed
```

The `neighbors` list itself is precomputed in `__init__`:

```python
radius = window_size // 2
self.neighbors = [
    (dr, dc)
    for dr in range(-radius, radius + 1)
    for dc in range(-radius, radius + 1)
    if not (dr == 0 and dc == 0)   # exclude the center pixel itself
]
```

#### g) Output & Interpretation

The output is a smoothed `(H, W, num_classes)` score map where each pixel's APS score vector has been pulled toward its local neighborhood's average. After smoothing, pixels in a spatially homogeneous region (where all neighbors agree the same class is likely) will have lower, more confident scores for that class; pixels near class boundaries — where neighbors disagree — will retain higher, more uncertain scores. This smoothed map is what gets thresholded by `q_hat` to build prediction sets.

#### h) Limitations

- With `lambda = 0.5`, note that `0.5 * original + 0.5 * neighbor_avg` does **not** preserve the original value's scale on its own when `n_count` neighbors are averaged uniformly — it's a fixed 50/50 blend regardless of how many neighbors contributed, so a pixel with only 1 valid neighbor is treated the same (50/50) as one with 24 valid neighbors (for `window_size=9`).
- Pixels at image borders or with few masked neighbors get less "spatial information" baked in (fewer neighbors to average), which could make border regions behave differently than interior regions.
- The smoothing is applied for `k` passes (`SACP_K = 1` here), but is **not** truly iterative-diffusion in the classic sense within a single pass — each pixel's update within one pass uses the *original* (pre-pass) neighbor values (`score_map`, not `smoothed`), avoiding order-dependence, but across passes (`k > 1`) the smoothing compounds.
- Computationally this is an `O(H * W * |neighbors|)` Python-level double loop, which becomes slow for larger window sizes (e.g., `window_size=9` has 80 neighbors per pixel) and for full-scene maps with ~100K pixels.

---

### Method: SACP Calibration and Prediction Set Construction

#### a) What it is

> This is the "rulebook writing" step. Using a held-out calibration set (pixels whose true labels are known but not used for training), SACP figures out exactly *how strict* the score threshold needs to be so that, when applied to brand-new evaluation pixels, the resulting prediction sets contain the true class at least `1 - alpha` (e.g., 95%) of the time.

#### b) Why it's used here

This is the heart of conformal prediction's statistical guarantee. Rather than trusting the model's raw probabilities directly (which can be over- or under-confident), SACP empirically measures how "surprising" the *true* labels of calibration pixels were (via APS scores, after spatial smoothing) and picks a threshold `q_hat` such that a user-chosen fraction (`1 - alpha`) of calibration true-class scores fall at or below it. Applying that same threshold to evaluation pixels — "include every class whose score is ≤ q_hat" — is what (under suitable exchangeability assumptions) yields the coverage guarantee.

#### c) How it works — Step by step

1. **Compute APS scores** for both the calibration probabilities (`calib_probs`) and the test/evaluation probabilities (`test_probs`), using `compute_aps_scores` in inference mode (no labels) — producing full `(N, num_classes)` score matrices for both sets.
2. **Populate a spatial score map**: create an all-zero `(H, W, num_classes)` array (`score_map`) and an all-`False` `(H, W)` boolean mask (`mask_map`). Place the calibration pixels' score vectors at their `(row, col)` coordinates and mark those positions `True` in the mask; do the same for the evaluation pixels.
3. **Iterative spatial smoothing**: apply `spatial_smoothing` to `score_map` (using `mask_map`) exactly `k` times (here `k=1`), producing `current_map`.
4. **Calibration quantile (`q_hat`)**:
   - For each calibration pixel `i` at coordinates `(r, c)`, extract `current_map[r, c, true_label_i]` — i.e., the *smoothed score the true class would have received*. Collect these into `fused_calib_scores`.
   - Compute the conformal quantile level: `q_level = ceil((n+1) * (1-alpha)) / n`, clipped to `[0, 1]`, where `n` is the number of calibration samples. This is the standard "finite-sample correction" formula from split conformal prediction — it ensures the *marginal* coverage guarantee holds exactly (in expectation) for finite `n`, not just asymptotically.
   - Compute `q_hat = quantile(fused_calib_scores, q_level, method='higher')` — the `method='higher'` argument means: if the desired quantile falls between two data points, round **up** to the higher one (a conservative choice that ensures the guarantee holds).
5. **Build prediction sets for evaluation pixels**: for each evaluation pixel `i` at `(r, c)`, the prediction set is `current_map[r, c, :] <= q_hat` — a boolean vector over classes, `True` for every class whose (smoothed) score doesn't exceed the threshold.
6. **Non-empty guarantee**: if a prediction set ends up with *no* `True` entries (every class's score exceeded `q_hat`), force the single class with the *minimum* score into the set — guaranteeing every pixel gets at least one candidate class.
7. Return the boolean prediction-sets array, `q_hat`, and the average set size on the evaluation set.

```
q_level = min(1, max(0, ceil((n+1)*(1-alpha)) / n))
q_hat   = quantile(fused_calib_scores, q_level, method='higher')

prediction_set(pixel) = { class c : smoothed_score[pixel, c] <= q_hat }
                         (or {argmin class} if that set would be empty)
```

#### d) ASCII Flow Diagram

```
calib_probs (N_cal, C)        test_probs (N_eval, C)
        |                              |
        v                              v
  compute_aps_scores()           compute_aps_scores()
  (inference mode, no labels)    (inference mode, no labels)
        |                              |
        v                              v
calib_scores_mat (N_cal, C)    test_scores_mat (N_eval, C)
        |                              |
        +--------- scatter into ------+
                  score_map (H, W, C)
                  mask_map  (H, W)
                          |
                          v
              spatial_smoothing x k  --> current_map
                          |
        +------------------------------------+
        |                                     |
        v                                     v
For calib pixels: gather                For eval pixels:
true-class smoothed scores              pred_set = current_map <= q_hat
        |                                     |
        v                                     v
  fused_calib_scores                    (fix empty sets: force argmin class)
        |                                     |
        v                                     |
  q_level = ceil((n+1)(1-alpha))/n            |
  q_hat = quantile(., q_level,'higher')       |
        |                                     |
        +---------------> q_hat used here ---+
                          |
                          v
              pred_sets, q_hat, avg_size
```

#### e) Worked Numerical Example

Suppose `alpha = 0.05` (target 95% coverage), `num_classes = 3`, and there are `n = 4` calibration pixels with the following **smoothed true-class scores** (`fused_calib_scores`): `[0.10, 0.30, 0.45, 0.90]`.

1. **Quantile level**: `q_level = ceil((4+1)*(1-0.05)) / 4 = ceil(5 * 0.95) / 4 = ceil(4.75) / 4 = 5 / 4 = 1.25`. Clipped to `[0,1]`: `q_level = min(1.0, max(0.0, 1.25)) = 1.0`.
2. **q_hat**: `quantile([0.10, 0.30, 0.45, 0.90], 1.0, method='higher')`. The 100th percentile with `method='higher'` is simply the maximum value: `q_hat = 0.90`.

   *(This illustrates an important edge case: with very few calibration samples and a small `alpha`, `q_level` can hit its upper clip of `1.0`, forcing `q_hat` to be the single largest observed calibration score — a conservative threshold that guarantees very wide/inclusive prediction sets.)*

3. **Building a prediction set** for one evaluation pixel with smoothed scores (per class) `[0.20, 0.85, 0.95]`:
   - Compare each to `q_hat = 0.90`: `[0.20 <= 0.90, 0.85 <= 0.90, 0.95 <= 0.90] = [True, True, False]`.
   - Prediction set = `{class 0, class 1}` (2 classes included).
   - Set is non-empty, so no fallback needed.

4. **An empty-set edge case** — suppose another evaluation pixel has scores `[0.95, 0.97, 0.99]` (all above `q_hat = 0.90`):
   - `[0.95<=0.90, 0.97<=0.90, 0.99<=0.90] = [False, False, False]` → empty set.
   - Fallback: find `argmin([0.95, 0.97, 0.99]) = 0` (class 0 has the smallest score), so force `pred_set = [True, False, False]`.
   - Final prediction set = `{class 0}` — a singleton, guaranteed non-empty.

#### f) Code Walkthrough

```python
def fit_calibrate(self, calib_probs, calib_labels, calib_indices,
                  test_probs, test_indices):
    """
    Calibrate q_hat on calib set, then return prediction sets for test set.
    """
    calib_scores_mat = self.compute_aps_scores(calib_probs)   # (N_cal, C) — score for every class, every calib pixel
    test_scores_mat  = self.compute_aps_scores(test_probs)    # (N_eval, C) — same for eval pixels

    # Populate spatial score map across the whole H x W grid
    score_map = np.zeros((self.H, self.W, self.num_classes), dtype=np.float64)
    mask_map  = np.zeros((self.H, self.W), dtype=bool)

    for i, (r, c) in enumerate(calib_indices):
        score_map[r, c] = calib_scores_mat[i]
        mask_map[r, c]  = True
    for i, (r, c) in enumerate(test_indices):
        score_map[r, c] = test_scores_mat[i]
        mask_map[r, c]  = True

    # Iterative spatial smoothing (k passes)
    current_map = score_map
    for _ in range(self.k):
        current_map = self.spatial_smoothing(current_map, mask_map)

    # Calibration quantile: gather the smoothed score of the TRUE class for each calib pixel
    fused_calib_scores = np.array([
        current_map[r, c, int(calib_labels[i])]
        for i, (r, c) in enumerate(calib_indices)
    ])
    n       = len(fused_calib_scores)
    q_level = np.ceil((n + 1) * (1 - self.alpha)) / n   # finite-sample conformal correction
    q_level = min(1.0, max(0.0, q_level))                # clip to valid quantile range
    q_hat   = float(np.quantile(fused_calib_scores, q_level, method='higher'))

    # Build prediction sets for evaluation pixels by thresholding
    pred_sets = np.zeros((len(test_indices), self.num_classes), dtype=bool)
    for i, (r, c) in enumerate(test_indices):
        pred_sets[i] = (current_map[r, c] <= q_hat)
        if not pred_sets[i].any():  # guarantee non-empty set
            pred_sets[i, int(np.argmin(current_map[r, c]))] = True

    avg_size = float(pred_sets.sum(axis=1).mean())
    return pred_sets, q_hat, avg_size
```

#### g) Output & Interpretation

- **`pred_sets`**: an `(N_eval, num_classes)` boolean array — each row is the prediction set for one evaluation pixel (which classes are considered "plausible enough").
- **`q_hat`**: the single calibrated threshold (a float between 0 and 1) applied uniformly to all evaluation pixels.
- **`avg_size`**: the mean number of classes per prediction set on the evaluation data.
- Interpretation: a **small average set size** close to 1 means the model + SACP are confident and precise (most pixels get a single predicted class). A **larger average set size** means more pixels are ambiguous enough that multiple classes had to be retained to meet the coverage target. The **empirical coverage** (computed separately in `compute_set_metrics`, described later) should be close to `1 - alpha` if the conformal guarantee is working as intended.

#### h) Limitations

- The conformal coverage guarantee from standard split-conformal theory assumes **exchangeability** between calibration and test data; the spatial smoothing step (which mixes information between calibration and evaluation pixels that are spatial neighbors) is a deviation from the textbook i.i.d./exchangeable setting, so the *exact* finite-sample guarantee may only hold approximately.
- `method='higher'` for the quantile is conservative — it can make `q_hat` slightly larger (and thus prediction sets slightly larger/wider) than the "exact" interpolated quantile would.
- With small calibration sets, `q_level` can saturate at `1.0` (as shown in the worked example), making `q_hat` equal to the single largest calibration score — sensitive to outliers.
- The empty-set fallback (forcing in the `argmin` class) means the *actual* coverage can be slightly different from the nominal `1-alpha` for those specific pixels, since they're guaranteed a class regardless of whether its score was ≤ `q_hat`.

---

### Method: Full-Scene Inference

#### a) What it is

> While calibration and evaluation only use labeled pixels (a subset of the image), this step runs the model over **every single pixel** in the 330×307 scene — including unlabeled ones — to produce a complete probability map and, eventually, a complete classification map of the whole image.

#### b) Why it's used here

The notebook produces visualizations (certain/uncertain maps, class maps) over the *entire* image, not just the labeled pixels used for calibration/evaluation. To do that, every pixel needs a `9×9×6` patch extracted around it (including pixels near the border, handled via edge-padding) and a probability vector computed by the model.

#### c) How it works — Step by step

1. Pad the normalized image by `patch_size // 2 = 4` pixels on each side using edge-padding (`mode='edge'`), so a full `9×9` patch can be extracted even for pixels at the image boundary.
2. Infer `num_classes` by running the model on a single test patch (the top-left `9×9×6` patch) and checking the output width.
3. Allocate an output array `prob_full` of shape `(H, W, num_classes)`.
4. Process the image **one column at a time** (looping `col` from `0` to `W-1`):
   - For every row in that column, extract the `9×9×6` patch centered (in the padded image) on `(row, col)`.
   - Stack all `H` patches for this column into a batch of shape `(H, 9, 9, 6)`.
   - Run `predict_probs` on this batch to get `(H, num_classes)` probabilities for the whole column at once.
   - Store the result in `prob_full[:, col, :]`.
5. Print a progress message every 50 columns (and on the final column).
6. Assert the final shape is `(H, W, num_classes)` as expected.

#### d) ASCII Flow Diagram

```
Normalized image (H, W, B)
        |
        v
Edge-pad by (patch_size // 2) on each side --> (H+2*pad, W+2*pad, B)
        |
        v
Infer num_classes from a single 9x9x6 patch
        |
        v
prob_full = zeros(H, W, num_classes)
        |
        v
For col in 0..W-1:
    |
    v
  For row in 0..H-1: extract 9x9x6 patch -> patchs[row]
    |
    v
  predict_probs(model, patchs)  --> (H, num_classes)
    |
    v
  prob_full[:, col, :] = result
    |
    v
  (every 50 cols) print progress
        |
        v
Final prob_full (H, W, num_classes)
```

#### e) Worked Numerical Example

Suppose (for a tiny toy scene) `H=2`, `W=1`, `patch_size=3`, `num_classes=2`, `B=1` band. The padded image (pad=1) might look like (showing just the single band, a 4×3 grid after padding a 2×1 image):

```
Original (H=2, W=1):        Padded (H+2=4, W+2=3), edge-replicated:
[1]                          [1 1 1]
[2]                          [1 1 1]
                              [2 2 2]
                              [2 2 2]
```

For `col=0` (the only column), and `patch_size=3`:
- `row=0` patch is `x_pad[0:3, 0:3, :]` → the top 3×3 block → `[[1,1,1],[1,1,1],[2,2,2]]`.
- `row=1` patch is `x_pad[1:4, 0:3, :]` → the bottom 3×3 block → `[[1,1,1],[2,2,2],[2,2,2]]`.

These two `3×3×1` patches are stacked into `patchs` of shape `(2, 3, 3, 1)` and passed through `predict_probs` in one batch call, returning `(2, 2)` — two probability vectors (one per row), which are written into `prob_full[:, 0, :]`. Repeating for every column eventually fills the entire `(H, W, num_classes)` array.

#### f) Code Walkthrough

```python
def predict_full_scene_probs(model, x_img, H, W, B, patch_size, batch_size=128):
    """
    Generate per-pixel softmax probabilities for the entire scene.

    Processes one column of pixels at a time to keep memory bounded.
    Logs progress every 50 columns.
    """
    pad   = patch_size // 2
    x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')  # replicate border pixels

    # Infer number of classes from a single test patch
    test_patch = x_pad[0:patch_size, 0:patch_size, :][None, ...]   # add batch dim -> (1, ps, ps, B)
    n_classes  = predict_probs(model, test_patch, batch_size=1).shape[1]

    prob_full = np.zeros((H, W, n_classes), dtype=np.float32)
    for col in range(W):
        patchs = np.zeros((H, patch_size, patch_size, B), dtype=np.float32)
        for row in range(H):
            patchs[row] = x_pad[row:row + patch_size, col:col + patch_size, :]  # patch centered at (row, col)
        prob_col          = predict_probs(model, patchs, batch_size=batch_size)  # (H, n_classes)
        prob_full[:, col, :] = prob_col
        if (col + 1) % 50 == 0 or (col + 1) == W:
            print(f'  full-scene progress: {col + 1}/{W}')

    assert prob_full.shape == (H, W, n_classes)
    return prob_full
```

#### g) Output & Interpretation

The output `prob_full` is an `(H, W, num_classes)` array: a full probability distribution for *every* pixel in the scene, regardless of whether it had a ground-truth label. This is the input to the full-scene APS scoring and spatial smoothing used to generate the "Certain vs Uncertain Map" and "Class Map with Uncertain Mask" visualizations — these maps show what SACP would predict (and how confident it is) across the entire image, not just the sampled evaluation pixels.

#### h) Limitations

- Column-by-column processing means `H` patches are extracted and stacked per iteration — for a 330×307 image this is 307 batches of up to 330 patches each; runtime scales with `H * W`.
- The function re-extracts patches with simple slicing in a nested Python loop (`for row in range(H)`), which is straightforward but not the most memory/compute-efficient approach for very large images.
- `num_classes` is inferred from the model's output on a single patch — if the model's output shape were somehow input-dependent (not the case for standard classifiers, but worth noting), this inference could be misleading.

---

### Method: Per-Model SACP Pipeline (Orchestration)

#### a) What it is

> This is the "project manager" function — for one model, it runs every other piece (inference, calibration, full-scene prediction, plotting, metric collection) in the right order and bundles all the results into one tidy package.

#### b) Why it's used here

Since the notebook needs to repeat the *entire* SACP analysis for 3 models × 4 window sizes = 12 total runs, this function encapsulates one (model, window_size) run so the outer loop (described next) can simply call it 12 times and collect the results.

#### c) How it works — Step by step

1. Start a timer (`t0`).
2. Run `predict_probs` on the calibration and evaluation patch sets to get `calib_probs` and `eval_probs`.
3. Instantiate a fresh `SpatialConformalPredictor` with the given `alpha`, `lambda_`, `k`, and `window_size` (and the fixed image dimensions `H`, `W`, `num_classes`, `seed`).
4. Call `sacp.fit_calibrate(...)` to get `pred_sets_eval`, `q_hat`, and `avg_size` (as described in the SACP Calibration method above).
5. Compute aggregate metrics (`compute_set_metrics`) and per-class coverage (`per_class_coverage_df`) on `pred_sets_eval` vs. `y_eval`.
6. Run `predict_full_scene_probs` to get probabilities for every pixel in the image.
7. Flatten `prob_full` to `(H*W, num_classes)`, compute APS scores for every pixel (`flat_scores`), and reshape back to `(H, W, num_classes)` as `current_map`.
8. If `k > 0`, apply `spatial_smoothing` to the *entire* image (all pixels masked `True`) `k` times — this mirrors the calibration step's smoothing but now over the full scene.
9. Threshold the smoothed full-scene score map by `q_hat` to get `pred_sets_full` (boolean, `(H, W, num_classes)`).
10. Compute `set_sizes_map` (how many classes are in each pixel's set) and `pred_class_map` (the model's single most-likely class per pixel, via `argmax`).
11. Build `combined_map`: where a pixel's set size is exactly 1 (certain), show its predicted class; otherwise (uncertain), mark it with a special "uncertain" class index (`num_classes`).
12. Build `pixel_counts_df` (counts of pixels per class plus "uncertain").
13. Generate four plots (as PNG buffers): per-class coverage bar chart, certain/uncertain binary map, masked class map, and pixel-count bar chart.
14. Stop the timer and assemble a `summary` dict with all the key metrics (model name, method, window size, target/empirical coverage, set-size stats, runtime, hyperparameters, `q_hat`, mean per-class coverage).
15. Assemble `tables` (a dict of small DataFrames for Excel export: Summary, Per-Class Coverage Values, Pixel Counts, SACP Parameters).
16. Return a dict with `model_name`, `summary`, `per_class_df`, `plot_buffers`, and `tables`.

#### d) ASCII Flow Diagram

```
model, x_cal, x_eval, x_img, alpha, lambda, k, window_size
        |
        v
 predict_probs(calib), predict_probs(eval)
        |
        v
 SpatialConformalPredictor(...).fit_calibrate(...)
        |                         \
        v                          \--> q_hat, avg_size
 pred_sets_eval
        |
        v
 compute_set_metrics + per_class_coverage_df
        |
        v
 predict_full_scene_probs(model, x_img)  --> prob_full (H,W,C)
        |
        v
 compute_aps_scores on flattened prob_full --> current_map (H,W,C)
        |
        v
 spatial_smoothing x k over WHOLE image
        |
        v
 pred_sets_full = current_map <= q_hat
        |
        v
 set_sizes_map, pred_class_map, combined_map, pixel_counts_df
        |
        v
 4 plots (coverage bar, certain/uncertain map, class map, pixel counts)
        |
        v
 summary dict + tables dict
        |
        v
 return { model_name, summary, per_class_df, plot_buffers, tables }
```

#### e) Worked Numerical Example

Suppose for `model_name='AlexNet'`, `window_size=3`, after `fit_calibrate` we get `q_hat = 0.62`, and on the full-scene `current_map`, three example pixels have smoothed score vectors (2 classes):

- Pixel A: `[0.30, 0.85]` → `set_sizes_map = sum([0.30<=0.62, 0.85<=0.62]) = sum([True, False]) = 1` → certain. `pred_class_map = argmax(prob_full[A]) = 0` (say). `combined_map[A] = 0`.
- Pixel B: `[0.55, 0.58]` → `set_sizes_map = sum([True, True]) = 2` → uncertain. `combined_map[B] = num_classes` (e.g., `2`, the "uncertain" index).
- Pixel C: `[0.70, 0.90]` → `set_sizes_map = sum([False, False]) = 0`. (Note: for the *full-scene* map, there's no empty-set fallback applied like in `fit_calibrate` — so `set_sizes_map` can legitimately be `0` here.) `combined_map[C] = num_classes` (since `0 != 1`, it falls into the "not exactly certain" / uncertain bucket too).

If the image had only these 3 pixels and `num_classes=2`, `pixel_counts_df` would show: `Class 0: 1`, `Class 1: 0`, `Uncertain: 2`.

The `summary` dict for this run might look like:
```
{
  'model_name': 'AlexNet', 'method': 'SACP', 'window_size': 3,
  'target_coverage': 0.95, 'empirical_coverage': 0.94,
  'avg_set_size': 1.3, 'median_set_size': 1.0,
  'singleton_rate': 0.75, 'empty_set_rate': 0.0,
  'runtime_sec': 42.7, 'alpha': 0.05, 'lambda': 0.5, 'k': 1,
  'q_hat': 0.62, 'mean_per_class_coverage': 0.93
}
```

#### f) Code Walkthrough

```python
def build_sacp_outputs_for_model(
    model_name, model,
    x_cal, y_cal, coords_cal,
    x_eval, y_eval, coords_eval,
    x_img, alpha, lambda_, k, window_size=3, batch_size=128,
):
    t0 = time.perf_counter()                                  # start timing this model's run

    # 1. Inference on calibration & evaluation splits
    calib_probs = predict_probs(model, x_cal,  batch_size=batch_size)
    eval_probs  = predict_probs(model, x_eval, batch_size=batch_size)

    # 2. SACP calibration
    sacp = SpatialConformalPredictor(
        height=H, width=W, num_classes=num_classes,
        lambda_=lambda_, alpha=alpha, k=k, window_size=window_size, seed=SEED,
    )
    pred_sets_eval, q_hat, avg_size = sacp.fit_calibrate(
        calib_probs=calib_probs, calib_labels=y_cal, calib_indices=coords_cal,
        test_probs=eval_probs,   test_indices=coords_eval,
    )

    metrics  = compute_set_metrics(pred_sets_eval, y_eval)     # coverage, set sizes, singleton/empty rates
    per_cls  = per_class_coverage_df(pred_sets_eval, y_eval, num_classes)

    # 3. Full-scene visualisation
    print(f'Generating full-scene probabilities for {model_name} ...')
    prob_full   = predict_full_scene_probs(
        model, x_img, H, W, B, PATCH_SIZE, batch_size=batch_size
    )

    flat_probs  = prob_full.reshape(-1, num_classes)           # (H*W, C)
    flat_scores = sacp.compute_aps_scores(flat_probs)          # APS scores for every pixel
    current_map = flat_scores.reshape(H, W, num_classes)       # back to spatial grid

    if k > 0:
        mask_map_full = np.ones((H, W), dtype=bool)            # smooth over the ENTIRE image
        iterator = (
            tqdm(range(k), desc=f'Smoothing full map ({model_name})')
            if k > 1 else range(k)
        )
        for _ in iterator:
            current_map = sacp.spatial_smoothing(current_map, mask_map_full)

    pred_sets_full  = (current_map <= q_hat)                   # threshold using the SAME q_hat from calibration
    set_sizes_map   = np.sum(pred_sets_full, axis=2)
    pred_class_map  = np.argmax(prob_full, axis=2)             # model's single best guess per pixel
    combined_map    = np.where(set_sizes_map == 1, pred_class_map, num_classes)  # certain -> class; else -> "uncertain"
    pixel_counts_df = build_pixel_counts_df(combined_map, num_classes)

    # 4. Plot buffers
    plot_buffers = {
        'Per-Class Coverage': make_per_class_coverage_plot(per_cls, alpha=alpha, title='SACP: Per-Class Coverage (Full Image)'),
        'Certain vs Uncertain Map': make_certain_uncertain_map_plot(set_sizes_map, title=f'Predictions with 95% Uncertainty Map\n(SACP — {model_name})'),
        'Class Map with Uncertain Mask': make_masked_class_map_plot(combined_map, n_classes=num_classes, title=f'Predictions with 95% Uncertainty Mask\n(SACP — {model_name})'),
        'Pixel Counts': make_pixel_count_plot(pixel_counts_df, title='Pixel Count per Class (Including Uncertain Regions)', n_classes=num_classes),
    }

    runtime = time.perf_counter() - t0
    summary = {
        'model_name': model_name, 'method': 'SACP', 'window_size': int(window_size),
        'target_coverage': float(1.0 - alpha), 'empirical_coverage': metrics['empirical_coverage'],
        'avg_set_size': metrics['avg_set_size'], 'median_set_size': metrics['median_set_size'],
        'singleton_rate': metrics['singleton_rate'], 'empty_set_rate': metrics['empty_set_rate'],
        'runtime_sec': float(runtime), 'alpha': float(alpha), 'lambda': float(lambda_), 'k': int(k),
        'q_hat': float(q_hat), 'mean_per_class_coverage': float(per_cls['class_coverage'].mean(skipna=True)),
    }
    tables = {
        'Summary': pd.DataFrame([summary]),
        'Per-Class Coverage Values': per_cls,
        'Pixel Counts': pixel_counts_df,
        'SACP Parameters': pd.DataFrame([{'q_hat': float(q_hat), 'alpha': float(alpha), 'lambda': float(lambda_), 'k': int(k), 'avg_set_size_eval': float(avg_size)}]),
    }
    return {'model_name': model_name, 'summary': summary, 'per_class_df': per_cls, 'plot_buffers': plot_buffers, 'tables': tables}
```

#### g) Output & Interpretation

The return value is a dictionary holding everything needed for one model's report sheet: a one-row `summary` (used for the cross-model comparison tables), a `per_class_df` (per-class coverage), four ready-to-embed plot images, and `tables` for the Excel sheet. The `summary['empirical_coverage']` vs. `summary['target_coverage']` comparison is the headline result — values close together indicate the conformal calibration is working as intended. `summary['avg_set_size']` and `singleton_rate` indicate how "decisive" the combined model+SACP system is.

#### h) Limitations

- Every (model, window_size) combination re-runs the full, expensive `predict_full_scene_probs` over ~101K pixels — this is the dominant cost and is repeated 12 times total (3 models × 4 window sizes), even though `x_img` and the model itself don't change across window sizes (only the SACP smoothing/threshold parameters do).
- The full-scene smoothing pass treats *every* pixel as masked (`mask_map_full = np.ones(...)`), including unlabeled pixels — these pixels' APS scores still influence each other's smoothed values even though they were never part of calibration.
- `q_hat`, computed from the labeled calibration/evaluation pixels only, is then applied to the *full-scene* smoothed map — the full-scene smoothing dynamics (every pixel has up to `|neighbors|` valid neighbors, vs. the sparser labeled-pixel-only mask used during `fit_calibrate`) differ from the conditions under which `q_hat` was calibrated, which is a structural approximation.

---

### Method: Window-Size Sweep & Excel Reporting

#### a) What it is

> This is the "outer loop and filing cabinet" — for each of the four window sizes (3, 5, 7, 9), it runs all three models through the pipeline above, then organizes everything (CSVs, an Excel workbook with one sheet per model plus comparison sheets) into a dedicated folder for that window size. Afterward, a final step combines all four window sizes' results into one master summary and comparison plot.

#### b) Why it's used here

The choice of spatial smoothing `window_size` is itself a hyperparameter whose effect on coverage and set size is unknown ahead of time. Sweeping over `[3, 5, 7, 9]` and recording results for all three models at each size lets the user empirically compare how window size trades off prediction-set size against coverage and per-class behavior, across model architectures.

#### c) How it works — Step by step

1. Initialize two empty lists, `all_windows_summaries` and `all_windows_per_class`, before the loop (these accumulate results across all window sizes for the final combined report).
2. **Outer loop** over `ws` in `SACP_WINDOW_SIZES = [3, 5, 7, 9]`:
   - Create a per-window output directory `results/window_<ws>/`.
   - **Inner loop** over the three models (`AlexNet_CNN`, `GFNet`, `ViT_UNet`): call `build_sacp_outputs_for_model(..., window_size=ws, ...)` for each, collecting outputs in `all_outputs`.
   - Build `summary_df` (one row per model, sorted by `model_name`) and `per_class_df` (concatenated per-class coverage for all 3 models, tagged with `model_name` and `window_size`).
   - Save `summary_df` and `per_class_df` as CSVs in the window's output directory.
   - Build an Excel workbook (`conformal_reports_SACP_ws<ws>_all_models.xlsx`) containing:
     - One sheet per model (via `write_model_sheet`, described below) named `SACP_<model_name>`.
     - A `Summary_Compact` sheet (the `summary_df`).
     - A `Run_Config` sheet (a one-row DataFrame recording `window_size`, `alpha`, `lambda`, `k`).
     - A `Compare_SACP` sheet (a subset of `summary_df` columns useful for cross-model comparison), with auto-widened columns.
   - Append `summary_df` and `per_class_df` to the running `all_windows_summaries` / `all_windows_per_class` lists.
3. **After the outer loop**: concatenate all four windows' summaries into `combined_summary_df` (12 rows: 4 windows × 3 models) and all per-class data into `combined_per_class_df`, save both as combined CSVs.
4. Generate a 3-panel cross-window comparison plot: empirical coverage vs. window size (with a red dashed line at the target `1-alpha`), average set size vs. window size, and mean per-class coverage vs. window size — each with one line per model.
5. **Final validation** (Section 10.0): for each window size, re-open the saved Excel workbook and assert that all expected sheet name prefixes exist (`Summary_Compact`, `Run_Config`, `SACP_AlexNet`, `SACP_GFNet`, `SACP_ViT`, `Compare_SACP`); assert the combined summary has exactly `12` rows; assert all coverage values are in `[0,1]`; assert all 4 window sizes are represented; and assert that per-class coverage isn't entirely `NaN` for any window size.

**The `write_model_sheet` helper** (Excel export detail):
- Writes the model name as a header.
- Iterates through each table in `output['tables']` (Summary, Per-Class Coverage Values, Pixel Counts, SACP Parameters), writing a label and the DataFrame, stacking them vertically with spacing.
- Iterates through each plot in `output['plot_buffers']`, writing a label and embedding the PNG image to the right of the tables (`img_col=9`), stacking them vertically with spacing (`img_row += 24`).

**Sheet name helpers** (`sanitize_sheet_name`, `make_sheet_name`) strip Excel-forbidden characters (`\ / * ? : [ ]`), truncate to Excel's 31-character sheet-name limit, and ensure uniqueness by appending `_1`, `_2`, etc. if a name is already used.

#### d) ASCII Flow Diagram

```
all_windows_summaries = [], all_windows_per_class = []

For ws in [3, 5, 7, 9]:
        |
        v
  Create results/window_<ws>/
        |
        v
  For model in {AlexNet, GFNet, ViT}:
        |
        v
    build_sacp_outputs_for_model(..., window_size=ws)
        |
        v
    all_outputs.append(out)
        |
        v
  summary_df, per_class_df  (3 rows / model-rows)
        |
        v
  Save CSVs (summary_ws<ws>.csv, per_class_ws<ws>.csv)
        |
        v
  Build Excel workbook:
    - SACP_<model> sheets (tables + plots)
    - Summary_Compact
    - Run_Config
    - Compare_SACP
        |
        v
  all_windows_summaries.append(summary_df)
  all_windows_per_class.append(per_class_df)

(after loop)
        |
        v
combined_summary_df  = concat(all_windows_summaries)   # 12 rows
combined_per_class_df = concat(all_windows_per_class)
        |
        v
Save combined CSVs
        |
        v
3-panel plot: coverage / avg_set_size / mean_per_class_coverage vs window_size
        |
        v
Final validation: assert sheets exist, 12 rows, coverage in [0,1], all 4 window sizes present
```

#### e) Worked Numerical Example

Suppose after running all 12 (model, window_size) combinations, `combined_summary_df` has these (illustrative) rows for `empirical_coverage` (target = `0.95`):

| model_name | window_size | empirical_coverage | avg_set_size |
|---|---|---|---|
| AlexNet | 3 | 0.94 | 1.20 |
| AlexNet | 5 | 0.95 | 1.35 |
| AlexNet | 7 | 0.96 | 1.50 |
| AlexNet | 9 | 0.97 | 1.70 |
| GFNet   | 3 | 0.93 | 1.15 |
| ... | ... | ... | ... |

**Final validation checks**, walked through:
1. `expected_rows = len([3,5,7,9]) * len({AlexNet,GFNet,ViT}) = 4 * 3 = 12`. If `combined_summary_df` has 12 rows, this passes.
2. `(combined_summary_df['empirical_coverage'] >= 0) & (... <= 1)` → for the example rows above (`0.94, 0.95, 0.96, 0.97, 0.93, ...`), all are within `[0,1]`, so `.all()` is `True`.
3. `set(combined_summary_df['window_size'].unique()) == {3,5,7,9}` → if every window size appears at least once across the 12 rows, this passes.
4. For each window size's `per_class_df`, `class_coverage.notna().any()` checks that at least one class has a non-`NaN` coverage value (a class could be `NaN` if it has zero support pixels in the evaluation set, per `per_class_coverage_df`'s `if support > 0 else np.nan` logic).

If, hypothetically, `combined_summary_df` had only 11 rows (e.g., one model's run for `window_size=9` failed silently and `all_outputs` was short one entry), the `expected_rows` assertion would raise: `AssertionError: Expected 12 rows in combined summary, got 11`.

#### f) Code Walkthrough

```python
# ── Accumulators for the cross-window summary (used in Cell 9.0) ─────────────
all_windows_summaries  = []   # collects one summary_df per window size
all_windows_per_class  = []   # collects one per_class_df per window size

# ── Outer sweep over every window size ──────────────────────────────────────
for ws in SACP_WINDOW_SIZES:
    print(f"\n{'#'*60}")
    print(f"  WINDOW SIZE = {ws}")
    print(f"{'#'*60}")

    ws_dir = OUTPUT_DIR / f'window_{ws}'         # one folder per window size
    ws_dir.mkdir(parents=True, exist_ok=True)

    all_outputs = []

    for model_key, model in models.items():
        model_name = MODEL_NAME_MAP.get(model_key, model_key)   # e.g. 'ViT_UNet' -> 'ViT'
        print(f"\n{'='*20} Running SACP for {model_name} (ws={ws}) {'='*20}")

        out = build_sacp_outputs_for_model(
            model_name=model_name, model=model,
            x_cal=x_cal, y_cal=y_cal, coords_cal=coords_cal,
            x_eval=x_eval, y_eval=y_eval, coords_eval=coords_eval,
            x_img=x_img, alpha=SACP_ALPHA, lambda_=SACP_LAMBDA, k=SACP_K,
            window_size=ws,                       # use the loop variable for this run's window size
            batch_size=BATCH_SIZE,
        )
        all_outputs.append(out)

    # Per-window summary & per-class DataFrames
    summary_df = (
        pd.DataFrame([o['summary'] for o in all_outputs])
        .sort_values('model_name')
        .reset_index(drop=True)
    )
    per_class_df = pd.concat(
        [o['per_class_df'].assign(model_name=o['model_name'], window_size=ws)
         for o in all_outputs],
        ignore_index=True,
    )

    summary_df.to_csv(ws_dir / f'summary_ws{ws}.csv', index=False)
    per_class_df.to_csv(ws_dir / f'per_class_ws{ws}.csv', index=False)

    # Excel workbook with per-model sheets + comparison sheets
    excel_path = ws_dir / f'conformal_reports_SACP_ws{ws}_all_models.xlsx'
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        used_names = set()

        for out in all_outputs:
            sname = make_sheet_name(f"SACP_{out['model_name']}", used_names)
            write_model_sheet(writer, workbook, out, sname)

        summary_df.to_excel(writer, sheet_name=make_sheet_name('Summary_Compact', used_names), index=False)

        pd.DataFrame([{'window_size': ws, 'alpha': SACP_ALPHA, 'lambda': SACP_LAMBDA, 'k': SACP_K}]).to_excel(
            writer, sheet_name=make_sheet_name('Run_Config', used_names), index=False
        )

        compare_cols = ['model_name', 'window_size', 'empirical_coverage', 'avg_set_size',
                         'median_set_size', 'singleton_rate', 'empty_set_rate',
                         'mean_per_class_coverage', 'q_hat', 'runtime_sec']
        compare_df = summary_df[[c for c in compare_cols if c in summary_df.columns]]
        compare_sheet = make_sheet_name('Compare_SACP', used_names)
        compare_df.to_excel(writer, sheet_name=compare_sheet, index=False)

        ws_xl = writer.sheets[compare_sheet]
        for i, col in enumerate(compare_df.columns):
            width = max(len(str(col)), compare_df[col].astype(str).str.len().max()) + 2
            ws_xl.set_column(i, i, width)
    print(f'Saved workbook: {excel_path}')

    all_windows_summaries.append(summary_df)
    all_windows_per_class.append(per_class_df)

print("\n✓ All window sizes complete.")
```

#### g) Output & Interpretation

On disk, the output is a directory tree:

```
results/
├── window_3/
│   ├── summary_ws3.csv
│   ├── per_class_ws3.csv
│   └── conformal_reports_SACP_ws3_all_models.xlsx
├── window_5/  (same structure)
├── window_7/  (same structure)
├── window_9/  (same structure)
├── combined_summary_all_windows.csv     (12 rows)
└── combined_per_class_all_windows.csv
```

Each Excel workbook lets a reader inspect, per model, the summary metrics, per-class coverage table, pixel counts, SACP parameters, and four diagnostic plots, plus compact cross-model comparison sheets. The final 3-panel plot (Section 9.0) is the headline visual: it shows, for each model, how empirical coverage tracks the `1-alpha` target line and how average/per-class set sizes change as the spatial smoothing window grows — larger windows generally pull in more neighbor information, which can either tighten or loosen prediction sets depending on how spatially homogeneous the true class layout is.

#### h) Limitations

- The combined summary's row count (`12`) is hard-coded as `len(SACP_WINDOW_SIZES) * len(MODEL_FILES)`; if either list changes, downstream code (including the final validation assertions) must be kept in sync.
- Re-running `predict_full_scene_probs` for the same model across all four window sizes is redundant (the model's predictions don't depend on `window_size` — only the SACP smoothing/threshold does); this could be cached/computed once per model and reused.
- The Excel `Compare_SACP` sheet's column-width calculation (`compare_df[col].astype(str).str.len().max()`) converts every value to a string first, which could produce unexpectedly wide columns for floating-point numbers with long decimal representations.
- The final validation in Section 10.0 checks *structural* correctness (sheet names exist, row counts match, values in valid ranges) but does not check *numerical correctness* of the underlying conformal guarantees (e.g., it doesn't independently verify that `empirical_coverage` is statistically consistent with `1 - alpha` within expected sampling variance).

---

## Results & Comparisons

The notebook itself does not contain hard-coded numeric results — all metrics are computed at runtime from the user's actual multispectral data and the three loaded models (`AlexNet_CNN`, `GFNet`, `ViT_UNet`), then written to `combined_summary_all_windows.csv` and `combined_per_class_all_windows.csv`. The **shape** of the final results table is fixed by the code, however:

- `combined_summary_df` has exactly **12 rows** (4 window sizes × 3 models), with columns: `model_name`, `method` (always `'SACP'`), `window_size`, `target_coverage` (always `1 - SACP_ALPHA = 0.95`), `empirical_coverage`, `avg_set_size`, `median_set_size`, `singleton_rate`, `empty_set_rate`, `runtime_sec`, `alpha`, `lambda`, `k`, `q_hat`, and `mean_per_class_coverage`.
- `combined_per_class_df` has one row per `(model_name, window_size, class_id)` combination, with `class_coverage` and `support_count` columns.

A comparison table in the format requested by the explainer skill, populated with **placeholder structure** (actual numbers depend on the run):

| Method | Window Size | Empirical Coverage | Avg Set Size | Singleton Rate | Notes |
|---|---|---|---|---|---|
| SACP — AlexNet | 3 | *(from CSV)* | *(from CSV)* | *(from CSV)* | Smallest spatial context |
| SACP — AlexNet | 5 | *(from CSV)* | *(from CSV)* | *(from CSV)* | |
| SACP — AlexNet | 7 | *(from CSV)* | *(from CSV)* | *(from CSV)* | |
| SACP — AlexNet | 9 | *(from CSV)* | *(from CSV)* | *(from CSV)* | Largest spatial context |
| SACP — GFNet | 3–9 | *(from CSV)* | *(from CSV)* | *(from CSV)* | Frequency-domain global filter backbone |
| SACP — ViT | 3–9 | *(from CSV)* | *(from CSV)* | *(from CSV)* | Transformer (CLS-token) backbone |

> **Results not shown in provided notebook.** The notebook produces these values at execution time by running inference on the user's models and data; this document describes the *computation* that produces them rather than specific numeric outcomes. The cross-window plot (Section 9.0) visualizes, for all three models simultaneously: (1) empirical coverage vs. window size with a red dashed reference line at the target `1 - alpha = 0.95`, (2) average prediction-set size vs. window size, and (3) mean per-class coverage vs. window size.

---

## Academic Paper Summary

**Problem Statement.** Pixel-wise classification of multispectral remote-sensing imagery with deep neural networks provides point predictions but no calibrated measure of predictive uncertainty. This work addresses the need for statistically valid, spatially-aware uncertainty quantification by applying a Spatial Adaptive Conformal Prediction (SACP) framework to three deep architectures — a convolutional network (AlexNet-style CNN), a frequency-domain global-filter network (GFNet), and a vision-transformer-based network (ViT-UNet) — evaluated on a six-band multispectral scene.

**Methodology.** Each trained classifier produces per-pixel class probability distributions over `9×9×6` input patches. Conformity scores are computed using the randomised Adaptive Prediction Sets (APS) procedure, which ranks classes by predicted probability and accumulates probability mass up to (and including a randomised fraction of) the class of interest. To exploit spatial autocorrelation inherent to image data, conformity scores are propagated across a local pixel neighborhood via an iterative spatial-smoothing operator parameterized by a window size and a mixing coefficient `lambda`. A calibration threshold `q_hat` is selected as the `ceil((n+1)(1-alpha))/n`-quantile (with conservative upper-rounding) of the smoothed true-class conformity scores on a held-out calibration set, following standard split-conformal methodology adapted to the spatial setting. Prediction sets for evaluation pixels and for the full scene are then formed by including every class whose smoothed conformity score does not exceed `q_hat`, with a fallback rule guaranteeing non-empty sets.

**Experimental Setup.** The dataset is a single `330 × 307` pixel, 6-band multispectral scene with per-pixel ground-truth class labels, min-max normalized per band. Labeled pixels are split 75/25 into a training pool (unused here, as models are pre-trained) and a test pool, which is further split 50/50 into calibration and evaluation sets via stratified sampling. The target miscoverage rate is `alpha = 0.05` (95% target coverage), the spatial mixing coefficient is `lambda = 0.5`, the number of smoothing iterations is `k = 1`, and the spatial window size is swept over `{3, 5, 7, 9}` to study its effect on coverage and set size. Evaluation metrics include empirical marginal coverage, average and median prediction-set size, singleton/empty-set rates, and mean per-class coverage, each reported per (model, window size) combination, yielding 12 total experimental conditions.

**Results Summary.** *(Results are generated at runtime from the user's data and are not hard-coded in the notebook; this section should be completed with the actual values from `combined_summary_all_windows.csv` once the notebook is executed.)* The expected analysis compares, across the three architectures and four window sizes, whether empirical coverage tracks the nominal 95% target, how average prediction-set size trades off against window size (larger spatial context potentially yielding either tighter or looser sets depending on class-map homogeneity), and which architecture achieves the best balance of high coverage and small/singleton-dominated prediction sets.

**Conclusion.** This work demonstrates an end-to-end pipeline for applying spatially-aware conformal prediction to multispectral image classifiers, producing per-pixel prediction sets with a target coverage guarantee alongside interpretable visualizations (certainty maps, masked class maps, per-class coverage). Limitations include the approximate nature of the coverage guarantee under spatial smoothing (which violates strict exchangeability assumptions of standard conformal theory), the computational cost of full-scene inference repeated across window-size sweeps, and the sensitivity of the calibrated threshold to small calibration-set sizes. Future directions include theoretical analysis of coverage guarantees under spatial dependence, more efficient full-scene inference (e.g., caching predictions across the window-size sweep), and extension to additional conformity score functions beyond APS.

---

## References

[1] Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. (Foundational text for conformal prediction.)

[2] Romano, Y., Sesia, M., & Candès, E. J. (2020). Classification with Valid and Adaptive Coverage. *Advances in Neural Information Processing Systems (NeurIPS)*. (Introduces the Adaptive Prediction Sets, APS, conformity score used in this notebook.)

[3] Angelopoulos, A. N., & Bates, S. (2023). Conformal Prediction: A Gentle Introduction. *Foundations and Trends in Machine Learning*. (General reference for split-conformal calibration, including the `ceil((n+1)(1-alpha))/n` quantile correction used here.)

[4] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*. (Origin of the AlexNet architecture referenced by the `AlexNet_CNN` model.)

[5] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global Filter Networks for Image Classification. *Advances in Neural Information Processing Systems (NeurIPS)*. (Origin of the GFNet architecture and its FFT-based `GlobalFilterLayer`.)

[6] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR)*. (Origin of the Vision Transformer patch-embedding and CLS-token design used in `PatchEncoderWithCLS` and `ViT_UNet`.)

[7] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*. (Relevant to the U-Net component implied by the `ViT_UNet` model name.)

[8] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*. (Source of `train_test_split` used for stratified calibration/evaluation splitting.)
