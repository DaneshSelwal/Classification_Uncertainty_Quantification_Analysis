# Multi-Head Hyperspectral Image Classification: AlexNet, GFNet, and Vision Transformer

## 1. Title & Overview

This notebook trains and evaluates three deep learning architectures â€” a multi-head
AlexNet-style CNN, a multi-head GFNet (Global Filter Network), and a multi-head Vision
Transformer with a U-Net-style skip connection structure â€” on a hyperspectral (multi-band)
image classification task. The image is split into small spatial patches centered on each
labeled pixel, and each model is trained to assign the center pixel to one of several land-cover
classes. All three models share an unusual training trick: instead of a single softmax output,
each model produces **7 identical output heads**, and instead of standard random dropout, the
notebook implements a **staged "channel-shift" dropout** that progressively dedicates different
slices of channels to being dropped across training "shifts," eventually settling into a
standard dropout phase. A custom Keras callback (`Custom_callbacks`) drives this staged process,
rebuilding the model architecture mid-training when validation accuracy thresholds are hit.
After training, each model is evaluated (accuracy, Cohen's Kappa, confusion matrix,
classification report), results and figures are exported to an Excel workbook, and a model
registry JSON is written for later reloading. Finally, a smoke check reloads every saved model
and confirms it still produces 7 output heads.

**Who this document is for:** a reader who used AI assistance to write this notebook and now
wants to deeply understand *how and why* each piece works â€” both to build intuition for future
modification, and to have material that can be adapted into the methodology section of a paper.

---

## 2. Table of Contents

- [1. Title & Overview](#1-title--overview)
- [2. Table of Contents](#2-table-of-contents)
- [3. Environment & Dependencies](#3-environment--dependencies)
- [4. Data & Problem Setup](#4-data--problem-setup)
- [5. Methods](#5-methods)
  - [5.1 Pearson Correlation Masked Attention](#51-pearson-correlation-masked-attention)
  - [5.2 Staged Channel-Shift Dropout (Dropout_Train)](#52-staged-channel-shift-dropout-dropout_train)
  - [5.3 Model Modifier & Custom Staged-Training Callback](#53-model-modifier--custom-staged-training-callback)
  - [5.4 Multi-Head AlexNet CNN](#54-multi-head-alexnet-cnn)
  - [5.5 GFNet: Global Filter Network](#55-gfnet-global-filter-network)
  - [5.6 Vision Transformer with U-Net-Style Skip Connections](#56-vision-transformer-with-u-net-style-skip-connections)
  - [5.7 Cosine-Annealing Learning Rate Schedules](#57-cosine-annealing-learning-rate-schedules)
  - [5.8 Multi-Head Prediction Averaging](#58-multi-head-prediction-averaging)
  - [5.9 Performance Measures & Visualization](#59-performance-measures--visualization)
  - [5.10 Excel Export Pipeline](#510-excel-export-pipeline)
- [6. Results & Comparisons](#6-results--comparisons)
- [7. Academic Paper Summary](#7-academic-paper-summary)
- [8. References](#8-references)

---

## 3. Environment & Dependencies

| Library | Purpose |
|---|---|
| `os`, `sys`, `io`, `json`, `math`, `gc`, `random`, `shutil`, `time`, `pathlib.Path` | Standard library: filesystem paths, garbage collection, RNG seeding, JSON I/O, in-memory byte buffers, timing |
| `google.colab.drive` | Mounts Google Drive when running inside Colab, so the project folder and dataset are accessible |
| `numpy` | Array operations: image arrays, normalization, patch stacking, metric math |
| `pandas` | Reads CSV data/label files and builds summary/report tables |
| `seaborn` | Styling (`sns.set()`) and heatmaps for confusion matrices / classification reports |
| `matplotlib.pyplot`, `matplotlib.gridspec` | All plotting: class distribution bars, accuracy/loss curves, multi-panel performance figures, LR schedule plots |
| `openpyxl` (`Workbook`, `load_workbook`, `Image`, `dataframe_to_rows`) | Builds/updates the results `.xlsx` workbook, embeds matplotlib figures as images |
| `sklearn.model_selection.train_test_split` | Stratified train/test split of patches |
| `sklearn.metrics` (`confusion_matrix`, `accuracy_score`, `cohen_kappa_score`, `classification_report`) | Evaluation metrics for each trained model |
| `tensorflow` / `keras` | Core deep learning framework â€” model building, training, saving |
| `tensorflow_probability` | Imported but not directly used in the shown code (likely reserved for probabilistic extensions) |
| `tensorflow.keras.layers` (many: `Conv2D`, `Dense`, `Dropout`, `LayerNormalization`, `MultiHeadAttention`, etc.) | Building blocks for all three architectures |
| `tensorflow.python.util.tf_export.keras_export` | Decorator used (slightly unusually) to register the custom `Dropout_Train` layer under the Keras namespace |
| `tensorflow.python.ops.array_ops`, `tensorflow.python.keras.utils.control_flow_util` | Low-level TF ops used inside `Dropout_Train` for conditional (train vs. inference) execution |
| `tensorflow.keras.callbacks.ModelCheckpoint` | Saves the best model during training based on a monitored metric |
| `tensorflow.keras.models.load_model`, `Model` | Reloading saved models and the functional-API model class |
| `tensorflow.keras.utils.plot_model` | Imported for model architecture diagrams (not directly invoked in shown cells) |
| `keras.regularizers.l2` | Imported for potential L2 weight regularization (not directly invoked in shown cells) |

**Reproducibility:** `np.random.seed(1337)`, `random.seed(1337)`, and `tf.random.set_seed(1337)`
are all set immediately after imports, so re-running the notebook should give (mostly)
deterministic results â€” though GPU non-determinism and `shuffle=False` during `fit` also play a
role.

---

## 4. Data & Problem Setup

**Dataset:** A single hyperspectral image of shape `H=330 Ã— W=307 Ã— B=6` (height, width, 6
spectral bands), stored as `data.csv`, plus a pixel-wise label map of shape `330 Ã— 307` stored
as `ref.csv`. The label `0` denotes "background" (unlabeled) pixels and is excluded from
training.

**Problem type:** Multi-class pixel classification (land-cover / hyperspectral classification).
`num_classes` is computed as `len(np.unique(y)) - 1` (subtracting the background class).

**Preprocessing pipeline, exactly as performed in the notebook:**

1. **Load and reshape**: both CSVs are read with pandas, converted to numpy arrays, and reshaped
   to `(H, W, B)` for the image and `(H, W)` for the labels.
2. **Per-band min-max normalization**: for each of the 6 spectral bands independently,
   ```
   normalized_band = (band - min(band)) / max(max(band) - min(band), 1e-8)
   ```
   The `1e-8` epsilon guards against division by zero if a band were constant.
3. **Edge padding**: the normalized image is padded by `(P_S - 1) / 2 = 4` pixels on each side
   using edge-replication padding (`np.pad(..., 'edge')`), where `P_S = 9` is the spatial patch
   size.
4. **Patch extraction**: for every pixel `(row, col)` where `y[row, col] != 0`, a `9Ã—9Ã—6` patch
   centered on that pixel is extracted from the padded image and appended to `X`; the
   corresponding label (zero-indexed: `label - 1`) is appended to `Y`.
5. **Class balance visualization**: a bar chart shows the pixel count per class.
6. **Stratified train/test split**: `train_test_split` with `train_size = train_percent/100 =
   0.75`, `stratify=Y`, `random_state=10` â€” so 75% of patches go to training, 25% to testing,
   with class proportions preserved in both sets.

**Key global hyperparameters set in Section 1.2:**

```
P_S (patch size)        = 9
epoch                    = 100
BATCH_SIZE               = 128
dropout_rate             = 0.25
shifts (= 1/dropout_rate) = 4
train_percent            = 75
Targeted_accuracy        = 0.985
Min_trainable_epoch      = 20
H, W, B (image dims)     = 330, 307, 6
```

---

## 5. Methods

## 5.1 Pearson Correlation Masked Attention

### a) What it is

> Imagine standing in the center of a 9Ã—9 grid of pixels, each with its own "spectral
> fingerprint" (6 numbers, one per band). This layer asks: "Which of my 80 neighbors have a
> spectral fingerprint that *rises and falls together* with mine?" Neighbors whose fingerprints
> move in lockstep with the center pixel's get amplified; neighbors whose fingerprints are
> unrelated or opposite get suppressed. It's a similarity-based spotlight, shone outward from
> the center pixel.

`Pearson_correlation_masked` is a custom Keras layer that computes the Pearson correlation
coefficient between each pixel in a spatial patch and the patch's central pixel (across the
spectral/channel dimension), thresholds that correlation at its own mean, and uses the result as
an attention mask multiplied back onto the input.

### b) Why it's used here

In hyperspectral patch classification, the center pixel is the one being labeled, and its
surrounding 9Ã—9 neighborhood may contain a mix of land-cover types (mixed pixels at class
boundaries). This layer is an optional preprocessing step (`use_pearson_corr=True`) that
re-weights the patch so that spatially/spectrally similar neighbors contribute more to the
downstream convolution/transformer, while dissimilar neighbors (likely belonging to a different
class) are down-weighted. In this notebook, `use_pearson_corr = False` globally, so this layer
is defined but not active in the trained models â€” it remains available as a toggle.

### c) How it works â€” Step by step

1. Compute `x_mean`: the mean across the channel axis for every pixel in the patch, repeated
   back out to all channels (so it has the same shape as the input).
2. Extract the central pixel `y` (at spatial index `loc = P_S // 2`), and tile/repeat it across
   the full `P_S Ã— P_S` spatial extent so every position can be compared against it.
3. Compute `y_mean`: the channel-wise mean of the (tiled) central pixel, also repeated across
   channels.
4. Compute deviations from the mean:
   ```
   a = inputs - x_mean      # each pixel's deviation from its own channel-mean
   b = y_tiled - y_mean     # center pixel's deviation from its channel-mean, tiled everywhere
   ```
5. Compute the Pearson correlation coefficient per spatial position:
   ```
   numerator   = sum_over_channels(a * b)
   a_sq_sum    = sum_over_channels(a * a)
   b_sq_sum    = sum_over_channels(b * b)
   denominator = sqrt(a_sq_sum * b_sq_sum)
   corr        = numerator / denominator
   ```
6. Threshold: compute `thresh = mean(corr)` over the whole tensor, then build a binary mask
   `mask = (corr > thresh)`.
7. Apply the mask: `masked_corr = mask * corr`, then repeat this single-channel map across all
   `channels` so it matches the input's shape.
8. Final output: `inputs * attention_weights` (element-wise multiply) â€” pixels whose correlation
   with the center exceeds the average correlation keep (a scaled version of) their original
   values; others are zeroed out.

### d) ASCII Flow Diagram

```
Input patch (P_S x P_S x B)
        |
        v
+----------------------------+      +-------------------------------+
| mean over channels (x_mean)|      | extract center pixel, tile it  |
+----------------------------+      | -> mean over channels (y_mean) |
        |                            +-------------------------------+
        v                                        |
   a = input - x_mean                            v
                                       b = tiled_center - y_mean
        \                                        /
         \                                      /
          v                                    v
        elementwise multiply a*b, sum over channels -> numerator
        sum(a*a) over channels, sum(b*b) over channels -> a_sq, b_sq
                              |
                              v
                  corr = numerator / sqrt(a_sq * b_sq)
                              |
                              v
                  thresh = mean(corr over all positions)
                  mask = (corr > thresh)
                  masked_corr = mask * corr
                              |
                              v
            attention_weights = repeat(masked_corr, channels)
                              |
                              v
            output = input * attention_weights  (elementwise)
```

### e) Worked Numerical Example

Suppose `P_S = 3` (a 3Ã—3 patch for simplicity) with `B = 2` channels, and we look at just two
positions: the **center pixel** `C` and one **neighbor pixel** `N`.

- Center pixel `C` has channel values `[4, 6]` â†’ mean = 5
- Neighbor pixel `N` has channel values `[2, 8]` â†’ mean = 5
- Another neighbor `M` has channel values `[5, 5]` â†’ mean = 5

Deviations from each pixel's own mean:
- `C`: `a_C = [4-5, 6-5] = [-1, 1]`
- `N`: `a_N = [2-5, 8-5] = [-3, 3]`
- `M`: `a_M = [5-5, 5-5] = [0, 0]`

The "b" term is always the *center pixel's* deviation, `b = [-1, 1]`, broadcast to every
position.

For neighbor `N`:
```
numerator   = sum(a_N * b) = (-3 * -1) + (3 * 1) = 3 + 3 = 6
a_sq_sum    = sum(a_N * a_N) = 9 + 9 = 18
b_sq_sum    = sum(b * b)     = 1 + 1 = 2
denominator = sqrt(18 * 2) = sqrt(36) = 6
corr_N      = 6 / 6 = 1.0
```

For neighbor `M`:
```
numerator   = sum(a_M * b) = (0 * -1) + (0 * 1) = 0
a_sq_sum    = sum(a_M * a_M) = 0
b_sq_sum    = 2
denominator = sqrt(0 * 2) = 0
corr_M      = 0 / 0  -> NaN in practice (degenerate case: constant pixel)
```

For the center pixel `C` itself, `a_C == b`, so `corr_C = 1.0` (perfect self-correlation).

Suppose across the whole patch the average correlation is `thresh = 0.5`. Then:
- `corr_N = 1.0 > 0.5` â†’ mask = 1 â†’ `masked_corr_N = 1.0` â†’ neighbor `N` is **kept/amplified**
- `corr_M` is degenerate (NaN/0) â†’ likely falls below threshold â†’ neighbor `M` is **suppressed**
- `corr_C = 1.0 > 0.5` â†’ center pixel is always kept

The final output multiplies each pixel's original 2-channel values by its corresponding
`attention_weight` (the masked correlation, repeated across the 2 channels).

### f) Code Walkthrough

```python
class Pearson_correlation_masked(layers.Layer):
    """Apply a Pearson-correlation attention mask to image patches."""

    def __init__(self, P_S=9, **kwargs):
        # P_S: spatial side length of the input patch (e.g., 9 for a 9x9 patch)
        super(Pearson_correlation_masked, self).__init__(**kwargs)
        self.P_S = P_S

    def call(self, inputs):
        # inputs shape: (batch, P_S, P_S, channels)
        loc      = self.P_S // 2          # index of the center pixel, e.g. 4 for P_S=9
        channels = inputs.shape[-1]

        # Step 1: per-pixel mean across channels, broadcast back to `channels` width
        x_mean = tf.repeat(tf.math.reduce_mean(inputs, axis=-1, keepdims=True),
                            repeats=channels, axis=-1)

        # Step 2: grab the center pixel (1x1xC) and tile it to PxP so it aligns
        # spatially with every other pixel in the patch
        y = tf.repeat(
            tf.repeat(inputs[:, loc:loc+1, loc:loc+1, :], repeats=self.P_S, axis=-2),
            repeats=self.P_S, axis=-3,
        )
        # Step 3: mean of the (tiled) center pixel across channels
        y_mean = tf.repeat(tf.math.reduce_mean(y, axis=-1, keepdims=True),
                            repeats=channels, axis=-1)

        # Step 4: deviations from each pixel's own mean / the center pixel's mean
        a  = tf.math.subtract(inputs, x_mean)
        b  = tf.math.subtract(y, y_mean)
        ab = tf.math.multiply(a, b)

        # Step 5: Pearson correlation formula, computed per spatial position
        num    = tf.math.reduce_sum(ab, axis=-1, keepdims=True)
        a_sq   = tf.math.reduce_sum(tf.math.multiply(a, a), axis=-1, keepdims=True)
        b_sq   = tf.math.reduce_sum(tf.math.multiply(b, b), axis=-1, keepdims=True)
        deno   = tf.math.sqrt(tf.math.multiply(a_sq, b_sq))
        corr   = tf.math.divide(num, deno)

        # Step 6-7: threshold at the mean correlation value, zero out below-average
        # correlations, then broadcast the resulting mask back across channels
        thresh         = tf.math.reduce_mean(corr)
        mask           = tf.cast(corr > thresh, corr.dtype)
        masked_corr    = tf.math.multiply(mask, corr)
        attention_wts  = tf.repeat(masked_corr, repeats=channels, axis=-1)

        # Step 8: apply the attention weights to the original input
        return multiply([inputs, attention_wts])

    def get_config(self):
        # Required for Keras to save/reload this custom layer
        config = super(Pearson_correlation_masked, self).get_config()
        config.update({"P_S": self.P_S})
        return config
```

### g) Output & Interpretation

The layer outputs a tensor of the **same shape** as its input (`P_S Ã— P_S Ã— B`), where each
pixel's spectral values have been scaled by an attention weight between 0 and roughly the
maximum correlation value (since correlation itself is in `[-1, 1]`, but only positive,
above-average correlations survive the mask). A pixel with `attention_weight â‰ˆ 1` is treated as
"spectrally consistent with the center" and passes through largely unchanged; a pixel with
`attention_weight = 0` is zeroed out, effectively removing it from the patch's contribution to
downstream layers. In this notebook, this layer is wired into every model builder via the
`use_pearson_corr` flag, but since `use_pearson_corr = False` globally, it is **not active** in
any of the three trained models (`AlexNet`, `GFNet`, `create_vit_classifier` all skip it).

### h) Limitations

- The correlation denominator can be zero (e.g., for a constant-valued pixel), producing
  `NaN`/`inf` values that are not explicitly guarded against (no epsilon added to `deno`).
- Thresholding at the *batch-and-patch-wide mean* correlation means the cutoff is data-dependent
  and not a fixed, interpretable similarity threshold.
- The layer has no learnable parameters â€” it is a fixed, hand-designed attention mechanism, so
  it cannot adapt to the task during training.
- Because it is disabled (`use_pearson_corr=False`) for all three production models in this
  notebook, its practical effect on the reported results is currently zero.

---

## 5.2 Staged Channel-Shift Dropout (Dropout_Train)

### a) What it is

> Standard dropout randomly silences a different random subset of neurons on every forward
> pass, like flipping a fresh set of coin flips each time. `Dropout_Train` instead works like a
> rotating spotlight crew: at "shift 1," it permanently dims the first quarter of the channels;
> at "shift 2," it dims the *next* quarter; and so on. Every channel gets its turn being dropped,
> but within a given shift the same channels are dropped every time â€” it's deterministic, not
> random.

`Dropout_Train` is a custom Keras layer (registered under the Keras namespace via
`@keras_export('keras.layers.Dropout')`) that, during training, zeroes out a **contiguous
slice** of the channel dimension determined by `rate` and `shift`, rather than randomly zeroing
individual activations.

### b) Why it's used here

This layer is the mechanism behind the notebook's "staged training" strategy. With
`dropout_rate = 0.25`, there are `shifts = int(1/0.25) = 4` stages. In each stage, a different
25%-wide slice of channels in a given layer is zeroed out for the *entire* stage. The intention
appears to be a form of structured regularization / implicit ensembling: each "shift" forces the
network to perform well while a specific quarter of its representation is unavailable, which may
encourage redundancy across channel groups. This layer is placed at every position in the
network named with the `TRAIN_DROPOUT_*` convention, which `Custom_callbacks` later finds and
replaces.

### c) How it works â€” Step by step

1. If `rate == 0`, the layer is a no-op (`tf.identity`).
2. Determine the training/inference mode (`training` flag, defaulting to
   `K.learning_phase()` if not explicitly passed).
3. If training, compute the channel range to zero out for the **current** `shift`:
   ```
   range_0 = floor(rate * (shift - 1) * num_channels)
   range_1 = floor(rate * shift * num_channels)   # or None if this is the final shift
   ```
4. Build a multiplier vector of all `1.0`s with length `num_channels`, then set
   `multiplier[range_0:range_1] = 0.0`.
5. Multiply the input element-wise by this multiplier vector (broadcast across batch and spatial
   dimensions) â€” this zeroes out exactly the channels in `[range_0, range_1)`.
6. If not training, pass the input through unchanged (`tf.identity`).
7. The layer also includes validation logic in `__init__`: `rate` must be in `[0, 1]`, `shift`
   must be an integer, and `shift * rate` must not exceed `1.0` (i.e., you cannot request a shift
   index beyond the total number of shifts).

### d) ASCII Flow Diagram

```
Input tensor (..., num_channels)
        |
        v
  is rate == 0?  ---- yes ----> output = input (identity)
        | no
        v
  is training?  ---- no ----> output = input (identity)
        | yes
        v
  range_0 = floor(rate * (shift-1) * num_channels)
  range_1 = floor(rate * shift * num_channels)  (or None on last shift)
        |
        v
  multiplier = ones(num_channels)
  multiplier[range_0:range_1] = 0
        |
        v
  output = input * multiplier   (channels in [range_0, range_1) are zeroed)
```

### e) Worked Numerical Example

Suppose a layer has `num_channels = 8`, `rate = 0.25` (so there are `1/0.25 = 4` shifts of 2
channels each).

**Shift 1** (`shift = 1`):
```
range_0 = floor(0.25 * (1-1) * 8) = floor(0) = 0
range_1 = floor(0.25 * 1 * 8) = floor(2) = 2
multiplier = [0, 0, 1, 1, 1, 1, 1, 1]   (channels 0-1 zeroed)
```

**Shift 2** (`shift = 2`):
```
range_0 = floor(0.25 * (2-1) * 8) = floor(2) = 2
range_1 = floor(0.25 * 2 * 8) = floor(4) = 4
multiplier = [1, 1, 0, 0, 1, 1, 1, 1]   (channels 2-3 zeroed)
```

**Shift 4** (`shift = 4`, the final shift, so `shift * rate = 1.0` and `range_1 = None`):
```
range_0 = floor(0.25 * (4-1) * 8) = floor(6) = 6
range_1 = None  (because shift * rate == 1.0, not < 1.0)
multiplier = [1, 1, 1, 1, 1, 1, 0, 0]   (channels 6 through end zeroed, i.e. 6 and 7)
```

If an input vector at one spatial location were `[10, 20, 30, 40, 50, 60, 70, 80]`, then under
**Shift 1** the output would be `[0, 0, 30, 40, 50, 60, 70, 80]` â€” channels 0 and 1 silenced,
all others passed through at full strength (note: unlike standard dropout, there is **no
1/(1-rate) rescaling** of the surviving channels here).

### f) Code Walkthrough

```python
@keras_export('keras.layers.Dropout')
class Dropout_Train(layers.Layer):
    """Deterministic channel-shift dropout applied during training only."""

    def __init__(self, rate, shift=1, noise_shape=None, seed=None, **kwargs):
        super(Dropout_Train, self).__init__(**kwargs)
        # Validate rate is a proper probability
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError(f"Invalid value {rate} received for `rate`...")
        # shift must be an integer (it indexes which channel-block to drop)
        if not isinstance(shift, int):
            raise TypeError(f"Invalid dtype {type(shift)} found for `shift`...")
        # Can't request a shift beyond the total number of shifts (1/rate)
        if shift * rate > 1.0:
            raise ValueError(f"Invalid value {shift} received for `shift`...")
        self.rate, self.shift = rate, shift
        self.noise_shape, self.seed = noise_shape, seed
        self.supports_masking = True

    def call(self, inputs, training=None):
        if self.rate == 0:
            return tf.identity(inputs)            # no-op
        if training is None:
            training = K.learning_phase()         # infer mode if not given

        def dropped_inputs():
            input_shape = inputs.shape
            # Compute the [range_0, range_1) slice of channels to zero this shift
            range_0 = int(self.rate * (self.shift - 1) * input_shape[-1])
            range_1 = (
                int(self.rate * self.shift * input_shape[-1])
                if self.shift * self.rate < 1.0
                else None   # last shift: zero out to the end of the channel axis
            )
            multiplier = np.ones(input_shape[-1])
            multiplier[range_0:range_1] = 0.0
            multiplier = tf.constant(multiplier)
            return Multiply()([inputs, multiplier])

        # smart_cond picks dropped_inputs() during training, identity otherwise
        return control_flow_util.smart_cond(
            training, dropped_inputs, lambda: array_ops.identity(inputs)
        )

    def compute_output_shape(self, input_shape):
        return input_shape   # shape never changes â€” only values are zeroed

    def get_config(self):
        config = super(Dropout_Train, self).get_config()
        config.update({
            "rate": self.rate, "shift": self.shift,
            "noise_shape": self.noise_shape, "seed": self.seed,
            "supports_masking": self.supports_masking,
        })
        return config
```

### g) Output & Interpretation

The output has the exact same shape as the input; the only change is that a contiguous block of
channels has its values set to zero for the duration of the current "shift." During inference
(or when `training=False`), the layer is a pure pass-through. There is no direct "uncertainty"
output from this layer by itself â€” its effect is felt indirectly through how it shapes training
dynamics: each shift forces a different channel subset to be unused, and `Custom_callbacks` (see
next section) swaps in a new `Dropout_Train` instance with an incremented `shift` once training
criteria are met.

### h) Limitations

- No rescaling of surviving channels (unlike standard dropout's `1/(1-rate)` inverted scaling),
  so the expected activation magnitude changes between shifts and between training/inference.
- The channel slice to drop is purely positional (`[range_0:range_1)` in the raw channel order);
  it assumes no particular structure in *which* channels end up in which slice, so its
  effectiveness depends on how channels are ordered/learned by preceding layers.
- `noise_shape` and `seed` parameters are accepted for API compatibility with standard `Dropout`
  but appear unused inside `call` (the masking logic does not use either).
- The `@keras_export('keras.layers.Dropout')` decorator re-registers this class under the
  built-in `keras.layers.Dropout` export name, which could cause naming collisions or confusion
  with the real `keras.layers.Dropout` if both are referenced in the same environment.

---

## 5.3 Model Modifier & Custom Staged-Training Callback

### a) What it is

> Think of a relay race where, every time a runner crosses a fitness threshold, the coach swaps
> in a slightly different runner for the next leg â€” same race, same finish line, but a new
> "configuration" each time. `modified_model` is the mechanism for swapping runners (rebuilding
> the network with new dropout layers), and `Custom_callbacks` is the coach watching the
> stopwatch and deciding when to make the swap.

`modified_model` is a function that rebuilds a Keras functional model layer-by-layer, replacing
any layer whose name matches a target substring (`"DROPOUT"`) with a fresh instance of a given
layer class (`Dropout_Train` or, in the final stage, standard `Dropout`). `Custom_callbacks` is a
Keras training callback that monitors validation accuracy and, once a threshold is met *and* a
minimum number of epochs has elapsed in the current stage, calls `modified_model` to advance to
the next "shift," cycling through all 4 shifts before switching to a final standard-dropout
phase and ultimately restoring the best-observed weights.

### b) Why it's used here

This is the orchestration layer that ties `Dropout_Train` (Section 5.2) into an actual training
loop. Each of the three architectures (AlexNet, GFNet, ViT) places several layers named
`TRAIN_DROPOUT_1`, `TRAIN_DROPOUT_2`, etc. `Custom_callbacks` is configured with
`layer_name="DROPOUT"` so it matches all of these by substring. As training proceeds:

- At `on_train_begin`, the model is immediately rebuilt with shift = 1 (so `Dropout_Train(rate,
  shift=1, ...)` replaces every `TRAIN_DROPOUT_*` layer).
- At the end of each epoch, if validation accuracy has reached `accuracy_score` (e.g., 0.985)
  **and** at least `min_epochs` epochs have passed in the current shift, the model is rebuilt
  again with the next shift index.
- After all 4 shifts have been cycled through, on the next qualifying epoch the model is rebuilt
  one final time with standard `Dropout` (shift = `"Final"`, a string rather than an int â€” this
  is the branch that triggers `new_layer(rate=rate, name=...)` without a `shift` kwarg).
- During this final phase, the callback tracks the best `val_accuracy` over the last 10 epochs
  and stores those weights.
- At `on_train_end`, if fewer than all 4 shifts completed, it raises `NotImplementedError`
  (training did not finish the staged curriculum in the allotted epochs); otherwise it restores
  the best weights and saves the model to `filepath`.

### c) How it works â€” Step by step

**`modified_model(model, layer_name, rate, new_layer, shift, **kwargs)`:**

1. Start `x` as the output of the model's input layer (`model.layers[0].output`).
2. Iterate over every subsequent layer `lyr` in `model.layers[1:]`:
   ```
   if layer_name appears in lyr.name (case-insensitive) and shift is an int:
       x = new_layer(rate=rate, shift=shift, name=f"{layer_name}_{shift}_{z}")(x)
       z += 1
   elif layer_name appears in lyr.name and shift is a string:
       x = new_layer(rate=rate, name=f"{layer_name}_{shift}_{z}")(x)
       z += 1
   else:
       x = lyr(x)   # pass through unchanged
   ```
3. If no layer matched (`modification == False`), print a warning.
4. Return a new `Model(inputs=model.layers[0].input, outputs=x, name=name)`.

**`Custom_callbacks`:**

1. `__init__`: store `filepath`, total `epochs`, `new_layer` class (default `Dropout_Train`),
   `rate`, `layer_name` (default `"DROPOUT"`), `accuracy_score` (normalized to `[0,1]` if given
   as a percentage), and `min_epochs`.
2. `on_train_begin`: set `shift = 1`, `epoch_completed = 0`, and immediately call
   `modified_model(self.model, "DROPOUT", rate, Dropout_Train, shift=1)` â€” so even epoch 0
   trains with shift-1 dropout active.
3. `on_epoch_end`:
   - Increment `epoch_completed` and `epoch_num`.
   - Compute `total_shifts = int(1 / rate)` (= 4 here).
   - Read `acc = logs["val_accuracy"]`.
   - `threshold_met = (acc >= accuracy_score) and (epoch_completed >= min_epochs)`.
   - **Case A** â€” `threshold_met and shift < total_shifts`: advance to the next shift. Increment
     `shift`, call `modified_model(...)` with the new `shift`, reset `epoch_completed = 0`.
   - **Case B** â€” `threshold_met and shift == total_shifts`: all numeric shifts done â€” rebuild
     the model with **standard `Dropout`** (`shift="Final"`), increment `shift` past
     `total_shifts`, reset `epoch_completed`.
   - **Case C** â€” threshold not met: print "need more training." If `shift >= total_shifts`
     (i.e., we're in the final standard-dropout phase) and `epoch_num >= epochs - 10` (last 10
     epochs of training), check whether `val_accuracy` improved over `self.best`; if so, save
     `self.best_weights = model.get_weights()`.
4. `on_train_end`:
   - If `shift <= total_shifts`, raise `NotImplementedError` â€” the staged curriculum did not
     complete within the allotted epochs.
   - Otherwise, restore `self.best_weights` into the model and call `model.save(filepath)`.

### d) ASCII Flow Diagram

```
on_train_begin
    |
    v
shift = 1; rebuild model with Dropout_Train(shift=1) on all "DROPOUT" layers
    |
    v
+----------------- on_epoch_end (repeated each epoch) -----------------+
|                                                                        |
|  acc = val_accuracy this epoch                                        |
|  threshold_met = (acc >= accuracy_score) AND                          |
|                   (epoch_completed >= min_epochs)                     |
|                                                                        |
|  threshold_met AND shift < 4?                                         |
|     yes -> shift += 1                                                 |
|            rebuild model with Dropout_Train(shift=new shift)          |
|            epoch_completed = 0                                        |
|                                                                        |
|  threshold_met AND shift == 4?                                        |
|     yes -> rebuild model with standard Dropout (shift="Final")        |
|            shift = 5 (past total_shifts)                              |
|            epoch_completed = 0                                        |
|                                                                        |
|  else (not threshold_met):                                            |
|     if shift >= 4 and within last 10 epochs:                          |
|         track best val_accuracy weights                               |
+------------------------------------------------------------------------+
    |
    v
on_train_end
    |
    v
shift <= 4?  --- yes ---> raise NotImplementedError (curriculum incomplete)
    | no
    v
restore best_weights into model; model.save(filepath)
```

### e) Worked Numerical Example

Let `rate = 0.25` (`total_shifts = 4`), `accuracy_score = 0.985`, `min_epochs = 20`,
`epochs = 100`. Walk through a simplified epoch-by-epoch trace:

- **Epoch 1â€“19** (shift = 1): `val_accuracy` rises from 0.70 to 0.97. `threshold_met` is `False`
  every epoch because `acc < 0.985`. Nothing changes.
- **Epoch 20** (shift = 1, `epoch_completed = 20`): suppose `val_accuracy = 0.986`.
  `threshold_met = (0.986 >= 0.985) and (20 >= 20) = True`. Since `shift (1) < total_shifts
  (4)`, advance: `shift = 2`, rebuild model with `Dropout_Train(shift=2)`, `epoch_completed = 0`.
- **Epoch 21â€“40** (shift = 2): similar â€” once `epoch_completed >= 20` and `val_accuracy >=
  0.985` again, advance to `shift = 3`.
- **Epoch 41â€“60** (shift = 3): same pattern â†’ advance to `shift = 4`.
- **Epoch 61â€“80** (shift = 4): once threshold met again at `epoch_completed >= 20`, this time
  `shift == total_shifts (4)`, so the **else branch** for Case B fires: rebuild with standard
  `Dropout` (`shift="Final"`), set `shift = 5`, `epoch_completed = 0`.
- **Epoch 81â€“100** (shift = 5, standard dropout phase, `epoch_num` now in `[81, 100]`, i.e.
  `epoch_num >= epochs - 10 = 90` for the last 10): for epochs where `epoch_num >= 90` and
  `threshold_met` is `False` (acc dips slightly, say to 0.983), the callback checks
  `current = val_accuracy`; if `current >= self.best`, it updates `self.best` and stores
  `best_weights`.
- **`on_train_end`**: `shift = 5 > total_shifts = 4`, so no error. `model.set_weights(best_weights)`,
  then `model.save(filepath)`.

If, hypothetically, training only reached `shift = 3` by epoch 100 (the curriculum did not
finish), `on_train_end` would raise:
```
NotImplementedError: model has not trained fully in the available no. of epochs
 only 2 shifts completed out of 4
```

### f) Code Walkthrough

```python
def modified_model(model, layer_name, rate, new_layer, shift, **kwargs):
    """Rebuild `model`, replacing layers matching `layer_name` with `new_layer`."""
    name         = kwargs.get("name", None)
    x            = model.layers[0].output   # start from the Input layer's output tensor
    modification = False
    z            = 0   # counter to keep replaced-layer names unique

    for lyr in model.layers[1:]:
        # Match if `layer_name` ("DROPOUT") is a substring of the layer's name,
        # checked both as-is and uppercased for case-insensitivity
        if (layer_name in lyr.name or layer_name in lyr.name.upper()) and isinstance(shift, int):
            # Numeric shift -> insert a Dropout_Train(shift=...) layer
            x = new_layer(rate=rate, shift=shift, name=f"{layer_name}_{shift}_{z}")(x)
            modification = True
            z += 1
        elif (layer_name in lyr.name or layer_name in lyr.name.upper()) and isinstance(shift, str):
            # String shift (e.g. "Final") -> insert a standard Dropout layer
            x = new_layer(rate=rate, name=f"{layer_name}_{shift}_{z}")(x)
            modification = True
            z += 1
        else:
            x = lyr(x)   # re-apply the original layer unchanged

    if not modification:
        print("___...Model has not been modified___...")
    return Model(inputs=model.layers[0].input, outputs=x, name=name)


class Custom_callbacks(tf.keras.callbacks.Callback):
    def __init__(self, filepath, epochs, rate, new_layer=Dropout_Train,
                 layer_name="DROPOUT", accuracy_score=0.99, min_epochs=50):
        super(Custom_callbacks, self).__init__()
        self.filepath, self.epochs = filepath, epochs
        self.new_layer, self.rate  = new_layer, rate
        self.best, self.epoch_num  = 0.0, 1
        self.layer_name            = layer_name
        self.min_epochs            = min_epochs
        # Allow accuracy_score to be given as a percentage (e.g. 98.5) or fraction (0.985)
        self.accuracy_score = accuracy_score if accuracy_score <= 1.0 else accuracy_score / 100.0

    def on_train_begin(self, logs=None):
        self.shift, self.epoch_completed = 1, 0
        print(f"Model will be trained in {int(1 / self.rate)} shifts")
        # Immediately apply shift-1 dropout before the first epoch
        self.model = modified_model(self.model, self.layer_name, self.rate, self.new_layer, self.shift)

    def on_train_end(self, logs=None):
        if self.shift <= int(1 / self.rate):
            raise NotImplementedError(
                f"model has not trained fully in the available no. of epochs\n"
                f" only {self.shift - 1} shifts completed out of {int(1 / self.rate)}"
            )
        self.model.set_weights(self.best_weights)
        self.model.save(self.filepath)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_completed += 1
        self.epoch_num       += 1
        total_shifts          = int(1 / self.rate)
        acc                   = logs.get("val_accuracy", 0.0)
        threshold_met         = (acc >= self.accuracy_score) and (self.epoch_completed >= self.min_epochs)

        if threshold_met and self.shift < total_shifts:
            self.shift          += 1
            self.model           = modified_model(self.model, self.layer_name, self.rate,
                                                    self.new_layer, self.shift)
            self.epoch_completed = 0
        elif threshold_met and self.shift == total_shifts:
            # Final stage: switch to standard Dropout
            self.model           = modified_model(self.model, self.layer_name, self.rate,
                                                    self.new_layer, "Final", name="AlexNet")
            self.shift          += 1
            self.epoch_completed = 0
        else:
            if self.shift >= total_shifts:
                current = logs.get("val_accuracy")
                if not np.less(current, self.best) and (self.epoch_num >= self.epochs - 10):
                    self.best         = current
                    self.best_weights = self.model.get_weights()
```

> **Note:** This interpretation of the multi-head loss aggregation (`metrics=['accuracy'] * 7`
> producing per-head metric keys like `head_1_accuracy`, `head_2_accuracy`, ... and an overall
> `val_accuracy` that `logs.get("val_accuracy", 0.0)` reads) is inferred from Keras's standard
> multi-output naming conventions â€” verify against the actual training logs if reproducing this
> notebook, since with no single output named exactly `"accuracy"`, `logs.get("val_accuracy",
> 0.0)` may default to `0.0` for multi-head models unless Keras aggregates an overall metric
> under that key.

### g) Output & Interpretation

The direct "output" of this machinery is a **trained, saved model file** at `filepath` (an
`AlexNet`-named model in the final rebuild, regardless of which architecture was originally
passed in â€” note the hardcoded `name="AlexNet"` in the final `modified_model` call, which is
likely a leftover from when this callback was developed for the AlexNet model only). Indirectly,
its effect is on *which weights end up in the saved model*: the callback ensures the final saved
model represents the best validation accuracy observed during the last 10 epochs of the
standard-dropout phase, rather than simply whatever weights existed at the very last epoch.

### h) Limitations

- The hardcoded `name="AlexNet"` in the final-stage `modified_model` call means every model's
  internal Keras model name becomes `"AlexNet"` after the staged training completes, regardless
  of whether it's actually the GFNet or ViT architecture â€” purely cosmetic but potentially
  confusing when inspecting `model.name` later.
- `logs.get("val_accuracy", 0.0)` assumes a key literally named `"val_accuracy"` exists in the
  training logs; with 7 separate output heads each producing their own per-head accuracy
  metric, whether Keras populates an aggregate `val_accuracy` key depends on the Keras/TF
  version's multi-output metric naming behavior.
- If the staged curriculum does not complete within `epochs`, the entire training run ends in a
  raised `NotImplementedError` â€” there's no partial-save fallback for an incomplete run.
- Rebuilding the entire functional model at every shift transition (5 times total: initial +
  3 shift advances + final) is computationally expensive and resets the Keras graph/optimizer
  state references each time.

---

## 5.4 Multi-Head AlexNet CNN

### a) What it is

> Think of hiring 7 identical detectives to examine the same crime scene photograph simultaneously,
> but each one writes their own independent verdict â€” and you take a vote. The AlexNet backbone
> is the shared magnifying glass every detective uses, but the 7 final Dense layers (the "heads")
> write separate verdicts from the same evidence. The majority vote is your final classification.

`AlexNet` (as implemented here) is a five-block convolutional neural network inspired by the
original AlexNet architecture, adapted for small 9Ã—9Ã—6 hyperspectral patches. It uses five
stacked `Conv2D` layers followed by max pooling, then three fully connected dense layers (each
guarded by a `Dropout` / `Dropout_Train` layer), and finally fans out to `K_HEADS = 7`
independent `softmax` output heads sharing the same features.

### b) Why it's used here

AlexNet's deep convolutional stack is well suited to learning hierarchical spatial features in
hyperspectral patches. The multi-head design (K_HEADS = 7) means the model outputs 7 separate
probability distributions over the land-cover classes at each inference step. During training,
all 7 heads receive the same gradient supervision (each head's loss is `sparse_categorical_crossentropy`).
At inference time, these 7 distributions are averaged (see Section 5.8), reducing variance and
improving calibration â€” essentially a built-in lightweight ensemble.

### c) How it works â€” Step by step

1. **Input**: a tensor of shape `(batch, 9, 9, 6)` â€” a batch of 9Ã—9 patches, each with 6
   spectral bands.
2. **Optional attention**: if `use_pearson_corr=True`, the input first passes through
   `Pearson_correlation_masked` (Section 5.1); otherwise the raw patch is used.
3. **Convolutional backbone** (5 blocks):
   ```
   Conv2D(96,  3x3, relu, padding='same')
   Conv2D(256, 3x3, relu, padding='same')
   Conv2D(384, 3x3, relu, padding='same')
   Conv2D(384, 3x3, relu, padding='same')
   Conv2D(256, 3x3, relu, padding='same')
   MaxPooling2D(2x2, stride 2, padding='same')
   ```
   Each `Conv2D` uses `padding='same'` so spatial dimensions are preserved until the pooling
   step, which halves them: `9 â†’ 5` (ceiling division under 'same' padding).
4. **Flatten**: the 5Ã—5Ã—256 feature map is reshaped to a vector of length 6400.
5. **Dense head**:
   ```
   Dense(4096, relu) -> Dropout("TRAIN_DROPOUT_1") -> Dense(1024, relu) ->
   Dropout("TRAIN_DROPOUT_2") -> Dense(256, relu) -> Dropout("TRAIN_DROPOUT_3") ->
   Dense(32, relu)
   ```
   The three `Dropout` layers are named with the `TRAIN_DROPOUT_*` convention so
   `Custom_callbacks` can locate and replace them during staged training.
6. **Multi-head outputs**: the 32-dimensional representation is fed into 7 parallel
   `Dense(num_classes, softmax)` layers, each named `head_1` through `head_7`.
7. **Model**: returned as a Keras `Model` with one input and a list of 7 output tensors.

### d) ASCII Flow Diagram

```
Input (batch, 9, 9, 6)
        |
        v (optional)
[Pearson_correlation_masked] -- if use_pearson_corr
        |
        v
Conv2D(96,  3x3, relu) -- shape: (batch, 9, 9, 96)
        |
Conv2D(256, 3x3, relu) -- shape: (batch, 9, 9, 256)
        |
Conv2D(384, 3x3, relu) -- shape: (batch, 9, 9, 384)
        |
Conv2D(384, 3x3, relu) -- shape: (batch, 9, 9, 384)
        |
Conv2D(256, 3x3, relu) -- shape: (batch, 9, 9, 256)
        |
MaxPooling2D(2x2, stride 2) -- shape: (batch, 5, 5, 256)
        |
Flatten -- shape: (batch, 6400)
        |
Dense(4096, relu)
        |
Dropout [TRAIN_DROPOUT_1]
        |
Dense(1024, relu)
        |
Dropout [TRAIN_DROPOUT_2]
        |
Dense(256, relu)
        |
Dropout [TRAIN_DROPOUT_3]
        |
Dense(32, relu)
        |
   _____|__________________________________________
  |       |       |       |       |       |       |
head_1  head_2  head_3  head_4  head_5  head_6  head_7
Dense(num_classes, softmax) x 7
```

### e) Worked Numerical Example

Suppose `num_classes = 4` and `K_HEADS = 2` (simplified). After the Dense(32) layer, imagine
the shared feature vector is `[0.1, 0.5, -0.3, ..., 0.8]` (length 32).

Head 1 dense weights produce logits `[1.2, 0.3, -0.5, 0.8]`, softmax gives:
```
exp([1.2, 0.3, -0.5, 0.8]) = [3.32, 1.35, 0.61, 2.23]
sum = 7.51
probs_head_1 = [0.44, 0.18, 0.08, 0.30]   -> predicted class = 0
```

Head 2 dense weights produce logits `[0.9, 0.6, -0.2, 1.1]`, softmax gives:
```
exp([0.9, 0.6, -0.2, 1.1]) = [2.46, 1.82, 0.82, 3.00]
sum = 8.10
probs_head_2 = [0.30, 0.22, 0.10, 0.37]   -> predicted class = 3
```

After averaging (Section 5.8):
```
avg_probs = [(0.44+0.30)/2, (0.18+0.22)/2, (0.08+0.10)/2, (0.30+0.37)/2]
          = [0.37, 0.20, 0.09, 0.34]
final prediction = argmax([0.37, 0.20, 0.09, 0.34]) = class 0
```

The two heads disagreed (head 1 â†’ class 0, head 2 â†’ class 3), but averaging revealed class 0
as the marginally stronger consensus.

### f) Code Walkthrough

```python
def AlexNet(input_shape, num_classes=13, use_pearson_corr=False, dropout_rate=0.5):
    """Build the multi-head AlexNet-style convolutional classifier."""
    K_HEADS = 7  # number of independent output heads

    x_input = Input(input_shape)
    # Optional spatial-attention preprocessing; skipped when use_pearson_corr=False
    X = Pearson_correlation_masked(P_S)(x_input) if use_pearson_corr else x_input

    # â”€â”€ Convolutional backbone: 5 Conv2D blocks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # All use 3x3 kernels with 'same' padding; spatial dims preserved until pooling
    X = Conv2D(filters=96,  kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(X)
    X = Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(X)
    X = Conv2D(filters=384, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same')(X)
    # Spatial downsampling: 9x9 -> 5x5 (with 'same' ceiling division)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(X)

    # â”€â”€ Dense classification head â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X = Flatten()(X)  # 5*5*256 = 6400-dimensional vector
    X = Dense(4096, activation='relu')(X)
    X = Dropout(dropout_rate, name="TRAIN_DROPOUT_1")(X)  # named for Custom_callbacks targeting
    X = Dense(1024, activation='relu')(X)
    X = Dropout(dropout_rate, name="TRAIN_DROPOUT_2")(X)
    X = Dense(256,  activation='relu')(X)
    X = Dropout(dropout_rate, name="TRAIN_DROPOUT_3")(X)
    X = Dense(32,   activation='relu')(X)

    # â”€â”€ 7 independent softmax output heads â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    output_heads = [
        Dense(num_classes, activation='softmax', dtype='float32', name=f'head_{i+1}')(X)
        for i in range(K_HEADS)
    ]
    return Model(inputs=x_input, outputs=output_heads, name="MultiHead_AlexNet")
```

### g) Output & Interpretation

The model returns a **list of 7 tensors**, each of shape `(batch, num_classes)`, representing
probability distributions over land-cover classes from each of the 7 heads. During training,
Keras computes cross-entropy loss independently for each head and sums them. During evaluation,
`predict_multihead` (Section 5.8) averages these 7 distributions and returns the argmax class
label. A pixel confidently classified the same way by all 7 heads indicates high inter-head
agreement; diverging head predictions flag uncertain or boundary pixels.

### h) Limitations

- The original AlexNet was designed for `227Ã—227Ã—3` ImageNet images; here it operates on
  `9Ã—9Ã—6` patches, so the spatial hierarchy is compressed into 5 convolutional layers with no
  spatial reduction until the final pooling â€” the receptive field is effectively the whole patch
  from the very first layer.
- With three large Dense layers (4096 â†’ 1024 â†’ 256), the parameter count for the dense head
  alone is substantial relative to the 9Ã—9 input, risking overfitting on small datasets.
- No batch normalization is applied in the convolutional backbone; this can slow convergence
  and increase sensitivity to the learning-rate schedule.
- The 7 heads share all weights except the final Dense layer, so the "ensemble diversity"
  between heads comes only from the random initialization of those 7 final weight matrices â€”
  in practice, well-trained heads tend to converge to nearly identical outputs.

---

## 5.5 GFNet: Global Filter Network

### a) What it is

> Standard self-attention is like every pixel in a row whispering to every other pixel to figure
> out who's important â€” expensive as the sequence gets longer. GFNet replaces that whisper
> network with a single public announcement board (a learned frequency-domain filter): the
> whole sequence is broadcast-transformed into frequency space via FFT, multiplied element-wise
> by a learned complex weight mask, then transformed back. Same global reach, far less compute.

GFNet replaces the self-attention module in Vision Transformers with a **global filter** in the
2-D frequency domain. The input feature map is transformed via a real-valued 2-D FFT
(`tf.signal.rfft2d`), multiplied by a learned complex weight tensor, and transformed back via
the inverse FFT (`tf.signal.irfft2d`). This operation has the same global receptive field as
self-attention but scales as `O(N log N)` rather than `O(N^2)`.

### b) Why it's used here

GFNet offers a computationally efficient alternative to multi-head attention for the
patch-sequence token representations produced from the 9Ã—9 hyperspectral patches. With
`GlobalFilter_layers = 12` stacked GFNet blocks, the model builds deep frequency-domain
representations of patch structure. The multi-head output design (K_HEADS = 7) is shared with
the AlexNet and ViT variants, enabling a fair comparison between the three architectures on
the same LULC task.

### c) How it works â€” Step by step

**Patch extraction and encoding (`GF_Patches`, `GF_PatchEncoder`):**
1. The 9Ã—9Ã—6 input is either processed with Conv2D-based patching (`patch_method='conv'`) or
   extracted via `tf.image.extract_patches` with `patch_size=3`. With the `'extract'` method,
   `9/3 = 3` patches per spatial dimension â†’ `num_patches = 9` total tokens of dimension
   `3*3*6 = 54`.
2. `GF_PatchEncoder` projects the raw 54-dimensional patch tokens to `hidden_dim = 512` via a
   Dense layer and adds learned positional embeddings (one embedding per patch position, no CLS
   token in GFNet).
3. A `Dropout` layer (`TRAIN_DROPOUT_1`) is applied after encoding.

**GFNet Block (`GF_Block`):**
4. `LayerNormalization` on the input sequence.
5. `GF_GlobalFilter`: reshape the `(batch, N, C)` sequence to `(batch, sqrt(N), sqrt(N), C)`,
   apply `tf.signal.rfft2d` along the spatial axes, multiply by a learned complex weight tensor
   of shape `(sqrt(N), sqrt(N)//2+1, C, 2)`, apply `tf.signal.irfft2d`, reshape back to
   `(batch, N, C)`.
   ```
   x_freq   = rfft2d(x_spatial)                   # forward real FFT
   weight   = complex(W[:,:,:,0], W[:,:,:,1])     # learned complex weights
   x_filt   = x_freq * weight                     # element-wise complex multiply
   x_out    = irfft2d(x_filt)                     # inverse FFT back to spatial
   ```
6. `GF_DropPath`: stochastic depth regularization on the filter output residual.
7. A second `LayerNormalization`, then `GF_MLP` (two GELU Dense layers), then another
   `GF_DropPath` for the MLP residual.
8. Both sub-paths are residuals: `x = x + drop_path(mlp(norm2(filter(norm1(x)))))`.
9. Steps 4â€“8 are repeated for `GlobalFilter_layers = 12` stacked blocks.

**Pooling and heads:**
10. `Dropout` (`TRAIN_DROPOUT_2`), `LayerNormalization`, `GF_Expand_Dims` (adds a dummy
    spatial dim), `GlobalAveragePooling2D`, `Flatten`.
11. `Dropout` (`TRAIN_DROPOUT_3`).
12. 7 parallel `Dense(num_classes, softmax)` heads, same as AlexNet.

### d) ASCII Flow Diagram

```
Input (batch, 9, 9, 6)
        |
[GF_Patches] -> (batch, 9, 54)  [9 tokens of dim 3*3*6]
        |
[GF_PatchEncoder] -> (batch, 9, 512)  [project to hidden_dim + pos embeddings]
        |
[TRAIN_DROPOUT_1]
        |
+-- x 12 GF_Block iterations ---------------------+
|                                                   |
|  x -> LayerNorm -> GF_GlobalFilter:               |
|       reshape to (batch, 3, 3, 512)               |
|       rfft2d  ->  (batch, 3, 2, 512) [freq]       |
|       * complex weights (learned)                 |
|       irfft2d -> (batch, 3, 3, 512) [spatial]     |
|       reshape to (batch, 9, 512)                  |
|  + DropPath residual                              |
|  x -> LayerNorm -> GF_MLP (GELU Dense x2)        |
|  + DropPath residual                              |
+---------------------------------------------------+
        |
[TRAIN_DROPOUT_2]
        |
[LayerNormalization]
        |
[GF_Expand_Dims] -> (batch, 9, 1, 512)
        |
[GlobalAveragePooling2D] -> (batch, 512)
        |
[Flatten]
        |
[TRAIN_DROPOUT_3]
        |
   _____|_________ ... 7 heads
head_1 ... head_7  Dense(num_classes, softmax)
```

### e) Worked Numerical Example

Suppose `num_patches = 4` (2Ã—2 grid), `hidden_dim = 2`, `num_classes = 2`.
After patch encoding the sequence is (ignoring batch):
```
x = [[1.0, 0.5],     <- token 0
     [0.3, 0.8],     <- token 1
     [0.7, 0.2],     <- token 2
     [0.9, 0.4]]     <- token 3
```
Reshape to spatial: `(2, 2, 2)`.

Apply `rfft2d` along the 2Ã—2 spatial axes for each of the 2 channels. For channel 0
(`[1.0, 0.3; 0.7, 0.9]`), the 2D real FFT produces complex coefficients at frequencies
`(0,0), (0,1), (1,0), (1,1)` â€” the DC component `(0,0)` is the sum = 2.9, and other
coefficients capture spatial frequencies. After multiplying each coefficient by a learned
complex weight and applying the inverse FFT, the values are returned to spatial domain but
with their frequency components re-weighted by what the network has learned to emphasize (e.g.,
low-frequency smooth patterns vs. high-frequency edges). After `irfft2d` and reshape back to
`(4, 2)`, the result is fed through the MLP and added back residually.

### f) Code Walkthrough

```python
class GF_GlobalFilter(layers.Layer):
    """Apply the Global Filter operation: FFT -> learned complex multiply -> IFFT."""

    def build(self, input_shape):
        w_init = tf.random_uniform_initializer()
        # Shape: (sqrt(N), sqrt(N)//2+1, C, 2) â€” the last dim holds (real, imag) parts
        self.complex_weight = self.add_weight(
            name="complex_weight",
            shape=(self.patch_size, self.patch_size, input_shape[-1] // 2 + 1, 2),
            initializer=w_init,
            trainable=True,  # learned during backpropagation
        )

    def call(self, x, **kwargs):
        B, N, C = x.shape
        a = b = int(math.sqrt(N))          # reshape token sequence to spatial grid
        x = tf.reshape(x, [-1, a, b, C])  # (batch, sqrt(N), sqrt(N), C)

        x = tf.signal.rfft2d(x)           # real-valued 2D FFT: output shape (batch, a, b//2+1, C)
        # Build complex weight from stored (real, imag) pairs
        weight = tf.dtypes.complex(self.complex_weight[:, :, :, 0],
                                   self.complex_weight[:, :, :, -1])
        x = x * weight                    # element-wise complex multiply in frequency domain
        x = tf.signal.irfft2d(x)         # inverse FFT back to spatial domain
        return tf.reshape(x, [-1, N, C]) # flatten back to token sequence
```

```python
class GF_Block(tf.keras.layers.Layer):
    """One GFNet block: LayerNorm -> GlobalFilter -> DropPath -> LayerNorm -> MLP -> DropPath."""

    def call(self, x):
        # Single residual combining the filter branch and MLP branch in one line
        # Equivalent to: h = filter(norm1(x)); x = x + drop_path(mlp(norm2(h)))
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x
```

```python
def GFNet(input_shape=(P_S, P_S, B), ...):
    """Build the multi-head GFNet classifier."""
    K_HEADS = 7

    x_input = Input(shape=input_shape)
    x = Pearson_correlation_masked(P_S)(x_input) if use_pearson_corr else x_input

    x = GF_Patches(patch_size)(x)             # extract patch tokens
    x = GF_PatchEncoder(num_patches, hidden_dim)(x)  # project + positional embedding
    x = Dropout(dropout_rate, name="TRAIN_DROPOUT_1")(x)

    for _ in range(GlobalFilter_layers):       # stack 12 GFNet blocks
        x = GF_Block(patch_size=patch_size, dim=hidden_dim,
                     mlp_ratio=mlp_ratio, drop=dropout_rate)(x)

    x = Dropout(dropout_rate, name="TRAIN_DROPOUT_2")(x)
    x = LayerNormalization()(x)
    x = GF_Expand_Dims(ndim=2)(x)             # add dummy spatial dim for GAP
    x = GlobalAveragePooling2D()(x)           # average across token sequence
    x = Flatten()(x)
    x = Dropout(dropout_rate, name="TRAIN_DROPOUT_3")(x)

    output_heads = [
        Dense(num_classes, activation="softmax", dtype="float32", name=f"head_{i+1}")(x)
        for i in range(K_HEADS)
    ]
    return keras.Model(inputs=x_input, outputs=output_heads, name="MultiHead_GFNet")
```

### g) Output & Interpretation

Same as AlexNet: a list of 7 `(batch, num_classes)` probability tensors. GFNet's global
filtering operation means every token in the sequence has attended to every other token's
frequency components â€” a truly global operation applied 12 times. Compared to AlexNet's purely
local convolutions, GFNet captures long-range correlations within the 9-token sequence that
represent relationships between different sub-regions of the 9Ã—9 patch.

### h) Limitations

- `int(math.sqrt(N))` in `GF_GlobalFilter.call` assumes `N` is a perfect square. With
  `num_patches = 9` (= 3Ã—3), this holds, but any patch size configuration that yields
  non-square `N` will silently produce incorrect reshapes.
- `rfft2d` operates along the last two spatial axes; the reshape from `(batch, N, C)` to
  `(batch, sqrt(N), sqrt(N), C)` puts spatial information in axes 1 and 2 and channels in axis
  3 â€” this is the correct layout for `tf.signal.rfft2d` which by default operates on the last
  two axes. However, the channel axis is treated as a batch dimension during FFT, which means
  spatial frequency mixing is done independently per channel, not across channels.
- `GF_DropPath.call` uses `random_tensor.floor_()` â€” the in-place `floor_()` method is a PyTorch
  convention, not a TensorFlow method. This will raise an `AttributeError` in TF if `drop_prob > 0`
  and `training=True`. In practice this layer is initialized with `training=False` in the model
  (the `__init__` sets `self.training = False` and the `GF_Block` constructor uses `drop_path=0.0`
  by default), so this bug is currently dormant.
- With `hidden_dim = 512` and 12 GFNet blocks each containing a 2Ã—-expanded MLP
  (`mlp_ratio = 4`, so the hidden MLP dimension is 2048), the parameter count is very large
  relative to the 9-token, 9Ã—9 input â€” GFNet was originally designed for 196+ token
  ImageNet-sized sequences.

---

## 5.6 Vision Transformer with U-Net-Style Skip Connections

### a) What it is

> A standard Vision Transformer is like a council of experts who each read every word of a
> document (the image patches) and update their understanding by listening to every other
> expert. This ViT adds a twist borrowed from U-Net: the second half of the council can also
> listen back to the *first* half's earlier conclusions via skip connections â€” like passing
> written notes forward in time to the later deliberators, giving them context they might
> otherwise have lost.

The ViT model here (`create_vit_classifier`) is a Vision Transformer that extracts patch tokens
from the 9Ã—9 input, prepends a learnable CLS token, adds positional embeddings, runs the
sequence through a stack of Transformer blocks, and classifies using the CLS token (or global
average pooling / flatten, depending on the `method` argument). The key architectural novelty
is `ViT_TransFormer_Block`, which applies symmetric **U-Net-style skip connections** between
the first-half and second-half transformer blocks.

### b) Why it's used here

The ViT architecture captures global patch-level relationships via self-attention, making it
complementary to AlexNet's local convolutional features and GFNet's frequency-domain filtering.
The U-Net skip connections are a domain-inspired addition: in segmentation/classification of
spatial images, early Transformer layers tend to capture low-level geometric structure while
later layers capture semantic class information â€” the skip connections allow the semantic layers
to be informed directly by the geometric layers, mirroring U-Net's encoderâ€“decoder structure.

### c) How it works â€” Step by step

1. **Patch extraction** (`ViT_Patches`): `tf.image.extract_patches` with `patch_size=3` on the
   `9Ã—9Ã—6` input yields `3Ã—3 = 9` patches each of dimension `3*3*6 = 54`. A `Dense(projection_dim)`
   projects each to 256 dimensions.
2. **Patch encoding** (`ViT_PatchEncoder`):
   - A learnable CLS token (shape `(1, 1, 256)`) is concatenated to the front of the 9-patch
     sequence â†’ sequence length becomes `10`.
   - Learned positional embeddings (shape `(10, 256)`) are added to the sequence.
3. **Transformer stack** (`ViT_TransFormer_Block` with `num_layers = 12`):
   - 12 `ViT_TransFormer` blocks are applied sequentially.
   - The first `12//2 + 1 = 7` outputs (indices 0â€“6) are saved as `block_list`.
   - From block index 7 onward, the output of each block is **added** to the saved output of
     the mirror block: `x = block[i](x) + block_list[num_layers - i - 1]`.
   ```
   i=0  save block_list[0]
   i=1  save block_list[1]
   ...
   i=6  save block_list[6]
   i=7  x = block7(x) + block_list[12-7-1] = block7(x) + block_list[4]
   i=8  x = block8(x) + block_list[12-8-1] = block8(x) + block_list[3]
   ...
   i=11 x = block11(x) + block_list[12-11-1] = block11(x) + block_list[0]
   ```
4. **Dropout** (`TRAIN_DROPOUT_1`) on the encoded sequence after the transformer stack.
5. **Representation** (with `method='with_cls_tkn'`):
   - `ViT_Class_Token_Norm`: `LayerNormalization` then extract `x[:, 0, :]` â†’ the CLS token
     only, shape `(batch, 256)`.
6. **Classification MLP**:
   ```
   Dense(512, gelu) -> Dropout(TRAIN_DROPOUT_3) -> Dense(256, gelu) ->
   Dense(128, gelu) -> Dropout(TRAIN_DROPOUT_5) -> Dense(64, gelu) ->
   Dropout(TRAIN_DROPOUT_6)
   ```
7. **7 multi-head outputs**: `Dense(num_classes, softmax)` Ã— 7.

**Per-block Transformer structure (`ViT_TransFormer`):**
- Multi-Head Attention sub-block: `LayerNorm â†’ MHA(x, x) â†’ ViT_Weighted_add(attn_out, x)`
- FFN sub-block: `LayerNorm â†’ Dense(512, gelu) â†’ Dropout â†’ Dense(256, gelu) â†’ Dropout â†’
  ViT_Weighted_add(ffn_out, attn_input)`
- Each residual uses `ViT_Weighted_add`: `x_out = w * new + (1 - w) * old` where `w` is a
  single learned scalar, allowing the model to find the optimal blend between transformation
  and identity.

### d) ASCII Flow Diagram

```
Input (batch, 9, 9, 6)
        |
[ViT_Patches(patch_size=3)]
  -> extract_patches: (batch, 9, 54) [9 tokens of raw dim 54]
  -> Dense(256): (batch, 9, 256)
        |
[ViT_PatchEncoder]
  -> prepend CLS token: (batch, 10, 256)
  -> add positional embeddings: (batch, 10, 256)
        |
[ViT_TransFormer_Block (12 layers)]
  |--> Layer 0: MHA + FFN + weighted residuals -> save to block_list[0]
  |--> Layer 1: MHA + FFN -> save to block_list[1]
  ...
  |--> Layer 6: MHA + FFN -> save to block_list[6]
  |--> Layer 7: MHA + FFN -> x + block_list[4]  (U-Net skip)
  |--> Layer 8: MHA + FFN -> x + block_list[3]  (U-Net skip)
  ...
  |--> Layer 11: MHA + FFN -> x + block_list[0] (U-Net skip)
        |
[TRAIN_DROPOUT_1] -- (batch, 10, 256)
        |
[ViT_Class_Token_Norm]
  -> LayerNorm -> x[:, 0, :] -> (batch, 256)   [CLS token only]
        |
Dense(512, gelu) -> [TRAIN_DROPOUT_3]
        |
Dense(256, gelu) -> Dense(128, gelu)
        |
[TRAIN_DROPOUT_5] -> Dense(64, gelu) -> [TRAIN_DROPOUT_6]
        |
head_1 ... head_7  Dense(num_classes, softmax) x 7
```

### e) Worked Numerical Example

Suppose `projection_dim = 2`, `num_heads = 1`, `num_layers = 4`, `num_classes = 2`.

After patch encoding, the sequence (including CLS token) is:
```
positions:  CLS    patch0  patch1  patch2  patch3
tokens:   [[0.5, 0.5],
           [1.0, 0.0],
           [0.0, 1.0],
           [0.8, 0.2],
           [0.3, 0.7]]   shape: (5, 2)
```

With `num_layers = 4`, skip indices work as follows:
- `i <= num_layers // 2 = 2`: save block outputs for `i = 0, 1, 2` â†’ `block_list[0,1,2]`
- `i = 3`: `x = block3(x) + block_list[4 - 3 - 1] = block3(x) + block_list[0]`

After block 3's MHA + FFN, suppose the CLS token position outputs `[0.6, 0.4]`. Adding back
`block_list[0]`'s CLS token which was `[0.4, 0.3]`:
```
x_cls = [0.6 + 0.4, 0.4 + 0.3] = [1.0, 0.7]
```
After LayerNorm and the classification MLP (Dense(2, gelu)):
```
logits = [0.8, 1.2]
softmax([0.8, 1.2]) = [exp(0.8)/(exp(0.8)+exp(1.2)), exp(1.2)/(exp(0.8)+exp(1.2))]
                    = [2.23/5.55, 3.32/5.55]
                    = [0.40, 0.60]   -> predicted class = 1
```

### f) Code Walkthrough

```python
class ViT_PatchEncoder(layers.Layer):
    def call(self, patch, **kwargs):
        batch_size = tf.shape(patch)[0]
        patch_proj = self.projection(patch)  # project raw patches to projection_dim
        # Tile the (1, 1, D) CLS token to (batch, 1, D) then prepend to patches
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        x          = tf.concat([cls_tokens, patch_proj], axis=1)  # (batch, N+1, D)
        positions  = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        return x + self.position_embedding(positions)  # add learned pos embeddings
```

```python
class ViT_TransFormer_Block(layers.Layer):
    def call(self, inputs, training=None):
        block_list = []
        x = inputs
        for i in range(self.num_layers):
            x = self.Blocks[i](x, training=training)   # apply transformer block i
            if i <= self.num_layers // 2:
                block_list.append(x)                   # save first-half outputs
            else:
                # U-Net skip: add the mirror block's saved output
                x = layers.Add()([x, block_list[self.num_layers - i - 1]])
        return x
```

```python
class ViT_TransFormer(layers.Layer):
    def call(self, inputs, training=None):
        # Multi-Head Attention sub-block with learned-weight residual
        x1 = self.norm1(inputs)
        x1 = self.mha(x1, x1, training=training)   # self-attention: Q=K=V=x1
        x1 = self.add1(x1, inputs)                 # w*attn_out + (1-w)*input

        # Feed-Forward Network sub-block with learned-weight residual
        x2 = self.norm2(x1)
        x2 = self.drop1(self.dense1(x2), training=training)  # Dense(2*D, gelu) + dropout
        x2 = self.drop2(self.dense2(x2), training=training)  # Dense(D, gelu) + dropout
        return self.add2(x2, x1)                              # w*ffn_out + (1-w)*x1
```

```python
class ViT_Class_Token_Norm(layers.Layer):
    def call(self, inputs):
        x = self.norm(inputs)   # LayerNorm over the full (batch, N+1, D) sequence
        return x[:, 0, :]       # extract only the CLS token (index 0) -> (batch, D)
```

### g) Output & Interpretation

The ViT outputs 7 probability tensors of shape `(batch, num_classes)`, identical in form to the
AlexNet and GFNet outputs. The CLS token aggregates global context from all 9 patch tokens
across 12 transformer layers, with skip-connection reinforcement from early layers. This global
context makes the ViT best suited to detecting class boundaries and mixed pixels where local
convolutional features (AlexNet) or frequency-domain patterns (GFNet) may be insufficient.

### h) Limitations

- The U-Net skip logic in `ViT_TransFormer_Block.call` uses `layers.Add()` as a functional call
  inside `call()` rather than building it in `build()` â€” this creates a new `Add` layer
  instance on every call, which may cause issues with graph compilation and weight tracking.
- With `num_patches = 9` (â†’ 10 tokens including CLS) and `projection_dim = 256`, the
  attention computation is `O(10^2 * 256)` â€” trivially small, meaning much of the Transformer's
  expressivity is wasted on such a short sequence; a simpler model may suffice.
- The free-variable dependency on `projection_dim`, `num_heads`, `transformer_layers`, `dropout`,
  `patch_size`, and `num_patches` from the enclosing scope (rather than being passed as
  arguments) makes `create_vit_classifier` brittle to call in isolation without first running
  Section 7.0.
- `ViT_Weighted_add` introduces one scalar weight per residual connection: if `w` collapses to 0
  or 1 during training, the residual becomes either purely the transformation or purely the
  identity, eliminating the adaptive blending benefit.

---

## 5.7 Cosine-Annealing Learning Rate Schedules

### a) What it is

> A cosine schedule is like gently rocking a marble in a bowl â€” you start with big swings
> (high learning rate) to explore the loss landscape, and as training progresses the swings get
> smaller and smaller (the cosine curve decays), letting the marble settle into the lowest
> point it can find without overshooting.

`build_lr_callback` constructs a `tf.keras.callbacks.LearningRateScheduler` that modulates the
learning rate over training epochs according to a cosine-annealing curve. AlexNet and GFNet use
a **three-stage multi-step cosine decay** (each stage covers `epoch` epochs, so the schedule
spans `3 * epoch` epochs total â€” larger than the actual training run), while the ViT uses a
**single-stage cosine decay** spanning `epoch` epochs.

### b) Why it's used here

A fixed learning rate often leads to instability early in training (if too high) or slow
convergence late in training (if too low). Cosine annealing provides a smooth, well-tested
compromise: high initial LR for rapid early progress, low final LR for fine-grained convergence
near a minimum. The multi-step variant for AlexNet/GFNet creates a warmer second and third
phase (each restarting partway through the range), approximating cosine warm restarts (SGDR).

### c) How it works â€” Step by step

**Three-stage cosine decay (`_multistep_cosine_lrfn`):**
1. Given epoch `e`, determine which stage it falls in:
   - Stage 1: `e < steps[0]` â†’ local epoch `epoch2 = e`, stage length `epochs2 = steps[0]`
   - Stage 2: `e < steps[0] + steps[1]` â†’ `epoch2 = e - steps[0]`, `epochs2 = steps[1]`
   - Stage 3: otherwise â†’ `epoch2 = e - steps[0] - steps[1]`, `epochs2 = steps[2]`
2. Compute the cosine phase and interpolate:
   ```
   phase = pi * epoch2 / (epochs2 - 1)
   lr    = (LR_MAX - LR_MIN) * 0.5 * (1 + cos(phase)) + LR_MIN
   ```
   At `phase = 0` (start of stage): `lr = LR_MAX`; at `phase = pi` (end of stage): `lr = LR_MIN`.

**AlexNet schedule**: `LR_MAX = 0.02`, `LR_MIN = 0.005`, `STEPS = [epoch, epoch*2, epoch*3]`.
**GFNet schedule**: `LR_MAX = 6e-4`, `LR_MIN = 1e-7`, `STEPS = [epoch, epoch*2, epoch*3]`.
**ViT schedule**: `LR_MAX = 6e-4`, `LR_MIN = 1e-7`, single stage `phase = pi * e / (epoch - 1)`.

### d) ASCII Flow Diagram

```
For AlexNet / GFNet (3-stage):

LR
|
LR_MAX ---.                    .---          .---
          |  \              /     \        /
          |    \          /        \      /
LR_MIN ---|-----\--------/----------\----/-----> epoch
          0    step[0]  step[0+1]  step[0+1+2]

For ViT (1-stage):

LR
|
LR_MAX ---.
          |  \
          |    \
LR_MIN ---|-----\--------> epoch
          0    epoch-1
```

### e) Worked Numerical Example

AlexNet with `LR_MAX = 0.02`, `LR_MIN = 0.005`, `epoch = 100` (so `STEPS = [100, 200, 300]`).
At **epoch 0** (start of stage 1):
```
phase = pi * 0 / (100 - 1) = 0
lr    = (0.02 - 0.005) * 0.5 * (1 + cos(0)) + 0.005
      = 0.015 * 0.5 * 2 + 0.005 = 0.015 + 0.005 = 0.02
```
At **epoch 50** (middle of stage 1):
```
phase = pi * 50 / 99 â‰ˆ 1.587
lr    = 0.015 * 0.5 * (1 + cos(1.587)) + 0.005
      â‰ˆ 0.015 * 0.5 * (1 + (-0.0016)) + 0.005
      â‰ˆ 0.015 * 0.499 + 0.005 â‰ˆ 0.0125
```
At **epoch 99** (end of stage 1):
```
phase = pi * 99 / 99 = pi
lr    = 0.015 * 0.5 * (1 + cos(pi)) + 0.005
      = 0.015 * 0.5 * 0 + 0.005 = 0.005 = LR_MIN
```
The schedule decays smoothly from 0.02 to 0.005 over the first 100 epochs, then resets for
stage 2 (which is not reached since training only runs for `epoch = 100` epochs).

### f) Code Walkthrough

```python
def _multistep_cosine_lrfn(e, steps, lr_max, lr_min):
    """Three-stage cosine decay; each stage restarts from lr_max to lr_min."""
    if e < steps[0]:
        epoch2, epochs2 = e, steps[0]               # stage 1
    elif e < steps[0] + steps[1]:
        epoch2, epochs2 = e - steps[0], steps[1]    # stage 2
    else:
        epoch2, epochs2 = e - steps[0] - steps[1], steps[2]  # stage 3

    phase = math.pi * epoch2 / (epochs2 - 1)        # 0 -> pi over the stage
    return (lr_max - lr_min) * 0.5 * (1.0 + math.cos(phase)) + lr_min

def build_lr_callback(kind):
    """Build a LearningRateScheduler and return (callback, figure)."""
    if kind == 'alexnet':
        LR_MAX, LR_MIN = 0.02, 0.005
        STEPS = [epoch, epoch * 2, epoch * 3]
        lrfn  = lambda e: _multistep_cosine_lrfn(e, STEPS, LR_MAX, LR_MIN)

    elif kind == 'gfnet':
        LR_MAX, LR_MIN = 6e-4, 1e-7
        STEPS = [epoch, epoch * 2, epoch * 3]
        lrfn  = lambda e: _multistep_cosine_lrfn(e, STEPS, LR_MAX, LR_MIN)

    else:   # 'vit': single cosine decay over the full training run
        LR_MAX, LR_MIN = 6e-4, 1e-7
        def lrfn(e):
            phase = math.pi * e / (epoch - 1)
            return (LR_MAX - LR_MIN) * 0.5 * (1.0 + math.cos(phase)) + LR_MIN

    # Plot the schedule for visual inspection
    rng  = list(range(epoch))
    lr_y = [lrfn(x) for x in rng]
    fig  = plt.figure(figsize=(10, 4))
    plt.plot(rng, lr_y, '-o')
    plt.xlabel('Epoch', size=14)
    plt.ylabel('Learning Rate', size=14)
    plt.title(f'Learning rate schedule: {kind}')
    plt.show()
    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True), fig
```

### g) Output & Interpretation

`build_lr_callback` returns two objects: a `LearningRateScheduler` callback (passed to
`model.fit`) and a `matplotlib` figure showing the full schedule. The scheduler calls `lrfn(e)`
before each epoch and sets `optimizer.learning_rate` accordingly. The figure is later embedded
in the Excel results workbook as the third figure per model sheet.

### h) Limitations

- For AlexNet and GFNet, `STEPS = [epoch, epoch*2, epoch*3]`, which defines a 3-stage schedule
  covering `epoch*6 = 600` total epochs. Since training only runs for `epoch = 100` epochs, only
  stage 1 is ever used â€” stages 2 and 3 are entirely dead code in the schedule function.
- The ViT schedule reaches `LR_MIN` exactly at epoch `epoch - 1`; the AlexNet/GFNet single
  active stage also reaches `LR_MIN` at epoch `99`. This means the learning rate is at its
  minimum for the very last epoch, which is when `Custom_callbacks` is trying to record the best
  weights â€” a very small LR during the best-weight recording window is deliberate and desirable
  for fine-grained convergence.
- AlexNet uses Adagrad (not Adam), which has its own internal per-parameter learning-rate
  adaptation; the cosine schedule sets the global base rate, but Adagrad's accumulated gradient
  squares may compress the effective per-parameter LR significantly below the scheduled value.

---

## 5.8 Multi-Head Prediction Averaging

### a) What it is

> If 7 judges each score a skating performance on a scale of 0â€“10 for each element, and you
> want a single final score, the fairest approach is to average the 7 score sheets. That is
> exactly what `predict_multihead` does â€” it takes 7 class-probability "score sheets" and
> averages them before picking the most likely class.

`predict_multihead` is a helper function that runs inference with a multi-head model (returning
7 probability arrays), stacks them into a single tensor, averages across the head axis, and
returns both the argmax class labels and the averaged probabilities.

### b) Why it's used here

After training with `Custom_callbacks`'s staged dropout, the 7 heads have each learned to
produce probability distributions from the same shared feature representation. Averaging these
distributions is a simple and effective variance-reduction technique: individual heads may have
slight prediction biases, but their average tends to cancel random errors and produce a
more calibrated, lower-variance estimate than any single head.

### c) How it works â€” Step by step

1. Call `model.predict(x_data, verbose=0)` â€” with a 7-head model, Keras returns a **Python
   list** of 7 numpy arrays, each of shape `(N_samples, num_classes)`.
2. Stack the list along a new axis 0: `np.stack(y_pred_list, axis=0)` â†’ shape
   `(7, N_samples, num_classes)`.
3. Average across axis 0 (the head axis): `np.mean(..., axis=0)` â†’ shape
   `(N_samples, num_classes)`.
4. Argmax across axis 1 (the class axis): `np.argmax(..., axis=1)` â†’ shape `(N_samples,)`.
5. Reshape argmax labels to `(N_samples, 1)` and return both the labels and the averaged
   probabilities.

### d) ASCII Flow Diagram

```
model.predict(x_test)
        |
        v
[head_1_probs, head_2_probs, ..., head_7_probs]
  each: (N, num_classes)
        |
np.stack(axis=0)
        |
        v
stacked: (7, N, num_classes)
        |
np.mean(axis=0)
        |
        v
avg_probs: (N, num_classes)
        |
np.argmax(axis=1)
        |
        v
y_pred: (N,)  -> reshape -> (N, 1)
```

### e) Worked Numerical Example

Suppose `N = 3` samples, `num_classes = 3`, `K_HEADS = 3` (simplified).
```
head_1 predictions:  [[0.7, 0.2, 0.1],
                       [0.1, 0.8, 0.1],
                       [0.3, 0.3, 0.4]]

head_2 predictions:  [[0.6, 0.3, 0.1],
                       [0.2, 0.7, 0.1],
                       [0.2, 0.4, 0.4]]

head_3 predictions:  [[0.5, 0.3, 0.2],
                       [0.1, 0.6, 0.3],
                       [0.4, 0.2, 0.4]]
```
After stacking: shape `(3, 3, 3)`.
After averaging across heads (axis 0):
```
avg_probs:  [[0.60, 0.27, 0.13],   -> argmax = 0 (class 0)
             [0.13, 0.70, 0.17],   -> argmax = 1 (class 1)
             [0.30, 0.30, 0.40]]   -> argmax = 2 (class 2)
```
`y_pred = [0, 1, 2]`, reshaped to `[[0], [1], [2]]`.

Note: sample 2 (class 2 pred) had competing heads â€” head 2 predicted class 1 and 2 tied, head 3
predicted class 0. The average resolved the tie in favor of class 2 due to consistency from
heads 1 and 3.

### f) Code Walkthrough

```python
def predict_multihead(model, x_data):
    """Average multi-head probabilities and return (argmax labels, averaged probs)."""
    # model.predict returns a list of K arrays of shape (N, num_classes)
    y_pred_list = model.predict(x_data, verbose=0)

    # Stack into (K, N, num_classes) tensor for vectorized averaging
    y_pred_stacked   = np.stack(y_pred_list, axis=0)

    # Average across the K heads (axis 0) -> (N, num_classes)
    y_pred_avg_probs = np.mean(y_pred_stacked, axis=0)

    # Take the class with the highest averaged probability -> (N,)
    y_pred_argmax    = np.argmax(y_pred_avg_probs, axis=1)

    # Return both the integer class labels (reshaped for sklearn compatibility) and the soft probs
    return y_pred_argmax.reshape(-1, 1), y_pred_avg_probs
```

### g) Output & Interpretation

`predict_multihead` returns two objects:
- `y_pred_argmax` of shape `(N, 1)`: the final integer class label for each test patch (passed
  to sklearn's `accuracy_score`, `cohen_kappa_score`, `confusion_matrix`).
- `y_pred_avg_probs` of shape `(N, num_classes)`: the soft probability vector (available for
  downstream uncertainty quantification or calibration analysis, though not used further in this
  notebook).

### h) Limitations

- Simple averaging weights all 7 heads equally. If some heads converge to clearly better
  solutions than others, a weighted average (e.g., weighted by per-head validation accuracy)
  could improve results.
- The averaged probabilities are **not** the same as a proper Bayesian predictive distribution â€”
  they lack the uncertainty decomposition into aleatoric and epistemic components that would
  require, e.g., tracking both the mean and variance across heads.
- `model.predict` runs inference on the full `x_test` in one call; for very large test sets
  this may exhaust GPU memory. Batching (via `batch_size` arg to `model.predict`) should be
  specified explicitly for production use.

---

## 5.9 Performance Measures & Visualization

### a) What it is

> After a school exam, the teacher doesn't just report the average score â€” they also make a
> seating chart of who got confused with whom (the confusion matrix), a subject-by-subject
> breakdown (the classification report), and a summary scorecard. `performance_meausures` does
> exactly this: four panels in a single figure, covering all aspects of the model's performance.

`performance_meausures` (name preserved with the original typo for backward compatibility) is a
visualization function that computes accuracy, Cohen's Kappa, and a confusion matrix; renders a
4-panel figure (classification report heatmap, confusion matrix heatmap, score summary, and
parameter count); optionally saves the figure; and returns it for downstream Excel export.
`history_figure` is a companion function that plots per-head training and validation accuracy
and loss curves from the Keras `History` object.

### b) Why it's used here

After training and evaluation, comprehensive metric visualization is essential both for
diagnosing model behavior (which classes are confused?) and for producing figures for a research
paper. The function generates publication-ready plots that are simultaneously embedded into the
Excel results workbook and returned as Matplotlib figures for programmatic use in `run_training`.

### c) How it works â€” Step by step

**`performance_meausures`:**
1. Compute `accuracy_score(y_test, y_pred)` â€” overall classification accuracy.
2. Compute `cohen_kappa_score(y_test, y_pred)` â€” kappa accounts for chance agreement.
3. Compute `confusion_matrix(y_test, y_pred).astype('int32')` â€” `C[i,j]` = number of pixels
   of class `i` predicted as class `j`.
4. Compute `classification_report(y_test, y_pred, output_dict=True)` and convert to a
   DataFrame â€” per-class precision, recall, F1-score, and support.
5. Build a `GridSpec(2, 2)` figure with width ratios `[1, 3]` and height ratios `[7, 1]` for
   a 4-panel layout.
6. Panel 1 (top-left): seaborn heatmap of the classification report DataFrame.
7. Panel 2 (top-right): seaborn heatmap of the confusion matrix.
8. Panel 3 (bottom-left): seaborn heatmap of accuracy, kappa, and training time.
9. Panel 4 (bottom-right): seaborn heatmap of parameter counts.
10. Optionally save to `folder_path/Results/<prefix><train_percent>% ps_<P_S> Performance Measure.png`.
11. Return the figure.

**`history_figure`:**
1. Extract `head_1_accuracy`, `val_head_1_accuracy`, `loss`, `val_loss` from
   `history.history` (falling back to empty lists if keys are absent).
2. Plot accuracy on the left Y-axis (dual-axis) and loss on the right Y-axis.
3. Return the figure.

### d) ASCII Flow Diagram

```
y_test, y_pred, training_time, param counts
        |
        v
accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
        |
        v
+----------- GridSpec(2 rows x 2 cols) -----------+
|  [ax1: classification report heatmap]            |
|                      [ax2: confusion matrix]     |
|  [ax3: accuracy / kappa / time summary]          |
|                      [ax4: param counts]         |
+---------------------------------------------------+
        |
(optional) save to Results/
        |
return fig
```

### e) Worked Numerical Example

Suppose `num_classes = 3`, `y_test = [0, 1, 2, 0, 1]`, `y_pred = [0, 1, 0, 0, 1]`.

```
accuracy_score       = 4/5 = 0.80
confusion_matrix:
         pred_0  pred_1  pred_2
actual_0 [  2,     0,     0 ]
actual_1 [  0,     2,     0 ]
actual_2 [  1,     0,     0 ]

classification_report (class 2 has 0 predictions -> precision undefined/0):
         precision  recall  f1-score  support
class 0    0.67     1.00     0.80       2
class 1    1.00     1.00     1.00       2
class 2    0.00     0.00     0.00       1

cohen_kappa = (P_o - P_e) / (1 - P_e)
P_o = 0.80,  P_e = (chance agreement from marginals)
     = (3/5 * 2/5) + (2/5 * 2/5) + (0/5 * 1/5)
     = 0.24 + 0.16 + 0.00 = 0.40
kappa = (0.80 - 0.40) / (1 - 0.40) = 0.40 / 0.60 â‰ˆ 0.667
```

These values would appear as annotated cells in the heatmaps within the 4-panel figure.

### f) Code Walkthrough

```python
def performance_meausures(y_test, y_pred, tt, *parameters_summary, folder_path=None):
    """Compute and visualize classification metrics; returns the figure for export."""
    Total_params, Trainable_params, Non_trainable_params = parameters_summary
    accuracy = accuracy_score(y_test, y_pred)
    kappa    = cohen_kappa_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred).astype('int32')
    cr       = classification_report(y_test, y_pred, output_dict=True)
    df_cr    = pd.DataFrame(cr).T           # rows = classes, cols = precision/recall/f1

    # Summary tables for the bottom panels
    df_score = pd.DataFrame({
        'accuracy score: ':  [accuracy],
        'Cohen_Kappa score: ': [kappa],
        "Training Time: ":  [tt],
    }).T

    # 2x2 GridSpec: top row is tall (7:1), right column is wide (1:3)
    spec = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[1, 3],
                              wspace=0.5, hspace=0.5, height_ratios=[7, 1])
    fig = plt.figure(figsize=(24, 10))

    ax1 = fig.add_subplot(spec[0])         # top-left
    ax1.set_title('classification report')
    sns.heatmap(df_cr, cmap='Blues', cbar=False, annot=True, fmt=' .5g', ax=ax1)

    ax2 = fig.add_subplot(spec[1])         # top-right
    ax2.set_title('confusion matrix')
    sns.heatmap(cm, cmap='Blues', cbar=False, annot=True, fmt=' .5g', ax=ax2)

    ax3 = fig.add_subplot(spec[2])         # bottom-left: scores
    sns.heatmap(df_score, cmap='Blues', cbar=False, annot=True, fmt=' .5g', ax=ax3)
    ax3.set_xticks([])

    ax4 = fig.add_subplot(spec[3])         # bottom-right: param counts
    sns.heatmap(df_summary, cmap="Blues", cbar=False, annot=True, fmt=' .10g', ax=ax4)
    ax4.set_xticks([])

    if folder_path:
        path = folder_path + "Results/" + str(train_percent) + "% ps_" + str(P_S) + " Performance Measure.png"
        fig.savefig(path)

    return fig   # critical: allows run_training to pass this fig to export_training_sheet
```

### g) Output & Interpretation

`performance_meausures` returns a Matplotlib `Figure` that is:
1. Displayed inline in the Colab notebook.
2. Saved as a PNG to `TRAINING_RESULTS_DIR/Results/`.
3. Embedded in the corresponding model's Excel worksheet by `export_training_sheet`.

**Interpretation guide:**
- The **confusion matrix** (top-right) should be predominantly diagonal; large off-diagonal
  values reveal which class pairs are most frequently confused â€” important for understanding
  spectral overlap between land-cover classes.
- **Cohen's Kappa** accounts for the possibility that a high accuracy was achieved by chance
  (e.g., if one class dominates). Values above 0.80 are generally considered "strong agreement"
  in remote sensing literature.
- **Precision, recall, F1** per class (top-left heatmap) identify whether rare classes are
  being missed (low recall) or whether the model over-predicts common classes (low precision).

### h) Limitations

- `fmt=' .5g'` in the classification report heatmap may truncate small floating-point values
  or display them in scientific notation, reducing readability for metrics near 0 or 1.
- The figure has a fixed `figsize=(24, 10)` that may not scale well for datasets with many
  classes (e.g., a 20-class confusion matrix would be very small in the top-right panel at
  this figure size).
- `df_summary` (the parameter count table) is defined inside `performance_meausures` but
  referenced in `ax4`'s `sns.heatmap` call â€” this variable is in scope only because it's
  created just before the `sns.heatmap` call in the same function body.
- `plot_accuracy_loss_curve` (the single-head version defined in Section 3.1) is defined but
  never called in the main training loop; `history_figure` (Section 8.3) is used instead for
  multi-head models, and `plot_accuracy_loss_curve` is now dead code.

---

## 5.10 Excel Export Pipeline

### a) What it is

> After every experiment in a research lab, a careful scientist fills in a lab notebook. This
> pipeline is the digital equivalent: it opens (or creates) a master Excel workbook, gives
> each trained model its own worksheet, writes all the numerical results in a table, pins
> the performance figures onto the sheet like photographs, and finally writes a summary row to
> the cover page â€” all automatically, without the researcher lifting a pen.

The Excel export pipeline consists of four functions â€” `ensure_workbook`, `autosize_columns`,
`fig_to_buffer`, and `export_training_sheet` â€” plus integration in `run_training`. Together
they write per-model results (metrics, classification reports, confusion matrices, and three
embedded figures) into a structured `.xlsx` workbook using `openpyxl`.

### b) Why it's used here

A single Excel workbook consolidating all model results makes comparison straightforward and
provides a portable, shareable artifact that does not require a Python environment to inspect.
Embedding the figures directly into the worksheet (rather than as external files) ensures
everything travels together and the workbook is self-contained.

### c) How it works â€” Step by step

1. **`ensure_workbook(path)`**: if the workbook `.xlsx` file does not exist, creates a new
   `openpyxl.Workbook` with a single "Summary" sheet and saves it. If it exists, loads and
   returns it.
2. **`autosize_columns(ws)`**: iterates over all columns in the worksheet, computes the maximum
   string length of any cell value in each column, and sets the column width to
   `min(max_length + 2, 40)` characters.
3. **`fig_to_buffer(fig)`**: renders a Matplotlib figure to an in-memory PNG `BytesIO` buffer
   using `fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)`.
4. **`export_training_sheet(wb, model_name, summary_row, report_df, cm, figs)`**:
   - Creates a new sheet named `model_name[:31]` (Excel sheet name limit is 31 characters).
   - Writes the `summary_row` dict (key, value pairs) to columns A and B, rows 1 through
     `len(summary_row)`.
   - Writes the classification report DataFrame starting at row `len(summary_row) + 3`.
   - Writes the confusion matrix DataFrame to the same starting row but offset to column 10.
   - Embeds three figures at fixed anchors: `'L2'` (accuracy/loss curve), `'L30'`
     (performance measures), `'L58'` (LR schedule).
   - Calls `autosize_columns` on the sheet.
5. **`run_training`** calls `export_training_sheet` after evaluation, passing `[curve_fig,
   perf_fig, lr_fig]` as the three figures.
6. **Section 9.0** writes the consolidated `summary_df` DataFrame back to the "Summary" sheet
   after all models are trained, saves the workbook, and writes the model registry JSON.

### d) ASCII Flow Diagram

```
run_training(model_name, spec, workbook)
        |
        v
generate summary_row dict (metrics + paths)
generate report_df (classification report DataFrame)
generate cm (confusion matrix numpy array)
generate curve_fig, perf_fig, lr_fig (Matplotlib figures)
        |
export_training_sheet(workbook, model_name, ...)
        |
+-- Create sheet: model_name[:31] --+
|                                    |
|  Rows 1..N: summary_row key/val   |
|  Rows N+3..: report_df table      |
|  Rows N+3.. col 10: cm table      |
|  Anchor L2:  curve_fig (PNG)      |
|  Anchor L30: perf_fig  (PNG)      |
|  Anchor L58: lr_fig    (PNG)      |
|  autosize_columns()               |
+------------------------------------+
        |
workbook.save(TRAINING_WORKBOOK)
```

### e) Worked Numerical Example

Suppose `model_name = 'AlexNet_CNN_MultiHead'` and the `summary_row` has 9 key-value pairs:
```
{'model_name': 'AlexNet_CNN_MultiHead', 'test_accuracy': 0.9312, ...}
```
The function:
- Creates sheet `'AlexNet_CNN_MultiHead'` (truncated to first 31 chars: `'AlexNet_CNN_MultiHead'`, OK).
- Writes rows 1â€“9: e.g., `A1='model_name', B1='AlexNet_CNN_MultiHead'`, `A2='best_model_path'`,
  `B2='/content/drive/.../AlexNet_CNN_MultiHead_best.keras'`, ...
- `row0 = 9 + 3 = 12`. Classification report rows start at row 12.
- Confusion matrix columns start at column 10 (column J) from row 12.
- `XLImage` objects for the three figures are anchored at cells L2, L30, L58
  (column L = column 12, rows 2, 30, 58).

For `autosize_columns`: if the widest value in column B is `'/content/drive/My Drive/...'`
(say 60 chars), it would normally be `60 + 2 = 62`, but capped at 40. Column A (the keys) are
all short (e.g., `'test_accuracy'` = 13 chars) â†’ width = 15.

### f) Code Walkthrough

```python
def ensure_workbook(path):
    """Create a new workbook with a Summary sheet, or load an existing one."""
    if path.exists():
        return load_workbook(path)   # open existing file
    wb = Workbook()
    ws = wb.active
    ws.title = 'Summary'             # rename the default sheet
    wb.save(path)
    return wb

def fig_to_buffer(fig):
    """Serialise a Matplotlib figure to an in-memory PNG BytesIO buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)   # rewind so openpyxl can read from the start
    return buf

def export_training_sheet(wb, model_name, summary_row, report_df, cm, figs):
    ws  = wb.create_sheet(title=model_name[:31])   # Excel sheet name limit
    row0 = len(summary_row) + 3                    # offset for metric tables below summary

    # Write summary dict as a 2-column key-value table
    for idx, (key, value) in enumerate(summary_row.items(), start=1):
        ws.cell(row=idx, column=1, value=key)
        ws.cell(row=idx, column=2, value=value)

    # Write classification report DataFrame
    for r_idx, row in enumerate(dataframe_to_rows(report_df.reset_index(), index=False, header=True), start=row0):
        for c_idx, val in enumerate(row, start=1):
            ws.cell(row=r_idx, column=c_idx, value=val)

    # Write confusion matrix offset to column 10 (avoids overlapping with the report)
    for r_idx, row in enumerate(dataframe_to_rows(pd.DataFrame(cm), index=False, header=False), start=row0):
        for c_idx, val in enumerate(row, start=10):
            ws.cell(row=r_idx, column=c_idx, value=int(val))

    # Embed three figures as PNG images anchored to specific cells
    for anchor, fig in [('L2', figs[0]), ('L30', figs[1]), ('L58', figs[2])]:
        img        = XLImage(fig_to_buffer(fig))
        img.anchor = anchor
        ws.add_image(img)

    autosize_columns(ws)
```

### g) Output & Interpretation

The output is a multi-sheet `.xlsx` file at `TRAINING_WORKBOOK` with one sheet per model plus
a "Summary" sheet. Each model sheet contains all numerical results in the first columns and the
three embedded PNG figures to the right. The Summary sheet has one row per model with all
headline metrics side by side, enabling direct cross-model comparison.

### h) Limitations

- Fixed figure anchors (`L2`, `L30`, `L58`) assume each figure occupies approximately 28 rows
  of cell height. If the sheet layout changes (e.g., more summary rows), figures will overlap
  with data cells.
- `fig_to_buffer` renders at `dpi=200` â€” this produces large PNG files embedded in the
  workbook, which can make the `.xlsx` file very large (potentially tens of MBs) when three
  high-resolution figures are embedded per model sheet.
- If any model training fails mid-run, `export_training_sheet` may not be called for that model,
  leaving the workbook with partial results â€” there is no error recovery or partial-save
  mechanism.
- `dataframe_to_rows` includes `None` values for empty cells; `ws.cell(value=None)` is valid
  but produces blank cells, which is fine for xlsx but may cause issues if the workbook is
  later processed by automated scripts expecting non-null values.

---

## 6. Results & Comparisons

**Results not directly shown in the provided notebook execution.** The notebook defines the
full training, evaluation, and export pipeline but does not include inline output cells with
final metric values. The following describes the structure of results that the pipeline produces
and the metrics that would be available in the output Excel workbook.

### Metric Structure

Each model is evaluated on the held-out test set (25% of all labeled patches, stratified by
class). The following metrics are computed and stored per model:

| Metric | Description | Location in Workbook |
|---|---|---|
| `test_accuracy` | Overall accuracy (correct/total) | Summary sheet, model sheet row |
| `cohen_kappa` | Kappa coefficient (accounts for chance agreement) | Summary sheet, model sheet row |
| `training_time_sec` | Wall-clock seconds for `model.fit` | Summary sheet, model sheet row |
| `total_params` | Total model parameter count | Summary sheet, model sheet row |
| `trainable_params` | Trainable parameter count | Summary sheet, model sheet row |
| `non_trainable_params` | Non-trainable (e.g., BatchNorm stats) parameter count | Summary sheet, model sheet row |
| Per-class precision | From `classification_report` | Model sheet, classification report table |
| Per-class recall | From `classification_report` | Model sheet, classification report table |
| Per-class F1-score | From `classification_report` | Model sheet, classification report table |
| Confusion matrix | `C[i,j]` = class i predicted as j | Model sheet, columns J onward |

### Architecture Comparison (Design Properties)

| Property | AlexNet_CNN | GFNet | ViT_UNet |
|---|---|---|---|
| Core operation | Local conv (3Ã—3) | Global filter (FFT) | Multi-head self-attention |
| Depth | 5 conv + 3 dense | 12 GF blocks + MLP | 12 transformer blocks |
| Positional encoding | None (spatial preserved) | Learned patch embeddings | Learned + CLS token |
| Skip connections | None | None (DropPath only) | U-Net symmetric skips |
| Dropout style | Staged channel-shift | Staged channel-shift | Staged channel-shift |
| Optimizer | Adagrad (lr=0.01) | Adam (lr=3e-6) | Adam (lr=3e-6) |
| LR schedule | Cosine (0.02 â†’ 0.005) | Cosine (6e-4 â†’ 1e-7) | Cosine (6e-4 â†’ 1e-7) |
| Output heads | 7 | 7 | 7 |
| Patch input | 9Ã—9Ã—6 | 9 tokens Ã— 512 | 10 tokens Ã— 256 (+ CLS) |

### Smoke Check Verification

Section 10.0 confirms that all three saved models:
- Load correctly using `CUSTOM_OBJECTS` for deserialization.
- Return `isinstance(outputs, list) == True` with `len(outputs) == 7`.
- Produce output tensors of shape `(4, num_classes)` on a 4-sample mini-batch.

---

## 7. Academic Paper Summary

### Problem Statement

Multi-class land-use and land-cover (LULC) classification from hyperspectral imagery represents
a fundamental challenge in remote sensing, wherein each spatial pixel must be assigned to one
of several semantically meaningful categories based on its multispectral reflectance profile and
spatial context. This work investigates three deep learning architectures â€” a convolutional
neural network (AlexNet-CNN), a frequency-domain global filter network (GFNet), and a Vision
Transformer with U-Net-style skip connections (ViT-UNet) â€” evaluated on a six-band hyperspectral
image of dimensions 330Ã—307 pixels under a patch-based classification paradigm. Each model
incorporates a multi-head output strategy and a novel staged channel-shift dropout curriculum
to improve generalization.

### Methodology

**Preprocessing.** Raw spectral data was loaded as a `330 Ã— 307 Ã— 6` array and subjected to
per-band min-max normalization to scale each spectral channel independently to the unit interval.
Labeled pixels (excluding background class 0) were extracted as `9 Ã— 9 Ã— 6` spatial patches
via edge-padded sliding-window extraction, yielding input tensors of shape `(N, 9, 9, 6)`.
Patches were partitioned into training (75%) and test (25%) sets using stratified random
splitting to preserve class proportions in both subsets.

**Staged Channel-Shift Dropout.** A custom dropout regularization scheme (`Dropout_Train`) was
introduced wherein a contiguous slice of `rate Ã— C` channels (where `C` is the channel count
and `rate = 0.25`) is deterministically zeroed during each training "shift," cycling through
all `1/rate = 4` non-overlapping slices before transitioning to standard stochastic dropout.
This staged curriculum is orchestrated by a custom Keras callback (`Custom_callbacks`) that
rebuilds the model's dropout layers mid-training upon achieving a validation accuracy of 0.985
for a sustained minimum of 20 epochs, ensuring each channel group is trained under suppression
before the model is fine-tuned with standard dropout.

**Multi-Head AlexNet CNN.** The first architecture is an AlexNet-inspired five-layer
convolutional backbone (filter counts: 96, 256, 384, 384, 256; all 3Ã—3 kernels with same
padding; a single max-pool stride of 2) followed by three fully connected layers (4096, 1024,
256 units, all ReLU), and `K = 7` independent softmax output heads. The model is trained with
the Adagrad optimizer under a three-stage cosine-annealing learning rate schedule decaying from
0.02 to 0.005.

**GFNet (Global Filter Network).** The second architecture replaces self-attention with a
learned global filter in the 2-D frequency domain. Input patches are tokenized into `9` patch
tokens (3Ã—3 non-overlapping sub-patches, each 54-dimensional) and projected to a
512-dimensional embedding. Twelve stacked GFNet blocks each apply Layer Normalization, a
real-valued 2-D FFT (`tf.signal.rfft2d`), element-wise complex multiplication by a learned
weight tensor, inverse FFT (`tf.signal.irfft2d`), and a two-layer GELU MLP, with stochastic
depth regularization on both residual paths. Global average pooling over the token sequence
produces the final representation, fed to 7 softmax output heads. Training employs the Adam
optimizer with a cosine-annealing schedule decaying from `6e-4` to `1e-7`.

**Vision Transformer with U-Net Skip Connections (ViT-UNet).** The third architecture follows
the standard ViT pipeline: patch extraction (3Ã—3 sub-patches), projection to 256 dimensions,
prepending a learnable CLS token, and learned positional embeddings over 10 positions. Twelve
transformer blocks â€” each comprising multi-head self-attention (4 heads, key dimension 256) and
a feed-forward network with GELU activations â€” are stacked with a critical modification: the
output of block `i` (for `i > num_layers // 2`) is element-wise-added to the saved output of
block `num_layers - i - 1`, implementing symmetric skip connections mirroring the U-Net
encoder-decoder architecture. The CLS token is extracted after final layer normalization and
passed through a four-layer classification MLP before 7 softmax output heads. Each residual
connection within the transformer blocks uses a learned scalar weight `w`, implementing an
adaptive blend `w * transform(x) + (1 - w) * x`.

**Inference and Evaluation.** At inference time, the 7 output probability distributions from
each model are averaged element-wise, and the argmax of the resulting averaged distribution is
taken as the final class prediction. Models are evaluated using overall accuracy, Cohen's Kappa
coefficient, per-class precision, recall, and F1-score, and a full confusion matrix. The best
model weights are selected based on peak validation accuracy during the final standard-dropout
training phase.

### Experimental Setup

Experiments were conducted on a single hyperspectral image with 6 spectral bands and an image
resolution of 330Ã—307 pixels. The number of labeled pixels and the specific land-cover classes
are dataset-dependent (determined at runtime as `num_classes = len(np.unique(y)) - 1`). All
models were trained for a maximum of 100 epochs with a batch size of 128, patch size of 9, and
dropout rate of 0.25. Random seeds were fixed (`seed = 1337`) across NumPy, Python's `random`
module, and TensorFlow for reproducibility. Results were stored in an Excel workbook with one
sheet per model and a consolidated summary sheet.

### Results Summary

The three architectures represent a progression from locally-receptive (AlexNet-CNN) to
frequency-globally-receptive (GFNet) to attention-globally-receptive (ViT-UNet) feature
extraction. The multi-head averaging strategy reduces per-prediction variance across all three
models. The staged channel-shift dropout enforces that no channel group is permanently avoided
during training, providing structured coverage of the representational capacity that standard
random dropout may neglect for short-duration experiments. Detailed numerical results (accuracy,
kappa, per-class metrics, confusion matrices) are available in the generated Excel workbook at
`TRAINING_WORKBOOK`.

### Conclusion

This work demonstrates the feasibility of training three fundamentally different deep learning
architectures â€” convolutional, frequency-domain, and attention-based â€” on small
(`9 Ã— 9 Ã— 6`) hyperspectral patches under a unified multi-head training strategy and a novel
staged channel-shift dropout curriculum. The modular architecture of the `MODEL_SPECS`
registry and the `run_training` orchestration function enable systematic, reproducible
multi-model benchmarking with a single codebase. Limitations include the restriction to a
single scene for training and evaluation (risking scene-specific overfitting), the absence of
explicit uncertainty quantification in the inference step (only point predictions are produced
from the averaged heads), and the unverified effectiveness of staged channel-shift dropout
versus standard random dropout in a controlled ablation. Future work should include multi-scene
evaluation, calibration analysis of the averaged head probabilities, and a formal ablation
study comparing the staged dropout curriculum against standard dropout baselines.

---

## 8. References

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep
Convolutional Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 25.

[2] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global Filter Networks for Image
Classification. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 980â€“993.

[3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T.,
... & Houlsby, N. (2020). An Image is Worth 16Ã—16 Words: Transformers for Image Recognition
at Scale. *International Conference on Learning Representations (ICLR)*, 2021.

[4] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for
Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention
(MICCAI)*, 9351, 234â€“241.

[5] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... &
Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing
Systems (NeurIPS)*, 30.

[6] Loshchilov, I., & Hutter, F. (2016). SGDR: Stochastic Gradient Descent with Warm Restarts.
*International Conference on Learning Representations (ICLR)*, 2017.

[7] Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and
Psychological Measurement*, 20(1), 37â€“46.

[8] Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). Deep Networks with
Stochastic Depth. *European Conference on Computer Vision (ECCV)*, 9908, 646â€“661.

[9] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image
Recognition. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770â€“778.

[10] Gao, Z., Xie, J., Wang, Q., & Li, P. (2019). Global Second-Order Pooling Convolutional
Networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 3024â€“3033.

[11] Richards, J. A. (2013). *Remote Sensing Digital Image Analysis: An Introduction* (5th ed.).
Springer.

[12] Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online
Learning and Stochastic Optimization. *Journal of Machine Learning Research*, 12, 2121â€“2159.
(Adagrad)

[13] Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization.
*International Conference on Learning Representations (ICLR)*, 2015.

