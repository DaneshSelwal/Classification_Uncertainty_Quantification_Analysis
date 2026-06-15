# CreDE: Credal Deep Ensemble Uncertainty Quantification for Multispectral Land-Cover Classification

## 1. Title & Overview

This notebook implements **CreDE (Credal Deep Ensemble)**, a method for quantifying
prediction uncertainty when classifying every pixel of a 6-band multispectral image
into land-cover classes. Instead of training a single classifier and trusting its
output blindly, the notebook loads **multiple independently-trained deep learning
models** (an ensemble) for each of three architectures — a CNN (AlexNet-style), a
Global Filter Network (GFNet), and a Vision Transformer with a U-Net-style structure
(ViT-UNet) — runs every model over the full scene, and then combines their
predictions using **credal set theory** (bounds on probability) to split the total
uncertainty into two interpretable parts: **aleatoric uncertainty** (uncertainty from
noisy/ambiguous data itself) and **epistemic uncertainty** (uncertainty from the
model "not being sure" — disagreement between ensemble members). The notebook then
produces large spatial maps showing where each type of uncertainty is high, bar charts
of how many pixels fall above chosen thresholds, a master CSV summary across all three
architectures, and a formatted Excel report bundling everything together.

**Who this document is for**: a reader who used AI assistance to write this notebook
and now wants to deeply understand *how and why* every piece works — both to learn the
underlying uncertainty-quantification theory and to be able to describe the method
formally in a research paper.

---

## 2. Table of Contents

- [Environment & Dependencies](#environment--dependencies)
- [Data & Problem Setup](#data--problem-setup)
- [Method: Multispectral Data Loading & Normalisation](#method-multispectral-data-loading--normalisation)
- [Method: Spatial Patch Extraction](#method-spatial-patch-extraction)
- [Method: Labeled Patch Extraction & Train/Test Split](#method-labeled-patch-extraction--traintest-split)
- [Method: Custom Keras Layers for Model Deserialisation](#method-custom-keras-layers-for-model-deserialisation)
- [Method: Ensemble Path Discovery](#method-ensemble-path-discovery)
- [Method: Homogeneous Ensemble Evaluation — CreDE Credal Bounds](#method-homogeneous-ensemble-evaluation--crede-credal-bounds)
- [Method: 6-Panel Spatial Uncertainty Mapping](#method-6-panel-spatial-uncertainty-mapping)
- [Method: Excel Report Generation](#method-excel-report-generation)
- [Method: Master Evaluation Loop](#method-master-evaluation-loop)
- [Results & Comparisons](#results--comparisons)
- [Academic Paper Summary](#academic-paper-summary)
- [References](#references)

---

## 3. Environment & Dependencies

| Library / Module | Purpose |
|---|---|
| `os`, `gc`, `glob`, `pathlib.Path` | Filesystem navigation, pattern-matching for model checkpoint files, and manual garbage collection to control RAM usage |
| `numpy` | Core array operations — normalisation, statistics (mean, variance, entropy), reshaping |
| `pandas` | Reading the flat CSV files containing image and label data; building summary tables |
| `seaborn` | Sets a clean plotting style (`whitegrid`) used by all matplotlib figures |
| `matplotlib.pyplot` | Drawing the multi-panel spatial uncertainty figures |
| `matplotlib.colors.ListedColormap` | Building discrete colour maps so each land-cover class gets a fixed colour |
| `matplotlib.patches.Patch` | Creating custom legend entries (e.g. "Certain" vs "Uncertain") |
| `tensorflow` / `tensorflow.keras.layers` | Loading and running the deep learning ensemble models, and defining custom layers |
| `sklearn.model_selection.train_test_split` | Splitting labeled pixels into train/test sets (stratified by class) |
| `sklearn.metrics` (`accuracy_score`, `cohen_kappa_score`, `confusion_matrix`, `classification_report`, `f1_score`, `log_loss`) | Imported for potential classification-quality evaluation (metric functions are defined but the master loop in this notebook focuses on uncertainty statistics rather than printing these per-model) |
| `openpyxl` (`Workbook`, styling classes, `Image`) | Building a formatted `.xlsx` report with styled tables and embedded plot images |
| `google.colab.drive` | Mounts Google Drive so the notebook (running on Colab) can read data and saved model checkpoints from persistent storage |

> **Note:** TensorFlow version is printed at the end of the imports cell
> (`print("TensorFlow:", tf.__version__)`) but the actual version string is not shown
> in the provided notebook output — it will appear when the cell is executed.

---

## 4. Data & Problem Setup

**Dataset.** The notebook works with a multispectral remote-sensing scene:

- **Shape**: `H = 330` rows × `W = 307` columns × `B = 6` spectral bands.
- **Source files**: a flat CSV `data.csv` (pixel values) and a flat CSV `ref.csv`
  (integer class labels), both reshaped into `(H, W, B)` and `(H, W)` arrays
  respectively.
- **Classes**: the number of land-cover classes is computed automatically as the
  count of unique positive label values: `num_classes = unique(y_img[y_img > 0]).size`.
  Label value `0` is treated as "unlabeled / background" and excluded from training
  and metric pixels (but every pixel, including label-0 pixels, is still classified
  during full-scene inference).

**Problem type.** This is a **pixel-wise multiclass classification** problem
(supervised land-cover classification), where each pixel is represented not just by
its own 6 spectral values but by a small spatial neighbourhood (a "patch") around it,
giving the models local spatial context.

**Preprocessing steps (in the order performed in the notebook):**

1. Load the raw pixel CSV and label CSV, reshape to `(H, W, B)` and `(H, W)`.
2. **Per-band min-max normalisation** to `[0, 1]` — each of the 6 bands is normalised
   independently using its own min and max across the whole scene.
3. **Edge-pad** the normalised image by `PATCH_SIZE // 2 = 4` pixels on each side
   (using `mode="edge"`, i.e. replicate the border pixels).
4. **Extract a `9×9×6` patch centered on every pixel** of the scene — this produces
   `scene_pixels_scaled` with shape `(330 × 307, 9, 9, 6) = (101,310, 9, 9, 6)`, used
   for full-scene inference.
5. Separately, **extract patches only for labeled pixels** (`y > 0`), producing `X`
   (patches) and `y_labels` (the label minus 1, so classes are zero-indexed).
6. **Stratified train/test split**: 75% train, 25% test (`TRAIN_PERCENT = 0.75`,
   `random_state = SEED = 42`), stratified by class so the class balance is preserved
   in both splits. The test labels are also one-hot encoded (`y_test_cat`).

> **Note:** the train/test split and `x_train_full` / `y_train_full` are computed but
> not directly consumed later in this notebook — this notebook focuses on
> **inference and uncertainty quantification** using pre-trained ensemble checkpoints
> rather than training. The split is likely retained from a training notebook for
> consistency / reproducibility (so the same `x_test` can be cross-checked against
> CreDE outputs at the labeled pixel locations).

---

## Method: Multispectral Data Loading & Normalisation

### a) What it is

> Think of this like taking six different black-and-white photographs of the same
> landscape — one for each spectral band — and then adjusting the brightness/contrast
> of *each photo independently* so that its darkest pixel becomes pure black (0) and
> its brightest pixel becomes pure white (1). This way, no single band dominates just
> because it happens to have larger raw numbers.

### b) Why it's used here

Different spectral bands (e.g., visible light vs. infrared) can have wildly different
raw value ranges. Neural networks train and predict much more reliably when all input
features are on a comparable scale. Per-band min-max normalisation puts every band
into `[0, 1]` without mixing information across bands.

### c) How it works — Step by step

1. Read the pixel CSV into a flat table and reshape it to `(H, W, B)`.
2. Read the label CSV into a flat table and reshape it to `(H, W)`.
3. For each band `b` from `0` to `B-1`:
   ```
   band_min = min(x[:, :, b])
   band_max = max(x[:, :, b])
   x_norm[:, :, b] = (x[:, :, b] - band_min) / max(band_max - band_min, 1e-8)
   ```
4. The `max(..., 1e-8)` guards against division by zero if a band happens to be
   perfectly constant.
5. Return the normalised image `x_norm` (float32, values in `[0, 1]`) and the integer
   label map `y`.

### d) ASCII Flow Diagram

```
data.csv (flat pixel values)         ref.csv (flat integer labels)
        |                                       |
        v                                       v
reshape -> (H, W, B)                   reshape -> (H, W)
        |
        v
for each band b in 0..B-1:
    band = x[:, :, b]
    x_norm[:, :, b] = (band - min(band)) / max(max(band) - min(band), 1e-8)
        |
        v
x_norm (H, W, B), values in [0, 1]   +   y (H, W) integer labels
```

### e) Worked Numerical Example

Suppose `B = 1` band and the scene is just a `2×2` image (instead of `330×307×6`),
with raw pixel values:

```
band values:
[ 10  40 ]
[ 20  60 ]
```

- `band_min = 10`, `band_max = 60`
- `range = max(60 - 10, 1e-8) = 50`
- Normalise each pixel: `(value - 10) / 50`

```
(10-10)/50 = 0.0
(40-10)/50 = 0.6
(20-10)/50 = 0.2
(60-10)/50 = 1.0
```

Result:

```
x_norm:
[ 0.0  0.6 ]
[ 0.2  1.0 ]
```

The darkest original pixel (10) becomes 0.0, the brightest (60) becomes 1.0, and the
others fall proportionally in between.

### f) Code Walkthrough

```python
def load_multispectral_6band(data_path, label_path, height, width, bands):
    # Read the flat pixel CSV and reshape into a (H, W, B) image cube
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(height, width, bands)

    # Read the flat label CSV and reshape into a (H, W) label map
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(height, width)

    # Allocate an output array of the same shape, same dtype
    x_norm = np.empty_like(x, dtype=np.float32)

    # Normalise each spectral band independently to [0, 1]
    for b in range(bands):
        band = x[:, :, b]
        band_min, band_max = np.min(band), np.max(band)
        # 1e-8 prevents division by zero if band_max == band_min
        x_norm[:, :, b] = (band - band_min) / max(band_max - band_min, 1e-8)

    return x_norm, y

print("Loading multispectral data...")
x_img, y_img = load_multispectral_6band(DATA_FILE, LABEL_FILE, H, W, B)

# Count distinct positive label values -> number of real land-cover classes
num_classes = int(np.unique(y_img[y_img > 0]).size)
print(f"Scene shape: {x_img.shape}  |  Classes: {num_classes}")
```

### g) Output & Interpretation

- `x_img`: the normalised `(330, 307, 6)` image cube, every value in `[0, 1]`, ready
  to be fed into the patch-extraction step.
- `y_img`: the `(330, 307)` integer label map, where `0` means "no label / background"
  and positive integers identify land-cover classes.
- `num_classes`: the count of distinct positive labels — used everywhere downstream
  (colour maps, one-hot encoding, bar chart categories, etc.).

### h) Limitations

- Per-band min-max normalisation is sensitive to outliers: a single extreme pixel
  value will compress the rest of that band's range.
- Normalisation statistics are computed from the *entire scene* (including any
  unlabeled pixels), so if the unlabeled region has very different reflectance values
  than the labeled region, the normalisation could be skewed.
- This is a fixed, deterministic preprocessing — it assumes the same normalisation
  bounds were used (or are appropriate) for the models being loaded; if the ensemble
  models were trained with different normalisation statistics, predictions could be
  miscalibrated.

---

## Method: Spatial Patch Extraction

### a) What it is

> Imagine cutting out a small square sticker (9×9 pixels) centered on every single
> pixel of a giant photo, so that for each pixel you also get to see its immediate
> neighbourhood. This gives the model "context" — it doesn't just see one pixel's
> colour, it sees the texture and pattern around it too.

### b) Why it's used here

The deep learning architectures used (CNN, GFNet, ViT-UNet) all expect small 2D
image patches as input rather than single pixel vectors, because spatial context
(what's around a pixel) is highly informative for land-cover classification — for
example, a single pixel's spectral signature alone might be ambiguous, but its
surrounding texture can disambiguate it.

### c) How it works — Step by step

1. Compute `pad = PATCH_SIZE // 2 = 4` (since `PATCH_SIZE = 9`).
2. Pad the normalised image on all sides by `pad` pixels using **edge replication**
   (`mode="edge"`), so patches near the image border don't run out of bounds.
3. Allocate an empty array `scene_pixels_scaled` of shape
   `(H * W, PATCH_SIZE, PATCH_SIZE, B)`.
4. For every pixel position `(r, c)` in the original (unpadded) `H × W` grid, slice
   out the `9×9×6` window from the padded image centered on that pixel:
   ```
   scene_pixels_scaled[idx] = x_pad[r : r+9, c : c+9, :]
   ```
5. Increment a flat index `idx` so the patches are stored in row-major (raster) order
   — this ordering matters later because the uncertainty/prediction outputs are
   reshaped back to `(H, W)` assuming this exact order.

### d) ASCII Flow Diagram

```
x_img (H, W, B) normalised image
        |
        v
np.pad(..., pad=4, mode="edge")
        |
        v
x_pad  (H+8, W+8, B)
        |
        v
for r in 0..H-1:
  for c in 0..W-1:
    patch = x_pad[r:r+9, c:c+9, :]   <- 9x9xB window
    scene_pixels_scaled[idx] = patch
    idx += 1
        |
        v
scene_pixels_scaled: (H*W, 9, 9, B)
```

### e) Worked Numerical Example

To keep this small, imagine a tiny `3×3` image with `B = 1` band and `PATCH_SIZE = 3`
(so `pad = 1`):

```
Original (3x3):
[ 1  2  3 ]
[ 4  5  6 ]
[ 7  8  9 ]
```

Edge-padding by 1 on each side (replicating border values) gives a `5×5` array:

```
[ 1 1 2 3 3 ]
[ 1 1 2 3 3 ]
[ 4 4 5 6 6 ]
[ 7 7 8 9 9 ]
[ 7 7 8 9 9 ]
```

For the center pixel of the original image, value `5` at position `(1,1)`, the
corresponding `3×3` patch centered on it is taken from `x_pad[1:4, 1:4]`:

```
[ 1 2 3 ]
[ 4 5 6 ]
[ 7 8 9 ]
```

— which is just the original image, because pixel `(1,1)` is already in the interior
and didn't need any padded values. For a *corner* pixel like `(0,0)` (value `1`), the
patch `x_pad[0:3, 0:3]` would be:

```
[ 1 1 2 ]
[ 1 1 2 ]
[ 4 4 5 ]
```

— note the repeated `1`s and `4`s, which come from the edge-replication padding.

### f) Code Walkthrough

```python
print("Extracting all spatial patches for full-scene inference...")

pad   = PATCH_SIZE // 2   # = 4, since PATCH_SIZE = 9
# Replicate border pixels by `pad` on each side so every pixel can have a full 9x9 window
x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

# Pre-allocate output: one 9x9xB patch per pixel in the original H x W grid
scene_pixels_scaled = np.empty((H * W, PATCH_SIZE, PATCH_SIZE, B), dtype=np.float32)

idx = 0
for r in range(H):           # iterate rows of the ORIGINAL (unpadded) image
    for c in range(W):       # iterate columns of the ORIGINAL (unpadded) image
        # x_pad[r:r+9, c:c+9, :] is centered on original pixel (r, c)
        # because x_pad is shifted by `pad` relative to x_img
        scene_pixels_scaled[idx] = x_pad[r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
        idx += 1

print(f"scene_pixels_scaled shape: {scene_pixels_scaled.shape}")
```

### g) Output & Interpretation

- `scene_pixels_scaled` has shape `(101310, 9, 9, 6)` — one patch per pixel of the
  `330×307` scene, ready to be fed in batches to each ensemble model for full-scene
  prediction.
- The **flat index order is row-major** (`idx` increments fastest over columns within
  a row), which is the same convention `np.reshape((H, W))` assumes later — this
  consistency is essential for the uncertainty maps to align spatially with the
  original image.

### h) Limitations

- Generates a very large array (`101,310 × 9 × 9 × 6` ≈ 49.7 million float32 values,
  roughly 200 MB) — memory-intensive for larger scenes or larger patch sizes.
- Edge-replication padding is a simple choice; it can slightly bias predictions for
  pixels very near the image border (their "neighbourhood" partially consists of
  duplicated values rather than true surrounding terrain).
- The nested Python `for` loop over `H × W` pixels is not vectorised — for very large
  scenes this extraction step could become a performance bottleneck.

---

## Method: Labeled Patch Extraction & Train/Test Split

### a) What it is

> This is like making a separate, smaller stack of "flashcards" — one for every pixel
> that actually has a known answer (a ground-truth label) — so you can later check
> how well a model's guesses match reality, and so models could originally be trained
> on a representative sample of each class.

### b) Why it's used here

Most of the scene's pixels may be unlabeled (`label == 0`). To evaluate or train a
model meaningfully, you need a set of pixels where you *know* the correct class.
This function isolates exactly those pixels (and their surrounding 9×9 patches),
and the subsequent split creates held-out test data with the same class proportions
as the full labeled set (via stratification).

### c) How it works — Step by step

1. Pad the normalised image the same way as before (`pad = patch_size // 2`, edge
   mode).
2. Find the `(row, col)` coordinates of every pixel where `y > 0` using
   `np.argwhere(y > 0)`.
3. For each such coordinate `(r, c)`:
   - Extract its `9×9×B` patch from the padded image.
   - Store the label as `y[r, c] - 1` (shifting classes from `1..K` to `0..K-1` for
     zero-indexed categorical encoding).
4. Return all patches `X`, all labels `y_labels`, and the original `(row, col)`
   coordinates.
5. Split `X` and `y_labels` into train (75%) and test (25%) sets using
   `train_test_split` with `stratify=y_labels` and `random_state=SEED`, ensuring each
   class is represented proportionally in both splits and the split is reproducible.
6. Convert `y_test` to one-hot format `y_test_cat` using
   `tf.keras.utils.to_categorical(y_test, num_classes)`.

### d) ASCII Flow Diagram

```
x_img (H,W,B), y_img (H,W)
        |
        v
pad x_img by `pad` (edge mode) -> x_pad
        |
        v
coords = argwhere(y_img > 0)     <- list of (r,c) for labeled pixels
        |
        v
for (r,c) in coords:
    patches[i] = x_pad[r:r+9, c:c+9, :]
    labels[i]  = y_img[r,c] - 1
        |
        v
X (N,9,9,B), y_labels (N,), coords (N,2)
        |
        v
train_test_split(X, y_labels, train_size=0.75, stratify=y_labels, random_state=42)
        |
        +--> x_train_full, y_train_full  (75%)
        +--> x_test, y_test              (25%)
                       |
                       v
              to_categorical -> y_test_cat
```

### e) Worked Numerical Example

Suppose there are only `N = 8` labeled pixels total, with classes
`y_labels = [0, 0, 0, 0, 1, 1, 1, 1]` (4 of class 0, 4 of class 1). With
`train_size = 0.75` and stratification, the split preserves the 50/50 class ratio:

- Train (75% of 8 = 6 pixels): 3 of class 0, 3 of class 1
- Test (25% of 8 = 2 pixels): 1 of class 0, 1 of class 1

If `num_classes = 2`, then `y_test = [0, 1]` becomes, after one-hot encoding:

```
y_test_cat:
[ 1, 0 ]   <- class 0
[ 0, 1 ]   <- class 1
```

### f) Code Walkthrough

```python
print("Extracting labeled patches for metric evaluation...")

def extract_labeled_patches(x, y, patch_size=9):
    pad = patch_size // 2
    # Edge-pad so patches near image borders are still fully populated
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

    # Find every pixel coordinate that has a real (positive) label
    coords = np.argwhere(y > 0)

    # Pre-allocate output arrays
    patches = np.empty((coords.shape[0], patch_size, patch_size, x.shape[-1]), dtype=np.float32)
    labels  = np.empty((coords.shape[0],), dtype=np.int32)

    for i, (r, c) in enumerate(coords):
        # Extract the patch centered on (r, c)
        patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
        # Shift label from 1-indexed to 0-indexed for categorical encoding
        labels[i] = int(y[r, c]) - 1

    return patches, labels, coords

X, y_labels, coords = extract_labeled_patches(x_img, y_img, PATCH_SIZE)

# Standard Split: 75% train / 25% test, stratified to preserve class balance
TRAIN_PERCENT = 0.75
x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y_labels, train_size=TRAIN_PERCENT, random_state=SEED, stratify=y_labels
)

# One-hot encode the test labels for metric computations that need probability vectors
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"Standard Test Set Extracted: {x_test.shape}")
```

### g) Output & Interpretation

- `X`, `y_labels`, `coords`: every labeled pixel's patch, class, and original
  position — the full labeled dataset.
- `x_train_full` / `y_train_full`: 75% of labeled data, stratified — would be used to
  (re)train models if needed.
- `x_test` / `y_test` / `y_test_cat`: 25% held-out labeled data — available for
  computing accuracy-style metrics at *known* pixel locations, complementing the
  scene-wide CreDE uncertainty maps (which cover *all* pixels, labeled or not).

### h) Limitations

- The `for i, (r, c) in enumerate(coords)` loop is a per-pixel Python loop — for
  scenes with many labeled pixels this can be slow compared to a vectorised
  implementation.
- `random_state=SEED` makes the split reproducible, but it is tied to the *current*
  ordering of `coords` (which depends on `np.argwhere`'s row-major scan) — if the
  underlying label raster changes, the exact split membership will change too.
- The one-hot encoding `y_test_cat` and the train/test arrays are computed but, in
  this particular notebook, not directly used by the CreDE evaluation loop later
  (which runs on the *full* scene, not just `x_test`). They appear to be
  carried over from a training/evaluation notebook for consistency.

---

## Method: Custom Keras Layers for Model Deserialisation

### a) What it is

> When you save a trained neural network that contains "home-made" pieces — building
> blocks that aren't in standard TensorFlow — Keras needs a blueprint to reconstruct
> those exact pieces when loading the model back. This section is that blueprint: four
> custom LEGO-brick designs (`PatchExtractor`, `PatchPositionEncoder`,
> `GlobalFilterLayer`, `PatchEncoderWithCLS`) that the saved `.keras` files reference
> by name.

### b) Why it's used here

The notebook loads pre-trained `.keras` model files for three different
architectures (AlexNet-style CNN, GFNet, ViT-UNet). At least the GFNet and ViT-UNet
architectures rely on these custom layers (patch tokenisation, positional encoding,
frequency-domain filtering, and a CLS-token transformer pattern). Without registering
these exact class definitions via `@tf.keras.utils.register_keras_serializable()`,
`tf.keras.models.load_model(...)` would fail with an "unknown layer" error.

### c) How it works — Step by step

For each custom layer:

1. Decorate the class with `@tf.keras.utils.register_keras_serializable()` so Keras
   can map a saved layer's class name back to this Python class.
2. Implement `__init__` to store configuration (e.g., `patch_size`, `num_patches`,
   `projection_dim`, `token_side`).
3. Implement `call()` — the forward computation performed when data flows through the
   layer.
4. Implement `get_config()` — returns a dictionary of the constructor arguments so the
   layer can be re-instantiated identically when loading.

The four layers, specifically:

**`PatchExtractor`** — splits an input image into non-overlapping
`patch_size × patch_size` tiles using `tf.image.extract_patches`, then flattens the
spatial grid of tiles into a sequence `(batch, num_patches, patch_dim)`. This is the
"tokenisation" step that turns a 2D image into a sequence, as required by
transformer-style architectures.

**`PatchPositionEncoder`** — takes a sequence of patch tokens, linearly projects each
token to `projection_dim` via a `Dense` layer, and adds a learned positional embedding
(one embedding vector per patch position) so the model knows *where* each patch sits
in the grid (since flattening loses spatial order information otherwise).

**`GlobalFilterLayer`** — the core building block of GFNet. It reshapes the token
sequence back into a square 2D grid, applies a 2D Fast Fourier Transform (FFT) to move
into the frequency domain, multiplies element-wise by a *learned complex-valued
filter* (`w_real + i*w_imag`), applies the inverse FFT to return to the spatial
domain, and flattens back to a sequence. This lets the model learn global spatial
filters efficiently (mixing information across the whole token grid) rather than only
local convolutional filters.

**`PatchEncoderWithCLS`** — similar to `PatchPositionEncoder`, but additionally
prepends a single learnable "CLS" (classification) token to the start of the
sequence before adding positional embeddings. The CLS token is a common
Vision-Transformer trick: as the sequence passes through transformer blocks, the CLS
token's final representation aggregates information from the whole image and is used
for the final classification decision.

### d) ASCII Flow Diagram

```
Image (batch, H, W, C)
        |
        v
[PatchExtractor]  -- splits into non-overlapping patch_size x patch_size tiles
        |
        v
Sequence of patch tokens (batch, num_patches, patch_dim)
        |
        +----------------------------+
        v                             v
[PatchPositionEncoder]        [PatchEncoderWithCLS]
  project -> Dense               project -> Dense
  + position embedding           prepend [CLS] token
        |                          + position embedding (num_patches+1)
        v                                  |
  Encoded tokens                  Encoded tokens incl. CLS
  (used by GFNet path)            (used by ViT-UNet path)
        |
        v
[GlobalFilterLayer]  (used inside GFNet blocks)
  reshape tokens -> square grid
  FFT2D -> multiply by complex filter (w_real + i*w_imag) -> inverse FFT2D
  reshape back -> sequence
```

### e) Worked Numerical Example

**PatchExtractor (toy example):** Suppose the input is a `4×4` single-channel image
and `patch_size = 2`:

```
Input image (4x4):
[ 1  2  3  4 ]
[ 5  6  7  8 ]
[ 9 10 11 12 ]
[13 14 15 16 ]
```

With non-overlapping `2×2` patches (stride = `patch_size`), there are
`(4/2) × (4/2) = 4` patches, each flattened to `4` values:

```
Patch (top-left):     [ 1, 2, 5, 6]
Patch (top-right):    [ 3, 4, 7, 8]
Patch (bottom-left):  [ 9,10,13,14]
Patch (bottom-right): [11,12,15,16]
```

Result: a sequence of shape `(num_patches=4, patch_dim=4)`.

**GlobalFilterLayer (conceptual):** If the token grid is `2×2` (i.e. `token_side=2`)
and a token's value at frequency position `(0,0)` after FFT is `a + bi`, and the
learned filter at that position is `w_real + w_imag * i`, the element-wise complex
multiplication is:

```
result_real = a*w_real - b*w_imag
result_imag = a*w_imag + b*w_real
```

After multiplying *every* frequency component this way, an inverse FFT converts the
filtered frequency-domain grid back into a spatial-domain grid — effectively a
learned, globally-acting filter (similar to how a graphic-equalizer boosts/cuts
specific frequency bands of audio, but here applied to spatial frequencies of the
token grid).

### f) Code Walkthrough

```python
@tf.keras.utils.register_keras_serializable()
class PatchExtractor(layers.Layer):
    """Extracts non-overlapping image patches and flattens them into a sequence."""

    def __init__(self, patch_size=3, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size  # side length of each square patch

    def call(self, images):
        # Slide a non-overlapping patch_size x patch_size window (stride = patch_size)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        batch, patch_dim = tf.shape(images)[0], tf.shape(patches)[-1]
        # Total number of patches = (grid_rows * grid_cols)
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
        # Flatten the 2D grid of patches into a 1D sequence
        return tf.reshape(patches, [batch, num_patches, patch_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})  # needed to rebuild this layer on load
        return cfg
```

```python
@tf.keras.utils.register_keras_serializable()
class PatchPositionEncoder(layers.Layer):
    """Projects patches to `projection_dim` and adds learned positional embeddings."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        # Linear projection from raw patch values to model dimension
        self.projection         = layers.Dense(projection_dim)
        # One learnable embedding vector per patch position
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # Add positional info so the model knows "where" each patch came from
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return cfg
```

```python
@tf.keras.utils.register_keras_serializable()
class GlobalFilterLayer(layers.Layer):
    """
    Applies a learnable complex-valued filter in the 2-D frequency domain.
    """

    def __init__(self, token_side, **kwargs):
        super().__init__(**kwargs)
        self.token_side = token_side  # side length of the square token grid

    def build(self, input_shape):
        channels = int(input_shape[-1])
        # Real and imaginary parts of a learnable complex filter, one per (row,col,channel)
        self.w_real = self.add_weight(
            name="w_real", shape=(self.token_side, self.token_side, channels),
            initializer="glorot_uniform", trainable=True
        )
        self.w_imag = self.add_weight(
            name="w_imag", shape=(self.token_side, self.token_side, channels),
            initializer="zeros", trainable=True  # starts as a purely real filter
        )
        super().build(input_shape)

    def call(self, x):
        batch    = tf.shape(x)[0]
        channels = tf.shape(x)[-1]

        # Reshape the flat token sequence back into a square 2D grid
        x_2d       = tf.reshape(x, [batch, self.token_side, self.token_side, channels])
        # Move to the frequency domain
        x_fft      = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
        # Combine learned real/imag parts into one complex filter
        w_complex  = tf.complex(self.w_real, self.w_imag)
        # Element-wise multiply in frequency domain == learned global filtering
        x_filtered = x_fft * w_complex
        # Back to spatial domain, keep only the real part
        x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))

        # Flatten back to a sequence
        return tf.reshape(x_spatial, [batch, self.token_side * self.token_side, channels])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"token_side": self.token_side})
        return cfg
```

```python
@tf.keras.utils.register_keras_serializable()
class PatchEncoderWithCLS(layers.Layer):
    """
    Projects patches, prepends a learnable [CLS] token, and adds positional embeddings.
    Used by the ViT-UNet architecture.
    """

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection         = layers.Dense(projection_dim)
        # +1 for the extra CLS token position
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1, output_dim=projection_dim
        )

    def build(self, input_shape):
        # A single learnable token, shared across the batch
        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, self.projection_dim),
            initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, patches):
        batch      = tf.shape(patches)[0]
        patch_proj = self.projection(patches)
        # Repeat the single CLS token once per batch element
        cls_tokens = tf.repeat(self.cls_token, repeats=batch, axis=0)
        # Prepend CLS token to the front of the patch-token sequence
        x          = tf.concat([cls_tokens, patch_proj], axis=1)
        positions  = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        return x + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return cfg
```

```python
# Maps the string class names stored inside the .keras files
# back to these Python class definitions, so load_model can reconstruct them.
CUSTOM_OBJECTS = {
    "PatchExtractor":      PatchExtractor,
    "PatchPositionEncoder": PatchPositionEncoder,
    "GlobalFilterLayer":   GlobalFilterLayer,
    "PatchEncoderWithCLS": PatchEncoderWithCLS,
}
```

### g) Output & Interpretation

- These classes themselves produce no "result" to interpret — they are
  *infrastructure* that makes `tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, ...)`
  succeed for the GFNet and ViT-UNet checkpoints later in the notebook.
- `CUSTOM_OBJECTS` is passed into every `load_model` call inside the ensemble
  evaluation function.

### h) Limitations

- **Exact match required**: the layer implementations must match what was used at
  training time bit-for-bit in terms of computation graph (the notebook explicitly
  warns "Do not alter any layer internals").
- `GlobalFilterLayer` assumes the token sequence length is a perfect square
  (`token_side * token_side`) — if a model used a non-square token grid this layer
  would fail or silently misbehave.
- Casting to `tf.complex64` and back means these operations run at limited numerical
  precision compared to `complex128`, which is rarely an issue for learned filters but
  is a fixed design choice.

---

## Method: Ensemble Path Discovery

### a) What it is

> This is a small "file finder" — like searching your computer for every photo named
> `vacation_*.jpg` in a folder, with a backup folder to check if the first one is
> empty.

### b) Why it's used here

The CreDE method needs **multiple independently trained model checkpoints** (the
"ensemble members") for each architecture. This function automatically locates all
the `.keras` files belonging to one architecture, so the main loop doesn't need
hard-coded file lists.

### c) How it works — Step by step

1. Build a glob pattern for the primary location:
   `MODEL_DIR / "{model_name}_ens_*_final.keras"`.
2. Search for files matching that pattern.
3. If no files are found, fall back to a legacy directory:
   `MODEL_DIR / "ensembles_old" / "{model_name}_ens_*_final.keras"`.
4. Return the list of matching file paths (could be empty if neither location has
   matches).

### d) ASCII Flow Diagram

```
model_name (e.g. "AlexNet_CNN")
        |
        v
pattern = MODEL_DIR / "AlexNet_CNN_ens_*_final.keras"
        |
        v
glob.glob(pattern) --> paths
        |
   paths empty? ----yes----> fallback pattern = MODEL_DIR/"ensembles_old"/"AlexNet_CNN_ens_*_final.keras"
        |  no                          |
        v                              v
   return paths                  glob.glob(fallback) --> paths
                                        |
                                        v
                                  return paths (possibly still empty)
```

### e) Worked Numerical Example

Suppose `MODEL_DIR/` contains these files:

```
AlexNet_CNN_ens_1_final.keras
AlexNet_CNN_ens_2_final.keras
AlexNet_CNN_ens_3_final.keras
GFNet_ens_1_final.keras
```

Calling `get_ensemble_paths("AlexNet_CNN")`:

- Primary pattern `AlexNet_CNN_ens_*_final.keras` matches **3 files** → returns all 3
  immediately, no fallback needed.

Calling `get_ensemble_paths("ViT_UNet")` (assume no matching files exist anywhere):

- Primary pattern matches **0 files**.
- Fallback pattern in `ensembles_old/` also matches **0 files**.
- Returns an **empty list** `[]`.

### f) Code Walkthrough

```python
def get_ensemble_paths(model_name):
    # Primary search: MODEL_DIR/<model_name>_ens_<anything>_final.keras
    primary_pattern  = str(MODEL_DIR / f"{model_name}_ens_*_final.keras")
    paths = glob.glob(primary_pattern)

    if not paths:  # Fallback to legacy directory if nothing found
        fallback_pattern = str(MODEL_DIR / "ensembles_old" / f"{model_name}_ens_*_final.keras")
        paths = glob.glob(fallback_pattern)

    return paths
```

### g) Output & Interpretation

- Returns a `list[str]` of absolute file paths.
- An **empty list** signals "no ensemble members available for this architecture" —
  the main loop checks for this and skips the architecture entirely with a printed
  message.
- A **non-empty list** becomes `ensemble_paths`, fed directly into
  `evaluate_homogeneous_ensemble`.

### h) Limitations

- Relies entirely on filename conventions (`{model_name}_ens_*_final.keras`) — if
  checkpoint files don't follow this naming pattern exactly, they won't be found.
- `glob.glob` does not guarantee any particular ordering of returned paths across
  platforms/filesystems, so "ensemble member 1, 2, 3..." order is not strictly
  guaranteed (though for CreDE's min/max-based aggregation, member *order* does not
  affect the final result — only *membership* matters).
- Does not validate that the found files actually belong to the same architecture
  (relies purely on the naming convention being followed correctly when the models
  were originally saved).

---

## Method: Homogeneous Ensemble Evaluation — CreDE Credal Bounds

### a) What it is

> Imagine asking 5 different weather forecasters (all trained on similar data, but
> each with slightly different experience) for tomorrow's chance of rain. Instead of
> just averaging their answers, CreDE looks at the **full range** of opinions — the
> lowest and highest probability anyone gave to "rain" — and uses that *range* itself
> as a measure of how uncertain the group is. A wide range between forecasters means
> high **epistemic** uncertainty (the *experts disagree*); even if they all agree, the
> probabilities themselves might still show genuine ambiguity in the weather pattern
> itself — that's **aleatoric** uncertainty (the *situation itself* is ambiguous).

### b) Why it's used here

For each architecture (CNN, GFNet, ViT-UNet), the notebook has several independently
trained model checkpoints ("ensemble members" — a *homogeneous* ensemble means all
members share the same architecture but different trained weights, e.g. from
different random initialisations or training runs). CreDE combines their outputs
into a **credal set** — a set of plausible probability distributions bounded by
per-class minimum and maximum predicted probabilities — and decomposes the resulting
uncertainty into aleatoric (AU), epistemic (EU), and total (TU) components for every
pixel in the scene.

### c) How it works — Step by step

1. For each model checkpoint path in `model_paths`:
   a. Load the model with `tf.keras.models.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS, safe_mode=False)`.
   b. Run `model.predict(input_data, batch_size=2048)` to get a probability vector
      per pixel, shape `(N, C)` where `N = H*W` and `C = num_classes`.
   c. Append this prediction array to `all_preds`.
   d. Delete the model object, clear the Keras backend session, and run garbage
      collection — freeing GPU/CPU memory before loading the next checkpoint.
2. Stack all members' predictions into a single tensor `stacked_preds` of shape
   `(M, N, C)` where `M` is the number of ensemble members.
3. Compute, for every pixel and class, the **minimum** and **maximum** probability
   across all `M` members:
   ```
   p_min[n, c] = min over m of stacked_preds[m, n, c]
   p_max[n, c] = max over m of stacked_preds[m, n, c]
   ```
4. Compute the per-class **credal spread**:
   ```
   delta_p[n, c] = p_max[n, c] - p_min[n, c]
   ```
5. Build `p_star`, the **lower probability distribution**, by normalising `p_min` so
   it sums to 1 across classes:
   ```
   p_star[n, c] = p_min[n, c] / (sum over c of p_min[n, c] + 1e-12)
   ```
   then clip to `[1e-12, 1.0]` to avoid `log(0)` in the next step.
6. **Aleatoric uncertainty (AU)** — the Shannon entropy of `p_star`:
   ```
   au[n] = -sum over c of ( p_star[n,c] * log(p_star[n,c]) )
   ```
7. **Epistemic uncertainty (EU)** — the mean credal spread across classes:
   ```
   eu[n] = mean over c of delta_p[n, c]
   ```
8. **Total uncertainty (TU)**:
   ```
   tu[n] = au[n] + eu[n]
   ```
9. **Predicted class** — the argmax of `p_star` (i.e., the class favoured by the
   *lower* credal bound):
   ```
   pred_class[n] = argmax over c of p_star[n, c]
   ```
10. Convert any TensorFlow tensors to NumPy arrays and return
    `(pred_class, p_star, au, eu, tu)`.

### d) ASCII Flow Diagram

```
model_paths = [m1.keras, m2.keras, ..., mM.keras]
        |
        v
for each model path:
    load model (with CUSTOM_OBJECTS)
    preds_m = model.predict(scene_pixels_scaled)   -> (N, C)
    all_preds.append(preds_m)
    delete model, clear session, gc.collect()
        |
        v
stacked_preds: (M, N, C)
        |
        +-----------------+
        v                 v
   p_min = min over M   p_max = max over M
        |                 |
        |                 v
        |          delta_p = p_max - p_min      (credal spread)
        |                                              |
        v                                              v
p_star = normalise(p_min)                     eu = mean_c(delta_p)
        |                                              |
        v                                              |
au = entropy(p_star)                                   |
        |                                              |
        +-----------------+---------------------------+
                           v
                    tu = au + eu

pred_class = argmax_c(p_star)

returns: pred_class, p_star, au, eu, tu
```

### e) Worked Numerical Example

Consider a tiny problem with **2 ensemble members** (`M=2`), **1 pixel** (`N=1`),
**3 classes** (`C=3`). The two models predict:

```
Model 1 prediction: [0.50, 0.30, 0.20]
Model 2 prediction: [0.30, 0.40, 0.30]
```

**Step 1 — stack:** `stacked_preds` shape is `(2, 1, 3)`.

**Step 2 — credal bounds (min/max across the 2 members, per class):**

```
p_min = [ min(0.50,0.30), min(0.30,0.40), min(0.20,0.30) ] = [0.30, 0.30, 0.20]
p_max = [ max(0.50,0.30), max(0.30,0.40), max(0.20,0.30) ] = [0.50, 0.40, 0.30]
```

**Step 3 — credal spread:**

```
delta_p = p_max - p_min = [0.50-0.30, 0.40-0.30, 0.30-0.20] = [0.20, 0.10, 0.10]
```

**Step 4 — normalise p_min to get p_star:**

```
sum(p_min) = 0.30 + 0.30 + 0.20 = 0.80
p_star = [0.30/0.80, 0.30/0.80, 0.20/0.80] = [0.375, 0.375, 0.25]
```

**Step 5 — aleatoric uncertainty (entropy of p_star), using natural log:**

```
au = -( 0.375*ln(0.375) + 0.375*ln(0.375) + 0.25*ln(0.25) )
   = -( 0.375*(-0.9808) + 0.375*(-0.9808) + 0.25*(-1.3863) )
   = -( -0.3678 - 0.3678 - 0.3466 )
   = -( -1.0822 )
   = 1.0822
```

**Step 6 — epistemic uncertainty (mean credal spread):**

```
eu = mean([0.20, 0.10, 0.10]) = (0.20+0.10+0.10)/3 = 0.40/3 ≈ 0.1333
```

**Step 7 — total uncertainty:**

```
tu = au + eu ≈ 1.0822 + 0.1333 = 1.2155
```

**Step 8 — predicted class:**

```
p_star = [0.375, 0.375, 0.25] -> argmax = class 0 (ties broken by first occurrence)
pred_class = 0
```

> **Interpretation of this toy result:** the two models *disagree* moderately
> (especially on class 0, where the spread is 0.20 — driving `eu`), and even the
> "worst-case" distribution `p_star` is fairly spread across classes 0 and 1 (driving
> a relatively high `au` too). Both AU and EU contribute to a high TU, meaning this
> pixel is genuinely hard to classify confidently for *both* reasons.

### f) Code Walkthrough

```python
def evaluate_homogeneous_ensemble(model_paths, input_data, batch_size=2048):
    all_preds = []

    for path in model_paths:
        print(f"  -> Loading & predicting: {Path(path).name}")
        # Load one ensemble member; compile=False since we only need forward inference,
        # custom_objects required for the custom layers, safe_mode=False allows
        # deserialising the custom @register_keras_serializable classes/lambdas.
        model = tf.keras.models.load_model(
            path, compile=False, custom_objects=CUSTOM_OBJECTS, safe_mode=False
        )
        # Predict class probabilities for every patch in the scene
        preds = model.predict(input_data, batch_size=batch_size, verbose=1)
        all_preds.append(preds)

        # Free memory before loading the next ensemble member
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    print("  -> Computing credal bounds...")
    # Stack all members' predictions: shape (num_models, num_pixels, num_classes)
    stacked_preds = tf.stack(all_preds, axis=0)

    # Per-pixel, per-class lower and upper bounds across the ensemble
    p_min = tf.reduce_min(stacked_preds, axis=0)
    p_max = tf.reduce_max(stacked_preds, axis=0)

    # Width of the credal interval per class -- "how much the models disagree"
    delta_p = p_max - p_min

    # Normalise the lower bounds into a valid probability distribution p_star
    p_star = p_min / (tf.reduce_sum(p_min, axis=-1, keepdims=True) + 1e-12)
    # Clip to avoid log(0) when computing entropy
    p_star = np.clip(p_star, 1e-12, 1.0)

    # Aleatoric uncertainty: entropy of the "worst-case" (lower-bound) distribution
    au = -np.sum(p_star * np.log(p_star), axis=-1)
    # Epistemic uncertainty: average disagreement (credal spread) across classes
    eu = np.mean(delta_p, axis=-1)
    # Total uncertainty: sum of the two components
    tu = au + eu

    # Final predicted class is taken from the lower-bound distribution p_star
    pred_class = np.argmax(p_star, axis=-1)

    # Ensure everything is returned as plain NumPy arrays
    to_np = lambda t: t.numpy() if hasattr(t, 'numpy') else t
    return to_np(pred_class), to_np(p_star), to_np(au), to_np(eu), to_np(tu)
```

### g) Output & Interpretation

- `pred_class`: shape `(N,)`, the predicted land-cover class for every pixel,
  decided from the *credal lower-bound* distribution `p_star`.
- `p_star`: shape `(N, C)`, the normalised lower-bound probability distribution per
  pixel — sums to 1 across classes.
- `au` (Aleatoric Uncertainty): high values mean the lower-bound probability
  distribution is itself spread across multiple classes — i.e., even in the
  "worst-case" reading of the ensemble, the situation looks genuinely ambiguous
  (e.g., a pixel that's a genuine mixture of two land-cover types).
- `eu` (Epistemic Uncertainty): high values mean the ensemble members disagree a lot
  (`p_max - p_min` is large) — i.e., the model itself is unsure, which in principle
  could be reduced with more/better training data or a better-fit model.
- `tu` (Total Uncertainty): the sum, used as an overall "how much should I distrust
  this pixel's prediction" signal.
- All three (`au`, `eu`, `tu`) are later compared against fixed thresholds
  (`AU_THRESH=0.5`, `EU_THRESH=0.2`, `TU_THRESH=0.7`) to flag pixels as "uncertain".

### h) Limitations

- **Sequential model loading** (one model at a time, with explicit memory cleanup)
  trades speed for memory safety — useful on memory-constrained environments like
  Colab, but slower than loading all members simultaneously.
- The credal-bound approach (`min`/`max` across members) is sensitive to **outlier
  ensemble members** — a single poorly-trained or anomalous model can dominate
  `p_min`/`p_max` for a class, since only the extreme values are used (not, e.g., a
  trimmed range or quantile-based bound).
- `p_star` is built from `p_min` only — it discards the "central tendency"
  (e.g. mean) of the ensemble entirely for the entropy computation, focusing
  specifically on a worst-case/lower-bound reading; this is a deliberate
  design choice of the CreDE framework but means `au` does not represent the entropy
  of the ensemble's "average" prediction.
- The `1e-12` clipping/epsilon values are small but arbitrary numerical-stability
  choices; for pixels where `sum(p_min)` is extremely close to zero (all models
  assign near-zero probability to every class for the *lower* bound), `p_star` could
  become a near-uniform distribution dominated by the epsilon term, potentially
  inflating `au` for those pixels.
- "Homogeneous ensemble" assumes all loaded checkpoints share the same architecture
  and output shape `(N, C)` — mixing architectures with different `C` would break the
  stacking step.

---

## Method: 6-Panel Spatial Uncertainty Mapping

> **Note:** The function is named `generate_spatial_crede_maps` and the figure layout
> is `3 rows × 4 columns = 12 panels`, but the section header in the notebook calls it
> "6-Panel" and the docstring says "standardised 6-panel figure" / "3x4 spatial
> uncertainty figure". This document follows the section title (6-Panel) while
> describing the actual `3×4` grid implemented in code — likely the figure was
> expanded from an earlier 6-panel version without updating all the labels/comments.

### a) What it is

> Picture a wall of 12 small maps of the same area, arranged in a 3×4 grid. The first
> map shows what the model thinks each spot is (e.g., forest, water, urban). The next
> three show, for each of the three uncertainty types (Aleatoric, Epistemic, Total), a
> simple "confident vs. not confident" map. Below those, the same three uncertainty
> types are shown again, but this time the "not confident" areas are painted grey
> *on top of* the prediction map, so you can see *which classes* tend to be uncertain.
> Finally, three bar charts count up how many pixels of each class fall into the
> "uncertain" category for each uncertainty type.

### b) Why it's used here

Numeric uncertainty values (`au`, `eu`, `tu` per pixel) are hard to interpret in
isolation. Turning them into **spatial maps** lets a human (or a paper reviewer)
immediately see *where* in the scene the model is least trustworthy, whether that
correlates with particular land-cover classes, edges between classes, or specific
regions — and the bar charts quantify *how many* pixels are affected.

### c) How it works — Step by step

1. Reshape the flat per-pixel arrays (`au`, `eu`, `tu`, `pred_class`) from `(H*W,)`
   back into `(H, W)` 2D maps.
2. Create three **binary masks** by thresholding each uncertainty map:
   ```
   au_mask = 1 where au_map > AU_THRESH else 0
   eu_mask = 1 where eu_map > EU_THRESH else 0
   tu_mask = 1 where tu_map > TU_THRESH else 0
   ```
3. Create three **"combined" overlay maps**: start from the predicted class map, but
   replace any pixel flagged as "uncertain" (mask == 1) with a special sentinel value
   `n_cls` (one beyond the last real class index), so it can be rendered in a distinct
   grey colour:
   ```
   combined_au = where(au_mask == 1, n_cls, pred_map)
   combined_eu = where(eu_mask == 1, n_cls, pred_map)
   combined_tu = where(tu_mask == 1, n_cls, pred_map)
   ```
4. Build three colour maps:
   - `cmap_base`: one fixed colour per class (no grey), for the raw prediction map.
   - `cmap_unc`: the same per-class colours **plus grey** as an extra colour for the
     "uncertain" sentinel value.
   - `cmap_binary`: just two colours (yellow = "Certain", dark navy = "Uncertain")
     for the binary masks.
5. Lay out a `3×4` grid of subplots (`figsize=(38, 26)`, a large poster-sized figure):
   - **Row 0**: base prediction map, then 3 binary certain/uncertain maps (one per
     uncertainty type), each with a yellow/navy legend.
   - **Row 1**: 3 grey-overlay "combined" maps (one per uncertainty type), 4th panel
     blank.
   - **Row 2**: 3 bar charts of pixel counts per class + an "Uncertain" bar (one per
     uncertainty type), 4th panel blank. Each bar is annotated with its exact count.
6. Add an overall figure title, tighten the layout, save the figure as a PNG (300 DPI)
   into `CREDE_OUT_DIR`, display it, and return the saved file path as a string.

### d) ASCII Flow Diagram

```
pred_class, au, eu, tu  (flat, shape (H*W,))
        |
        v
reshape each to (H, W)  -> pred_map, au_map, eu_map, tu_map
        |
        v
threshold each uncertainty map:
   au_mask = au_map > AU_THRESH
   eu_mask = eu_map > EU_THRESH
   tu_mask = tu_map > TU_THRESH
        |
        v
combined_X = where(X_mask==1, n_cls, pred_map)   for X in {au, eu, tu}
        |
        v
3x4 figure layout:
  Row0: [pred_map] [au binary] [eu binary] [tu binary]
  Row1: [combined_au] [combined_eu] [combined_tu] [blank]
  Row2: [au bar chart] [eu bar chart] [tu bar chart] [blank]
        |
        v
save PNG (300 dpi) -> CREDE_OUT_DIR/{model_name}_CreDE_spatial_maps.png
        |
        v
return save_path (string)
```

### e) Worked Numerical Example

Imagine a tiny `2×2` scene (`H=2, W=2`), `n_cls = 2` classes (0 and 1), and:

```
pred_map:
[ 0  1 ]
[ 1  0 ]

au_map:
[ 0.6  0.3 ]
[ 0.7  0.2 ]

AU_THRESH = 0.5
```

**Step 1 — au_mask** (1 where `au_map > 0.5`):

```
au_mask:
[ 1  0 ]
[ 1  0 ]
```

**Step 2 — combined_au** (replace masked pixels with sentinel `n_cls = 2`):

```
combined_au:
[ 2  1 ]
[ 2  0 ]
```

So pixels at `(0,0)` and `(1,0)` — which were predicted as class 0 and class 1
respectively — are now both shown as the sentinel value `2` (rendered grey,
"uncertain"), while `(0,1)` stays class 1 and `(1,1)` stays class 0.

**Step 3 — bar chart counts.** Using `np.unique(combined_au, return_counts=True)` on
the array `[2, 1, 2, 0]`:

```
value 0 -> count 1
value 1 -> count 1
value 2 -> count 2
```

With `bar_lbls = ['Class 0', 'Class 1', 'Uncertain']`, the bar heights would be
`[1, 1, 2]` — i.e., 1 pixel of class 0, 1 pixel of class 1, and 2 pixels flagged
"Uncertain" (by aleatoric threshold).

### f) Code Walkthrough

```python
# Shared colour palette (up to 10 classes + grey for uncertain pixels)
CLASS_COLORS = [
    '#0000FF', '#00FF00', '#FF0000', '#00FFFF', '#FF00FF',
    '#FFFF00', '#A52A2A', '#FFA500', '#7FFF00', '#8A2BE2'
]

def generate_spatial_crede_maps(
    model_name, pred_class_scene, p_star_scene, au_scene, eu_scene, tu_scene,
    H=330, W=307, au_thresh=0.5, eu_thresh=0.2, tu_thresh=0.7
):
    """Produce and save a 3x4 spatial uncertainty figure for CreDE."""
    print(f'  -> Generating 3x4 spatial maps for {model_name}...')
    n_cls = p_star_scene.shape[-1]  # number of real classes

    # Reshape flat per-pixel arrays back into (H, W) spatial grids
    au_map   = au_scene.reshape((H, W))
    eu_map   = eu_scene.reshape((H, W))
    tu_map   = tu_scene.reshape((H, W))
    pred_map = pred_class_scene.reshape((H, W))

    # Binary "uncertain" masks via fixed thresholds
    au_mask = (au_map > au_thresh).astype(int)
    eu_mask = (eu_map > eu_thresh).astype(int)
    tu_mask = (tu_map > tu_thresh).astype(int)

    # Combined maps: prediction everywhere EXCEPT pixels flagged uncertain,
    # which get the sentinel value `n_cls` (rendered as grey)
    combined_au = np.where(au_mask == 1, n_cls, pred_map)
    combined_eu = np.where(eu_mask == 1, n_cls, pred_map)
    combined_tu = np.where(tu_mask == 1, n_cls, pred_map)

    cmap_base   = ListedColormap(CLASS_COLORS[:n_cls])               # classes only
    cmap_unc    = ListedColormap(CLASS_COLORS[:n_cls] + ['#808080']) # classes + grey
    cmap_binary = ListedColormap(['#FFFF00', '#001F3F'])             # certain / uncertain

    bar_lbls = [f'Class {i}' for i in range(n_cls)] + ['Uncertain']
    bar_cols = CLASS_COLORS[:n_cls] + ['#808080']

    fig, axes = plt.subplots(3, 4, figsize=(38, 26))
    fig.suptitle(f'{model_name} — CreDE Uncertainty Maps (Absolute Thresholds)',
                 fontsize=24, fontweight='bold', y=0.99)

    # ── Row 0: Base prediction + 3 binary maps
    axes[0, 0].imshow(pred_map, cmap=cmap_base, vmin=0, vmax=n_cls - 1)
    axes[0, 0].set_title('Base Prediction Map', fontsize=15)
    axes[0, 0].axis('off')

    binary_specs = [
        (axes[0, 1], au_mask, f'Aleatoric (AU > {au_thresh})'),
        (axes[0, 2], eu_mask, f'Epistemic (EU > {eu_thresh})'),
        (axes[0, 3], tu_mask, f'Total      (TU > {tu_thresh})'),
    ]
    for ax, mask, label in binary_specs:
        ax.imshow(mask, cmap=cmap_binary, vmin=0, vmax=1)
        ax.set_title(f'Certain vs Uncertain\n{label}', fontsize=15, pad=10)
        ax.axis('off')
        ax.legend(
            handles=[Patch(facecolor='#FFFF00', label='Certain'),
                     Patch(facecolor='#001F3F', label='Uncertain')],
            loc='upper left', bbox_to_anchor=(0.0, -0.02), borderaxespad=0,
            fontsize=11, framealpha=0.9, ncol=2
        )

    # ── Row 1: Grey overlay maps (AU, EU, TU) + blank
    overlay_specs = [
        (axes[1, 0], combined_au, f'Aleatoric (AU > {au_thresh})'),
        (axes[1, 1], combined_eu, f'Epistemic (EU > {eu_thresh})'),
        (axes[1, 2], combined_tu, f'Total      (TU > {tu_thresh})'),
    ]
    for ax, combined, label in overlay_specs:
        ax.imshow(combined, cmap=cmap_unc, vmin=0, vmax=n_cls)
        ax.set_title(f'Grey Overlay — {label}', fontsize=15, pad=10)
        ax.axis('off')

    axes[1, 3].axis('off')  # unused panel

    # ── Row 2: Bar charts (AU, EU, TU) + blank
    bar_specs = [
        (axes[2, 0], combined_au, f'Aleatoric (AU > {au_thresh})'),
        (axes[2, 1], combined_eu, f'Epistemic (EU > {eu_thresh})'),
        (axes[2, 2], combined_tu, f'Total      (TU > {tu_thresh})'),
    ]
    for ax, combined, label in bar_specs:
        # Count how many pixels fall into each value (class 0..n_cls-1, or n_cls = "Uncertain")
        uniq, cnt = np.unique(combined, return_counts=True)
        c_dict   = {int(k): int(v) for k, v in zip(uniq, cnt)}
        bar_vals = [c_dict.get(i, 0) for i in range(n_cls + 1)]
        ax.bar(bar_lbls, bar_vals, color=bar_cols, edgecolor='black')
        ax.set_title(f'Pixel Counts — {label}', fontsize=15, pad=10)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.set_ylabel('Pixel Count', fontsize=12)
        max_val = max(bar_vals, default=1)
        # Annotate each bar with its exact count
        for i, v in enumerate(bar_vals):
            ax.text(i, v + max_val * 0.01,
                    f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylim(0, max_val * 1.12)  # leave headroom for the text labels

    axes[2, 3].axis('off')  # unused panel

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_path = CREDE_OUT_DIR / f'{model_name}_CreDE_spatial_maps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'  -> Saved: {save_path}')
    return str(save_path)
```

### g) Output & Interpretation

- A single PNG file per architecture, `{model_name}_CreDE_spatial_maps.png`, saved at
  300 DPI, containing all 12 panels.
- **Base Prediction Map**: the model's classification of the scene, one colour per
  class.
- **Binary Certain/Uncertain maps**: quick visual sense of *how much* of the scene
  exceeds each uncertainty threshold — large dark-navy areas mean widespread
  uncertainty of that type.
- **Grey-overlay maps**: show *which classes* (and where) the uncertain pixels would
  otherwise have been predicted as — useful for spotting, e.g., "epistemic
  uncertainty is concentrated at the boundary between class 2 and class 5".
- **Bar charts**: precise pixel counts per class plus an "Uncertain" bar — useful for
  reporting summary statistics (e.g., "12.4% of pixels exceeded the epistemic
  uncertainty threshold").
- **High uncertainty** (above threshold) = the model's prediction for that pixel
  should be treated with caution (for AU: the situation may be inherently ambiguous;
  for EU: more/better training data or a different model might help; for TU: either
  or both).
- **Low uncertainty** (below threshold) = the prediction is likely reliable by this
  metric.

### h) Limitations

- The thresholds (`AU_THRESH=0.5`, `EU_THRESH=0.2`, `TU_THRESH=0.7`) are **fixed,
  hand-chosen absolute values** (the figure title even says "Absolute Thresholds") —
  they are not calibrated per-dataset or per-architecture, so the same numeric
  threshold may mean something different for a different ensemble size or class
  count.
- The `3×4` grid has 4 unused/blank panels (`axes[1,3]` and `axes[2,3]`, plus the
  `axes[0,0]` slot is the only one in row 0 column 0 that isn't part of the
  binary/overlay/bar triplets) — the layout has spare capacity that isn't used for
  additional information.
- `figsize=(38, 26)` produces a very large image; this is fine for a saved PNG report
  but could be unwieldy to view directly without scaling down.
- The colour palette `CLASS_COLORS` supports up to 10 classes; if `num_classes > 10`,
  `CLASS_COLORS[:n_cls]` would simply be shorter than needed and `ListedColormap`
  would behave unpredictably (likely cycling or erroring).

---

## Method: Excel Report Generation

### a) What it is

> This is like a small in-house "report designer" — a set of helper functions that
> take plain data tables and images and arrange them into a polished, colour-coded
> Excel workbook, complete with a styled header row, alternating row shading, and
> embedded plot images — similar to what a human analyst might do manually in Excel,
> but automated.

### b) Why it's used here

After computing CreDE uncertainty metrics for each architecture and generating
spatial-map PNGs, the notebook needs a single shareable deliverable. An `.xlsx` file
with a formatted summary table (Sheet 1) and embedded plot images (Sheet 2) is more
presentable and portable than a raw CSV plus a folder of PNGs.

### c) How it works — Step by step

1. Define reusable Excel styling constants: header fill colour, alternating-row fill
   colour, header/body/title fonts, cell alignment, and thin cell borders.
2. `_style_header_row(ws, row, col_start, col_end)`: applies the header fill, font,
   centered alignment, and border to every cell in a given row range.
3. `_style_data_rows(ws, row_start, row_end, col_start, col_end)`: for each data row,
   applies alternating background fill (every even row gets the light-blue
   `_ALT_FILL`), body font, left alignment, and border.
4. `_write_df_to_sheet(ws, df, start_row, start_col, title)`: writes a pandas
   DataFrame into a worksheet starting at a given cell:
   - Optionally writes a merged title row above the table.
   - Writes the column headers and styles them.
   - Writes each data row, **rounding floats to 6 decimal places**.
   - Styles the data rows with alternating shading.
   - Auto-sizes each column's width based on the longest value (capped at 40
     characters).
   - Returns the row number just after the written table (so additional content
     could be placed below it).
5. `create_crede_excel_report(out_dir, summary_df, plot_paths)`:
   - Creates a new `Workbook`.
   - **Sheet "CreDE Summary"**: writes `summary_df` starting at row 1, with the title
     "CreDE — Inference Summary", using `_write_df_to_sheet`.
   - **Sheet "Plots"** (only created if `plot_paths` is non-empty): for each
     `(label, img_path)` pair, writes the label as a bold text header, then loads the
     PNG via `openpyxl.drawing.image.Image`, **rescales it proportionally to a target
     width of 900px**, embeds it into the sheet, and adjusts the row height/spacing so
     subsequent images don't overlap.
   - Saves the workbook to `out_dir / 'CreDE_Results.xlsx'`.
   - Prints a confirmation message and returns the saved path.

### d) ASCII Flow Diagram

```
summary_df (pandas DataFrame)     plot_paths = [(label1, path1), (label2, path2), ...]
        |                                          |
        v                                          v
Workbook()                              Workbook.create_sheet('Plots')
  |                                          |
  v                                          v
Sheet 'CreDE Summary'              for (label, img_path) in plot_paths:
  _write_df_to_sheet(                 write label (bold)
     title='CreDE — Inference           load image, rescale to width=900px
     Summary')                          embed image, advance row pointer
  |                                          |
  +------------------+-----------------------+
                      v
            wb.save(out_dir/'CreDE_Results.xlsx')
                      |
                      v
              return xlsx_path (string)
```

### e) Worked Numerical Example

**Column-width auto-sizing.** Suppose a DataFrame has a column `Model` with values
`["AlexNet_CNN_CreDE", "GFNet_CreDE", "ViT_UNet_CreDE"]`. The header text `"Model"` has
length 5. The values have lengths `18, 11, 14`. The maximum of all of these is `18`.
The computed column width is:

```
min(max_len + 4, 40) = min(18 + 4, 40) = min(22, 40) = 22
```

So the `Model` column gets a width of `22`.

**Image rescaling.** Suppose a saved spatial-map PNG has original dimensions
`orig_w = 11400`, `orig_h = 7800` (pixels, matching the `figsize=(38,26)` at 300 DPI).
The target width is `900`:

```
scale = 900 / 11400 ≈ 0.0789
new_width  = 11400 * 0.0789 ≈ 900
new_height = 7800  * 0.0789 ≈ 616
```

So the embedded image becomes roughly `900 × 616` pixels — proportionally shrunk to
fit nicely in the spreadsheet.

### f) Code Walkthrough

```python
# ── Excel Export Helpers ──────────────────────────────────────────────────────
_HDR_FILL   = PatternFill('solid', start_color='1F4E79')   # dark blue header background
_ALT_FILL   = PatternFill('solid', start_color='D6E4F0')   # light blue alternating rows
_HDR_FONT   = Font(name='Arial', bold=True, color='FFFFFF', size=11)  # white bold header text
_BODY_FONT  = Font(name='Arial', size=10)
_TITLE_FONT = Font(name='Arial', bold=True, size=13)
_CENTER     = Alignment(horizontal='center', vertical='center', wrap_text=True)
_LEFT       = Alignment(horizontal='left',   vertical='center')
_THIN_SIDE  = Side(style='thin', color='AAAAAA')
_THIN_BORDER= Border(left=_THIN_SIDE, right=_THIN_SIDE,
                     top=_THIN_SIDE,  bottom=_THIN_SIDE)

def _style_header_row(ws, row, col_start, col_end):
    # Apply header styling to every cell in the given row range
    for c in range(col_start, col_end + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill   = _HDR_FILL
        cell.font   = _HDR_FONT
        cell.alignment = _CENTER
        cell.border = _THIN_BORDER

def _style_data_rows(ws, row_start, row_end, col_start, col_end):
    for r in range(row_start, row_end + 1):
        # Alternate shading: even rows get the light-blue fill, odd rows stay default
        fill = _ALT_FILL if r % 2 == 0 else PatternFill()
        for c in range(col_start, col_end + 1):
            cell = ws.cell(row=r, column=c)
            cell.font      = _BODY_FONT
            cell.fill      = fill
            cell.alignment = _LEFT
            cell.border    = _THIN_BORDER

def _write_df_to_sheet(ws, df, start_row=1, start_col=1, title=None):
    r = start_row
    if title:
        # Optional merged title cell spanning the table's width
        cell = ws.cell(row=r, column=start_col, value=title)
        cell.font      = _TITLE_FONT
        cell.alignment = _LEFT
        ws.merge_cells(start_row=r, start_column=start_col,
                       end_row=r, end_column=start_col + len(df.columns) - 1)
        r += 1

    # Write column headers
    for j, col_name in enumerate(df.columns, start=start_col):
        ws.cell(row=r, column=j, value=col_name)
    _style_header_row(ws, r, start_col, start_col + len(df.columns) - 1)
    r += 1

    data_start = r
    for _, row_data in df.iterrows():
        for j, val in enumerate(row_data, start=start_col):
            # Round floats to 6 decimals for readability; leave other types as-is
            ws.cell(row=r, column=j, value=round(float(val), 6)
                    if isinstance(val, (float, np.floating)) else val)
        r += 1
    _style_data_rows(ws, data_start, r - 1, start_col, start_col + len(df.columns) - 1)

    # Auto-size each column based on its longest entry (header or data), capped at 40
    for j, col_name in enumerate(df.columns, start=start_col):
        max_len = max(len(str(col_name)),
                      max((len(str(ws.cell(row=row, column=j).value or ''))
                           for row in range(data_start, r)), default=0))
        ws.column_dimensions[get_column_letter(j)].width = min(max_len + 4, 40)
    return r

def create_crede_excel_report(out_dir, summary_df, plot_paths):
    """Auto-creates CreDE_Results.xlsx with Summary and Plots."""
    xlsx_path = Path(out_dir) / 'CreDE_Results.xlsx'
    wb = Workbook()

    # ── Sheet 1: Summary ──────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = 'CreDE Summary'
    ws1.row_dimensions[1].height = 30
    _write_df_to_sheet(ws1, summary_df, start_row=1, title='CreDE — Inference Summary')

    # ── Sheet 2: Plots ────────────────────────────────────────────────────────
    if plot_paths:
        ws2 = wb.create_sheet('Plots')
        title_cell = ws2.cell(row=1, column=1, value='CreDE — Spatial Uncertainty Maps')
        title_cell.font = Font(name='Arial', bold=True, size=14)

        img_row = 3
        for label, img_path in plot_paths:
            if not Path(img_path).exists():
                continue  # skip if a plot wasn't actually generated/saved
            ws2.cell(row=img_row, column=1, value=label).font = Font(name='Arial', bold=True, size=11)
            img_row += 1

            xl_img = XLImage(img_path)
            orig_w, orig_h = xl_img.width, xl_img.height
            target_w = 900
            scale    = target_w / orig_w if orig_w > 0 else 1
            xl_img.width  = int(orig_w * scale)
            xl_img.height = int(orig_h * scale)

            ws2.add_image(xl_img, f'A{img_row}')
            ws2.row_dimensions[img_row].height = xl_img.height * 0.75
            # Advance the row pointer past this image, plus some spacing
            img_row += int(xl_img.height / 15) + 3

    wb.save(xlsx_path)
    print(f'\n✅ Excel report saved → {xlsx_path}')
    return str(xlsx_path)
```

### g) Output & Interpretation

- A single file: `CreDE_Results.xlsx` inside `CREDE_OUT_DIR`.
- **"CreDE Summary" sheet**: one row per architecture, with columns for mean AU, EU,
  TU, and per-class pixel counts — a quick numeric overview comparable across
  architectures.
- **"Plots" sheet**: the full spatial-map figure for each architecture, embedded
  directly in the spreadsheet at a manageable size, with a bold label above each.
- This file is the main shareable deliverable summarising the entire CreDE analysis
  across all evaluated architectures.

### h) Limitations

- Row-height/image-spacing calculations (`xl_img.height * 0.75` and
  `img_row += int(xl_img.height / 15) + 3`) are heuristic "magic numbers" tuned for a
  specific image aspect ratio; very differently-shaped images might not space
  perfectly.
- `_write_df_to_sheet` rounds floats to 6 decimals but leaves integer-like columns
  (e.g., pixel counts) as whatever type they were in the DataFrame — if a count column
  were accidentally float-typed, it would also get rounded to 6 decimals (cosmetically
  harmless here since counts are whole numbers, but worth noting).
- The column-width formula `min(max_len + 4, 40)` is a simple heuristic; very long
  text values will be truncated visually (capped at width 40) even though the
  underlying cell value is unchanged.
- If `plot_paths` is empty (no architectures produced plots, e.g., because no
  ensembles were found for any architecture), only the "CreDE Summary" sheet is
  created — its table would then also be empty/absent in practice if
  `master_results` itself is empty (see the Master Evaluation Loop).

---

## Method: Master Evaluation Loop

### a) What it is

> This is the "conductor" of the whole notebook — it goes through each of the three
> model architectures one at a time, runs the full CreDE pipeline (load ensemble →
> compute uncertainty → draw maps), tidies up memory after each one, and finally
> stitches all the architectures' results together into one master spreadsheet and
> report.

### b) Why it's used here

Running CreDE for three different architectures back-to-back, with careful memory
management between them, lets the notebook produce a single, directly-comparable
summary across all architectures without manual re-running for each one.

### c) How it works — Step by step

1. Initialise empty lists `master_results` (per-architecture summary dicts) and
   `plot_entries` (label + saved-plot-path tuples).
2. Define `architectures = ['AlexNet_CNN', 'GFNet', 'ViT_UNet']`.
3. For each `model_name` in `architectures`:
   a. Print a banner announcing the architecture being evaluated.
   b. Call `get_ensemble_paths(model_name)`. If empty, print a "skipping" message and
      `continue` to the next architecture.
   c. **Step 1**: call `evaluate_homogeneous_ensemble(ensemble_paths, scene_pixels_scaled, batch_size=2048)`
      to get `(pred_class, p_star, au, eu, tu)` for the whole scene.
   d. **Step 2**: call `generate_spatial_crede_maps(...)` with these arrays plus
      `H, W` and the three thresholds, producing and saving the PNG; record
      `(label, path)` in `plot_entries`.
   e. **Step 3**: compute per-class pixel counts from `pred_class` via
      `np.unique(pred_class, return_counts=True)`, then append a summary dict to
      `master_results` containing the model name (with `_CreDE` suffix), mean AU, mean
      EU, mean TU, and one `Class_{k}_Pixels` entry per class present.
   f. **Step 4**: delete the large arrays (`pred_class, p_star, au, eu, tu`), call
      `tf.keras.backend.clear_session()` and `gc.collect()` to free memory before
      moving to the next architecture.
4. After the loop, if `master_results` is non-empty:
   - Build `df_summary = pd.DataFrame(master_results)`.
   - Save it to `CREDE_OUT_DIR / "CreDE_Master_Summary.csv"`.
   - Print the saved path and the full table (`df_summary.to_string(index=False)`).
   - Call `create_crede_excel_report(CREDE_OUT_DIR, df_summary, plot_entries)` to
     produce the combined Excel report.

Two metric helper functions (`multiclass_brier_score`, `expected_calibration_error`)
are also defined in this section but are **not called** anywhere in the provided
notebook — see Limitations below.

### d) ASCII Flow Diagram

```
master_results = []
plot_entries   = []
architectures  = ['AlexNet_CNN', 'GFNet', 'ViT_UNet']

for model_name in architectures:
        |
        v
  ensemble_paths = get_ensemble_paths(model_name)
        |
   empty? --yes--> print "skipping" --> continue to next architecture
        | no
        v
  (pred_class, p_star, au, eu, tu) = evaluate_homogeneous_ensemble(...)
        |
        v
  saved_plot_path = generate_spatial_crede_maps(...)
  plot_entries.append((label, saved_plot_path))
        |
        v
  pixel_counts = unique-counts of pred_class
  master_results.append({ Model, Mean_AU, Mean_EU, Mean_TU, Class_k_Pixels... })
        |
        v
  del pred_class, p_star, au, eu, tu
  clear_session(); gc.collect()
        |
        v
  (loop continues to next architecture)

after loop:
  if master_results not empty:
      df_summary = DataFrame(master_results)
      save -> CreDE_Master_Summary.csv
      print df_summary
      create_crede_excel_report(...) -> CreDE_Results.xlsx
```

### e) Worked Numerical Example

Suppose only `AlexNet_CNN` and `GFNet` have ensemble checkpoints (`ViT_UNet` is
skipped), `num_classes = 3`, and the per-architecture results are:

```
AlexNet_CNN:
  pred_class pixel counts: class0=40000, class1=35000, class2=26310
  mean(au) = 0.42, mean(eu) = 0.15, mean(tu) = 0.57

GFNet:
  pred_class pixel counts: class0=38000, class1=37000, class2=26310
  mean(au) = 0.39, mean(eu) = 0.11, mean(tu) = 0.50
```

`master_results` would become:

```
[
  {"Model": "AlexNet_CNN_CreDE", "Mean_AU": 0.42, "Mean_EU": 0.15, "Mean_TU": 0.57,
   "Class_0_Pixels": 40000, "Class_1_Pixels": 35000, "Class_2_Pixels": 26310},
  {"Model": "GFNet_CreDE", "Mean_AU": 0.39, "Mean_EU": 0.11, "Mean_TU": 0.50,
   "Class_0_Pixels": 38000, "Class_1_Pixels": 37000, "Class_2_Pixels": 26310}
]
```

`df_summary` (as a table):

```
| Model              | Mean_AU | Mean_EU | Mean_TU | Class_0_Pixels | Class_1_Pixels | Class_2_Pixels |
|--------------------|---------|---------|---------|----------------|----------------|----------------|
| AlexNet_CNN_CreDE  | 0.42    | 0.15    | 0.57    | 40000          | 35000          | 26310          |
| GFNet_CreDE        | 0.39    | 0.11    | 0.50    | 38000          | 37000          | 26310          |
```

This table is saved as `CreDE_Master_Summary.csv` and embedded into
`CreDE_Results.xlsx`.

### f) Code Walkthrough

```python
# -----------------------------
# Metric Helpers (defined but not invoked in the provided notebook)
# -----------------------------
def multiclass_brier_score(y_onehot, y_prob):
    # Mean squared difference between predicted probabilities and one-hot truth,
    # summed across classes, averaged across samples
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

def expected_calibration_error(y_true, y_prob, n_bins=15):
    # Measures how well predicted confidence matches actual accuracy,
    # by bucketing predictions into confidence bins and comparing
    # average confidence vs. average accuracy within each bin
    confidences, predictions = np.max(y_prob, axis=1), np.argmax(y_prob, axis=1)
    correct = (predictions == y_true).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1] if i == n_bins - 1 else confidences < bin_edges[i+1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(correct[in_bin]) - np.mean(confidences[in_bin])) * prop
    return float(ece)
```

```python
master_results = []
plot_entries = []

architectures = ['AlexNet_CNN', 'GFNet', 'ViT_UNet']

for model_name in architectures:
    print(f"\n{'='*60}\n  Evaluating CreDE: {model_name}\n{'='*60}")

    ensemble_paths = get_ensemble_paths(model_name)
    if not ensemble_paths:
        print(f"  -> No ensemble models found for {model_name}. Skipping.")
        continue

    # Step 1: Run homogeneous ensemble evaluation -> CreDE uncertainty decomposition
    pred_class, p_star, au, eu, tu = evaluate_homogeneous_ensemble(
        ensemble_paths, scene_pixels_scaled, batch_size=2048
    )

    # Step 2: Generate and save the 3x4 spatial maps figure
    saved_plot_path = generate_spatial_crede_maps(
        model_name, pred_class, p_star, au, eu, tu,
        H=H, W=W, au_thresh=AU_THRESH, eu_thresh=EU_THRESH, tu_thresh=TU_THRESH
    )
    plot_entries.append((f'{model_name} — Spatial Uncertainty Maps', saved_plot_path))

    # Step 3: Accumulate per-architecture summary metrics
    unique, counts = np.unique(pred_class, return_counts=True)
    pixel_counts   = dict(zip(unique, counts))

    master_results.append({
        "Model":    f"{model_name}_CreDE",
        "Mean_AU":  float(np.mean(au)),
        "Mean_EU":  float(np.mean(eu)),
        "Mean_TU":  float(np.mean(tu)),
        **{f"Class_{int(k)}_Pixels": int(v) for k, v in pixel_counts.items()}
    })

    # Step 4: Aggressively free RAM before next architecture
    del pred_class, p_star, au, eu, tu
    tf.keras.backend.clear_session()
    gc.collect()

# --- Export master summary & Excel Report ---
if master_results:
    df_summary = pd.DataFrame(master_results)

    # Save CSV
    csv_path = CREDE_OUT_DIR / "CreDE_Master_Summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary → {csv_path}")
    print(df_summary.to_string(index=False))

    # Save Excel Report
    create_crede_excel_report(
        out_dir=CREDE_OUT_DIR,
        summary_df=df_summary,
        plot_paths=plot_entries
    )
```

### g) Output & Interpretation

- `master_results`: a list of dicts, one per *successfully evaluated* architecture
  (architectures with no ensemble checkpoints are silently skipped).
- `CreDE_Master_Summary.csv`: a CSV with one row per architecture, columns
  `Model, Mean_AU, Mean_EU, Mean_TU, Class_0_Pixels, Class_1_Pixels, ...` — directly
  comparable across architectures.
- `CreDE_Results.xlsx`: the combined, styled Excel report (Summary + Plots sheets).
- **Interpretation**: lower `Mean_AU`/`Mean_EU`/`Mean_TU` generally indicate an
  architecture whose ensemble is, on average, both more internally consistent
  (low EU = members agree) and more "confidently decisive" in its worst-case
  distribution (low AU). Comparing `Class_k_Pixels` columns across architectures
  shows how differently each architecture partitions the scene.

### h) Limitations

- If a class present in one architecture's `pred_class` is absent in another's, the
  resulting `df_summary` will have `NaN` for that `Class_k_Pixels` cell in the row
  where it's missing (pandas fills missing dict keys with `NaN` when building a
  DataFrame from a list of differently-keyed dicts) — this is not explicitly handled.
- If **no** architecture has available ensemble checkpoints, `master_results` stays
  empty and **neither the CSV nor the Excel report is created** — the notebook would
  finish silently without producing the master summary or report (though individual
  per-architecture plots, if any were generated before the `continue`, would still
  exist — though in practice if `ensemble_paths` is empty, `continue` happens before
  any plot is made, so truly nothing would be produced).
- `multiclass_brier_score` and `expected_calibration_error` are defined but never
  called — these appear to be prepared for future evaluation against `y_test` /
  `y_test_cat` (the held-out labeled test set from Section 3.3) but are not wired into
  the current pipeline. As written, the notebook does not report classification
  accuracy/calibration against ground truth — only ensemble-derived uncertainty
  statistics over the full (mostly unlabeled) scene.
- The loop processes architectures **sequentially and only once each** — there's no
  retry logic if `evaluate_homogeneous_ensemble` or `generate_spatial_crede_maps`
  raises an exception partway through (e.g., a corrupted checkpoint file would halt
  the entire loop for that and all subsequent architectures unless wrapped in
  try/except, which it is not).

---

## 6. Results & Comparisons

> **Note:** No executed cell outputs (printed tables, figures, or saved-file
> confirmations) are included in the provided notebook export — only the source
> code. Results not shown in provided notebook.

Based on the code, when run successfully the notebook would produce, **for each
architecture with available ensemble checkpoints** (`AlexNet_CNN`, `GFNet`,
`ViT_UNet`):

1. A `{model_name}_CreDE_spatial_maps.png` file (3×4 grid of maps and bar charts).
2. One row in the master summary table with columns:

```
| Model              | Mean_AU | Mean_EU | Mean_TU | Class_0_Pixels | Class_1_Pixels | ... |
|--------------------|---------|---------|---------|----------------|----------------|-----|
| AlexNet_CNN_CreDE  |   ?     |   ?     |   ?     |       ?        |       ?        | ... |
| GFNet_CreDE        |   ?     |   ?     |   ?     |       ?        |       ?        | ... |
| ViT_UNet_CreDE     |   ?     |   ?     |   ?     |       ?        |       ?        | ... |
```

3. A combined `CreDE_Master_Summary.csv` and `CreDE_Results.xlsx` (Summary + Plots
   sheets) inside `CREDE_OUT_DIR` (`.../Classification/ensemble/results/`).

> **Note:** Since actual numeric results are not present in the provided notebook
> export, the table above is shown with placeholders (`?`) rather than fabricated
> numbers. When the notebook is executed, this table should be filled in directly
> from `CreDE_Master_Summary.csv`.

---

## 7. Academic Paper Summary

### Problem Statement

Pixel-wise land-cover classification from multispectral remote-sensing imagery is
subject to two distinct sources of predictive uncertainty: aleatoric uncertainty
arising from inherent ambiguity in the spectral signal (e.g., mixed pixels at class
boundaries), and epistemic uncertainty arising from limitations of the learned model
itself. This work addresses the need to quantify and spatially localise both forms of
uncertainty across an entire classified scene, for three deep learning architectures
of differing inductive biases: a convolutional network (AlexNet-style CNN), a global
spectral filtering network (GFNet), and a hybrid Vision Transformer with a U-Net-style
structure (ViT-UNet).

### Methodology

The proposed pipeline, termed CreDE (Credal Deep Ensemble), operates on
homogeneous ensembles of independently trained models per architecture. For a
given architecture, each ensemble member is loaded sequentially and produces a
per-pixel class-probability vector for the entire scene (preprocessed via per-band
min-max normalisation and 9×9 spatial patch extraction). The ensemble's predictions
are aggregated by computing, for each pixel and class, the minimum and maximum
predicted probability across members, forming a credal set bounded by `p_min` and
`p_max`. The lower bound `p_min` is normalised to a valid probability distribution
`p_star`, whose Shannon entropy defines the Aleatoric Uncertainty (AU). The mean
per-class width of the credal interval (`p_max - p_min`) defines the Epistemic
Uncertainty (EU), and Total Uncertainty (TU) is the sum AU + EU. The predicted class
for each pixel is taken as the argmax of `p_star`. The three custom Keras layers
(`PatchExtractor`, `PatchPositionEncoder`/`PatchEncoderWithCLS`, and
`GlobalFilterLayer`) implement the patch-tokenisation, positional encoding, and
learned frequency-domain filtering operations required by the GFNet and ViT-UNet
architectures, registered for serialisation to allow checkpoint reloading.

### Experimental Setup

The dataset is a 330×307-pixel, 6-band multispectral scene with an associated integer
land-cover label raster. Labeled pixels (where the label is positive) are extracted
with 9×9 spatial patches and split 75/25 into train and test sets, stratified by
class, for potential downstream evaluation against ground truth (though this notebook
focuses on full-scene inference rather than test-set scoring). For each of the three
architectures, all available ensemble checkpoints (matching a `{model}_ens_*_final.keras`
naming pattern, with a legacy fallback directory) are loaded and run over every
9×9×6 patch of the full scene (101,310 pixels) in batches of 2048. Three fixed
absolute thresholds (AU > 0.5, EU > 0.2, TU > 0.7) define "uncertain" pixels for
visualisation purposes.

### Results Summary

For each architecture, CreDE produces a 3×4 spatial figure comprising the base
prediction map, three binary certain/uncertain masks (one per uncertainty type),
three grey-overlay maps showing the class identity of uncertain regions, and three
bar charts quantifying pixel counts per class and per uncertainty category. A master
summary table aggregates, per architecture, the mean AU, mean EU, mean TU, and
per-class pixel counts of the predicted scene, exported as both CSV and a styled
Excel workbook with embedded figures. The provided notebook export does not include
executed-cell numeric outputs, so specific comparative values (e.g., which
architecture achieves the lowest mean epistemic uncertainty) are not available in this
document and would need to be read from the generated `CreDE_Master_Summary.csv`.

### Conclusion

This work demonstrates an end-to-end, memory-conscious pipeline for decomposing
predictive uncertainty in multispectral land-cover classification into aleatoric and
epistemic components via credal-set bounds derived from homogeneous deep ensembles,
across three architecturally distinct deep learning models. The pipeline produces
both visual (spatial maps) and tabular (CSV/Excel) artifacts suitable for further
analysis or reporting. Limitations include reliance on fixed absolute uncertainty
thresholds rather than data-driven calibration, the use of only the lower credal bound
(`p_min`) to define the aleatoric component (discarding ensemble-mean information),
sensitivity of min/max credal bounds to outlier ensemble members, and the presence of
unused calibration-metric helper functions (Brier score, expected calibration error)
that suggest a planned but unimplemented quantitative evaluation against held-out
ground-truth labels. Future work could incorporate these calibration metrics against
the held-out test set, explore data-driven or per-class uncertainty thresholds, and
investigate robustness of the credal bounds to ensemble size and member quality.

---

## 8. References

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*. — Foundational reference for the AlexNet-style CNN architecture.

[2] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global Filter Networks for Image Classification. *Advances in Neural Information Processing Systems (NeurIPS)*. — Source for the GFNet architecture and the frequency-domain `GlobalFilterLayer` (FFT-based learnable filtering).

[3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR)*. — Source for the Vision Transformer patch-tokenisation, positional encoding, and CLS-token pattern used in `PatchEncoderWithCLS`.

[4] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*. — Reference for the U-Net-style encoder-decoder structure referenced by the "ViT-UNet" architecture name.

[5] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *Advances in Neural Information Processing Systems (NeurIPS)*. — Foundational reference for using ensembles of independently trained deep models to estimate predictive uncertainty, the basis for the "homogeneous ensemble" approach.

[6] Hüllermeier, E., & Waegeman, W. (2021). Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods. *Machine Learning*, 110(3), 457–506. — General reference for the aleatoric/epistemic uncertainty decomposition framework used throughout this notebook.

[7] Levi, M. Y., & Gurevich, S. (or related authors) — Credal set / imprecise probability formulations for uncertainty quantification (credal sets bounded by p_min/p_max), underlying the "CreDE" (Credal Deep Ensemble) naming and the lower/upper probability bound approach.

> **Note:** Reference [7] is included because the notebook's central method (CreDE,
> credal bounds via per-class min/max across ensemble members) is conceptually rooted
> in imprecise-probability / credal-set theory; the exact originating paper for the
> specific "CreDE" name/formulation is not identifiable from the notebook contents
> alone and should be verified by the author against their source material.

[8] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *International Conference on Machine Learning (ICML)*. — Reference for the Expected Calibration Error (ECE) metric defined (though not yet invoked) in the notebook.
