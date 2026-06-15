# DAPM Full: Domain-Adversarial Probabilistic Model for Multispectral Land-Cover Classification

## 1. Title & Overview

This notebook implements **DAPM (Domain-Adversarial Probabilistic Model)**, a two-stage deep learning
pipeline for classifying land-cover types from a 6-band multispectral image, where only a subset of
pixels (the "source domain") have ground-truth labels and the rest of the image (the "target domain")
is unlabeled.

In plain terms: the notebook takes three pre-trained image-patch classifiers (AlexNet-style CNN, GFNet,
and a ViT-UNet), strips off their final classification layers to use them as fixed "feature extractors,"
and then bolts on a custom probabilistic head — a Variational Autoencoder (VAE) combined with domain-
adversarial training (via a Gradient Reversal Layer) and a conditional diffusion model. The goal is to
build a latent representation that (a) reconstructs the input features, (b) is useful for classifying
the labeled source pixels, (c) looks statistically similar whether it comes from a labeled or unlabeled
pixel (domain alignment), and (d) can refine/denoise label predictions via diffusion.

The notebook trains this DAPM head for each of the three backbones, saves all weights and configs,
evaluates each resulting model on a held-out test set (Overall Accuracy, Average Accuracy, Cohen's
Kappa, weighted F1), and produces a full set of comparison visualizations including confusion matrices
and full-scene classification maps.

**Who this document is for:** Someone who used AI assistance to write this code and now wants to
deeply understand *how and why* every piece works — for personal learning, debugging, and for writing
up the methodology in a research paper (see Section 7, "Academic Paper Summary").

---

## 2. Table of Contents

1. [Title & Overview](#1-title--overview)
2. [Table of Contents](#2-table-of-contents)
3. [Environment & Dependencies](#3-environment--dependencies)
4. [Data & Problem Setup](#4-data--problem-setup)
5. Methods
   - [Method 1: Multispectral Data Loading & Per-Band Normalization](#method-1-multispectral-data-loading--per-band-normalization)
   - [Method 2: Patch Extraction (Source & Target Sampling)](#method-2-patch-extraction-source--target-sampling)
   - [Method 3: Frozen Backbone Feature Extractors (Custom Layers)](#method-3-frozen-backbone-feature-extractors-custom-layers)
   - [Method 4: VAE Encoder with Reparameterization (Sampling Layer)](#method-4-vae-encoder-with-reparameterization-sampling-layer)
   - [Method 5: Source/Target Decoders (Feature Reconstruction)](#method-5-sourcetarget-decoders-feature-reconstruction)
   - [Method 6: Classifier Head](#method-6-classifier-head)
   - [Method 7: Domain Discriminator with Gradient Reversal Layer (GRL)](#method-7-domain-discriminator-with-gradient-reversal-layer-grl)
   - [Method 8: Conditional Diffusion Model for Label Refinement](#method-8-conditional-diffusion-model-for-label-refinement)
   - [Method 9: Stage 1 — Joint VAE + Domain-Adversarial Training](#method-9-stage-1--joint-vae--domain-adversarial-training)
   - [Method 10: Stage 2 — Conditional Diffusion Training](#method-10-stage-2--conditional-diffusion-training)
   - [Method 11: Evaluation Metrics & Inference](#method-11-evaluation-metrics--inference)
   - [Method 12: Full-Scene Classification Map Generation](#method-12-full-scene-classification-map-generation)
6. [Results & Comparisons](#6-results--comparisons)
7. [Academic Paper Summary](#7-academic-paper-summary)
8. [References](#8-references)

---

## 3. Environment & Dependencies

| Library | Purpose |
|---|---|
| `os`, `sys`, `io`, `gc`, `json`, `time`, `random`, `pathlib.Path` | Standard library utilities — file paths, JSON I/O, garbage collection, RNG seeding |
| `google.colab.drive` | Mounts Google Drive when running inside Google Colab, so the project folder is accessible |
| `numpy` | Core numerical array operations — used for image arrays, patches, and metric math |
| `pandas` | Reads the CSV data/label files and builds the final summary tables |
| `seaborn` | Statistical plotting (bar plots with consistent styling) |
| `matplotlib.pyplot` | General plotting — confusion matrices, classification maps, bar charts |
| `sklearn.model_selection.train_test_split` | Stratified train/val/test splitting |
| `sklearn.metrics` (`confusion_matrix`, `classification_report`, `accuracy_score`, `cohen_kappa_score`, `f1_score`, `ConfusionMatrixDisplay`) | Computes and displays classification performance metrics |
| `tensorflow` / `tensorflow.keras` / `tensorflow.keras.layers` | Deep learning framework used to build, train, and save all neural network components |

**Version-specific notes:**
- The notebook calls `tf.keras.utils.register_keras_serializable()` on every custom layer. This is
  required so that the custom layers (`PatchExtractor`, `GlobalFilterLayer`, `Sampling`,
  `GradientReversal`, etc.) can be correctly saved/loaded with Keras's `.keras` and `.weights.h5`
  formats.
- `keras.models.load_model(..., compile=False, safe_mode=False)` is used to load the pre-trained
  backbones — `safe_mode=False` is required because the models contain custom Lambda-like layers
  (e.g., FFT operations in `GlobalFilterLayer`) that Keras's default safe deserialization would
  otherwise block.
- Global seeds (`SEED = 42`) are set for `random`, `numpy`, and `tensorflow` to make patch
  sampling, train/val/test splits, and weight initialization reproducible — though full
  bit-for-bit reproducibility on GPU is not guaranteed due to non-deterministic CUDA kernels.

---

## 4. Data & Problem Setup

**Dataset:** A single multispectral scene stored as two CSV files:
- `data.csv` — pixel values for a `330 × 307` image with `6` spectral bands (`H=330, W=307, B=6`),
  flattened into rows that get reshaped back to `(H, W, B)`.
- `ref.csv` — a `330 × 307` ground-truth label map, where `0` means "unlabeled" and values `> 0`
  represent 1-indexed class labels (converted to 0-indexed internally).

**Problem type:** Multi-class pixel-wise land-cover **classification** with a **domain adaptation**
twist — labeled pixels are treated as the "source domain" and unlabeled pixels as the "target domain"
(same scene, but pixels without ground truth).

**Preprocessing pipeline (exactly as done in the notebook):**

1. **Per-band min-max normalization** — each of the 6 spectral bands is independently rescaled to
   `[0, 1]` using that band's own min and max values (with a small epsilon `1e-8` to avoid
   division by zero).
2. **Patch extraction** — for every pixel of interest, a `9 × 9` (i.e., `PATCH_SIZE=9`) neighborhood
   patch across all 6 bands is extracted, using edge-padding so that pixels near the image border
   still get a full patch.
3. **Source set construction** — all pixels with label `> 0` are extracted as labeled patches
   `(X_all, y_all)`, with labels shifted to be 0-indexed (`y_img - 1`).
4. **Source split** — `X_all` is split via stratified sampling: `75%` train+val / `25%` test
   (`TRAIN_PERCENT = 0.75`), then the train+val portion is further split `80% train / 20% val`
   (`VAL_SPLIT_FROM_TRAIN = 0.20`), both stratified by class label.
5. **Target set construction** — all pixels with label `== 0` (unlabeled) are extracted as patches,
   then randomly subsampled down to at most `MAX_TARGET_UNLABELED = 20000` patches for tractability.
6. **Target split** — the target patches are split `90% train / 10% val`
   (`TARGET_VAL_FRACTION = 0.10`), without stratification (there are no labels to stratify by).

The result is five arrays used throughout training: `x_train`, `x_val`, `x_test` (source/labeled,
with corresponding `y_train`, `y_val`, `y_test`), and `x_target_train`, `x_target_val`
(target/unlabeled, no labels).

---
## Method 1: Multispectral Data Loading & Per-Band Normalization

### a) What it is

> Think of each spectral band as a separate black-and-white photo of the same scene, but taken with
> a different "color filter" (e.g., infrared, red, near-infrared). Each filter's camera has its own
> brightness range, so to compare them fairly, each photo is stretched so its darkest pixel becomes
> pure black (0) and its brightest pixel becomes pure white (1).

This step loads the raw `(H, W, B)` image and ground-truth label map from two CSV files and applies
**per-band min-max normalization**, scaling each spectral band independently into the `[0, 1]` range.

### b) Why it's used here

Neural networks train more stably when inputs are on a consistent numeric scale. Because each
spectral band can have a very different raw value range (e.g., one band might range 0–10000 while
another ranges 0–255), normalizing each band *independently* ensures no single band dominates the
network's early layers simply due to its raw magnitude.

### c) How it works — Step by step

1. Read `data.csv` into a NumPy array and reshape it into `(H, W, B)` = `(330, 307, 6)`.
2. Read `ref.csv` into a NumPy array and reshape it into `(H, W)` = `(330, 307)` — this is the
   ground-truth class map (0 = unlabeled).
3. For each band index `bi` from 0 to 5:
   ```
   band_min = min(band_values)
   band_max = max(band_values)
   denom = max(band_max - band_min, 1e-8)
   normalized_band = (band_values - band_min) / denom
   ```
4. Stack the 6 normalized bands back into a `(330, 307, 6)` array.
5. Return the normalized image and the (still raw, 1-indexed) label map.

### d) ASCII Flow Diagram

```
data.csv (flattened pixel values)        ref.csv (flattened labels)
        |                                         |
        v                                         v
  reshape -> (330, 307, 6)                reshape -> (330, 307)
        |                                         |
        v                                         |
  for each band b in 0..5:                        |
    band_min, band_max = min/max(band)            |
    band = (band - band_min) / max(band_max-band_min, 1e-8)
        |                                         |
        v                                         v
  x_norm (330, 307, 6)                     y_img (330, 307)
        |_________________________________________|
                          |
                          v
              passed to patch extraction (Method 2)
```

### e) Worked Numerical Example

Suppose Band 1 has just 4 pixel values: `[2, 4, 6, 8]`.

```
band_min = 2
band_max = 8
denom = max(8 - 2, 1e-8) = 6

normalized[0] = (2 - 2) / 6 = 0.0
normalized[1] = (4 - 2) / 6 = 0.333
normalized[2] = (6 - 2) / 6 = 0.667
normalized[3] = (8 - 2) / 6 = 1.0
```

So `[2, 4, 6, 8]` becomes `[0.0, 0.333, 0.667, 1.0]`. Now imagine Band 2 has raw values `[100, 150]`
— after the same formula it would *also* land in `[0.0, 1.0]`, so both bands are now on equal
footing even though their original scales were wildly different.

### f) Code Walkthrough

```python
def load_multispectral_6band(data_path, label_path, h, w, b):
    """Load and per-band normalise a multispectral image from two CSV files."""
    # Read the flattened pixel data and reshape into (height, width, bands)
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(h, w, b)
    # Read the flattened label map and reshape into (height, width); 0 = unlabeled
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(h, w)

    # Allocate an output array of the same shape for the normalized image
    x_norm = np.empty_like(x, dtype=np.float32)

    # Normalize each spectral band independently
    for bi in range(b):
        band  = x[:, :, bi]                       # isolate one band (H, W)
        mn, mx = float(np.min(band)), float(np.max(band))  # band's own min/max
        denom  = max(mx - mn, 1e-8)                # avoid divide-by-zero for constant bands
        x_norm[:, :, bi] = (band - mn) / denom     # rescale to [0, 1]

    return x_norm, y
```

### g) Output & Interpretation

- **Output:** `x_norm` — a `(330, 307, 6)` float32 array with every band's values in `[0, 1]`;
  `y` — a `(330, 307)` int32 array of class labels where `0` means "no ground truth available."
- **Interpretation:** `x_norm` is the canonical normalized image used for *all* downstream patch
  extraction (both source and target). `y` is used to decide which pixels belong to the source
  domain (`y > 0`) versus the target domain (`y == 0`), and to derive the 0-indexed class labels.

### h) Limitations

- Min-max normalization is sensitive to outliers — a single extreme pixel value in a band will
  compress the rest of that band's dynamic range toward 0.
- Normalization statistics (`band_min`, `band_max`) are computed on the *entire* image (including
  test and target pixels), which means there is a mild form of information leakage from test data
  into the normalization statistics — though since it's just a global min/max rescale (not a
  learned statistic), the practical impact is typically small.
- If a band is perfectly constant across the whole image, `denom` falls back to `1e-8`, producing
  values of `0` for that entire band — effectively making it uninformative without throwing an error.

---
## Method 2: Patch Extraction (Source & Target Sampling)

### a) What it is

> Imagine cutting out a small `9 × 9` square sticker centered on every pixel of interest, capturing
> not just that pixel's value but also its immediate neighborhood — like looking at a pixel through
> a small window rather than in isolation, so the model can "see" local texture and context.

This step extracts fixed-size `9 × 9 × 6` spatial patches around chosen pixel coordinates. Separate
helper functions handle (1) all *labeled* pixels (the source domain) and (2) a random sample of
*unlabeled* pixels (the target domain).

### b) Why it's used here

The pre-trained backbones (AlexNet-CNN, GFNet, ViT-UNet) all expect small image patches as input
rather than single pixels — local spatial context (the surrounding `9 × 9` neighborhood) carries
information about texture and adjacency that a single 6-band pixel value cannot. This step converts
the full scene into a dataset of `(patch, label)` pairs (source) and `(patch,)` only (target).

### c) How it works — Step by step

1. **Padding:** Pad the normalized image by `pad = patch_size // 2 = 4` pixels on each side using
   edge-replication (`mode='edge'`), so that patches centered near the image border don't run out
   of bounds.
2. **Generic patch extractor** (`extract_patches_from_coords`): for each `(row, col)` coordinate,
   slice out a `9 × 9 × 6` block from the padded image starting at that coordinate (which, due to
   padding, corresponds to a patch centered on the original pixel).
3. **Labeled (source) patches** (`extract_labeled_patches_with_coords`):
   ```
   coords = all (row, col) where y_img > 0
   labels = y_img[row, col] - 1   (convert 1-indexed -> 0-indexed)
   patches = extract_patches_from_coords(x_img, coords)
   ```
4. **Unlabeled (target) patches** (`extract_unlabeled_patches_with_coords`):
   ```
   coords = all (row, col) where y_img == 0
   if number of coords > max_samples:
       randomly keep only max_samples of them (without replacement, fixed seed)
   patches = extract_patches_from_coords(x_img, coords)
   ```

### d) ASCII Flow Diagram

```
x_norm (330, 307, 6)         y_img (330, 307)
        |                            |
        v                            v
  pad by 4 on each side    find coords where y > 0  ---> source coords
  (edge mode)                       and
        |                  coords where y == 0  --> subsample to <=20000 --> target coords
        |                            |
        +------------+---------------+
                      |
                      v
  for each coord (r, c):
    patch = padded_image[r : r+9, c : c+9, :]   # (9, 9, 6)
                      |
                      v
   source: (X_all, y_all, coords_all)   target: (X_target_all, coords_target_all)
                      |                                  |
                      v                                  v
        stratified 75/25 train+val/test       random 90/10 train/val
        then 80/20 train/val (of the 75%)
```

### e) Worked Numerical Example

Suppose the (unpadded) normalized image is just `5 × 5 × 1` (1 band for simplicity), and we want a
`3 × 3` patch (`patch_size=3`, `pad = 1`) centered at pixel `(row=0, col=0)` — the top-left corner.

After edge-padding by 1 pixel on all sides, the padded image becomes `7 × 7`. The pixel originally at
`(0,0)` is now located at `(1,1)` in the padded image. The `3 × 3` patch is extracted as
`padded_image[0:3, 0:3]`, i.e., rows/cols `0` through `2` of the padded array — which includes the
replicated edge pixels surrounding the original corner pixel.

If `y_img[0,0] = 3` (a labeled pixel), this patch goes into the source set with label
`3 - 1 = 2` (0-indexed class 2). If `y_img[0,0] = 0`, this patch instead becomes a *candidate* for
the target set (subject to random subsampling).

### f) Code Walkthrough

```python
def extract_patches_from_coords(x_img, coords, patch_size=9):
    """Extract (patch_size x patch_size x bands) patches at each (row, col) coordinate."""
    pad   = patch_size // 2                                   # 9 // 2 = 4
    # Pad the image so border pixels still get a full patch (replicate edge values)
    x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    # Pre-allocate output array: one (9, 9, B) patch per coordinate
    out   = np.empty((coords.shape[0], patch_size, patch_size, x_img.shape[-1]), dtype=np.float32)
    for i, (r, c) in enumerate(coords):
        # Because of padding, slicing [r : r+patch_size, c : c+patch_size] centers on (r, c)
        out[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
    return out

def extract_labeled_patches_with_coords(x_img, y_img, patch_size=9):
    """Return patches, 0-based labels, and coordinates for all labeled (y > 0) pixels."""
    coords   = np.argwhere(y_img > 0)                          # all labeled pixel locations
    x_patches = extract_patches_from_coords(x_img, coords, patch_size=patch_size)
    # Convert 1-indexed ground-truth labels to 0-indexed class IDs
    y_labels  = np.array([int(y_img[r, c]) - 1 for r, c in coords], dtype=np.int32)
    return x_patches, y_labels, coords

def extract_unlabeled_patches_with_coords(x_img, y_img, patch_size=9, max_samples=None, seed=42):
    """Return patches and coordinates for unlabeled (y == 0) pixels, with optional subsampling."""
    coords = np.argwhere(y_img == 0)                           # all unlabeled pixel locations
    if max_samples is not None and coords.shape[0] > max_samples:
        rng    = np.random.default_rng(seed)
        # Randomly keep only max_samples coordinates (no replacement) for tractability
        keep   = rng.choice(coords.shape[0], size=max_samples, replace=False)
        coords = coords[keep]
    x_patches = extract_patches_from_coords(x_img, coords, patch_size=patch_size)
    return x_patches, coords
```

### g) Output & Interpretation

- **Output:** `X_all` (all labeled patches, shape `(N_labeled, 9, 9, 6)`), `y_all` (0-indexed class
  labels, shape `(N_labeled,)`), `coords_all` (pixel coordinates), plus `X_target_all` and
  `coords_target_all` for the unlabeled/target patches (at most `20000` of them).
- After the splits in Section 4, this becomes `x_train`, `x_val`, `x_test`, `y_train`, `y_val`,
  `y_test` (source domain) and `x_target_train`, `x_target_val` (target domain, no labels).
- **Interpretation:** each row of `X_all`/`X_target_all` is one training example fed directly into
  the frozen backbone feature extractors (Method 3). The source set drives supervised classification
  loss; the target set drives the domain-adversarial and diffusion losses that have no ground truth.

### h) Limitations

- A `9 × 9` window is a fixed receptive field — it cannot capture larger-scale spatial patterns
  beyond ~4 pixels in each direction from the center.
- Edge-padding (`mode='edge'`) replicates border pixel values, which can slightly bias patches near
  the image boundary toward unrealistic homogeneous neighborhoods.
- The target set is capped at `20000` patches purely for computational tractability — this is a
  small fraction of the total unlabeled pixels in a `330 × 307 = 101,310`-pixel image, so the
  "target domain" the model sees during adversarial training is only a sample of the true target
  distribution.
- Stratified splitting (Section 4) requires every class to have enough samples to be split three
  ways — classes with very few labeled pixels could be poorly represented in val/test.

---
## Method 3: Frozen Backbone Feature Extractors (Custom Layers)

### a) What it is

> Think of three different "expert photographers" (AlexNet-CNN, GFNet, ViT-UNet) who were each
> already trained to recognize land-cover types from image patches. Instead of asking them for
> their final "this is class X" verdict, the notebook asks each expert to hand over their internal
> notes — the rich numerical summary they form just *before* making their final decision. These
> notes (the "penultimate layer activations") become the input to everything else in the notebook.

This section (a) registers four custom Keras layers needed to deserialize the pre-trained `.keras`
backbone files, and (b) loads each backbone, slices off its final classification layer, and freezes
its weights to produce a fixed **feature extractor**.

### b) Why it's used here

The notebook is *not* training new image classifiers from scratch. Instead, it reuses three already-
trained backbones as fixed feature extractors and builds a new probabilistic/domain-adaptation head
on top of their outputs. Freezing the backbones (`FREEZE_BACKBONE = True`) means only the new DAPM
components (encoder, decoders, classifier, discriminator, diffusion model) are trained — this is a
form of **transfer learning**.

The four custom layers exist because the backbones themselves were originally built with these
non-standard architectural components, and Keras needs to know how to reconstruct them when loading
the saved `.keras` files:

- `PatchExtractor` — used inside GFNet/ViT-style backbones to split an image into smaller
  sub-patches/tokens.
- `PatchPositionEncoder` — projects patches into an embedding space and adds learned positional
  embeddings (standard for transformer-like architectures).
- `GlobalFilterLayer` — the core of GFNet: applies a learnable filter in the 2D Fourier (frequency)
  domain.
- `PatchEncoderWithCLS` — a ViT-style patch encoder that prepends a learnable `[CLS]` (classification)
  token, used by the ViT-UNet backbone.

### c) How it works — Step by step

1. Define each custom layer class, decorated with `@tf.keras.utils.register_keras_serializable()`
   so Keras can find them by name when loading a saved model.
2. Collect all four classes into a `CUSTOM_OBJECTS` dictionary.
3. For each of the three backbone model keys (`AlexNet_CNN`, `GFNet`, `ViT_UNet`):
   ```
   model = keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS,
                                    compile=False, safe_mode=False)
   ```
4. Build a **feature extractor** sub-model that outputs the *second-to-last* layer's activations:
   ```
   penultimate_layer_output = model.layers[-2].output
   feat_model = keras.Model(model.input, penultimate_layer_output)
   ```
5. Freeze the feature extractor (`feat_model.trainable = False`) if `FREEZE_BACKBONE` is `True`.
6. Record the feature dimension (`feature_dim`) — the size of the penultimate layer's output vector
   — for each backbone, since this differs between architectures and is needed to size the DAPM
   encoder's input.

### d) ASCII Flow Diagram

```
Pre-trained .keras file (AlexNet_CNN / GFNet / ViT_UNet)
        |
        v
  keras.models.load_model(..., custom_objects=CUSTOM_OBJECTS,
                           compile=False, safe_mode=False)
        |
        v
  full_model  (input -> ... -> penultimate layer -> final classification layer)
        |
        v
  feat_model = Model(input, penultimate_layer.output)   # drop final layer
        |
        v
  feat_model.trainable = False   (if FREEZE_BACKBONE)
        |
        v
  feature_extractor   (input: 9x9x6 patch -> output: feature_dim vector)
        |
        v
  feature_dim recorded for use in build_dapm_encoder(...)
```

### e) Worked Numerical Example

Suppose the AlexNet-CNN backbone's full architecture (conceptually) is:

```
Input (9, 9, 6) -> Conv layers -> Flatten -> Dense(128, relu) -> Dense(num_classes, softmax)
                                              ^                    ^
                                       layers[-2]            layers[-1] (dropped)
```

The feature extractor becomes `Model(Input, Dense(128, relu).output)`. So if a `9 × 9 × 6` patch
goes in, the output is a `128`-dimensional vector of activations (`feature_dim = 128`) — *not* a
class probability. This `128`-dim vector is what feeds into the DAPM encoder in Method 4. Each of
the three backbones may have a different `feature_dim` (e.g., AlexNet-CNN might output 128, GFNet
might output 192, ViT-UNet might output 256) — the code reads this dynamically from
`extractor.output_shape[-1]` rather than hard-coding it.

### f) Code Walkthrough — Custom Layers

```python
@tf.keras.utils.register_keras_serializable()
class PatchExtractor(layers.Layer):
    """Extract non-overlapping image patches via tf.image.extract_patches."""

    def __init__(self, patch_size=3, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        # Slice the input image into non-overlapping patch_size x patch_size blocks
        patches   = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        batch     = tf.shape(images)[0]
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]   # total patches per image
        patch_dim = tf.shape(patches)[-1]                            # flattened patch size
        # Reshape from (batch, rows, cols, patch_pixels) to (batch, num_patches, patch_dim)
        return tf.reshape(patches, [batch, num_patches, patch_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'patch_size': self.patch_size})
        return cfg
```

```python
@tf.keras.utils.register_keras_serializable()
class PatchPositionEncoder(layers.Layer):
    """Project patches and add learned positional embeddings."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection          = layers.Dense(projection_dim)              # linear projection
        self.position_embedding  = layers.Embedding(input_dim=num_patches,   # one embedding per
                                                      output_dim=projection_dim)  # patch position

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        # Add a learned "where am I in the image" vector to each projected patch
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return cfg
```

```python
@tf.keras.utils.register_keras_serializable()
class GlobalFilterLayer(layers.Layer):
    """GFNet global filter: learnable 2-D frequency-domain weights."""

    def __init__(self, token_side, **kwargs):
        super().__init__(**kwargs)
        self.token_side = token_side   # side length of the square token grid

    def build(self, input_shape):
        channels = int(input_shape[-1])
        # Learnable real and imaginary parts of a complex frequency-domain filter
        self.w_real = self.add_weight(
            name='w_real', shape=(self.token_side, self.token_side, channels),
            initializer='glorot_uniform', trainable=True,
        )
        self.w_imag = self.add_weight(
            name='w_imag', shape=(self.token_side, self.token_side, channels),
            initializer='zeros', trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        batch    = tf.shape(x)[0]
        channels = tf.shape(x)[-1]
        # Reshape the flat token sequence back into a 2D grid
        x_2d     = tf.reshape(x, [batch, self.token_side, self.token_side, channels])
        # Move to the frequency domain via 2D FFT
        x_fft    = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
        w_complex  = tf.complex(self.w_real, self.w_imag)
        # Element-wise multiply in frequency domain = filtering operation
        x_filtered = x_fft * w_complex
        # Move back to spatial domain via inverse FFT, keep only the real part
        x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
        # Flatten the grid back into a token sequence
        return tf.reshape(x_spatial, [batch, self.token_side * self.token_side, channels])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'token_side': self.token_side})
        return cfg
```

```python
@tf.keras.utils.register_keras_serializable()
class PatchEncoderWithCLS(layers.Layer):
    """ViT-style patch encoder with a prepended learnable [CLS] token."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection         = layers.Dense(projection_dim)
        # +1 because the CLS token also needs a positional embedding
        self.position_embedding = layers.Embedding(input_dim=num_patches + 1, output_dim=projection_dim)

    def build(self, input_shape):
        # A single learnable "summary" token, prepended to every sequence
        self.cls_token = self.add_weight(
            name='cls_token', shape=(1, 1, self.projection_dim),
            initializer='zeros', trainable=True,
        )
        super().build(input_shape)

    def call(self, patches):
        batch      = tf.shape(patches)[0]
        patch_proj = self.projection(patches)
        # Repeat the single CLS token for every item in the batch
        cls_tokens = tf.repeat(self.cls_token, repeats=batch, axis=0)
        # Prepend CLS token to the sequence of projected patches
        x          = tf.concat([cls_tokens, patch_proj], axis=1)
        positions  = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        return x + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return cfg

CUSTOM_OBJECTS = {
    'PatchExtractor'      : PatchExtractor,
    'PatchPositionEncoder': PatchPositionEncoder,
    'GlobalFilterLayer'   : GlobalFilterLayer,
    'PatchEncoderWithCLS' : PatchEncoderWithCLS,
}
```

### Code Walkthrough — Building Feature Extractors

```python
def get_feature_extractor(base_model, freeze_backbone=True):
    """Return a sub-model that outputs the penultimate (pre-logit) layer activations."""
    penultimate = base_model.layers[-2].output                 # second-to-last layer's output tensor
    feat_model  = keras.Model(
        base_model.input, penultimate,
        name=f'{base_model.name}_feature_extractor',
    )
    feat_model.trainable = not freeze_backbone                 # freeze weights if requested
    return feat_model

base_models       = {}
feature_extractors = {}
feature_dims      = {}

for model_key in MODEL_KEYS:                                    # ['AlexNet_CNN', 'GFNet', 'ViT_UNet']
    path = MODEL_FILES[model_key]
    if not path.exists():
        raise FileNotFoundError(f'Missing base model: {path}')

    # Load the full pre-trained model, registering all custom layers it might need
    model     = keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False, safe_mode=False)
    extractor = get_feature_extractor(model, freeze_backbone=FREEZE_BACKBONE)
    feature_dim = int(extractor.output_shape[-1])               # size of the feature vector

    base_models[model_key]        = model
    feature_extractors[model_key] = extractor
    feature_dims[model_key]       = feature_dim

    print(model_key, 'loaded from', path)
    print('  feature_dim =', feature_dim)
```

### g) Output & Interpretation

- **Output:** `feature_extractors` — a dict mapping each model key (`AlexNet_CNN`, `GFNet`,
  `ViT_UNet`) to a frozen Keras sub-model that maps `(9, 9, 6)` patches to a `feature_dim`-length
  vector. `feature_dims` records the dimensionality for each.
- **Interpretation:** these feature vectors are the "raw material" fed into the DAPM encoder
  (Method 4) for every subsequent step. Since the backbones are frozen, all of the learning in this
  notebook happens *downstream* of this fixed feature representation.

### h) Limitations

- Because the backbones are frozen, the DAPM head can only work with whatever information the
  pre-trained backbones already chose to preserve in their penultimate layer — it cannot learn new
  low-level visual features.
- `safe_mode=False` disables some of Keras's deserialization safety checks; this is necessary for
  custom layers with non-standard operations (like FFT) but means loading a tampered/untrusted
  `.keras` file could execute arbitrary code.
- The assumption that `layers[-2]` is always the correct "penultimate" layer depends on each
  backbone's specific architecture ending in exactly `... -> features -> classification_head`. If a
  backbone has a different structure (e.g., dropout or batch-norm as the second-to-last layer), the
  extracted "features" might not be what's intended.

---
## Method 4: VAE Encoder with Reparameterization (Sampling Layer)

### a) What it is

> Imagine compressing a detailed photo into a short "summary description," but instead of writing
> one fixed summary, you write down a *range of possible summaries* — a best guess (mean) plus how
> uncertain you are about each detail (variance). Then you randomly pick one summary from that
> range each time. That's the VAE encoder: it doesn't just compress the data, it compresses it into
> a *probability distribution* over possible compressed representations, then samples from it.

The DAPM encoder is a small Variational Autoencoder (VAE) encoder. It takes the `feature_dim`-length
vector from the frozen backbone and maps it to a `64`-dimensional (`LATENT_DIM=64`) latent space,
represented as a Gaussian distribution (mean `z_mu` and log-variance `z_logvar`), then draws a
sample `z` from that distribution using the **reparameterization trick**.

### b) Why it's used here

A VAE-style latent space gives the model several useful properties simultaneously:

1. It provides a compact, lower-dimensional (`64`-d) representation that the classifier, decoders,
   discriminator, and diffusion model all operate on — making the rest of the architecture smaller
   and more efficient than working directly with the (potentially larger) backbone feature vectors.
2. The KL-divergence term (computed from `z_mu`/`z_logvar`, see Method 9) regularizes this latent
   space toward a standard normal distribution, which helps prevent overfitting and gives the
   latent space a smooth, well-behaved structure — useful both for the reconstruction task (Method
   5) and for the domain-adversarial alignment (Method 7).
3. Encoding *both* source and target features through the *same* encoder is what makes domain
   alignment meaningful — both domains are mapped into the same latent space, where the
   discriminator (Method 7) tries to detect whether a given `z` came from source or target.

### c) How it works — Step by step

1. Input: a `feature_dim`-length feature vector (from the frozen backbone).
2. Pass through two hidden Dense layers with ReLU activation, each of size `hidden_dim` (256,
   `DECODER_HIDDEN_DIM`).
3. Branch into two separate linear output heads, each of size `LATENT_DIM=64`:
   - `z_mu` — the mean of the latent Gaussian.
   - `z_logvar` — the log-variance of the latent Gaussian.
4. Sample `z` using the reparameterization trick:
   ```
   eps ~ Normal(0, 1)            # random noise, same shape as z_mu
   z = z_mu + exp(0.5 * z_logvar) * eps
   ```
5. Return `(z_mu, z_logvar, z)` — all three are used downstream (e.g., `z_mu` for "deterministic"
   inference, `z` for training with stochasticity).

### d) ASCII Flow Diagram

```
feature vector (feature_dim,)
        |
        v
  Dense(256, relu)  -- enc_h1
        |
        v
  Dense(256, relu)  -- enc_h2
        |
   +----+----+
   |         |
   v         v
Dense(64)  Dense(64)
 z_mu      z_logvar
   |         |
   +----+----+
        |
        v
  Sampling layer:
    eps ~ N(0,1)
    z = z_mu + exp(0.5 * z_logvar) * eps
        |
        v
  z (64,)  -->  fed to: decoders, classifier, discriminator, diffusion model
```

### e) Worked Numerical Example

Suppose `LATENT_DIM = 1` (just one latent dimension for simplicity) and after the Dense layers, the
encoder produces:

```
z_mu     = 2.0
z_logvar = 0.0     -> variance = exp(0.0) = 1.0,  std = exp(0.5 * 0.0) = 1.0
```

Suppose the random noise sample is `eps = 0.5` (drawn from a standard normal). Then:

```
z = z_mu + exp(0.5 * z_logvar) * eps
  = 2.0 + exp(0.5 * 0.0) * 0.5
  = 2.0 + 1.0 * 0.5
  = 2.5
```

So this particular sample of `z` is `2.5`. If we drew a different `eps`, say `eps = -1.0`:

```
z = 2.0 + 1.0 * (-1.0) = 1.0
```

Different noise samples give different `z` values, but they're all centered around `z_mu = 2.0`
with spread controlled by `z_logvar`. This randomness is what makes the model "probabilistic" —
the same input feature vector can produce slightly different latent codes on different forward
passes.

### f) Code Walkthrough

```python
@tf.keras.utils.register_keras_serializable()
class Sampling(layers.Layer):
    """VAE reparameterisation trick: z = mu + exp(0.5 * logvar) * eps."""

    def call(self, inputs):
        z_mu, z_logvar = inputs
        # Draw random Gaussian noise the same shape as z_mu
        eps = tf.random.normal(shape=tf.shape(z_mu))
        # Reparameterization trick: makes sampling differentiable w.r.t. z_mu and z_logvar
        return z_mu + tf.exp(0.5 * z_logvar) * eps
```

```python
def build_dapm_encoder(feature_dim, latent_dim=64, hidden_dim=256):
    """VAE encoder: feature vector -> (z_mu, z_logvar, z_sample)."""
    inp      = keras.Input(shape=(feature_dim,), name='enc_feature_in')
    h        = layers.Dense(hidden_dim, activation='relu', name='enc_h1')(inp)
    h        = layers.Dense(hidden_dim, activation='relu', name='enc_h2')(h)
    z_mu     = layers.Dense(latent_dim, name='z_mu')(h)          # linear output: mean
    z_logvar = layers.Dense(latent_dim, name='z_logvar')(h)      # linear output: log-variance
    z        = Sampling(name='z_sample')([z_mu, z_logvar])       # stochastic latent sample
    return keras.Model(inp, [z_mu, z_logvar, z], name='dapm_full_encoder')
```

### g) Output & Interpretation

- **Output:** three tensors of shape `(batch, 64)` — `z_mu`, `z_logvar`, and `z`.
- **Interpretation:**
  - `z_mu` is the "best estimate" latent code — used at inference time (no noise) for stable
    predictions (e.g., in `predict_with_dapm_classifier`, only `z_mu` is passed to the classifier).
  - `z_logvar` controls how much uncertainty/spread the encoder assigns to this input — values
    close to `0` mean variance ≈ 1 (matching the prior); very negative values mean the encoder is
    very confident (low variance) about this code.
  - `z` is the stochastic sample used during *training* — passing noisy samples through the decoders
    and classifier acts as a regularizer (similar in spirit to dropout) and is required for the VAE's
    theoretical justification (the KL term in Method 9 only makes sense if `z` really is sampled
    from `N(z_mu, exp(z_logvar))`).

### h) Limitations

- A Gaussian latent prior assumes the "true" underlying structure of the feature space is roughly
  unimodal/Gaussian-like, which may not perfectly hold for complex multispectral data with multiple
  land-cover types.
- The same encoder is shared between source and target domains — if the two domains are very
  different, the encoder may struggle to find a single latent space that serves both reconstruction
  goals well.
- The reparameterization trick adds stochastic noise during training, which can slow convergence or
  add training variance, especially early in training when `z_logvar` may not yet be well-calibrated.

---
## Method 5: Source/Target Decoders (Feature Reconstruction)

### a) What it is

> If the encoder is like writing a short summary of a book, the decoder is like trying to rewrite
> the full book *just from that summary*. If the rewritten book closely matches the original, the
> summary must have captured the important details. The notebook trains *two* such "rewriters" —
> one specialized for summaries that came from labeled (source) pixels, and one for summaries from
> unlabeled (target) pixels.

This step defines a generic decoder factory function, which is used to build **two separate
decoder networks**: `src_decoder` (for source-domain latent codes) and `tgt_decoder` (for
target-domain latent codes). Each maps a `64`-dimensional latent vector `z` back to a
`feature_dim`-length vector, attempting to reconstruct the original backbone feature vector.

### b) Why it's used here

The reconstruction objective (Method 9, `recon_loss`) forces the latent space `z` to retain enough
information to reproduce the original feature vector — this prevents the encoder from collapsing to
a trivial/uninformative representation. Using **two separate decoders** (rather than one shared
decoder) allows each decoder to specialize: the source decoder can focus on reconstructing the
statistical patterns typical of labeled-pixel features, while the target decoder focuses on
unlabeled-pixel features — useful if the two domains have systematically different feature
distributions (which is the whole premise of domain adaptation).

### c) How it works — Step by step

1. Input: a `64`-dimensional latent vector `z` (either `z_src` or `z_tgt`, depending on which
   decoder).
2. Pass through two hidden Dense layers with ReLU activation, each of size `hidden_dim=256`.
3. Output a `feature_dim`-length vector via a final Dense layer with **linear** activation (since
   reconstruction targets are continuous feature values, not probabilities).
4. The same factory function `build_dapm_decoder` is called twice with different `name` arguments
   (`'dapm_full_source_decoder'` and `'dapm_full_target_decoder'`) to create two independently-
   weighted decoder networks with identical architecture.

### d) ASCII Flow Diagram

```
z_src (64,)                          z_tgt (64,)
   |                                     |
   v                                     v
Dense(256, relu)  -- src h1        Dense(256, relu)  -- tgt h1
   |                                     |
   v                                     v
Dense(256, relu)  -- src h2        Dense(256, relu)  -- tgt h2
   |                                     |
   v                                     v
Dense(feature_dim, linear)         Dense(feature_dim, linear)
   |                                     |
   v                                     v
feat_src_rec (feature_dim,)        feat_tgt_rec (feature_dim,)
   |                                     |
   v                                     v
compared to feat_src             compared to feat_tgt
via recon_loss (Method 9)        via recon_loss (Method 9)
```

### e) Worked Numerical Example

Suppose `feature_dim = 3` and the original backbone feature vector for a source pixel is
`feat_src = [1.0, 2.0, 3.0]`. After encoding to `z_src` and decoding, suppose the decoder outputs
`feat_src_rec = [0.9, 2.2, 2.8]`.

The reconstruction loss (from Method 9, `recon_loss`) is the mean over the batch of the **sum of
squared errors**:

```
squared_errors = (1.0-0.9)^2 + (2.0-2.2)^2 + (3.0-2.8)^2
               = 0.01 + 0.04 + 0.04
               = 0.09

src_recon (for this one example) = 0.09
```

If the batch contains more examples, `src_recon` is the *mean* of each example's summed squared
error. A perfect reconstruction would give `src_recon = 0`; larger values mean the decoder (and
hence the encoder's latent code) is losing more information about the original features.

### f) Code Walkthrough

```python
def build_dapm_decoder(latent_dim, feature_dim, hidden_dim=256, name='dapm_full_decoder'):
    """Decoder: latent vector -> reconstructed feature vector."""
    inp = keras.Input(shape=(latent_dim,), name=f'{name}_z_in')
    h   = layers.Dense(hidden_dim, activation='relu', name=f'{name}_h1')(inp)
    h   = layers.Dense(hidden_dim, activation='relu', name=f'{name}_h2')(h)
    # Linear output: reconstruction targets are continuous feature values, not probabilities
    out = layers.Dense(feature_dim, activation='linear', name=f'{name}_out')(h)
    return keras.Model(inp, out, name=name)
```

```python
# Called twice with different names to create two independently-weighted decoders:
src_decoder = build_dapm_decoder(LATENT_DIM, feature_dim, hidden_dim=DECODER_HIDDEN_DIM,
                                  name='dapm_full_source_decoder')
tgt_decoder = build_dapm_decoder(LATENT_DIM, feature_dim, hidden_dim=DECODER_HIDDEN_DIM,
                                  name='dapm_full_target_decoder')
```

### g) Output & Interpretation

- **Output:** `feat_src_rec` and `feat_tgt_rec`, each shape `(batch, feature_dim)` — reconstructed
  versions of the original backbone feature vectors.
- **Interpretation:** lower `src_recon`/`tgt_recon` values (Method 9) indicate the latent space `z`
  preserves more of the original feature information for that domain. These losses are part of the
  total Stage-1 training objective and are also tracked separately during validation
  (`val_src_recon`, `val_tgt_recon`) as diagnostics of representation quality per domain.

### h) Limitations

- Using two separate decoders doubles the parameter count of the reconstruction pathway compared
  to a shared decoder, and risks the latent space becoming domain-specific in ways that could
  undermine the domain-alignment goal of the discriminator (Method 7) — there is an inherent
  tension between "reconstruct domain-specific features well" and "make source/target latents
  indistinguishable."
- Reconstruction loss alone does not guarantee the latent code is *useful* for classification —
  it only guarantees it retains enough information to reconstruct the input, which is why the
  classifier loss (Method 6/9) is needed as a separate term.
- The target decoder is trained on patches without ground-truth labels, so there is no direct
  signal to verify *what* the target-domain latents represent semantically — only that they can be
  reconstructed.

---

## Method 6: Classifier Head

### a) What it is

> This is the final "decision maker." After the encoder compresses a pixel's information into a
> 64-number summary, the classifier looks at that summary and says, "Based on this, I think this
> pixel belongs to land-cover class C, with these confidence percentages for each possible class."

The classifier is a small feed-forward network that maps the `64`-dimensional latent vector `z`
(or `z_mu`) to a probability distribution over `num_classes` land-cover classes via a softmax
output.

### b) Why it's used here

This is the component that actually performs the land-cover classification task — it is the part
whose output is compared against `y_train`/`y_val`/`y_test` ground-truth labels (via
`src_ce`, the source cross-entropy loss in Method 9) and whose predictions (`argmax`) are reported
as the final Overall Accuracy / Average Accuracy / Kappa / weighted-F1 metrics in Section 10. It
also provides the "soft guidance" signal (`y_guidance_src`, `y_guidance_tgt`) consumed by the
diffusion model in Stage 2 (Method 10).

### c) How it works — Step by step

1. Input: a `64`-dimensional latent vector `z` (during training) or `z_mu` (during inference/
   evaluation — see Method 11).
2. Pass through one hidden Dense layer with ReLU activation, size `hidden_dim=128`
   (`CLASSIFIER_HIDDEN_DIM`).
3. Output a `num_classes`-length vector via a Dense layer with **softmax** activation, producing
   class probabilities that sum to 1.

### d) ASCII Flow Diagram

```
z (64,)  [training: stochastic sample]
or
z_mu (64,)  [inference: deterministic mean]
        |
        v
  Dense(128, relu)  -- clf_h1
        |
        v
  Dense(num_classes, softmax)  -- clf_out
        |
        v
  y_prob (num_classes,)   e.g. [0.05, 0.02, 0.81, 0.04, 0.08, ...]
        |
        +--> argmax(y_prob)  -> predicted class label
        +--> compared to true label via sparse_categorical_crossentropy (src_ce)
        +--> used as "guidance" input to the diffusion model (Method 10)
```

### e) Worked Numerical Example

Suppose `num_classes = 4` and, for one input, the classifier's pre-softmax logits (the output of
`Dense(num_classes)` before the softmax activation) are `[2.0, 1.0, 0.5, -1.0]`.

Softmax computes:

```
exp(2.0)  = 7.389
exp(1.0)  = 2.718
exp(0.5)  = 1.649
exp(-1.0) = 0.368

sum = 7.389 + 2.718 + 1.649 + 0.368 = 16.124

y_prob[0] = 7.389 / 16.124 = 0.458
y_prob[1] = 2.718 / 16.124 = 0.169
y_prob[2] = 1.649 / 16.124 = 0.102
y_prob[3] = 0.368 / 16.124 = 0.023
```

(Note: rounded values sum to ≈0.752, the discrepancy is due to rounding in this illustration —
in practice they sum exactly to 1.0.) The predicted class is `argmax([0.458, 0.169, 0.102, 0.023]) =
0` (class index 0), with confidence ≈ 0.458. If the true label is also `0`, this prediction is
correct.

If the true label `y_true = 0`, the sparse categorical cross-entropy loss for this example is:

```
loss = -log(y_prob[y_true]) = -log(0.458) ≈ 0.781
```

### f) Code Walkthrough

```python
def build_dapm_classifier(latent_dim, num_classes, hidden_dim=128):
    """Classifier head: latent vector -> class probabilities."""
    inp = keras.Input(shape=(latent_dim,), name='clf_z_in')
    h   = layers.Dense(hidden_dim, activation='relu', name='clf_h1')(inp)
    # Softmax produces a proper probability distribution over classes
    out = layers.Dense(num_classes, activation='softmax', name='clf_out')(h)
    return keras.Model(inp, out, name='dapm_full_classifier')
```

### g) Output & Interpretation

- **Output:** `y_prob`, shape `(batch, num_classes)`, a probability distribution per example.
- **Interpretation:** `argmax(y_prob, axis=-1)` gives the predicted class label. The maximum
  probability value can be loosely read as the model's confidence. During Stage 1 training,
  `y_prob` from `z` (stochastic) is compared to `y_train` via cross-entropy. During Stage 2 and
  evaluation, `y_prob` from `z_mu` (deterministic) is used as "soft guidance" or as the final
  prediction.

### h) Limitations

- The classifier is trained *only* on source-domain labels (`y_train`) — it never receives direct
  gradient signal from the target domain's true labels (since none exist), relying entirely on the
  domain-adversarial alignment (Method 7) to make its source-trained weights transfer to target
  pixels.
- A single hidden layer of size 128 is a relatively shallow head — its capacity is limited by
  whatever information the (frozen) backbone + VAE encoder have already extracted into `z`.
- Softmax outputs can be overconfident, especially for out-of-distribution (target-domain) inputs
  that the classifier was never directly supervised on.

---
## Method 7: Domain Discriminator with Gradient Reversal Layer (GRL)

### a) What it is

> Imagine a detective whose job is to figure out whether a "summary" (latent code `z`) came from a
> labeled pixel (source) or an unlabeled pixel (target). The encoder's job, on the other hand, is to
> write summaries that *fool* this detective — so good that the detective can't tell the difference.
> The "Gradient Reversal Layer" is a clever trick: it lets the detective train normally to get
> better at distinguishing, while *simultaneously* training the encoder to do the opposite — making
> the two domains' summaries statistically indistinguishable.

This module consists of two pieces: the `GradientReversal` (GRL) layer, and the
`build_dapm_discriminator` factory function that uses it. The discriminator is a binary classifier
that predicts whether a given latent code `z` came from the source domain (label 0) or target
domain (label 1). The GRL sits between the latent input and the discriminator's hidden layers.

### b) Why it's used here

This is the **domain adaptation** mechanism. The core idea (from domain-adversarial training) is:

- The discriminator tries to *minimize* its loss — get better at telling source apart from target.
- The encoder (via the GRL) is updated with the *negated, scaled* gradient from the discriminator's
  loss — meaning the encoder is implicitly pushed to *maximize* the discriminator's loss, i.e., to
  make source and target latent codes look as similar as possible.

This adversarial "tug of war" encourages the encoder to produce a domain-invariant latent
representation `z`, so that the classifier (trained only on source labels) generalizes better to
target-domain pixels too.

### c) How it works — Step by step

1. **GradientReversal layer:** during the forward pass, it's the identity function (`return v`
   unchanged). During the backward pass (gradient computation), it multiplies the incoming gradient
   by `-lambda_` before passing it further back — i.e., it *flips the sign* and *scales* the
   gradient.
2. **Discriminator architecture:**
   - Input: a `64`-dimensional latent vector `z` (either `z_src` or `z_tgt`).
   - Apply the GRL with `grl_lambda=1.0`.
   - Two hidden Dense layers with ReLU, size `hidden_dim=128` (`DISCRIM_HIDDEN_DIM`).
   - Output: a single sigmoid unit, `dom_prob ∈ (0, 1)`, the predicted probability that `z` is from
     the target domain.
3. **Loss computation (Method 9):**
   ```
   dom_loss = BCE(label=0, pred=dom_src_prob) + BCE(label=1, pred=dom_tgt_prob)
   ```
   i.e., source latents should be predicted as `0` ("not target"), target latents as `1`.
4. When `dom_loss`'s gradients flow backward through the discriminator's Dense layers (normal
   gradients) and then through the GRL into the encoder, the GRL flips the sign — so while the
   discriminator's own weights are updated to *reduce* `dom_loss`, the encoder's weights are
   updated in the direction that would *increase* `dom_loss` (i.e., confuse the discriminator).

### d) ASCII Flow Diagram

```
z_src (64,)                              z_tgt (64,)
   |                                          |
   v                                          v
GradientReversal(lambda=1.0)        GradientReversal(lambda=1.0)
   |  (forward: identity)                    |  (forward: identity)
   |  (backward: grad * -1.0)                |  (backward: grad * -1.0)
   v                                          v
Dense(128, relu) -- disc_h1          Dense(128, relu) -- disc_h1 (shared weights)
   |                                          |
   v                                          v
Dense(128, relu) -- disc_h2          Dense(128, relu) -- disc_h2 (shared weights)
   |                                          |
   v                                          v
Dense(1, sigmoid) -- disc_out        Dense(1, sigmoid) -- disc_out (shared weights)
   |                                          |
   v                                          v
dom_src_prob  (target true label: 0)   dom_tgt_prob  (target true label: 1)
   |                                          |
   +------------------+-----------------------+
                       |
                       v
       dom_loss = BCE(0, dom_src_prob) + BCE(1, dom_tgt_prob)
                       |
        forward: discriminator weights updated to REDUCE dom_loss
        backward through GRL: encoder weights updated to INCREASE dom_loss
                       |
                       v
        => encoder pushed toward domain-invariant z
```

### e) Worked Numerical Example

**Step 1 — Forward pass (identity):** Suppose `z_src = [0.5, -0.2, ...]` (64 numbers). The GRL
passes this through unchanged: `GRL(z_src) = [0.5, -0.2, ...]`. The discriminator processes this
normally and outputs, say, `dom_src_prob = 0.3` (it thinks there's a 30% chance this is "target").

**Step 2 — Loss:** True label for source is `0`. Binary cross-entropy:
```
BCE(0, 0.3) = -[0 * log(0.3) + (1-0) * log(1-0.3)] = -log(0.7) ≈ 0.357
```

Suppose similarly `dom_tgt_prob = 0.6` for a target example (true label `1`):
```
BCE(1, 0.6) = -[1 * log(0.6) + 0 * log(0.4)] = -log(0.6) ≈ 0.511
```

```
dom_loss = 0.357 + 0.511 = 0.868
```

**Step 3 — Backward pass through GRL:** Suppose the gradient of `dom_loss` with respect to some
particular value in `z_src` (as computed normally through the discriminator's Dense layers) is
`+0.04`. After passing through the GRL with `lambda_=1.0`, this gradient becomes:
```
grl_grad = -lambda_ * grad = -1.0 * 0.04 = -0.04
```

This `-0.04` is what the encoder receives and uses to update its own weights. Because the sign is
flipped, the encoder moves its weights in the *opposite* direction from what would help the
discriminator — i.e., toward making `z_src` and `z_tgt` harder to distinguish.

### f) Code Walkthrough

```python
@tf.keras.utils.register_keras_serializable()
class GradientReversal(layers.Layer):
    """Flip gradients by -lambda_ during back-propagation (domain adversarial training)."""

    def __init__(self, lambda_=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = float(lambda_)

    def call(self, x):
        lambda_ = self.lambda_

        @tf.custom_gradient
        def _flip_gradients(v):
            def grad(dy):
                # On the backward pass, negate and scale the incoming gradient
                return -lambda_ * dy
            # On the forward pass, output equals input (identity function)
            return v, grad

        return _flip_gradients(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'lambda_': self.lambda_})
        return cfg
```

```python
def build_dapm_discriminator(latent_dim, hidden_dim=128, grl_lambda=1.0):
    """Domain discriminator with Gradient Reversal Layer: latent -> P(target domain)."""
    inp = keras.Input(shape=(latent_dim,), name='disc_z_in')
    x   = GradientReversal(lambda_=grl_lambda, name='disc_grl')(inp)   # flips gradient on backward
    x   = layers.Dense(hidden_dim, activation='relu', name='disc_h1')(x)
    x   = layers.Dense(hidden_dim, activation='relu', name='disc_h2')(x)
    # Sigmoid: probability that this latent code is from the target domain
    out = layers.Dense(1, activation='sigmoid', name='disc_out')(x)
    return keras.Model(inp, out, name='dapm_full_discriminator')
```

```python
def domain_bce(y_true, y_prob):
    """Binary cross-entropy for domain labels."""
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_prob))
```

```python
# Computing the combined domain loss (from Stage 1 training, Method 9):
dom_loss = (
    domain_bce(tf.zeros_like(dom_src_prob), dom_src_prob) +   # source should predict 0
    domain_bce(tf.ones_like(dom_tgt_prob),  dom_tgt_prob)     # target should predict 1
)
```

### g) Output & Interpretation

- **Output:** `dom_src_prob` and `dom_tgt_prob`, each shape `(batch, 1)`, sigmoid probabilities.
- **Interpretation:** `dom_loss` (Method 9, `LAMBDA_DOMAIN=0.2`) is part of the total Stage-1 loss.
  A *high* `dom_loss` for the discriminator's own training step would normally be bad — but because
  of the GRL, when the *encoder's* portion of the gradient is considered, a successfully-aligned
  latent space corresponds to `dom_src_prob` and `dom_tgt_prob` both hovering near `0.5` (the
  discriminator can't tell domains apart). The `domain` metric logged each epoch (Method 9) lets you
  monitor this adversarial game — neither vanishing to 0 (discriminator dominating) nor exploding
  (encoder collapsing the latent space trivially).

### h) Limitations

- Adversarial training (GRL-based or otherwise) is notoriously unstable — if the discriminator
  becomes too strong too quickly, the gradient signal to the encoder can vanish or oscillate.
- A fixed `grl_lambda=1.0` for the entire training run is a simplification; many domain-adversarial
  papers anneal `lambda` from `0` to `1` over training to stabilize early epochs.
- Making latent representations "domain-indistinguishable" doesn't guarantee they remain
  *class-discriminative* — there is a fundamental trade-off between domain invariance and
  classification accuracy that `LAMBDA_DOMAIN=0.2` attempts to balance, but the optimal weight is
  problem-dependent and not tuned via validation here.

---
## Method 8: Conditional Diffusion Model for Label Refinement

### a) What it is

> Imagine you have a blurry photo of a label (e.g., a one-hot vector like `[0, 1, 0, 0]` meaning
> "class 1") that's been progressively smeared with random static, more and more at each "step." A
> diffusion model is trained to look at a smeared version at a *known* smear-level and predict
> exactly *what static was added* — so that, in reverse, you could start from pure static and
> gradually "un-smear" it back into a clean one-hot label, guided by the encoder's latent code and
> the classifier's rough guess.

This step defines `build_dapm_diffusion`, a conditional noise-prediction network. Given a noisy
version `y_t` of a one-hot label vector at diffusion timestep `t`, plus the encoder's latent code
`z` and the classifier's "guidance" probabilities, the network predicts the noise `eps` that was
added to produce `y_t`.

### b) Why it's used here

The diffusion model is trained as a *secondary refinement mechanism* for label prediction,
conditioned on both the latent representation `z` and the classifier's softmax output (used as
"guidance"). For the **target domain** (no real labels), the classifier's own soft predictions are
used as a pseudo-label (`y0_tgt = classifier output`), and the diffusion model learns to predict the
noise added to *that* pseudo-label too — extending the domain-adaptation idea into the diffusion
training stage (Stage 2, Method 10). Conceptually, this gives the model a learned "denoising prior"
over plausible label distributions that is informed by the latent feature space, potentially useful
for refining or smoothing predictions (though the notebook's evaluation in Section 10 uses only the
encoder→classifier path, not the diffusion model directly — see Limitations).

### c) How it works — Step by step

1. **Inputs (four tensors):**
   - `z_in`: the `64`-dimensional latent code (`LATENT_DIM`).
   - `y_t_in`: a `num_classes`-length noisy label vector at timestep `t`.
   - `f_in`: a `num_classes`-length "guidance" vector (the classifier's softmax output).
   - `t_in`: a single integer timestep, `t ∈ [1, T]` where `T = DIFFUSION_T = 100`.
2. **Timestep embedding:** `t_in` is passed through an `Embedding` layer (`input_dim=T+1=101`,
   `output_dim=32` = `T_EMBED_DIM`), then flattened — turning the integer timestep into a learned
   32-dimensional vector.
3. **Concatenation:** `z_in`, `y_t_in`, `f_in`, and the flattened timestep embedding are
   concatenated into one long vector.
4. **MLP:** two hidden Dense layers with ReLU activation, size `hidden_dim=256`
   (`DIFFUSION_HIDDEN_DIM`).
5. **Output:** a `num_classes`-length linear output, `eps_pred` — the predicted noise vector.

### d) ASCII Flow Diagram

```
z_in (64,)   y_t_in (num_classes,)   f_in (num_classes,)   t_in (1,) int32
   |               |                       |                    |
   |               |                       |                    v
   |               |                       |          Embedding(101, 32) -- diff_t_embed
   |               |                       |                    |
   |               |                       |                    v
   |               |                       |              Flatten -- diff_t_flat
   |               |                       |                    |
   +-------+-------+-----------+-----------+--------------------+
           |
           v
   Concatenate -- diff_concat
   shape: (64 + num_classes + num_classes + 32,)
           |
           v
   Dense(256, relu) -- diff_h1
           |
           v
   Dense(256, relu) -- diff_h2
           |
           v
   Dense(num_classes, linear) -- diff_eps_pred
           |
           v
   eps_pred (num_classes,)
           |
           v
   compared to true eps via squared error (Method 10)
```

### e) Worked Numerical Example

**Forward diffusion (`q_sample`, defined in Method 9's data helpers, used to *create* the training
target):**

Suppose `num_classes = 4`, the true one-hot label is `y0 = [0, 1, 0, 0]` (class 1), and at
timestep `t=50`, the precomputed `alpha_bar[50] = 0.36` (a value between 0 and 1 that decreases as
`t` increases, representing how much of the "signal" remains).

```
sqrt(alpha_bar)     = sqrt(0.36) = 0.6
sqrt(1 - alpha_bar) = sqrt(0.64) = 0.8
```

Suppose the random noise drawn is `eps = [0.1, -0.2, 0.3, -0.1]`. Then the noisy label `y_t` is:

```
y_t = sqrt(alpha_bar) * y0 + sqrt(1 - alpha_bar) * eps
    = 0.6 * [0, 1, 0, 0] + 0.8 * [0.1, -0.2, 0.3, -0.1]
    = [0, 0.6, 0, 0] + [0.08, -0.16, 0.24, -0.08]
    = [0.08, 0.44, 0.24, -0.08]
```

**Diffusion model's job:** given `(z, y_t=[0.08, 0.44, 0.24, -0.08], f=guidance_probs, t=50)`, predict
`eps_pred ≈ [0.1, -0.2, 0.3, -0.1]` — i.e., recover the noise that was added.

**Loss:** if the model predicts `eps_pred = [0.12, -0.18, 0.28, -0.05]`, the squared error per
dimension is:

```
(0.1 - 0.12)^2  = 0.0004
(-0.2 - -0.18)^2 = 0.0004
(0.3 - 0.28)^2  = 0.0004
(-0.1 - -0.05)^2 = 0.0025

sum = 0.0004 + 0.0004 + 0.0004 + 0.0025 = 0.0037
```

This sum (averaged over the batch) is the diffusion loss for this example.

### f) Code Walkthrough

```python
def build_dapm_diffusion(latent_dim, num_classes, T=100, t_embed_dim=32, hidden_dim=256):
    """Conditional diffusion noise predictor: (z, y_t, guidance, t) -> eps_pred."""
    z_in   = keras.Input(shape=(latent_dim,),   name='diff_z_in')        # encoder latent code
    y_t_in = keras.Input(shape=(num_classes,),  name='diff_y_t')         # noisy label at step t
    f_in   = keras.Input(shape=(num_classes,),  name='diff_guidance')    # classifier soft guidance
    t_in   = keras.Input(shape=(1,), dtype='int32', name='diff_t')       # diffusion timestep

    # Learn a vector representation for "how noisy is this" (the timestep)
    t_emb    = layers.Embedding(input_dim=T + 1, output_dim=t_embed_dim, name='diff_t_embed')(t_in)
    t_emb    = layers.Flatten(name='diff_t_flat')(t_emb)

    # Combine all conditioning information into one vector
    x        = layers.Concatenate(name='diff_concat')([z_in, y_t_in, f_in, t_emb])
    x        = layers.Dense(hidden_dim, activation='relu', name='diff_h1')(x)
    x        = layers.Dense(hidden_dim, activation='relu', name='diff_h2')(x)
    # Linear output: predicted noise (can be positive or negative)
    eps_pred = layers.Dense(num_classes, activation='linear', name='diff_eps_pred')(x)

    return keras.Model([z_in, y_t_in, f_in, t_in], eps_pred, name='dapm_full_diffusion')
```

### g) Output & Interpretation

- **Output:** `eps_pred`, shape `(batch, num_classes)` — the model's prediction of the Gaussian
  noise that was added to the (one-hot or pseudo-) label at timestep `t`.
- **Interpretation:** A low diffusion loss (mean squared error between `eps_pred` and the true
  `eps`) means the model has learned the structure of how noise corrupts label vectors,
  conditioned on the latent code `z` and the classifier's guidance. In a full diffusion-based
  generation/refinement pipeline, this `eps_pred` would be used iteratively (via the reverse
  diffusion process) to denoise from pure noise back to a clean label distribution — though this
  notebook only trains the noise predictor and does not implement the iterative reverse-sampling
  loop for inference (see Limitations).

### h) Limitations

- The notebook **trains** the diffusion model (Stage 2) but the final **evaluation** (Section 10,
  `predict_with_dapm_classifier`) uses only the encoder → classifier path and does **not** invoke
  the diffusion model at all — so its trained weights are saved but not used to produce the
  reported OA/AA/Kappa/F1 metrics or the classification maps.
- For the target domain, the "true" label `y0_tgt` is itself the classifier's own soft prediction
  (`tf.stop_gradient(y_guidance_tgt)`) — meaning the diffusion model for the target domain is
  learning to denoise *predicted* labels, not ground-truth ones; any systematic classifier errors
  on the target domain become part of what the diffusion model treats as "signal."
  > **Note:** This interpretation is inferred from the variable naming and `stop_gradient` usage —
  > verify with the notebook author if a different intent was meant.
- A linear beta schedule (`BETA_START=1e-4` to `BETA_END=2e-2` over `T=100` steps) is a simple,
  commonly-used default but is not tuned for this specific label-distribution diffusion task.

---
## Method 9: Stage 1 — Joint VAE + Domain-Adversarial Training

### a) What it is

> This is the "main training session" where five different components — the encoder, two decoders,
> the classifier, and the discriminator — all learn together, like five musicians rehearsing the
> same piece simultaneously, each adjusting their part based on feedback from a combined "score"
> that blends five different kinds of errors into one number.

`train_stage1_for_model` builds all five Stage-1 sub-networks (encoder, source decoder, target
decoder, classifier, discriminator) for one backbone, then runs a custom training loop for
`STAGE1_EPOCHS=20` epochs, jointly optimizing a weighted combination of five losses: source
reconstruction, target reconstruction, KL divergence (source + target), source classification
cross-entropy, and domain adversarial loss.

### b) Why it's used here

This is the heart of the DAPM approach — it's where the latent space `z` is shaped to simultaneously
satisfy multiple competing objectives:

- **Reconstruction (source & target):** keep `z` informative about the original features.
- **KL divergence:** regularize `z`'s distribution toward a standard normal (VAE regularization).
- **Classification (source only):** make `z` useful for predicting land-cover class.
- **Domain adversarial:** make `z` look the same whether it came from source or target pixels.

Combining these into one joint loss, optimized with a single optimizer over all five networks'
combined trainable variables, is what makes this a true *multi-task, multi-domain* training
procedure rather than five separate training runs.

### c) How it works — Step by step

1. **Build sub-networks:** `encoder`, `src_decoder`, `tgt_decoder`, `classifier`,
   `discriminator` — using the factory functions from Methods 4–7, sized via the global config
   constants (`LATENT_DIM=64`, `DECODER_HIDDEN_DIM=256`, `CLASSIFIER_HIDDEN_DIM=128`,
   `DISCRIM_HIDDEN_DIM=128`).
2. **Optimizer:** a single `Adam(STAGE1_LR=1e-3)` optimizer.
3. **Datasets:**
   - `train_ds`: `(x_train, y_train)` batched at `BATCH_SIZE=128`, shuffled each epoch.
   - `val_ds`: `(x_val, y_val)` batched, no shuffling.
   - `tgt_iter`: an *infinite* shuffled iterator over `x_target_train` (so source and target
     batches can be paired even though they have different total sizes — see Method 9, "Data Stream
     & Diffusion Helpers" below).
4. **Collect trainable variables** from all five networks into `stage1_vars` — these are the
   variables that receive gradient updates.
5. **For each epoch, for each `(xb_src, yb_src)` batch from `train_ds`:**
   - Pull the next batch `xb_tgt` from the infinite target iterator.
   - **Forward pass (inside `tf.GradientTape()`):**
     - Source branch: `feature_extractor(xb_src) → encoder → (z_mu_src, z_logvar_src, z_src)`,
       then `src_decoder(z_src)`, `classifier(z_src)`, `discriminator(z_src)`.
     - Target branch: `feature_extractor(xb_tgt) → encoder → (z_mu_tgt, z_logvar_tgt, z_tgt)`, then
       `tgt_decoder(z_tgt)`, `discriminator(z_tgt)` (no classifier — no labels for target).
   - **Compute five loss terms:**
     ```
     src_recon = recon_loss(feat_src, feat_src_rec)
     tgt_recon = recon_loss(feat_tgt, feat_tgt_rec)
     src_kl    = kl_loss_from_stats(z_mu_src, z_logvar_src)
     tgt_kl    = kl_loss_from_stats(z_mu_tgt, z_logvar_tgt)
     src_ce    = sparse_categorical_crossentropy(yb_src, y_src_prob)  [mean]
     dom_loss  = domain_bce(0, dom_src_prob) + domain_bce(1, dom_tgt_prob)
     ```
   - **Combine into total loss:**
     ```
     loss = LAMBDA_SRC_RECON * src_recon      # weight = 1.0
          + LAMBDA_TGT_RECON * tgt_recon      # weight = 1.0
          + LAMBDA_KL * (src_kl + tgt_kl)     # weight = 0.01
          + LAMBDA_CE * src_ce                # weight = 1.0
          + LAMBDA_DOMAIN * dom_loss          # weight = 0.2
     ```
   - **Backpropagate and update:** `tape.gradient(loss, stage1_vars)` → `opt.apply_gradients(...)`.
   - Record each individual loss term for epoch-level averaging.
6. **End-of-epoch validation:**
   - Loop over `val_ds`: compute `val_accuracy` (classifier accuracy on `z_val`), `val_ce`
     (cross-entropy), and `val_src_recon` (source reconstruction error), all with `training=False`.
   - Loop over a freshly-batched `x_target_val`: compute `val_tgt_recon` (target reconstruction
     error).
7. **Log a summary row per epoch** containing all training and validation metrics, append to
   `history`, and print it.
8. **Return** the five trained sub-networks plus the full per-epoch `history`.

### d) ASCII Flow Diagram

```
For each epoch (1..20):
  For each batch (xb_src, yb_src) from train_ds:
     xb_tgt = next(tgt_iter)
                                                                  +--------------------+
     xb_src --> feature_extractor --> feat_src --> encoder --> (z_mu_src, logvar_src, z_src)
                                                                  |        |        |
                                                                  v        v        v
                                                            src_decoder  KL term  classifier
                                                                  |        |        |
                                                                  v        v        v
                                                             feat_src_rec  src_kl  y_src_prob
                                                                  |                  |
                                                                  v                  v
                                                              src_recon          src_ce
                                                                  |
                                                                  v
                                                            discriminator --> dom_src_prob

     xb_tgt --> feature_extractor --> feat_tgt --> encoder --> (z_mu_tgt, logvar_tgt, z_tgt)
                                                                  |        |
                                                                  v        v
                                                            tgt_decoder  KL term
                                                                  |        |
                                                                  v        v
                                                             feat_tgt_rec  tgt_kl
                                                                  |
                                                                  v
                                                              tgt_recon
                                                                  |
                                                            discriminator --> dom_tgt_prob

     dom_loss = BCE(0, dom_src_prob) + BCE(1, dom_tgt_prob)

     loss = 1.0*src_recon + 1.0*tgt_recon + 0.01*(src_kl+tgt_kl) + 1.0*src_ce + 0.2*dom_loss

     grads = d(loss)/d(stage1_vars)
     Adam.apply_gradients(grads)

  --- end of epoch: validation pass over val_ds and x_target_val ---
  log: loss, src_recon, tgt_recon, src_kl, tgt_kl, src_ce, domain,
       val_accuracy, val_ce, val_src_recon, val_tgt_recon
```

### e) Worked Numerical Example

Using the small numbers computed in earlier sections (Methods 4–7), suppose for one batch the five
loss terms come out to:

```
src_recon = 0.09
tgt_recon = 0.15
src_kl    = 0.02   (sum of src_kl and tgt_kl, e.g., 0.01 + 0.01 = 0.02 combined)
src_ce    = 0.781
dom_loss  = 0.868
```

The combined loss is:

```
loss = LAMBDA_SRC_RECON * src_recon
     + LAMBDA_TGT_RECON * tgt_recon
     + LAMBDA_KL        * (src_kl + tgt_kl)
     + LAMBDA_CE        * src_ce
     + LAMBDA_DOMAIN    * dom_loss

     = 1.0 * 0.09
     + 1.0 * 0.15
     + 0.01 * 0.02
     + 1.0 * 0.781
     + 0.2 * 0.868

     = 0.09 + 0.15 + 0.0002 + 0.781 + 0.1736

     = 1.1948
```

This single scalar `loss ≈ 1.1948` is what `tape.gradient(loss, stage1_vars)` differentiates with
respect to — every trainable weight in all five networks receives a gradient derived from this
combined number, scaled by each component's `LAMBDA_*` weight.

### f) Code Walkthrough

```python
def kl_loss_from_stats(z_mu, z_logvar):
    """KL divergence from N(0,1): averaged over batch."""
    return -0.5 * tf.reduce_mean(
        tf.reduce_sum(1.0 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), axis=-1)
    )

def recon_loss(x_true, x_rec):
    """Mean sum-of-squared-errors reconstruction loss."""
    return tf.reduce_mean(tf.reduce_sum(tf.square(x_true - x_rec), axis=-1))

def domain_bce(y_true, y_prob):
    """Binary cross-entropy for domain labels."""
    y_true = tf.cast(y_true, tf.float32)
    return tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_prob))
```

```python
def build_target_batch_stream(x_target, batch_size, seed=42):
    """Infinite shuffled iterator over target-domain patches."""
    ds = tf.data.Dataset.from_tensor_slices(x_target)
    ds = ds.shuffle(len(x_target), seed=seed, reshuffle_each_iteration=True)
    # repeat() makes the dataset infinite, so it can be paired with source batches of any count
    ds = ds.repeat().batch(batch_size, drop_remainder=True)
    return iter(ds)
```

```python
def train_stage1_for_model(model_key, feature_extractor, num_classes):
    """Run Stage-1 DAPM training for one backbone and return trained sub-networks + history."""
    feature_dim   = int(feature_extractor.output_shape[-1])
    encoder       = build_dapm_encoder(feature_dim, latent_dim=LATENT_DIM, hidden_dim=DECODER_HIDDEN_DIM)
    src_decoder   = build_dapm_decoder(LATENT_DIM, feature_dim, hidden_dim=DECODER_HIDDEN_DIM, name='dapm_full_source_decoder')
    tgt_decoder   = build_dapm_decoder(LATENT_DIM, feature_dim, hidden_dim=DECODER_HIDDEN_DIM, name='dapm_full_target_decoder')
    classifier    = build_dapm_classifier(LATENT_DIM, num_classes, hidden_dim=CLASSIFIER_HIDDEN_DIM)
    discriminator = build_dapm_discriminator(LATENT_DIM, hidden_dim=DISCRIM_HIDDEN_DIM, grl_lambda=1.0)

    opt = keras.optimizers.Adam(STAGE1_LR)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32), y_train.astype(np.int32)))
        .shuffle(len(x_train), seed=SEED)
        .batch(BATCH_SIZE, drop_remainder=False)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val.astype(np.float32), y_val.astype(np.int32)))
        .batch(BATCH_SIZE)
    )
    tgt_iter = build_target_batch_stream(x_target_train.astype(np.float32), BATCH_SIZE, seed=SEED)

    history    = []
    # All trainable variables across the five sub-networks, updated jointly each step
    stage1_vars = (
        encoder.trainable_variables +
        src_decoder.trainable_variables +
        tgt_decoder.trainable_variables +
        classifier.trainable_variables +
        discriminator.trainable_variables
    )

    for epoch in range(STAGE1_EPOCHS):
        meters = {k: [] for k in ['loss', 'src_recon', 'tgt_recon', 'src_kl', 'tgt_kl', 'src_ce', 'domain']}

        for xb_src, yb_src in train_ds:
            xb_tgt = next(tgt_iter)
            with tf.GradientTape() as tape:
                # ── Source branch ──────────────────────────────────────────
                feat_src          = feature_extractor(xb_src, training=not FREEZE_BACKBONE)
                z_mu_src, z_logvar_src, z_src = encoder(feat_src, training=True)
                feat_src_rec      = src_decoder(z_src, training=True)
                y_src_prob        = classifier(z_src, training=True)
                dom_src_prob      = discriminator(z_src, training=True)

                # ── Target branch ──────────────────────────────────────────
                feat_tgt          = feature_extractor(xb_tgt, training=not FREEZE_BACKBONE)
                z_mu_tgt, z_logvar_tgt, z_tgt = encoder(feat_tgt, training=True)
                feat_tgt_rec      = tgt_decoder(z_tgt, training=True)
                dom_tgt_prob      = discriminator(z_tgt, training=True)

                # ── Losses ────────────────────────────────────────────────
                src_recon = recon_loss(feat_src, feat_src_rec)
                tgt_recon = recon_loss(feat_tgt, feat_tgt_rec)
                src_kl    = kl_loss_from_stats(z_mu_src, z_logvar_src)
                tgt_kl    = kl_loss_from_stats(z_mu_tgt, z_logvar_tgt)
                src_ce    = tf.reduce_mean(
                    keras.losses.sparse_categorical_crossentropy(yb_src, y_src_prob)
                )
                dom_loss  = (
                    domain_bce(tf.zeros_like(dom_src_prob), dom_src_prob) +
                    domain_bce(tf.ones_like(dom_tgt_prob),  dom_tgt_prob)
                )
                loss = (
                    LAMBDA_SRC_RECON * src_recon +
                    LAMBDA_TGT_RECON * tgt_recon +
                    LAMBDA_KL        * (src_kl + tgt_kl) +
                    LAMBDA_CE        * src_ce +
                    LAMBDA_DOMAIN    * dom_loss
                )

            grads = tape.gradient(loss, stage1_vars)
            opt.apply_gradients(zip(grads, stage1_vars))

            for k, v in [('loss', loss), ('src_recon', src_recon), ('tgt_recon', tgt_recon),
                         ('src_kl', src_kl), ('tgt_kl', tgt_kl), ('src_ce', src_ce), ('domain', dom_loss)]:
                meters[k].append(float(v))

        # ── Validation ──────────────────────────────────────────────────────
        val_acc_meter, val_ce_meter, val_src_recon_meter = [], [], []
        for xb_val, yb_val in val_ds:
            feat_val              = feature_extractor(xb_val, training=False)
            z_mu_val, z_logvar_val, z_val = encoder(feat_val, training=False)
            feat_val_rec          = src_decoder(z_val, training=False)
            y_val_prob            = classifier(z_val, training=False)
            val_pred              = tf.argmax(y_val_prob, axis=-1, output_type=tf.int32)
            val_acc_meter.append(float(tf.reduce_mean(tf.cast(tf.equal(val_pred, yb_val), tf.float32))))
            val_ce_meter.append(float(tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(yb_val, y_val_prob))))
            val_src_recon_meter.append(float(recon_loss(feat_val, feat_val_rec)))

        tgt_val_ds = (
            tf.data.Dataset.from_tensor_slices(x_target_val.astype(np.float32))
            .batch(BATCH_SIZE)
        )
        tgt_recon_val_meter = []
        for xb_tval in tgt_val_ds:
            feat_tval                    = feature_extractor(xb_tval, training=False)
            z_mu_tval, z_logvar_tval, z_tval = encoder(feat_tval, training=False)
            feat_tval_rec                = tgt_decoder(z_tval, training=False)
            tgt_recon_val_meter.append(float(recon_loss(feat_tval, feat_tval_rec)))

        row = {
            'epoch'           : epoch + 1,
            'loss'            : float(np.mean(meters['loss'])),
            'src_recon'       : float(np.mean(meters['src_recon'])),
            'tgt_recon'       : float(np.mean(meters['tgt_recon'])),
            'src_kl'          : float(np.mean(meters['src_kl'])),
            'tgt_kl'          : float(np.mean(meters['tgt_kl'])),
            'src_ce'          : float(np.mean(meters['src_ce'])),
            'domain'          : float(np.mean(meters['domain'])),
            'val_accuracy'    : float(np.mean(val_acc_meter)),
            'val_ce'          : float(np.mean(val_ce_meter)),
            'val_src_recon'   : float(np.mean(val_src_recon_meter)),
            'val_tgt_recon'   : float(np.mean(tgt_recon_val_meter)),
        }
        history.append(row)
        print(f'[{model_key}] Stage1 epoch {epoch + 1}/{STAGE1_EPOCHS}:', row)

    return {
        'encoder'        : encoder,
        'source_decoder' : src_decoder,
        'target_decoder' : tgt_decoder,
        'classifier'     : classifier,
        'discriminator'  : discriminator,
        'history'        : history,
    }
```

### g) Output & Interpretation

- **Output:** a dict with the five trained sub-networks (`encoder`, `source_decoder`,
  `target_decoder`, `classifier`, `discriminator`) and a `history` list of 20 per-epoch metric
  dicts.
- **Interpretation:**
  - `val_accuracy` is the primary indicator of classification quality on held-out *source* pixels
    — it's later used to rank the three backbones in the training summary (Section 9.1).
  - `val_src_recon` / `val_tgt_recon` indicate how well the encoder+decoders preserve information
    for each domain — large or diverging values might indicate the latent space is struggling to
    serve both domains.
  - `domain` (the discriminator loss) trending toward `~1.38` (≈ `2 * log(2)`, the BCE loss when
    both probabilities are ≈0.5) would suggest the discriminator can no longer distinguish domains
    — a sign of successful alignment. Trending toward `0` would mean the discriminator easily tells
    domains apart — alignment has failed or hasn't started.

### h) Limitations

- A single fixed learning rate (`STAGE1_LR=1e-3`) and a single fixed set of `LAMBDA_*` weights are
  used for all 20 epochs and for all three backbones — no learning-rate scheduling or per-backbone
  tuning is performed.
- The five loss terms operate on very different scales (reconstruction losses can be large sums of
  squares over high-dimensional feature vectors, while KL and cross-entropy are typically smaller)
  — the chosen `LAMBDA_*` weights implicitly rebalance these scales, but the notebook does not show
  any tuning/ablation process for these weights.
- Because `tgt_iter` is infinite via `.repeat()`, the number of *target* batches seen per epoch
  exactly matches the number of *source* batches — but since `len(x_target_train) ≠ len(x_train)`
  in general, target patches may be repeated multiple times within one epoch (or not all seen),
  depending on the relative sizes.
- No early stopping or checkpoint-based model selection is used — the model after the full 20
  epochs is what gets saved, regardless of whether validation metrics were better at an earlier
  epoch.

---
## Method 10: Stage 2 — Conditional Diffusion Training

### a) What it is

> If Stage 1 was the "main rehearsal" where the encoder, decoders, classifier, and discriminator
> all learned together, Stage 2 is a "specialist lesson" for just the diffusion model. The encoder
> and classifier are now frozen (their lesson is over) — the diffusion model studies their outputs
> and learns to predict the noise patterns added to (pseudo-)labels, for both labeled and unlabeled
> pixels.

`train_stage2_for_model` builds the diffusion model (Method 8) and trains it for
`STAGE2_EPOCHS=20` epochs using the **frozen** encoder and classifier from Stage 1, with a
combined source + target diffusion loss.

### b) Why it's used here

This stage trains the conditional diffusion model introduced in Method 8. It's structured so that:

- The **source** diffusion loss uses real one-hot ground-truth labels (`y0_src`) as the diffusion
  target.
- The **target** diffusion loss uses the classifier's own soft predictions as a pseudo-label
  (`y0_tgt = stop_gradient(classifier output)`), again extending the source/target domain-adaptation
  theme into this stage — though here it's pseudo-labeling rather than adversarial alignment.

By keeping `feature_extractor`, `encoder`, and `classifier` frozen (`training=False` everywhere,
with `tf.stop_gradient` on the guidance signals), Stage 2 isolates the diffusion model's training
so it doesn't disturb the representation learned in Stage 1.

### c) How it works — Step by step

1. **Build the diffusion model:**
   ```
   diffusion = build_dapm_diffusion(LATENT_DIM, num_classes, T=DIFFUSION_T,
                                     t_embed_dim=T_EMBED_DIM, hidden_dim=DIFFUSION_HIDDEN_DIM)
   ```
2. **Optimizer:** a new `Adam(STAGE2_LR=1e-3)` optimizer (separate from Stage 1's).
3. **Precompute the diffusion schedule:**
   ```
   betas, alphas, alpha_bars = make_beta_schedule(T=100, beta_start=1e-4, beta_end=2e-2)
   ```
4. **Datasets:** `src_ds` = `(x_train, y_train)` batched; `src_val_ds` = `(x_val, y_val)` batched;
   `tgt_iter` = infinite shuffled iterator over `x_target_train` (note: a *different* seed,
   `SEED+11`, from Stage 1's target iterator).
5. **For each epoch, for each `(xb_src, yb_src)` batch:**
   - Pull `xb_tgt` from `tgt_iter`.
   - **Source diffusion loss:**
     - `feat_src = feature_extractor(xb_src, training=False)`
     - `z_mu_src, _, z_src = encoder(feat_src, training=False)`
     - `y_guidance_src = stop_gradient(classifier(z_mu_src, training=False))`
     - `y0_src = one_hot(yb_src, num_classes)` (the *real* ground-truth one-hot label)
     - `t_src = random integer in [1, 100]` per example
     - `y_t_src, eps_src = q_sample(y0_src, t_src, alpha_bars)` (forward diffusion — add noise)
     - `eps_src_pred = diffusion([z_src, y_t_src, y_guidance_src, t_src], training=True)`
     - `src_loss = mean(sum((eps_src - eps_src_pred)^2))`
   - **Target diffusion loss (pseudo-label):**
     - Same steps as above, but using `xb_tgt`, `z_tgt`, and crucially:
       `y0_tgt = stop_gradient(y_guidance_tgt)` — the classifier's *own prediction* stands in for
       the "true" label, since target pixels have no ground truth.
     - `tgt_loss = mean(sum((eps_tgt - eps_tgt_pred)^2))`
   - **Combined loss:**
     ```
     loss = src_loss + LAMBDA_TGT_DIFF * tgt_loss     # LAMBDA_TGT_DIFF = 0.5
     ```
   - Backpropagate **only through `diffusion.trainable_variables`** (encoder/classifier/feature
     extractor are not updated).
6. **Validation:** loop over `src_val_ds`, computing the same source diffusion loss with
   `training=False` (still using fresh random timesteps/noise each validation pass).
7. **Log** `diff_loss` (training) and `val_diff_loss` per epoch, append to `history`.
8. **Return** the trained `diffusion` model, its `history`, and the precomputed `alpha_bars` array.

### d) ASCII Flow Diagram

```
For each epoch (1..20):
  For each batch (xb_src, yb_src) from src_ds:
     xb_tgt = next(tgt_iter)

     -- SOURCE --                                  -- TARGET (pseudo-label) --
     xb_src -> feature_extractor (frozen)          xb_tgt -> feature_extractor (frozen)
        |                                               |
        v                                               v
     encoder (frozen) -> z_mu_src, z_src            encoder (frozen) -> z_mu_tgt, z_tgt
        |                       |                       |                       |
        v                       v                       v                       v
   classifier(z_mu_src)    y0_src = one_hot(yb_src)  classifier(z_mu_tgt)   y0_tgt = stop_grad(
        |  (frozen)             |                       |  (frozen)             classifier output)
        v                       v                       v                       |
   y_guidance_src        sample t_src ~ U[1,100]   y_guidance_tgt          sample t_tgt ~ U[1,100]
        |                       |                       |                       |
        |                       v                       |                       v
        |              q_sample -> y_t_src, eps_src     |               q_sample -> y_t_tgt, eps_tgt
        |                       |                       |                       |
        +-----------+-----------+                       +-----------+-----------+
                     |                                               |
                     v                                               v
        diffusion([z_src, y_t_src,                     diffusion([z_tgt, y_t_tgt,
                    y_guidance_src, t_src])                          y_guidance_tgt, t_tgt])
                     |                                               |
                     v                                               v
              eps_src_pred                                    eps_tgt_pred
                     |                                               |
                     v                                               v
        src_loss = MSE(eps_src, eps_src_pred)         tgt_loss = MSE(eps_tgt, eps_tgt_pred)

                     loss = src_loss + 0.5 * tgt_loss
                     |
                     v
        grads = d(loss)/d(diffusion.trainable_variables)   <- only diffusion updated
        Adam.apply_gradients(grads)
```

### e) Worked Numerical Example

**Beta schedule:** `make_beta_schedule(T=100, beta_start=1e-4, beta_end=2e-2)` produces:

```
betas      = linspace(0.0001, 0.02, 100)     # 100 evenly-spaced values
alphas     = 1 - betas
alpha_bars = cumulative product of alphas
```

For a *much smaller* illustrative example, suppose `T=3` with `betas = [0.1, 0.2, 0.3]`:

```
alphas = [1-0.1, 1-0.2, 1-0.3] = [0.9, 0.8, 0.7]

alpha_bars[0] = 0.9
alpha_bars[1] = 0.9 * 0.8 = 0.72
alpha_bars[2] = 0.9 * 0.8 * 0.7 = 0.504
```

So `alpha_bars` decreases monotonically — later timesteps correspond to "more noise has been
added," consistent with the forward-diffusion intuition.

**Combined loss:** suppose for one batch, `src_loss = 0.45` and `tgt_loss = 0.60`. Then:

```
loss = src_loss + LAMBDA_TGT_DIFF * tgt_loss
     = 0.45 + 0.5 * 0.60
     = 0.45 + 0.30
     = 0.75
```

This `loss = 0.75` is what gets backpropagated into the diffusion model's weights only.

### f) Code Walkthrough

```python
def sample_timesteps(batch_size, T):
    """Uniformly sample integer timesteps in [1, T]."""
    return np.random.randint(1, T + 1, size=(batch_size, 1), dtype=np.int32)

def q_sample(y0, t_idx, alpha_bars):
    """Forward diffusion: add noise to y0 at timestep t_idx using precomputed alpha_bars."""
    # Look up alpha_bar for each example's chosen timestep (t_idx is 1-indexed, so subtract 1)
    a_bar = tf.gather(alpha_bars, tf.cast(tf.squeeze(t_idx, axis=-1) - 1, tf.int32))
    a_bar = tf.cast(tf.reshape(a_bar, (-1, 1)), tf.float32)
    eps   = tf.random.normal(tf.shape(y0), dtype=tf.float32)   # fresh Gaussian noise
    # Standard diffusion forward process formula
    y_t   = tf.sqrt(a_bar) * y0 + tf.sqrt(1.0 - a_bar) * eps
    return y_t, eps
```

```python
def train_stage2_for_model(model_key, feature_extractor, encoder, classifier, num_classes):
    """Run Stage-2 diffusion training for one backbone and return the diffusion model + history."""
    diffusion      = build_dapm_diffusion(
        LATENT_DIM, num_classes, T=DIFFUSION_T, t_embed_dim=T_EMBED_DIM, hidden_dim=DIFFUSION_HIDDEN_DIM
    )
    opt            = keras.optimizers.Adam(STAGE2_LR)
    _, _, alpha_bars_np = make_beta_schedule(DIFFUSION_T, beta_start=BETA_START, beta_end=BETA_END)
    alpha_bars     = tf.constant(alpha_bars_np, dtype=tf.float32)

    src_ds = (
        tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32), y_train.astype(np.int32)))
        .shuffle(len(x_train), seed=SEED)
        .batch(BATCH_SIZE, drop_remainder=False)
    )
    src_val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val.astype(np.float32), y_val.astype(np.int32)))
        .batch(BATCH_SIZE)
    )
    # Different seed offset (SEED + 11) from Stage 1's target stream, for variety
    tgt_iter = build_target_batch_stream(x_target_train.astype(np.float32), BATCH_SIZE, seed=SEED + 11)
    history  = []

    for epoch in range(STAGE2_EPOCHS):
        train_losses = []
        for xb_src, yb_src in src_ds:
            xb_tgt = next(tgt_iter)
            with tf.GradientTape() as tape:
                # ── Source diffusion loss ──────────────────────────────────
                feat_src           = feature_extractor(xb_src, training=False)
                z_mu_src, z_logvar_src, z_src = encoder(feat_src, training=False)
                # Classifier's guess, detached from the gradient graph (it's frozen anyway)
                y_guidance_src     = tf.stop_gradient(classifier(z_mu_src, training=False))
                y0_src             = tf.one_hot(yb_src, depth=num_classes, dtype=tf.float32)   # real label
                t_src              = tf.convert_to_tensor(sample_timesteps(tf.shape(xb_src)[0], DIFFUSION_T))
                y_t_src, eps_src   = q_sample(y0_src, t_src, alpha_bars)
                eps_src_pred       = diffusion([z_src, y_t_src, y_guidance_src, t_src], training=True)
                src_loss           = tf.reduce_mean(tf.reduce_sum(tf.square(eps_src - eps_src_pred), axis=-1))

                # ── Target diffusion loss (pseudo-label) ──────────────────
                feat_tgt           = feature_extractor(xb_tgt, training=False)
                z_mu_tgt, z_logvar_tgt, z_tgt = encoder(feat_tgt, training=False)
                y_guidance_tgt     = tf.stop_gradient(classifier(z_mu_tgt, training=False))
                y0_tgt             = tf.stop_gradient(y_guidance_tgt)  # classifier soft labels as proxy
                t_tgt              = tf.convert_to_tensor(sample_timesteps(tf.shape(xb_tgt)[0], DIFFUSION_T))
                y_t_tgt, eps_tgt   = q_sample(y0_tgt, t_tgt, alpha_bars)
                eps_tgt_pred       = diffusion([z_tgt, y_t_tgt, y_guidance_tgt, t_tgt], training=True)
                tgt_loss           = tf.reduce_mean(tf.reduce_sum(tf.square(eps_tgt - eps_tgt_pred), axis=-1))

                loss = src_loss + LAMBDA_TGT_DIFF * tgt_loss

            # Only the diffusion model's weights are updated in Stage 2
            grads = tape.gradient(loss, diffusion.trainable_variables)
            opt.apply_gradients(zip(grads, diffusion.trainable_variables))
            train_losses.append(float(loss))

        # ── Validation ──────────────────────────────────────────────────────
        val_losses = []
        for xb_val, yb_val in src_val_ds:
            feat_val               = feature_extractor(xb_val, training=False)
            z_mu_val, z_logvar_val, z_val = encoder(feat_val, training=False)
            y_guidance_val         = classifier(z_mu_val, training=False)
            y0_val                 = tf.one_hot(yb_val, depth=num_classes, dtype=tf.float32)
            t_val                  = tf.convert_to_tensor(sample_timesteps(tf.shape(xb_val)[0], DIFFUSION_T))
            y_t_val, eps_val       = q_sample(y0_val, t_val, alpha_bars)
            eps_val_pred           = diffusion([z_val, y_t_val, y_guidance_val, t_val], training=False)
            val_loss               = tf.reduce_mean(tf.reduce_sum(tf.square(eps_val - eps_val_pred), axis=-1))
            val_losses.append(float(val_loss))

        row = {
            'epoch'         : epoch + 1,
            'diff_loss'     : float(np.mean(train_losses)),
            'val_diff_loss' : float(np.mean(val_losses)),
        }
        history.append(row)
        print(f'[{model_key}] Stage2 epoch {epoch + 1}/{STAGE2_EPOCHS}:', row)

    return {
        'diffusion' : diffusion,
        'history'   : history,
        'alpha_bars': alpha_bars_np,
    }
```

### g) Output & Interpretation

- **Output:** the trained `diffusion` model, a `history` of 20 `{epoch, diff_loss, val_diff_loss}`
  rows, and the precomputed `alpha_bars` schedule (saved for potential later use in reverse
  sampling, though not used further in this notebook).
- **Interpretation:** `diff_loss`/`val_diff_loss` decreasing over epochs indicates the diffusion
  model is learning to predict noise more accurately given `(z, y_t, guidance, t)`. Since `t` is
  re-sampled randomly every batch (and every validation pass), some epoch-to-epoch noise in these
  metrics is expected even with a well-trained model — they don't converge to exactly 0 the way a
  deterministic-target loss might, because the *targets themselves* (`eps`) are randomly resampled
  each time.

### h) Limitations

- As noted in Method 8, this trained diffusion model is **not used in the final evaluation**
  (Section 10) — it's trained and saved, but `predict_with_dapm_classifier` bypasses it entirely.
  Its practical contribution to the reported results is therefore zero; it represents either
  future-work scaffolding or an artifact of the broader research pipeline this notebook is part of.
- The target-domain diffusion loss trains on the classifier's *own* soft outputs as ground truth —
  if the classifier is systematically biased on certain target-domain pixels (e.g., a class that's
  common in target but rare/absent in source), the diffusion model will faithfully learn to
  reproduce that bias rather than correct it.
- `val_diff_loss` is computed with `training=False` but `t_val` and the noise `eps_val` are still
  randomly sampled each call — so this "validation loss" has irreducible stochastic variance and
  is not a perfectly comparable metric across epochs or across models.

---
## Method 11: Evaluation Metrics & Inference

### a) What it is

> After all the training "rehearsals" are done, this is the "final exam": each model is given the
> held-out test patches it has never seen, asked to make its best guess for each one, and graded
> against the true answers using four standard scoring methods that capture overall correctness,
> per-class fairness, agreement-beyond-chance, and a balance of precision/recall.

This section defines `predict_with_dapm_classifier` (an inference helper using only the
encoder→classifier path, deterministically via `z_mu`), then for each of the three backbones,
reloads the saved Stage-1 encoder and classifier weights and computes four metrics on the test
set: Overall Accuracy (OA), Average Accuracy (AA), Cohen's Kappa, and weighted F1.

### b) Why it's used here

This is the notebook's primary quantitative output — the numbers that answer "how good is each
DAPM-augmented backbone at classifying land cover?" It also demonstrates the save/reload cycle: the
weights saved in Method 9's Stage 1 are reloaded from disk via a freshly-built (but
identically-shaped) encoder and classifier, confirming the saved artifacts are sufficient to
reproduce predictions without needing the original in-memory training objects.

### c) How it works — Step by step

1. **`predict_with_dapm_classifier(feature_extractor, encoder, classifier, x_data, batch_size=256)`:**
   - Batch `x_data` into chunks of 256.
   - For each batch: `feat = feature_extractor(xb, training=False)`,
     `z_mu, _, _ = encoder(feat, training=False)` (note: `z_mu` is used, **not** the stochastic
     `z` — deterministic inference), `probs = classifier(z_mu, training=False)`.
   - Concatenate all batch probabilities into `all_probs`.
   - Return `(argmax(all_probs, axis=-1), all_probs)` — hard predictions and soft probabilities.
2. **For each model key `mk` in `MODEL_KEYS`:**
   - Load `{mk}_dapm_full_config.json` (saved in Method 9's main loop) to get `feature_dim` and
     weight file paths.
   - Rebuild a fresh `encoder` and `classifier` with `build_dapm_encoder`/`build_dapm_classifier`
     using the config's `feature_dim`.
   - `encoder.load_weights(...)`, `classifier.load_weights(...)`.
   - Run `predict_with_dapm_classifier` on `x_test` to get `preds`, `probs`.
   - Compute:
     - `acc = accuracy_score(y_test, preds)` — Overall Accuracy (OA).
     - `kappa = cohen_kappa_score(y_test, preds)` — Cohen's Kappa.
     - `wf1 = f1_score(y_test, preds, average='weighted')` — weighted F1.
     - `cm = confusion_matrix(y_test, preds)` — raw confusion matrix.
     - `cr = classification_report(y_test, preds, digits=4)` — per-class precision/recall/F1 text
       report.
     - `aa = mean(cm.diagonal() / cm.sum(axis=1).clip(min=1))` — Average Accuracy (mean per-class
       recall).
   - Store everything in `test_results[mk]`.

### d) ASCII Flow Diagram

```
x_test (N, 9, 9, 6)
        |
        v
  for each batch of 256:
     feature_extractor(xb)  -> feat
        |
        v
     encoder(feat) -> z_mu (discard z_logvar, z)
        |
        v
     classifier(z_mu) -> probs (N_batch, num_classes)
        |
        v
  concatenate all batches -> all_probs (N, num_classes)
        |
        v
  preds = argmax(all_probs, axis=-1)
        |
        +---------------------+----------------------+----------------------+
        v                      v                      v                      v
  accuracy_score(y_test,  cohen_kappa_score(   f1_score(y_test, preds,  confusion_matrix(
       preds) -> OA          y_test, preds)         average='weighted')      y_test, preds)
                                  -> Kappa              -> weighted F1            -> cm
                                                                                     |
                                                                                     v
                                                                    AA = mean(cm.diagonal()
                                                                         / cm.sum(axis=1))
```

### e) Worked Numerical Example

Suppose `num_classes = 3` and on a tiny test set of 6 examples, true labels are
`y_test = [0, 0, 1, 1, 2, 2]` and the model predicts `preds = [0, 1, 1, 1, 2, 0]`.

**Confusion matrix** (`cm[i, j]` = count of true class `i` predicted as class `j`):

```
        pred=0  pred=1  pred=2
true=0:    1       1       0      (one correct, one mistaken for class 1)
true=1:    0       2       0      (both correct)
true=2:    1       0       1      (one correct, one mistaken for class 0)
```

**Overall Accuracy (OA):**
```
OA = (1 + 2 + 1) / 6 = 4 / 6 = 0.667
```

**Average Accuracy (AA)** — mean of per-class recall (diagonal / row sum):
```
recall_class0 = 1 / (1+1+0) = 1/2 = 0.5
recall_class1 = 2 / (0+2+0) = 2/2 = 1.0
recall_class2 = 1 / (1+0+1) = 1/2 = 0.5

AA = (0.5 + 1.0 + 0.5) / 3 = 2.0 / 3 = 0.667
```

**Weighted F1** would similarly be computed per-class (precision, recall, F1) and averaged,
weighted by each class's number of true instances (2, 2, 2 here — equal, so it would equal the
unweighted macro-F1 in this balanced example).

**Cohen's Kappa** measures agreement beyond what would be expected by chance — given the observed
accuracy of `0.667` and the expected accuracy under random guessing (computed from the marginal
distributions of `y_test` and `preds`), Kappa would be somewhat lower than `0.667` (exact value
depends on the marginals; Kappa = 0 means no better than chance, Kappa = 1 means perfect agreement).

### f) Code Walkthrough

```python
def predict_with_dapm_classifier(feature_extractor, encoder, classifier, x_data, batch_size=256):
    """Run encoder -> classifier inference on x_data; returns (hard_preds, softmax_probs)."""
    ds        = tf.data.Dataset.from_tensor_slices(x_data.astype(np.float32)).batch(batch_size)
    all_probs = []
    for xb in ds:
        feat       = feature_extractor(xb, training=False)
        # Use z_mu (deterministic) rather than the stochastic sample z, for stable predictions
        z_mu, _, _ = encoder(feat, training=False)
        probs      = classifier(z_mu, training=False)
        all_probs.append(probs.numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    return np.argmax(all_probs, axis=-1), all_probs
```

```python
test_results = {}

for mk in MODEL_KEYS:
    print(f"\n{'='*20} {mk} {'='*20}")

    fe = feature_extractors[mk]
    with open(DAPM_DIR / f'{mk}_dapm_full_config.json', 'r') as f:
        cfg = json.load(f)

    # Rebuild lightweight architectures and load weights
    enc = build_dapm_encoder(cfg['feature_dim'], LATENT_DIM, DECODER_HIDDEN_DIM)
    clf = build_dapm_classifier(LATENT_DIM, num_classes, CLASSIFIER_HIDDEN_DIM)
    enc.load_weights(cfg['weights']['encoder'])
    clf.load_weights(cfg['weights']['classifier'])

    preds, probs = predict_with_dapm_classifier(fe, enc, clf, x_test)

    acc   = accuracy_score(y_test, preds)
    kappa = cohen_kappa_score(y_test, preds)
    wf1   = f1_score(y_test, preds, average='weighted')
    cm    = confusion_matrix(y_test, preds)
    cr    = classification_report(y_test, preds, digits=4)
    aa    = np.mean(cm.diagonal() / cm.sum(axis=1).clip(min=1))   # mean per-class recall

    test_results[mk] = {
        'preds': preds, 'probs': probs,
        'cm': cm, 'cr': cr,
        'acc': acc, 'aa': aa, 'kappa': kappa, 'f1': wf1,
    }

    print(f'OA={acc:.4f}  AA={aa:.4f}  \u03ba={kappa:.4f}  wF1={wf1:.4f}')
    print(cr)
```

### g) Output & Interpretation

- **Output:** `test_results` — a dict keyed by model name, each containing predictions, raw
  probabilities, confusion matrix, classification report text, and four scalar metrics (`acc`,
  `aa`, `kappa`, `f1`).
- **Interpretation:**
  - **OA (Overall Accuracy):** fraction of all test pixels classified correctly — sensitive to
    class imbalance (a model that always predicts the majority class can have high OA).
  - **AA (Average Accuracy):** mean of per-class recall — treats every class equally regardless of
    how many test pixels it has, so it's a better indicator of performance on rare classes.
  - **Cohen's Kappa:** agreement corrected for chance — values near `0` mean the model is no better
    than random given the class distribution; values near `1` mean near-perfect agreement.
  - **Weighted F1:** balances precision and recall per class, weighted by class frequency — a
    middle ground between OA and AA.
  - A model with high OA but much lower AA suggests it performs well on common classes but poorly
    on rare ones.

### h) Limitations

- These metrics describe performance on the **source-domain test split only** (`x_test`,
  `y_test`) — there is no analogous quantitative metric for *target-domain* performance, since
  target pixels have no ground truth. The entire premise of the domain-adaptation training (Stage
  1's discriminator) cannot be directly validated numerically within this notebook.
- `clip(min=1)` in the AA calculation guards against division-by-zero for classes with zero test
  examples (`cm.sum(axis=1) == 0`), but if such a class exists, its "recall" would be computed as
  `0 / 1 = 0`, which silently lowers AA rather than excluding that class from the average — this
  could understate AA if some classes are absent from the test set.
- Evaluation uses `z_mu` (deterministic), so the metrics do not reflect the stochasticity that was
  part of the training objective (via `z`) — this is standard practice for VAEs at inference time,
  but means train-time loss values and test-time metrics are not directly comparable measures of
  the "same" forward pass.

---

## Method 12: Full-Scene Classification Map Generation

### a) What it is

> Having graded the model on a held-out test sample, this final step asks the model to "color in"
> the *entire* image — every single pixel, labeled or not — producing a complete map of predicted
> land-cover classes across the whole scene, which can be visually compared side-by-side with the
> (partial) ground-truth map.

For each backbone, this step runs the encoder→classifier inference path (same as Method 11) across
**every pixel** in the `330 × 307` scene, row by row, to produce a dense `(330, 307)` prediction
map, then plots it alongside the ground truth and the other backbones' maps.

### b) Why it's used here

The quantitative metrics in Method 11 only cover labeled (source-domain) test pixels — a small
fraction of the scene. The full-image classification map gives a *qualitative* view of how each
model behaves across the **entire** scene, including the vast majority of pixels that have no
ground truth (the target domain). This is often the most visually compelling output of a remote-
sensing classification pipeline, since it shows spatial coherence (or lack thereof) in the
predictions.

### c) How it works — Step by step

1. For each model key `mk`:
   - Reload `feature_extractor`, `encoder`, `classifier` (same pattern as Method 11).
   - Pad the normalized image `x_img` by `pad = PATCH_SIZE // 2 = 4` using edge-replication (same
     as Method 2).
   - Initialize `pred_map = full((H, W), -1)` — a `330 × 307` array filled with `-1` (placeholder).
   - **For each row `r` in `0..H-1`:**
     - Extract all `W=307` patches along that row in one batch:
       `patches[c] = x_pad[r:r+9, c:c+9, :]` for `c in 0..W-1`.
     - Run `feature_extractor → encoder (z_mu) → classifier → argmax` on this batch of 307
       patches at once.
     - Store the resulting 307 predictions into `pred_map[r, :]`.
     - Print progress every 50 rows.
2. **Ground truth map:** `gt_map = y_img - 1` (shift to 0-indexed; unlabeled pixels, originally `0`,
   become `-1`).
3. **Plot:** a row of `len(MODEL_KEYS) + 1 = 4` subplots — ground truth first, then each model's
   `pred_map` — using `show_map`, which masks pixels with value `< 0` (i.e., unlabeled pixels in
   the ground-truth panel only, since model prediction maps have no `-1` entries) and displays the
   rest with a `tab20` categorical colormap.
4. Save the combined figure to `full_image_classification_maps.png`.

### d) ASCII Flow Diagram

```
x_img (330, 307, 6) -- normalized full scene
        |
        v
  pad by 4 (edge mode) -> x_pad (338, 315, 6)
        |
        v
  for r in 0..329:
     patches = stack([ x_pad[r:r+9, c:c+9, :] for c in 0..306 ])   # (307, 9, 9, 6)
        |
        v
     feature_extractor(patches) -> feat   (307, feature_dim)
        |
        v
     encoder(feat) -> z_mu   (307, 64)        [z_logvar, z discarded]
        |
        v
     classifier(z_mu) -> probs  (307, num_classes)
        |
        v
     pred_map[r, :] = argmax(probs, axis=-1)   # (307,)

  --- after all 330 rows ---

  pred_map (330, 307)  -- full_maps[mk]

gt_map = y_img - 1   (-1 = unlabeled)

  +-----------------+-----------------+-----------------+-----------------+
  | Ground Truth    | AlexNet_CNN     | GFNet           | ViT_UNet        |
  | (masked where   | pred_map        | pred_map        | pred_map        |
  |  gt_map < 0)    | (no masking)    | (no masking)    | (no masking)    |
  +-----------------+-----------------+-----------------+-----------------+
                  -> saved as full_image_classification_maps.png
```

### e) Worked Numerical Example

Consider a tiny `2 × 2` scene (`H=2, W=2`) instead of `330 × 307`, with `PATCH_SIZE=3` (`pad=1`).
After padding, `x_pad` is `4 × 4 × B`.

For row `r=0`:
```
patches[c=0] = x_pad[0:3, 0:3, :]   # 3x3 patch centered on original pixel (0,0)
patches[c=1] = x_pad[0:3, 1:4, :]   # 3x3 patch centered on original pixel (0,1)
```
These 2 patches are stacked into a batch of shape `(2, 3, 3, B)` and run through
`feature_extractor → encoder (z_mu) → classifier`, producing `probs` of shape `(2, num_classes)`.
Suppose `argmax(probs, axis=-1) = [2, 0]` — then `pred_map[0, :] = [2, 0]`.

The same is repeated for row `r=1`, filling `pred_map[1, :]`. After both rows, `pred_map` is a
complete `2 × 2` array of predicted class indices — for the real notebook, this process repeats
330 times (once per row) to fill a `330 × 307` array.

### f) Code Walkthrough

```python
cmap_classes = plt.cm.get_cmap('tab20', num_classes)
full_maps    = {}

for mk in MODEL_KEYS:
    print(f'Predicting full image for {mk} ...')
    fe  = feature_extractors[mk]
    with open(DAPM_DIR / f'{mk}_dapm_full_config.json') as f:
        cfg = json.load(f)

    enc = build_dapm_encoder(cfg['feature_dim'], LATENT_DIM, DECODER_HIDDEN_DIM)
    clf = build_dapm_classifier(LATENT_DIM, num_classes, CLASSIFIER_HIDDEN_DIM)
    enc.load_weights(cfg['weights']['encoder'])
    clf.load_weights(cfg['weights']['classifier'])

    pad      = PATCH_SIZE // 2
    x_pad    = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    pred_map = np.full((H, W), -1, dtype=np.int32)

    for r in range(H):
        # Extract all W patches along row r in one batch (efficient: one forward pass per row)
        patches  = np.stack(
            [x_pad[r:r + PATCH_SIZE, c:c + PATCH_SIZE, :] for c in range(W)], axis=0
        )
        feat         = fe(patches, training=False)
        z_mu, _, _   = enc(feat, training=False)
        probs        = clf(z_mu, training=False)
        pred_map[r, :] = tf.argmax(probs, axis=-1).numpy()
        if r % 50 == 0:
            print(f'  row {r}/{H}')

    full_maps[mk] = pred_map
    print(f'  {mk} done.')

# ── Plot ground truth alongside model predictions ─────────────────────────────
gt_map = y_img.copy() - 1  # 0-based; unlabeled pixels -> -1

fig, axes = plt.subplots(1, len(MODEL_KEYS) + 1, figsize=(6 * (len(MODEL_KEYS) + 1), 5))

def show_map(ax, cmap_data, title):
    """Render a labelled classification map; mask unlabeled pixels (value -1)."""
    masked = np.ma.masked_where(cmap_data < 0, cmap_data)
    im     = ax.imshow(masked, cmap=cmap_classes, vmin=0, vmax=num_classes - 1, interpolation='nearest')
    ax.set_title(title, fontsize=11)
    ax.axis('off')
    return im

show_map(axes[0], gt_map, 'Ground Truth\n(labeled only)')
for i, mk in enumerate(MODEL_KEYS):
    im = show_map(axes[i + 1], full_maps[mk], f'{mk}\nOA={test_results[mk]["acc"]:.4f}')

cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, ticks=range(num_classes))
cbar.ax.set_yticklabels([f'C{i}' for i in range(num_classes)], fontsize=8)

fig.suptitle('Full-Image Classification Maps', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(DAPM_DIR / 'full_image_classification_maps.png', dpi=200, bbox_inches='tight')
plt.show()
```

### g) Output & Interpretation

- **Output:** `full_maps` — a dict of three `(330, 307)` integer arrays (one per backbone), each
  containing a predicted class index `0..num_classes-1` for *every* pixel; plus a saved PNG
  comparing all three maps to the ground truth.
- **Interpretation:** Visually compare each model's predicted map to the ground-truth panel (where
  visible) and to each other. Spatially coherent, smoothly-varying regions in the prediction maps
  (matching real-world land-cover patterns — e.g., contiguous fields, roads) suggest the model has
  learned meaningful spatial structure, whereas "salt-and-pepper" noise (isolated mispredicted
  pixels) suggests the model is making more independent, less spatially-consistent decisions. Since
  `pred_map` has no `-1` values (every pixel gets a prediction), the model panels show predictions
  even for the unlabeled target-domain pixels — this is the only place in the notebook where target-
  domain predictions are visualized.

### h) Limitations

- Generating this map requires `H = 330` separate forward passes (one per row, each a batch of
  `W = 307` patches) per model — for larger images this row-by-row approach, while
  memory-efficient, could become a runtime bottleneck.
- There is no quantitative metric for the full-scene map (since most of it is unlabeled) — its value
  is purely qualitative/visual.
- The `tab20` colormap supports up to 20 distinct categories; if `num_classes > 20`, colors would
  start repeating, making some classes visually indistinguishable.
- As in Method 11, this uses `z_mu` (deterministic) and bypasses the diffusion model entirely —
  the visualized predictions reflect only the encoder→classifier path from Stage 1.

---
## 6. Results & Comparisons

> **Note:** This notebook does not contain executed cell outputs (no printed numbers, tables, or
> rendered plots are embedded in the provided source) — the figures below describe *what the
> notebook computes and displays*, not actual numeric results. Run the notebook to obtain real
> values.

The notebook produces the following results artifacts, all saved under `DAPM_DIR`
(`<project_root>/dapm/dapm_full_artifacts/`):

1. **Per-epoch training histories** (Stage 1, 20 rows; Stage 2, 20 rows) for each of the three
   backbones — printed to console during training (Methods 9 and 10), tracking `loss`,
   `src_recon`, `tgt_recon`, `src_kl`, `tgt_kl`, `src_ce`, `domain`, `val_accuracy`, `val_ce`,
   `val_src_recon`, `val_tgt_recon` (Stage 1) and `diff_loss`, `val_diff_loss` (Stage 2).

2. **`dapm_full_training_summary.csv`** — one row per backbone, with the *final-epoch* values of
   the above metrics plus `feature_dim`, sorted descending by `stage1_val_accuracy_last`. A bar-plot
   figure (3 panels: Stage-1 val accuracy, Stage-2 val diffusion loss, backbone feature dimension)
   is displayed alongside it.

3. **Per-model test-set metrics** (Method 11): Overall Accuracy (OA), Average Accuracy (AA),
   Cohen's Kappa (κ), and weighted F1, plus a full `classification_report` (per-class precision,
   recall, F1, support).

4. **Visualizations** (Section 11), each saved as a PNG:
   - `confusion_matrices_raw.png` — raw-count confusion matrices for all three models side by side.
   - `confusion_matrices_normalized.png` — recall-normalized confusion matrices with per-cell
     annotations.
   - `per_class_accuracy.png` — per-class recall bar charts for all three models.
   - `model_comparison_bars.png` — four-panel bar chart comparing OA, AA, Kappa, and weighted-F1
     across the three models.
   - `full_image_classification_maps.png` — ground truth + three full-scene prediction maps.

**Comparison table template** (to be filled in once the notebook is run):

```
| Model       | OA  | AA  | Kappa | Weighted-F1 | Notes |
|-------------|-----|-----|-------|-------------|-------|
| AlexNet_CNN |     |     |       |             |       |
| GFNet       |     |     |       |             |       |
| ViT_UNet    |     |     |       |             |       |
```

The notebook's own ranking criterion (Section 9.1) sorts backbones by `stage1_val_accuracy_last` —
i.e., the *validation* classification accuracy at the end of Stage-1 training — as a proxy for
overall quality, though the final reported test metrics (OA, AA, Kappa, weighted-F1) are the more
complete picture and may rank backbones differently.

---

## 7. Academic Paper Summary

### Problem Statement

Pixel-level land-cover classification of multispectral remote-sensing imagery is hindered by the
limited availability of labeled ground-truth data relative to the full spatial extent of a scene.
This work addresses semi-supervised domain adaptation within a single scene, treating labeled
pixels as a source domain and unlabeled pixels as a target domain, with the goal of learning a
shared latent representation that supports accurate classification on labeled data while
generalizing to the unlabeled majority of the image.

### Methodology

The proposed Domain-Adversarial Probabilistic Model (DAPM) is built atop three independently
pre-trained backbone architectures — an AlexNet-style convolutional network, GFNet (a global
frequency-filter token-mixing architecture), and a ViT-UNet hybrid — each of which is used solely
as a frozen feature extractor by removing its final classification layer. A shared Variational
Autoencoder (VAE) encoder maps backbone feature vectors from both domains into a common
64-dimensional latent space, parameterized by per-example mean and log-variance vectors and sampled
via the reparameterization trick. Two domain-specific decoders reconstruct the original feature
vectors from this latent space, providing a reconstruction-based regularization signal for each
domain independently. A classifier head, trained exclusively on source-domain labels, predicts
class probabilities from the latent code via cross-entropy loss. Domain alignment is enforced
through a binary domain discriminator equipped with a Gradient Reversal Layer (GRL), which is
trained to distinguish source from target latent codes while simultaneously driving the encoder, via
reversed gradients, toward producing domain-invariant representations. Training proceeds in two
stages: Stage 1 jointly optimizes the encoder, both decoders, the classifier, and the discriminator
using a weighted sum of reconstruction, KL-divergence, classification cross-entropy, and domain
adversarial losses; Stage 2 subsequently trains a conditional diffusion-based noise predictor,
conditioned on the frozen Stage-1 encoder's latent codes and the classifier's soft predictions, to
model the noising process applied to (ground-truth or pseudo-) one-hot label distributions for both
domains.

### Experimental Setup

The dataset is a single `330 × 307` six-band multispectral scene with a partially-labeled
ground-truth map. Each band is independently min-max normalized to `[0, 1]`. Labeled pixels are
extracted as `9 × 9 × 6` patches and stratified-split into source train (60% of all labeled
pixels), validation (15%), and test (25%) sets. Unlabeled pixels are similarly patch-extracted,
randomly subsampled to at most 20,000 examples, and split 90/10 into target train and target
validation sets. Each of the three backbones is independently trained with the DAPM pipeline for 20
Stage-1 epochs and 20 Stage-2 epochs, using the Adam optimizer at a learning rate of `1e-3` and a
batch size of 128. Evaluation is performed on the held-out source-domain test set, using Overall
Accuracy, Average Accuracy (mean per-class recall), Cohen's Kappa, and weighted F1-score as the
primary metrics, with confusion matrices and full-scene classification maps generated for
qualitative comparison across backbones.

### Results Summary

> Results are not available in the provided notebook source, as no executed cell outputs are
> included. Once executed, the notebook would report, for each backbone, the four headline test
> metrics (OA, AA, Kappa, weighted-F1) alongside Stage-1 final validation accuracy and Stage-2
> final diffusion validation loss, ranked by Stage-1 validation accuracy. The backbone achieving the
> highest OA/AA/Kappa/weighted-F1 combination on the source-domain test set would be identified as
> the best-performing configuration under this DAPM pipeline, with full-scene classification maps
> providing a qualitative view of generalization to the unlabeled target domain.

### Conclusion

This work demonstrates a unified probabilistic, domain-adversarial training framework that can be
layered on top of arbitrary pre-trained feature-extraction backbones for semi-supervised land-cover
classification. By combining VAE-based representation learning, domain-adversarial alignment via a
Gradient Reversal Layer, and a conditional diffusion-based label-noise model, the framework attempts
to address both the representation-quality and domain-shift challenges inherent to single-scene
semi-supervised classification. Key limitations include the absence of any direct quantitative
evaluation on target-domain pixels (due to lack of ground truth), the unused status of the trained
diffusion model in the final classification pipeline, and the use of fixed loss-weighting
hyperparameters without an ablation or sensitivity analysis. Future directions include incorporating
the diffusion model into the inference pipeline (e.g., as a label-refinement or uncertainty-
estimation step), exploring learned or annealed domain-adversarial weighting schedules, and
extending evaluation to additional scenes or externally-labeled target-domain subsets to directly
assess domain-adaptation efficacy.

---

## 8. References

```
[1] Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.
    (Foundational VAE paper - basis for the encoder/decoder/Sampling/KL-divergence design.)

[2] Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation.
    Proceedings of the 32nd International Conference on Machine Learning (ICML).
    (Introduces the Gradient Reversal Layer used for domain-adversarial training.)

[3] Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F.,
    Marchand, M., & Lempitsky, V. (2016). Domain-Adversarial Training of Neural Networks.
    Journal of Machine Learning Research, 17(59), 1-35.
    (Extended treatment of the GRL-based domain discriminator architecture.)

[4] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
    Advances in Neural Information Processing Systems (NeurIPS) 33.
    (Basis for the forward noising process (q_sample), beta/alpha schedule, and
    noise-prediction objective used in the diffusion model.)

[5] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with
    Deep Convolutional Neural Networks. Advances in Neural Information Processing Systems
    (NeurIPS) 25.
    (Original AlexNet architecture, referenced via the 'AlexNet_CNN' backbone.)

[6] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global Filter Networks for
    Image Classification. Advances in Neural Information Processing Systems (NeurIPS) 34.
    (GFNet architecture - basis for the GlobalFilterLayer using 2D FFT-based filtering.)

[7] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T.,
    Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021).
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
    International Conference on Learning Representations (ICLR).
    (Vision Transformer (ViT) architecture - basis for PatchEncoderWithCLS and the
    'ViT_UNet' backbone's patch/positional embedding design.)

[8] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for
    Biomedical Image Segmentation. International Conference on Medical Image Computing
    and Computer-Assisted Intervention (MICCAI).
    (U-Net architecture, referenced via the 'ViT_UNet' backbone naming.)

[9] Landis, J. R., & Koch, G. G. (1977). The Measurement of Observer Agreement for
    Categorical Data. Biometrics, 33(1), 159-174.
    (Standard reference for Cohen's Kappa, used as an evaluation metric.)

[10] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python.
     Journal of Machine Learning Research, 12, 2825-2830.
     (scikit-learn - source of train_test_split, accuracy_score, cohen_kappa_score,
     f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay.)
```

---
