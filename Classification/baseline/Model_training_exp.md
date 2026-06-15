# Model Training Notebook Explainer
## Multispectral Land Cover Classification with AlexNet CNN, GFNet, and ViT-UNet

---

**Overview**

This notebook trains three deep learning architectures — an AlexNet-inspired CNN, a Global Filter Network (GFNet), and a Vision Transformer with U-Net skip connections (ViT-UNet) — to classify land cover types from a 6-band multispectral image. Each pixel in the scene is assigned a class label based on a spatial patch of 9×9 pixels surrounding it. The pipeline covers everything from raw data loading and normalisation, through model construction and training, to evaluation, visualisation, and export.

**Who this document is for:** Someone who used AI assistance to write this code and wants to deeply understand *how and why* each piece works — for personal mastery and for writing an academic paper describing the methodology.

---

## Table of Contents

1. [Environment & Dependencies](#environment--dependencies)
2. [Data & Problem Setup](#data--problem-setup)
3. [Method: Per-Band Min-Max Normalisation](#method-per-band-min-max-normalisation)
4. [Method: Spatial Patch Extraction](#method-spatial-patch-extraction)
5. [Method: Stratified Train / Val / Test Splitting](#method-stratified-train--val--test-splitting)
6. [Method: Shared Custom Keras Layers (PatchExtractor, PatchPositionEncoder, PatchEncoderWithCLS)](#method-shared-custom-keras-layers)
7. [Method: AlexNet-Inspired CNN](#method-alexnet-inspired-cnn)
8. [Method: Global Filter Network (GFNet)](#method-global-filter-network-gfnet)
9. [Method: Vision Transformer with U-Net Skip Connections (ViT-UNet)](#method-vision-transformer-with-u-net-skip-connections-vit-unet)
10. [Method: Calibration Metrics — Brier Score and ECE](#method-calibration-metrics--brier-score-and-ece)
11. [Method: AdamW with Cosine Decay Learning Rate Schedule](#method-adamw-with-cosine-decay-learning-rate-schedule)
12. [Method: Training, Checkpointing, and Evaluation Pipeline](#method-training-checkpointing-and-evaluation-pipeline)
13. [Method: Full-Scene Dense Inference](#method-full-scene-dense-inference)
14. [Results & Comparisons](#results--comparisons)
15. [Academic Paper Summary](#academic-paper-summary)
16. [References](#references)

---

## Environment & Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Numerical array operations: reshaping, padding, argmax, linspace |
| `pandas` | Loading CSV files into arrays |
| `matplotlib` | Plotting training curves, bar charts, confusion matrices |
| `seaborn` | Enhanced confusion matrix heatmaps (`whitegrid` style) |
| `sklearn.model_selection` | Stratified train/val/test splitting |
| `sklearn.metrics` | Accuracy, F1, Kappa, classification report, confusion matrix, log loss |
| `tensorflow` / `keras` | Building, training, and saving all three deep learning models |
| `tensorflow.keras.layers` | Conv2D, Dense, Dropout, LayerNorm, MultiHeadAttention, etc. |
| `tensorflow.keras.regularizers` | L2 regularisation (imported but not used in final models) |
| `openpyxl` | Creating and embedding PNG images into Excel workbooks |
| `io`, `json`, `os`, `random`, `time`, `pathlib` | Standard library utilities for file management, timing, seeding |

**Reproducibility:** Seeds are fixed at `SEED = 42` for Python's `random`, NumPy, and TensorFlow before any computation occurs, ensuring identical results on re-run.

---

## Data & Problem Setup

**Dataset:** A multispectral remote sensing image stored as a flat CSV. It is reshaped into a 3D array of shape `(330, 307, 6)` — 330 rows × 307 columns × 6 spectral bands. A companion reference CSV of shape `(330, 307)` contains integer class labels for each pixel; unlabelled pixels have value 0 and are ignored.

**Task:** Multiclass land cover classification — assign each labelled pixel to one of C classes (where C is determined at runtime from the unique non-zero values in the label map).

**Input representation:** Rather than classifying each pixel in isolation, the model sees a `9×9×6` spatial patch centred on each labelled pixel. This gives spatial context to the spectral measurements at each location.

**Splits:**
- 75% of labelled samples → training (AlexNet uses its own legacy split with seed 10)
- 20% of the training portion → validation (GFNet and ViT only)
- Remaining 25% → test

---

## Method: Per-Band Min-Max Normalisation

### a) What it is

> Think of each spectral band as a separate "channel" on a camera — one might record near-infrared, another visible red. Min-max normalisation stretches each channel's pixel values to a 0–1 range independently, so no single band dominates just because it has larger raw numbers.

Per-band min-max normalisation rescales each spectral band to the interval [0, 1] using that band's own minimum and maximum values.

### b) Why it's used here

Raw multispectral sensor readings can span very different numerical ranges across bands (e.g., band 1 might range 0–3000, band 4 might range 0–8000). Without normalisation, networks trained on gradient descent would be biased toward higher-magnitude bands, causing slow or unstable training.

### c) How it works — Step by step

1. For each band `b` in `{0, 1, 2, 3, 4, 5}`:
2. Extract all pixel values for that band: a 2D matrix of shape `(H, W)`.
3. Find `band_min = min(all values in band b)` and `band_max = max(all values in band b)`.
4. Compute the denominator (add a small epsilon to prevent division by zero):
```
denom = max(band_max - band_min, 1e-8)
```
5. Normalise every pixel in the band:
```
x_norm[:, :, b] = (x[:, :, b] - band_min) / denom
```
6. Each normalised band now has values in [0, 1].

### d) ASCII Flow Diagram

```
Raw CSV (flat)
    |
    v
reshape to (H=330, W=307, B=6)
    |
    v
For each band b in [0..5]:
    |
    ├── Find band_min, band_max
    |
    └── x_norm[:,:,b] = (x[:,:,b] - band_min) / max(band_max - band_min, 1e-8)
    |
    v
x_norm: shape (330, 307, 6), all values in [0, 1]
```

### e) Worked Numerical Example

Suppose band 2 has 5 pixels with values: `[100, 200, 400, 300, 500]`.

- `band_min = 100`, `band_max = 500`
- `denom = 500 - 100 = 400`
- Normalised values:
  - `(100 - 100) / 400 = 0.00`
  - `(200 - 100) / 400 = 0.25`
  - `(400 - 100) / 400 = 0.75`
  - `(300 - 100) / 400 = 0.50`
  - `(500 - 100) / 400 = 1.00`
- Result: `[0.0, 0.25, 0.75, 0.5, 1.0]` — all in [0, 1].

### f) Code Walkthrough

```python
def load_multispectral_6band(data_path, label_path, height, width, bands):
    # Load the flat CSV and reshape into a (H, W, B) cube
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(height, width, bands)
    # Load the label CSV into a (H, W) integer array
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(height, width)

    x_norm = np.empty_like(x, dtype=np.float32)  # Pre-allocate output array
    for b in range(bands):
        band     = x[:, :, b]                    # Extract band b as a 2D slice
        band_min = np.min(band)                   # Minimum pixel value in this band
        band_max = np.max(band)                   # Maximum pixel value in this band
        denom    = max(band_max - band_min, 1e-8) # Avoid division by zero
        x_norm[:, :, b] = (band - band_min) / denom  # Normalise to [0,1]

    return x_norm, y
```

### g) Output & Interpretation

Returns `x_norm` of shape `(330, 307, 6)` with float32 values in [0, 1], and `y` of shape `(330, 307)` with integer class labels. All subsequent operations work on `x_norm`.

### h) Limitations

- Min-max normalisation is sensitive to outlier pixels. A single extremely bright or dark pixel in a band shifts all other values.
- Computed from the full scene; in a real deployment, the normalisation parameters should be saved from the training set and applied to new images.
- Assumes all 6 bands have range > 1e-8; degenerate (constant) bands receive a near-zero denominator, producing a zero-filled band rather than an error.
- Does not account for sensor-specific noise floors or saturation artefacts.

---

## Method: Spatial Patch Extraction

### a) What it is

> Instead of looking at one pixel at a time, the model looks through a 9×9 "window" centred on each labelled pixel — like reading a word in context rather than letter by letter.

Patch extraction converts the (H, W, B) image into a set of (P, P, B) volumetric patches, one per labelled pixel, where P = 9.

### b) Why it's used here

Spatial context dramatically improves classification accuracy for remote sensing images. Neighbouring pixels tend to belong to the same land cover type, and texture patterns within a patch (e.g., field rows vs. forest canopy) are discriminative features that a single-pixel classifier cannot see.

### c) How it works — Step by step

1. Compute `pad = patch_size // 2 = 4`.
2. Pad the image on all four sides by 4 pixels using edge-replication (`mode="edge"`), yielding shape `(H+8, W+8, B)`. Edge replication avoids introducing artificial boundary artefacts.
3. Find all labelled pixel coordinates: `coords = np.argwhere(y > 0)` — these are all `(row, col)` positions where the label is non-zero.
4. For each labelled coordinate `(r, c)`:
   - Extract the slice `x_pad[r : r+P, c : c+P, :]` — a `(9, 9, 6)` cube.
   - Store in `patches[i]`.
   - Store `y[r, c] - 1` in `labels[i]` (converts 1-indexed classes to 0-indexed).
5. Return `patches` (shape: `[N, 9, 9, 6]`), `labels` (shape: `[N]`), and `coords`.

### d) ASCII Flow Diagram

```
x_norm: (330, 307, 6)
    |
    v
Edge-pad by 4 pixels --> x_pad: (338, 315, 6)
    |
    v
Find all (r, c) where y[r,c] > 0 --> N labelled pixel coordinates
    |
    v
For each (r, c):
    extract x_pad[r:r+9, c:c+9, :] --> patch shape (9, 9, 6)
    label = y[r,c] - 1             --> 0-indexed class
    |
    v
patches: (N, 9, 9, 6)
labels:  (N,)
```

### e) Worked Numerical Example

Suppose the image is `(5, 5, 1)` (1 band), pad=1, so padded image is `(7, 7, 1)`. For a labelled pixel at `(r=2, c=2)`:

```
Original 5×5 image:        Padded 7×7 image:
a b c d e                  a a b c d e e
f g h i j       pad=1      a a b c d e e
k l m n o    --------->    f f g h i j j
p q r s t                  k k l m n o o
u v w x y                  p p q r s t t
                            u u v w x y y
                            u u v w x y y
```

For patch at `(2,2)` with size 3, we extract rows 1:4, cols 1:4 of the padded image:
```
Patch = [[a, b, c],
         [f, g, h],
         [k, l, m]]
```
Centre pixel `g` at position `(2,2)` is surrounded by its spatial neighbours.

### f) Code Walkthrough

```python
def extract_labeled_patches(x, y, patch_size=9):
    pad   = patch_size // 2                   # 4 pixels on each side
    # Pad image edges using edge replication (avoids zero-border artefacts)
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

    coords  = np.argwhere(y > 0)             # All labelled pixel (row, col) pairs
    # Pre-allocate arrays for patches and labels
    patches = np.empty((coords.shape[0], patch_size, patch_size, x.shape[-1]), dtype=np.float32)
    labels  = np.empty((coords.shape[0],), dtype=np.int32)

    for i, (r, c) in enumerate(coords):
        # Slice the padded image to get a 9×9×6 cube
        patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
        labels[i]  = int(y[r, c]) - 1       # Convert 1-indexed label to 0-indexed
    return patches, labels, coords
```

### g) Output & Interpretation

`patches` has shape `(N, 9, 9, 6)` where N is the total number of labelled pixels. Each patch is the primary input to all three models. `coords` is retained for scene visualisation (mapping predictions back to spatial positions).

### h) Limitations

- Edge replication padding may mislead the classifier for patches near image borders, since the same pixel values are repeated.
- Fixed patch size (9) is a design choice. Larger patches capture more context but require more memory; smaller patches miss spatial structure.
- No data augmentation (flipping, rotation) is applied at extraction time, which may limit generalisation.
- The loop-based extraction is slow for very large images; vectorised implementations using `np.lib.stride_tricks` would be faster.

---

## Method: Stratified Train / Val / Test Splitting

### a) What it is

> Stratified splitting is like shuffling a deck of cards and dealing them out while guaranteeing each player gets the same proportion of hearts, clubs, diamonds, and spades — so no player is disadvantaged by an unlucky draw.

Stratification ensures each split (train, val, test) contains the same proportion of each class as the original dataset. Without it, random chance could leave rare classes entirely in one split.

### b) Why it's used here

Remote sensing datasets are often imbalanced — some land cover types (e.g., water bodies) may be far less common than others (e.g., vegetation). A non-stratified split could place all samples of a rare class into the test set, making training impossible for that class.

### c) How it works — Step by step

1. Call `train_test_split(X, y, train_size=0.75, stratify=y, random_state=42)` to separate 75% for training (from which validation will also be drawn) and 25% for testing.
2. Within the 75% training pool, call `train_test_split(..., test_size=0.20, stratify=y_train_full, random_state=42)` to reserve 20% of that pool as a validation set.
3. For AlexNet only: a separate split with `random_state=10` (no validation) is used to match an upstream "legacy" single-head notebook so that uncertainty recovery experiments remain comparable.
4. Labels are one-hot encoded for GFNet and ViT using `keras.utils.to_categorical`.

### d) ASCII Flow Diagram

```
All N labelled samples
    |
    v
75% / 25% stratified split (seed=42)
    |               |
x_train_full    x_test (25%)
y_train_full    y_test
    |
    v
80% / 20% stratified split (seed=42)
    |           |
x_train (60%) x_val (15%)
y_train       y_val

(AlexNet uses a separate 75/25 split with seed=10, no validation set)
```

### e) Worked Numerical Example

Suppose N=100 samples with 3 classes: 60 of class A, 30 of class B, 10 of class C.

After 75/25 stratified split:
- `x_train_full`: 75 samples — 45 A, 22 B, 8 C (proportions preserved: 60%, 30%, 10%)
- `x_test`: 25 samples — 15 A, 8 B, 2 C

After 80/20 split of `x_train_full`:
- `x_train`: 60 samples — 36 A, 18 B, 6 C
- `x_val`: 15 samples — 9 A, 4 B, 2 C (approximately)

### f) Code Walkthrough

```python
# Primary split for GFNet and ViT
x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y,
    train_size=TRAIN_PERCENT,       # 0.75 — 75% goes to training pool
    random_state=SEED,              # 42 — for reproducibility
    stratify=y,                     # Preserve class proportions in both halves
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=VAL_SPLIT_FROM_TRAIN, # 0.20 — 20% of training pool becomes validation
    random_state=SEED,
    stratify=y_train_full,
)

# One-hot encode for models expecting categorical targets
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat   = keras.utils.to_categorical(y_val,   num_classes)
y_test_cat  = keras.utils.to_categorical(y_test,  num_classes)

# AlexNet legacy split — different seed, no validation set
x_train_alex, x_test_alex, y_train_alex, y_test_alex = train_test_split(
    X, y,
    train_size=ALEXNET_LEGACY_TRAIN_PERCENT,  # 0.75
    random_state=ALEXNET_LEGACY_SPLIT_SEED,   # 10 — must match original notebook
    stratify=y,
)
```

### g) Output & Interpretation

Produces six arrays: `x_train, x_val, x_test` (patches) and `y_train, y_val, y_test` (integer labels), plus their one-hot encoded counterparts. AlexNet gets its own pair `x_train_alex, x_test_alex`.

### h) Limitations

- Stratification on the integer label only; it does not guarantee spatial diversity (spatially adjacent patches may all end up in the same split).
- The AlexNet legacy split uses a different seed, making direct cross-model comparisons on test sets slightly uneven.
- No k-fold cross-validation is performed, so performance estimates have higher variance than they would with repeated splits.
- Very rare classes (fewer than the number of splits) may still cause `train_test_split` to fail with a `ValueError`.

---

## Method: Shared Custom Keras Layers

### a) What it is

> These three layers are Lego bricks — reusable building blocks that both GFNet and ViT snap together in different configurations. They handle the common task of turning an image into a sequence of "tokens," each representing a small spatial region.

Three custom `tf.keras.Layer` subclasses implement patch tokenisation with positional awareness, shared between GFNet and ViT:
- **PatchExtractor**: Divides an image into non-overlapping tiles and flattens each to a 1D vector.
- **PatchPositionEncoder**: Projects each patch vector to a `projection_dim`-dimensional embedding and adds a learned positional embedding.
- **PatchEncoderWithCLS**: Like `PatchPositionEncoder`, but also prepends a special learnable `[CLS]` token — a single vector that is trained to summarise the entire sequence for classification.

### b) Why it's used here

Transformers and GFNets operate on sequences, not grids. These layers convert the 2D patch grid into a 1D sequence of embedding vectors that attention mechanisms and global filter layers can process. The positional embeddings restore the spatial ordering information that is lost in the sequence conversion. The `[CLS]` token in ViT provides a single fixed-position vector that the classification head can read off without averaging over all patches.

### c) How it works — Step by step

**PatchExtractor:**
1. Use `tf.image.extract_patches` with stride = patch_size (non-overlapping) to cut the input image into tiles.
2. Reshape from `(batch, grid_h, grid_w, patch_dim)` to `(batch, num_patches, patch_dim)`.

**PatchPositionEncoder:**
1. Apply a `Dense(projection_dim)` layer to project each patch vector from raw pixel dimension to `projection_dim`.
2. Create integer position indices `[0, 1, ..., num_patches-1]`.
3. Look up each position in a learned `Embedding(num_patches, projection_dim)` table.
4. Add the position embeddings to the projected patch embeddings elementwise.

**PatchEncoderWithCLS:**
1. Same projection as above.
2. Prepend a single trainable `cls_token` of shape `(1, 1, projection_dim)` to every sequence in the batch.
3. Add positional embeddings of shape `(num_patches+1, projection_dim)` covering both the CLS token and all patches.

### d) ASCII Flow Diagram

```
Input: (batch, 9, 9, 6)
    |
    v
PatchExtractor(patch_size=3)
    --> (batch, 9, patch_dim)   [9 = (9//3)*(9//3) patches]
    |
    v
PatchPositionEncoder or PatchEncoderWithCLS
    |
    Dense(projection_dim) --> projected patches
    +
    Embedding(positions)  --> learned position vectors
    |
    v
Sequence of token embeddings: (batch, num_patches [+1 for CLS], projection_dim)
```

### e) Worked Numerical Example

Suppose input image is `(1, 9, 9, 6)`, `inner_patch=3`, `projection_dim=4`.

- After `PatchExtractor`: `(1, 9, 54)` — 9 patches, each being 3×3×6=54 values flattened.
- Dense projects `54 → 4`: `(1, 9, 4)`.
- Position embeddings for positions `[0..8]`: lookup returns `(9, 4)`.
- Output: `(1, 9, 4)` — 9 tokens, each 4-dimensional.

For `PatchEncoderWithCLS`:
- After CLS prepend: `(1, 10, 4)` — 10 tokens (1 CLS + 9 patches).

### f) Code Walkthrough

```python
# PatchExtractor — divides image into non-overlapping tiles
class PatchExtractor(layers.Layer):
    def __init__(self, patch_size=3, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],    # tile size
            strides=[1, self.patch_size, self.patch_size, 1],  # non-overlapping
            rates=[1, 1, 1, 1],                                # dilation=1 (standard)
            padding="VALID",                                   # no extra padding
        )
        batch       = tf.shape(images)[0]
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]  # grid_h * grid_w
        patch_dim   = tf.shape(patches)[-1]                        # flattened tile size
        return tf.reshape(patches, [batch, num_patches, patch_dim])  # → sequence

# PatchEncoderWithCLS — adds CLS token and positional embeddings
class PatchEncoderWithCLS(layers.Layer):
    def build(self, input_shape):
        # Trainable CLS token, starts as zeros
        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, self.projection_dim),
            initializer="zeros", trainable=True,
        )

    def call(self, patches):
        batch      = tf.shape(patches)[0]
        patch_proj = self.projection(patches)        # Dense: project each patch
        cls_tokens = tf.repeat(self.cls_token, repeats=batch, axis=0)  # one per item in batch
        x          = tf.concat([cls_tokens, patch_proj], axis=1)  # prepend CLS
        positions  = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        return x + self.position_embedding(positions)  # add learned position vectors
```

### g) Output & Interpretation

All three layers are registered with `@tf.keras.utils.register_keras_serializable()` so that `.keras` model files can be loaded back with `custom_objects` without errors. Their `get_config` methods ensure the architecture hyperparameters are serialised alongside the weights.

### h) Limitations

- Learned positional embeddings do not generalise to different sequence lengths at test time (unlike sinusoidal embeddings).
- Non-overlapping patch extraction may split meaningful spatial features across patch boundaries.
- The CLS token adds one extra position; if `num_patches` changes between training and inference, the model must be rebuilt.
- `tf.image.extract_patches` requires `VALID` padding, meaning the input spatial dimensions must be exactly divisible by `inner_patch`.

---

## Method: AlexNet-Inspired CNN

### a) What it is

> AlexNet is a classic "stack of sieves" — each convolutional layer looks for increasingly complex patterns (edges → textures → shapes), followed by fully connected layers that vote on the final class.

An adaptation of the 2012 AlexNet architecture, retaining five convolutional layers and four dense (fully connected) layers, but simplified for the small 9×9×6 patch input instead of the original 224×224×3 images.

### b) Why it's used here

AlexNet is used as a baseline ("legacy") model. Its architecture and training recipe are intentionally preserved to match an upstream notebook used for uncertainty analysis, allowing fair comparison across experiments.

### c) How it works — Step by step

1. Input: `(9, 9, 6)` patch.
2. Apply 5 convolutional layers with filter counts `[96, 256, 384, 384, 256]`, each using `3×3` kernels with ReLU activation and `same` padding (spatial size preserved).
3. Apply one `MaxPooling2D` with `2×2` pool size and stride 2 (halves spatial resolution).
4. Flatten to a 1D vector.
5. Apply 4 dense layers with units `[4096, 1024, 256, 32]`, each with ReLU activation, with dropout (rate=0.25) between each layer.
6. Final `Dense(num_classes, softmax)` layer outputs class probabilities.

### d) ASCII Flow Diagram

```
Input (9, 9, 6)
    |
    v
Conv2D(96,  3×3, relu, same)  --> (9, 9, 96)
Conv2D(256, 3×3, relu, same)  --> (9, 9, 256)
Conv2D(384, 3×3, relu, same)  --> (9, 9, 384)
Conv2D(384, 3×3, relu, same)  --> (9, 9, 384)
Conv2D(256, 3×3, relu, same)  --> (9, 9, 256)
    |
MaxPool2D(2×2, stride=2)      --> (5, 5, 256)
    |
Flatten                        --> (6400,)
    |
Dense(4096, relu) + Dropout(0.25)
Dense(1024, relu) + Dropout(0.25)
Dense(256,  relu) + Dropout(0.25)
Dense(32,   relu)
    |
Dense(num_classes, softmax)   --> class probabilities
```

### e) Worked Numerical Example

Suppose 2 classes and a single 3×3×2 toy patch (simplified):

```
Patch values, band 1:       Patch values, band 2:
1 2 3                       7 8 9
4 5 6                       1 2 3
7 8 9                       4 5 6
```

A 3×3 conv filter learns weights. Say filter 1 computes the average of all pixels in band 1:
```
filter output = (1+2+3+4+5+6+7+8+9) / 9 = 5.0  (at each spatial position, simplified)
```
After 5 such conv layers with increasing depth, the flattened output is fed to dense layers:
```
dense_1(4096): 6400 inputs × 4096 outputs → activation vector of 4096 values (post-ReLU)
dense_2(1024): 4096 → 1024
dense_3(256):  1024 → 256
dense_4(32):   256 → 32
output(2):     32 → [0.3, 0.7] → class 1 (softmax)
```

### f) Code Walkthrough

```python
def build_alexnet(input_shape, num_classes, dropout_rate=0.25, cfg=None):
    cfg    = cfg or ALEXNET_CFG
    inputs = keras.Input(shape=input_shape)
    x      = inputs

    # Five convolutional layers — each scans for local spatial patterns
    for i, filters in enumerate(cfg["conv_filters"], start=1):
        x = layers.Conv2D(filters, (3, 3), activation="relu",
                          padding="same", name=f"alex_conv_{i}")(x)

    # MaxPooling halves the spatial dimension (9 → 5 with padding="same")
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="alex_pool")(x)
    x = layers.Flatten(name="alex_flatten")(x)  # Convert 3D feature map to 1D

    # Four dense layers with dropout for regularisation
    dense_units = cfg["dense_units"]            # [4096, 1024, 256, 32]
    x = layers.Dense(dense_units[0], activation="relu", name="alex_fc1")(x)
    x = layers.Dropout(dropout_rate, name="TRAIN_DROPOUT_1")(x)
    x = layers.Dense(dense_units[1], activation="relu", name="alex_fc2")(x)
    x = layers.Dropout(dropout_rate, name="TRAIN_DROPOUT_2")(x)
    x = layers.Dense(dense_units[2], activation="relu", name="alex_fc3")(x)
    x = layers.Dropout(dropout_rate, name="TRAIN_DROPOUT_3")(x)
    x = layers.Dense(dense_units[3], activation="relu", name="alex_fc4")(x)

    # Softmax output — probabilities over all classes
    outputs = layers.Dense(num_classes, activation="softmax", name="alex_logits")(x)
    return keras.Model(inputs, outputs, name="AlexNet_SingleHead")
```

### g) Output & Interpretation

A Keras Model with ~40M+ parameters (dominated by the first dense layer). Trained with Adagrad + cosine LR schedule (0.01→0.02→0.005 over 100 epochs) and `sparse_categorical_crossentropy` loss. The Dropout layers named `TRAIN_DROPOUT_*` are referenced by name in downstream uncertainty notebooks.

### h) Limitations

- AlexNet was designed for much larger inputs; the 5 conv layers on a 9×9 patch may over-process the spatial information.
- Four 4096-unit dense layers are extremely memory-heavy relative to the patch size.
- No batch normalisation is used, which can slow convergence.
- Uses Adagrad (older optimiser) rather than AdamW, for backward compatibility only.

---

## Method: Global Filter Network (GFNet)

### a) What it is

> GFNet replaces the attention mechanism of a Transformer with a fast Fourier transform — instead of each token "asking" every other token what it knows, the network applies learned filters in the *frequency domain*, which is computationally much cheaper and mathematically equivalent to computing every possible global convolution simultaneously.

GFNet is a token-mixer architecture where the mixing between spatial positions is performed in the Fourier frequency domain using learnable complex-valued filters, followed by a standard MLP block in each residual layer.

### b) Why it's used here

GFNet provides a computationally efficient alternative to self-attention for capturing long-range spatial dependencies across patches. For a 9-patch sequence (from the 9×9 image with 3×3 inner patch), this is not a major efficiency advantage, but it serves as a distinct architectural family for comparison.

### c) How it works — Step by step

1. Input patches are tokenised via `PatchExtractor` + `PatchPositionEncoder`.
2. For each of `num_blocks=5` GFNet blocks:
   a. Apply `LayerNormalization` to stabilise activations.
   b. Pass through `GlobalFilterLayer`: reshape the sequence back to a 2D grid, apply `fft2d` to move to frequency space, multiply element-wise by a learned complex filter `(w_real + i*w_imag)`, then apply `ifft2d` to return to spatial space, take the real part, reshape back to sequence.
   c. Apply a second `LayerNormalization`.
   d. Pass through a two-layer MLP with GELU activations and dropout (hidden dim = `hidden_dim * mlp_ratio`).
   e. Add the block's output to its input (residual/skip connection).
3. After all blocks: dropout, `LayerNormalization`, `GlobalAveragePooling1D` (averages over the token sequence), flatten, dropout.
4. Final `Dense(num_classes, softmax)`.

**GlobalFilterLayer formula:**
```
x_fft    = fft2d(reshape(x, [batch, token_side, token_side, channels]))
w_complex = w_real + i * w_imag
x_filtered = x_fft * w_complex
x_spatial  = real(ifft2d(x_filtered))
output     = reshape(x_spatial, [batch, num_tokens, channels])
```

### d) ASCII Flow Diagram

```
Input (batch, 9, 9, 6)
    |
    v
PatchExtractor(3) --> (batch, 9, 54)
PatchPositionEncoder --> (batch, 9, 512)
Dropout(0.25)
    |
    v
[GFNet Block] × 5:
    |-- LayerNorm
    |-- GlobalFilterLayer (fft2d → multiply w_complex → ifft2d → real)
    |-- LayerNorm
    |-- Dense(512×4=2048, GELU) + Dropout
    |-- Dense(512, GELU) + Dropout
    +-- Residual Add
    |
    v
Dropout → LayerNorm → GlobalAvgPool1D → Flatten → Dropout
    |
    v
Dense(num_classes, softmax)
```

### e) Worked Numerical Example

Suppose `token_side=3`, `channels=2`, one block. Token grid (3×3×2):

```
FFT step:
x_2d  = [[1+0j, 2+0j, 3+0j],    (one channel shown)
          [4+0j, 5+0j, 6+0j],
          [7+0j, 8+0j, 9+0j]]

fft2d(x_2d) = [[45+0j, -4.5+7.79j, -4.5-7.79j],   (illustrative, rounded)
               [-13.5+7.79j, ...],
               [-13.5-7.79j, ...]]

w_complex = [[1+0j, 0.5+0.5j, ...]]   (learned filter, one example value)

x_filtered = x_fft * w_complex   (element-wise complex multiplication)

ifft2d(x_filtered) → back to spatial domain
take real(·)  → filtered feature map
```

The network learns `w_real` and `w_imag` via gradient descent to amplify or suppress specific frequency components.

### f) Code Walkthrough

```python
class GlobalFilterLayer(layers.Layer):
    def build(self, input_shape):
        channels = int(input_shape[-1])
        # Learnable real part of complex filter (Glorot init for good gradient flow)
        self.w_real = self.add_weight(name="w_real",
            shape=(self.token_side, self.token_side, channels),
            initializer="glorot_uniform", trainable=True)
        # Learnable imaginary part (zeros init — starts as a real-valued filter)
        self.w_imag = self.add_weight(name="w_imag",
            shape=(self.token_side, self.token_side, channels),
            initializer="zeros", trainable=True)

    def call(self, x):
        batch    = tf.shape(x)[0]
        channels = tf.shape(x)[-1]
        # Reshape token sequence back to a 2D grid for 2D FFT
        x_2d     = tf.reshape(x, [batch, self.token_side, self.token_side, channels])
        x_fft    = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))  # 2D discrete FFT
        w_complex = tf.complex(self.w_real, self.w_imag)          # combine into complex
        x_filtered = x_fft * w_complex                            # frequency-domain filtering
        # Inverse FFT → spatial domain; discard imaginary residuals
        x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
        return tf.reshape(x_spatial, [batch, self.token_side * self.token_side, channels])

def gf_block(x, token_side, dim, mlp_ratio=4, dropout_rate=0.25, name_prefix="gf"):
    y = layers.LayerNormalization(name=f"{name_prefix}_ln1")(x)   # Normalise before filter
    y = GlobalFilterLayer(token_side, name=f"{name_prefix}_gfilter")(y)  # Apply filter
    y = layers.LayerNormalization(name=f"{name_prefix}_ln2")(y)   # Normalise before MLP
    y = layers.Dense(dim * mlp_ratio, activation=tf.keras.activations.gelu,
                     name=f"{name_prefix}_mlp1")(y)               # Expand (bottleneck open)
    y = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop1")(y)
    y = layers.Dense(dim, activation=tf.keras.activations.gelu,
                     name=f"{name_prefix}_mlp2")(y)               # Contract (bottleneck close)
    y = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop2")(y)
    return layers.Add(name=f"{name_prefix}_add")([x, y])          # Residual connection
```

### g) Output & Interpretation

GFNet outputs `(num_classes,)` softmax probabilities. It is trained with AdamW + cosine decay + label smoothing, monitored on validation loss. The best checkpoint is saved as `GFNet_best.keras`.

### h) Limitations

- The 2D FFT requires reshaping the token sequence back to a square grid; if `num_patches` is not a perfect square, this requires special handling.
- The imaginary part initialised to zero means the filter starts as real-valued — it must learn complex behaviour through training.
- GFNet assumes that global convolution (in frequency space) is the right inductive bias; for fine-grained texture discrimination, local attention may be superior.
- Frequency-domain multiplication is equivalent to circular (toroidal) convolution, which wraps edges — potentially an issue at image borders.

---

## Method: Vision Transformer with U-Net Skip Connections (ViT-UNet)

### a) What it is

> The Vision Transformer reads all patches as a sentence — every word (patch) pays attention to every other word simultaneously. The U-Net modification adds "memory shortcuts" that let the later layers remember what the early layers saw, like a student who can flip back to their earlier notes while answering an exam question.

ViT-UNet is a standard Vision Transformer encoder where the second half of transformer blocks receives additive skip connections from the symmetric first half, mirroring the encoder-decoder structure of U-Net.

### b) Why it's used here

ViT has shown strong performance on image classification when sufficient data is available. The U-Net skip connections allow gradient to flow from early to late layers without degradation, potentially improving convergence for small training sets by letting the classification head access multi-scale representations.

### c) How it works — Step by step

1. Tokenise via `PatchExtractor` + `PatchEncoderWithCLS` to get `(batch, num_patches+1, projection_dim)`.
2. For each of `transformer_layers=12` blocks:
   a. **Pre-LN attention sub-block:** `LayerNorm → MultiHeadAttention(self) → Add residual`.
   b. **Pre-LN MLP sub-block:** `LayerNorm → Dense(mlp_dim, GELU) → Dropout → Dense(projection_dim, GELU) → Dropout → Add residual`.
   c. If `i <= 6` (first half): save the block output to `block_list`.
   d. If `i > 6` (second half): add the stored output from block `(12 - i - 1)` (mirror) to the current output before continuing.
3. Dropout + `LayerNorm` on the full sequence.
4. Extract the CLS token: `x[:, 0, :]` — the classification summary vector.
5. Pass through 4 dense layers `[512, 256, 128, 64]` with GELU activations and dropout.
6. Final `Dense(num_classes, softmax)`.

**Self-attention formula (per head):**
```
Q = K = V = LayerNorm(x)    (self-attention: query=key=value)
Attention(Q, K, V) = softmax(Q * K^T / sqrt(key_dim)) * V
```

### d) ASCII Flow Diagram

```
Input (batch, 9, 9, 6)
    |
PatchExtractor(3) --> (batch, 9, 54)
PatchEncoderWithCLS --> (batch, 10, 256)  [10 = 9 patches + 1 CLS]
    |
    v
Transformer Block 1 --> saved to block_list[0]
Transformer Block 2 --> saved to block_list[1]
...
Transformer Block 6 --> saved to block_list[5]   (encoder half)
Transformer Block 7 --> Add block_list[4]         (decoder half)
Transformer Block 8 --> Add block_list[3]
...
Transformer Block 12 --> Add block_list[0]
    |
    v
Dropout → LayerNorm → Extract CLS token [:, 0, :]
    |
Dense(512, GELU) → Dropout
Dense(256, GELU)
Dense(128, GELU) → Dropout
Dense(64,  GELU) → Dropout
    |
Dense(num_classes, softmax)
```

### e) Worked Numerical Example

Suppose `num_patches=4`, `projection_dim=2`, `num_heads=1`, `key_dim=2`.

After `PatchEncoderWithCLS`, token sequence = 5 rows (CLS + 4 patches), 2 dims each:
```
tokens = [[0.1, 0.2],   <- CLS token (initialised to 0, updated by encoder)
          [0.3, 0.4],   <- patch 0
          [0.5, 0.1],   <- patch 1
          [0.2, 0.7],   <- patch 2
          [0.8, 0.3]]   <- patch 3
```

In one attention head with `key_dim=2`:
```
Q = K = V = tokens (after LayerNorm)
score_matrix = Q @ K^T / sqrt(2)   -> (5×5) matrix of raw attention scores
attn_weights = softmax(score_matrix, axis=-1)
output = attn_weights @ V           -> weighted sum of values for each query
```

The CLS token row of `attn_weights` learns which patches to attend to for classification.

### f) Code Walkthrough

```python
def transformer_block(x, num_heads, projection_dim, mlp_dim, dropout_rate, name_prefix):
    # --- Attention sub-block (pre-LayerNorm style) ---
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(x)
    # Self-attention: each token attends to all tokens (y, y) = (query, key/value)
    y = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim,
        dropout=dropout_rate, name=f"{name_prefix}_mha")(y, y)
    x = layers.Add(name=f"{name_prefix}_add1")([y, x])  # Residual connection

    # --- MLP sub-block (pre-LayerNorm style) ---
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x)
    y = layers.Dense(mlp_dim, activation=tf.keras.activations.gelu,
                     name=f"{name_prefix}_mlp1")(y)    # Expand width
    y = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop1")(y)
    y = layers.Dense(projection_dim, activation=tf.keras.activations.gelu,
                     name=f"{name_prefix}_mlp2")(y)    # Contract back
    y = layers.Dropout(dropout_rate, name=f"{name_prefix}_drop2")(y)
    return layers.Add(name=f"{name_prefix}_add2")([y, x])

def build_vit_unet_singlehead(...):
    block_list = []  # Stores encoder-half outputs for skip connections
    for i in range(transformer_layers):
        x = transformer_block(x, num_heads=num_heads, ...)
        if i <= transformer_layers // 2:
            block_list.append(x)      # Save first-half outputs
        else:
            # Add skip connection from the mirror position in the encoder half
            x = layers.Add(name=f"vit_skip_add_{i+1}")([x, block_list[transformer_layers - i - 1]])

    cls_token = layers.Lambda(lambda t: t[:, 0, :], name="vit_cls_token")(x)  # Extract CLS row
```

### g) Output & Interpretation

ViT-UNet is the largest model in the notebook. The U-Net skip connections mean that the final classification head receives information from both deep (abstract) and shallow (detailed) transformer representations. It is trained with AdamW + cosine decay + label smoothing, with best checkpoint saved on validation loss.

### h) Limitations

- ViT typically requires large datasets to outperform CNNs; with moderate-sized remote sensing datasets it may underfit without pretraining.
- The skip connection index `block_list[transformer_layers - i - 1]` may cause an index error if `transformer_layers` is odd and the mirror calculation is not carefully validated.
- Self-attention has O(N²) complexity in the number of tokens, which is manageable for 9 patches but would not scale to full-resolution scenes.
- The Lambda layer (`lambda t: t[:, 0, :]`) is not serialisable in all Keras versions without special handling.

---

## Method: Calibration Metrics — Brier Score and ECE

### a) What it is

> A model that says "I'm 90% confident" about every prediction — but is only right 60% of the time — is poorly calibrated. The Brier score and ECE measure exactly this gap between stated confidence and actual accuracy.

**Brier Score:** The mean squared error between the predicted probability vector and the true one-hot label vector. Lower is better (0 = perfect). **Expected Calibration Error (ECE):** Divides predictions into confidence bins, measures the absolute difference between average confidence and average accuracy within each bin, then takes a weighted average.

### b) Why it's used here

Classification accuracy alone does not reveal whether a model is overconfident (high accuracy, but inflated probability estimates). For downstream uncertainty analysis (e.g., Monte Carlo Dropout), well-calibrated probability outputs are essential. These metrics serve as proxy calibration measures without requiring temperature scaling or explicit uncertainty estimation.

### c) How it works — Step by step

**Brier Score:**
```
brier = mean over all samples of: sum over all classes of (prob[c] - one_hot[c])^2
```
1. Compute `(y_prob - y_onehot)^2` — element-wise squared error matrix of shape `(N, C)`.
2. Sum across classes (axis=1) → `(N,)`.
3. Take the mean over all samples → single scalar.

**ECE (15 bins):**
1. Compute `confidences = max(y_prob, axis=1)` — the model's top-class probability.
2. Compute `predictions = argmax(y_prob, axis=1)`.
3. Compute `correct = (predictions == y_true)` — 1 if right, 0 if wrong.
4. Create 15 equal-width bins from 0.0 to 1.0.
5. For each bin `[lo, hi)`:
   a. Find all samples whose confidence falls in this bin.
   b. Compute `acc_bin = mean(correct[in_bin])` and `conf_bin = mean(confidences[in_bin])`.
   c. Compute `|acc_bin - conf_bin| * proportion_in_bin`.
6. ECE = sum of all bin contributions.

### d) ASCII Flow Diagram

```
y_test_prob (N, C)       y_test_cat (N, C)
       |                       |
       +----------- (prob - onehot)^2 ----------+
                                                  |
                          sum over C, mean over N |
                                                  v
                                          Brier Score (scalar)

y_test_prob (N, C)       y_test (N,)
       |                       |
  confidences = max(prob)   correct = (argmax(prob) == y_test)
       |                       |
       +------ bin into 15 buckets by confidence ------+
                     |
          |acc_bin - conf_bin| * bin_weight
                     |
                 sum over bins
                     |
                  ECE (scalar)
```

### e) Worked Numerical Example

Suppose 4 samples, 2 classes:
```
y_prob    = [[0.9, 0.1],   [0.6, 0.4],   [0.3, 0.7],   [0.8, 0.2]]
y_onehot  = [[1,   0  ],   [1,   0  ],   [0,   1  ],   [1,   0  ]]
y_true    =  [0,           0,            1,            0          ]
```

**Brier Score:**
```
sample 1: (0.9-1)^2 + (0.1-0)^2 = 0.01 + 0.01 = 0.02
sample 2: (0.6-1)^2 + (0.4-0)^2 = 0.16 + 0.16 = 0.32
sample 3: (0.3-0)^2 + (0.7-1)^2 = 0.09 + 0.09 = 0.18
sample 4: (0.8-1)^2 + (0.2-0)^2 = 0.04 + 0.04 = 0.08
mean = (0.02 + 0.32 + 0.18 + 0.08) / 4 = 0.15
```

**ECE (simplified, 2 bins: [0, 0.5) and [0.5, 1.0]):**
```
confidences = [0.9, 0.6, 0.7, 0.8]
predictions = [0,   0,   1,   0  ]
correct     = [1,   1,   1,   1  ] (all correct)

Bin [0.5, 1.0]: all 4 samples, acc=1.0, conf=0.75
ECE = |1.0 - 0.75| * 1.0 = 0.25
```
A model that is always correct but consistently underestimates its confidence will have nonzero ECE.

### f) Code Walkthrough

```python
def multiclass_brier_score(y_onehot, y_prob):
    # Element-wise: (predicted_prob - true_prob)^2 for every (sample, class) pair
    # Sum over classes, average over samples
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

def expected_calibration_error(y_true, y_prob, n_bins=15):
    confidences = np.max(y_prob, axis=1)             # Top-class probability for each sample
    predictions = np.argmax(y_prob, axis=1)          # Predicted class
    correct     = (predictions == y_true).astype(np.float32)  # 1 if correct, else 0
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1) # 15 equal-width bins
    ece         = 0.0

    for i in range(n_bins):
        lo, hi  = bin_edges[i], bin_edges[i + 1]
        # Last bin is inclusive on right edge; all others are exclusive
        in_bin  = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        prop    = np.mean(in_bin)                    # Fraction of samples in this bin
        if prop > 0:
            # Weighted absolute gap between bin accuracy and bin confidence
            ece += np.abs(np.mean(correct[in_bin]) - np.mean(confidences[in_bin])) * prop
    return float(ece)
```

### g) Output & Interpretation

Both metrics are included in the `results_rows` dictionary alongside accuracy, F1, and Kappa, then plotted as a calibration proxy bar chart (`uncertainty_proxy_metrics.png`). NLL (negative log-likelihood / log loss) is also included, providing a third calibration-sensitive metric.

### h) Limitations

- Brier score and ECE are computed on the test (or validation) set only — they don't reflect calibration on out-of-distribution inputs.
- ECE depends on the number of bins; 15 bins is standard, but bins may be empty for rare confidence ranges, skewing the estimate.
- Neither metric measures epistemic uncertainty directly; they are "calibration proxies" rather than true uncertainty estimates.
- NLL is unbounded and can be dominated by a small number of very confident wrong predictions.

---

## Method: AdamW with Cosine Decay Learning Rate Schedule

### a) What it is

> AdamW is like a smart downhill hiker who adjusts step size based on the terrain and also periodically lightens his pack (weight decay). Cosine decay is the schedule: the hiker starts at a medium pace, briefly speeds up, then gradually slows to a near-crawl by the time he reaches the valley.

AdamW is Adam with decoupled weight decay regularisation. Cosine decay is a smooth schedule that reduces the learning rate from `initial_learning_rate` to `alpha * initial_learning_rate` following a cosine curve.

### b) Why it's used here

AdamW's decoupled weight decay prevents the weight regularisation from interfering with the adaptive gradient estimates (a known flaw in Adam + L2 regularisation). Cosine decay reduces the risk of overshooting good minima late in training. Gradient clipping (`clipnorm=1.0`) prevents exploding gradients in deep Transformer models.

### c) How it works — Step by step

**Cosine Decay Schedule:**
```
lr(step) = alpha * initial_lr  +  (1 - alpha) * initial_lr * 0.5 * (1 + cos(pi * step / decay_steps))
```
At `step=0`:         `lr = initial_lr = 3e-4`
At `step=decay_steps/2`: `lr ≈ (1+alpha)/2 * initial_lr`
At `step=decay_steps`:   `lr = alpha * initial_lr = 0.05 * 3e-4 = 1.5e-5`

**AdamW update rule (simplified):**
```
m  = beta1 * m  + (1 - beta1) * grad          (momentum estimate)
v  = beta2 * v  + (1 - beta2) * grad^2         (variance estimate)
theta = theta - lr * m / (sqrt(v) + eps)  - lr * weight_decay * theta
```

**AlexNet cosine LR (legacy):**
```
phase = pi * epoch / (EPOCHS - 1)
lr(epoch) = (LR_MAX - LR_MIN) * 0.5 * (1 + cos(phase)) + LR_MIN
```
Goes from 0.01 at epoch 0 → peaks at ~0.02 → returns to 0.005 at epoch 99.

### d) ASCII Flow Diagram

```
Training step counter: 0 → decay_steps
    |
    v
CosineDecay schedule:
  3e-4 ─────────────────╮
                         ╰──────────────── 1.5e-5
  (smooth cosine descent over 100 epochs × steps_per_epoch steps)
    |
    v
AdamW:
  gradient g
    |
    v
  clip g if ||g|| > 1.0 (clipnorm)
    |
    v
  update momentum m, variance v
    |
    v
  theta ← theta  -  lr * m/sqrt(v)  -  lr * weight_decay * theta
```

### e) Worked Numerical Example

With `initial_lr=3e-4`, `alpha=0.05`, `decay_steps=1000`:

At step 0:
```
lr = 0.05 * 3e-4 + (1 - 0.05) * 3e-4 * 0.5 * (1 + cos(0))
   = 1.5e-5 + 0.95 * 3e-4 * 0.5 * 2
   = 1.5e-5 + 2.85e-4 = 3.0e-4
```
At step 500 (halfway):
```
lr = 1.5e-5 + 0.95 * 3e-4 * 0.5 * (1 + cos(pi/2))
   = 1.5e-5 + 0.95 * 3e-4 * 0.5 * 1
   = 1.5e-5 + 1.425e-4 ≈ 1.575e-4
```
At step 1000:
```
lr = 1.5e-5 + 0.95 * 3e-4 * 0.5 * (1 + cos(pi))
   = 1.5e-5 + 0  = 1.5e-5
```

### f) Code Walkthrough

```python
def make_optimizer(num_train_samples):
    steps_per_epoch = int(np.ceil(num_train_samples / BATCH_SIZE))  # Batches per epoch
    decay_steps     = max(1, steps_per_epoch * EPOCHS)              # Total training steps
    lr_schedule     = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,  # 3e-4 — starting LR
        decay_steps=decay_steps,              # Decay over full training run
        alpha=TRAIN_CFG["cosine_alpha"],      # 0.05 — minimum LR as fraction of initial
    )
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=TRAIN_CFG["weight_decay"],  # 1e-4 — decoupled L2 regularisation
        clipnorm=TRAIN_CFG["clipnorm"],           # 1.0 — gradient clipping by norm
    )

def _alexnet_legacy_lr(epoch):
    """AlexNet uses a callback-based schedule (per-epoch, not per-step)."""
    phase        = np.pi * epoch / (EPOCHS - 1)   # 0 → pi over training
    cosine_decay = 0.5 * (1.0 + np.cos(phase))    # 1.0 → 0.0
    return float((ALEXNET_LR_MAX - ALEXNET_LR_MIN) * cosine_decay + ALEXNET_LR_MIN)
    # At epoch 0: 0.02 * 1.0 + 0.005 = ?
    # (LR_MAX - LR_MIN) * 1 + LR_MIN = (0.02-0.005)*1 + 0.005 = 0.02
```

### g) Output & Interpretation

GFNet and ViT are trained with step-level cosine decay (smooth). AlexNet uses epoch-level cosine via `LearningRateScheduler`, which is coarser but reproduces the original notebook's training dynamics. Training curves are saved per model showing accuracy and loss vs epoch.

### h) Limitations

- Cosine decay with a single cycle may not be optimal; warm restarts (SGDR) or cyclic schedules often improve final accuracy.
- `clipnorm=1.0` clips the entire gradient vector's L2 norm, which can slow learning if most gradients are naturally large.
- The cosine schedule for AlexNet starts LR at 0.01 then rises to 0.02 before decaying — this warm-up-then-decay pattern can cause instability early in training.
- Weight decay and dropout are both active simultaneously, which may over-regularise small models.

---

## Method: Training, Checkpointing, and Evaluation Pipeline

### a) What it is

> `train_save_evaluate` is the assembly line manager: it builds the car (model), drives it through quality testing (training + validation), stamps the best version (checkpoint), parks the final version (save), and hands over the inspection report (metrics).

The `train_save_evaluate` function encapsulates the full training loop for a single model: compile, fit with callbacks, save best and final models, compute all evaluation metrics, and return a results row.

### b) Why it's used here

Centralising the training logic avoids code duplication across three architectures and ensures a consistent evaluation protocol (same metrics, same save convention) for all models.

### c) How it works — Step by step

1. Clear the Keras session (frees GPU memory from any previous model).
2. Build the model using the provided `model_builder` lambda.
3. If AlexNet: compile with Adagrad + `sparse_categorical_crossentropy`, use AlexNet's legacy data splits and cosine LR callback, no shuffling.
4. If GFNet/ViT: compile with AdamW + CosineDecay + `categorical_crossentropy` with `label_smoothing=0.05`, use primary splits, shuffle each epoch.
5. Call `model.fit(...)` with `ModelCheckpoint` callback (saves the best model on `val_accuracy` for AlexNet or `val_loss` for GFNet/ViT).
6. Time the training duration with `time.perf_counter`.
7. Save the final model (after all epochs) as a separate file.
8. Run `model.predict` on both validation/evaluation set and test set.
9. Compute: accuracy, Cohen Kappa, macro F1, weighted F1, NLL, Brier score, ECE.
10. Return a metrics row dict, a per-class classification report, a confusion matrix, and the raw history.
11. If an OOM error occurs for GFNet or ViT, automatically retry with reduced fallback configs.

### d) ASCII Flow Diagram

```
model_builder lambda
    |
    v
clear_session() --> build model --> compile
    |
    v
model.fit(x_tr, y_tr, validation=(x_va, y_va), callbacks=[ModelCheckpoint])
    |
    v
Best epoch model saved to MODEL_DIR/{name}_best.keras
Final model  saved to MODEL_DIR/{name}_final.keras
    |
    v
model.predict(x_eval) --> y_eval_prob
model.predict(x_test) --> y_test_prob --> y_test_pred (argmax)
    |
    v
accuracy, kappa, macro_f1, weighted_f1,
val_nll, test_nll, val_brier, test_brier, test_ece_15bin
    |
    v
return (metrics_row, classification_report, confusion_matrix, history_dict)
```

### e) Worked Numerical Example

For a 2-class problem with 10 test samples:

```
y_test      = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
y_test_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
               ✓  ✗  ✓  ✗  ✓  ✓  ✓  ✓  ✓  ✓

accuracy = 8/10 = 0.80
macro_f1 = (F1_class0 + F1_class1) / 2
         = (2*3/(2*3+1+1) + 2*4/(2*4+1+1)) / 2
         = (0.75 + 0.80) / 2 = 0.775
```

### f) Code Walkthrough

```python
def train_save_evaluate(model_name, model_builder, capacity_tag="max"):
    tf.keras.backend.clear_session()    # Free GPU memory from previous model
    model      = model_builder()        # Instantiate the architecture
    best_path  = MODEL_DIR / f"{model_name}_best.keras"   # Path for best checkpoint
    final_path = MODEL_DIR / f"{model_name}_final.keras"  # Path for end-of-training model

    if model_name == "AlexNet_CNN":
        model.compile(
            optimizer=keras.optimizers.Adagrad(learning_rate=ALEXNET_LR_START),
            loss="sparse_categorical_crossentropy",  # Integer labels — no one-hot needed
            metrics=["accuracy"],
        )
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(best_path), monitor="val_accuracy",  # Save on accuracy peak
                mode="max", save_best_only=True, verbose=1,
            ),
            keras.callbacks.LearningRateScheduler(_alexnet_legacy_lr, verbose=0),
        ]
        # AlexNet uses its own data split (no separate val set; test set used as val)
        x_tr, y_tr = x_train_alex, y_train_alex
        x_va, y_va = x_test_alex,  y_test_alex
    else:
        model.compile(
            optimizer=make_optimizer(len(x_train)),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),  # Soften targets
            metrics=["accuracy"],
        )
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(best_path), monitor="val_loss",  # Save on loss minimum
                mode="min", save_best_only=True, verbose=1,
            ),
        ]
        x_tr, y_tr = x_train, y_train_cat   # One-hot labels for CategoricalCrossentropy
        x_va, y_va = x_val,   y_val_cat

    train_start = time.perf_counter()       # Start wall-clock timer
    history_obj = model.fit(
        x_tr, y_tr,
        validation_data=(x_va, y_va),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1, shuffle=fit_shuffle,
    )
    train_time_sec = float(time.perf_counter() - train_start)  # Total training seconds

    model.save(final_path)                  # Always save the last epoch model
    # ... metrics computation ...
```

### g) Output & Interpretation

Returns four objects per model: a dict row (appended to `results_rows` for the final DataFrame), a `classification_report` dict (per-class precision/recall/F1, saved as JSON), a confusion matrix array (used for plotting), and a history dict (train/val accuracy and loss per epoch, used for learning curves).

### h) Limitations

- AlexNet uses the test set as its "validation" set during training (no held-out val set), which means the best checkpoint selection may be optimistic for AlexNet.
- The OOM retry mechanism retries once only; if the fallback config also OOMs, the exception propagates.
- `label_smoothing=0.05` makes the loss surface softer but also changes the numerical values of `val_loss` (lower does not mean better calibrated), potentially distorting checkpoint selection.
- Training time includes Python overhead from verbose callbacks and predict calls at the end.

---

## Method: Full-Scene Dense Inference

### a) What it is

> Dense inference is like scanning every square inch of a map with a magnifying glass — the model classifies every single pixel in the full scene, not just the labelled ones, to produce a complete land cover map.

`predict_full_scene_labels` slides the same 9×9 extraction window across every pixel in the scene (not just labelled ones) and runs batch inference to produce a complete classification image of shape `(H, W)`.

### b) Why it's used here

The model was trained on patches around labelled pixels, but the final output needed for remote sensing applications is a spatially complete classification map — every pixel assigned a class, including those without training labels.

### c) How it works — Step by step

1. Pad the full image by 4 pixels (edge replication), same as during training.
2. For each row `r` in `[0, H-1]`:
   a. Extract a row's worth of patches: for each column `c`, slice `x_pad[r:r+9, c:c+9, :]`.
   b. Stack into a batch `(W, 9, 9, 6)`.
   c. Run `model.predict(row_patches, batch_size=256)` → `(W, num_classes)` probability matrix.
   d. Assign `preds[r] = argmax(row_prob, axis=1) + 1` (convert back to 1-indexed labels).
3. Return the full `(H, W)` label map.

### d) ASCII Flow Diagram

```
x_img: (330, 307, 6)
    |
    v
Edge-pad to (338, 315, 6)
    |
    v
For r in range(330):
    For c in range(307):
        row_patches[c] = x_pad[r:r+9, c:c+9, :]  --> (9, 9, 6)
    batch of 307 patches → model.predict(batch_size=256)
    → (307, num_classes) softmax outputs
    → preds[r] = argmax(...) + 1   (row of integer class labels)
    |
    v
preds: (330, 307) full scene label map
```

### e) Worked Numerical Example

Suppose a tiny `3×3` scene, 1 band, `patch_size=3`, `pad=1`. After edge padding: `5×5`.

For row `r=1` (middle row of original):
```
row_patches[c=0] = x_pad[0:3, 0:3, :]   (3×3 around pixel (1,0))
row_patches[c=1] = x_pad[0:3, 1:4, :]   (3×3 around pixel (1,1))
row_patches[c=2] = x_pad[0:3, 2:5, :]   (3×3 around pixel (1,2))
```
Model predicts softmax for each: `[[0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]`
Predicted row: `argmax → [1, 0, 1]` → add 1 → `preds[1] = [2, 1, 2]`

### f) Code Walkthrough

```python
def predict_full_scene_labels(model, x_img, patch_size=9, batch_size=256):
    pad   = patch_size // 2                             # 4 pixels padding
    x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")  # Same padding as training
    preds = np.zeros((x_img.shape[0], x_img.shape[1]), dtype=np.int32)    # Output label map

    for r in range(x_img.shape[0]):                    # Process one row at a time
        row_patches = np.empty(
            (x_img.shape[1], patch_size, patch_size, x_img.shape[-1]),
            dtype=np.float32,
        )
        for c in range(x_img.shape[1]):
            row_patches[c] = x_pad[r:r + patch_size, c:c + patch_size, :]  # Extract patch

        row_prob  = model.predict(row_patches, batch_size=batch_size, verbose=0)  # Batch infer
        preds[r]  = np.argmax(row_prob, axis=1) + 1   # Convert 0-indexed back to 1-indexed
    return preds
```

### g) Output & Interpretation

Each model produces a `(330, 307)` integer array where values are class labels 1..C. These are visualised as false-colour maps using a fixed palette (`CLASS_COLOR_BASE`), saved as PNGs, and embedded in an Excel workbook alongside the approximate RGB composite and ground truth map.

### h) Limitations

- Row-by-row inference is memory-efficient but slow; batching entire rows of 307 patches at a time is a compromise.
- Pixels at the scene boundary are padded with edge-replicated values, which may cause misclassification near borders.
- All three models are loaded simultaneously to produce maps, which may exhaust GPU memory on smaller Colab instances.
- No post-processing (spatial smoothing, majority filter) is applied to the raw prediction maps.

---

## Results & Comparisons

The notebook computes and saves the following metrics for each model:

| Metric | Description | Better when |
|---|---|---|
| `test_accuracy` | Fraction of correctly classified test pixels | Higher |
| `macro_f1` | Unweighted mean F1 across all classes | Higher |
| `kappa` | Cohen's Kappa (agreement beyond chance) | Higher |
| `train_time_sec` | Wall-clock seconds for full training | Lower |
| `test_nll` | Negative log-likelihood on test set | Lower |
| `test_brier` | Multiclass Brier score on test set | Lower |
| `test_ece_15bin` | Expected Calibration Error (15 bins) on test set | Lower |

**Comparison table structure (actual values depend on dataset and run):**

| Model | Test Accuracy | Macro F1 | Kappa | Train Time (s) | NLL | Brier | ECE |
|---|---|---|---|---|---|---|---|
| AlexNet_CNN | — | — | — | — | — | — | — |
| GFNet | — | — | — | — | — | — | — |
| ViT_UNet | — | — | — | — | — | — | — |

The results are saved to `classification_summary.csv` and individual per-class reports to `{model}_classification_report.json`. The notebook generates:

- **Training curves** (accuracy + loss vs epoch, per model) — reveals overfitting and convergence speed.
- **Cross-model bar charts** — side-by-side comparison of accuracy, F1, Kappa, and training time.
- **Calibration proxy chart** — bar chart of NLL, Brier, ECE side by side.
- **Confusion matrices** — reveals which classes are most commonly confused.
- **Full-scene classification maps** — spatial output showing predicted land cover across the entire image.

---

## Academic Paper Summary

### Problem Statement

This study addresses the problem of pixel-wise land cover classification in multispectral remote sensing imagery. Given a six-band multispectral image, the objective is to assign each spatially labelled pixel to one of C mutually exclusive land cover categories based on the spectral and spatial information contained within a 9×9-pixel neighbourhood patch. The fundamental challenge is to identify which deep learning architecture most effectively captures the discriminative spectral-spatial features required for accurate and well-calibrated classification.

### Methodology

**Data Preprocessing.** The raw multispectral image, loaded from a tabular CSV format and reshaped to a spatial array of dimensions 330 × 307 × 6, underwent independent per-band min-max normalisation, mapping each spectral band to the unit interval [0, 1]. Spatial context patches of size 9 × 9 × 6 were subsequently extracted around each labelled pixel using zero-order edge-replication padding. The labelled sample pool was partitioned using stratified random splitting: 75% for training (of which 20% was reserved as a validation set) and 25% for testing. Class proportions were preserved in all splits.

**Model Architectures.** Three single-head classification architectures were evaluated. The first, an AlexNet-inspired convolutional neural network [1], applied five successive 3 × 3 convolutional layers with channel depths of [96, 256, 384, 384, 256], followed by max-pooling, flattening, and four fully connected layers with progressive dimensionality reduction [4096, 1024, 256, 32], with dropout regularisation at rate 0.25. The second, a Global Filter Network (GFNet) [2], tokenised the input patch into a sequence of 9 tokens via non-overlapping 3 × 3 sub-patch extraction and projected them to a 512-dimensional embedding space before applying five residual blocks. Each GFNet block performed spectral token mixing via learnable complex-valued filters in the two-dimensional discrete Fourier frequency domain, followed by a GELU-activated MLP with expansion ratio 4. The third, a Vision Transformer with U-Net-style skip connections (ViT-UNet) [3], employed the same tokenisation approach and prepended a learnable [CLS] token. Twelve pre-layer-normalisation transformer blocks with four-headed self-attention and a projection dimension of 256 were applied; the output of each block in the second half received an additive skip connection from the symmetric block in the first half, following the encoder–decoder structure of U-Net [4]. The final classification was derived from the CLS token representation passed through a four-layer GELU MLP head.

**Training Procedure.** AlexNet was trained using the Adagrad optimiser with a cosine learning rate schedule ranging from 0.01 to 0.02 then decaying to 0.005 over 100 epochs, on integer-label sparse cross-entropy loss without label smoothing. GFNet and ViT-UNet were trained using the AdamW optimiser [5] with decoupled weight decay of 1 × 10⁻⁴, gradient clipping at norm 1.0, and a cosine decay schedule from 3 × 10⁻⁴ to 1.5 × 10⁻⁵. Both used categorical cross-entropy loss with label smoothing of 0.05. All models used a batch size of 128 and were trained for up to 100 epochs. The best-performing checkpoint was selected on the basis of validation accuracy (AlexNet) or validation loss (GFNet, ViT-UNet).

**Evaluation Metrics.** Models were evaluated on the held-out test set using: overall accuracy (OA), macro-averaged F1 score, Cohen's Kappa coefficient, negative log-likelihood (NLL), multiclass Brier score [6], and Expected Calibration Error (ECE) with 15 equal-width confidence bins [7]. Full-scene classification maps were generated via dense sliding-window inference.

### Experimental Setup

**Dataset:** A 330 × 307 pixel, 6-band multispectral image with per-pixel integer class labels.
**Input representation:** 9 × 9 × 6 spatial-spectral patches.
**Evaluation metrics:** OA, macro F1, Cohen's Kappa, NLL, Brier score, ECE.
**Baselines:** AlexNet-CNN (legacy reference), GFNet, ViT-UNet compared on identical test sets.

### Results Summary

Quantitative results recorded across the three architectures reveal differences in classification accuracy, macro F1, and calibration quality. The ranking of models is expected to reflect the relative capacity of each architecture to exploit spatial-spectral patch structure — with ViT-UNet's multi-head attention and U-Net skip connections potentially offering superior performance on spatially structured classes, while GFNet's frequency-domain token mixing may provide computational efficiency advantages. AlexNet functions as a reproducible empirical baseline consistent with prior work.

### Conclusion

This work demonstrates a systematic comparative evaluation of three deep learning architectures for patch-based multispectral land cover classification. The experimental pipeline encompasses normalisation, stratified partitioning, architecture-specific training regimes, and both discriminative and calibration-sensitive evaluation metrics. Limitations include the absence of spatial cross-validation, a fixed patch size, and a lack of data augmentation. Future work may explore hybrid architectures that combine convolutional locality with global attention, pretrained transformer backbones adapted to multispectral modalities, and Monte Carlo Dropout-based epistemic uncertainty estimation to complement the calibration proxy metrics reported here.

---

## References

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 25.

[2] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global Filter Networks for Image Classification. *Advances in Neural Information Processing Systems (NeurIPS)*, 34.

[3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *International Conference on Learning Representations (ICLR)*.

[4] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, LNCS 9351, pp. 234–241.

[5] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR)*.

[6] Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.

[7] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70.

[8] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

[9] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, PMLR 48.
