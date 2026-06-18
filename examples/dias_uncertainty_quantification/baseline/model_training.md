# Multispectral Patch-Based Classification Pipeline: Theory & Implementation Summary

> **One-line description:** A comparative deep-learning framework that trains three architectures — AlexNet-CNN, Global Filter Network (GFNet), and a ViT with U-Net skip connections — on spatially extracted patches from a 6-band multispectral image to produce pixel-wise land-cover classification maps.

---

## 1. Overview & Intuition

Remote sensing images contain rich spatial and spectral information at each pixel. Rather than classifying pixels in isolation using only their spectral vector, patch-based classification embeds each pixel in its local spatial context by centring a fixed-size neighbourhood window around it and feeding that window — a *patch* — to a classifier. This allows the model to exploit texture, shape, and structural cues that are invisible in single-pixel approaches.

The pipeline in this notebook operates on a 6-band multispectral scene of 330 × 307 pixels. Every labelled pixel becomes one training sample: a 9 × 9 × 6 tensor that captures the pixel's neighbourhood across all bands. Across a scene with thousands of labelled pixels, this forms a rich supervised dataset amenable to deep learning.

Three architectures sit at the centre of the comparison. **AlexNet** represents the classical CNN paradigm: convolutional filters learn local spatial features hierarchically, followed by dense layers for classification. **GFNet** replaces spatial self-attention with learnable global filters in the 2-D frequency domain, achieving log-linear complexity while capturing long-range interactions. **ViT-UNet** applies a pure Transformer encoder with multi-head self-attention to patch sequences, augmented by U-Net-style residual skip connections that feed shallow encoder states into the symmetric decoder layers to preserve multi-scale representations.

Each model is trained independently, evaluated against the same held-out test set (for GFNet and ViT) or a common train/test split (for AlexNet, which uses a legacy configuration for downstream uncertainty compatibility), and their predictions are projected back onto the full scene grid to produce coloured classification maps. The notebook is therefore both a performance benchmark and a source of calibrated probability outputs for downstream uncertainty estimation work.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let the multispectral scene be represented as a tensor $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$, where $H = 330$, $W = 307$, and $B = 6$ (spectral bands). Each band is normalised independently to $[0, 1]$ via min-max scaling:

$$\tilde{x}_{h,w,b} = \frac{x_{h,w,b} - \min_b}{\max_b - \min_b + \epsilon}$$

**Where:**
- $x_{h,w,b}$ — raw reflectance value at pixel $(h,w)$ in band $b$
- $\min_b, \max_b$ — band-wise minimum and maximum over the full scene
- $\epsilon = 10^{-8}$ — numerical stability constant to prevent division by zero

A reference label map $\mathcal{Y} \in \{0, 1, \ldots, K\}^{H \times W}$ assigns a class identity to each pixel (0 = unlabelled background; $1 \ldots K$ = land-cover classes). Only labelled pixels ($y_{h,w} > 0$) are used for training and evaluation.

### 2.2 Patch Extraction

For each labelled pixel at position $(r, c)$, a square neighbourhood of side $P = 9$ is extracted from the edge-padded scene:

$$\mathbf{X}^{(i)} = \tilde{\mathcal{X}}_{\text{pad}}\bigl[r:r+P,\; c:c+P,\; :\bigr] \in \mathbb{R}^{P \times P \times B}$$

**Where:**
- $\tilde{\mathcal{X}}_{\text{pad}}$ — the normalised scene padded by $\lfloor P/2 \rfloor = 4$ pixels on each side using edge-replication
- $P = 9$ — patch side length (spatial context window)
- $B = 6$ — number of spectral bands

The class label for sample $i$ is $y^{(i)} = \mathcal{Y}[r, c] - 1 \in \{0, \ldots, K-1\}$ (shifted to zero-indexed integers for Keras compatibility).

### 2.3 Per-Band Min-Max Normalisation

Normalisation is applied per band rather than globally to prevent spectral bands with large dynamic ranges from dominating those with smaller ranges:

$$\tilde{x}_b = \frac{x_b - \mu_b^{\min}}{\mu_b^{\max} - \mu_b^{\min} + \epsilon}$$

This ensures all input features lie in $[0, 1]$ before patch extraction.

### 2.4 The Three Architecture Families

#### 2.4.1 AlexNet-style CNN

AlexNet classifies a patch $\mathbf{X}^{(i)}$ by stacking convolutional layers (each followed by ReLU activations) to build hierarchical feature maps, then pooling and flattening into a vector that is processed by fully connected layers:

$$\hat{\mathbf{p}}^{(i)} = \text{softmax}\bigl(\mathbf{W}_{\text{fc}} \cdot \phi_{\text{conv}}(\mathbf{X}^{(i)})\bigr)$$

**Where:**
- $\phi_{\text{conv}}$ — composition of 5 convolutional layers with filter counts $[96, 256, 384, 384, 256]$, followed by max-pooling and flattening
- $\mathbf{W}_{\text{fc}}$ — weights of the fully connected head (units: $[4096, 1024, 256, 32, K]$)
- $\hat{\mathbf{p}}^{(i)} \in [0,1]^K$ — predicted class probability vector

Dropout ($p = 0.25$) is inserted between dense layers to regularise training.

#### 2.4.2 Global Filter Network (GFNet)

GFNet replaces self-attention with a frequency-domain token mixing operation. The patch is first divided into non-overlapping inner patches (side $= 3$), yielding a sequence of $N = 9$ tokens, each of dimension $d = 512$.

The central operation in each GFNet block is the **Global Filter Layer**:

$$\tilde{\mathbf{X}}_{\ell} = \mathcal{F}^{-1}\!\bigl[\mathcal{F}[\mathbf{X}_{\ell}] \odot \mathbf{W}_{\ell}\bigr]$$

**Where:**
- $\mathcal{F}[\cdot]$ — 2-D discrete Fourier transform (DFT), converting the token grid to frequency domain
- $\mathbf{W}_{\ell} \in \mathbb{C}^{T \times T \times d}$ — learnable complex filter weights (real part $\mathbf{w}^r$ and imaginary part $\mathbf{w}^i$ are trained separately), where $T = \sqrt{N}$ is the token grid side
- $\odot$ — element-wise Hadamard product in frequency space
- $\mathcal{F}^{-1}[\cdot]$ — inverse 2-D DFT, mapping filtered features back to spatial domain
- $\tilde{\mathbf{X}}_{\ell}$ — token sequence after frequency filtering in block $\ell$

**What this means:** A standard convolutional filter sees only a local neighbourhood. By operating in the Fourier domain, the global filter simultaneously modulates *all* spatial frequencies — effectively computing a weighted interaction between every pair of spatial positions with only $O(N \log N)$ operations (via FFT), compared to $O(N^2)$ for self-attention.

Each GFNet block combines the global filter with a two-layer MLP and a residual connection:

$$\mathbf{X}_{\ell+1} = \mathbf{X}_{\ell} + \text{MLP}\bigl(\text{LN}(\tilde{\mathbf{X}}_{\ell})\bigr)$$

After $L = 5$ such blocks, global average pooling collapses the token sequence to a single vector, which is passed through a linear classification head.

#### 2.4.3 Vision Transformer with U-Net Skip Connections (ViT-UNet)

The ViT processes the patch as a sequence of tokens. Each inner patch (side $= 3$) is linearly projected to a $d = 256$ dimensional embedding, a learnable **[CLS] token** is prepended, and learned positional embeddings are added:

$$\mathbf{Z}_0 = \bigl[\mathbf{z}_{\text{cls}};\; \mathbf{E}_1;\ldots;\mathbf{E}_N\bigr] + \mathbf{P}$$

**Where:**
- $\mathbf{z}_{\text{cls}} \in \mathbb{R}^d$ — trainable classification token initialised to zero
- $\mathbf{E}_i = \mathbf{W}_p \cdot \text{vec}(\text{patch}_i) \in \mathbb{R}^d$ — linear patch embedding for token $i$
- $\mathbf{P} \in \mathbb{R}^{(N+1) \times d}$ — learned positional embedding matrix

Each Transformer block applies pre-layer-normalisation multi-head self-attention (MHSA) followed by a GELU MLP:

$$\mathbf{Z}'_{\ell} = \text{MHSA}\bigl(\text{LN}(\mathbf{Z}_{\ell})\bigr) + \mathbf{Z}_{\ell}$$
$$\mathbf{Z}_{\ell+1} = \text{MLP}\bigl(\text{LN}(\mathbf{Z}'_{\ell})\bigr) + \mathbf{Z}'_{\ell}$$

The **U-Net skip connections** add the output of block $i$ (encoder side) to the output of the symmetric block $L - i$ (decoder side) for $i \leq \lfloor L/2 \rfloor$:

$$\mathbf{Z}_{L-i} \leftarrow \mathbf{Z}_{L-i} + \mathbf{Z}_{i}, \quad i = 1, \ldots, \lfloor L/2 \rfloor$$

**What this means:** Early Transformer blocks capture low-level patch structure; later blocks capture high-level semantics. The skip connections allow the final classification layers to access both levels simultaneously, analogous to how U-Net skip paths in convolutional segmentation models recover fine-grained spatial detail lost during downsampling.

After $L = 12$ blocks, the [CLS] token representation is extracted, passed through a four-layer GELU MLP classification head, and projected to a $K$-way softmax output.

### 2.5 Loss Functions and Optimisation

**AlexNet** uses standard sparse categorical cross-entropy with an Adagrad optimiser and a cosine learning-rate schedule that cycles between $\text{LR}_{\min} = 0.005$ and $\text{LR}_{\max} = 0.02$.

**GFNet and ViT** use label-smoothed categorical cross-entropy with smoothing factor $\epsilon_s = 0.05$:

$$\mathcal{L}_{\text{smooth}} = (1 - \epsilon_s)\,\mathcal{L}_{\text{CE}} + \frac{\epsilon_s}{K}$$

**Where:**
- $\mathcal{L}_{\text{CE}}$ — standard cross-entropy between softmax predictions and one-hot targets
- $\epsilon_s = 0.05$ — label smoothing factor
- $K$ — number of classes

Label smoothing prevents the model from becoming overconfident on training samples, which improves calibration. Both models are optimised with AdamW (weight decay $= 10^{-4}$, gradient clipping at norm 1.0) under a cosine decay schedule with final learning rate $\alpha = 0.05 \times \text{LR}_{\text{init}}$.

### 2.6 Calibration Metrics

Beyond accuracy, two calibration-aware metrics are tracked.

**Multiclass Brier Score** measures the mean squared error between the predicted probability vector $\hat{\mathbf{p}}^{(i)}$ and the one-hot target vector $\mathbf{e}_{y^{(i)}}$:

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} \bigl(\hat{p}_k^{(i)} - \mathbf{1}[y^{(i)} = k]\bigr)^2$$

A perfectly confident and correct model scores 0; a model that is maximally wrong scores 2 (for $K \geq 2$).

**Expected Calibration Error (ECE)** partitions the test set into $M = 15$ equal-width confidence bins $\{B_m\}$ and measures the weighted gap between average confidence and average accuracy within each bin:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \bigl|\overline{\text{acc}}(B_m) - \overline{\text{conf}}(B_m)\bigr|$$

**Where:**
- $\overline{\text{acc}}(B_m)$ — fraction of samples in bin $m$ correctly classified
- $\overline{\text{conf}}(B_m)$ — mean maximum predicted probability in bin $m$
- $|B_m|$ — number of samples in bin $m$

A perfectly calibrated model has ECE $= 0$.

---

## 3. Algorithm

**Input:** Multispectral scene $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$, label map $\mathcal{Y} \in \mathbb{Z}^{H \times W}$, patch size $P$, number of classes $K$, architecture config $\theta$

**Output:** Trained model weights, classification metrics, scene-level prediction maps

1. **Normalise** each of the $B$ spectral bands independently to $[0, 1]$ using per-band min-max scaling.
2. **Extract patches**: for every pixel $(r, c)$ with $\mathcal{Y}[r,c] > 0$, extract the $P \times P \times B$ neighbourhood patch and record the class label $k = \mathcal{Y}[r,c] - 1$.
3. **Split**: stratified random split of labelled samples into train (75%), validation (20% of train), and test (25%) sets. AlexNet uses a separate legacy split (seed 10, no validation set) for compatibility with downstream uncertainty notebooks.
4. **Build model**: instantiate the chosen architecture (AlexNet / GFNet / ViT-UNet) with the configured hyperparameters.
5. **Compile**: choose optimiser and loss (Adagrad + sparse CE for AlexNet; AdamW + label-smoothed CE for GFNet/ViT).
6. **Train** for 100 epochs with the configured batch size (128). Save the best checkpoint (best validation accuracy for AlexNet; best validation loss for others). On `ResourceExhaustedError`, automatically retry with a smaller fallback config.
7. **Evaluate** on the test set: compute accuracy, Cohen's κ, macro-F1, weighted-F1, NLL, Brier score, and ECE.
8. **Save results** to CSV (summary) and JSON (per-model classification report).
9. **Visualise**: generate training curves, cross-model bar charts, calibration proxy charts, and confusion matrices.
10. **Dense scene inference**: load the best saved model and slide the $P \times P$ window across every pixel in the full scene (row by row), producing an $(H \times W)$ predicted label map.
11. **Export**: save individual PNGs of the RGB composite, ground-truth map, and each model's prediction map; embed all in an Excel workbook.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_training.ipynb`

### 4.1 Reproducibility & Configuration (Section 2.0)

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

**What this does:** Seeds Python, NumPy, and TensorFlow random states to the same value, ensuring that all stochastic operations (weight initialisation, data shuffling, dropout masks) produce the same result on repeated runs.

**Why:** Reproducibility is essential for a comparative study. Without it, differences between runs could be attributed to randomness rather than architectural differences.

The notebook uses a single configuration cell (`Section 2.0`) as the canonical source of truth for all hyperparameters. This includes patch geometry, split ratios, learning rates, architecture sizes, and file paths. Changing a single value in this cell propagates throughout the entire notebook — a best practice for experiment management.

---

### 4.2 Multispectral Loading & Normalisation (Section 3.1)

```python
def load_multispectral_6band(data_path, label_path, height, width, bands):
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(height, width, bands)
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(height, width)
    x_norm = np.empty_like(x, dtype=np.float32)
    for b in range(bands):
        band = x[:, :, b]
        denom = max(np.max(band) - np.min(band), 1e-8)
        x_norm[:, :, b] = (band - np.min(band)) / denom
    return x_norm, y
```

**What this does:** Reads the raw spectral data from CSV into a 3-D array (H × W × B), reads the class labels into a 2-D integer array, and applies independent per-band min-max normalisation. The `max(..., 1e-8)` guard prevents division by zero if a band is constant across the scene.

**Why:** Normalising to [0, 1] per band prevents scale-induced gradient imbalance during training. Per-band (rather than global) normalisation respects the fact that different sensors or spectral ranges may have vastly different dynamic ranges.

---

### 4.3 Patch Extraction (Section 3.1)

```python
def extract_labeled_patches(x, y, patch_size=9):
    pad = patch_size // 2
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    coords = np.argwhere(y > 0)
    patches = np.empty((coords.shape[0], patch_size, patch_size, x.shape[-1]), dtype=np.float32)
    labels = np.empty((coords.shape[0],), dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
        labels[i] = int(y[r, c]) - 1
    return patches, labels, coords
```

**What this does:** Pads the scene with edge-replicated borders (4 pixels on each side for a 9 × 9 patch), then iterates over all labelled pixel positions and extracts the centred 9 × 9 × 6 neighbourhood. Labels are converted from 1-indexed to 0-indexed integers.

**Why:** Edge padding ensures pixels near the scene boundary still receive a full-sized patch without cropping. Using `mode="edge"` replicates boundary values, which is preferable to zero-padding because it doesn't introduce an artificial discontinuity. Storing the coordinates alongside the patches enables projecting predictions back onto the spatial grid later.

---

### 4.4 Custom Keras Layers (Section 4.1)

Three custom layers underpin the GFNet and ViT architectures. All inherit from `layers.Layer` and implement `get_config()` for serialisation, enabling `model.save()` and reloading with `custom_objects`.

**`PatchExtractor`**: uses `tf.image.extract_patches` to divide the 9 × 9 input into non-overlapping 3 × 3 sub-patches, producing $N = 9$ tokens, each of dimension $9 \times B = 54$ raw features.

**`PatchPositionEncoder`**: projects the raw patch tokens to `hidden_dim` via a dense layer and adds a learned position embedding table of shape $(N, d)$. This is the standard ViT patch embedding stage.

**`PatchEncoderWithCLS`**: extends `PatchPositionEncoder` by prepending a trainable [CLS] token (shape $(1, 1, d)$ tiled to the batch) and expanding the position embedding to $N+1$ positions. The [CLS] token accumulates global context through self-attention and is extracted after the final Transformer block for classification.

---

### 4.5 AlexNet Architecture (Section 4.2)

```python
for i, filters in enumerate(cfg["conv_filters"], start=1):
    x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
x = layers.Flatten()(x)
for units in cfg["dense_units"]:
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
```

**What this does:** Five 3 × 3 convolutional layers with filter counts [96, 256, 384, 384, 256] and ReLU activations extract local spatial features. A single max-pooling layer downsamples the feature map. The flattened feature vector is passed through four dense layers with dropout between each, and the final softmax layer produces class probabilities.

**Why:** This structure follows the AlexNet spirit: shallow convolutional feature extraction followed by a deep fully connected classifier. The adaptation to 3 × 3 kernels (rather than the original AlexNet's 11 × 11 kernels) is appropriate because the input patches are only 9 × 9 pixels — large kernels would see very little multi-scale information. All convolutional layers have `padding="same"` so the spatial dimensions are preserved until the single pooling layer.

---

### 4.6 Global Filter Layer (Section 4.3)

```python
def call(self, x):
    x_2d = tf.reshape(x, [batch, self.token_side, self.token_side, channels])
    x_fft = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
    w_complex = tf.complex(self.w_real, self.w_imag)
    x_filtered = x_fft * w_complex
    x_spatial = tf.math.real(tf.signal.ifft2d(x_filtered))
    return tf.reshape(x_spatial, [batch, self.token_side * self.token_side, channels])
```

**What this does:** Reshapes the token sequence back into a 2-D grid ($\sqrt{N} \times \sqrt{N}$), applies the 2-D FFT channel-wise to get frequency-domain representations, multiplies element-wise by learnable complex weights (real and imaginary stored separately), applies the inverse FFT, and takes the real part of the result.

**Why:** The Fourier transform diagonalises convolution operators. A learnable complex filter in the frequency domain is mathematically equivalent to a learnable convolution in the spatial domain, but with global (all-to-all) receptive field rather than a limited kernel window — at the cost of only $O(N \log N)$ operations via the FFT algorithm. The `tf.math.real()` projection discards imaginary artefacts that arise because the input is real-valued but the complex multiplication can produce non-zero imaginary components.

---

### 4.7 ViT Transformer Block and U-Net Skip Connections (Section 4.4)

```python
for i in range(transformer_layers):
    x = transformer_block(x, ...)
    if i <= transformer_layers // 2:
        block_list.append(x)
    else:
        x = layers.Add()([x, block_list[transformer_layers - i - 1]])
```

**What this does:** The first $\lfloor L/2 \rfloor + 1$ blocks form the "encoder" — their outputs are saved in `block_list`. The remaining "decoder" blocks have the corresponding saved encoder output added (via an `Add` layer) to the current representation before proceeding to the next block. The exact pairing is symmetric: block $L - i - 1$ in the encoder is added to block $i$ in the decoder.

**Why:** Deep Transformer stacks can progressively lose early-layer structural information as attention patterns become more abstract. The skip additions feed earlier representations (local structure, patch boundaries) directly into later processing stages, mitigating this information loss — exactly the role that skip paths play in U-Net for spatial segmentation. After the stack, only the [CLS] token at position 0 is extracted for classification; the patch tokens are discarded.

---

### 4.8 Training Loop with OOM Fallback (Section 6.0)

```python
try:
    row, report, cm, history = train_save_evaluate(model_name, builder, ...)
except tf.errors.ResourceExhaustedError:
    if model_name == "GFNet":
        row, ... = train_save_evaluate(model_name,
            lambda: build_gfnet_with_cfg(GFNET_FALLBACK_CFG), capacity_tag="fallback")
```

**What this does:** Wraps each model's training in a `try/except` block. If TensorFlow raises a `ResourceExhaustedError` (GPU out-of-memory), the session is cleared and training is retried immediately with a reduced-capacity configuration — `hidden_dim` reduced from 512 → 384 for GFNet; `projection_dim` reduced from 256 → 192 and `transformer_layers` from 12 → 8 for ViT-UNet.

**Why:** Colab's GPU memory is finite and non-deterministically allocated. Rather than failing the notebook, automatic fallback lets the training pipeline complete with the largest configuration the hardware supports. The `capacity_tag` field in the results row records which config was actually used, preserving experimental transparency.

---

### 4.9 Dense Scene Inference (Section 8.2)

```python
def predict_full_scene_labels(model, x_img, patch_size=9, batch_size=256):
    pad = patch_size // 2
    x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    preds = np.zeros((x_img.shape[0], x_img.shape[1]), dtype=np.int32)
    for r in range(x_img.shape[0]):
        row_patches = ...  # extract all W patches for row r
        row_prob = model.predict(row_patches, batch_size=batch_size, verbose=0)
        preds[r] = np.argmax(row_prob, axis=1) + 1
    return preds
```

**What this does:** Pads the full scene and iterates row by row, extracting one batch of $W = 307$ patches (one per column), running the model to get class probabilities, taking the argmax, and writing the 1-indexed predicted class back into a $(H \times W)$ array.

**Why:** Predicting row by row (rather than flattening all pixels at once) limits peak memory consumption to $W \times P \times P \times B \approx 307 \times 9 \times 9 \times 6 \approx 150\,\text{K}$ floats per batch, which fits comfortably in memory regardless of total scene size. Processing in mini-batches of 256 within each row further reduces memory pressure.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

We construct a toy 6-class multispectral classification problem with $K = 6$ classes, $n = 18$ calibration samples (3 per class), $\alpha = 0.10$, and patch features represented by a single scalar softmax confidence per class for clarity. In practice the patch is 9 × 9 × 6, but the classification head produces a 6-dimensional probability vector — it is this vector we work with here.

**Calibration set — predicted probability vectors (true class in **bold**):**

| Sample | True Class | $\hat{p}_1$ | $\hat{p}_2$ | $\hat{p}_3$ | $\hat{p}_4$ | $\hat{p}_5$ | $\hat{p}_6$ |
|--------|------------|-------------|-------------|-------------|-------------|-------------|-------------|
| 1 | 1 | **0.72** | 0.10 | 0.08 | 0.04 | 0.03 | 0.03 |
| 2 | 1 | **0.65** | 0.14 | 0.10 | 0.05 | 0.03 | 0.03 |
| 3 | 1 | **0.58** | 0.18 | 0.12 | 0.06 | 0.04 | 0.02 |
| 4 | 2 | 0.08 | **0.74** | 0.08 | 0.04 | 0.03 | 0.03 |
| 5 | 2 | 0.10 | **0.63** | 0.14 | 0.07 | 0.04 | 0.02 |
| 6 | 2 | 0.12 | **0.55** | 0.16 | 0.09 | 0.05 | 0.03 |
| 7 | 3 | 0.05 | 0.09 | **0.71** | 0.08 | 0.04 | 0.03 |
| 8 | 3 | 0.06 | 0.10 | **0.62** | 0.12 | 0.06 | 0.04 |
| 9 | 3 | 0.07 | 0.12 | **0.53** | 0.15 | 0.08 | 0.05 |
| 10 | 4 | 0.04 | 0.05 | 0.08 | **0.70** | 0.08 | 0.05 |
| 11 | 4 | 0.05 | 0.07 | 0.10 | **0.60** | 0.12 | 0.06 |
| 12 | 4 | 0.06 | 0.08 | 0.12 | **0.52** | 0.14 | 0.08 |
| 13 | 5 | 0.03 | 0.04 | 0.06 | 0.09 | **0.73** | 0.05 |
| 14 | 5 | 0.04 | 0.05 | 0.08 | 0.11 | **0.64** | 0.08 |
| 15 | 5 | 0.05 | 0.06 | 0.10 | 0.13 | **0.57** | 0.09 |
| 16 | 6 | 0.03 | 0.03 | 0.05 | 0.07 | 0.09 | **0.73** |
| 17 | 6 | 0.04 | 0.04 | 0.06 | 0.08 | 0.10 | **0.68** |
| 18 | 6 | 0.05 | 0.05 | 0.08 | 0.10 | 0.13 | **0.59** |

**Test set — three representative samples:**

| Test | True Class | $\hat{p}_1$ | $\hat{p}_2$ | $\hat{p}_3$ | $\hat{p}_4$ | $\hat{p}_5$ | $\hat{p}_6$ | Scenario |
|------|------------|-------------|-------------|-------------|-------------|-------------|-------------|----------|
| T1 | 3 | 0.05 | 0.07 | **0.78** | 0.05 | 0.03 | 0.02 | Easy (top class = 0.78) |
| T2 | 2 | 0.12 | **0.41** | 0.28 | 0.10 | 0.05 | 0.04 | Borderline (true class not dominant) |
| T3 | 4 | 0.20 | 0.19 | 0.18 | **0.22** | 0.12 | 0.09 | Ambiguous (no class dominates) |

---

### 5.1 Softmax Argmax Classifier (Baseline)

Before considering any uncertainty method, the standard classifier simply predicts the argmax of the softmax output.

#### Step A — Predictions on calibration set

| Sample | True Class | Predicted Class | $\hat{p}_{\text{true}}$ | Correct? |
|--------|------------|-----------------|--------------------------|----------|
| 1–3 | 1 | 1 | 0.72 / 0.65 / 0.58 | ✓ ✓ ✓ |
| 4–6 | 2 | 2 | 0.74 / 0.63 / 0.55 | ✓ ✓ ✓ |
| 7–9 | 3 | 3 | 0.71 / 0.62 / 0.53 | ✓ ✓ ✓ |
| 10–12 | 4 | 4 | 0.70 / 0.60 / 0.52 | ✓ ✓ ✓ |
| 13–15 | 5 | 5 | 0.73 / 0.64 / 0.57 | ✓ ✓ ✓ |
| 16–18 | 6 | 6 | 0.73 / 0.68 / 0.59 | ✓ ✓ ✓ |

The argmax classifier is correct on all 18 calibration samples — a 100% calibration accuracy on this clean toy dataset.

#### Step B — Predictions on test set

| Test | True Class | Predicted Class | $\hat{p}_{\text{pred}}$ | Correct? |
|------|------------|-----------------|--------------------------|----------|
| T1 | 3 | 3 | 0.78 | ✓ |
| T2 | 2 | 2 | 0.41 | ✓ |
| T3 | 4 | 4 | 0.22 | ✓ |

All three test samples receive the correct argmax prediction. However, T2 and T3 have low peak confidence, indicating the model is genuinely uncertain — this is the information that calibration metrics capture.

---

### 5.2 Expected Calibration Error (ECE) Calculation

This example demonstrates how the notebook computes ECE with $M = 15$ equal-width bins.

**Step A — Collect confidence–correctness pairs on the calibration set:**

Confidence = max predicted probability; correctness = argmax matches true class.

| Sample | Confidence | Correct? |
|--------|-----------|---------|
| 1 | 0.72 | 1 |
| 2 | 0.65 | 1 |
| 3 | 0.58 | 1 |
| 4 | 0.74 | 1 |
| 5 | 0.63 | 1 |
| 6 | 0.55 | 1 |
| 7 | 0.71 | 1 |
| 8 | 0.62 | 1 |
| 9 | 0.53 | 1 |
| 10 | 0.70 | 1 |
| 11 | 0.60 | 1 |
| 12 | 0.52 | 1 |
| 13 | 0.73 | 1 |
| 14 | 0.64 | 1 |
| 15 | 0.57 | 1 |
| 16 | 0.73 | 1 |
| 17 | 0.68 | 1 |
| 18 | 0.59 | 1 |

**Step B — Assign to bins (width = 1/15 ≈ 0.0667):**

Bin boundaries relevant to our data: [0.467, 0.533), [0.533, 0.600), [0.600, 0.667), [0.667, 0.733), [0.733, 0.800)

| Bin Range | Samples | $\overline{\text{acc}}$ | $\overline{\text{conf}}$ | $|\Delta|$ | Weight | ECE contrib. |
|-----------|---------|------------------------|--------------------------|-----------|--------|--------------|
| [0.467, 0.533) | 9, 12 → conf: 0.53, 0.52 | 1.00 | 0.525 | 0.475 | 2/18 | 0.0528 |
| [0.533, 0.600) | 3, 6, 15, 18 → 0.58, 0.55, 0.57, 0.59 | 1.00 | 0.573 | 0.427 | 4/18 | 0.0949 |
| [0.600, 0.667) | 2, 5, 8, 11, 14, 17 → 0.65, 0.63, 0.62, 0.60, 0.64, 0.68 | 1.00 | 0.637 | 0.363 | 6/18 | 0.1210 |
| [0.667, 0.733) | 7, 10, 4, 13, 16, 1 → 0.71, 0.70, 0.74, 0.73, 0.73, 0.72 | 1.00 | 0.722 | 0.278 | 6/18 | 0.0927 |

**Step C — Sum ECE contributions:**

$$\text{ECE} = 0.0528 + 0.0949 + 0.1210 + 0.0927 = 0.361$$

**Interpretation:** Even though the model achieves 100% accuracy on this toy set, the ECE of 0.361 indicates poor calibration — the model's confidence (≈ 0.55–0.73) substantially undershoots its actual accuracy (1.00). In other words, the model is underconfident on this clean dataset. A real model with noisier classes and lower accuracy would show a different pattern — possibly overconfidence — and the ECE would reflect the mismatch in the opposite direction.

---

### 5.3 Multiclass Brier Score Calculation

Using the calibration set probability vectors:

$$\text{BS} = \frac{1}{18} \sum_{i=1}^{18} \sum_{k=1}^{6} (\hat{p}_k^{(i)} - \mathbf{1}[y^{(i)} = k])^2$$

For sample 1 (true class 1, $\hat{\mathbf{p}} = [0.72, 0.10, 0.08, 0.04, 0.03, 0.03]$):

$$\text{BS}_1 = (0.72 - 1)^2 + (0.10 - 0)^2 + (0.08)^2 + (0.04)^2 + (0.03)^2 + (0.03)^2$$
$$= 0.0784 + 0.0100 + 0.0064 + 0.0016 + 0.0009 + 0.0009 = 0.0982$$

For sample 9 (true class 3, $\hat{\mathbf{p}} = [0.07, 0.12, 0.53, 0.15, 0.08, 0.05]$):

$$\text{BS}_9 = (0.07)^2 + (0.12)^2 + (0.53 - 1)^2 + (0.15)^2 + (0.08)^2 + (0.05)^2$$
$$= 0.0049 + 0.0144 + 0.2209 + 0.0225 + 0.0064 + 0.0025 = 0.2716$$

Sample 9 contributes a much higher Brier score than sample 1 because the model is less confident about class 3 (0.53 vs 0.72). Averaging across all 18 samples gives the overall Brier score.

**Interpretation:** The Brier score rewards sharpness (high confidence on the true class) and penalises overconfidence on wrong classes. A model that assigns $\hat{p} = 1.0$ to the correct class always scores 0; uniform assignment ($\hat{p}_k = 1/K$ for all $k$) scores $\frac{2(K-1)}{K} = \frac{10}{6} \approx 1.67$ for $K = 6$.

---

### 5.4 Test Sample Analysis

For the three test samples, applying the argmax classifier:

| Test | True Class | Predicted | Peak Confidence | Brier Score | Interpretation |
|------|------------|-----------|-----------------|-------------|----------------|
| T1 | 3 | 3 | 0.78 | low | Easy: model is sharp and correct |
| T2 | 2 | 2 | 0.41 | medium | Borderline: correct but uncertain |
| T3 | 4 | 4 | 0.22 | high | Ambiguous: barely correct, high uncertainty |

For T3, the Brier score contribution:

$$\text{BS}_{T3} = (0.20)^2 + (0.19)^2 + (0.18)^2 + (0.22-1)^2 + (0.12)^2 + (0.09)^2$$
$$= 0.04 + 0.0361 + 0.0324 + 0.6084 + 0.0144 + 0.0081 = 0.7394$$

This high per-sample Brier score reflects the nearly uniform probability distribution — the model is genuinely unsure. In the downstream conformal prediction notebooks, T3 would likely receive a large prediction set (many classes included to achieve coverage) while T1 would receive a singleton set.

---

## 6. References

[1] Krizhevsky, A., Sutskever, I., and Hinton, G.E. "ImageNet Classification with Deep Convolutional Neural Networks." *Advances in Neural Information Processing Systems 25 (NeurIPS)*, 2012. [Link](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

[2] Rao, Y., Zhao, W., Zhu, Z., Lu, J., and Zhou, J. "Global Filter Networks for Image Classification." *Advances in Neural Information Processing Systems 34 (NeurIPS)*, 2021. [arXiv:2107.00645](https://arxiv.org/abs/2107.00645)

[3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

[4] Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

[5] Ahmad, M., et al. "Hyperspectral Image Classification — Traditional to Deep Models: A Survey for Future Prospects." *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2022. [arXiv:2101.06116](https://arxiv.org/abs/2101.06116)

[6] Niculescu-Mizil, A. and Caruana, R. "Predicting Good Probabilities with Supervised Learning." *Proceedings of the 22nd International Conference on Machine Learning (ICML)*, 2005. [ACM DL](https://dl.acm.org/doi/10.1145/1102351.1102430)
