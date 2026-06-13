# Multi-Architecture Multispectral Patch Classification: Theory & Implementation Summary

> **One-line description:** Trains and evaluates three deep learning classifiers — an AlexNet-inspired CNN, a Global Filter Network (GFNet), and a Vision Transformer with U-Net skip connections — on a 7-class multispectral remote-sensing dataset, with per-model calibration metrics to support downstream uncertainty analysis.

---

## 1. Overview & Intuition

### What Problem Does This Solve?

Remote sensing images captured by multispectral sensors contain multiple spectral bands beyond the visible range, allowing discrimination of land-cover classes (e.g., vegetation types, water bodies, built-up areas) that would be indistinguishable in standard RGB imagery. Pixel-wise classification of such scenes — assigning each labelled pixel to one of several semantic classes — is a core task in geospatial analysis.

The notebook addresses this problem by:

1. Extracting 9×9×6 spatial patches centred on each labelled pixel from a 330×307×6 multispectral image.
2. Training three architecturally distinct classifiers on those patches.
3. Recording not only accuracy but also calibration metrics (Brier score, Expected Calibration Error, Negative Log-Likelihood) so that each model can later serve as a base for conformal or Bayesian uncertainty estimation.

### Why Multiple Architectures?

Different inductive biases capture different aspects of the data:

- **AlexNet-CNN** exploits local spatial convolutions — efficient for textures and edges, with a well-understood training recipe.
- **GFNet** operates in the frequency domain, learning *global* filter patterns that span the entire receptive field in a single element-wise operation. This is efficient yet expressive for capturing long-range spectral–spatial correlations.
- **ViT-UNet** applies self-attention across patch tokens with encoder–decoder skip connections, combining the global context of Transformers with multi-scale feature reuse akin to U-Net.

Together, these three models provide a calibration baseline for ensemble or uncertainty-quantification studies.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let the raw multispectral image be a tensor $\mathbf{X} \in \mathbb{R}^{H \times W \times B}$ where $H = 330$, $W = 307$, $B = 6$ (spectral bands). A corresponding label map $\mathbf{Y} \in \{0, 1, \ldots, C-1\}^{H \times W}$ assigns a class (or background) to each pixel.

**Per-band normalisation** maps each band $b$ to $[0,1]$:
$$\hat{x}_{h,w,b} = \frac{x_{h,w,b} - \min_b}{\max_b - \min_b + \varepsilon}, \quad \varepsilon = 10^{-8}$$

**Where:**
- $x_{h,w,b}$ — raw reflectance value at pixel $(h,w)$, band $b$
- $\min_b,\, \max_b$ — per-band minimum and maximum
- $\varepsilon$ — small constant preventing division by zero

**Patch extraction:** Around every labelled pixel $(r,c)$, a $P \times P$ patch (with $P = 9$) is extracted after edge-padding, yielding a dataset $\mathcal{D} = \{(\mathbf{p}_i, y_i)\}_{i=1}^{N}$ with $\mathbf{p}_i \in \mathbb{R}^{9 \times 9 \times 6}$ and $y_i \in \{0,\ldots,6\}$ (7 classes, IDs shifted to be zero-indexed).

**Stratified splits:**

| Split | Fraction | Samples (approx.) |
|-------|----------|-------------------|
| Train (GFNet/ViT) | 60% | 10,343 |
| Validation | 15% | 2,586 |
| Test | 25% | 4,310 |
| Train (AlexNet, separate seed) | 75% | 12,929 |

### 2.2 Classification Objective

All three models minimise a cross-entropy loss. For GFNet and ViT, **label smoothing** is applied:

$$\mathcal{L}_{\text{smooth}} = (1 - \alpha)\,\mathcal{L}_{\text{CE}} + \frac{\alpha}{C}\,\sum_{c=1}^{C} \log \hat{p}_c$$

**Where:**
- $\mathcal{L}_{\text{CE}}$ — standard categorical cross-entropy
- $\alpha = 0.05$ — smoothing factor
- $C = 7$ — number of classes
- $\hat{p}_c$ — predicted softmax probability for class $c$

Label smoothing penalises over-confident predictions, improving calibration.

AlexNet uses standard sparse categorical cross-entropy (integer labels, no smoothing).

---

## 3. Architecture Descriptions

### 3.1 AlexNet-Inspired CNN

The AlexNet-inspired CNN adapts the classic five-convolution architecture for small 9×9 patches.

**Key operations:**

$$\mathbf{h}^{(l)} = \text{ReLU}\!\left(\mathbf{W}^{(l)} * \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l = 1,\ldots,5$$

After five conv layers (filters: 96, 256, 384, 384, 256), a 2×2 max-pool collapses spatial dimensions, then four dense layers (4096 → 1024 → 256 → 32 units) with Dropout ($p = 0.25$) lead to a 7-way softmax head.

**Optimiser:** Adagrad with a cosine learning-rate schedule oscillating between 0.005 and 0.02 — preserving the original legacy recipe needed for downstream uncertainty recovery.

### 3.2 Global Filter Network (GFNet)

GFNet replaces self-attention with frequency-domain filtering, achieving log-linear (rather than quadratic) complexity in the sequence length.

#### 3.2.1 Patch Tokenisation

The 9×9 input is divided into non-overlapping 3×3 inner patches, yielding a sequence of $T = (9/3)^2 = 9$ tokens. Each token (of dimension $3 \times 3 \times 6 = 54$) is projected to a hidden dimension $d = 512$ by a learnable linear layer, and sinusoidal positional embeddings are added.

#### 3.2.2 Global Filter Layer — Core Operation

The core of each GFNet block:

**Step 1 — 2-D FFT:**
$$\mathbf{X}_f = \mathcal{F}_{2D}(\mathbf{X}_s) \in \mathbb{C}^{T_h \times T_w \times d}$$

**Step 2 — Complex-valued element-wise multiplication:**
$$\widetilde{\mathbf{X}}_f = \mathbf{X}_f \odot \mathbf{K}, \quad \mathbf{K} = \mathbf{K}_r + j\,\mathbf{K}_i$$

**Step 3 — Inverse 2-D FFT (taking real part):**
$$\mathbf{X}_s' = \text{Re}\!\left[\mathcal{F}^{-1}_{2D}(\widetilde{\mathbf{X}}_f)\right]$$

**Where:**
- $\mathbf{X}_s \in \mathbb{R}^{T_h \times T_w \times d}$ — spatial token grid (reshaped from sequence)
- $\mathcal{F}_{2D}$ — 2-D Discrete Fourier Transform
- $\mathbf{K}_r, \mathbf{K}_i \in \mathbb{R}^{T_h \times T_w \times d}$ — learnable real and imaginary filter weights
- $\odot$ — Hadamard (element-wise) product in the frequency domain

**What this means:** Each learnable complex weight $K_{f,c}$ scales the contribution of frequency component $f$ in channel $c$. Because the filter operates globally across all frequencies simultaneously, the network can learn to suppress or amplify specific spatial frequencies (e.g., edges, textures) in a single operation — equivalent to a globally-receptive convolution but with $O(T \log T)$ cost instead of $O(T^2)$.

#### 3.2.3 GFNet Residual Block

Each block applies the filter inside a residual path followed by a GELU MLP:

$$\mathbf{y} = \mathbf{x} + \text{MLP}\!\left(\text{LN}_2\!\left[\text{GlobalFilter}\!\left(\text{LN}_1(\mathbf{x})\right)\right]\right)$$

The MLP expands the channel dimension by ratio 4 (to $4 \times 512 = 2048$), then contracts back. There are 5 such blocks.

**Final head:** Layer norm → GlobalAveragePooling1D → Flatten → Dropout → 7-class softmax.

### 3.3 Vision Transformer with U-Net Skip Connections (ViT-UNet)

#### 3.3.1 Patch Encoding with CLS Token

Patches are extracted at 3×3 inner size (yielding 9 tokens) and projected to dimension 256. A learnable **[CLS] token** is prepended, and learned positional embeddings (for 10 positions) are added:

$$\mathbf{z}_0 = \left[\mathbf{c};\, \mathbf{E}\mathbf{p}_1;\, \mathbf{E}\mathbf{p}_2;\, \ldots;\, \mathbf{E}\mathbf{p}_9\right] + \mathbf{P}$$

**Where:**
- $\mathbf{c} \in \mathbb{R}^{256}$ — learnable CLS token
- $\mathbf{E}$ — linear patch projection matrix
- $\mathbf{P} \in \mathbb{R}^{10 \times 256}$ — learned positional embedding table

#### 3.3.2 Pre-LN Transformer Block

Each of the 12 transformer blocks applies multi-head self-attention in a pre-normalised form:

$$\mathbf{y} = \mathbf{x} + \text{MHSA}\!\left(\text{LN}(\mathbf{x})\right)$$
$$\mathbf{z} = \mathbf{y} + \text{MLP}_{\text{GELU}}\!\left(\text{LN}(\mathbf{y})\right)$$

**Where:**
- $\text{MHSA}$ — Multi-Head Self-Attention with 4 heads, key-dimension 256
- $\text{LN}$ — Layer Normalisation
- $\text{MLP}$ — two-layer feed-forward network with GELU activations (dim: 256 → 512 → 256)

The **Multi-Head Self-Attention** computes:

$$\text{MHSA}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

across $H = 4$ parallel heads, where $d_k = 256$ per head.

#### 3.3.3 U-Net Skip Connections

Blocks are arranged in an encoder–decoder pattern. For 12 blocks:

- **Encoder** (blocks 1–7): outputs are saved to a list.
- **Decoder** (blocks 8–12): each decoder block output is added element-wise to the *mirrored* encoder output before proceeding.

Specifically, at decoder block $i$ (0-indexed from 7), the skip add is:

$$\mathbf{z}_i = \mathbf{x}_i + \text{block\_list}[L - i - 1]$$

where $L = 12$. This allows the model to directly reuse early-layer spatial features in later layers — the key idea from U-Net — preventing loss of low-level detail through deep stacking.

#### 3.3.4 Classification Head

After the 12 transformer blocks:
1. Extract CLS token: $\mathbf{v} = \mathbf{z}_{12}[:, 0, :]$
2. Four dense GELU layers (512 → 256 → 128 → 64) with Dropout
3. Final 7-class softmax

---

## 4. Calibration Metrics

The notebook measures three calibration signals beyond accuracy:

### 4.1 Multiclass Brier Score

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left(\hat{p}_{i,c} - y_{i,c}\right)^2$$

**Where:**
- $\hat{p}_{i,c}$ — predicted probability for sample $i$, class $c$
- $y_{i,c}$ — one-hot label ($1$ if $y_i = c$, else $0$)

Brier score ranges from 0 (perfect) to 2 (worst). It penalises both wrong predictions and over-confident wrong predictions.

### 4.2 Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

**Where:**
- $B_m$ — set of samples whose max-class confidence falls in the $m$-th bin of $[0,1]$ (15 equal-width bins)
- $\text{acc}(B_m)$ — fraction of correctly classified samples in bin $m$
- $\text{conf}(B_m)$ — mean max-class confidence in bin $m$

ECE measures whether the model's confidence reflects its actual accuracy. A well-calibrated model has ECE ≈ 0.

### 4.3 Negative Log-Likelihood (NLL)

$$\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{p}_{i, y_i}$$

NLL is sensitive to extreme mis-predictions (assigning near-zero probability to the true class).

---

## 5. Algorithm

**Input:** Raw multispectral image $\mathbf{X}$ (H×W×B), label map $\mathbf{Y}$  
**Output:** Three trained classifiers + performance/calibration metrics

1. **Normalise** each spectral band to $[0,1]$ independently.
2. **Extract patches** of size $9 \times 9$ around every labelled pixel; shift class IDs to $\{0,\ldots,6\}$.
3. **Split** data into train/val/test (stratified by class). AlexNet uses a separate legacy split.
4. **For each architecture** (AlexNet-CNN, GFNet, ViT-UNet):
   a. Build the model according to the architecture config.
   b. Compile with the appropriate loss, optimiser, and LR schedule.
   c. Train for 100 epochs; save the best checkpoint by monitored metric (val\_accuracy for AlexNet, val\_loss for others).
   d. Evaluate on the held-out test set.
   e. Compute accuracy, Cohen's kappa, macro-F1, weighted-F1, NLL, Brier score, ECE.
5. **Sort** results by test accuracy; save summary CSV and per-model JSON reports.
6. **Plot** training curves, cross-model bar charts, and confusion matrices.

---

## 6. Implementation Walkthrough

> Based on the uploaded notebook: `Model_training.ipynb`

### 6.1 Band Normalisation and Patch Extraction

```python
def load_multispectral_6band(data_path, label_path, height, width, bands):
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(height, width, bands)
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(height, width)
    for b in range(bands):
        band = x[:, :, b]
        denom = max(band.max() - band.min(), 1e-8)
        x_norm[:, :, b] = (band - band.min()) / denom
    return x_norm, y

def extract_labeled_patches(x, y, patch_size=9):
    pad = patch_size // 2
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    coords = np.argwhere(y > 0)
    for i, (r, c) in enumerate(coords):
        patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
        labels[i]  = int(y[r, c]) - 1   # 1-indexed → 0-indexed
```

**What this does:** Loads the CSV-encoded raster data, reshapes it, and normalises each band. Patches are extracted via edge-padding so boundary pixels are included without introducing artefacts. Labels are converted from 1-indexed to 0-indexed class IDs.  
**Why:** Edge-padding preserves the border pixels without zero-filling, which would artificially appear as a new spectral signature.

### 6.2 Global Filter Layer (Core of GFNet)

```python
def call(self, x):
    x_2d    = tf.reshape(x, [batch, self.token_side, self.token_side, channels])
    x_fft   = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
    w_complex = tf.complex(self.w_real, self.w_imag)
    x_filtered = x_fft * w_complex
    x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
    return tf.reshape(x_spatial, [batch, self.token_side * self.token_side, channels])
```

**What this does:** Reshapes the flattened token sequence into a 2-D grid, applies FFT, multiplies by a learnable complex filter, inverts back to the spatial domain, and flattens again.  
**Why:** The FFT allows the filter to act on the entire token grid simultaneously in frequency space. This is equivalent to a global, learned spatial convolution but executes in $O(T \log T)$ rather than $O(T^2)$.

### 6.3 ViT-UNet Skip Connections

```python
block_list = []
for i in range(transformer_layers):   # 12 blocks
    x = transformer_block(x, ...)
    if i <= transformer_layers // 2:
        block_list.append(x)          # encoder: save output
    else:
        x = layers.Add(...)([x, block_list[transformer_layers - i - 1]])  # decoder: skip
```

**What this does:** The first 7 blocks act as an encoder; each output is stored. The remaining 5 blocks act as a decoder, adding the saved encoder outputs (in reverse order) to the current token sequence.  
**Why:** Skip connections let the classifier head access low-level spatial detail that might be lost through successive self-attention layers, mirroring the U-Net design principle.

### 6.4 Calibration Computation

```python
def expected_calibration_error(y_true, y_prob, n_bins=15):
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct = (predictions == y_true).astype(np.float32)
    for i in range(n_bins):
        in_bin = (confidences >= lo) & (confidences < hi)
        ece += np.abs(np.mean(correct[in_bin]) - np.mean(confidences[in_bin])) * np.mean(in_bin)
```

**What this does:** Bins predictions by their maximum softmax confidence and computes the weighted absolute gap between average confidence and average accuracy in each bin.  
**Why:** Calibration is critical for downstream uncertainty methods; a model that is 80% confident should be correct 80% of the time.

---

## 7. Worked Numerical Example

**Setup:** Suppose we have 5 labelled pixels, 3 classes, and a 5×5 patch size (for simplicity).

**Step 1 — Raw values in band 1 (say):** $[0.1, 0.3, 0.5, 0.7, 0.9]$.  
After min-max normalisation: $\min = 0.1,\, \max = 0.9 \Rightarrow$ normalised values $= [0, 0.25, 0.5, 0.75, 1.0]$.

**Step 2 — Patch for pixel at position (2,2):** Extract rows/columns 0–4 (all) from the padded array — a 5×5×B patch.

**Step 3 — GFNet Global Filter (toy, 4 tokens, 2 channels):**

Token sequence after patch embedding:  
$\mathbf{X}_s = \begin{bmatrix}1+0j & 2+0j \\ 0+0j & 1+0j\end{bmatrix}$ (2×2 grid, 1 channel shown)

2-D FFT:  
$\mathbf{X}_f = \mathcal{F}_{2D}(\mathbf{X}_s) = \begin{bmatrix}4+0j & 0+0j \\ 2+0j & 0+0j\end{bmatrix}$  
(DC component = sum = 4; off-diagonal entries encode spatial frequency content)

Learnable filter (random init):  
$\mathbf{K} = \begin{bmatrix}0.5+0.1j & 1.0+0j \\ 0.8-0.2j & 0.3+0.5j\end{bmatrix}$

Element-wise multiplication:  
$\widetilde{\mathbf{X}}_f = \begin{bmatrix}2.0+0.4j & 0 \\ 1.6-0.4j & 0\end{bmatrix}$

Inverse FFT → take real part → reshape to token sequence → add to residual path and continue to MLP.

**Step 4 — Calibration example:**

After training, model outputs softmax probabilities for 3 test samples:

| True Label | $\hat{p}_0$ | $\hat{p}_1$ | $\hat{p}_2$ | Predicted | Confidence | Correct? |
|------------|------------|------------|------------|-----------|------------|---------|
| 0          | 0.85       | 0.10       | 0.05       | 0         | 0.85       | Yes     |
| 1          | 0.20       | 0.70       | 0.10       | 1         | 0.70       | Yes     |
| 2          | 0.30       | 0.30       | 0.40       | 2         | 0.40       | Yes     |

Brier score (class 0 sample): $(0.85-1)^2 + (0.10-0)^2 + (0.05-0)^2 = 0.0225 + 0.01 + 0.0025 = 0.035$

ECE bin [0.7, 0.9]: contains 2 samples, avg confidence = 0.775, accuracy = 1.0 → gap = 0.225, weight = 2/3.

---

## 8. Results Summary (from Notebook Outputs)

| Model       | Test Accuracy | Cohen's κ | Macro F1 | Brier Score | ECE (15 bins) | Train Time (s) |
|-------------|--------------|-----------|----------|-------------|---------------|----------------|
| **GFNet**   | **99.74%**   | 0.9965    | 0.9971   | 0.00458     | 0.00970       | 787            |
| ViT-UNet    | 99.49%       | 0.9931    | 0.9935   | 0.01117     | 0.03719       | 1,474          |
| AlexNet-CNN | 99.35%       | 0.9912    | 0.9907   | 0.00971     | 0.00286       | 747            |

All three models achieve over 99% accuracy on the 7-class multispectral task after 100 epochs. GFNet leads on most metrics while being significantly faster than ViT-UNet. AlexNet achieves the best ECE, suggesting its confidence estimates are the best-calibrated despite slightly lower accuracy. These models now serve as baselines for downstream conformal prediction or uncertainty quantification experiments.

---

## 9. References

[1] Krizhevsky, A., Sutskever, I., and Hinton, G. E. "ImageNet Classification with Deep Convolutional Neural Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 2012. [Link](https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

[2] Rao, Y., Zhao, W., Zhu, Z., Lu, J., and Zhou, J. "Global Filter Networks for Image Classification." *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. [arXiv:2107.00645](https://arxiv.org/abs/2107.00645)

[3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." *International Conference on Learning Representations (ICLR)*, 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

[4] Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

[5] Niculescu-Mizil, A., and Caruana, R. "Predicting Good Probabilities with Supervised Learning." *International Conference on Machine Learning (ICML)*, 2005.
