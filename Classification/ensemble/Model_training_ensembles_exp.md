# Deep Ensembles for Hyperspectral Image Classification: Theory & Implementation Summary

> **One-line description:** Train M=5 independently seeded copies of each neural architecture (AlexNet-CNN, GFNet, ViT-UNet) on 9×9 spectral patches, then aggregate their softmax outputs to obtain ensemble predictions with improved calibration and robustness.

---

## 1. Overview & Intuition

A single neural network trained with gradient descent converges to one point in a high-dimensional weight space. Because the loss landscape is non-convex and contains many local minima of similar quality, another training run from a different random initialisation will converge to a different solution. The two solutions may agree on easy, well-represented inputs but diverge on ambiguous or out-of-distribution ones. **Deep Ensembles** exploit this diversity deliberately: train M networks independently and average their predictions. Where the members agree, the ensemble is confident; where they disagree, the variance reveals genuine uncertainty.

This is especially valuable for hyperspectral remote-sensing classification, where spatial context (captured via 9×9 patches), spectral overlap between land-cover classes, and the relatively small labelled pixel count all conspire to produce overconfident single-model predictions. An ensemble of five members trained with different random seeds approximates sampling from different basins of the loss landscape, providing a low-cost but effective proxy for epistemic (model) uncertainty without modifying the underlying architecture or requiring Bayesian inference.

The notebook implements this strategy across three architecturally distinct backbones — a legacy convolutional network (AlexNet-CNN), a frequency-domain token mixer (GFNet), and a Vision Transformer with symmetric skip connections (ViT-UNet) — so that downstream uncertainty methods can later pool predictions from architecturally diverse members, further increasing diversity and coverage.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathcal{X} = \mathbb{R}^{P \times P \times B}$ be the input space of square spectral patches of side $P$ with $B$ spectral bands, and let $\mathcal{Y} = \{1, \ldots, C\}$ be a set of $C$ land-cover classes. A training dataset $\mathcal{D} = \{(x_n, y_n)\}_{n=1}^{N}$ consists of labeled patches extracted from a multispectral scene of spatial size $H \times W$.

Each ensemble member $m \in \{1, \ldots, M\}$ is a neural network $f_{\theta_m} : \mathcal{X} \to \Delta^{C-1}$ that maps a patch to a probability simplex (the softmax output), with parameters $\theta_m$ drawn from a different random initialisation seed $s_m$.

### 2.2 Ensemble Predictive Distribution

At test time, the ensemble prediction is the **uniformly weighted average** of all member softmax distributions:

$$\hat{p}(y \mid x) = \frac{1}{M} \sum_{m=1}^{M} f_{\theta_m}(x)$$

**Where:**
- $x \in \mathcal{X}$ — the test patch
- $M$ — ensemble size (M = 5 in this notebook)
- $f_{\theta_m}(x) \in \Delta^{C-1}$ — the softmax probability vector produced by member $m$
- $\hat{p}(y \mid x)$ — the ensemble's marginal predictive distribution

**What this means:** The ensemble acts as a uniformly weighted mixture model. Each member "votes" with its full probability distribution, not just its argmax label. This soft averaging propagates calibration information and allows the ensemble to express genuine multi-modal uncertainty when members disagree.

### 2.3 Epistemic Uncertainty via Predictive Variance

The variance of ensemble member predictions around their mean is a practical proxy for epistemic (model) uncertainty:

$$\text{Var}[\hat{p}] = \frac{1}{M} \sum_{m=1}^{M} \left( f_{\theta_m}(x) - \hat{p}(y \mid x) \right)^2$$

High variance across members on a test sample indicates a region of input space where the training data provided insufficient information to uniquely constrain the weights — the signature of epistemic uncertainty. Low variance with confident $\hat{p}$ indicates reliable prediction.

### 2.4 Diversity via Independent Random Initialisation

Diversity is produced by drawing each member's initial weights from a different random seed:

$$\theta_m^{(0)} \sim \mathcal{I}(s_m), \quad s_m = 42 + m, \quad m \in \{1, \ldots, 5\}$$

**Where:**
- $\mathcal{I}(s)$ — the random weight initialiser seeded with $s$ (Glorot uniform for most layers, zeros for GFNet imaginary filter weights)
- $s_m$ — per-member seed, offset from a base seed of 42

Because the loss landscape is non-convex with many minima of similar training loss, independent initialisation causes each member to descend to a different basin. The resulting parameter diversity is the source of disagreement on ambiguous inputs and is the core mechanism behind ensemble uncertainty.

### 2.5 Calibration Metrics

Beyond accuracy, the notebook tracks three calibration metrics that measure how well ensemble probabilities reflect true class frequencies.

**Brier Score** (mean squared probability error):

$$\text{BS} = \frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} \left( \hat{p}(y{=}c \mid x_n) - \mathbb{1}[y_n = c] \right)^2$$

**Expected Calibration Error (ECE)** partitions predictions into $K$ equal-width confidence bins and measures the weighted average gap between mean confidence and mean accuracy within each bin:

$$\text{ECE} = \sum_{k=1}^{K} \frac{|B_k|}{N} \left| \text{acc}(B_k) - \text{conf}(B_k) \right|$$

**Where:**
- $B_k$ — the set of samples whose maximum predicted probability falls in the $k$-th bin
- $\text{acc}(B_k)$ — fraction of correctly classified samples in $B_k$
- $\text{conf}(B_k)$ — mean maximum probability in $B_k$
- $K = 15$ bins used in the notebook

**What this means:** ECE = 0 implies perfect calibration (a model that says "80% confident" is right 80% of the time). Lower Brier score and ECE indicate a more trustworthy probability estimator.

**Negative Log-Likelihood (NLL)**:

$$\text{NLL} = -\frac{1}{N} \sum_{n=1}^{N} \log \hat{p}(y_n \mid x_n)$$

NLL is a strictly proper scoring rule — it simultaneously rewards accuracy and calibration, penalising overconfident wrong predictions more than well-calibrated wrong ones.

### 2.6 Global Filter Operation (GFNet Block)

The GFNet member replaces multi-head self-attention with a learnable frequency-domain filter. For a token sequence reshaped into a 2D spatial grid $\mathbf{x} \in \mathbb{R}^{S \times S \times D}$:

$$\mathbf{X} = \mathcal{F}_{2D}(\mathbf{x})$$
$$\tilde{\mathbf{X}} = \mathbf{X} \odot \mathbf{K}, \quad \mathbf{K} = \mathbf{K}_r + i\,\mathbf{K}_i$$
$$\hat{\mathbf{x}} = \mathcal{F}^{-1}_{2D}(\tilde{\mathbf{X}})$$

**Where:**
- $\mathcal{F}_{2D}$ — 2D discrete Fourier transform (FFT2)
- $\mathbf{K}_r, \mathbf{K}_i \in \mathbb{R}^{S \times S \times D}$ — learnable real and imaginary filter weights
- $\odot$ — element-wise (Hadamard) multiplication in the frequency domain
- $\mathcal{F}^{-1}_{2D}$ — 2D inverse FFT

**What this means:** Rather than computing attention scores between all pairs of tokens ($O(N^2)$ cost), GFNet mixes information globally in the frequency domain at $O(N \log N)$ cost. Each learnable complex weight $\mathbf{K}[u,v,d]$ acts as a bandpass/bandstop gate for frequency $(u,v)$ in channel $d$, allowing the network to selectively amplify or suppress spatial frequency patterns (e.g., edges, textures, spectral gradients).

### 2.7 ViT-UNet Transformer Block with Skip Connections

Each Vision Transformer block applies pre-layer-normalised multi-head self-attention (MHSA) followed by a feed-forward network (FFN), each with a residual connection:

$$\mathbf{x}' = \mathbf{x} + \text{MHSA}(\text{LN}(\mathbf{x}))$$
$$\mathbf{x}'' = \mathbf{x}' + \text{FFN}(\text{LN}(\mathbf{x}'))$$

The notebook's ViT-UNet variant adds **symmetric skip connections** between encoder-side blocks $i \leq L/2$ and corresponding decoder-side blocks $i > L/2$:

$$\mathbf{x}_{i} \leftarrow \mathbf{x}_{i} + \mathbf{x}_{L - i - 1}, \quad i > \lfloor L/2 \rfloor$$

**Where:**
- $L$ — total number of transformer blocks (12 in primary config, 8 in fallback)
- $\mathbf{x}_{L - i - 1}$ — the saved output of the symmetric encoder block

**What this means:** The skip connections form a U-Net-like bottleneck, allowing shallow features (local spectral patterns) to bypass the deep blocks and be reused in classification, which aids in preserving fine-grained spatial detail through the attention stack.

---

## 3. Algorithm

**Input:** Labelled multispectral patch dataset $\mathcal{D}$, architecture family $\mathcal{A} \in \{\text{AlexNet, GFNet, ViT-UNet}\}$, ensemble size $M=5$  
**Output:** $M$ trained model checkpoints per architecture, per-member calibration metrics

1. Load and normalise the multispectral image per band with min–max scaling.
2. Extract 9×9 patches centred on all labelled pixels; label-shift classes to 0-indexed.
3. Split patches into train (75%) / val (20% of train) / test (25%) with stratified sampling.
4. For each member $m = 1, \ldots, M$:
   a. Set global random seeds to $42 + m$ across Python, NumPy, and TensorFlow.
   b. Instantiate the model from the architecture config; fall back to a smaller config on out-of-memory error.
   c. Compile with the architecture-specific optimiser and loss (see §4.2–4.3 below).
   d. Register a `ModelCheckpoint` callback to save the best weights during training.
   e. Train for up to 100 epochs with early stopping via checkpoint selection.
   f. Save final `.keras` model locally, then copy to Google Drive.
   g. Compute test predictions; record accuracy, macro-F1, Cohen's κ, NLL, Brier score, and ECE (15-bin).
5. Assemble a summary DataFrame of all M per-member metric rows, sorted by test accuracy.
6. Run full-scene patch-by-patch inference for each saved model; export classification maps.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_training_ensembles.md`

### 4.1 Per-Band Min–Max Normalisation

```python
for b in range(bands):
    band     = x[:, :, b]
    band_min = np.min(band)
    band_max = np.max(band)
    denom    = max(band_max - band_min, 1e-8)
    x_norm[:, :, b] = (band - band_min) / denom
```

**What this does:** Independently scales each of the 6 spectral bands to [0, 1].  
**Why:** Prevents high-dynamic-range bands (e.g., SWIR vs. visible) from dominating activations in early layers. The `1e-8` floor guards against constant bands.

### 4.2 Patch Extraction

```python
x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
for i, (r, c) in enumerate(coords):
    patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
    labels[i]  = int(y[r, c]) - 1
```

**What this does:** Extracts a 9×9 spatial neighbourhood around every labelled pixel, using edge-padding to handle border pixels without information loss.  
**Why:** The local patch provides spatial context (texture, shape boundaries) beyond the single-pixel spectrum, which all three architectures exploit differently (CNN via convolutions, GFNet via frequency filtering, ViT via self-attention across 3×3 sub-patches).

### 4.3 Ensemble Training Loop with Per-Member Seeds

```python
for i in range(1, M + 1):
    seed_val = 42 + i
    tf.random.set_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    tf.keras.backend.clear_session()
    model = builder()
```

**What this does:** Before building each member, all random number generators are re-seeded with a distinct value and the Keras session is cleared to reset all layer weight initialisations.  
**Why:** This is the core mechanism of deep ensemble diversity. Without distinct seeds, members trained on the same data with the same default initialisation would converge to very similar solutions and provide little variance signal.

### 4.4 GFNet Global Filter Layer

```python
x_fft      = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
w_complex  = tf.complex(self.w_real, self.w_imag)
x_filtered = x_fft * w_complex
x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
```

**What this does:** Applies a learned complex-valued bandpass filter in the frequency domain: FFT → element-wise multiply → IFFT.  
**Why:** This replaces the $O(N^2)$ self-attention of a standard transformer with an $O(N \log N)$ frequency-domain operation, enabling efficient global token mixing over the 3×3 sub-patch grid (9 tokens) within each 9×9 input patch.

### 4.5 ViT-UNet Skip Connection Assembly

```python
block_list = []
for i in range(transformer_layers):
    x = transformer_block(...)
    if i <= transformer_layers // 2:
        block_list.append(x)
    else:
        x = layers.Add(...)([x, block_list[transformer_layers - i - 1]])
```

**What this does:** Saves the output of the first half of the transformer stack into `block_list`; during the second half, each block's output is added to its mirror block's output.  
**Why:** Mirrors the U-Net encoder–decoder pattern in the token dimension. The encoder stores increasingly abstract spectral-spatial representations, and the skip connections inject them back at matching resolution levels during decoding, mitigating representation collapse at the CLS token.

### 4.6 Optimiser Factory (AdamW + Cosine Decay)

```python
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,   # 3e-4
    decay_steps=decay_steps,
    alpha=TRAIN_CFG['cosine_alpha'],       # 0.05
)
return keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=TRAIN_CFG['weight_decay'],  # 1e-4
    clipnorm=TRAIN_CFG['clipnorm'],           # 1.0
)
```

**What this does:** Builds an AdamW optimiser with a cosine annealing schedule that decays from `3e-4` to a floor of `3e-4 × 0.05 = 1.5e-5` over the full training run.  
**Why:** Cosine decay smoothly reduces the learning rate without sharp drops, helping members settle into sharper minima at the end of training. Weight decay (L2 on weights, not gradients) acts as regularisation, reducing overfit on the limited labelled pixels.

### 4.7 Label Smoothing (GFNet / ViT Only)

```python
loss=keras.losses.CategoricalCrossentropy(label_smoothing=TRAIN_CFG['label_smoothing'])
# label_smoothing = 0.05
```

**What this does:** Replaces the one-hot target $[0,\ldots,1,\ldots,0]$ with a smoothed version $[\frac{\epsilon}{C}, \ldots, 1 - \epsilon + \frac{\epsilon}{C}, \ldots, \frac{\epsilon}{C}]$ where $\epsilon = 0.05$.  
**Why:** Prevents the network from driving softmax logits to $\pm\infty$ to minimise cross-entropy loss exactly. The result is better-calibrated output probabilities — a prerequisite for ensemble calibration metrics to be meaningful.

### 4.8 ECE Computation

```python
bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
for i in range(n_bins):
    in_bin = (confidences >= lo) & (confidences < hi)
    acc_bin  = np.mean(correct[in_bin])
    conf_bin = np.mean(confidences[in_bin])
    ece     += np.abs(acc_bin - conf_bin) * prop
```

**What this does:** Groups predictions by their maximum softmax confidence into 15 equal-width bins, then computes the sample-weighted average absolute gap between accuracy and confidence in each bin.  
**Why:** ECE directly measures calibration error and is the primary signal for whether ensemble probability outputs can be trusted as uncertainty estimates in downstream conformal prediction or decision-making pipelines.

---

## 5. Worked Numerical Example

**Setup:** Consider a toy problem with C = 3 classes and M = 3 ensemble members applied to a single test patch.

**Member predictions (softmax outputs):**

| Member | Class 1 | Class 2 | Class 3 |
|--------|---------|---------|---------|
| $f_{\theta_1}$ | 0.70 | 0.20 | 0.10 |
| $f_{\theta_2}$ | 0.60 | 0.30 | 0.10 |
| $f_{\theta_3}$ | 0.50 | 0.40 | 0.10 |

**Step 1 — Ensemble prediction (average):**

$$\hat{p}(y \mid x) = \frac{1}{3}([0.70, 0.20, 0.10] + [0.60, 0.30, 0.10] + [0.50, 0.40, 0.10])$$
$$= [0.60, 0.30, 0.10]$$

The predicted class is argmax = Class 1, with confidence 0.60.

**Step 2 — Per-class variance (epistemic uncertainty proxy):**

For Class 1: $\frac{1}{3}((0.70 - 0.60)^2 + (0.60 - 0.60)^2 + (0.50 - 0.60)^2) = \frac{0.01 + 0 + 0.01}{3} = 0.0067$

For Class 2: $\frac{1}{3}((0.20 - 0.30)^2 + (0.30 - 0.30)^2 + (0.40 - 0.30)^2) = 0.0067$

For Class 3: 0.0 (all members agree on 0.10)

**Step 3 — Brier Score** (assuming true class = Class 1, one-hot = [1, 0, 0]):

$$\text{BS} = (0.60 - 1)^2 + (0.30 - 0)^2 + (0.10 - 0)^2 = 0.16 + 0.09 + 0.01 = 0.26$$

**Step 4 — Contribution to ECE:**

This sample has confidence 0.60, so it falls in the bin [0.533, 0.600]. If the accuracy in that bin across all test samples is 0.55 (members are slightly overconfident), the bin contributes $|0.55 - 0.60| \times \text{proportion of samples in bin} = 0.05 \times 0.12 = 0.006$ to the ECE.

**Result:** The ensemble assigns Class 1 with moderate confidence (0.60), shows non-trivial variance on Class 2 (0.0067), and perfectly agrees on Class 3 (0.0 variance). A downstream conformal predictor might include Class 1 and Class 2 in the prediction set to guarantee coverage, reflecting the ensemble's expressed ambiguity between the two.

---

## 6. References

[1] Lakshminarayanan, B., Pritzel, A., & Blundell, C. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *Advances in Neural Information Processing Systems (NeurIPS)*, 2017. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474)

[2] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. "Global Filter Networks for Image Classification." *Advances in Neural Information Processing Systems (NeurIPS)*, 2021. [arXiv:2107.00645](https://arxiv.org/abs/2107.00645)

[3] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *International Conference on Learning Representations (ICLR)*, 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

[4] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. "On Calibration of Modern Neural Networks." *International Conference on Machine Learning (ICML)*, 2017. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

[5] Ronneberger, O., Fischer, P., & Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
