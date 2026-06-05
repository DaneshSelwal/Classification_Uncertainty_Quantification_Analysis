# Deep Ensemble Training & CreDE Uncertainty Estimation: Theory & Implementation Summary

> **One-line description:** Train M independently-seeded neural networks on multispectral patch data, then use their output spread to compute credal-set bounds that decompose predictive uncertainty into aleatoric and epistemic components.

---

## 1. Overview & Intuition

### 1.1 The Two-Notebook Pipeline

These two notebooks form a complete end-to-end pipeline for deep learning–based classification of multispectral remote-sensing imagery with principled uncertainty quantification. The first notebook (`Model_training_ensembles.ipynb`) builds and trains the ensemble; the second (`Model_uncertainty_CreDE.ipynb`) applies Credal Deep Ensemble (CreDE) inference to the saved models to produce spatially resolved uncertainty maps.

### 1.2 Why Ensembles?

A single neural network produces a point estimate: one softmax probability vector per input. That output gives no indication of whether the model is confident because the input is easy, or merely confident-seeming because it has never seen anything like this class before. Deep ensembles address this by training multiple networks independently (different random seeds → different loss-landscape minima). When the members agree, the prediction is reliable; when they diverge, epistemic uncertainty is high.

### 1.3 Why Credal Sets?

Standard deep ensembles typically reduce the member predictions to a single mean probability. The CreDE approach instead treats the ensemble as a *credal set* — a convex set of probability distributions bounded below by the member-wise minimum and above by the member-wise maximum. This interval representation directly encodes how much the models disagree about each class probability. The width of the interval is a natural, interpretable measure of epistemic uncertainty, and the entropy of the lower-bound distribution separately captures aleatoric (data-level) uncertainty.

### 1.4 Applied Context

The task is patch-based classification of a 330×307 multispectral scene with 6 spectral bands and up to 10 land-cover classes. Three architectures are used: a legacy AlexNet-style CNN, a Global Filter Network (GFNet) that learns in the frequency domain, and a Vision Transformer with U-Net skip connections (ViT-UNet). Five independently seeded copies of each architecture are trained (M=5), producing an ensemble whose outputs feed into the CreDE inference engine.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathbf{x} \in \mathbb{R}^{P \times P \times B}$ denote a multispectral patch of spatial size $P \times P$ with $B=6$ spectral bands, and let $y \in \{0, \ldots, C-1\}$ be the integer class label ($C$ classes). Each ensemble member $f_m$ is a neural network that outputs a softmax probability vector:

$$f_m(\mathbf{x}) = \hat{\mathbf{p}}_m \in \Delta^{C-1}$$

where $\Delta^{C-1}$ is the probability simplex. Training $M=5$ members with different seeds $\{42+1, 42+2, \ldots, 42+5\}$ yields predictions $\{\hat{\mathbf{p}}_1, \ldots, \hat{\mathbf{p}}_M\}$ for each input.

### 2.2 Credal Set Construction

The key idea is to enclose all member predictions within a probability interval per class. For class $c$, define:

$$p_{\min}^{(c)} = \min_{m=1}^{M} \hat{p}_m^{(c)}, \qquad p_{\max}^{(c)} = \max_{m=1}^{M} \hat{p}_m^{(c)}$$

**Where:**
- $\hat{p}_m^{(c)}$ — softmax probability assigned to class $c$ by ensemble member $m$
- $p_{\min}^{(c)}$ — tightest lower bound on class-$c$ probability across all members
- $p_{\max}^{(c)}$ — tightest upper bound on class-$c$ probability across all members

**What this means:** The credal set is the convex hull of all member distributions. The interval $[p_{\min}^{(c)},\, p_{\max}^{(c)}]$ captures how much the ensemble "knows it doesn't know" about class $c$.

### 2.3 Normalised Lower Credal Probabilities ($p^*$)

The lower-bound vector $\mathbf{p}_{\min}$ may not sum to one, so it is normalised to form the *credal representative*:

$$p^{*(c)} = \frac{p_{\min}^{(c)}}{\sum_{c'} p_{\min}^{(c')} + \varepsilon}$$

**Where:**
- $\varepsilon = 10^{-12}$ — numerical stability constant
- $p^{*(c)}$ — normalised lower-bound probability for class $c$

**What this means:** $\mathbf{p}^*$ is the most conservative (least confident) proper probability distribution consistent with the ensemble. It represents the cautious inner-bound prediction. The $\arg\max$ of $\mathbf{p}^*$ is used as the final predicted class label.

### 2.4 Uncertainty Decomposition

Three scalar uncertainty measures are derived from the credal bounds:

**Aleatoric Uncertainty (AU)** — entropy of the normalised lower credal distribution:

$$\text{AU}(\mathbf{x}) = -\sum_{c=1}^{C} p^{*(c)} \log p^{*(c)}$$

High AU means the *lowest* plausible distribution is already uncertain — this is irreducible data noise.

**Epistemic Uncertainty (EU)** — mean spread across the credal interval:

$$\text{EU}(\mathbf{x}) = \frac{1}{C} \sum_{c=1}^{C} \left( p_{\max}^{(c)} - p_{\min}^{(c)} \right)$$

High EU means the ensemble members strongly disagree — this reflects reducible model uncertainty (e.g., insufficient training data in this region).

**Total Uncertainty (TU)**:

$$\text{TU}(\mathbf{x}) = \text{AU}(\mathbf{x}) + \text{EU}(\mathbf{x})$$

**Where:**
- $\Delta p^{(c)} = p_{\max}^{(c)} - p_{\min}^{(c)}$ — the credal width for class $c$

**What this means:** AU and EU are complementary. A pixel may be uncertain because the scene genuinely sits on a class boundary (AU) or because the models haven't seen similar spectral signatures in training (EU). Separating these allows different corrective actions: gather more data (EU) or accept irreducible ambiguity (AU).

### 2.5 The Global Filter Network Block (Key Architecture)

The GFNet replaces self-attention with a learnable complex filter in the 2D frequency domain. For a token sequence $\mathbf{x} \in \mathbb{R}^{L \times D}$ reshaped to a spatial grid $\mathbb{R}^{S \times S \times D}$ (where $L = S^2$):

$$\mathbf{y} = \text{Re}\left[\mathcal{F}^{-1}\!\left(\mathcal{F}(\mathbf{x}) \odot \mathbf{W}\right)\right]$$

**Where:**
- $\mathcal{F}$ — 2D discrete Fourier transform (FFT2D)
- $\mathbf{W} = \mathbf{W}_r + i\mathbf{W}_i \in \mathbb{C}^{S \times S \times D}$ — learnable complex filter weights
- $\odot$ — element-wise multiplication in the frequency domain
- $\text{Re}[\cdot]$ — real part of the inverse FFT output

**What this means:** Each learnable filter weight acts as a global frequency selector. The FFT operation has $O(L \log L)$ complexity, far cheaper than $O(L^2)$ self-attention. This is how GFNet captures long-range spectral dependencies across the entire patch grid at low cost.

---

## 3. Algorithm

### 3.1 Ensemble Training (Notebook 1)

**Input:** Multispectral image $(H \times W \times B)$; label raster; architecture choice (AlexNet / GFNet / ViT-UNet); $M=5$  
**Output:** $M$ saved `.keras` model files per architecture

1. Load multispectral CSV; apply per-band min-max normalisation to $[0,1]$.
2. Extract $P \times P$ patches centred on every labeled pixel; assign zero-indexed class labels.
3. Stratified train/val/test split (75% / 20% of train / remaining).
4. For $m = 1$ to $M$:
   a. Set seeds $\{42+m\}$ for Python, NumPy, TensorFlow.
   b. Build the chosen architecture from its config dict.
   c. Compile with AdamW + cosine-decay LR schedule (AlexNet uses Adagrad + cosine schedule).
   d. Train for 100 epochs with label smoothing ($\epsilon = 0.05$), weight decay ($10^{-4}$), and gradient clip-norm.
   e. Save best-validation-accuracy weights and final `.keras` file.
5. Record accuracy, F1, Cohen's Kappa, NLL, Brier score, ECE (15-bin) per member.
6. Plot training curves, metric comparisons, and confusion matrices.
7. Run full-scene patch inference; export classification maps to PNG and Excel.

### 3.2 CreDE Inference (Notebook 2)

**Input:** $M$ saved `.keras` ensemble members; full-scene patch array $(H \cdot W \times P \times P \times B)$  
**Output:** Per-pixel class label, $\mathbf{p}^*$, AU, EU, TU maps

1. Load multispectral data; extract all $H \cdot W$ overlapping patches.
2. For $m = 1$ to $M$: load model, predict on all scene patches in batches, append $\hat{\mathbf{p}}_m$; delete model, clear session (RAM management).
3. Stack predictions: $\hat{\mathbf{P}} \in \mathbb{R}^{M \times HW \times C}$.
4. Compute $\mathbf{p}_{\min}$, $\mathbf{p}_{\max}$ by reduce-min/max over the $M$ axis.
5. Normalise $\mathbf{p}_{\min}$ to obtain $\mathbf{p}^*$.
6. Compute AU (entropy of $\mathbf{p}^*$), EU (mean credal width), TU (AU + EU).
7. Threshold each map (AU > 0.5, EU > 0.2, TU > 0.7); generate binary certain/uncertain masks.
8. Produce 3×4 spatial figure (base map, binary masks, grey-overlay maps) per architecture.
9. Export summary CSV and styled Excel report.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebooks: `Model_training_ensembles.ipynb` and `Model_uncertainty_CreDE.ipynb`

### 4.1 Per-Band Normalisation (Both Notebooks)

```python
for b in range(bands):
    band = x[:, :, b]
    band_min, band_max = np.min(band), np.max(band)
    x_norm[:, :, b] = (band - band_min) / max(band_max - band_min, 1e-8)
```

**What this does:** Applies independent min-max normalisation to each spectral band, rescaling to $[0,1]$.  
**Why:** Multispectral bands have different physical scales (e.g., near-IR vs visible). Per-band normalisation prevents any single band from dominating the loss gradient and ensures the network receives consistently scaled inputs regardless of sensor calibration differences.

### 4.2 Patch Extraction

```python
pad   = patch_size // 2
x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
for i, (r, c) in enumerate(coords):
    patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
    labels[i]  = int(y[r, c]) - 1
```

**What this does:** Pads the scene with edge-replicated values, then extracts a $9 \times 9$ spatial context window centred on every labeled pixel.  
**Why:** Patch-based classification gives the model spatial neighbourhood context — the spectral profile of nearby pixels carries discriminative information. Edge padding avoids introducing artificial zero-valued boundary artefacts.

### 4.3 GFNet Global Filter Layer

```python
x_2d      = tf.reshape(x, [batch, self.token_side, self.token_side, channels])
x_fft     = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
w_complex = tf.complex(self.w_real, self.w_imag)
x_filtered = x_fft * w_complex
x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
return tf.reshape(x_spatial, [batch, self.token_side * self.token_side, channels])
```

**What this does:** Reshapes the token sequence into a 2D grid, transforms to the frequency domain via FFT2D, multiplies element-wise by learnable complex weights, then transforms back.  
**Why:** This is the core of GFNet. The complex weights $\mathbf{W}_r + i\mathbf{W}_i$ are initialised with glorot-uniform real parts and zero imaginary parts, then trained end-to-end. Each weight position corresponds to a specific spatial frequency; the network learns which frequencies are most discriminative for the classification task. The FFT/IFFT are non-parametric transforms — all trainable parameters live in $\mathbf{W}$.

### 4.4 Ensemble Training Loop (M=5 Members)

```python
for i in range(1, M + 1):
    seed_val = 42 + i
    tf.random.set_seed(seed_val); np.random.seed(seed_val); random.seed(seed_val)
    tf.keras.backend.clear_session()
    model = builder()
    # compile, fit, checkpoint ...
```

**What this does:** Trains five fully independent copies of the chosen architecture, each with a different global seed. Each member gets its own weight initialisation and data-shuffling trajectory.  
**Why:** Independence is the key property of deep ensembles. Members that start from different random initialisations converge to different loss-landscape basins, producing functionally diverse predictions. This diversity is what makes their disagreement a meaningful signal for epistemic uncertainty. `clear_session()` between members prevents any weight state leaking from one run to the next.

### 4.5 Credal Bound Computation

```python
stacked_preds = tf.stack(all_preds, axis=0)   # (M, N, C)
p_min = tf.reduce_min(stacked_preds, axis=0)
p_max = tf.reduce_max(stacked_preds, axis=0)
delta_p = p_max - p_min

p_star = p_min / (tf.reduce_sum(p_min, axis=-1, keepdims=True) + 1e-12)
p_star = np.clip(p_star, 1e-12, 1.0)

au = -np.sum(p_star * np.log(p_star), axis=-1)
eu = np.mean(delta_p, axis=-1)
tu = au + eu
```

**What this does:** Stacks all $M$ member prediction matrices into a $(M \times N \times C)$ tensor, reduces along the member axis to obtain per-class bounds, then computes the three uncertainty measures.  
**Why:** This is the entire CreDE inference step. `reduce_min/max` over the first (member) axis gives $p_{\min}$ and $p_{\max}$ without ever needing to store all predictions in memory simultaneously — they were accumulated sequentially precisely because the full scene array $(HW \times P \times P \times B)$ is large. Each model is also deleted immediately after prediction (`del model; clear_session(); gc.collect()`) to prevent OOM errors on GPU/TPU.

### 4.6 Thresholded Uncertainty Maps

```python
au_mask = (au_map > au_thresh).astype(int)   # AU_THRESH = 0.5
eu_mask = (eu_map > eu_thresh).astype(int)   # EU_THRESH = 0.2
tu_mask = (tu_map > tu_thresh).astype(int)   # TU_THRESH = 0.7

combined_au = np.where(au_mask == 1, n_cls, pred_map)
```

**What this does:** Binarises each continuous uncertainty map using fixed thresholds, then creates an overlay where uncertain pixels are shown in grey and certain pixels retain their predicted class colour.  
**Why:** Continuous uncertainty values are hard to interpret visually across a full scene. Thresholded binary masks immediately highlight spatially coherent uncertain regions (e.g., class boundaries, shadows, mixed pixels) that need further review. The grey-overlay variant lets the analyst see both the classification and its confidence simultaneously.

---

## 5. Worked Numerical Example

**Setup:** 3-class problem, 3-member mini-ensemble ($M=3$), 2-pixel scene ($N=2$).

Suppose the three members produce the following softmax outputs for pixel A:

| Member | Class 0 | Class 1 | Class 2 |
|--------|---------|---------|---------|
| $m=1$  | 0.70    | 0.20    | 0.10    |
| $m=2$  | 0.60    | 0.30    | 0.10    |
| $m=3$  | 0.65    | 0.15    | 0.20    |

**Step 1 — Credal bounds:**

$$\mathbf{p}_{\min} = [0.60,\; 0.15,\; 0.10], \qquad \mathbf{p}_{\max} = [0.70,\; 0.30,\; 0.20]$$

$$\Delta\mathbf{p} = [0.10,\; 0.15,\; 0.10]$$

**Step 2 — Normalise lower bound:**

$$\text{sum}(\mathbf{p}_{\min}) = 0.85, \qquad \mathbf{p}^* = [0.706,\; 0.176,\; 0.118]$$

**Step 3 — Predicted class:** $\arg\max(\mathbf{p}^*) = 0$ ✓

**Step 4 — Aleatoric uncertainty (entropy of $\mathbf{p}^*$):**

$$\text{AU} = -(0.706\ln 0.706 + 0.176\ln 0.176 + 0.118\ln 0.118) \approx 0.826$$

**Step 5 — Epistemic uncertainty (mean credal width):**

$$\text{EU} = \frac{0.10 + 0.15 + 0.10}{3} \approx 0.117$$

**Step 6 — Total uncertainty:**

$$\text{TU} = 0.826 + 0.117 = 0.943$$

**Interpretation:** Pixel A is predicted as Class 0, but the ensemble disagrees measurably (EU ≈ 0.12 → above the 0.2 threshold would flag this at EU_THRESH=0.2? No — it is below). AU is moderately high (0.83) because even the most conservative distribution is not peaked. A pixel with AU=0.05 and EU=0.40 would tell a different story: the conservative distribution is very confident, but members disagree strongly — a textbook case of epistemic uncertainty that more training data could resolve.

---

## 6. References

[1] Wang, K., Cuzzolin, F., Kudukkil Manchingal, S., Shariatmadar, K., Moens, D., & Hallez, H. "Credal Deep Ensembles for Uncertainty Quantification." *NeurIPS 2024*. [https://proceedings.neurips.cc/paper_files/paper/2024/hash/911fc798523e7d4c2e9587129fcf88fc-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2024/hash/911fc798523e7d4c2e9587129fcf88fc-Abstract-Conference.html)

[2] Lakshminarayanan, B., Pritzel, A., & Blundell, C. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS 2017*. [https://arxiv.org/abs/1612.01474](https://arxiv.org/abs/1612.01474)

[3] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. "Global Filter Networks for Image Classification." *NeurIPS 2021 / IEEE T-PAMI 2023*. [https://arxiv.org/abs/2107.00645](https://arxiv.org/abs/2107.00645)

[4] Wang, K., et al. "Credal Wrapper of Model Averaging for Uncertainty Estimation in Classification." *arXiv:2405.15047*, 2024. [https://arxiv.org/abs/2405.15047](https://arxiv.org/abs/2405.15047)

[5] Wang, K., et al. "Credal Ensemble Distillation for Uncertainty Quantification." *arXiv:2511.13766*, 2025. [https://arxiv.org/abs/2511.13766](https://arxiv.org/abs/2511.13766)
