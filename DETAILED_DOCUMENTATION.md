# 📘 Detailed Technical Documentation

## Classification Uncertainty Quantification Analysis

> A comprehensive, modular deep-learning framework for quantifying predictive uncertainty in pixel-level multispectral remote-sensing classification, implementing twelve state-of-the-art uncertainty quantification paradigms across three neural network architectures.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure — File-by-File](#2-repository-structure--file-by-file)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Architectures](#4-model-architectures)
   - [4.1 AlexNet-style CNN](#41-alexnet-style-cnn)
   - [4.2 Global Filter Network (GFNet)](#42-global-filter-network-gfnet)
   - [4.3 Vision Transformer with U-Net Skips (ViT-UNet)](#43-vision-transformer-with-u-net-skips-vit-unet)
5. [Method 1 — Baseline Uncertainty (MC Dropout + Conformal Prediction)](#5-method-1--baseline-uncertainty)
6. [Method 2 — CREDIT (Credal Ensemble Distillation)](#6-method-2--credit-credal-ensemble-distillation)
7. [Method 3 — DAPM (Deep Adaptive Predictive Modeling)](#7-method-3--dapm-deep-adaptive-predictive-modeling)
8. [Method 4 — Deep Ensembles & CreDE (Credal Deep Ensembles)](#8-method-4--deep-ensembles--crede)
9. [Method 5 — MultiCP (Multi-Head Conformal Prediction)](#9-method-5--multicp-multi-head-conformal-prediction)
10. [Method 6 — SACP (Self-Adaptive Conformal Prediction)](#10-method-6--sacp-self-adaptive-conformal-prediction)
11. [Method 7 — SCMCP (Spatial Multi-Head Conformal Prediction)](#11-method-7--scmcp-spatial-multi-head-conformal-prediction)
12. [Method 8 — Focal Loss & CB Focal Loss](#12-method-8--focal-loss--cb-focal-loss)
13. [Method 9 — EDL & EDL_v2 (Evidential Deep Learning)](#13-method-9--edl--edl_v2)
14. [Method 10 — CDL (Credal Deep Learning)](#14-method-10--cdl-credal-deep-learning)
15. [Method 11 — MambaHSI](#15-method-11--mambahsi)
16. [Method 12 — DOFA (Dynamic Wavelength Tokenization)](#16-method-12--dofa-dynamic-wavelength-tokenization)
17. [Comparative Summary of All Methods](#17-comparative-summary-of-all-methods)
18. [Evaluation Metrics & Calibration](#13-evaluation-metrics--calibration)
19. [Master Hyperparameter Reference](#14-master-hyperparameter-reference)
20. [Examples Directory](#15-examples-directory)
21. [References](#16-references)

---

## 1. Project Overview

This framework addresses the critical need for **reliability and trustworthiness** in machine learning models deployed for Earth Observation and spatial analysis. Standard classifiers produce point predictions — a single class label — but offer no measure of *how certain* that prediction is. In high-stakes remote sensing applications (land-use planning, disaster response, environmental monitoring), knowing *where the model is uncertain* is just as important as the prediction itself.

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Out-of-Distribution Detection** | Identify pixels where the model is likely to fail |
| **Calibrated Probabilistic Outputs** | Ensure that a 90% confidence score actually corresponds to ~90% accuracy |
| **Valid Prediction Sets** | Conformal Prediction produces sets of classes that contain the true label with a user-specified probability (e.g., 95%) |
| **Spatial Uncertainty Maps** | Visualize pixel-level uncertainty across entire remote sensing scenes |
| **Aleatoric vs. Epistemic Decomposition** | Separate irreducible data noise from reducible model ignorance |

### Technology Stack

- **Framework:** TensorFlow 2.10+ / Keras
- **Language:** Python 3.9+
- **Environment:** Designed for Google Colab (GPU), adaptable to local environments
- **Reproducibility:** Seeded RNGs (seed=42), deterministic data splits, checkpointed models

---

## 2. Repository Structure — File-by-File

```
Classification_Uncertainty_Quantification_Analysis/
│
├── README.md                          # High-level project overview and quick-start guide
├── DETAILED_DOCUMENTATION.md          # This file — comprehensive technical reference
├── LICENSE                            # Open-source license (Apache 2.0)
├── .gitignore                         # Git exclusion rules
│
├── Classification/                    # 📂 Main source code directory
│   │
│   ├── data/                          # 📊 Dataset directory
│   │   ├── data.csv                   # Band-stacked multispectral features (330×307×6 bands)
│   │   └── ref.csv                    # Ground-truth label map (330×307, 7 classes, 1-indexed)
│   │
│   ├── baseline/                      # 📂 Method 1: Baseline Uncertainty
│   │   ├── Model_training.ipynb       # Training notebook — AlexNet, GFNet, ViT-UNet
│   │   ├── Model_training.md          # Detailed summary of training pipeline
│   │   ├── Model_uncertainty_comparison.ipynb  # 5 Conformal Prediction methods
│   │   ├── Model_uncertainty_comparison.md     # Detailed summary of CP methods
│   │   └── results/                   # Output directory
│   │       ├── classification_summary.csv        # Per-model accuracy/kappa/F1
│   │       ├── *_classification_report.json      # Per-class precision/recall/F1
│   │       ├── training_plots/                   # Loss/accuracy curves
│   │       ├── scene_visualizations/             # Full-scene prediction maps
│   │       └── uncertainty_results/              # CP prediction set maps & stats
│   │
│   ├── credit/                        # 📂 Method 2: CREDIT
│   │   ├── Model_training_credit.ipynb           # Knowledge distillation from ensemble to dual-head student
│   │   ├── Model_training_credit.md              # Detailed summary of CREDIT pipeline
│   │   └── results/                              # AU/EU/TU maps and summary tables
│   │
│   ├── dapm/                          # 📂 Method 3: DAPM
│   │   ├── Model_training_dapm_full.ipynb        # VAE + Domain-Adversarial + Diffusion training
│   │   ├── Model_training_dapm_full.md           # Detailed summary of DAPM training
│   │   ├── Model_uncertainty_dapm_full.ipynb     # Welch t-test uncertainty quantification
│   │   ├── Model_uncertainty_dapm_full.md        # Detailed summary of DAPM UQ
│   │   └── results/                              # P-value maps, uncertainty masks
│   │
│   ├── ensemble/                      # 📂 Method 4: Deep Ensembles + CreDE
│   │   ├── Model_training_ensembles.ipynb        # M=5 ensemble training per architecture
│   │   ├── Model_training_ensembles.md           # Detailed summary of ensemble training
│   │   ├── Model_uncertainty_CreDE.ipynb          # Credal Deep Ensemble uncertainty
│   │   ├── Model_uncertainty_CreDE.md             # Detailed summary of CreDE UQ
│   │   └── results/                              # AU/EU/TU maps, credal set analysis
│   │
│   ├── multicp/                       # 📂 Method 5: MultiCP
│   │   ├── Model_training_multihead.ipynb        # Multi-head (K=7) model training
│   │   ├── Model_training_multihead.md           # Detailed summary of multi-head training
│   │   ├── Model_uncertainty_multicp.ipynb       # Multi-head conformal prediction
│   │   ├── Model_uncertainty_multicp.md          # Detailed summary of MultiCP UQ
│   │   └── results/                              # Prediction sets, head sweep analysis
│   │
│   ├── sacp/                          # 📂 Method 6: SACP
│   │   ├── Model_sacp_comparison.ipynb           # Spatial-aware conformal prediction
│   │   ├── Model_sacp_comparison.md              # Detailed summary of SACP
│   │   └── results/                              # Per-window coverage reports
│   │
│   └── multicp_sacp/                  # 📂 Method 7: SCMCP (Spatial MultiCP)
│       ├── Model_uncertainty_multicp_sacp.ipynb   # Combined spatial + multi-head CP
│       ├── Model_uncertainty_multicp_sacp.md      # Detailed summary of SCMCP
│       └── results/                              # Spatial uncertainty maps
│
└── examples/                          # 📁 Pre-configured dataset example suites
    ├── indian_pines_uncertainty_quantification/   # Classic hyperspectral benchmark
    ├── pavia_uncertainty_quantification/          # Pavia Centre dataset
    ├── pavia_university_uncertainty_quantification/  # Pavia University dataset
    ├── 6_band_uncertainty_quantification/         # 6-band multispectral
    ├── 372_band_uncertainty_quantification/       # Full hyperspectral (372 bands)
    ├── dias_uncertainty_quantification/           # DIAS satellite data
    ├── hisar_3mts_uncertainty_quantification/     # Hisar 3-month temporal stack
    ├── hissar_25_uncertainty_quantification/      # Hisar 25-band
    ├── multispectral_uncertainty_quantification/  # Generic multispectral template
    └── planet_data_hisar_uncertainty_quantification/  # PlanetScope Hisar data
```

---

## 3. Data Pipeline

### 3.1 Input Data Format

| Property | Value |
|----------|-------|
| **Scene dimensions** | H=330 rows × W=307 columns |
| **Spectral bands** | B=6 (multispectral) |
| **Number of classes** | K=7 (land-cover categories, 1-indexed in raw data) |
| **Feature file** | `data.csv` — flattened band-stacked pixel values |
| **Label file** | `ref.csv` — per-pixel class labels (0 = background/unlabelled) |

### 3.2 Preprocessing Steps

1. **Load:** Read CSV files → reshape to tensors `X ∈ ℝ^{330×307×6}` and `Y ∈ ℤ^{330×307}`
2. **Per-band min-max normalisation:**

$$x^{norm}_b = \frac{x_b - \min(x_b)}{\max(x_b) - \min(x_b) + \epsilon}, \quad \epsilon = 10^{-8}$$

3. **Patch extraction:** For every labelled pixel at position (r, c), extract a P×P×B spatial neighbourhood patch (P=9) centred on that pixel. Edge pixels are handled with replicate padding (4 pixels each side).
4. **Label adjustment:** Shift from 1-indexed to 0-indexed (`y' = y - 1`).
5. **Stratified splitting:**
   - **AlexNet:** 75% train / 25% test (legacy seed=10)
   - **GFNet & ViT-UNet:** 75% train (with 20% internal validation) / 25% test (seed=42)

### 3.3 Dense Scene Inference

For full-scene spatial maps, inference is performed row-by-row:
- Sliding window across all H×W=101,370 pixels
- Mini-batch of W=307 patches per row, `batch_size=256`
- Output: Full `(H × W)` predicted label/probability map

---

## 4. Model Architectures

All three architectures take the same input — a `9×9×6` multispectral patch — and output a softmax probability vector over K classes.

### 4.1 AlexNet-style CNN

```
Input: 9×9×6
  │
  ├── Conv2D(96, 3×3, same, ReLU)
  ├── Conv2D(256, 3×3, same, ReLU)
  ├── Conv2D(384, 3×3, same, ReLU)
  ├── Conv2D(384, 3×3, same, ReLU)
  ├── Conv2D(256, 3×3, same, ReLU)
  │
  ├── MaxPooling2D(2×2, stride 2)
  ├── Flatten
  │
  ├── Dense(4096, ReLU) → Dropout(0.25)
  ├── Dense(1024, ReLU) → Dropout(0.25)
  ├── Dense(256, ReLU)  → Dropout(0.25)
  ├── Dense(32, ReLU)   → Dropout(0.25)
  │
  └── Dense(K, softmax) → output p̂ ∈ [0,1]^K
```

**Key properties:**
- Purely local receptive fields within the 9×9 patch
- No explicit long-range spatial mixing
- Dropout layers provide stochasticity for MC Dropout uncertainty
- **Optimizer:** Adagrad with cosine cycling LR ∈ [0.005, 0.02]

---

### 4.2 Global Filter Network (GFNet)

```
Input: 9×9×6
  │
  ├── PatchExtractor(3×3)  → 9 tokens of dim 54
  ├── PatchPositionEncoder → Linear projection to d=512 + positional embeddings
  │
  ├── × 5 Global Filter Blocks:
  │     ├── LayerNorm
  │     ├── Reshape to 3×3 grid
  │     ├── 2D FFT
  │     ├── Hadamard product with learnable complex filter W_ℓ ∈ ℂ^{3×3×d}
  │     ├── Inverse 2D FFT → take real part
  │     ├── Reshape back to token sequence
  │     ├── Residual connection
  │     ├── LayerNorm
  │     └── 2-layer MLP + residual
  │
  ├── GlobalAveragePooling
  └── Dense(K, softmax) → output p̂ ∈ [0,1]^K
```

**Core mathematical operation:**

$$\tilde{\mathbf{X}}_\ell = \mathcal{F}^{-1}_{2D}\left[\mathcal{F}_{2D}[\mathbf{X}_\ell] \odot \mathbf{W}_\ell\right]$$

**Key properties:**
- Multiplication in frequency domain = circular convolution with a globally-supported learned kernel
- Every output token depends on every input token → **global receptive field** at O(N log N) cost
- Much cheaper than self-attention (O(N²)) for the same global mixing
- **Fallback config (OOM):** hidden_dim reduced from 512 → 384
- **Optimizer:** AdamW (weight decay 10⁻⁴), cosine decay LR, gradient clip norm 1.0

---

### 4.3 Vision Transformer with U-Net Skips (ViT-UNet)

```
Input: 9×9×6
  │
  ├── PatchExtractor(3×3)      → 9 tokens of dim 54
  ├── PatchEncoderWithCLS       → Linear projection to d=256, + [CLS] token, + positional embeddings
  │                               Sequence length = 10 (9 patches + 1 CLS)
  │
  ├── × 12 Transformer Blocks (Pre-LayerNorm):
  │     ├── LayerNorm → Multi-Head Self-Attention → Residual
  │     └── LayerNorm → GELU MLP → Residual
  │
  │   U-Net skip connections:
  │     Blocks 1–6 = "Encoder" (outputs cached)
  │     Blocks 7–12 = "Decoder" (additive skip from mirror block)
  │     Z_{12} += Z_1, Z_{11} += Z_2, ..., Z_7 += Z_6
  │
  ├── Extract [CLS] token
  ├── Dense(512→256→128→64, GELU) with 4-layer MLP head
  └── Dense(K, softmax) → output p̂ ∈ [0,1]^K
```

**Self-Attention:**

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

**Key properties:**
- Combines global context (self-attention in later layers) with local detail (skip connections from early layers)
- [CLS] token aggregates sequence-level information for classification
- **Fallback config (OOM):** projection_dim 256→192, transformer_layers 12→8
- **Optimizer:** AdamW (weight decay 10⁻⁴), cosine decay LR, gradient clip norm 1.0

---

### Custom Keras Layers (Shared Across Architectures)

| Layer | Purpose | Used By |
|-------|---------|---------|
| `PatchExtractor` | `tf.image.extract_patches` for 3×3 non-overlapping sub-patches | GFNet, ViT-UNet |
| `PatchPositionEncoder` | Dense projection + learned positional embedding table | GFNet |
| `PatchEncoderWithCLS` | Projection + trainable [CLS] token + positional embeddings | ViT-UNet |
| `GlobalFilterLayer` | FFT → Hadamard product → inverse FFT (frequency-domain filtering) | GFNet |

All custom layers use `@register_keras_serializable` for checkpoint save/load compatibility.

---

## 5. Method 1 — Baseline Uncertainty

**Directory:** `Classification/baseline/`  
**Files:** `Model_training.ipynb/.md`, `Model_uncertainty_comparison.ipynb/.md`

### 5.1 Training Phase

Standard single-model training for each of the three architectures (AlexNet, GFNet, ViT-UNet). This establishes the **performance floor** — what the models achieve without any advanced uncertainty method.

| Config | AlexNet | GFNet & ViT-UNet |
|--------|---------|-------------------|
| **Loss** | Sparse categorical CE | Label-smoothed categorical CE (ε_s=0.05) |
| **Optimizer** | Adagrad | AdamW (weight decay 10⁻⁴, grad clip 1.0) |
| **LR Schedule** | Cosine cycling [0.005, 0.02] | Cosine decay to 0.05 × LR_init |
| **Epochs** | 100 | 100 |
| **Batch Size** | 128 | 128 |

### 5.2 Uncertainty Phase — Five Conformal Prediction Methods

All five methods share a common setup:

- **Data split for CP:** Train (75%), Calibration (12.5%, n≈2155), Evaluation (12.5%)
- **Target coverage:** 1 − α = 0.95
- **Core nonconformity score:** s(x, y) = 1 − π̂_y(x)
- **Probability sanitisation:** NaN/Inf → 0, clip to [0,1], row-normalise to simplex

#### Method A: Split Conformal Prediction (SplitCP)

Single global threshold from all calibration scores:

$$\hat{q} = \text{Quantile}\left(\{s_i\},\; \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$

$$\widehat{C}(x) = \{y : \hat{\pi}_y(x) \geq 1 - \hat{q}\}$$

**Guarantee:** Marginal coverage ≥ 1−α (distribution-free, requires only exchangeability).

#### Method B: Class-Conditional CP (CcCP)

Per-class threshold from class-specific calibration scores:

$$\hat{q}_c = \text{Quantile}\left(\{s_i : y_i = c\},\; \frac{\lceil(n_c+1)(1-\alpha)\rceil}{n_c}\right)$$

$$\widehat{C}^{CcCP}(x) = \{y : \hat{\pi}_y(x) \geq 1 - \hat{q}_y\}$$

**Guarantee:** Class-conditional coverage (with sufficient per-class calibration data).

#### Method C: Rank Calibrated Class-Conditional CP (RC3P)

Adds a **rank gate** on top of CcCP to prune unnecessary classes:

1. Top-k error matrix: ε_c^k = fraction of class-c calibration samples where true rank > k
2. Truncated alpha: α̃_c = α − Δ / √(n/K), with Δ=0.1
3. Per-class rank limit: smallest k such that ε_c^k < α̃_c
4. Grid search over mixing parameter η ∈ [0,1] to minimise average set size

$$\widehat{C}^{RC3P}(x) = \{y : \hat{\pi}_y(x) \geq 1 - \hat{q}_y \;\text{AND}\; r_f(x,y) \leq \hat{k}(y)\}$$

**Key property:** Strictly smaller prediction sets than CcCP via rank pruning.

#### Method D: Clustered CP (ClCP)

Groups semantically similar classes using feature-space clustering:

1. Extract penultimate-layer embeddings per calibration sample
2. Compute per-class mean embeddings μ_c
3. K-Means clustering with K_g=4 clusters on class means
4. Per-cluster threshold: q̂_g from all calibration samples in cluster g

$$\widehat{C}^{ClCP}(x) = \{y : \hat{\pi}_y(x) \geq 1 - \hat{q}_{g(y)}\}$$

**Key property:** Pools similar classes for better threshold estimation when per-class data is limited.

#### Method E: RAPS (Regularised Adaptive Prediction Sets)

Uses a **cumulative probability + rank penalty** nonconformity score:

$$s^{RAPS}(x,y) = \sum_{j=1}^{L(y)} \hat{\pi}_{o_j}(x) + \lambda \cdot \max(L(y) - k_{reg}, 0)$$

where L(y) = rank of true class, λ=0.01, k_reg=1.

**Key property:** Never produces empty sets; top-1 class always included; tends toward singleton sets when classifiers are well-calibrated.

### 5.3 Spatial Uncertainty Mapping

For all CP methods, thresholds are applied pixel-wise to the full-scene probability cube (330 × 307 × K):
- **Singleton sets** → classified by argmax ("certain")
- **Set size 0 or ≥2** → marked as "uncertain" (labelled as class K)
- Output: spatial maps showing certain vs. uncertain regions across the scene

---

## 6. Method 2 — CREDIT (Credal Ensemble Distillation)

**Directory:** `Classification/credit/`  
**Files:** `Model_training_credit.ipynb/.md`  
**Reference:** Wang et al., "Credal Ensemble Distillation for Uncertainty Quantification," arXiv:2511.13766

### 6.1 Core Idea

CREDIT compresses a **5-model deep ensemble** into a **single dual-head student network** via knowledge distillation. Instead of producing a single softmax prediction, the student predicts a **class-wise probability interval** (a *credal set*) — enabling a single forward pass to recover both aleatoric and epistemic uncertainty.

### 6.2 Teacher-Derived Training Targets

From M=5 independently trained teacher ensemble members, per-class min/max across ensemble:

$$p_{min,c}(x) = \min_{m=1,...,M} \pi_m(x)_c \qquad p_{max,c}(x) = \max_{m=1,...,M} \pi_m(x)_c$$

**Derived targets:**

$$\Delta p_{true,c}(x) = p_{max,c}(x) - p_{min,c}(x) \quad \text{[epistemic target: interval width]}$$

$$p^*_{true,c}(x) = \frac{p_{min,c}(x)}{\sum_j p_{min,j}(x) + \epsilon} \quad \text{[aleatoric target: renormalized minimum]}$$

> **Important:** p*_true is NOT the ensemble mean — it is the **most conservative (lowest) per-class vote**, renormalized to form a valid probability distribution.

### 6.3 Dual-Head Student Architecture

The student reuses an existing backbone up to its penultimate feature layer z(x), then attaches **two parallel heads:**

```
Backbone → z(x)
              ├── Softmax Head:  p̂*(x) = softmax(W₁·z + b₁)   [aleatoric proxy]
              └── Sigmoid Head:  Δ̂p(x) = σ(W₂·z + b₂)          [epistemic proxy]
```

- **Softmax** for the lower-bound head: p̂* must be a valid probability distribution (sums to 1)
- **Sigmoid** for the width head: interval widths are independent per class (need NOT sum to 1)

### 6.4 Loss Function

$$\mathcal{L}(\phi) = \text{KL}(p^*_{true} \,\|\, \hat{p}^*_\phi) + \lambda \cdot \text{MSE}(\Delta p_{true},\, \hat{\Delta p}_\phi)$$

| Component | Loss Type | Weight |
|-----------|-----------|--------|
| Aleatoric head (p*) | KL-divergence | 1.0 |
| Epistemic head (Δp) | Mean Squared Error | λ = 0.5 |

### 6.5 Inference-Time Uncertainty Decomposition

$$\text{AU}(x) = -\sum_c \hat{p}^*_c(x) \cdot \log(\hat{p}^*_c(x) + \epsilon) \quad \text{[Shannon entropy of conservative belief]}$$

$$\text{EU}(x) = \frac{1}{C} \sum_c \hat{\Delta p}_c(x) \quad \text{[Mean predicted interval width]}$$

$$\text{TU}(x) = \text{AU}(x) + \text{EU}(x) \quad \text{[Total uncertainty]}$$

| Uncertainty | High When... | Type |
|-------------|--------------|------|
| **AU** (Aleatoric) | Belief spread across many classes | Irreducible data ambiguity |
| **EU** (Epistemic) | Wide intervals (ensemble disagreement) | Reducible model ignorance |
| **TU** (Total) | Either or both | Combined |

### 6.6 Key Advantage

Only **1 forward pass** at inference vs. M=5 for the full ensemble — significant computational savings with minimal information loss.

---

## 7. Method 3 — DAPM (Deep Adaptive Predictive Modeling)

**Directory:** `Classification/dapm/`  
**Files:** `Model_training_dapm_full.ipynb/.md`, `Model_uncertainty_dapm_full.ipynb/.md`

### 7.1 Core Idea

DAPM is a two-stage, domain-adaptive model combining:
1. **Variational Autoencoder (VAE)** for compact probabilistic feature encoding
2. **Domain-Adversarial Training** (Gradient Reversal Layer) for domain-invariant representations
3. **Conditional Denoising Diffusion** over label distributions for probabilistic uncertainty modeling

At inference, DAPM generates N=30 stochastic class-probability samples per pixel via the diffusion chain, then applies a **Welch t-test** to flag uncertain pixels.

### 7.2 Architecture Overview

```
 ┌─────────────────────────────────────────────┐
 │ Frozen Backbone (AlexNet/GFNet/ViT-UNet)    │
 │ Input 9×9×6 → Feature vector h ∈ ℝ^d       │
 └──────────────────┬──────────────────────────┘
                    │
 ┌──────────────────▼──────────────────────────┐
 │ Shared VAE Encoder (MLP: 256→256)           │
 │ h → z_mu ∈ ℝ^64, z_logvar ∈ ℝ^64           │
 │ Reparameterization: z = μ + σ⊙ε, ε~N(0,I)  │
 └────┬───────────┬────────────┬───────────────┘
      │           │            │
      ▼           ▼            ▼
 ┌────────┐ ┌─────────┐ ┌──────────────────────┐
 │Source   │ │Target   │ │Domain Discriminator   │
 │Decoder  │ │Decoder  │ │(128-unit MLP + GRL)   │
 │(recon h)│ │(recon h)│ │P(target|z)            │
 └────────┘ └─────────┘ └──────────────────────┘
      │                        │
      ▼                        │
 ┌────────────┐                │
 │Classifier  │                │
 │(MLP→softmax│  ←── trained on source labels only
 │ K classes) │
 └────────────┘
      │
      ▼
 ┌────────────────────────────────────────┐
 │ Conditional Diffusion Denoiser         │
 │ Input: z ⊕ y_t ⊕ guidance g ⊕ t_emb  │
 │ Output: predicted noise ε̂ ∈ ℝ^K       │
 │ (T=100 timesteps, linear β schedule)   │
 └────────────────────────────────────────┘
```

### 7.3 Stage 1: VAE + Adversarial + Classifier Training

Jointly trains encoder, both decoders, classifier, and discriminator:

$$\mathcal{L}_1 = \lambda_{src}\mathcal{L}_{recon}^s + \lambda_{tgt}\mathcal{L}_{recon}^t + \lambda_{KL}(\text{KL}_{src} + \text{KL}_{tgt}) + \lambda_{CE}\mathcal{L}_{CE} + \lambda_{dom}\mathcal{L}_{dom}$$

| Component | Loss | Weight |
|-----------|------|--------|
| Source reconstruction | MSE(h, ĥ) | λ_src = 1.0 |
| Target reconstruction | MSE(h, ĥ) | λ_tgt = 1.0 |
| KL regularization | KL(N(μ,σ²) ‖ N(0,I)) | λ_KL = 0.01 |
| Classification | Cross-entropy (source only) | λ_CE = 1.0 |
| Domain adversarial | BCE with GRL | λ_dom = 0.2 |

**Gradient Reversal Layer (GRL):** Forward pass = identity; backward pass = negate gradient by −λ. Forces the encoder to produce domain-invariant latent codes that fool the discriminator.

### 7.4 Stage 2: Diffusion Training (Encoder + Classifier Frozen)

Trains **only** the diffusion denoiser network:

**Forward diffusion (closed-form):**

$$y_t = \sqrt{\bar{\alpha}_t} \cdot y_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

- y₀ for source: one-hot true label
- y₀ for target: stop-gradiented classifier soft prediction (pseudo-label)

**Reverse diffusion (inference — T=100 iterative steps):**

$$y_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(y_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \cdot D_\xi(z, y_t, g, t)\right) + \sqrt{\beta_t} \cdot \epsilon'$$

### 7.5 Uncertainty Quantification — Welch T-Test

For each pixel, generate N=30 independent probability samples:

1. Sample N latent codes: z^(n) = μ_z + σ_z ⊙ ε^(n), ε~N(0,I)
2. Run full T-step reverse diffusion for each z^(n)
3. Apply softmax → N probability vectors

Then, for each pixel:
- Identify **top-1 class c₁** and **top-2 class c₂** from the mean prediction
- Extract their probability distributions across N samples
- Apply **Welch two-sample t-test:**

$$t = \frac{\bar{g}_1 - \bar{g}_2}{\sqrt{s_1^2/N + s_2^2/N}}$$

**Decision rule:**
- p > 0.05 → **UNCERTAIN** (top-1 and top-2 statistically indistinguishable)
- p ≤ 0.05 → **CERTAIN** (top-1 significantly dominates)

> **Why Welch (not Student's t):** Does not assume equal variance between the two groups — appropriate since top-1 and top-2 distributions can have different spreads.

---

## 8. Method 4 — Deep Ensembles & CreDE

**Directory:** `Classification/ensemble/`  
**Files:** `Model_training_ensembles.ipynb/.md`, `Model_uncertainty_CreDE.ipynb/.md`

### 8.1 Ensemble Construction

- **Ensemble size:** M=5 independently-seeded members per architecture
- **Seeds:** `seed_val = 42 + i` for i=1..5
- **Diversity source:** Only weight initialization, dropout masks, and shuffle order differ. Architecture and training data remain **identical**.
- **Memory management:** Models trained sequentially with `tf.keras.backend.clear_session()` between members

### 8.2 Standard Deep Ensemble Uncertainty

**Ensemble predictive distribution:**

$$\bar{p}(y|x) = \frac{1}{M}\sum_{m=1}^{M} p_m(y|x)$$

**Total–Aleatoric–Epistemic decomposition (Depeweg et al. 2018):**

$$\underbrace{H[\bar{p}(y|x)]}_{\text{Total}} = \underbrace{\frac{1}{M}\sum_{m=1}^{M}H[p_m(y|x)]}_{\text{Aleatoric}} + \underbrace{\left(H[\bar{p}] - \frac{1}{M}\sum_m H[p_m]\right)}_{\text{Epistemic (Mutual Information)}}$$

### 8.3 CreDE — Credal Deep Ensemble (Training-Free Variant)

This implementation uses a **lighter, training-free variant** of CreDE (Wang et al., NeurIPS 2024). Instead of training specialised "CreNets," it forms the credal set from the **empirical envelope** (min/max) of ensemble members' softmax outputs.

#### Credal Bounds

$$\underline{p}_i(x) = \min_{m=1,...,M} q_{m,i}(x) \qquad \overline{p}_i(x) = \max_{m=1,...,M} q_{m,i}(x)$$

**Credal set:**

$$\mathcal{Q}(x) = \{q \in \Delta^{C-1} : \underline{p}_i(x) \leq q_i \leq \overline{p}_i(x) \;\forall\; i\}$$

#### Interval Width and Conservative Distribution

$$\Delta p_i(x) = \overline{p}_i(x) - \underline{p}_i(x)$$

$$p^*_i(x) = \frac{\underline{p}_i(x)}{\sum_{j=1}^{C}\underline{p}_j(x) + \epsilon}$$

#### CreDE Uncertainty Decomposition

$$\text{AU}(x) = -\sum_{i=1}^{C} p^*_i(x) \ln p^*_i(x) \quad \text{[Entropy of conservative distribution]}$$

$$\text{EU}(x) = \frac{1}{C}\sum_{i=1}^{C}\Delta p_i(x) \quad \text{[Mean interval width]}$$

$$\text{TU}(x) = \text{AU}(x) + \text{EU}(x)$$

#### Binary Uncertainty Masking

$$\text{mask}_{AU}(x) = \mathbb{1}[\text{AU}(x) > \tau_{AU}], \quad \tau_{AU} = 0.5$$

$$\text{mask}_{EU}(x) = \mathbb{1}[\text{EU}(x) > \tau_{EU}], \quad \tau_{EU} = 0.2$$

$$\text{mask}_{TU}(x) = \mathbb{1}[\text{TU}(x) > \tau_{TU}], \quad \tau_{TU} = 0.7$$

#### Prediction Rule (Maximin)

$$\hat{c}(x) = \arg\max_i p^*_i(x)$$

### 8.4 Standard Ensemble vs. CreDE Comparison

| Aspect | Standard Deep Ensemble | CreDE |
|--------|------------------------|-------|
| Combination rule | Average softmax: p̄ = (1/M)∑p_m | Min/max envelope → credal set |
| Representative distribution | p̄ (mixture mean) | p* (normalized lower bound) |
| Aleatoric uncertainty | Mean member entropy | Shannon entropy of p* |
| Epistemic uncertainty | Mutual information | Mean interval width |
| Prediction rule | argmax p̄ | argmax p* (maximin) |
| Key advantage | Information-theoretic decomposition | Preserves disagreement info that averaging erases |

---

## 9. Method 5 — MultiCP (Multi-Head Conformal Prediction)

**Directory:** `Classification/multicp/`  
**Files:** `Model_training_multihead.ipynb/.md`, `Model_uncertainty_multicp.ipynb/.md`

### 9.1 Multi-Head Architecture

Each backbone has **K=7 independent softmax heads** sharing the same feature representation:

$$\hat{\mathbf{p}}_k = \text{softmax}(W_k \mathbf{z} + \mathbf{b}_k), \quad k = 1, \ldots, 7$$

**Total ensemble:** 3 architectures × 7 heads = **21 classifiers**.

**Joint loss:** Sum of per-head cross-entropy losses:

$$\mathcal{L} = \sum_{k=1}^{K} \mathcal{L}_{CE}(\hat{\mathbf{p}}_k, y) = -\sum_{k=1}^{K} \log \hat{p}_{k,y}$$

#### Staged Channel-Shift Dropout (Novel)

A deterministic dropout strategy that cyclically zeroes different channel groups during training:
- Channels divided into S = 1/r contiguous groups (r=0.25 → S=4 stages)
- At stage s: channels in range [⌊r(s-1)C_f⌋, ⌊rsC_f⌋) are zeroed
- Transition trigger: `val_accuracy ≥ 0.985` AND at least 20 epochs in current stage
- Final stage swaps to standard stochastic Dropout for fine-tuning

### 9.2 MultiCP Conformal Prediction

Apply split conformal prediction **independently to each head**, then **intersect**:

#### Nonconformity Score Functions

**RAPS (Regularized Adaptive Prediction Sets):**

$$s^{(k)}_{RAPS}(x, y) = \sum_{j=1}^{o^{(k)}_x(y)-1} \hat{\pi}^{(k)}_{(j)}(x) + u \cdot \hat{\pi}^{(k)}_{(o)}(x) + \lambda \cdot (o^{(k)}_x(y) - k_{reg})^+$$

**SAPS (Sorted Adaptive Prediction Sets):**

$$s^{(k)}_{SAPS}(x, y) = \begin{cases} u \cdot \hat{\pi}^{(k)}_{max}(x) & \text{if } o=1 \\ \hat{\pi}^{(k)}_{max}(x) + (o - 2 + u)\lambda & \text{otherwise} \end{cases}$$

#### Per-Head Calibration and Prediction Sets

$$\hat{q}^{(k)} = \text{Quantile}_{1-\alpha}(s^{(k)}_1, \ldots, s^{(k)}_n)$$

$$\mathcal{C}^{(k)}(x) = \{y \in \mathcal{Y} : s^{(k)}(x, y) \leq \hat{q}^{(k)}\}$$

#### Joint (Intersection) Coverage

$$\mathcal{C}_{joint}(x) = \bigcap_{k=1}^{K} \mathcal{C}^{(k)}(x)$$

**Coverage guarantee:** Each individual head has ≥ 1−α marginal coverage. Under positive head dependence (realistic), joint coverage ≈ 1−α; under independence, coverage ≥ (1−α)^K.

### 9.3 Normalised Uncertainty Score

$$u(x) = \frac{\bar{S}(x)}{C}, \quad \text{uncertain}(x) = \mathbb{1}[u(x) \geq Q_{1-\xi}(\{u(x_j)\})]$$

where ξ = 0.10 (top 10% flagged as uncertain).

---

## 10. Method 6 — SACP (Self-Adaptive Conformal Prediction)

**Directory:** `Classification/sacp/`  
**Files:** `Model_sacp_comparison.ipynb/.md`

### 10.1 Core Idea

SACP smooths the **nonconformity score itself** across a local spatial neighbourhood before calibrating. This leverages the spatial structure of remote sensing data — nearby pixels tend to have similar uncertainty patterns.

### 10.2 Base Score: Randomized APS

$$S(x, y) = \sum_{j=1}^{o(y,x)-1} \pi_{(j)}(x) + U \cdot \pi_{(o(y,x))}(x), \quad U \sim \text{Uniform}(0,1)$$

### 10.3 Spatial Score Aggregation

$$V_t(B_i, y) = \lambda \cdot V_{t-1}(B_i, y) + \lambda \cdot \frac{1}{|\mathcal{N}_{valid}(r,c)|} \sum_{(r',c') \in \mathcal{N}_{valid}} V_{t-1}(B_{(r',c')}, y)$$

- V₀(B_i, y) = S(B_i, y) (un-smoothed base score)
- k smoothing iterations (default: k=1)
- λ = 0.5 (mixing weight for **both** self and neighbour terms)
- Window sizes swept: w ∈ {3, 5, 7, 9}

### 10.4 Calibration and Prediction Sets

$$\hat{\tau} = \inf\left\{s : \frac{|\{i : \hat{s}_i \leq s\}|}{n} \geq \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right\}$$

$$\hat{\mathcal{C}}_{1-\alpha}(B_{n+1}) = \{y \in \mathcal{Y} : V_k(B_{n+1}, y) \leq \hat{\tau}\}$$

**Coverage guarantee:** Preserved at ≥ 1−α despite smoothing (proven in original SACP paper).

**Empty set safeguard:** If Ĉ = ∅, include argmin_y V_k(B,y).

---

## 11. Method 7 — SCMCP (Spatial Multi-Head Conformal Prediction)

**Directory:** `Classification/multicp_sacp/`  
**Files:** `Model_uncertainty_multicp_sacp.ipynb/.md`

### 11.1 Core Innovation

SCMCP unifies MultiCP and SACP with a **critical architectural distinction**: spatial smoothing is applied to the **softmax probability vectors** (in probability space, with mandatory renormalisation) **before** score computation — not to the scores themselves.

### 11.2 Smoothing in Probability Space

$$\tilde{P}^{(k)}(r,c) = (1-\lambda) \hat{\pi}^{(k)}(r,c) + \lambda \cdot \frac{1}{|\mathcal{N}_w(r,c)|} \sum_{(r',c') \in \mathcal{N}_w(r,c)} \hat{\pi}^{(k)}(r',c')$$

**Mandatory renormalisation after smoothing:**

$$\hat{P}^{(k)}_{smooth}(r,c) = \frac{\tilde{P}^{(k)}(r,c)}{\sum_{c'} \tilde{P}^{(k)}(r,c,c') + \varepsilon}$$

### 11.3 Key Differences: SACP vs. SCMCP

| Aspect | SACP | SCMCP |
|--------|------|-------|
| **What is smoothed** | Nonconformity scores | Softmax probabilities |
| **Blending formula** | λ·own + λ·avg (both down-scaled) | (1−λ)·own + λ·avg (proper convex mix) |
| **Renormalisation** | Not needed (scores ≠ probabilities) | Mandatory after each pass |
| **Number of heads** | 1 (single-head per model) | K=7 (multi-head) |
| **Cal/Eval separation** | Same spatial volume | **Separate** spatial volumes |

### 11.4 Full SCMCP Pipeline

1. Multi-head inference → P ∈ [0,1]^{K×N×C}
2. Build neighbour offsets for window w
3. Spatial smoothing of calibration probabilities (separate volume) with renormalisation
4. Spatial smoothing of evaluation probabilities (separate volume)
5. Compute APS or SAPS scores on **smoothed** probabilities
6. Head sweep n_H = 1...K:
   - Per-head quantile thresholds
   - Per-head prediction sets
   - Intersect across heads → joint prediction sets
7. Per-class coverage analysis
8. Full-scene smoothing and uncertainty mapping
9. Sweep over w ∈ {3, 5, 7, 9} and scoring methods ∈ {APS, SAPS}

### 11.5 Joint Intersection

$$\mathcal{C}_{SCMCP}(x_i) = \bigcap_{k=1}^{K} \mathcal{C}^{(k)}(x_i) = \{c \in \mathcal{Y} : \forall k,\; s^{(k)}(c) \leq \hat{q}^{(k)}\}$$

**Coverage bounds:**
- Under independence: P(y ∈ ∩_k C^(k)) ≥ (1−α)^K
- Under positive dependence (realistic): P(y ∈ ∩_k C^(k)) ≥ 1−α

---

## 12. Method 8 — Focal Loss & CB Focal Loss

**Directory:** `Classification/trials/focal_loss/` & `Classification/trials/cb_focal_loss/`

Focuses on resolving class imbalance via standard Focal Loss and Class-Balanced Focal Loss. Enhances the predictive confidence for minority classes and adjusts the margin of prediction sets appropriately.

---

## 13. Method 9 — EDL & EDL_v2

**Directory:** `Classification/trials/edl/` & `Classification/trials/edl_v2/`

Implements Evidential Deep Learning based on Subjective Logic. Replaces softmax with a Dirichlet distribution output to quantify epistemic and aleatoric uncertainty without the need for sampling or ensembling.

---

## 14. Method 10 — CDL (Credal Deep Learning)

**Directory:** `Classification/trials/cdl/`

Extends evidential principles using imprecise probabilities and credal sets. Captures severe uncertainty and conflicting evidence by tracking the bounds of allowable probability distributions.

---

## 15. Method 11 — MambaHSI

**Directory:** `Classification/mambahsi/`

Adapts the state-of-the-art Mamba State-Space Model architecture for Hyperspectral Image classification. Offers linear scaling for long-range spatial-spectral sequences, solving the quadratic bottleneck of Vision Transformers.

---

## 16. Method 12 — DOFA (Dynamic Wavelength Tokenization)

**Directory:** `Classification/dofa/`

Dynamic Wavelength Tokenization framework incorporating DOFA Spectral and DOFA Hiera Fusion approaches. Designed to handle varying continuous spectral channels across multi-sensor remote sensing payloads, ensuring highly stable token generation.

---

## 17. Comparative Summary of All Methods

| # | Method | UQ Paradigm | Heads | Spatial? | Key Output | Compute Cost |
|---|--------|-------------|-------|----------|------------|-------------|
| 1 | **Baseline** | Conformal Prediction (5 variants) | 1 | No | Prediction sets with coverage guarantees | Low (single model) |
| 2 | **CREDIT** | Knowledge Distillation + Credal Sets | 2 (dual-head student) | No | AU/EU/TU maps from single pass | Low (1 forward pass) |
| 3 | **DAPM** | VAE + Diffusion + Welch t-test | 1 | No | P-value maps, uncertain pixel masks | High (N×T diffusion steps) |
| 4 | **CreDE** | Deep Ensemble + Credal Envelope | M=5 members | No | AU/EU/TU maps + credal set analysis | Medium (M forward passes) |
| 5 | **MultiCP** | Multi-Head Conformal Prediction | K=7 | No | Intersected prediction sets | Low (single model, K heads) |
| 6 | **SACP** | Spatial-Aware Conformal Prediction | 1 | Yes (scores) | Spatially-coherent prediction sets | Low-Medium |
| 7 | **SCMCP** | Spatial MultiCP (combined) | K=7 | Yes (probabilities) | Spatially-coherent intersected sets | Medium |

### Uncertainty Decomposition Comparison

| Method | Aleatoric Uncertainty | Epistemic Uncertainty |
|--------|----------------------|----------------------|
| **CREDIT** | Entropy of conservative belief p* | Mean interval width (1/C)∑Δp |
| **DAPM** | Implicit in diffusion variance | Welch t-test p-value |
| **CreDE** | Entropy of normalized lower bound | Mean credal set width |
| **Standard Ensemble** | Mean member entropy | Mutual information (H[p̄] − mean H[p_m]) |

---

## 18. Evaluation Metrics & Calibration

### Classification Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Overall Accuracy (OA)** | Correct / Total | Fraction of correctly classified pixels |
| **Cohen's Kappa (κ)** | (OA − p_e)/(1 − p_e) | Agreement correcting for chance |
| **Macro-F1** | (1/K)∑F1_k | Unweighted average of per-class F1 |
| **Weighted-F1** | ∑(n_k/N)·F1_k | Class-frequency weighted average |

### Calibration Metrics

**Multiclass Brier Score:**

$$\text{BS} = \frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}(\hat{p}_k^{(i)} - \mathbb{1}[y^{(i)}=k])^2$$

**Expected Calibration Error (ECE, 15-bin):**

$$\text{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{N}|\overline{\text{acc}}(B_m) - \overline{\text{conf}}(B_m)|$$

**Negative Log-Likelihood:**

$$\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N}\log p_i(y_i|x_i)$$

### Conformal Prediction Metrics

| Metric | Description |
|--------|-------------|
| **Empirical Coverage** | Fraction of test samples where true class ∈ prediction set |
| **Average Set Size** | Mean |C(x)| across test set — smaller = more informative |
| **Singleton Rate** | Fraction of prediction sets with exactly one class |
| **Empty-Set Rate** | Fraction of empty prediction sets |
| **Per-Class Coverage** | Marginal coverage broken down by class |

---

## 19. Master Hyperparameter Reference

### Training Hyperparameters

| Parameter | Value | Scope |
|-----------|-------|-------|
| Patch size P | 9×9 | All models |
| Spectral bands B | 6 | Input |
| Number of classes K | 7 | Task |
| Epochs | 100 | All training |
| Batch size | 128 | Training |
| Inference batch size | 256 (baseline), 2048 (CreDE) | Inference |
| SEED | 42 (main), 10 (AlexNet legacy split) | Reproducibility |
| Train/Test split | 75% / 25% | All |
| Label smoothing ε_s | 0.05 | GFNet, ViT-UNet |
| AlexNet dropout | 0.25 | AlexNet |
| AlexNet LR | cosine cycling [0.005, 0.02] | Adagrad |
| AdamW weight decay | 10⁻⁴ | GFNet, ViT-UNet |
| Gradient clip norm | 1.0 | GFNet, ViT-UNet |
| Numerical ε | 10⁻⁸ (normalisation), 10⁻¹² (probability ops) | Stability |

### Architecture-Specific

| Parameter | AlexNet | GFNet | ViT-UNet |
|-----------|---------|-------|----------|
| Conv filters | [96,256,384,384,256] | — | — |
| Dense layers | [4096,1024,256,32] | — | [512,256,128,64] |
| Hidden dim | — | 512 (fallback: 384) | 256 (fallback: 192) |
| Blocks/Layers | 5 conv + 4 dense | 5 GF blocks | 12 transformer layers |
| Inner patch | — | 3×3 | 3×3 |
| Tokens | — | 9 | 10 (9+CLS) |
| Activation | ReLU | ReLU | GELU |

### Method-Specific

| Parameter | Value | Method |
|-----------|-------|--------|
| α (CP miscoverage) | 0.05 | All CP methods |
| M (ensemble size) | 5 | Ensemble, CreDE, CREDIT |
| K (multi-heads) | 7 | MultiCP, SCMCP |
| λ (CREDIT loss weight) | 0.5 | CREDIT |
| τ_AU / τ_EU / τ_TU | 0.5 / 0.2 / 0.7 | CreDE thresholds |
| N (diffusion samples) | 30 | DAPM |
| T (diffusion timesteps) | 100 | DAPM |
| β_start / β_end | 10⁻⁴ / 2×10⁻² | DAPM diffusion schedule |
| α_ttest | 0.05 | DAPM Welch test |
| λ (spatial blend) | 0.5 | SACP, SCMCP |
| k_iter (smoothing rounds) | 1 | SACP, SCMCP |
| w (window sizes) | {3, 5, 7, 9} | SACP, SCMCP |
| RAPS λ / k_reg | 0.01 / 1 | Baseline RAPS, MultiCP |
| RC3P Δ | 0.1 | Baseline RC3P |
| ClCP K_g (clusters) | 4 | Baseline ClCP |
| ECE bins | 15 | All calibration |
| Staged shift accuracy threshold | 0.985 | MultiCP training |

---

## 20. Examples Directory

The `examples/` directory contains **10 pre-configured dataset suites**, each mirroring the full `Classification/` directory structure (baseline, credit, dapm, ensemble, multi_cp, sacp, data). These serve as ready-to-use templates for different remote sensing datasets:

| Example Suite | Description |
|---------------|-------------|
| `indian_pines_uncertainty_quantification` | Classic AVIRIS hyperspectral benchmark (224 bands, 16 classes) |
| `pavia_uncertainty_quantification` | ROSIS Pavia Centre urban scene |
| `pavia_university_uncertainty_quantification` | ROSIS Pavia University campus |
| `6_band_uncertainty_quantification` | 6-band multispectral (template-compatible) |
| `372_band_uncertainty_quantification` | Full hyperspectral (372 bands) |
| `dias_uncertainty_quantification` | DIAS satellite data |
| `hisar_3mts_uncertainty_quantification` | Hisar 3-month temporal composite |
| `hissar_25_uncertainty_quantification` | Hisar 25-band multispectral |
| `multispectral_uncertainty_quantification` | Generic multispectral template |
| `planet_data_hisar_uncertainty_quantification` | PlanetScope commercial satellite data (Hisar) |

Each suite has its own `data/` subdirectory and method-specific notebooks pre-configured for that dataset's band count and class structure.

---

## 21. References

### Research Papers

| Citation | Method | Link |
|----------|--------|------|
| Wang, K. et al. (2026) | CREDIT: Credal Ensemble Distillation | [arXiv:2511.13766](https://arxiv.org/abs/2511.13766) |
| Angelopoulos et al. (2021) | RAPS: Regularised Adaptive Prediction Sets | [arXiv:2009.14193](https://arxiv.org/abs/2009.14193) |
| Huang et al. (2024) | SAPS: Sorted Adaptive Prediction Sets | — |
| Romano, Sesia & Candès (2020) | Conformal Prediction via Label Ranking | [arXiv:2310.06430](https://arxiv.org/abs/2310.06430) |
| Wang et al. (NeurIPS 2024) | CreDE: Credal Deep Ensembles | — |
| Lakshminarayanan et al. (NeurIPS 2017) | Deep Ensembles | [arXiv:1612.01474](https://arxiv.org/abs/1612.01474) |
| Guo et al. (ICML 2017) | On Calibration of Modern Neural Networks | [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) |
| Ganin & Lempitsky (2015) | Domain-Adversarial Neural Networks | — |
| Kingma & Welling (2014) | VAE: Variational Autoencoders | — |
| Ho et al. (2020) | DDPM: Denoising Diffusion Probabilistic Models | — |
| Dhariwal & Nichol (2021) | Classifier Guidance for Diffusion Models | — |
| Rao et al. (NeurIPS 2021) | GFNet: Global Filter Networks | — |
| Dosovitskiy et al. (ICLR 2021) | ViT: Vision Transformer | — |
| Ronneberger et al. (MICCAI 2015) | U-Net | — |
| Depeweg et al. (ICML 2018) | Uncertainty Decomposition in Ensembles | — |
| Hüllermeier & Waegeman (ML 2021) | Aleatoric vs Epistemic Uncertainty | — |
| Brier (1950) | Brier Score | — |
| Welch (1947) | Welch t-test | — |

### Libraries & Frameworks

- **TensorFlow / Keras:** Deep learning framework for all model architectures
- **SACP:** [GitHub Repository](https://github.com/J4ckLiu/SACP)

---

_This documentation was auto-generated from the repository's source code and method summary files. For the most granular implementation details, refer to the individual `.md` files within each method's directory._

_This repository is developed under the guidance of **Dr. Mahesh Pal**._
