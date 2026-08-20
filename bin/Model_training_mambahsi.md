# Multispectral Patch-Based Classification Pipeline: MambaHSI Theory & Implementation

> **One-line description:** A deep-learning framework that trains a spatial-spectral state-space model (MambaHSI) across three capacities (Small, Base, Large) on spatially extracted patches from a 6-band multispectral image to produce pixel-wise land-cover classification maps.

---

## 1. Overview & Intuition

Remote sensing images contain rich spatial and spectral information. Patch-based classification leverages local spatial context by extracting a neighbourhood window (patch) around each pixel. 

This notebook introduces **MambaHSI**, a model inspired by the Mamba selective state-space architecture. Unlike self-attention (which scales quadratically with sequence length), Mamba scales linearly while maintaining a global receptive field. To adapt Mamba for 2D spatial patches, MambaHSI divides the 9×9 patch into non-overlapping 3×3 tokens. It processes this sequence using a bidirectional selective-scan mechanism (`tf.scan` over the sequence axis) that allows the state space to capture non-causal context. Three capacities (Small, Base, Large) are trained independently and evaluated, generating both classification maps and calibrated probability outputs for downstream uncertainty estimation.

---

## 2. Mathematical Framework

### 2.1 Problem Setup and Patch Extraction

Let the multispectral scene be $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$, where $H = 330$, $W = 307$, and $B = 6$. Each band is normalised independently to $[0, 1]$ via min-max scaling.
For each labelled pixel, a square patch of side $P = 9$ is extracted:
$$\mathbf{X}^{(i)} = \tilde{\mathcal{X}}_{\text{pad}}\bigl[r:r+P,\; c:c+P,\; :\bigr] \in \mathbb{R}^{P \times P \times B}$$

### 2.2 MambaHSI Architecture

The patch is divided into non-overlapping inner patches (side $= 3$), yielding $N = (9/3)^2 = 9$ tokens. A `PatchPositionEncoder` projects each token to dimension `hidden_dim` and adds a learned positional embedding.

The core of MambaHSI is the **SelectiveSSMBlock**, which applies a bidirectional selective state-space scan.
The forward pass uses a **SpectralScanLayer**:
$$u_t = \mathbf{W}_{\text{in}} x_t$$
$$g_t = \sigma(\mathbf{W}_{\text{gate}} x_t)$$
$$\tilde{u}_t = u_t \odot g_t$$
$$h_t = \sigma(\mathbf{w}_{\text{decay}}) \odot h_{t-1} + \tilde{u}_t$$
$$y_t = \mathbf{W}_{\text{out}} h_t$$
Where $\mathbf{w}_{\text{decay}}$ is a learned log-decay parameter. The selection mechanism (input-dependent gate $g_t$) allows the model to selectively remember or forget context.
Because images lack a causal direction, the sequence is also scanned backwards. The forward and backward representations are concatenated, projected back to `hidden_dim`, and added to the residual stream. This is followed by a two-layer GELU MLP.

After stacking multiple SSM blocks (Small: 2, Base: 4, Large: 6), global average pooling collapses the token sequence into a single vector, which is passed through a dense softmax layer for classification.

### 2.3 Loss Functions and Optimisation

MambaHSI uses label-smoothed categorical cross-entropy with a smoothing factor $\epsilon_s = 0.05$. Optimization uses AdamW with a cosine decay schedule. Calibration metrics like Multiclass Brier Score and Expected Calibration Error (ECE) are tracked.

---

## 3. Algorithm

**Input:** Multispectral scene, label map, patch size $P=9$, num classes $K=7$.
**Output:** Trained model weights (Small, Base, Large), metrics, prediction maps.

1. **Normalise** each band independently.
2. **Extract patches** of size 9×9×6.
3. **Split** stratified into train (75%), validation (20% of train), and test.
4. **Build models**: Instantiate MambaHSI (Small, Base, Large).
5. **Train** for 100 epochs using AdamW + Cosine Decay. On `ResourceExhaustedError`, automatically retry Base/Large models with reduced fallback configs.
6. **Evaluate** on the test set computing accuracy, Cohen's κ, macro/weighted-F1, Brier score, and ECE.
7. **Dense scene inference**: slide the window across the full scene to produce predicted label maps.

---

## 4. Implementation Walkthrough

- **`SpectralScanLayer`**: Approximates Mamba's fused selective-scan with a pure Keras `tf.scan` loop, applying an input-dependent sigmoid gate before the linear recurrence.
- **`SelectiveSSMBlock`**: Wraps the bidirectional scan (`scan_fwd` and `scan_bwd`) and the GELU MLP within pre-LayerNorm residual connections.
- **`build_mambahsi`**: Factory function that strings together patch extraction, positional encoding, $N$ SSM blocks, GAP, and the classification head.
- **Fallback Logic**: Wraps training in a try/except block catching `tf.errors.ResourceExhaustedError` to gracefully handle Colab GPU OOM errors by falling back to `MAMBAHSI_BASE_FALLBACK_CFG` or `MAMBAHSI_LARGE_FALLBACK_CFG`.
