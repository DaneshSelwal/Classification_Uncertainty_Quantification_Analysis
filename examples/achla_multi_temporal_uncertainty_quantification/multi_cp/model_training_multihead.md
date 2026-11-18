# Multi-Head Deep Learning Ensemble for Multispectral Image Classification: Theory & Implementation Summary

> **One-line description:** A shared-backbone, multi-head sub-ensemble framework that trains three deep architectures — AlexNet-CNN, GFNet, and Vision Transformer (ViT-UNet) — each emitting seven independent softmax heads, whose outputs are averaged at inference to yield calibrated class probability distributions for patch-based multispectral image classification.

---

## 1. Overview & Intuition

### 1.1 The Problem

Multispectral remote sensing images consist of spatial pixels annotated with a small number of spectral bands. Classifying each pixel into land-cover categories (e.g. urban, vegetation, water bodies, bare soil) is a fundamental task with applications in agriculture, urban planning, and environmental monitoring. Because a single pixel's class depends not just on its own spectral signature but also on its spatial context, the predominant strategy is to extract a spatial neighbourhood (a patch) centred on the target pixel and present that patch to a classifier.

A persistent difficulty in this setting is that neural networks trained with standard softmax cross-entropy loss produce overconfident, poorly-calibrated predictions. A model can output a high-probability classification even when it is genuinely uncertain — a problem that propagates into any downstream uncertainty-aware pipeline (e.g. conformal prediction, active learning, or Bayesian decision-making).

### 1.2 Why Multi-Head Sub-Ensembles?

Deep Ensembles — training $M$ independent models and averaging their predictions — is a powerful approach that improves both accuracy and calibration. However, a full ensemble multiplies training cost by $M$ and requires $M$ separate inference passes. The **deep sub-ensemble** (DSE) paradigm addresses this by sharing a large common backbone and diverging only in a final set of classification heads. Each head sees the same learned representation but is independently optimized via a separate loss term, introducing diversity through stochastic gradient descent dynamics and random weight initialization. At inference, a single forward pass through the shared trunk suffices; only the $K$ small head outputs need to be aggregated.

This notebook extends that concept by training **three architecturally distinct backbones**, each equipped with $K = 7$ independent softmax heads, producing a cross-architecture ensemble of 21 classifiers in total. Aggregating across diverse architectural inductive biases (local convolution, global frequency filtering, and self-attention) is expected to yield richer epistemic diversity than any single-architecture sub-ensemble.

### 1.3 Architecture Diversity

The three backbones chosen represent three philosophically distinct approaches to feature extraction from spatial patches:

**AlexNet-CNN** extracts hierarchical local features through stacked convolutions. Each layer learns increasingly abstract spatial patterns. Its inductive bias — local connectivity and translation equivariance — is well-suited to detecting textures and edge statistics that distinguish land-cover types.

**GFNet (Global Filter Network)** replaces self-attention with learnable filters applied in the frequency domain via the 2-D Fast Fourier Transform. Rather than computing pixel-pair interactions in spatial space (quadratic cost), GFNet modulates the Fourier spectrum of a patch directly, achieving global receptive fields at log-linear cost. It is designed to capture long-range spectral correlations invisible to purely local convolutional filters.

**ViT-UNet (Vision Transformer with U-Net skip connections)** divides each patch into a sequence of sub-patches, treats them as tokens, and routes them through multi-head self-attention layers with a symmetric skip-connection (U-Net) structure. This allows both long-range interactions across the patch (via attention) and fine-grained local preservation (via skip connections). A learnable class token ([CLS]) aggregates the full patch representation for classification.

### 1.4 Staged Dropout Training (Deterministic Channel Shift)

A novel training stabilization mechanism called **staged channel-shift dropout** is applied across all three architectures. Instead of randomly zeroing neurons at each forward pass, the dropout layer divides channels into $S = 1/r$ contiguous groups (where $r$ is the dropout rate) and cycles through them across training stages. Only the currently designated group of channels is zeroed; all other channels receive gradient signal. Once a validation accuracy threshold is reached, the model transitions to the next stage, shifting the zeroed group. This guarantees every channel is trained fully without being permanently dropped, improving gradient flow and convergence stability compared to standard stochastic dropout.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathbf{x} \in \mathbb{R}^{P \times P \times B}$ be a spatial patch of size $P \times P$ pixels with $B$ spectral bands centred on a target pixel, and let $y \in \{0, 1, \dots, C-1\}$ be its class label. The dataset is a stratified split of labeled pixels extracted from a multispectral image of shape $H \times W \times B$.

In this notebook:
- $P = 9$ (patch size)
- $B = 6$ (spectral bands)
- $C$ = number of distinct land-cover classes (derived from the label map, excluding background class 0)
- Training split: 75% train / 25% test
- $K = 7$ independent output heads per model

### 2.2 Per-Band Min-Max Normalisation

Before patch extraction, each spectral band $b$ is independently normalized to $[0, 1]$:

$$x_{i,j,b}^{\text{norm}} = \frac{x_{i,j,b} - \min_b(x)}{\max_b(x) - \min_b(x) + \epsilon}$$

**Where:**
- $x_{i,j,b}$ — pixel value at row $i$, column $j$, band $b$
- $\min_b(x),\ \max_b(x)$ — minimum and maximum values across all spatial positions in band $b$
- $\epsilon = 10^{-8}$ — numerical guard to prevent division by zero

**What this means:** Each spectral band is re-scaled independently so that all bands lie in a comparable numeric range, preventing bands with larger absolute values from dominating the learning signal during training.

### 2.3 Patch Extraction

For a target pixel $(r, c)$, the spatial neighbourhood patch is extracted from the edge-padded image:

$$\mathbf{x}^{(r,c)} = x_{\text{padded}}[r : r+P,\ c : c+P,\ :] \in \mathbb{R}^{P \times P \times B}$$

Edge padding with $\lfloor (P-1)/2 \rfloor$ replicated pixels ensures every labeled pixel, including those on the image border, has a valid $P \times P$ neighbourhood.

### 2.4 Multi-Head Softmax Output

Each backbone $f_\theta$ maps an input patch $\mathbf{x}$ to a shared feature representation $\mathbf{z} = f_\theta(\mathbf{x}) \in \mathbb{R}^D$. This representation is passed to $K$ independent linear heads:

$$\hat{\mathbf{p}}_k = \text{softmax}(W_k \mathbf{z} + \mathbf{b}_k), \quad k = 1, \dots, K$$

**Where:**
- $\hat{\mathbf{p}}_k \in \Delta^{C-1}$ — predicted probability simplex from head $k$ (lies on the $(C-1)$-dimensional probability simplex)
- $W_k \in \mathbb{R}^{C \times D},\ \mathbf{b}_k \in \mathbb{R}^C$ — weight matrix and bias vector unique to head $k$
- $K = 7$ — number of heads

**What this means:** The backbone is shared; all heads see the same features. However, each head has its own independently learned projection from feature space to class logits. Diversity arises because the $K$ heads are initialized differently and receive independent gradient updates.

### 2.5 Loss Function (Multi-Head Sparse Categorical Cross-Entropy)

The multi-head model is trained jointly by summing sparse categorical cross-entropy losses across all $K$ heads:

$$\mathcal{L} = \sum_{k=1}^{K} \mathcal{L}_{\text{CE}}(\hat{\mathbf{p}}_k, y) = -\sum_{k=1}^{K} \log \hat{p}_{k,y}$$

**Where:**
- $\hat{p}_{k,y}$ — the probability assigned to the true class $y$ by head $k$
- The same label $y$ supervises all $K$ heads

**What this means:** Every head receives the same supervision signal, but their independent parameterisations create functionally diverse classifiers that will disagree in proportion to their epistemic uncertainty.

### 2.6 Inference by Probability Averaging

At inference time, the outputs of all $K$ heads are averaged to produce a single calibrated predictive distribution:

$$\bar{\mathbf{p}} = \frac{1}{K} \sum_{k=1}^{K} \hat{\mathbf{p}}_k$$

The final class prediction is then:

$$\hat{y} = \arg\max_{c}\ \bar{p}_c$$

**What this means:** Head averaging approximates a mixture model. When heads agree (low variance), $\bar{\mathbf{p}}$ is sharply peaked — high confidence. When heads disagree (high variance), $\bar{\mathbf{p}}$ is flatter — lower maximum probability signalling genuine uncertainty. This calibration property is valuable for downstream conformal or probabilistic methods.

### 2.7 GFNet Global Filter Operation

The core operation distinguishing GFNet from CNNs and ViTs is spectral-domain token mixing. Given a patch token tensor $\mathbf{X} \in \mathbb{R}^{H_p \times W_p \times C}$:

**Step 1 — 2-D Real FFT:**
$$\hat{\mathbf{X}} = \mathcal{F}_{2D}(\mathbf{X}) \in \mathbb{C}^{H_p \times \lfloor W_p/2 \rfloor + 1 \times C}$$

**Step 2 — Hadamard product with a learnable complex global filter $\mathbf{K}$:**
$$\tilde{\mathbf{X}} = \hat{\mathbf{X}} \odot \mathbf{K}$$

**Step 3 — 2-D Inverse Real FFT:**
$$\mathbf{X}' = \mathcal{F}^{-1}_{2D}(\tilde{\mathbf{X}}) \in \mathbb{R}^{H_p \times W_p \times C}$$

**Where:**
- $\mathcal{F}_{2D}$ — 2-D real-to-complex discrete Fourier Transform (`tf.signal.rfft2d`)
- $\mathbf{K} \in \mathbb{C}^{H_p \times (W_p/2+1) \times C}$ — learnable global filter, stored as a real-valued weight with two components (real and imaginary parts)
- $\odot$ — element-wise (Hadamard) complex multiplication
- $\mathcal{F}^{-1}_{2D}$ — inverse transform (`tf.signal.irfft2d`)

**What this means:** In the Fourier domain, multiplication with a filter $\mathbf{K}$ is equivalent to convolution with a spatial kernel of the same spatial extent as the patch. Since $\mathbf{K}$ spans the entire frequency representation, its effective spatial receptive field encompasses all positions simultaneously — hence "global filter." The complexity is $\mathcal{O}(N \log N)$ rather than the quadratic $\mathcal{O}(N^2)$ of self-attention.

### 2.8 Vision Transformer Self-Attention

In the ViT-UNet backbone, each transformer block applies scaled dot-product multi-head attention to a sequence of patch tokens:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

**Where:**
- $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d_k}$ — query, key, and value matrices derived from the token sequence
- $N$ — number of tokens (patches + 1 CLS token)
- $d_k = D / H$ — per-head key dimension (projection dimension divided by number of heads)
- $\sqrt{d_k}$ — scaling factor preventing softmax saturation for large $d_k$

The output for each head is a weighted sum of value vectors, where the weight between token $i$ and token $j$ reflects their semantic similarity. Stacking $H$ independent attention heads and projecting the concatenated outputs gives multi-head attention.

### 2.9 U-Net Skip Connections in the Transformer Stack

The `ViT_TransFormer_Block` applies a symmetric residual structure analogous to the U-Net encoder–decoder:

- **Encoding phase** (first $\lfloor L/2 \rfloor$ transformer layers): intermediate activations are stored in a `block_list`.
- **Decoding phase** (remaining layers): each layer output is added element-wise to its mirror counterpart from the encoding phase:

$$\mathbf{x}_i^{\text{dec}} = \text{TransFormerBlock}_i(\mathbf{x}_{i-1}^{\text{dec}}) + \mathbf{x}_{L-i-1}^{\text{enc}}$$

**What this means:** Skip connections propagate fine-grained token representations from earlier (less contextualized) layers directly to later (more abstract) layers, preventing the loss of spatial locality that can occur in deep attention stacks.

### 2.10 Pearson Correlation Attention (Optional Preprocessing)

When `use_pearson_corr=True`, the `Pearson_correlation_masked` layer pre-processes each patch by computing the Pearson correlation coefficient $\rho$ between every spatial position's spectral vector and that of the central pixel:

$$\rho_{i,j} = \frac{\sum_b (x_{i,j,b} - \bar{x}_{i,j})(x_{\text{ctr},b} - \bar{x}_\text{ctr})}{\sqrt{\sum_b (x_{i,j,b} - \bar{x}_{i,j})^2 \cdot \sum_b (x_{\text{ctr},b} - \bar{x}_\text{ctr})^2}}$$

Correlations above the patch mean are retained; the rest are zeroed, creating a spatial attention mask that emphasizes pixels spectrally similar to the target pixel before the backbone processes the patch.

### 2.11 Staged Channel-Shift Dropout

Let a layer have $C_f$ feature channels and dropout rate $r$ (giving $S = 1/r$ total stages). At training stage $s$ (one-indexed):

$$\text{mask}_j = \begin{cases} 0 & \text{if}\ j \in [\lfloor r(s-1)C_f \rfloor,\ \lfloor r \cdot s \cdot C_f \rfloor) \\ 1 & \text{otherwise} \end{cases}$$

$$\mathbf{h}' = \mathbf{h} \odot \text{mask}$$

**Where:**
- $j$ — channel index
- $s$ — current shift index
- The zeroed slice moves rightward by $\lfloor r \cdot C_f \rfloor$ channels each time the accuracy threshold is reached

**What this means:** Unlike random dropout (which may never train some channels and overtrain others in finite epochs), channel-shift dropout guarantees that every channel is both trained (when unmasked) and regularized (when masked) over the course of training.

### 2.12 Cosine Annealing Learning Rate Schedules

**Single-stage (ViT):**
$$\eta_e = (\eta_{\max} - \eta_{\min}) \cdot \frac{1 + \cos(\pi e / (E-1))}{2} + \eta_{\min}$$

**Three-stage multi-step cosine (AlexNet, GFNet):** The total epoch budget is divided into three equal stages. Within each stage, the above formula restarts from $\eta_{\max}$ to $\eta_{\min}$.

**Where:**
- $e$ — current epoch (within-stage)
- $E$ — total epochs in the current stage
- $\eta_{\max}, \eta_{\min}$ — maximum and minimum learning rates (model-specific)

**What this means:** Cosine annealing smoothly reduces the learning rate, allowing the optimizer to converge carefully near minima rather than overshooting them.

---

## 3. Algorithm

**Input:** Multispectral image $\mathbf{I} \in \mathbb{R}^{H \times W \times B}$, label map $\mathbf{L} \in \{0, \dots, C\}^{H \times W}$  
**Output:** Trained multi-head models (AlexNet, GFNet, ViT-UNet), model registry JSON, Excel results workbook

1. **Normalize** each spectral band independently to $[0, 1]$ using per-band min-max.
2. **Extract patches:** For every labeled pixel $(r, c)$ with $L_{r,c} \neq 0$, extract a $P \times P \times B$ neighbourhood from the edge-padded image; record zero-indexed label $L_{r,c} - 1$.
3. **Split** the patch dataset into 75% train / 25% test using stratified random sampling.
4. **For each model** $m \in \{$AlexNet, GFNet, ViT-UNet$\}$:
   1. **Build** the backbone with $K = 7$ independent softmax heads.
   2. **Compile** with $K$ `sparse_categorical_crossentropy` losses and corresponding accuracy metrics.
   3. **Configure** the staged-dropout callback (`Custom_callbacks`) targeting `TRAIN_DROPOUT_*` layers, with shift trigger at `Targeted_accuracy = 0.985` and minimum `20` epochs per shift.
   4. **Build** a cosine-annealing learning-rate scheduler (three-stage for AlexNet and GFNet; single-stage for ViT).
   5. **Train** for `epoch = 100` epochs with `batch_size = 128`, using `ModelCheckpoint` to save the best `val_head_1_accuracy` checkpoint.
   6. **Reload** the best checkpoint; copy to a final path.
   7. **Evaluate:** Call `predict_multihead` — stack the $K$ head outputs, average across heads, argmax to get labels — then compute accuracy, Cohen's Kappa, and a full classification report.
   8. **Export** training curves, performance figures, and a learning-rate schedule plot to the Excel workbook.
5. **Write** a consolidated Summary sheet to the Excel workbook.
6. **Save** a `model_registry_multihead.json` mapping each model name to its checkpoint paths and metadata.
7. **Smoke-check** each saved model: assert it returns exactly 7 output tensors on a 4-sample mini-batch.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_training_multihead.md`

### 4.1 Data Preparation and Patch Extraction

```python
# Per-band normalisation
for band_idx in range(B):
    band  = x[:, :, band_idx]
    denom = max(float(np.max(band) - np.min(band)), 1e-8)
    x[:, :, band_idx] = (band - np.min(band)) / denom

# Edge-pad the image, then slide to collect P×P×B patches
padded_x = np.pad(x, [(pad_width, pad_width), (pad_width, pad_width), (0, 0)], 'edge')
for row_idx in range(H):
    for col_idx in range(W):
        if y[row_idx][col_idx] != 0:
            patch = padded_x[row_idx:row_idx + P_S, col_idx:col_idx + P_S, :]
            X.append(patch)
            Y.append(y[row_idx][col_idx] - 1)
```

**What this does:** Normalizes each band independently, then uses a sliding window to collect labelled $9 \times 9 \times 6$ patches. Background pixels (label 0) are excluded.  
**Why:** Per-band normalization prevents spectral bands with large absolute values from overwhelming gradient updates. Edge padding (rather than zero-padding) avoids artificial spectral discontinuities at image borders.

---

### 4.2 Pearson Correlation Attention Layer

```python
class Pearson_correlation_masked(layers.Layer):
    def call(self, inputs):
        loc = self.P_S // 2
        # ... compute Pearson r between each pixel and central pixel ...
        thresh       = tf.math.reduce_mean(corr)
        mask         = tf.cast(corr > thresh, corr.dtype)
        attention_wts = tf.repeat(mask * corr, repeats=channels, axis=-1)
        return multiply([inputs, attention_wts])
```

**What this does:** Computes the Pearson correlation coefficient of each pixel's spectral vector against the central pixel's spectrum, thresholds at the patch mean, and multiplies the input by the resulting attention weights.  
**Why:** Pixels with a spectrally dissimilar signature to the target are likely from a different class (context noise). Down-weighting them before the backbone reduces inter-class confusion in contextually mixed patches.

---

### 4.3 Staged Channel-Shift Dropout

```python
class Dropout_Train(layers.Layer):
    def call(self, inputs, training=None):
        def dropped_inputs():
            range_0 = int(self.rate * (self.shift - 1) * input_shape[-1])
            range_1 = int(self.rate * self.shift * input_shape[-1])
            multiplier = np.ones(input_shape[-1])
            multiplier[range_0:range_1] = 0.0
            return Multiply()([inputs, tf.constant(multiplier)])
        return smart_cond(training, dropped_inputs, lambda: identity(inputs))
```

**What this does:** Applies a deterministic binary mask that zeros a fixed contiguous slice of channels during the forward pass. The `shift` attribute (set by `Custom_callbacks`) determines which slice is zeroed.  
**Why:** Deterministic channel masking ensures every channel receives gradient signal in at least $S-1$ out of $S$ training stages, unlike stochastic dropout where some channels may rarely be activated in finite training.

---

### 4.4 Custom Staged-Training Callback

```python
class Custom_callbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        threshold_met = (acc >= self.accuracy_score) and (epoch_completed >= self.min_epochs)
        if threshold_met and self.shift < total_shifts:
            self.model = modified_model(self.model, self.layer_name, self.rate,
                                        self.new_layer, self.shift + 1)
            self.shift += 1
            self.epoch_completed = 0
        elif threshold_met and self.shift == total_shifts:
            # Replace with standard Dropout for final convergence
            self.model = modified_model(..., shift="Final")
```

**What this does:** Monitors `val_accuracy` at each epoch end. When the threshold (98.5%) is reached and at least 20 epochs have elapsed in the current shift, it rebuilds the model with the next shift's dropout configuration. After all $S = 4$ shifts complete, it swaps in standard stochastic `Dropout` for a final fine-tuning phase.  
**Why:** Staged transitions ensure that the model has stabilized (met the accuracy target) before removing the regularizing mask from a channel group. This prevents early collapse from aggressive dropout removal.

---

### 4.5 AlexNet Multi-Head Model Builder

```python
def AlexNet(input_shape, num_classes=13, dropout_rate=0.5):
    K_HEADS = 7
    X = Conv2D(96,  (3,3), activation='relu', padding='same')(X_input)
    X = Conv2D(256, (3,3), activation='relu', padding='same')(X)
    X = Conv2D(384, (3,3), activation='relu', padding='same')(X)
    X = Conv2D(384, (3,3), activation='relu', padding='same')(X)
    X = Conv2D(256, (3,3), activation='relu', padding='same')(X)
    X = MaxPooling2D((2,2))(X)
    X = Flatten()(X)
    X = Dense(4096, activation='relu')(X)
    X = Dropout(dropout_rate, name="TRAIN_DROPOUT_1")(X)
    X = Dense(1024, activation='relu')(X)
    X = Dropout(dropout_rate, name="TRAIN_DROPOUT_2")(X)
    X = Dense(256, activation='relu')(X)
    X = Dropout(dropout_rate, name="TRAIN_DROPOUT_3")(X)
    X = Dense(32, activation='relu')(X)
    output_heads = [Dense(num_classes, activation='softmax', name=f'head_{i+1}')(X)
                    for i in range(K_HEADS)]
    return Model(inputs=x_input, outputs=output_heads, name="MultiHead_AlexNet")
```

**What this does:** Builds a 5-block convolutional backbone (96→256→384→384→256 filters) with max-pooling, then a 4-layer dense tower (4096→1024→256→32), and fans out to 7 independent softmax heads.  
**Why:** The AlexNet architecture provides local spatial–spectral feature extraction. Named `TRAIN_DROPOUT_*` layers are targeted by the staged callback. The narrow final Dense(32) bottleneck encourages all 7 heads to compete for a compact shared representation.

---

### 4.6 GFNet Multi-Head Model Builder

```python
def GFNet(input_shape, hidden_dim=512, GlobalFilter_layers=12, ...):
    x = GF_Patches(patch_size)(x_input)       # patch tokenisation
    x = GF_PatchEncoder(num_patches, hidden_dim)(x)  # project + add positions
    x = Dropout(dropout_rate, name="TRAIN_DROPOUT_1")(x)
    for _ in range(GlobalFilter_layers):       # 12 global filter blocks
        x = GF_Block(patch_size, hidden_dim, mlp_ratio, ...)(x)
    x = Dropout(dropout_rate, name="TRAIN_DROPOUT_2")(x)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dropout(dropout_rate, name="TRAIN_DROPOUT_3")(x)
    output_heads = [Dense(num_classes, activation="softmax", name=f"head_{i+1}")(x)
                    for i in range(K_HEADS)]
```

**What this does:** Converts patches to tokens, applies positional encoding, stacks 12 GFNet blocks (each: LayerNorm → GlobalFilter → residual → LayerNorm → MLP → residual), pools via GlobalAveragePooling, and fans to 7 heads.  
**Why:** The 12-layer depth gives the model many rounds of frequency-domain filtering. GlobalAveragePooling summarises all token positions into a single feature vector, which is then split across 7 heads.

---

### 4.7 ViT-UNet Multi-Head Model Builder

```python
def create_vit_classifier(k_heads=7, method='with_cls_tkn', ...):
    patches         = ViT_Patches(patch_size, embed_dim=projection_dim)(x0)
    encoded_patches = ViT_PatchEncoder(num_patches, projection_dim)(patches)  # + CLS token
    encoded_patches = ViT_TransFormer_Block(transformer_layers, num_heads, ...)(encoded_patches)
    # U-Net skip connections inside ViT_TransFormer_Block
    representation  = ViT_Class_Token_Norm()(encoded_patches)  # extract CLS token
    x = Dense(512, activation=gelu)(representation)
    x = Dense(256, activation=gelu)(x)
    x = Dense(128, activation=gelu)(x)
    x = Dense(64,  activation=gelu)(x)
    output_heads = [Dense(num_classes, activation='softmax', name=f'head_{i+1}')(x)
                    for i in range(k_heads)]
```

**What this does:** Extracts sub-patches, prepends a trainable CLS token, adds learned positional embeddings, passes through 12 transformer layers with U-Net skip connections, extracts the CLS token's normalized representation, routes through a 4-layer GELU MLP, and fans to 7 heads.  
**Why:** The CLS token aggregates global patch context. U-Net residuals carry early-layer spatial detail into later abstract layers. The GELU activations (smoother than ReLU) are standard for transformer-based architectures.

---

### 4.8 Multi-Head Prediction and Evaluation

```python
def predict_multihead(model, x_data):
    y_pred_list    = model.predict(x_data, verbose=0)  # list of K arrays
    y_pred_stacked = np.stack(y_pred_list, axis=0)     # shape (K, N, C)
    y_pred_avg     = np.mean(y_pred_stacked, axis=0)   # shape (N, C)
    y_pred_argmax  = np.argmax(y_pred_avg, axis=1)     # shape (N,)
    return y_pred_argmax.reshape(-1, 1), y_pred_avg
```

**What this does:** Collects all $K=7$ softmax distributions into a $(K \times N \times C)$ array, averages over the head axis, and takes the argmax to produce hard class labels.  
**Why:** Averaging probability distributions (rather than voting over argmax labels) preserves the full predictive uncertainty information. A downstream conformal predictor can apply a threshold directly to $\bar{\mathbf{p}}$ rather than only seeing discrete votes.

---

### 4.9 Learning Rate Scheduling

```python
def _multistep_cosine_lrfn(e, steps=[100, 200, 300], lr_max, lr_min):
    if e < steps[0]:
        epoch2, epochs2 = e, steps[0]
    elif e < steps[0] + steps[1]:
        epoch2, epochs2 = e - steps[0], steps[1]
    else:
        epoch2, epochs2 = e - steps[0] - steps[1], steps[2]
    phase = math.pi * epoch2 / (epochs2 - 1)
    return (lr_max - lr_min) * 0.5 * (1.0 + math.cos(phase)) + lr_min
```

**What this does:** Implements a three-stage piecewise cosine decay — AlexNet uses $\eta_{\max}=0.02, \eta_{\min}=0.005$; GFNet uses $\eta_{\max}=6\times10^{-4}, \eta_{\min}=10^{-7}$. ViT uses a single-stage cosine decay with the same GFNet bounds.  
**Why:** Multiple restarts (via the stage resets) help models escape sub-optimal local minima. The lower bounds prevent the learning rate from collapsing to zero before training stabilizes.

---

## 5. Worked Numerical Example

### 5.0 Shared Toy Setup

We use a simplified scenario with $K_{\text{classes}} = 4$, $B = 3$ spectral bands, and patches of size $3 \times 3 \times 3$. We focus on the core algorithmic component: **multi-head probability aggregation** applied to the same test inputs for all three architectures (illustrated with synthetic head outputs).

**Calibration set** ($n = 12$ patches, 3 per class, labels zero-indexed):

| Sample | True Class | Backbone produces... |
|--------|-----------|---------------------|
| 1 | 0 | (used to train heads; detailed in Step A) |
| ... | ... | ... |

For compactness, this example demonstrates the **inference aggregation** algorithm — the step that is architecturally identical across all three models — by hand with synthetic multi-head outputs.

---

### 5.1 Multi-Head Aggregation: Worked Inference Example

**Notation:** 4 classes (0, 1, 2, 3), $K = 7$ heads.

**Test sample: Easy case** — true class = 1, model confident.

Suppose the 7 head softmax outputs for a single test patch are:

| Head | $\hat{p}_0$ | $\hat{p}_1$ | $\hat{p}_2$ | $\hat{p}_3$ |
|------|-----------|-----------|-----------|-----------|
| 1 | 0.05 | **0.82** | 0.08 | 0.05 |
| 2 | 0.03 | **0.85** | 0.07 | 0.05 |
| 3 | 0.04 | **0.80** | 0.10 | 0.06 |
| 4 | 0.06 | **0.78** | 0.11 | 0.05 |
| 5 | 0.05 | **0.83** | 0.09 | 0.03 |
| 6 | 0.04 | **0.81** | 0.10 | 0.05 |
| 7 | 0.03 | **0.84** | 0.09 | 0.04 |

**Step A — Average across K = 7 heads:**

$$\bar{p}_0 = \frac{0.05+0.03+0.04+0.06+0.05+0.04+0.03}{7} = \frac{0.30}{7} \approx 0.043$$

$$\bar{p}_1 = \frac{0.82+0.85+0.80+0.78+0.83+0.81+0.84}{7} = \frac{5.73}{7} \approx 0.819$$

$$\bar{p}_2 = \frac{0.08+0.07+0.10+0.11+0.09+0.10+0.09}{7} = \frac{0.64}{7} \approx 0.091$$

$$\bar{p}_3 = \frac{0.05+0.05+0.06+0.05+0.03+0.05+0.04}{7} = \frac{0.33}{7} \approx 0.047$$

**Averaged distribution:** $\bar{\mathbf{p}} = [0.043, 0.819, 0.091, 0.047]$

**Prediction:** $\hat{y} = \arg\max\ \bar{\mathbf{p}} = 1$ ✓ (matches true class)

---

**Test sample: Borderline case** — true class = 2, model less confident.

| Head | $\hat{p}_0$ | $\hat{p}_1$ | $\hat{p}_2$ | $\hat{p}_3$ |
|------|-----------|-----------|-----------|-----------|
| 1 | 0.10 | 0.25 | **0.45** | 0.20 |
| 2 | 0.12 | 0.30 | **0.38** | 0.20 |
| 3 | 0.08 | 0.28 | **0.42** | 0.22 |
| 4 | 0.15 | 0.35 | **0.32** | 0.18 |
| 5 | 0.09 | 0.27 | **0.44** | 0.20 |
| 6 | 0.11 | 0.32 | **0.36** | 0.21 |
| 7 | 0.10 | 0.29 | **0.41** | 0.20 |

**Averaged distribution:**

$$\bar{p}_1 \approx \frac{0.25+0.30+0.28+0.35+0.27+0.32+0.29}{7} = \frac{2.06}{7} \approx 0.294$$

$$\bar{p}_2 \approx \frac{0.45+0.38+0.42+0.32+0.44+0.36+0.41}{7} = \frac{2.78}{7} \approx 0.397$$

**Averaged distribution:** $\bar{\mathbf{p}} \approx [0.107, 0.294, 0.397, 0.202]$

**Prediction:** $\hat{y} = 2$ ✓ — correct, but with lower confidence (0.397 vs 0.819). A conformal predictor with threshold $\tau = 0.35$ would produce a prediction set $\{1, 2, 3\}$, reflecting genuine uncertainty.

---

**Test sample: Ambiguous case** — true class = 0, heads strongly disagree.

| Head | $\hat{p}_0$ | $\hat{p}_1$ | $\hat{p}_2$ | $\hat{p}_3$ |
|------|-----------|-----------|-----------|-----------|
| 1 | **0.38** | 0.30 | 0.20 | 0.12 |
| 2 | 0.28 | **0.35** | 0.22 | 0.15 |
| 3 | **0.35** | 0.32 | 0.18 | 0.15 |
| 4 | 0.25 | **0.38** | 0.22 | 0.15 |
| 5 | **0.40** | 0.28 | 0.18 | 0.14 |
| 6 | 0.27 | **0.36** | 0.23 | 0.14 |
| 7 | **0.33** | 0.31 | 0.22 | 0.14 |

**Averaged distribution:**

$$\bar{p}_0 \approx \frac{0.38+0.28+0.35+0.25+0.40+0.27+0.33}{7} = \frac{2.26}{7} \approx 0.323$$

$$\bar{p}_1 \approx \frac{0.30+0.35+0.32+0.38+0.28+0.36+0.31}{7} = \frac{2.30}{7} \approx 0.329$$

**Averaged distribution:** $\bar{\mathbf{p}} \approx [0.323, 0.329, 0.207, 0.141]$

**Prediction:** $\hat{y} = 1$ ✗ — incorrect (true class 0). The heads are split between class 0 and class 1, and the average merely reflects this ambiguity. A conformal prediction set at $1-\alpha = 0.90$ would include at least $\{0, 1\}$, correctly flagging uncertainty. This is precisely the calibration property that makes multi-head averaging useful for downstream uncertain-aware inference.

---

### 5.2 Summary Table

| Test Sample | True Class | Predicted Class | Max $\bar{p}$ | Set $\{c : \bar{p}_c > 0.30\}$ | Covered? |
|-------------|-----------|----------------|--------------|-------------------------------|---------|
| Easy | 1 | 1 | 0.819 | {1} | ✓ |
| Borderline | 2 | 2 | 0.397 | {2} | ✓ |
| Ambiguous | 0 | 1 | 0.329 | {0, 1} | ✓ (if set) |

**Observation:** The multi-head ensemble correctly identifies uncertainty by producing flatter averaged distributions on hard samples. A single-head model would suppress this signal; the multi-head average preserves it.

---

### 5.3 Cross-Architecture Comparison

| Architecture | Head Diversity Source | Inductive Bias | Expected Strength |
|---|---|---|---|
| AlexNet-CNN | DSE head initialization + channel-shift dropout | Local convolution, translation equivariance | Texture & edge features |
| GFNet | DSE head initialization + global frequency filter | Frequency-domain global receptive field | Long-range spectral correlations |
| ViT-UNet | DSE head initialization + multi-head attention | Sequence self-attention + skip residuals | Context-aware patch reasoning |

Combining all three in a conformal calibration pipeline means the averaged probabilities reflect not only within-architecture disagreement (across 7 heads) but also cross-architecture epistemic variation — a richer uncertainty signal than any single model family can provide.

---

## 6. References

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. "ImageNet Classification with Deep Convolutional Neural Networks." *Advances in Neural Information Processing Systems 25 (NeurIPS 2012)*. [Link](https://dl.acm.org/doi/10.5555/2999134.2999257)

[2] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *ICLR 2021*. [Link](https://arxiv.org/abs/2010.11929)

[3] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. "Global Filter Networks for Image Classification." *Advances in Neural Information Processing Systems 34 (NeurIPS 2021)*. [Link](https://arxiv.org/abs/2107.00645)

[4] Valdenegro-Toro, M. "Deep Sub-Ensembles for Fast Uncertainty Estimation in Image Classification." *NeurIPS Bayesian Deep Learning Workshop 2019*. [Link](https://arxiv.org/abs/1910.08168)

[5] Lakshminarayanan, B., Pritzel, A., & Blundell, C. "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." *NeurIPS 2017*. [Link](https://arxiv.org/abs/1612.01474)

[6] Paoletti, M. E., Haut, J. M., Plaza, J., & Plaza, A. "Deep learning classifiers for hyperspectral imaging: A review." *ISPRS Journal of Photogrammetry and Remote Sensing, 2019*. [Link](https://www.sciencedirect.com/science/article/pii/S0924271619302187)

[7] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. "Attention is All You Need." *NeurIPS 2017*. [Link](https://arxiv.org/abs/1706.03762)
