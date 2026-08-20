# Multispectral Patch-Based Classification Pipeline with Class-Balanced Focal Loss: Theory & Implementation Summary

> **One-line description:** A comparative deep-learning framework that trains three architectures — AlexNet-CNN, Global Filter Network (GFNet), and a ViT with U-Net skip connections — using a Class-Balanced Focal Loss to handle severe class imbalance and hard-to-classify pixels, producing pixel-wise land-cover classification maps.

---

## 1. Overview & Intuition

Like the baseline framework, this pipeline extracts $9 \times 9 \times 6$ multispectral patches to provide spatial context to three models: **AlexNet** (local convolutional features), **GFNet** (frequency-domain global filtering), and **ViT-UNet** (self-attention with multi-scale skip connections). 

However, real-world remote sensing datasets often suffer from severe class imbalance and spectral overlap, where standard cross-entropy loss causes the model to over-focus on abundant classes and easy-to-classify pixels. To combat this, this notebook introduces the **Class-Balanced Focal Loss**. It dynamically scales the loss based on two components:
1. **Focal down-weighting ($\gamma$)**: Reduces the loss contribution of easy, high-confidence samples, forcing the optimiser to focus on hard, ambiguous pixels.
2. **Class-Balanced weighting ($\alpha$)**: Replaces a flat scalar weight with a per-class weight vector derived from the *effective number of samples*. This ensures rare classes receive a proportionally larger training signal without overwhelming the loss gradient.

The resulting models produce better-calibrated confidence scores, especially for underrepresented classes, which are evaluated via Brier Score and Expected Calibration Error (ECE) before being used for dense scene inference.

---

## 2. Mathematical Framework

### 2.1 Problem Setup and Patch Extraction

Let the multispectral scene be $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$. Each band is normalised independently via min-max scaling to $[0,1]$ to prevent bands with large dynamic ranges from dominating.
For every labelled pixel in the reference map $\mathcal{Y}$, a $P \times P$ spatial neighbourhood ($P=9$) is extracted to form a patch $\mathbf{X}^{(i)} \in \mathbb{R}^{P \times P \times B}$.

### 2.2 Effective Number of Samples

Rather than weighting classes by the inverse of their raw frequency, the pipeline computes the *effective number of samples* $E_{n_c}$ for each class $c$. This accounts for the diminishing marginal benefit of additional data points due to data overlap:

$$E_{n_c} = \frac{1 - \beta^{n_c}}{1 - \beta}$$

**Where:**
- $n_c$ — raw number of training samples for class $c$
- $\beta = 0.999$ — hyperparameter controlling the asymptote of the effective volume (closer to 1 means slower saturation)

The raw class weight is the inverse of the effective number, renormalised so the mean weight across all classes is $1$:
$$\alpha_c \propto \frac{1}{E_{n_c}}$$
This yields the fixed, per-class vector `CB_FOCAL_ALPHA`.

### 2.3 Class-Balanced Focal Loss

The standard Focal Loss is modified by the per-class effective-sample weights. For a sample with true class $c$ and predicted probability vector $\hat{\mathbf{p}}$, the loss is:

$$\mathcal{L}_{\text{CB-Focal}} = -\alpha_c \cdot (1 - \hat{p}_c)^\gamma \cdot \log(\hat{p}_c)$$

**Where:**
- $\hat{p}_c$ — the model's predicted probability for the true class $c$
- $\gamma = 2.0$ — focusing parameter that down-weights well-classified examples
- $\alpha_c$ — the class-balanced weight computed above

For GFNet and ViT-UNet, this loss is applied over the softmax probabilities. For AlexNet, a sparse variant computes the same formula directly from integer targets.

### 2.4 The Three Architectures

- **AlexNet-style CNN**: Five 3×3 convolutional layers extract local spatial features, followed by a max-pooling layer and four dense layers with dropout ($p=0.25$).
- **GFNet**: Non-overlapping 3×3 tokens are passed to a Global Filter Layer that applies a 2-D discrete Fourier transform (DFT), multiplies element-wise by learnable complex weights, and applies the inverse DFT. This achieves $O(N \log N)$ all-to-all spatial interactions.
- **ViT-UNet**: A Vision Transformer processes flattened 3×3 tokens with a prepended [CLS] token. Symmetrical skip connections (U-Net style) bridge early encoder layers directly to later decoder layers, preserving local multi-scale context that attention might otherwise blur.

---

## 3. Algorithm

**Input:** Multispectral scene $\mathcal{X}$, label map $\mathcal{Y}$, $P=9$, $K$ classes, CB-Focal configs ($\beta=0.999, \gamma=2.0$).
**Output:** Trained models, classification metrics, dense scene prediction maps.

1. **Normalise** each spectral band independently to $[0, 1]$.
2. **Extract patches** of $9 \times 9 \times 6$ for each labelled pixel.
3. **Compute Class Weights**: Calculate $E_{n_c}$ and $\alpha_c$ on the training split.
4. **Split Data**: Create stratified train (75%), val (20% of train), and test (25%) sets.
5. **Build Models**: Instantiate AlexNet, GFNet, and ViT-UNet architectures.
6. **Compile**: Use AdamW (GFNet/ViT) or Adagrad (AlexNet) with the `categorical_focal_loss` / `sparse_categorical_focal_loss`.
7. **Train** for 100 epochs, tracking best validation metrics.
8. **Evaluate**: Compute accuracy, F1 scores, Brier score, and ECE on the test set.
9. **Visualise & Export**: Slide the models over the full spatial grid to produce pixel-wise classification maps. Save maps and metrics.

---

## 4. Implementation Walkthrough

### 4.1 Class-Balanced Alpha Computation (Section 3.3)
```python
class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
effective_num  = 1.0 - np.power(CB_BETA, class_counts)
cb_alpha_raw   = (1.0 - CB_BETA) / np.clip(effective_num, 1e-12, None)
cb_alpha_norm  = cb_alpha_raw / cb_alpha_raw.sum() * num_classes
CB_FOCAL_ALPHA = tf.constant(cb_alpha_norm, dtype=tf.float32)
```
The raw inverse-effective-sample weights are scaled so their mean is exactly 1, preventing the overall learning rate from being implicitly scaled down.

### 4.2 Focal Loss Implementation (Section 5.1)
```python
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        eps    = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight        = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss_fn
```
Notice how `alpha` here accepts the tensor `CB_FOCAL_ALPHA`. For each pixel, the vector multiplication `weight * cross_entropy` scales the loss by the pre-computed class weight and dynamically squashes the gradient using $(1 - y_{pred})^\gamma$.

### 4.3 Training Fallback and Inference (Sections 6.0 & 8.0)
Like the baseline, training employs an automatic `ResourceExhaustedError` fallback that scales down model dimensions (e.g., GFNet `hidden_dim` 512 → 384) to survive GPU limits on Colab. Scene inference operates strictly row-by-row to bound memory consumption tightly.
