# Multispectral Patch-Based Classification Pipeline with Focal Loss: Theory & Implementation Summary

> **One-line description:** A comparative deep-learning framework that trains three architectures — AlexNet-CNN, Global Filter Network (GFNet), and a ViT with U-Net skip connections — on spatially extracted patches from a 6-band multispectral image using Focal Loss to produce pixel-wise land-cover classification maps.

---

## 1. Overview & Intuition

Remote sensing images contain rich spatial and spectral information at each pixel. Patch-based classification embeds each pixel in its local spatial context by centring a fixed-size neighbourhood window around it and feeding that window — a *patch* — to a classifier. 

The pipeline in this notebook operates on a 6-band multispectral scene. Every labelled pixel becomes one training sample: a 9 × 9 × 6 tensor that captures the pixel's neighbourhood across all bands. 

Three architectures are compared: **AlexNet** (classical CNN), **GFNet** (Global Filter Network in the 2-D frequency domain), and **ViT-UNet** (Vision Transformer with multi-head self-attention and U-Net-style skip connections).

A major deviation from the baseline training is the use of **Focal Loss** instead of standard cross-entropy. Focal Loss addresses class imbalance and difficult-to-classify pixels by down-weighting the loss contribution of well-classified ("easy") examples. This forces the optimizer to focus on harder, ambiguous, or minority-class pixels.

---

## 2. Mathematical Framework

### 2.1 Focal Loss

Focal Loss modifies standard Cross-Entropy (CE) by introducing a modulating factor $(1 - p_t)^\gamma$ and a balancing weight $\alpha$:

$$\text{FL}(p_t) = -\alpha (1 - p_t)^\gamma \log(p_t)$$

**Where:**
- $p_t$ — the model's estimated probability for the true class.
- $\gamma \geq 0$ — the focusing parameter. In this notebook, $\gamma = 2.0$. It smoothly adjusts the rate at which easy examples are down-weighted.
- $\alpha \in [0, 1]$ — a balancing factor. In this notebook, $\alpha = 0.25$.

For the categorical version (used by GFNet and ViT):
$$\mathcal{L}_{\text{focal\_categorical}} = \sum_{c=1}^{K} -\alpha (1 - \hat{p}_c)^\gamma \cdot y_c \cdot \log(\hat{p}_c)$$
where $y_c$ is the one-hot encoded ground truth.

### 2.2 Patch Extraction & Architecture
- **Input:** Tensor $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$. Bands are min-max normalised independently.
- **Patch Extraction:** For each labelled pixel, a $P \times P \times B$ patch (where $P=9$) is extracted using edge-padding.
- **AlexNet:** Convolutional layers followed by dense layers.
- **GFNet:** Replaces self-attention with a Global Filter layer applying a 2-D DFT.
- **ViT-UNet:** Standard Transformer encoder blocks with symmetric U-Net skip connections.

---

## 3. Algorithm

1. **Normalise** each spectral band independently to $[0, 1]$.
2. **Extract patches** of size 9 × 9 × 6 for labelled pixels.
3. **Split** into train, validation, and test sets.
4. **Build model**: AlexNet / GFNet / ViT-UNet.
5. **Compile** using the custom Focal Loss functions (`sparse_categorical_focal_loss` for AlexNet, `categorical_focal_loss` for GFNet/ViT).
6. **Train** for 100 epochs, tracking metrics and saving best checkpoints.
7. **Evaluate** on the test set, computing accuracy, kappa, macro-F1, Brier score, and ECE.
8. **Dense scene inference** over the full image to generate prediction maps.

---

## 4. Implementation Walkthrough

### 4.1 Custom Focal Loss Implementation

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
*What this does:* It calculates the cross-entropy and scales it by the dynamic weight based on the prediction confidence. If $p_t$ is high, the weight approaches 0, reducing the loss for that sample.

### 4.2 Model Compilation
AlexNet uses `sparse_categorical_focal_loss` combined with Adagrad optimization, while GFNet and ViT use `categorical_focal_loss` with an AdamW optimizer and cosine learning-rate decay.
