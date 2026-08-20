# Multispectral Patch-Based Classification Pipeline: Conformal Regularisation

> **One-line description:** A comparative deep-learning framework that trains three architectures (AlexNet-CNN, GFNet, ViT-UNet) on spatially extracted patches from a 6-band multispectral image, augmented with a differentiable conformal-training regulariser to optimise downstream prediction sets.

---

## 1. Overview & Intuition

Standard cross-entropy loss often produces overconfident models. While label smoothing mitigates this, it doesn't explicitly penalise the probability mass spread across incorrect classes. 

This notebook modifies the baseline training pipeline by introducing a **Conformal-training regulariser** (following Stutz et al., 2022). By explicitly penalising the model for placing meaningful probability mass on non-target classes during training, the resulting predictions yield significantly tighter Split/RAPS conformal sets downstream without requiring complex re-weighting schemes.

The three architectures (AlexNet, GFNet, ViT-UNet) remain structurally identical to the baseline, but the loss function is adapted to anticipate post-hoc uncertainty quantification.

---

## 2. Mathematical Framework

### 2.1 Problem Setup and Patch Extraction
The problem formulation remains identical to the baseline: patches of size $P=9$ are extracted from the $H \times W \times B$ scene, and spectral bands are individually min-max normalised.

### 2.2 The Conformal Regularised Loss Function

Instead of standard categorical cross-entropy, the models are trained using a **categorical conformal loss**:

$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(y, \hat{\mathbf{p}}) + \lambda_{\text{reg}} \sum_{c \neq y} \hat{p}_c $$

**Where:**
- $\mathcal{L}_{\text{CE}}$ — standard cross-entropy between predictions and targets.
- $\lambda_{\text{reg}} = 0.1$ — weight of the set-size penalty.
- $\sum_{c \neq y} \hat{p}_c$ — the false-class probability mass.

This regulariser explicitly minimises the size of the prediction sets that will be constructed downstream, as methods like RAPS accumulate probability mass to achieve coverage. By pushing down the mass of all false classes, the model avoids generating long "tails" of uncertain probabilities.

### 2.3 Architectures & Calibration Metrics
- **AlexNet:** Uses `sparse_categorical_conformal_loss` with integer targets.
- **GFNet & ViT:** Use `categorical_conformal_loss` with one-hot targets.
- **Metrics:** Brier Score and Expected Calibration Error (ECE) are tracked.

---

## 3. Algorithm

1. **Normalise** and **Extract patches** from the scene.
2. **Split** into train, validation, and test sets.
3. **Build model**: AlexNet, GFNet, or ViT-UNet.
4. **Compile** using the custom conformal loss functions (`lambda_reg=0.1`).
5. **Train** for 100 epochs, saving the best models.
6. **Evaluate** accuracy, NLL, Brier score, and ECE.
7. **Dense scene inference**: generate full prediction maps.

---

## 4. Implementation Walkthrough

### 4.1 Conformal-training regularizer (Section 5.1)

```python
def categorical_conformal_loss(lambda_reg=0.1):
    def loss_fn(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        false_class_mass = tf.reduce_sum(y_pred * (1.0 - y_true), axis=-1)
        return ce + lambda_reg * false_class_mass
    return loss_fn
```

**What this does:** It isolates the sum of probabilities assigned to all incorrect classes (`1.0 - y_true` masks out the correct class). This sum is multiplied by `lambda_reg` and added to the standard cross-entropy loss.

**Why:** It acts as a differentiable surrogate for conformal prediction set size. It directly tells the optimiser that leaving residual mass on wrong classes is bad, even if the correct class has the highest probability.

---

## 5. Worked Numerical Example

Consider a sample with $K=4$ classes. The true class is $y = 1$ (one-hot: `[0, 1, 0, 0]`).
The model predicts $\hat{\mathbf{p}} = [0.10, 0.70, 0.15, 0.05]$.

**1. Cross-Entropy Component:**
$$ \mathcal{L}_{\text{CE}} = -\ln(0.70) \approx 0.356 $$

**2. Conformal Regulariser Component ($\lambda_{\text{reg}} = 0.1$):**
The false class probabilities are $0.10, 0.15, 0.05$. Their sum is $0.30$.
$$ \text{Penalty} = 0.1 \times 0.30 = 0.03 $$

**3. Total Loss:**
$$ \mathcal{L}_{\text{total}} = 0.356 + 0.03 = 0.386 $$

In subsequent epochs, the optimiser will push the 0.10, 0.15, and 0.05 values down further, sharpening the distribution around the true class and making post-hoc conformal sets smaller.
