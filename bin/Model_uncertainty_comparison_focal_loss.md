# Model Uncertainty Comparison (Focal Loss Models): Theory & Implementation

> **One-line description:** A pipeline applying post-hoc Conformal Prediction (CP) methods to calibrate and quantify predictive uncertainty for focal-loss-trained multispectral image classifiers.

---

## 1. Overview & Intuition

Standard softmax outputs from deep neural networks are often poorly calibrated. Conformal Prediction (CP) provides a rigorous mathematical framework to convert heuristic uncertainty measures into statistically valid prediction sets. For a given error rate $\alpha$, CP guarantees that the true class is contained in the prediction set with probability at least $1 - \alpha$. 

This notebook loads models trained using **Focal Loss** (AlexNet, GFNet, ViT-UNet) and applies multiple CP methods:
1. **Split Conformal Prediction (SplitCP)**
2. **Class-Conditional Conformal Prediction (CcCP)**
3. **Rank Calibrated Class-Conditional CP (RC3P)**

---

## 2. Mathematical Framework

### 2.1 Split Conformal Prediction
A calibration set is used to compute non-conformity scores.
$$s_i = 1 - \hat{p}_{y_i}(x_i)$$
The threshold $\hat{q}$ is computed as the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile of $\{s_1, \dots, s_n\}$.
Prediction sets for new samples are formed by including all classes $c$ where $1 - \hat{p}_c(x) \leq \hat{q}$.

### 2.2 Class-Conditional CP
Instead of a single global threshold, CcCP computes a separate threshold $\hat{q}_c$ for each class, providing valid coverage margins for minority classes, which is crucial for imbalanced remote sensing datasets.

---

## 3. Algorithm

1. **Load Data & Models:** Load the full image, patch dataset, and the pre-trained focal loss models.
2. **Split Data:** Partition the held-out test data into calibration and evaluation subsets.
3. **Apply Conformal Methods:** For each model and method, compute quantiles/thresholds using the calibration set.
4. **Evaluate Sets:** Generate prediction sets on the evaluation data. Compute empirical coverage, average set size, and singleton rates.
5. **Full-Scene Mapping:** Run inference on the full image and construct spatial uncertainty maps highlighting pixels with multi-class prediction sets.
6. **Export:** Save plots and summary metrics (Excel/CSV).

---

## 4. Implementation Walkthrough

### 4.1 CP Threshold Computation
```python
def conformal_qhat(scores, alpha):
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return safe_quantile(scores, q_level)
```
*What this does:* It dynamically finds the conformal quantile $\hat{q}$ such that $1-\alpha$ proportion of calibration scores fall below it.

### 4.2 Full Scene Mapping
The notebook extracts patches across the entire scene and checks prediction thresholds per pixel. Pixels where the set size $>1$ are masked as "Uncertain" to visualize model doubt spatially.
