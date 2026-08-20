# Conformal Prediction Uncertainty Quantification (Conformal Regularised Models)

> **One-line description:** Evaluates five Conformal Prediction (CP) methods on multispectral image classifiers trained with a conformal regulariser, generating statistically guaranteed prediction sets and uncertainty maps.

---

## 1. Overview & Intuition

Post-hoc Conformal Prediction converts heuristic model probabilities into rigorous prediction sets that contain the true class with a user-specified probability (e.g., $1 - \alpha = 95\%$). 

Because the underlying models (AlexNet, GFNet, ViT-UNet) in this specific pipeline were trained with a **conformal regulariser**, their non-target probabilities are structurally suppressed. This means CP methods can generate much tighter (smaller) sets than they would on standard models, vastly reducing the number of "uncertain" multi-class predictions while preserving the strict mathematical coverage guarantee.

---

## 2. Mathematical Framework

The notebook implements five CP approaches:

### 2.1 Split Conformal Prediction (SplitCP)
Computes a single global threshold $\hat{q}$ from the calibration set's non-conformity scores ($1 - \hat{p}_{y}$). Prediction sets include all classes $c$ where $\hat{p}_c \geq (1 - \hat{q})$.

### 2.2 Class-Conditional CP (CcCP)
Computes a separate $\hat{q}_c$ for each class $c$. This ensures that coverage is maintained marginally for every individual class, addressing class imbalances.

### 2.3 Rank Calibrated Class-Conditional CP (RC3P)
Searches for optimal truncated rank limits and per-class thresholds to minimise the average set size while strictly maintaining class-conditional coverage.

### 2.4 Cluster-Conditioned CP (ClusterCP)
Uses KMeans ($k=4$) to cluster the raw image features. It then computes a distinct threshold $\hat{q}_k$ for each cluster, allowing the model to be more conservative in difficult feature-space regions and sharper in easy regions.

### 2.5 Regularized Adaptive Predictive Sets (RAPS)
Sorts probabilities in descending order and accumulates them until the cumulative mass exceeds a threshold. A size penalty ($k_{reg}=1, \lambda=0.01$) is applied to discourage overly large sets.

---

## 3. Algorithm

1. **Load pre-trained models** (AlexNet, GFNet, ViT-UNet).
2. **Split data** into calibration (50% of test pool) and evaluation sets.
3. **Compute Thresholds:** For each model and each CP method, calculate the $\hat{q}$ thresholds on the calibration set.
4. **Evaluate Metrics:** Calculate empirical coverage, average set size, and singleton/empty set rates on the evaluation set.
5. **Full-Scene Inference:** Generate pixel-wise prediction sets for the entire multispectral image.
6. **Visualise & Export:** Create spatial maps highlighting "certain" (singleton set) vs "uncertain" (multi-class set) regions and save metrics to Excel.

---

## 4. Implementation Walkthrough

### 4.1 Safely Loading Models (Section 6.0)
```python
def load_models(model_files, custom_objects):
    ...
    # Verifies that models are loaded from trusted paths before applying unsafe deserialization 
    # to handle custom layers like GFNet's GlobalFilterLayer.
```

### 4.2 Split Conformal Prediction (Section 8.1)
```python
calib_scores = 1.0 - prob_cal[np.arange(len(y_cal)), y_cal]
q_hat = conformal_qhat(calib_scores, alpha)
pred_sets_eval = prob_eval >= (1.0 - q_hat)
```
**What this does:** Extracts the probability assigned to the ground-truth class for all calibration samples, calculates the $(1-\alpha)$ quantile, and forms prediction sets for new data by thresholding.

### 4.3 RAPS implementation (Section 8.5)
```python
reg_penalty = lam_reg * np.maximum(0, np.arange(1, n_classes + 1) - k_reg)
cal_scores = cum_prob[np.arange(num_cal), cal_ranks[np.arange(num_cal), y_cal] - 1] + reg_penalty[cal_ranks[np.arange(num_cal), y_cal] - 1]
```
**Why:** RAPS is particularly synergistic with the conformal regulariser used during training, as both explicitly target and penalise the distribution of probability mass across long tails of incorrect classes.
