# Conformal Prediction for Class-Balanced Focal Loss Models: Theory & Implementation Summary

> **One-line description:** A post-hoc uncertainty quantification framework applying five conformal prediction methods to three architectures (AlexNet, GFNet, ViT-UNet) trained with Class-Balanced Focal Loss, generating statistically rigorous, valid prediction sets for a multispectral remote sensing scene.

---

## 1. Overview & Intuition

Standard neural network softmax scores are notoriously uncalibrated. While the underlying models in this folder were trained with Class-Balanced Focal Loss to mitigate imbalance and improve calibration on hard examples, they still do not provide formal statistical guarantees.

**Conformal Prediction (CP)** solves this by converting point predictions into *prediction sets*. Given a user-defined error rate $\alpha = 0.05$, CP guarantees that the true class is contained within the prediction set with a probability of at least $1 - \alpha$ ($95\%$).

This notebook implements five distinct conformal prediction techniques:
1. **Split Conformal Prediction (SplitCP)** — A global threshold approach.
2. **Class-Conditional Conformal Prediction (CcCP)** — Evaluates specific thresholds per class to guarantee coverage for rare classes.
3. **Rank-Calibrated Class-Conditional CP (RC3P)** — Balances rank-based truncation with class-conditional thresholds to shrink set sizes.
4. **Clustered Conformal Prediction (ClCP)** — Clusters classes by latent feature embeddings and applies local cluster-level thresholds.
5. **Regularised Adaptive Prediction Sets (RAPS)** — Sorts probabilities and adds classes until a cumulative threshold is met, regularised to discourage unnecessarily large sets.

The notebook reads the pre-trained weights, runs inference across a calibration set to learn non-conformity thresholds, applies those thresholds to an evaluation set, and maps the resulting sets onto the full remote-sensing image to distinguish "certain" (singleton set) from "uncertain" (multi-element set) pixels.

---

## 2. Mathematical Framework

Let $\alpha \in (0, 1)$ be the target error rate (e.g., $0.05$). A non-conformity scoring function $s(x, y)$ measures how "unusual" it is for $y$ to be the true label for $x$. We compute scores on a held-out calibration set of size $N$ and define $\hat{q}$ as the $\lceil (N+1)(1-\alpha) \rceil / N$ empirical quantile of the scores.

### 2.1 Split Conformal Prediction (SplitCP)
- **Score:** $s(x, y) = 1 - \hat{p}_y(x)$, where $\hat{p}_y(x)$ is the softmax probability for the true class.
- **Set Construction:** $\mathcal{C}(x) = \{ k : 1 - \hat{p}_k(x) \leq \hat{q} \}$.
- **Pros/Cons:** Simple and guarantees marginal coverage, but can severely under-cover hard or rare classes.

### 2.2 Class-Conditional Conformal Prediction (CcCP)
- **Mechanic:** Instead of one global quantile, compute a separate $\hat{q}_c$ for each class $c$ using only calibration samples where the true label is $c$.
- **Set Construction:** $\mathcal{C}(x) = \{ k : 1 - \hat{p}_k(x) \leq \hat{q}_k \}$.
- **Pros/Cons:** Guarantees coverage independently for every class, but prediction sets can become overly large due to noisy thresholds for rare classes.

### 2.3 Rank-Calibrated Class-Conditional CP (RC3P)
- **Mechanic:** Computes a top-$K$ accuracy matrix across all classes. It searches for a rank truncation limit $k_c$ for each class $c$ such that the error rate before $k_c$ satisfies a modified $\alpha$. The non-conformity threshold $\hat{q}_c$ is then computed strictly over samples that rank within $k_c$.
- **Set Construction:** A class is included only if it passes the threshold *and* its probability rank is $\leq k_c$.
- **Pros/Cons:** Maintains high class-conditional coverage while significantly reducing the average set size compared to standard CcCP.

### 2.4 Clustered Conformal Prediction (ClCP)
- **Mechanic:** Extracts penultimate-layer feature embeddings from the calibration set and runs K-Means ($k=4$) to group structurally similar classes. A separate threshold $\hat{q}_m$ is computed for each cluster $m$.
- **Set Construction:** $\mathcal{C}(x) = \{ k : 1 - \hat{p}_k(x) \leq \hat{q}_{cluster(k)} \}$.
- **Pros/Cons:** Balances the robustness of global SplitCP with the specificity of CcCP by sharing statistical strength among similar classes.

### 2.5 Regularised Adaptive Prediction Sets (RAPS)
- **Mechanic:** Sorts the predicted probabilities in descending order: $\pi_1, \pi_2, \ldots, \pi_K$. The score is the cumulative probability mass up to the true class, plus a penalty $\lambda \max(rank - k_{reg}, 0)$.
- **Set Construction:** Adds classes in descending order of probability until the sum of their probabilities (plus the rank penalty) exceeds the globally calibrated threshold $\hat{q}_{RAPS}$.
- **Pros/Cons:** Adapts naturally to the instance-level difficulty (flat distributions yield large sets; peaked distributions yield singletons), with the regularisation preventing excessively large sets.

---

## 3. Algorithm

**Input:** Pre-trained models (AlexNet, GFNet, ViT-UNet), multispectral scene, train/calib/eval split configuration, target $\alpha = 0.05$.
**Output:** Extensive evaluation reports and dense full-scene maps differentiating "Certain" vs "Uncertain" pixels.

1. **Load Models & Data:** Instantiate `PatchExtractor`, `GlobalFilterLayer`, etc., as `custom_objects` and deserialize Keras `.keras` weights safely.
2. **Split Data:** Partition the non-training pool into an equal 50/50 Calibration and Evaluation split.
3. **Cache Probabilities:** To save runtime, run dense prediction over the entire spatial grid for each model once, caching the resulting $(H, W, K)$ probability arrays.
4. **For Each Model:**
   - **Run SplitCP, CcCP, RC3P, ClCP, RAPS:** 
     - Compute specific $\hat{q}$ quantiles from the calibration set.
     - Evaluate valid sets on the evaluation set, recording average set size, marginal coverage, and per-class coverage.
   - **Map Construction (Except RAPS):** Apply the computed thresholds to the full scene array to produce a prediction set mask. Visualise singleton sets as the predicted class, and multi-element (or empty) sets as a masked "Uncertain" color.
5. **Generate Visuals:** Produce comparative bar charts tracking empirical coverage and set sizes.
6. **Export:** Combine summary DataFrames, per-class tables, spatial map images, and distribution plots into a comprehensive multi-sheet Excel workbook.

---

## 4. Implementation Walkthrough

### 4.1 Reliable Quantile Computation (Section 7.1)
```python
def conformal_qhat(scores, alpha):
    n       = len(scores)
    if n == 0: return 1.0
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return safe_quantile(scores, q_level)
```
Uses the standard conformal prediction formula with a small-sample correction `(n + 1) / n`. `safe_quantile` uses the `higher` interpolation method to ensure exact marginal coverage bounds.

### 4.2 RC3P Threshold and Rank Truncation Search (Section 8.3)
```python
meets_thresh_eval = prob_eval >= (1.0 - q_hats.reshape(1, -1))
meets_rank_eval   = eval_ranks <= np.array(suit_indices).reshape(1, -1)
pred_sets_eval    = meets_thresh_eval & meets_rank_eval
```
RC3P enforces a dual constraint: the probability must exceed the class-conditional threshold $\hat{q}_c$, and the class must fall within the top $k_c$ probability ranks. This sharply cuts down the inclusion of low-probability noise classes.

### 4.3 Feature Embedding Extraction for Clustered CP (Section 8.4)
```python
feat_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
emb_cal    = feat_model.predict(x_cal)
...
global_mean = emb_cal.mean(axis=0)
km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
cluster_assignments = km.fit_predict(class_means)
```
The algorithm dynamically peels back the classification head (the last dense layer) to extract semantic embeddings. K-Means is then applied to the average feature vector of each class to form semantic clusters.

### 4.4 RAPS Score Computation (Section 8.5)
```python
def raps_score_single(prob_row, true_label, lam=0.01, k_reg=1):
    order      = np.argsort(prob_row)[::-1]
    rank       = int(np.where(order == true_label)[0][0])
    cumulative = float(np.sum(prob_row[order[:rank]]))
    penalty    = float(lam) * max(rank - int(k_reg), 0)
    return cumulative + penalty
```
The score is derived from the sorted probabilities up to the true class index, applying a linear penalty for classes ranked below $k_{reg}$. This guarantees that the threshold incorporates a soft cap on set size.
