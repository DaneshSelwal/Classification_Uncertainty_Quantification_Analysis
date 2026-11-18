# Conformal Prediction for Multi-Class Remote Sensing Classification: Theory & Implementation Summary

> **One-line description:** A comparative framework applying five conformal prediction methods to multispectral image patch classification, providing distribution-free coverage guarantees for three deep learning models (AlexNet, GFNet, ViT-UNet) across a 7-class land-cover task.

---

## 1. Overview & Intuition

Standard deep learning classifiers output a single predicted class for each input, but they offer no principled guarantee about *how often* that prediction is correct. In safety-critical or high-stakes domains — such as remote sensing land-cover mapping — we need more than a point prediction. We need to know: *which set of labels is plausible for this input?* And crucially, we need a formal guarantee that the true label is in that set with at least a pre-specified probability.

Conformal Prediction (CP) is a post-hoc framework that wraps any pre-trained classifier and returns a **prediction set** — a subset of class labels — that is guaranteed to contain the true label with probability at least $1 - \alpha$, where $\alpha$ is a user-specified error tolerance. This guarantee holds without any assumptions about the classifier architecture or the data distribution, requiring only that calibration and evaluation samples are exchangeable (i.i.d.). The set size becomes the signal of uncertainty: a singleton set means the model is confident; a multi-label set signals ambiguity; an empty set (theoretically possible under some thresholding rules) indicates extreme distributional mismatch.

The limitation of the basic approach (Split CP) is that its coverage guarantee is **marginal**: averaged over all samples. In practice, some classes may be systematically under-covered while others are over-covered. This is especially pronounced in imbalanced datasets or when classes have very different difficulty levels. This motivates class-conditional and cluster-conditional variants that provide stronger, per-group guarantees — at the cost of requiring more calibration data per group.

This notebook implements and compares **five conformal prediction methods** applied to patch-based multispectral image classification with three deep models:

1. **Split CP (SplitCP)** — the canonical baseline
2. **Class-Conditional CP (CcCP)** — per-class calibration thresholds
3. **Rank Calibrated Class-Conditional CP (RC3P)** — rank-pruned class-conditional CP
4. **Clustered CP (ClCP)** — classes grouped by embedding similarity before calibration
5. **RAPS** — rank-regularized adaptive prediction sets

The task is a 7-class multispectral land-cover classification over a 330×307 pixel scene with 6 spectral bands, using 9×9 spatial patches centred on each labeled pixel.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathcal{X} \subseteq \mathbb{R}^{p \times p \times B}$ be the input space (image patches of size $p \times p$ with $B$ spectral bands) and $\mathcal{Y} = \{0, 1, \ldots, K{-}1\}$ the label space with $K$ classes. A pre-trained classifier $f : \mathcal{X} \to \Delta^{K-1}$ maps each patch to a probability simplex vector $\hat{\pi}(x) = [\hat{\pi}_0(x), \ldots, \hat{\pi}_{K-1}(x)]$.

The dataset is partitioned into three disjoint sets:
- **Training set** $\mathcal{D}_\text{train}$ (75% of all labeled pixels): used to train $f$.
- **Calibration set** $\mathcal{D}_\text{cal} = \{(x_i, y_i)\}_{i=1}^{n}$ (12.5%): used to compute conformal thresholds.
- **Evaluation set** $\mathcal{D}_\text{eval}$ (12.5%): held out entirely for reporting empirical coverage and set-size metrics.

In this notebook: $n = 2155$ calibration samples, $K = 7$ classes, $B = 6$ bands, $p = 9$ pixels, $\alpha = 0.05$ (targeting 95% coverage).

---

### 2.2 The Nonconformity Score

At the heart of all CP methods is a **nonconformity score** $s(x, y)$ that measures how *unusual* it would be for the true label to be $y$ given input $x$. A low score means the label is consistent with the model; a high score means it is surprising.

For all methods in this notebook, the score is defined as:

$$s(x, y) = 1 - \hat{\pi}_y(x)$$

**Where:**
- $\hat{\pi}_y(x)$ — the model's predicted probability for the true class $y$ given input $x$
- $s(x,y) \in [0, 1]$ — score close to 0 means the model assigns high probability to the correct class; close to 1 means the model is wrong or uncertain

**What this means:** We are penalising the model whenever it assigns low softmax probability to the ground-truth label. Calibration will find the threshold $\hat{q}$ such that this score falls below $\hat{q}$ for at least $1-\alpha$ of calibration samples — and this threshold is then used to build prediction sets at test time.

---

### 2.3 Split Conformal Prediction: Threshold and Coverage Guarantee

#### Calibration

Given calibration scores $\{s_i = 1 - \hat{\pi}_{y_i}(x_i)\}_{i=1}^{n}$, the conformal quantile threshold is:

$$\hat{q} = \text{Quantile}\!\left(\{s_i\}_{i=1}^n,\ \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

**Where:**
- $n$ — number of calibration samples
- $\alpha$ — target miscoverage rate (here 0.05)
- $\lceil \cdot \rceil$ — ceiling function
- The "higher" interpolation is used, ensuring the quantile is at least as large as the $(1-\alpha)$ fraction of scores

**What this means:** We find the smallest threshold $\hat{q}$ that would have covered at least $\lceil (n+1)(1-\alpha)\rceil$ of the $n$ calibration points. The $+1$ correction accounts for the unseen test point and ensures the finite-sample coverage guarantee.

#### Prediction Set Construction

At test time, for a new input $x_\text{test}$:

$$\widehat{C}(x_\text{test}) = \left\{ y \in \mathcal{Y} : 1 - \hat{\pi}_y(x_\text{test}) \leq \hat{q} \right\} = \left\{ y : \hat{\pi}_y(x_\text{test}) \geq 1 - \hat{q} \right\}$$

#### Marginal Coverage Guarantee

For any new test point $(x_\text{test}, y_\text{test})$ drawn exchangeably with the calibration data:

$$\mathbb{P}\!\left(y_\text{test} \in \widehat{C}(x_\text{test})\right) \geq 1 - \alpha$$

This guarantee is **marginal** — averaged over the joint randomness in calibration and test data. It does not guarantee $1-\alpha$ coverage for any particular class or subgroup.

---

### 2.4 Class-Conditional Conformal Prediction (CcCP)

To address the limitation of marginal coverage, CcCP computes a separate threshold $\hat{q}_c$ for each class $c$:

$$\hat{q}_c = \text{Quantile}\!\left(\{s_i : y_i = c\}_{i \in \mathcal{D}_\text{cal}},\ \frac{\lceil (n_c+1)(1-\alpha) \rceil}{n_c}\right)$$

**Where:**
- $n_c = |\{i : y_i = c\}|$ — number of calibration samples belonging to class $c$
- $\hat{q}_c$ — the class-specific conformal threshold

The prediction set becomes:

$$\widehat{C}^\text{CcCP}(x) = \left\{ y : \hat{\pi}_y(x) \geq 1 - \hat{q}_y \right\}$$

Each class now uses its own threshold. Classes that are harder for the model (lower average probability on their own calibration samples) will have a higher $\hat{q}_c$, resulting in more inclusive prediction sets for those classes. Under sufficient calibration data per class, CcCP guarantees class-conditional coverage: $\mathbb{P}(y \in \widehat{C}(x) \mid y = c) \geq 1-\alpha$ for each $c$.

---

### 2.5 Rank Calibrated Class-Conditional CP (RC3P)

CcCP can produce unnecessarily large prediction sets because it checks all $K$ classes without regard to whether a class is plausibly ranked near the top. RC3P introduces a **rank gate** per class to prune out implausible candidates.

#### Top-$k$ Error Matrix

For each class $c$ and rank level $k$, define the top-$k$ error as the fraction of calibration samples of class $c$ for which the true class is *not* among the $k$ highest-probability predictions:

$$\epsilon_c^k = 1 - \frac{1}{n_c}\sum_{i: y_i = c} \mathbf{1}[r_f(x_i, c) \leq k]$$

**Where:**
- $r_f(x, c)$ — the rank of class $c$ in the model's probability ordering for input $x$ (rank 1 = highest probability)

#### Truncated Calibration Level

The effective per-class alpha is tightened by a gap term:

$$\tilde{\alpha}_c = \alpha - \frac{\Delta}{\sqrt{n/K}}$$

where $\Delta = 0.1$ is a truncation gap hyperparameter and $n/K$ approximates samples per class. The minimum valid rank limit for class $c$ is the smallest $k$ such that $\epsilon_c^k < \tilde{\alpha}_c$.

#### RC3P Prediction Set

RC3P searches over a mixing parameter $\eta \in [0, 1]$ to find the combination of rank limits and score thresholds that minimises average set size while maintaining coverage. The final prediction set for a test input $x$ is:

$$\widehat{C}^\text{RC3P}(x) = \left\{ y \in \mathcal{Y} : \hat{\pi}_y(x) \geq 1 - \hat{q}_y \text{ AND } r_f(x, y) \leq \hat{k}(y) \right\}$$

**Where:**
- $\hat{q}_y$ — per-class conformal score threshold
- $\hat{k}(y)$ — per-class rank limit (only include class $y$ if it is ranked within the top $\hat{k}(y)$ for this input)

**What this means:** A class must pass *two gates* to be in the prediction set: its model probability must exceed a class-specific threshold, and its rank must be within a class-specific limit. This dual constraint significantly reduces set sizes compared to CcCP while preserving class-conditional coverage.

---

### 2.6 Clustered Conformal Prediction (ClCP)

When the number of calibration samples per class $n_c$ is small (as is common with many classes), per-class thresholds are noisy. ClCP takes a middle path between the single global threshold of Split CP and the per-class thresholds of CcCP, by grouping similar classes together.

#### Clustering Step

A feature extractor (the penultimate layer of the model) maps each calibration sample to an embedding vector. The mean embedding per class is computed:

$$\mu_c = \frac{1}{n_c}\sum_{i: y_i = c} \phi(x_i)$$

where $\phi(x)$ denotes the penultimate-layer embedding. K-Means with $K_g = 4$ clusters is then applied to $\{\mu_c\}_{c=0}^{K-1}$, assigning each class to a cluster $g(c) \in \{0, \ldots, K_g{-}1\}$.

#### Cluster-Level Calibration

For each cluster $g$, a single threshold is computed from all calibration samples whose true class belongs to cluster $g$:

$$\hat{q}_g = \text{Quantile}\!\left(\{1 - \hat{\pi}_{y_i}(x_i) : g(y_i) = g\},\ \frac{\lceil (n_g+1)(1-\alpha)\rceil}{n_g}\right)$$

The prediction set for a test point is:

$$\widehat{C}^\text{ClCP}(x) = \left\{ y : \hat{\pi}_y(x) \geq 1 - \hat{q}_{g(y)} \right\}$$

**What this means:** Classes that produce similar embedding patterns are calibrated together, pooling their calibration data. This reduces variance in threshold estimates while still allowing some heterogeneity across clusters with different difficulty levels.

---

### 2.7 Regularised Adaptive Prediction Sets (RAPS)

The APS family of methods uses a different nonconformity score that is *rank-adaptive*: the score for a sample accounts for not just whether the true class has high probability, but how much probability mass is assigned to all classes ranked above it.

#### RAPS Nonconformity Score

For a sample $(x, y)$, let the classes be sorted in descending probability order as $o_1, o_2, \ldots, o_K$ (so $\hat{\pi}_{o_1}(x) \geq \hat{\pi}_{o_2}(x) \geq \ldots$). Let $L(y) = |\{j : \hat{\pi}_{o_j}(x) > \hat{\pi}_y(x)\}|$ be the 0-based rank of the true class. The RAPS score is:

$$s^\text{RAPS}(x, y) = \sum_{j=1}^{L(y)} \hat{\pi}_{o_j}(x) + \lambda \cdot \max(L(y) - k_\text{reg},\ 0)$$

**Where:**
- $\sum_{j=1}^{L(y)} \hat{\pi}_{o_j}(x)$ — cumulative probability mass of all classes ranked *above* the true class
- $\lambda$ — regularisation strength (here $\lambda = 0.01$)
- $k_\text{reg}$ — rank threshold above which the penalty activates (here $k_\text{reg} = 1$)
- The penalty term $\lambda \cdot \max(L(y) - k_\text{reg}, 0)$ discourages inclusion of low-probability classes by inflating their nonconformity score

**What this means:** If the true class is ranked first, the score is 0 (no probability mass is ranked above it, no penalty). If it is ranked second, the score equals the probability of the first-ranked class. The regularisation term additionally penalises when the true class is ranked beyond $k_\text{reg}$, encouraging the algorithm to produce smaller sets by making it harder for very low-ranked classes to be included.

#### RAPS Prediction Set Construction

The calibration threshold $\hat{q}$ is found from RAPS scores on the calibration set via the standard conformal quantile formula. At test time, the prediction set is built greedily: starting from the highest-probability class and adding classes in descending probability order until the RAPS threshold would be exceeded:

$$\widehat{C}^\text{RAPS}(x) = \text{smallest prefix of sorted classes such that } s^\text{RAPS}(x, y) \leq \hat{q} \text{ for each } y \text{ included}$$

At minimum, the top-1 class is always included. RAPS maintains the marginal coverage guarantee of Split CP.

---

## 3. Algorithm

**Inputs:** Calibration set $\mathcal{D}_\text{cal}$, evaluation set $\mathcal{D}_\text{eval}$, pre-trained model $f$, miscoverage rate $\alpha = 0.05$  
**Output:** Prediction sets for every evaluation pixel; empirical coverage, average set size, per-class coverage; spatial uncertainty maps

1. **Compute model probabilities** on calibration and evaluation sets using batched inference with softmax normalisation and NaN/Inf sanitisation.
2. **Compute full-scene probabilities** by sliding a patch window over the entire 330×307 image, building a probability cube of shape $(H, W, K)$.
3. **For each method, calibration phase:**
   - *SplitCP:* Compute $1 - \hat{\pi}_{y_i}$ for each calibration sample; take the $(1-\alpha)$ conformal quantile.
   - *CcCP:* Repeat SplitCP separately for each class $c$, storing $\hat{q}_c$.
   - *RC3P:* Build the top-$k$ error matrix; find truncated rank limits per class; grid-search the mixing parameter $\eta$ to minimise average set size subject to coverage.
   - *ClCP:* Extract penultimate-layer embeddings; compute per-class mean embeddings; K-Means into 4 clusters; compute a conformal quantile per cluster.
   - *RAPS:* Compute RAPS nonconformity scores on calibration set; take the conformal quantile.
4. **Evaluation phase:** Apply learned thresholds (and rank limits for RC3P) to evaluation set to form prediction sets; compute empirical coverage, average set size, singleton rate, empty-set rate, and per-class coverage.
5. **Spatial mapping** (SplitCP, CcCP, RC3P, ClCP): Apply thresholds pixel-wise to the full-scene probability cube; mark pixels with singleton prediction sets as "certain" (classified by argmax); mark all others as "uncertain".
6. **Output:** Aggregate all results into an Excel workbook with per-method, per-model sheets and comparison plots.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_uncertainty_comparison.ipynb`

### 4.1 Probability Normalisation

```python
def normalize_probs(prob, eps=1e-12):
    prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    rs = prob.sum(axis=-1, keepdims=True)
    rs = np.where(rs <= eps, 1.0, rs)
    return prob / rs
```

**What this does:** Sanitises raw model outputs by replacing invalid values, clipping to $[0,1]$, and row-normalising to a valid probability simplex.  
**Why:** Neural networks with softmax outputs should already produce valid probabilities, but numerical edge cases (e.g., extremely large logits, NaN propagation) can violate this; conformal scores assume valid probabilities.

---

### 4.2 Conformal Quantile Computation

```python
def conformal_qhat(scores, alpha):
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return safe_quantile(scores, q_level)
```

**What this does:** Implements the finite-sample correction to the quantile level: instead of taking the $(1-\alpha)$ quantile directly, it inflates the level to $\lceil(n+1)(1-\alpha)\rceil / n$.  
**Why:** This "+1" correction accounts for the unseen test point in the exchangeability argument, ensuring the marginal coverage guarantee holds exactly in finite samples rather than just asymptotically.

---

### 4.3 Split CP Calibration and Prediction

```python
# Calibration
calib_scores = 1.0 - prob_cal[np.arange(len(y_cal)), y_cal]
q_hat = conformal_qhat(calib_scores, alpha)

# Evaluation — prediction sets
pred_sets_eval = prob_eval >= (1.0 - q_hat)
```

**What this does:** For each calibration sample $i$, computes the nonconformity score as one minus the model's probability for the true class. The single global threshold `q_hat` is then used to threshold all class probabilities at test time.  
**Why:** The prediction set $\{y : \hat{\pi}_y(x) \geq 1 - \hat{q}\}$ is a simple vectorisable operation — no per-sample loop needed.

---

### 4.4 Full-Scene Map Generation

```python
pred_sets_full = prob_full >= (1.0 - q_hat)
set_sizes_map  = np.sum(pred_sets_full, axis=2)
pred_class_map = np.argmax(prob_full, axis=2)
combined_map   = np.where(set_sizes_map == 1, pred_class_map, n_classes)
```

**What this does:** Applies the conformal threshold to the full $(H, W, K)$ probability cube. Pixels with exactly one class in their prediction set are labelled with that class; all other pixels (set size 0 or $\geq 2$) are labelled as "uncertain" (coded as class index $K$).  
**Why:** This spatialises uncertainty directly: singleton sets represent high-confidence pixels; multi-element or empty sets represent regions where the model is ambiguous.

---

### 4.5 RC3P Rank Calibration

```python
# Compute per-class 0-based rank of the true label
cal_ranks = np.argsort(np.argsort(-prob_cal, axis=1), axis=1) + 1

# Find minimum k such that top-k error < truncated alpha
for c in range(n_classes):
    valid_k = np.where(err_matrix[:, c] < tc_alpha)[0]
    suit_k.append(valid_k[0] + 1 if len(valid_k) > 0 else n_classes)
```

**What this does:** `np.argsort(np.argsort(...))` computes the dense rank of each class within the sorted probability list (rank 1 = highest probability). The minimum rank limit per class is the smallest $k$ for which fewer than $\tilde{\alpha}$ fraction of calibration samples have the true class ranked below $k$.  
**Why:** This is the "truncated gap" step of RC3P: it identifies the rank boundary beyond which including a class would be statistically wasteful because the model almost never needs to look that far down its ranked list to find the correct class.

---

### 4.6 RAPS Score Computation

```python
def raps_score_single(prob_row, true_label, lam=0.01, k_reg=1):
    order     = np.argsort(prob_row)[::-1]
    rank      = int(np.where(order == true_label)[0][0])
    cumulative = float(np.sum(prob_row[order[:rank]]))
    penalty   = float(lam) * max(rank - int(k_reg), 0)
    return cumulative + penalty
```

**What this does:** For a single calibration sample, sorts classes by descending probability, finds the 0-based rank of the true label, sums the probability mass of all classes ranked above it, and adds the regularisation penalty if the true label is ranked beyond `k_reg`.  
**Why:** This sample-level implementation avoids vectorised complexity; RAPS set construction is sequential (greedy top-down), so per-sample loops are appropriate here.

---

## 5. Worked Numerical Example — All Five Methods

This section walks through every computation performed by the notebook's backend for all five conformal prediction methods, using a shared toy dataset. Every number is derived from scratch so you can verify that the backend logic is correct by following along manually.

---

### 5.0 Shared Setup

**Problem:** $K = 4$ classes (labelled 0, 1, 2, 3), $\alpha = 0.10$ (targeting 90% coverage), $n = 12$ calibration samples, 3 evaluation test samples.

**Calibration set — softmax probability matrix and true labels:**

| Sample $i$ | True $y_i$ | $\hat{\pi}_0$ | $\hat{\pi}_1$ | $\hat{\pi}_2$ | $\hat{\pi}_3$ |
|------------|-----------|--------------|--------------|--------------|--------------|
| 1  | 0 | **0.70** | 0.15 | 0.10 | 0.05 |
| 2  | 0 | **0.60** | 0.25 | 0.10 | 0.05 |
| 3  | 0 | **0.50** | 0.30 | 0.12 | 0.08 |
| 4  | 1 | 0.10 | **0.75** | 0.10 | 0.05 |
| 5  | 1 | 0.08 | **0.65** | 0.20 | 0.07 |
| 6  | 1 | 0.15 | **0.55** | 0.20 | 0.10 |
| 7  | 2 | 0.05 | 0.10 | **0.80** | 0.05 |
| 8  | 2 | 0.10 | 0.15 | **0.65** | 0.10 |
| 9  | 2 | 0.20 | 0.20 | **0.45** | 0.15 |
| 10 | 3 | 0.05 | 0.05 | 0.10 | **0.80** |
| 11 | 3 | 0.10 | 0.10 | 0.15 | **0.65** |
| 12 | 3 | 0.15 | 0.20 | 0.25 | **0.40** |

*(Bold = the probability at the true class; each row sums to 1.00)*

**Three evaluation test samples (used to build prediction sets):**

| Test | True $y$ | $\hat{\pi}_0$ | $\hat{\pi}_1$ | $\hat{\pi}_2$ | $\hat{\pi}_3$ |
|------|---------|-------|-------|-------|-------|
| T1 | 0 | **0.65** | 0.20 | 0.10 | 0.05 |
| T2 | 2 | 0.25 | 0.20 | **0.40** | 0.15 |
| T3 | 3 | 0.10 | 0.30 | 0.25 | **0.35** |

> T2 and T3 are deliberately difficult: T2 is borderline (class 2 is only weakly the highest), T3 is ambiguous (no class dominates clearly). This lets each method reveal different behaviour.

---

### 5.1 Method 1: Split Conformal Prediction (SplitCP)

Notebook function: `build_split_outputs_for_model`

#### Step 1 — Compute nonconformity scores on calibration set

For each calibration sample: $s_i = 1 - \hat{\pi}_{y_i}(x_i)$

| Sample | True $y_i$ | $\hat{\pi}_{y_i}$ | Score $s_i = 1 - \hat{\pi}_{y_i}$ |
|--------|-----------|-------------------|-------------------------------------|
| 1  | 0 | 0.70 | **0.30** |
| 2  | 0 | 0.60 | **0.40** |
| 3  | 0 | 0.50 | **0.50** |
| 4  | 1 | 0.75 | **0.25** |
| 5  | 1 | 0.65 | **0.35** |
| 6  | 1 | 0.55 | **0.45** |
| 7  | 2 | 0.80 | **0.20** |
| 8  | 2 | 0.65 | **0.35** |
| 9  | 2 | 0.45 | **0.55** |
| 10 | 3 | 0.80 | **0.20** |
| 11 | 3 | 0.65 | **0.35** |
| 12 | 3 | 0.40 | **0.60** |

All 12 scores: `[0.30, 0.40, 0.50, 0.25, 0.35, 0.45, 0.20, 0.35, 0.55, 0.20, 0.35, 0.60]`

#### Step 2 — Compute the conformal quantile $\hat{q}$

$$q_\text{level} = \frac{\lceil (n+1)(1-\alpha) \rceil}{n} = \frac{\lceil 13 \times 0.90 \rceil}{12} = \frac{\lceil 11.7 \rceil}{12} = \frac{12}{12} = 1.0$$

Sort all 12 scores in ascending order:

`[0.20, 0.20, 0.25, 0.30, 0.35, 0.35, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]`

The quantile at level 1.0 with "higher" interpolation = the maximum value = **$\hat{q} = 0.60$**

Probability threshold for inclusion: $1 - \hat{q} = 1 - 0.60 = \mathbf{0.40}$

> Any class with predicted probability $\geq 0.40$ enters the prediction set.

#### Step 3 — Build prediction sets for the three test samples

**Test T1** (true class = 0): probs = [0.65, 0.20, 0.10, 0.05]

| Class | Prob | $\geq 0.40$? | In set? |
|-------|------|-------------|---------|
| 0 | 0.65 | Yes | ✓ |
| 1 | 0.20 | No | ✗ |
| 2 | 0.10 | No | ✗ |
| 3 | 0.05 | No | ✗ |

→ Prediction set: **{0}** | Size = 1 | True class 0 ∈ {0} → **COVERED ✓**

**Test T2** (true class = 2): probs = [0.25, 0.20, 0.40, 0.15]

| Class | Prob | $\geq 0.40$? | In set? |
|-------|------|-------------|---------|
| 0 | 0.25 | No | ✗ |
| 1 | 0.20 | No | ✗ |
| 2 | 0.40 | Yes (exactly equal) | ✓ |
| 3 | 0.15 | No | ✗ |

→ Prediction set: **{2}** | Size = 1 | True class 2 ∈ {2} → **COVERED ✓**

**Test T3** (true class = 3): probs = [0.10, 0.30, 0.25, 0.35]

| Class | Prob | $\geq 0.40$? | In set? |
|-------|------|-------------|---------|
| 0 | 0.10 | No | ✗ |
| 1 | 0.30 | No | ✗ |
| 2 | 0.25 | No | ✗ |
| 3 | 0.35 | No | ✗ |

→ Prediction set: **{}** (empty set) | Size = 0 | True class 3 ∉ {} → **NOT COVERED ✗**

#### Summary — SplitCP

| | $\hat{q}$ | Threshold $1-\hat{q}$ | T1 set | T2 set | T3 set | T1 covered | T2 covered | T3 covered |
|-|-----------|-----------------------|--------|--------|--------|-----------|-----------|-----------|
| SplitCP | 0.60 | 0.40 | {0} | {2} | {} | ✓ | ✓ | ✗ |

Empirical coverage on these 3 test points = 2/3 = **67%** (below 90% target — expected with only 3 evaluation samples; the guarantee is over the joint distribution, not small finite samples).

---

### 5.2 Method 2: Class-Conditional CP (CcCP)

Notebook function: `build_classconditional_outputs_for_model`

#### Step 1 — Separate calibration scores by class

**Class 0** (samples 1, 2, 3): scores = [0.30, 0.40, 0.50], $n_0 = 3$

**Class 1** (samples 4, 5, 6): scores = [0.25, 0.35, 0.45], $n_1 = 3$

**Class 2** (samples 7, 8, 9): scores = [0.20, 0.35, 0.55], $n_2 = 3$

**Class 3** (samples 10, 11, 12): scores = [0.20, 0.35, 0.60], $n_3 = 3$

#### Step 2 — Compute per-class conformal quantile $\hat{q}_c$

For each class: $q_\text{level} = \lceil (n_c + 1)(1 - \alpha) \rceil / n_c = \lceil 4 \times 0.90 \rceil / 3 = \lceil 3.6 \rceil / 3 = 4/3 = 1.\overline{3}$

Since $q_\text{level}$ is clipped at 1.0 (per `min(1.0, ...)` in the code), the quantile is always the **maximum** of each class's scores.

| Class $c$ | Scores | $q_\text{level}$ (clipped) | $\hat{q}_c$ = max |
|-----------|--------|--------------------------|-------------------|
| 0 | [0.30, 0.40, 0.50] | 1.0 | **0.50** |
| 1 | [0.25, 0.35, 0.45] | 1.0 | **0.45** |
| 2 | [0.20, 0.35, 0.55] | 1.0 | **0.55** |
| 3 | [0.20, 0.35, 0.60] | 1.0 | **0.60** |

Probability inclusion thresholds $1 - \hat{q}_c$:

| Class 0 | Class 1 | Class 2 | Class 3 |
|---------|---------|---------|---------|
| $1 - 0.50 = \mathbf{0.50}$ | $1 - 0.45 = \mathbf{0.55}$ | $1 - 0.55 = \mathbf{0.45}$ | $1 - 0.60 = \mathbf{0.40}$ |

> Each class now has its own threshold. The prediction set rule is: **include class $c$ if $\hat{\pi}_c(x) \geq 1 - \hat{q}_c$**.

#### Step 3 — Build prediction sets for the three test samples

**Test T1** (true class = 0): probs = [0.65, 0.20, 0.10, 0.05]

| Class $c$ | Prob $\hat{\pi}_c$ | Threshold $1 - \hat{q}_c$ | Prob ≥ Threshold? | In set? |
|-----------|--------------------|--------------------------|-------------------|---------|
| 0 | 0.65 | 0.50 | 0.65 ≥ 0.50 → Yes | ✓ |
| 1 | 0.20 | 0.55 | 0.20 ≥ 0.55 → No | ✗ |
| 2 | 0.10 | 0.45 | 0.10 ≥ 0.45 → No | ✗ |
| 3 | 0.05 | 0.40 | 0.05 ≥ 0.40 → No | ✗ |

→ Prediction set: **{0}** | Size = 1 | **COVERED ✓**

**Test T2** (true class = 2): probs = [0.25, 0.20, 0.40, 0.15]

| Class $c$ | Prob $\hat{\pi}_c$ | Threshold $1 - \hat{q}_c$ | Prob ≥ Threshold? | In set? |
|-----------|--------------------|--------------------------|-------------------|---------|
| 0 | 0.25 | 0.50 | 0.25 ≥ 0.50 → No | ✗ |
| 1 | 0.20 | 0.55 | 0.20 ≥ 0.55 → No | ✗ |
| 2 | 0.40 | 0.45 | 0.40 ≥ 0.45 → No | ✗ |
| 3 | 0.15 | 0.40 | 0.15 ≥ 0.40 → No | ✗ |

→ Prediction set: **{}** (empty) | Size = 0 | **NOT COVERED ✗**

> Compare to SplitCP: SplitCP included class 2 (0.40 ≥ 0.40), but CcCP's class-2-specific threshold is stricter (0.45), so class 2 is excluded.

**Test T3** (true class = 3): probs = [0.10, 0.30, 0.25, 0.35]

| Class $c$ | Prob $\hat{\pi}_c$ | Threshold $1 - \hat{q}_c$ | Prob ≥ Threshold? | In set? |
|-----------|--------------------|--------------------------|-------------------|---------|
| 0 | 0.10 | 0.50 | 0.10 ≥ 0.50 → No | ✗ |
| 1 | 0.30 | 0.55 | 0.30 ≥ 0.55 → No | ✗ |
| 2 | 0.25 | 0.45 | 0.25 ≥ 0.45 → No | ✗ |
| 3 | 0.35 | 0.40 | 0.35 ≥ 0.40 → No | ✗ |

→ Prediction set: **{}** (empty) | Size = 0 | **NOT COVERED ✗**

#### Summary — CcCP

| Class | $\hat{q}_c$ | Threshold $1 - \hat{q}_c$ |
|-------|------------|--------------------------|
| 0 | 0.50 | 0.50 |
| 1 | 0.45 | 0.55 |
| 2 | 0.55 | 0.45 |
| 3 | 0.60 | 0.40 |

| Test | Set | Covered? |
|------|-----|---------|
| T1 (true=0) | {0} | ✓ |
| T2 (true=2) | {} | ✗ |
| T3 (true=3) | {} | ✗ |

---

### 5.3 Method 3: RC3P — Rank Calibrated Class-Conditional CP

Notebook function: `build_rc3p_outputs_for_model`, `compute_rc3p_qhats_and_sets`

RC3P adds a second gate: a class is only included in the prediction set if (a) its probability exceeds the class-specific threshold AND (b) its rank is within a class-specific rank limit $\hat{k}(c)$.

#### Step 1 — Compute rank of the true class for each calibration sample

For each sample, sort classes by descending probability and assign rank 1 to the highest. The rank of the true class is what matters.

| Sample | True $y_i$ | Probs [0,1,2,3] | Sorted order (desc) | Rank of $y_i$ |
|--------|-----------|-----------------|---------------------|--------------|
| 1  | 0 | [0.70,0.15,0.10,0.05] | 0,1,2,3 | **1** |
| 2  | 0 | [0.60,0.25,0.10,0.05] | 0,1,2,3 | **1** |
| 3  | 0 | [0.50,0.30,0.12,0.08] | 0,1,2,3 | **1** |
| 4  | 1 | [0.10,0.75,0.10,0.05] | 1,0,2,3 | **1** |
| 5  | 1 | [0.08,0.65,0.20,0.07] | 1,2,0,3 | **1** |
| 6  | 1 | [0.15,0.55,0.20,0.10] | 1,2,0,3 | **1** |
| 7  | 2 | [0.05,0.10,0.80,0.05] | 2,1,0,3 | **1** |
| 8  | 2 | [0.10,0.15,0.65,0.10] | 2,1,0,3 | **1** |
| 9  | 2 | [0.20,0.20,0.45,0.15] | 2,0,1,3 | **1** |
| 10 | 3 | [0.05,0.05,0.10,0.80] | 3,2,0,1 | **1** |
| 11 | 3 | [0.10,0.10,0.15,0.65] | 3,2,0,1 | **1** |
| 12 | 3 | [0.15,0.20,0.25,0.40] | 3,2,1,0 | **1** |

All 12 calibration samples have the true class ranked **1st** — the model is well-calibrated on the calibration set.

#### Step 2 — Build the top-$k$ error matrix

For class $c$ and rank level $k$: $\epsilon_c^k = $ fraction of class-$c$ calibration samples where the true class rank is $> k$.

Since every sample has true class rank = 1:

| $k$ | $\epsilon_0^k$ | $\epsilon_1^k$ | $\epsilon_2^k$ | $\epsilon_3^k$ |
|-----|---------------|---------------|---------------|---------------|
| 1 | 0/3 = **0.00** | 0/3 = **0.00** | 0/3 = **0.00** | 0/3 = **0.00** |
| 2 | 0/3 = **0.00** | 0/3 = **0.00** | 0/3 = **0.00** | 0/3 = **0.00** |
| 3 | 0/3 = **0.00** | 0/3 = **0.00** | 0/3 = **0.00** | 0/3 = **0.00** |
| 4 | 0/3 = **0.00** | 0/3 = **0.00** | 0/3 = **0.00** | 0/3 = **0.00** |

The error matrix `err_matrix` (shape $K \times K$ in the code) has rows = $k$ values 1..K and columns = classes.

#### Step 3 — Compute truncated calibration level $\tilde{\alpha}$

$$\tilde{\alpha} = \alpha - \frac{\Delta}{\sqrt{n/K}} = 0.10 - \frac{0.10}{\sqrt{12/4}} = 0.10 - \frac{0.10}{\sqrt{3}} = 0.10 - \frac{0.10}{1.732} = 0.10 - 0.0577 = \mathbf{0.0423}$$

#### Step 4 — Find minimum valid rank limit $\hat{k}(c)$ for each class

For class $c$: find the smallest $k$ such that $\epsilon_c^k < \tilde{\alpha} = 0.0423$.

Since $\epsilon_c^1 = 0.00 < 0.0423$ for every class, the minimum valid rank limit for all classes is $k = 1$.

$$\hat{k}(0) = \hat{k}(1) = \hat{k}(2) = \hat{k}(3) = \mathbf{1}$$

> This says: for every class, the model almost always ranks the true class first, so we only need to look at rank-1 candidates.

#### Step 5 — Grid search over mixing parameter $\eta$

The code searches $\eta \in \{0.0, 0.5, 1.0\}$ (3 values since $k_\text{max} = 1$, $K = 4$, so `mix_paras = np.linspace(0, 1, K - k_max + 1) = np.linspace(0, 1, 4)`).

For each $\eta$, the trial rank limit per class is:
$$k_\text{trial}(c) = \lceil (1 - \eta) \times \hat{k}(c) + K \times \eta \rceil = \lceil (1-\eta) \times 1 + 4\eta \rceil$$

| $\eta$ | $k_\text{trial}$ (same for all classes) | Effective range |
|--------|----------------------------------------|-----------------|
| 0.00 | $\lceil 1.0 \rceil = 1$ | Only rank-1 candidates |
| 0.33 | $\lceil 0.67 + 1.33 \rceil = \lceil 2.0 \rceil = 2$ | Rank 1–2 candidates |
| 0.67 | $\lceil 0.33 + 2.67 \rceil = \lceil 3.0 \rceil = 3$ | Rank 1–3 candidates |
| 1.00 | $\lceil 4.0 \rceil = 4$ | All candidates |

For each $\eta$, the class-specific effective alpha is:
$$\hat{\alpha}_c(\eta) = \tilde{\alpha} - \epsilon_c^{k_\text{trial}} = 0.0423 - 0.00 = 0.0423$$

The per-class score threshold at $k_\text{trial} = 1$ (filtering calibration samples whose true class has rank $\leq 1$, i.e., all samples):

Class 0 calibration samples with rank ≤ 1: all 3. Scores: [0.30, 0.40, 0.50].
$\hat{q}_0 = \text{Quantile}([0.30, 0.40, 0.50],\ \lceil 4 \times (1-0.0423) \rceil / 3) = \text{Quantile}([0.30, 0.40, 0.50],\ \lceil 3.83 \rceil / 3) = \text{Quantile}([0.30, 0.40, 0.50],\ 4/3)$

Since $4/3 = 1.\overline{3} > 1.0$, clip to 1.0 → $\hat{q}_0 = \max([0.30, 0.40, 0.50]) = \mathbf{0.50}$

Repeating for all classes with $\eta = 0$ (same formula, different score sets):

| Class | Scores at rank ≤ 1 | $\hat{q}_c$ |
|-------|--------------------|-------------|
| 0 | [0.30, 0.40, 0.50] | **0.50** |
| 1 | [0.25, 0.35, 0.45] | **0.45** |
| 2 | [0.20, 0.35, 0.55] | **0.55** |
| 3 | [0.20, 0.35, 0.60] | **0.60** |

Now compute average prediction set size on evaluation set with $\eta = 0$, $k_\text{trial} = 1$:

For a sample to be in class $c$'s prediction set it must satisfy:
- $\hat{\pi}_c(x) \geq 1 - \hat{q}_c$ (score gate), AND
- rank of class $c$ in the prediction is $\leq 1$ (rank gate)

The rank gate $\leq 1$ means: **only the top-ranked class can be included**. Since only one class is rank-1, every prediction set has size ≤ 1.

For the 3 evaluation samples: set sizes = [1, 1, 0] → average = 0.667.

With $\eta = 1.0$, $k_\text{trial} = 4$ (all classes allowed) and the rank gate is trivially satisfied → this degenerates to CcCP, with average set size also approximately 0.667 on these samples.

Since $\eta = 0$ gives the smallest average set size (= 0.667 ≤ all other $\eta$), the algorithm selects:

$$\text{best}: \hat{k}(c) = 1 \text{ for all } c, \quad \hat{q}_0 = 0.50,\ \hat{q}_1 = 0.45,\ \hat{q}_2 = 0.55,\ \hat{q}_3 = 0.60$$

#### Step 6 — Build RC3P prediction sets for test samples

Prediction rule: include class $c$ iff $\hat{\pi}_c(x) \geq 1 - \hat{q}_c$ AND $\text{rank}(c) \leq \hat{k}(c) = 1$

**Test T1** (true=0): probs = [0.65, 0.20, 0.10, 0.05]

Sorted descending: class 0 (0.65) > class 1 (0.20) > class 2 (0.10) > class 3 (0.05)
Ranks: class 0 → rank 1, class 1 → rank 2, class 2 → rank 3, class 3 → rank 4

| Class | Prob | Threshold | Score gate? | Rank | Rank ≤ 1? | Both gates? |
|-------|------|-----------|------------|------|-----------|-------------|
| 0 | 0.65 | 0.50 | ✓ | 1 | ✓ | **IN** |
| 1 | 0.20 | 0.55 | ✗ | 2 | ✗ | out |
| 2 | 0.10 | 0.45 | ✗ | 3 | ✗ | out |
| 3 | 0.05 | 0.40 | ✗ | 4 | ✗ | out |

→ Set: **{0}** | **COVERED ✓**

**Test T2** (true=2): probs = [0.25, 0.20, 0.40, 0.15]

Sorted descending: class 2 (0.40) > class 0 (0.25) > class 1 (0.20) > class 3 (0.15)
Ranks: class 2 → rank 1, class 0 → rank 2, class 1 → rank 3, class 3 → rank 4

| Class | Prob | Threshold | Score gate? | Rank | Rank ≤ 1? | Both gates? |
|-------|------|-----------|------------|------|-----------|-------------|
| 0 | 0.25 | 0.50 | ✗ | 2 | ✗ | out |
| 1 | 0.20 | 0.55 | ✗ | 3 | ✗ | out |
| 2 | 0.40 | 0.45 | ✗ | 1 | ✓ | **Score gate fails → out** |
| 3 | 0.15 | 0.40 | ✗ | 4 | ✗ | out |

→ Set: **{}** | **NOT COVERED ✗**

> Class 2 passes the rank gate (it is rank 1) but fails the score gate (0.40 < 0.45). The dual-gate requirement is strict.

**Test T3** (true=3): probs = [0.10, 0.30, 0.25, 0.35]

Sorted descending: class 3 (0.35) > class 1 (0.30) > class 2 (0.25) > class 0 (0.10)
Ranks: class 3 → rank 1, class 1 → rank 2, class 2 → rank 3, class 0 → rank 4

| Class | Prob | Threshold | Score gate? | Rank | Rank ≤ 1? | Both gates? |
|-------|------|-----------|------------|------|-----------|-------------|
| 0 | 0.10 | 0.50 | ✗ | 4 | ✗ | out |
| 1 | 0.30 | 0.55 | ✗ | 2 | ✗ | out |
| 2 | 0.25 | 0.45 | ✗ | 3 | ✗ | out |
| 3 | 0.35 | 0.40 | ✗ | 1 | ✓ | **Score gate fails → out** |

→ Set: **{}** | **NOT COVERED ✗**

#### Summary — RC3P

| Class | $\hat{q}_c$ | $1 - \hat{q}_c$ | Rank limit $\hat{k}(c)$ |
|-------|------------|-----------------|------------------------|
| 0 | 0.50 | 0.50 | 1 |
| 1 | 0.45 | 0.55 | 1 |
| 2 | 0.55 | 0.45 | 1 |
| 3 | 0.60 | 0.40 | 1 |

| Test | Set | Covered? |
|------|-----|---------|
| T1 (true=0) | {0} | ✓ |
| T2 (true=2) | {} | ✗ |
| T3 (true=3) | {} | ✗ |

With $\hat{k} = 1$ for all classes and the model's probabilities on test samples never simultaneously satisfying both gates for borderline cases, RC3P here produces the same sets as CcCP. In practice with larger calibration sets and lower-confidence models, RC3P's rank gate provides meaningful additional pruning over CcCP.

---

### 5.4 Method 4: Clustered Conformal Prediction (ClCP)

Notebook function: `build_clustered_outputs_for_model`

ClCP groups classes by embedding similarity and calibrates at the cluster level. We simulate this with toy embeddings.

#### Step 1 — Construct class mean embeddings

In the notebook, embeddings come from the penultimate layer. We use a 2-D toy embedding for clarity.

Per-class mean embeddings (imagine these came from the model's second-to-last layer):

| Class | Mean embedding $\mu_c$ |
|-------|------------------------|
| 0 | [0.9, 0.1] |
| 1 | [0.8, 0.2] |
| 2 | [0.1, 0.9] |
| 3 | [0.2, 0.8] |

#### Step 2 — K-Means clustering with $K_g = 2$ clusters

> The notebook uses $K_g = 4$ in general. We use 2 here to make the grouping visible with 4 classes.

K-Means initialises two centroids and iterates until convergence. With the embeddings above, two natural clusters emerge:

**Cluster A (classes 0 and 1):** embeddings near [0.85, 0.15]  
**Cluster B (classes 2 and 3):** embeddings near [0.15, 0.85]

Assignments:
- Class 0 → **Cluster A**
- Class 1 → **Cluster A**
- Class 2 → **Cluster B**
- Class 3 → **Cluster B**

#### Step 3 — Compute per-cluster nonconformity scores

**Cluster A** — calibration samples from classes 0 and 1 (samples 1–6):

Scores (from column $s_i = 1 - \hat{\pi}_{y_i}$ computed earlier):
- Class 0: [0.30, 0.40, 0.50]
- Class 1: [0.25, 0.35, 0.45]
- All cluster A scores: [0.30, 0.40, 0.50, 0.25, 0.35, 0.45] ($n_A = 6$)

**Cluster B** — calibration samples from classes 2 and 3 (samples 7–12):

- Class 2: [0.20, 0.35, 0.55]
- Class 3: [0.20, 0.35, 0.60]
- All cluster B scores: [0.20, 0.35, 0.55, 0.20, 0.35, 0.60] ($n_B = 6$)

#### Step 4 — Compute per-cluster conformal quantile $\hat{q}_g$

$$q_\text{level} = \frac{\lceil (n_g + 1)(1-\alpha) \rceil}{n_g} = \frac{\lceil 7 \times 0.90 \rceil}{6} = \frac{\lceil 6.3 \rceil}{6} = \frac{7}{6} = 1.1\overline{6}$$

Clip to 1.0 → quantile = maximum of each cluster's scores.

**Cluster A:** sorted = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50] → $\hat{q}_A = \mathbf{0.50}$

**Cluster B:** sorted = [0.20, 0.20, 0.35, 0.35, 0.55, 0.60] → $\hat{q}_B = \mathbf{0.60}$

Per-class thresholds derived from cluster membership:

| Class | Cluster | $\hat{q}_\text{cluster}$ | Threshold $1 - \hat{q}_\text{cluster}$ |
|-------|---------|--------------------------|----------------------------------------|
| 0 | A | 0.50 | **0.50** |
| 1 | A | 0.50 | **0.50** |
| 2 | B | 0.60 | **0.40** |
| 3 | B | 0.60 | **0.40** |

> Notice: classes 2 and 3 share the same (looser) threshold 0.40 because they are calibrated together. This is the key difference from CcCP.

#### Step 5 — Build prediction sets for the three test samples

Prediction rule: include class $c$ if $\hat{\pi}_c(x) \geq 1 - \hat{q}_{g(c)}$

**Test T1** (true=0): probs = [0.65, 0.20, 0.10, 0.05]

| Class | Prob | Cluster | Threshold | In set? |
|-------|------|---------|-----------|---------|
| 0 | 0.65 | A | 0.50 | 0.65 ≥ 0.50 → **✓** |
| 1 | 0.20 | A | 0.50 | 0.20 ≥ 0.50 → ✗ |
| 2 | 0.10 | B | 0.40 | 0.10 ≥ 0.40 → ✗ |
| 3 | 0.05 | B | 0.40 | 0.05 ≥ 0.40 → ✗ |

→ Set: **{0}** | **COVERED ✓**

**Test T2** (true=2): probs = [0.25, 0.20, 0.40, 0.15]

| Class | Prob | Cluster | Threshold | In set? |
|-------|------|---------|-----------|---------|
| 0 | 0.25 | A | 0.50 | 0.25 ≥ 0.50 → ✗ |
| 1 | 0.20 | A | 0.50 | 0.20 ≥ 0.50 → ✗ |
| 2 | 0.40 | B | 0.40 | 0.40 ≥ 0.40 → **✓** |
| 3 | 0.15 | B | 0.40 | 0.15 ≥ 0.40 → ✗ |

→ Set: **{2}** | **COVERED ✓**

> ClCP covers T2 where CcCP and RC3P did not. The reason: Cluster B's threshold (0.40) is looser than CcCP's class-2-specific threshold (0.45), because class 3's harder calibration samples pull the cluster threshold up — and class 2 benefits from this shared looseness.

**Test T3** (true=3): probs = [0.10, 0.30, 0.25, 0.35]

| Class | Prob | Cluster | Threshold | In set? |
|-------|------|---------|-----------|---------|
| 0 | 0.10 | A | 0.50 | 0.10 ≥ 0.50 → ✗ |
| 1 | 0.30 | A | 0.50 | 0.30 ≥ 0.50 → ✗ |
| 2 | 0.25 | B | 0.40 | 0.25 ≥ 0.40 → ✗ |
| 3 | 0.35 | B | 0.40 | 0.35 ≥ 0.40 → ✗ |

→ Set: **{}** | **NOT COVERED ✗**

#### Summary — ClCP

| Cluster | Classes | $\hat{q}_g$ | Threshold $1 - \hat{q}_g$ |
|---------|---------|------------|--------------------------|
| A | {0, 1} | 0.50 | 0.50 |
| B | {2, 3} | 0.60 | 0.40 |

| Test | Set | Covered? |
|------|-----|---------|
| T1 (true=0) | {0} | ✓ |
| T2 (true=2) | {2} | ✓ |
| T3 (true=3) | {} | ✗ |

---

### 5.5 Method 5: RAPS — Regularised Adaptive Prediction Sets

Notebook function: `build_raps_outputs_for_model`, `raps_score_single`, `raps_set_single`

Parameters: $\lambda = 0.01$, $k_\text{reg} = 1$

RAPS uses a fundamentally different nonconformity score that accounts for cumulative probability mass above the true class.

#### Step 1 — Compute RAPS nonconformity scores on calibration set

For each sample: sort classes descending, find the 0-based rank $L$ of the true class, sum probabilities of all classes above it, add the penalty.

$$s_i^\text{RAPS} = \underbrace{\sum_{j=1}^{L} \hat{\pi}_{o_j}}_{\text{cumulative mass above true class}} + \underbrace{\lambda \cdot \max(L - k_\text{reg},\ 0)}_{\text{penalty if rank} > k_\text{reg}}$$

where $o_1, o_2, \ldots$ is the descending probability ordering and $L$ is the 0-based rank of the true class (true class is at position $L$, with $L=0$ meaning rank 1, i.e., top of the list).

| Sample | True $y_i$ | Sorted desc order | 0-based rank $L$ | Cumulative above | Penalty $0.01 \times \max(L-1, 0)$ | RAPS score |
|--------|-----------|-------------------|-----------------|-----------------|--------------------------------------|------------|
| 1  | 0 | 0,1,2,3 | 0 | 0.00 | $0.01 \times 0 = 0.000$ | **0.000** |
| 2  | 0 | 0,1,2,3 | 0 | 0.00 | 0.000 | **0.000** |
| 3  | 0 | 0,1,2,3 | 0 | 0.00 | 0.000 | **0.000** |
| 4  | 1 | 1,0,2,3 | 0 | 0.00 | 0.000 | **0.000** |
| 5  | 1 | 1,2,0,3 | 0 | 0.00 | 0.000 | **0.000** |
| 6  | 1 | 1,2,0,3 | 0 | 0.00 | 0.000 | **0.000** |
| 7  | 2 | 2,1,0,3 | 0 | 0.00 | 0.000 | **0.000** |
| 8  | 2 | 2,1,0,3 | 0 | 0.00 | 0.000 | **0.000** |
| 9  | 2 | 2,0,1,3 | 0 | 0.00 | 0.000 | **0.000** |
| 10 | 3 | 3,2,0,1 | 0 | 0.00 | 0.000 | **0.000** |
| 11 | 3 | 3,2,0,1 | 0 | 0.00 | 0.000 | **0.000** |
| 12 | 3 | 3,2,1,0 | 0 | 0.00 | 0.000 | **0.000** |

All RAPS calibration scores = **0.000** because the true class is always ranked first (rank 1, $L = 0$), so there is no probability mass above it and no penalty applies.

#### Step 2 — Compute the RAPS quantile $\hat{q}^\text{RAPS}$

All 12 scores = 0.00. Sorted: $[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]$

$$q_\text{level} = \frac{\lceil 13 \times 0.90 \rceil}{12} = \frac{12}{12} = 1.0 \quad \Rightarrow \quad \hat{q}^\text{RAPS} = \max(0,\ldots,0) = \mathbf{0.00}$$

#### Step 3 — Build RAPS prediction sets for test samples

The notebook function `raps_set_single` works greedily: start from the top-ranked class and include it; add the next class only if the running cumulative score (plus penalty) does not exceed $\hat{q}^\text{RAPS} = 0.00$.

**Test T1** (true=0): probs = [0.65, 0.20, 0.10, 0.05]

Sorted descending: class 0 (0.65), class 1 (0.20), class 2 (0.10), class 3 (0.05)

Greedy inclusion:
- **Class 0** (rank 0, $L=0$): cumulative before adding = 0.00; penalty = $0.01 \times \max(0-1, 0) = 0.00$; score to check = $0.00 + 0.00 = 0.00 \leq \hat{q} = 0.00$ → **INCLUDE**. Running cumulative after adding = 0.65.
- **Class 1** (rank 1, $L=1$): cumulative before = 0.65; penalty = $0.01 \times \max(1-1, 0) = 0.00$; score = $0.65 + 0.00 = 0.65 > 0.00$ → **STOP**.

→ Set: **{0}** | **COVERED ✓**

**Test T2** (true=2): probs = [0.25, 0.20, 0.40, 0.15]

Sorted descending: class 2 (0.40), class 0 (0.25), class 1 (0.20), class 3 (0.15)

- **Class 2** (rank 0): score to check = $0.00 + 0.00 = 0.00 \leq 0.00$ → **INCLUDE**. Running cumulative = 0.40.
- **Class 0** (rank 1): score = $0.40 + 0.00 = 0.40 > 0.00$ → **STOP**.

→ Set: **{2}** | **COVERED ✓**

**Test T3** (true=3): probs = [0.10, 0.30, 0.25, 0.35]

Sorted descending: class 3 (0.35), class 1 (0.30), class 2 (0.25), class 0 (0.10)

- **Class 3** (rank 0): score = $0.00 + 0.00 = 0.00 \leq 0.00$ → **INCLUDE**. Running cumulative = 0.35.
- **Class 1** (rank 1): score = $0.35 + 0.00 = 0.35 > 0.00$ → **STOP**.

→ Set: **{3}** | **COVERED ✓**

#### Step 4 — Minimum guarantee: top-1 is always included

The notebook enforces `if not pred_set.any(): pred_set[order[0]] = True`. Since every set already contains the top-1 class above, this fallback never fires here.

#### Summary — RAPS

| $\hat{q}^\text{RAPS}$ | Interpretation |
|-----------------------|----------------|
| 0.00 | The model always places the true class first during calibration; the threshold collapses to zero, forcing prediction sets to be exactly the top-1 class. |

| Test | Set | Covered? |
|------|-----|---------|
| T1 (true=0) | {0} | ✓ |
| T2 (true=2) | {2} | ✓ |
| T3 (true=3) | {3} | ✓ |

RAPS achieves 100% coverage on these 3 test samples — not because the method is perfect, but because the calibration data led to $\hat{q} = 0$, which makes RAPS always produce singleton sets containing only the top-1 predicted class. This mirrors exactly what is observed in the notebook's actual run (singleton_rate = 1.00, empirical_coverage ≈ 0.999 for all three models).

---

### 5.6 Cross-Method Comparison on Test Samples

| Test sample | True class | SplitCP | CcCP | RC3P | ClCP | RAPS |
|-------------|-----------|---------|------|------|------|------|
| T1 | 0 | {0} ✓ | {0} ✓ | {0} ✓ | {0} ✓ | {0} ✓ |
| T2 | 2 | {2} ✓ | {} ✗ | {} ✗ | {2} ✓ | {2} ✓ |
| T3 | 3 | {} ✗ | {} ✗ | {} ✗ | {} ✗ | {3} ✓ |

**Key takeaways visible in these 3 examples:**

**T1 (easy, high confidence):** All five methods agree — class 0 has probability 0.65, well above every method's threshold. Universal singleton coverage.

**T2 (borderline, class 2 at exactly 0.40):**
- SplitCP includes it because 0.40 ≥ 0.40 (threshold is exactly met).
- CcCP excludes it because class 2's per-class threshold is stricter: 0.45 > 0.40.
- RC3P also excludes it (same per-class threshold as CcCP under $\hat{k} = 1$).
- ClCP includes it because cluster B's shared threshold 0.40 is looser (class 3's harder scores pulled the cluster-level quantile up).
- RAPS includes it because the top-1 class (class 2) always enters the greedy set.

**T3 (ambiguous, no class dominates):**
- SplitCP, CcCP, RC3P, ClCP all produce empty sets — the model's top probability (0.35 for class 3) is below every method's acceptance threshold.
- RAPS produces {3} because its greedy mechanism always admits the top-ranked class, regardless of how low its probability is. This is why RAPS achieves ~100% coverage in practice: it never produces empty sets.

**Threshold summary across methods for each class:**

| Class | SplitCP threshold | CcCP threshold | RC3P threshold | ClCP threshold | RAPS threshold (implicit) |
|-------|------------------|---------------|---------------|---------------|--------------------------|
| 0 | 0.40 | 0.50 | 0.50 | 0.50 | top-1 auto-included |
| 1 | 0.40 | 0.55 | 0.55 | 0.50 | top-1 auto-included |
| 2 | 0.40 | 0.45 | 0.45 | 0.40 | top-1 auto-included |
| 3 | 0.40 | 0.40 | 0.40 | 0.40 | top-1 auto-included |

SplitCP is the most uniform (same threshold for all classes). CcCP and RC3P are the most selective for easy classes (classes 0 and 1) and lenient for hard classes (class 3). ClCP pools classes 0+1 and 2+3 separately, landing between the two extremes. RAPS bypasses explicit probability thresholds entirely.

---

## 6. Empirical Results Summary

The notebook ran all five methods across three models on the 7-class scene, targeting 95% coverage ($\alpha = 0.05$). Key observations from the execution output:

| Method | AlexNet Coverage | GFNet Coverage | ViT Coverage | Avg Set Size | Note |
|--------|-----------------|----------------|--------------|-------------|------|
| SplitCP | 0.958 | 0.952 | 0.948 | ~0.95 | Marginal; near-target |
| CcCP | 0.964 | 0.961 | 0.955 | ~0.96 | Slightly conservative |
| RC3P | 0.967 | 0.966 | 0.958 | ~0.96 | Tightest class-cond. |
| ClCP | 0.957 | 0.950 | 0.951 | ~0.95 | Near-target, clustered |
| RAPS | 0.999 | 0.998 | 0.993 | **1.000** | Massively over-covered |

All set sizes have median 1.0 (singleton sets are the norm for this well-calibrated high-accuracy model), meaning the classifier is confident enough on most pixels that only one class passes the threshold. RAPS produces exactly singleton sets for every evaluation sample (100% singleton rate), indicating the regularisation parameters chosen ($\lambda = 0.01$, $k_\text{reg} = 1$) effectively collapse all prediction sets to the top-1 class — at the cost of heavy over-coverage (~99.9% vs the 95% target).

---

## 7. References

[1] Papadopoulos, H., Proedrou, K., Vovk, V., and Gammerman, A. "Inductive Confidence Machines for Regression." *ECML 2002*, Springer, pp. 345–356, 2002. (Founding paper for inductive / split conformal prediction.)

[2] Vovk, V., Gammerman, A., and Shafer, G. *Algorithmic Learning in a Random World.* Springer, 2005. (Canonical reference for conformal prediction theory.) [https://link.springer.com/book/9780387001524](https://link.springer.com/book/9780387001524)

[3] Angelopoulos, A. N., Bates, S., Malik, J., and Jordan, M. I. "Uncertainty Sets for Image Classifiers using Conformal Prediction." *arXiv:2009.14193*, 2020. (Introduces RAPS.) [https://arxiv.org/abs/2009.14193](https://arxiv.org/abs/2009.14193)

[4] Shi, Y., Ghosh, S., Belkhouja, T., Doppa, J. R., and Yan, Y. "Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration." *NeurIPS 2024*. (Introduces RC3P.) [https://arxiv.org/abs/2406.06818](https://arxiv.org/abs/2406.06818)

[5] Ding, T., Angelopoulos, A. N., Bates, S., Jordan, M. I., and Tibshirani, R. J. "Class-Conditional Conformal Prediction with Many Classes." *NeurIPS 2023*. (Introduces Clustered CP.) [https://arxiv.org/abs/2306.09335](https://arxiv.org/abs/2306.09335)

[6] Romano, Y., Sesia, M., and Candès, E. "Classification with Valid and Adaptive Coverage." *NeurIPS 2020*. (Introduces APS, the score function RAPS extends.) [https://arxiv.org/abs/2006.02544](https://arxiv.org/abs/2006.02544)

[7] Angelopoulos, A. N. and Bates, S. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *arXiv:2107.07511*, 2021. [https://arxiv.org/abs/2107.07511](https://arxiv.org/abs/2107.07511)
