# Model Uncertainty Comparison: Theory & Implementation Summary

> **One-line description:** A comparative framework applying five conformal prediction methods — SplitCP, CcCP, RC3P, ClCP, and RAPS — to multispectral image classification, producing guaranteed-coverage prediction sets and spatial uncertainty maps across three deep learning models.

---

## 1. Overview & Intuition

Standard deep classifiers output a single predicted class for each input, but this single-point prediction carries no formal guarantee about how often it is correct. When a model outputs "Class 3" with 85% softmax confidence, that number is not statistically calibrated — it reflects the model's internal score, not a provable coverage probability. In safety-critical domains such as remote sensing and land-cover classification, knowing *when the model is uncertain* is as important as knowing what it predicts.

Conformal prediction (CP) solves this by wrapping any pre-trained classifier in a post-hoc procedure that converts its raw probability outputs into **prediction sets** — sets of candidate classes that are guaranteed to contain the true label with a user-specified probability (e.g., 95%), regardless of the underlying model architecture or data distribution. The only assumption required is that the calibration data and test data are **exchangeable** (loosely, drawn i.i.d. from the same distribution).

This notebook implements and compares five CP variants, each making a different trade-off between coverage type and prediction set size efficiency. Applied to multispectral 6-band satellite patches classified by three models (AlexNet, GFNet, ViT), the framework also produces full spatial uncertainty maps over the entire scene — pixels whose prediction set contains more than one class are marked as uncertain, giving an interpretable spatial view of model confidence.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathcal{X}$ be the input space (multispectral image patches) and $\mathcal{Y} = \{0, 1, \ldots, K-1\}$ a finite set of $K$ classes. A pre-trained classifier $\hat{f}: \mathcal{X} \to \Delta^{K-1}$ maps each patch to a probability simplex vector $\hat{p}(x) = [\hat{p}_1(x), \ldots, \hat{p}_K(x)]$.

The dataset is partitioned into:
- **Calibration set** $\mathcal{D}_{cal} = \{(x_i, y_i)\}_{i=1}^{n}$: used to fit thresholds (never seen by the model during training)
- **Evaluation set** $\mathcal{D}_{eval}$: used to measure empirical coverage and set sizes

All five methods use the calibration set to derive a threshold $\hat{q}$ (or a vector of thresholds), then construct a prediction set $\mathcal{C}(x) \subseteq \mathcal{Y}$ for each test point.

The **nonconformity score** for a calibration point $(x_i, y_i)$ in the simplest variant is:

$$s_i = 1 - \hat{p}_{y_i}(x_i)$$

**Where:**
- $\hat{p}_{y_i}(x_i)$ — the softmax probability assigned to the *true* class $y_i$
- $s_i \in [0, 1]$ — high when the model is wrong or uncertain about the true class; low when the model is confident and correct

**What this means:** A low nonconformity score means the model strongly predicted the correct class; a high score means the model gave it low probability.

### 2.2 Conformal Quantile Threshold

Given $n$ calibration nonconformity scores $\{s_1, \ldots, s_n\}$ and a target miscoverage level $\alpha$ (here $\alpha = 0.05$), the threshold is:

$$\hat{q} = \text{Quantile}\!\left(\{s_1, \ldots, s_n\},\; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

**Where:**
- $\alpha$ — target error rate (here 0.05, seeking 95% coverage)
- $\lceil \cdot \rceil$ — ceiling function; the "+1" adjustment ensures finite-sample validity

**What this means:** $\hat{q}$ is the score value such that at least $1-\alpha$ of the calibration samples have a score at or below it. Choosing `method='higher'` interpolation ensures the guarantee is met even with finite data.

### 2.3 Prediction Set Construction

Given threshold $\hat{q}$, the prediction set for a test point $x$ is:

$$\mathcal{C}(x) = \{y \in \mathcal{Y} : \hat{p}_y(x) \geq 1 - \hat{q}\}$$

**What this means:** Include class $y$ in the set if the model's predicted probability for that class exceeds the threshold $1 - \hat{q}$. If the model is confident, the set will contain only one class (a "singleton"); if uncertain, multiple classes are included.

### 2.4 Marginal Coverage Guarantee (Split CP)

The fundamental theorem of conformal prediction guarantees:

$$\mathbb{P}\!\left(Y_{\text{test}} \in \mathcal{C}(X_{\text{test}})\right) \geq 1 - \alpha$$

This holds for *any* classifier and *any* data distribution satisfying exchangeability — no distributional assumptions, no model assumptions.

### 2.5 Class-Conditional Coverage (CcCP and RC3P)

Marginal coverage ensures the guarantee holds *on average*, but a single threshold $\hat{q}$ may under-cover rare classes and over-cover common ones. Class-conditional methods replace the single threshold with one threshold per class:

$$\hat{q}_c = \text{Quantile}\!\left(\{s_i : y_i = c\},\; \frac{\lceil (n_c + 1)(1-\alpha) \rceil}{n_c}\right)$$

**Where:**
- $n_c = |\{i : y_i = c\}|$ — number of calibration samples for class $c$
- The resulting guarantee is $\mathbb{P}(Y \in \mathcal{C}(X) \mid Y = c) \geq 1 - \alpha$ for each $c$

The prediction set becomes:

$$\mathcal{C}(x) = \{y \in \mathcal{Y} : \hat{p}_y(x) \geq 1 - \hat{q}_y\}$$

### 2.6 RC3P: Rank-Calibrated Class-Conditional Threshold

RC3P (Shi et al., NeurIPS 2024) augments class-conditional CP with **label rank filtering**. For each class $c$, let the **top-$k$ error** be:

$$\text{err}_k^c = 1 - \frac{1}{n_c}\sum_{i: y_i=c} \mathbf{1}\!\left[\text{rank}_i^c \leq k\right]$$

**Where:**
- $\text{rank}_i^c$ — the rank of class $c$ in the sorted probability vector for sample $i$ (rank 1 = highest probability)

A **suitable rank limit** $k_c^*$ is found as the smallest $k$ such that $\text{err}_k^c < \alpha - \delta/\sqrt{n/K}$, where $\delta$ is a truncation gap hyperparameter. The calibration is then restricted to samples where the true class ranked within $k_c^*$, and the prediction set enforces both a score threshold *and* a rank filter:

$$\mathcal{C}(x) = \{y \in \mathcal{Y} : \hat{p}_y(x) \geq 1 - \hat{q}_y^{RC3P} \;\text{ and }\; \text{rank}_y(x) \leq k_y^*\}$$

**What this means:** RC3P prunes candidate classes that are both low-probability *and* low-ranked by the model, yielding smaller prediction sets than CcCP while still guaranteeing class-wise coverage.

### 2.7 Clustered CP (ClCP): Grouping by Feature Similarity

Rather than one threshold per class (which suffers from small calibration counts in rare classes), ClCP groups similar classes into $K'$ clusters and computes one threshold per cluster.

Class embeddings are formed as the mean penultimate-layer feature vector over each class's calibration samples. K-means ($K' = 4$ clusters here) assigns each class $c$ to a cluster $m(c)$. The per-cluster threshold is:

$$\hat{q}_{m} = \text{conformalQhat}\!\left(\{1 - \hat{p}_{y_i}(x_i) : m(y_i) = m\},\; \alpha\right)$$

The prediction set uses the cluster threshold for each candidate class:

$$\mathcal{C}(x) = \{y \in \mathcal{Y} : \hat{p}_y(x) \geq 1 - \hat{q}_{m(y)}\}$$

**What this means:** Classes with similar feature distributions share a threshold, giving more stable estimates than class-wise calibration when some classes are rare.

### 2.8 RAPS: Regularised Adaptive Prediction Sets

RAPS (Angelopoulos et al., 2020) uses a richer nonconformity score that accumulates sorted probabilities and adds a size penalty:

$$s_i^{RAPS} = \sum_{j=1}^{r_i} \hat{p}_{\pi_j}(x_i) + \lambda \cdot \max(r_i - k_{reg},\; 0)$$

**Where:**
- $\pi_1, \pi_2, \ldots$ — classes sorted in descending probability order
- $r_i$ — the rank of the true label $y_i$ in this sorted order
- $\lambda$ — regularisation strength (here $\lambda = 0.01$)
- $k_{reg}$ — regularization onset rank (here $k_{reg} = 1$)

At test time, the prediction set is built by adding classes in probability-descending order until the cumulative sum plus penalty would exceed $\hat{q}$:

$$\mathcal{C}(x) = \{y : y \text{ is included before score exceeds } \hat{q}\}$$

**What this means:** RAPS penalises including tail classes (those ranked far below the top), discouraging overly large prediction sets while preserving the marginal coverage guarantee.

---

## 3. Algorithm

The five methods share a common calibration–evaluation structure. The general procedure is:

**Input:** Pre-trained model $\hat{f}$, calibration set $\mathcal{D}_{cal}$, evaluation set $\mathcal{D}_{eval}$, $\alpha$, full-scene image cube  
**Output:** Prediction sets for evaluation samples, per-class coverage metrics, spatial uncertainty maps

1. Compute softmax probabilities $\hat{p}_{cal} = \hat{f}(x_{cal})$ and $\hat{p}_{eval} = \hat{f}(x_{eval})$
2. Compute nonconformity scores on the calibration set (method-specific formula)
3. Derive threshold(s) $\hat{q}$ (or $\{\hat{q}_c\}$ or $\{\hat{q}_m\}$) from calibration scores at level $\lceil(n+1)(1-\alpha)\rceil/n$
4. Build prediction sets $\mathcal{C}(x)$ for each evaluation sample using the threshold(s)
5. Measure empirical coverage $= \frac{1}{|\mathcal{D}_{eval}|}\sum \mathbf{1}[y_i \in \mathcal{C}(x_i)]$ and average set size
6. For map-based methods (SplitCP, CcCP, RC3P), run inference over every pixel in the full scene and apply the same threshold to produce a spatial uncertainty map

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_uncertainty_comparison.ipynb`

### 4.1 Data Pipeline (Cells 9–11)

```python
def load_multispectral_6band(data_path, label_path, h, w, b):
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(h, w, b)
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(h, w)
    # Per-band min-max normalisation
    for bi in range(b):
        band = x[:, :, bi]
        x_norm[:, :, bi] = (band - band.min()) / max(band.max() - band.min(), 1e-8)
    return x_norm, y
```

**What this does:** Reads the 6-band multispectral image from CSV, reshapes it to a (330, 307, 6) spatial cube, and normalizes each spectral band independently to [0, 1].

**Why:** Band-wise normalisation prevents bands with larger physical magnitudes from dominating the model's feature space.

```python
def extract_labeled_patches(x_img, y_img, patch_size=9):
    coords    = np.argwhere(y_img > 0)   # only labeled pixels
    for i, (r, c) in enumerate(coords):
        x_patches[i] = x_pad[r:r+patch_size, c:c+patch_size, :]
        y_labels[i]  = int(y_img[r, c]) - 1  # 0-based labels
```

**What this does:** Extracts 9×9×6 patches centred on every labeled (non-background) pixel. Labels are converted from 1-based to 0-based indexing.

**Why:** The models operate on local neighbourhood patches, which encode spatial context without requiring full-image inference.

### 4.2 Core Conformal Utilities (Cell 19)

```python
def conformal_qhat(scores, alpha):
    n       = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return safe_quantile(scores, q_level)
```

**What this does:** Computes the calibration threshold $\hat{q}$ using the finite-sample-valid quantile level $\lceil(n+1)(1-\alpha)\rceil / n$, clamped to [0, 1].

**Why:** The $(n+1)$ factor corrects for the fact that we want the guarantee to hold for the *next* test point, not just the calibration set.

### 4.3 Split Conformal Prediction (Cell 26)

```python
def build_split_outputs_for_model(...):
    calib_scores = 1.0 - prob_cal[np.arange(len(y_cal)), y_cal]
    q_hat        = conformal_qhat(calib_scores, alpha)
    pred_sets_eval = prob_eval >= (1.0 - q_hat)
```

**What this does:** Computes the nonconformity score (1 minus the true-class probability) for each calibration point, finds $\hat{q}$, then includes all classes whose predicted probability meets the threshold.

**Why:** This is the simplest CP variant — one threshold for all classes, giving marginal coverage.

### 4.4 Class-Conditional CP (Cell 28)

```python
q_hats = np.zeros(n_classes)
for c in range(n_classes):
    mask = (y_cal == c)
    scores_c  = 1.0 - prob_cal[mask, c]
    q_hats[c] = conformal_qhat(scores_c, alpha)

thresholds     = 1.0 - q_hats.reshape(1, -1)
pred_sets_eval = prob_eval >= thresholds
```

**What this does:** Computes a separate threshold for each class using only calibration samples belonging to that class.

**Why:** A global threshold may systematically under-cover certain classes; per-class thresholds enforce the coverage guarantee for each class independently.

### 4.5 RC3P (Cell 30)

```python
# Build top-k accuracy matrix
ranks = np.argsort(np.argsort(-prob, axis=1), axis=1) + 1
acc_matrix[k-1, c] = np.mean(class_ranks <= k)

# Find minimum k where top-k error < adjusted alpha
tc_alpha = alpha - (truncated_gap / np.sqrt(num_samples_per_class))
suit_k[c] = first k where err_matrix[k-1, c] < tc_alpha

# Calibration restricted to samples where true class is within rank limit
idx = (y_cal == c) & (cal_ranks[:, c] <= test_indices[c])
scores = 1.0 - prob_cal[idx, c]
q_hats[c] = conformal_qhat(scores, test_alphas[c])

# Prediction set: both score AND rank threshold must be met
pred_sets = (prob_eval >= 1 - q_hats) & (eval_ranks <= suit_indices)
```

**What this does:** For each class, finds the minimum rank limit $k_c^*$ that controls top-$k$ error below a tightened alpha, then calibrates only on within-rank-limit samples, and finally enforces both a score and a rank gate at test time.

**Why:** The rank gate prunes tail classes that the model nearly never considers, yielding prediction sets substantially smaller than CcCP while guaranteeing class-wise coverage.

### 4.6 Clustered CP (Cell 32)

```python
# Extract penultimate-layer embeddings
feat_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
emb_cal = feat_model.predict(x_cal, ...)

# Cluster classes by mean embedding
class_means = [emb_cal[y_cal==c].mean(axis=0) for c in range(n_classes)]
km = KMeans(n_clusters=k, ...)
cluster_assignments = km.fit_predict(class_means)

# Per-cluster calibration
for cluster_id in range(k):
    cls  = np.where(cluster_assignments == cluster_id)[0]
    mask = np.isin(y_cal, cls)
    scores = 1.0 - prob_cal[mask, y_cal[mask]]
    q_hats_per_cluster[cluster_id] = conformal_qhat(scores, alpha)
```

**What this does:** Extracts deep feature embeddings from the penultimate model layer, computes a mean embedding per class, clusters classes with K-means, then calibrates one threshold per cluster.

**Why:** Classes with similar feature-space representations tend to have similar nonconformity score distributions — sharing a threshold across such classes gives better-calibrated estimates than using all data (too coarse) or one class alone (too few samples).

### 4.7 RAPS (Cell 34)

```python
def raps_score_single(prob_row, true_label, lam=0.01, k_reg=1):
    order      = np.argsort(prob_row)[::-1]
    rank       = int(np.where(order == true_label)[0][0])
    cumulative = float(np.sum(prob_row[order[:rank]]))
    penalty    = float(lam) * max(rank - k_reg, 0)
    return cumulative + penalty
```

**What this does:** Sums the model's probabilities for all classes ranked above the true class, then adds a penalty proportional to how far down the rank the true class appears (beyond the `k_reg` onset).

**Why:** The cumulative sum score captures how much probability mass sits above the true label. The penalty discourages large sets by penalising samples where the true class is deeply buried — this regularises score calibration and yields tighter prediction sets.

### 4.8 Spatial Uncertainty Maps (Cells 26, 28, 30)

```python
pred_sets_full = prob_full >= (1.0 - q_hat)   # shape: (H, W, K)
set_sizes_map  = np.sum(pred_sets_full, axis=2)  # shape: (H, W)
combined_map   = np.where(set_sizes_map == 1, pred_class_map, n_classes)
```

**What this does:** Applies the calibrated threshold to every pixel's probability vector in the full scene. Pixels with a singleton prediction set are assigned their class; pixels with multi-class sets are marked as "uncertain" (class index $K$).

**Why:** This converts abstract statistical uncertainty into a spatial mask that directly shows *where* in the image the classifier is reliable and where it is not, which is operationally valuable for downstream analysis.

---

## 5. Worked Numerical Example

**Setup:** 5 calibration samples, 3 classes, $\alpha = 0.05$. Softmax probabilities and true labels:

| Sample | $\hat{p}_0$ | $\hat{p}_1$ | $\hat{p}_2$ | True $y$ |
|--------|-------------|-------------|-------------|----------|
| 1      | 0.70        | 0.20        | 0.10        | 0        |
| 2      | 0.10        | 0.80        | 0.10        | 1        |
| 3      | 0.60        | 0.25        | 0.15        | 0        |
| 4      | 0.05        | 0.10        | 0.85        | 2        |
| 5      | 0.50        | 0.35        | 0.15        | 0        |

**Step 1 — Compute nonconformity scores** (Split CP):
$$s_1 = 1 - 0.70 = 0.30,\quad s_2 = 1 - 0.80 = 0.20,\quad s_3 = 1 - 0.60 = 0.40$$
$$s_4 = 1 - 0.85 = 0.15,\quad s_5 = 1 - 0.50 = 0.50$$

Sorted scores: $\{0.15, 0.20, 0.30, 0.40, 0.50\}$

**Step 2 — Compute $\hat{q}$:**
$$q_{level} = \frac{\lceil (5+1)(1-0.05) \rceil}{5} = \frac{\lceil 5.70 \rceil}{5} = \frac{6}{5} \to \text{clamped to } 1.0$$

With only 5 calibration points and $\alpha = 0.05$, the quantile level saturates at 1.0, so $\hat{q} = 0.50$ (the maximum score).

**Step 3 — Build prediction set for a test point** $x_{test}$ with $\hat{p} = [0.55, 0.30, 0.15]$:
Threshold = $1 - \hat{q} = 1 - 0.50 = 0.50$

$$\mathcal{C}(x_{test}) = \{y : \hat{p}_y \geq 0.50\} = \{0\}$$

Class 0 ($\hat{p}_0 = 0.55 \geq 0.50$) is included; classes 1 and 2 are not. The result is a singleton set — the model is sufficiently confident.

**Step 4 — With a more uncertain test point** $\hat{p} = [0.40, 0.35, 0.25]$:
$$\mathcal{C}(x_{test}) = \{y : \hat{p}_y \geq 0.50\} = \{\}$$

Since no class exceeds 0.50, the set would be empty. In the notebook, RAPS guarantees at least the top-1 class is always included; for Split CP, this signals high uncertainty — none of the classes pass the threshold.

**Interpretation:** A prediction set $\{0\}$ means the classifier is confident and the corresponding pixel would be coloured as class 0 on the spatial map. A multi-class or empty set would be coloured grey (uncertain).

---

## 6. References

[1] Shi, Y., Ghosh, S., Belkhouja, T., Doppa, J. R., and Yan, Y. "Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration." *NeurIPS 2024*. [arXiv:2406.06818](https://arxiv.org/abs/2406.06818)

[2] Angelopoulos, A. N., Bates, S., Malik, J., and Jordan, M. I. "Uncertainty Sets for Image Classifiers using Conformal Prediction." *ICLR 2021*. [arXiv:2009.14193](https://arxiv.org/abs/2009.14193)

[3] Angelopoulos, A. N. and Bates, S. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *arXiv 2021*. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)

[4] Ding, T., Angelopoulos, A., Bates, S., Jordan, M., and Tibshirani, R. "Class-Conditional Conformal Prediction with Many Classes." *NeurIPS 2023*. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/cb931eddd563f8d473c355518ce8601c-Paper-Conference.pdf)

[5] Romano, Y., Sesia, M., and Candès, E. J. "Classification with Valid and Adaptive Coverage." *NeurIPS 2020*. [arXiv:2006.02544](https://arxiv.org/abs/2006.02544)
