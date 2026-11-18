# Multi-Head Conformal Prediction (MultiCP): Theory & Implementation Summary

> **One-line description:** MultiCP applies split conformal prediction independently across K parallel output heads of a neural network, intersects their per-head prediction sets, and uses the resulting joint coverage and average set size to quantify pixel-level uncertainty in multispectral image classification.

---

## 1. Overview & Intuition

Uncertainty quantification in deep learning classifiers is a long-standing challenge: standard softmax outputs are frequently overconfident, and their raw probability values do not reliably reflect true class membership uncertainty. Conformal Prediction (CP) addresses this by providing a statistically rigorous wrapper around any pre-trained model, constructing *prediction sets* — sets of candidate classes that are guaranteed to contain the true label with at least a user-specified probability — without placing any distributional assumptions on the data beyond exchangeability.

Standard split CP applies a single score function to a single model output. The limitation is that the resulting prediction set reflects only one view of the model's output distribution. In complex multi-spectral remote sensing applications, where land-cover classes can be spectrally ambiguous, a single conformal pass may yield uninformative sets — either trivially small (missing the true class) or excessively large (including many irrelevant classes).

MultiCP extends split CP to architectures with multiple parallel output heads. Each head is trained with the same backbone but through a progressive structured-dropout training schedule that forces each head to learn complementary representations of the same input. At inference time, each head independently produces a softmax probability vector, and CP is applied to each head separately. The intersection of all per-head prediction sets produces the final joint set: only classes that survive every head's conformal threshold are included. The *joint coverage* — the fraction of test pixels where the true class survives all heads simultaneously — is the key metric. As the number of heads grows, the coverage and the average set size both evolve, and the head-sweep plot captures this progression.

For spatial applications such as full-scene multispectral mapping, MultiCP additionally assigns each pixel a *normalised set size* (average set size divided by number of classes), which becomes a continuous uncertainty signal. Pixels in the top fraction of this signal are labelled *uncertain* and masked in the final classification map, producing a binary certain/uncertain spatial map with a direct probabilistic interpretation.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathcal{X} = \mathbb{R}^{P_S \times P_S \times B}$ be the input space of spectral patches of spatial extent $P_S \times P_S$ and $B$ spectral bands, and let $\mathcal{Y} = \{0, 1, \ldots, C-1\}$ be the set of $C$ land-cover classes.

A multi-head model $f: \mathcal{X} \to (\Delta^{C-1})^K$ maps each patch to $K$ independent softmax probability vectors:
$$f(x) = \bigl(\hat{\mathbf{p}}^{(1)}(x), \hat{\mathbf{p}}^{(2)}(x), \ldots, \hat{\mathbf{p}}^{(K)}(x)\bigr)$$

where $\hat{\mathbf{p}}^{(k)}(x) \in \Delta^{C-1}$ denotes the softmax output of head $k$, and $\Delta^{C-1}$ is the $(C-1)$-simplex (vectors summing to 1 with non-negative entries).

The dataset is split three ways: a training set $\mathcal{D}_{\text{train}}$, a calibration set $\mathcal{D}_{\text{cal}} = \{(x_i, y_i)\}_{i=1}^{N}$, and a test set $\mathcal{D}_{\text{test}}$. All three are assumed to be drawn exchangeably from the same underlying distribution.

Within $\mathcal{D}_{\text{cal}}$, the notebook further partitions it into a *cell-selection set* $\mathcal{D}_{\text{cells}}$ (a small random fraction, default 5%) and a *re-calibration set* $\mathcal{D}_{\text{re-cal}}$. The cell-selection set is used purely for Voronoi visualisation; the re-calibration set drives the conformal thresholds.

The error rate $\alpha \in (0,1)$ is set by the user (default $\alpha = 0.05$, targeting 95% coverage). The number of heads is $K = 7$ and the number of classes is $C = 7$.

### 2.2 Nonconformity Score Functions

For each head $k$, a nonconformity score $s^{(k)}(x, y)$ measures how *unusual* it is to assign class $y$ to input $x$. Two score functions are used in the notebook.

#### 2.2.1 RAPS (Regularized Adaptive Prediction Sets)

RAPS, introduced by Angelopoulos et al. (2021), builds on the Adaptive Prediction Set (APS) score by adding a regularisation term that penalises including tail classes. Let $\hat{\pi}^{(k)}_{(j)}(x)$ be the $j$-th largest softmax probability from head $k$, and let $o_x^{(k)}(y)$ be the rank of class $y$ among all classes sorted in descending probability order (rank 1 = most probable). Then:

$$s^{(k)}_{\text{RAPS}}(x, y) = \sum_{j=1}^{o_x^{(k)}(y)-1} \hat{\pi}^{(k)}_{(j)}(x) + u \cdot \hat{\pi}^{(k)}_{(o)}(x) + \lambda \cdot \bigl(o_x^{(k)}(y) - k_{\text{reg}}\bigr)^+$$

**Where:**
- $\hat{\pi}^{(k)}_{(j)}(x)$ — the $j$-th ranked softmax probability under head $k$
- $o_x^{(k)}(y)$ — the rank of the true class $y$ in descending probability order under head $k$
- $u \sim \mathcal{U}[0,1]$ — a uniform random tiebreaker for randomised coverage
- $\lambda$ — regularisation strength (penalises large sets)
- $k_{\text{reg}}$ — rank threshold below which regularisation kicks in
- $(\cdot)^+$ — the positive part, i.e. $\max(0, \cdot)$

**What this means:** The RAPS score accumulates the probability mass of all classes more confident than $y$, adds a fractional contribution from $y$ itself, and adds a penalty for any class ranked lower than $k_{\text{reg}}$. A low score means the model places $y$ near the top of its ranking; a high score means the model is uncertain about or actively disfavours $y$.

#### 2.2.2 SAPS (Sorted Adaptive Prediction Sets)

SAPS, introduced by Huang et al. (2024), discards all probability values except the maximum softmax probability, replacing cumulative mass with rank-based information:

$$s^{(k)}_{\text{SAPS}}(x, y) = \begin{cases} u \cdot \hat{\pi}^{(k)}_{\max}(x) & \text{if } o_x^{(k)}(y) = 1 \\ \hat{\pi}^{(k)}_{\max}(x) + \bigl(o_x^{(k)}(y) - 2 + u\bigr)\lambda & \text{otherwise} \end{cases}$$

**Where:**
- $\hat{\pi}^{(k)}_{\max}(x)$ — the maximum softmax probability under head $k$
- $o_x^{(k)}(y)$ — the rank of class $y$ under head $k$
- $\lambda$ — a weight controlling how much the rank contributes to the score
- $u \sim \mathcal{U}[0,1]$ — random tiebreaker

**What this means:** SAPS retains only the top probability as a scale anchor, and uses the rank of the true class to determine set inclusion. This reduces sensitivity to miscalibration in the softmax tail, which is common when models are overconfident.

### 2.3 Per-Head Calibration Threshold

For each head $k$, the nonconformity scores are computed on the re-calibration set $\mathcal{D}_{\text{re-cal}}$ of size $n$:

$$s^{(k)}_i = s^{(k)}(x_i, y_i), \quad i = 1, \ldots, n$$

These scores are collected along the true-class dimension: the score for sample $i$ is the nonconformity of its true label under head $k$. The calibration threshold for head $k$ is then the empirical $(1-\alpha)$ quantile:

$$\hat{q}^{(k)} = \text{Quantile}_{1-\alpha}\bigl(s^{(k)}_1, s^{(k)}_2, \ldots, s^{(k)}_n\bigr)$$

In the notebook, this is implemented as:
```python
cal_true = Dre_cal_scores[np.arange(K)[:, None], np.arange(N_cal), Dre_cal_target]
q = np.quantile(cal_true, 1 - alpha, axis=1)
```

Each of the $K$ heads yields its own threshold $\hat{q}^{(k)}$.

### 2.4 Per-Head Prediction Sets

For a test sample $x_{\text{test}}$, the prediction set under head $k$ is:

$$\mathcal{C}^{(k)}(x_{\text{test}}) = \bigl\{ y \in \mathcal{Y} : s^{(k)}(x_{\text{test}}, y) \leq \hat{q}^{(k)} \bigr\}$$

A class is included if and only if its nonconformity score falls within the calibrated threshold. This is implemented as a boolean mask over the score matrix:

```python
pred_sets = test_scores <= q[:, None, None]
# shape: (K, N_test, C) — True means class c is in head k's prediction set for sample i
```

### 2.5 Coverage Guarantee (Marginal)

For any single head $k$, split CP guarantees marginal coverage: if the calibration and test data are exchangeable, then

$$\Pr\bigl(Y_{\text{test}} \in \mathcal{C}^{(k)}(X_{\text{test}})\bigr) \geq 1 - \alpha$$

This follows directly from the quantile construction: setting $\hat{q}^{(k)}$ at the $(1-\alpha)$ empirical quantile ensures that at most an $\alpha$ fraction of calibration true-class scores exceed the threshold (Vovk et al., 2005; Shafer and Vovk, 2008).

### 2.6 Joint Coverage Across All Heads (MultiCP)

The joint prediction set is the intersection of all per-head sets:

$$\mathcal{C}_{\text{joint}}(x) = \bigcap_{k=1}^{K} \mathcal{C}^{(k)}(x)$$

A test sample is *covered* if and only if the true class survives all $K$ per-head thresholds simultaneously:

$$\text{covered}_i = \bigwedge_{k=1}^{K} \mathbf{1}\bigl[y_i \in \mathcal{C}^{(k)}(x_i)\bigr]$$

The empirical joint coverage is:

$$\widehat{\text{cov}} = \frac{1}{N_{\text{test}}} \sum_{i=1}^{N_{\text{test}}} \text{covered}_i$$

In the notebook:
```python
covered = np.all(pred_sets[np.arange(K)[:, None], np.arange(N_valid), test_target[valid]], axis=0)
return covered.mean(), pred_sets.sum(axis=2).mean(), pred_sets
```

**Important:** The joint coverage is not independently guaranteed at $1-\alpha$. Each head individually has the $1-\alpha$ coverage guarantee, but the intersection can be stricter (lower coverage) or more permissive depending on how correlated the heads are. The head-sweep plot in the notebook tracks *empirical* coverage as heads are added from $k=1$ to $k=K$, making the relationship between head count and coverage a key diagnostic.

### 2.7 Average Set Size

The average set size across the test set is a measure of *efficiency*: the smaller the sets, the more informative the predictions. It is computed as:

$$\overline{|S|} = \frac{1}{K \cdot N_{\text{test}}} \sum_{k=1}^{K} \sum_{i=1}^{N_{\text{test}}} |\mathcal{C}^{(k)}(x_i)|$$

In practice, the notebook averages the boolean set-membership tensor over the heads and samples axes: `pred_sets.sum(axis=2).mean()`.

### 2.8 Normalised Uncertainty Score and Binary Spatial Map

For each pixel in the full scene, let $\bar{S}(x)$ be the average set size across all $K$ heads:

$$u(x) = \frac{\bar{S}(x)}{C}$$

This normalises the set size to $[0, 1]$, with $u=0$ meaning every head produces a singleton set and $u=1$ meaning every head includes all classes.

A pixel is declared *uncertain* if its normalised uncertainty exceeds the $(1 - \xi)$ quantile of $u$ across all scene pixels, where $\xi = \texttt{UNCERTAIN\_FRACTION} = 0.10$:

$$\text{uncertain}(x) = \mathbf{1}\bigl[u(x) \geq \hat{\tau}_\xi\bigr], \quad \hat{\tau}_\xi = Q_{1-\xi}(\{u(x_j)\}_j)$$

Pixels with ground-truth label 7 (unlabelled background) are also forced to *uncertain*. The resulting binary map has direct probabilistic meaning: the certain region is the part of the scene where the MultiCP procedure, at the chosen error rate, is confident enough to assign a single class prediction.

---

## 3. Algorithm

**Input:** Pre-trained multi-head model $f$, calibration set $\mathcal{D}_{\text{cal}}$, test set $\mathcal{D}_{\text{test}}$, error rate $\alpha$, scoring method $\in \{\text{RAPS}, \text{SAPS}\}$, head sweep range $1 \ldots K$, uncertain fraction $\xi$

**Output:** Head-sweep DataFrame (coverage & set size vs. heads), per-class coverage DataFrame, binary uncertainty map, prediction class map, pixel counts

1. **Extract multi-head outputs:** Run the model in prediction-only mode on $\mathcal{D}_{\text{cal}}$ and $\mathcal{D}_{\text{test}}$, stacking head outputs to shape $(K, N, C)$.

2. **Compute nonconformity scores:** Apply the chosen score function (RAPS or SAPS) element-wise via `compute_scores`. Result shape: $(K, N, C)$.

3. **Split calibration set:** Randomly separate a small fraction of calibration samples into $\mathcal{D}_{\text{cells}}$ (for Voronoi visualisation); the remainder forms $\mathcal{D}_{\text{re-cal}}$.

4. **Per-head threshold computation (re-calibration):** For each head $k$, extract the score of the true class for every sample in $\mathcal{D}_{\text{re-cal}}$, and compute the $(1-\alpha)$ empirical quantile $\hat{q}^{(k)}$.

5. **Prediction sets on test data:** For each head $k$ and test sample $i$, include class $c$ if $s^{(k)}(x_i, c) \leq \hat{q}^{(k)}$, producing boolean tensor of shape $(K, N_{\text{test}}, C)$.

6. **Joint coverage and set size:** A sample is covered if and only if the true class is included in all $K$ heads. Average the boolean results to obtain empirical coverage and average set size.

7. **Head sweep:** Repeat steps 3–6 for each prefix $k = 1, \ldots, K$, recording (coverage, set size) at each step.

8. **Per-class coverage:** For each class $c$, compute the fraction of test samples of class $c$ that are covered by the joint prediction set.

9. **Full-scene binary mapping:**
   a. Extract every pixel patch from the padded scene image.
   b. Run multi-head inference on all scene patches.
   c. Apply the score function and conformal thresholds from step 4.
   d. Compute normalised uncertainty $u(x)$ for each pixel.
   e. Threshold at $(1-\xi)$ quantile to produce binary certain/uncertain map.
   f. Force background pixels (label 7) to uncertain.

10. **Visualise and export:** Generate head-sweep line charts, per-class coverage bar charts, binary uncertainty map, class prediction map, pixel count bar chart, Voronoi cell-selection diagram. Write all results to Excel.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_uncertainty_multicp.md`

### 4.1 Multi-Head Output Extraction

```python
def get_multihead_outputs(model, x_data, batch_size=128):
    outputs = model.predict(x_data, batch_size=batch_size, verbose=0)
    if not isinstance(outputs, list):
        outputs = [outputs]
    return np.stack(outputs, axis=0)
```

**What this does:** Runs a single forward pass through the multi-head Keras model. Because Keras returns a list of head outputs (one per `Dense` output layer), this function stacks them into a single 3-D array of shape $(K, N, C)$.

**Why:** The stacked representation is the natural tensor layout for vectorised per-head operations in NumPy. All downstream conformal math is written to operate on this $(K, N, C)$ shape, so this function is the data-format contract between the model and the CP machinery.

### 4.2 Calibration Set Partitioning

```python
def generate_Dcal_Dcells_sets(cal_scores, cal_target, fraction=0.05, seed=42):
    K, N, _ = cal_scores.shape
    rng = np.random.default_rng(seed)
    n_cells = max(1, int(N * fraction))
    idx_cells = rng.choice(N, n_cells, replace=False)

    Dcells_scores = cal_scores[:, idx_cells, cal_target[idx_cells].astype(int)].T
    Dcells_target = cal_target[idx_cells]

    mask = np.ones(N, dtype=bool); mask[idx_cells] = False
    Dre_cal_scores = cal_scores[:, mask, :]
    Dre_cal_target = cal_target[mask]
    return Dcells_scores, Dcells_target, Dre_cal_scores, Dre_cal_target
```

**What this does:** Randomly selects 5% of calibration samples as the *cell-selection* set $\mathcal{D}_{\text{cells}}$. For these samples, only the true-class score is retained (shape: $n_{\text{cells}} \times K$), which is used later to drive the Voronoi ordering visualisation. The remaining 95% forms $\mathcal{D}_{\text{re-cal}}$, which carries the full score tensor $(K, N_{\text{re-cal}}, C)$ and is used to compute the conformal thresholds.

**Why:** Separating the visualisation set from the threshold-computation set avoids any circularity. The Voronoi diagram of $\mathcal{D}_{\text{cells}}$ scores provides intuition about how much of the calibration space is visited; the re-cal set provides the statistical quantile estimates.

### 4.3 Core Conformal Inference: `main_algo`

```python
def main_algo(Dcells_scores, Dcells_target, Dre_cal_scores, Dre_cal_target,
              test_scores, test_target, alpha, config):
    K, N_cal = Dre_cal_scores.shape[0], Dre_cal_scores.shape[1]
    # Extract true-class scores from re-calibration set
    cal_true = Dre_cal_scores[np.arange(K)[:, None], np.arange(N_cal), Dre_cal_target]
    # Per-head (1-alpha) quantile threshold
    q = np.quantile(cal_true, 1 - alpha, axis=1)
    # Prediction sets: shape (K, N_test, C)
    pred_sets = test_scores <= q[:, None, None]
    # Coverage: true class included in all K heads?
    valid = (test_target >= 0) & (test_target < pred_sets.shape[2])
    covered = np.all(pred_sets[np.arange(K)[:, None],
                               np.arange(np.sum(valid)),
                               test_target[valid]], axis=0)
    return covered.mean(), pred_sets.sum(axis=2).mean(), pred_sets
```

**What this does:**
- `cal_true`: shape $(K, N_{\text{re-cal}})$ — true-class score for each calibration sample under each head.
- `q`: shape $(K,)$ — per-head $(1-\alpha)$ quantile; the conformal threshold vector.
- `pred_sets`: boolean $(K, N_{\text{test}}, C)$ — `True` means class $c$ is in head $k$'s prediction set for test sample $i$.
- `covered`: boolean vector of length $N_{\text{test}}$ — `True` if and only if the true class passes all $K$ per-head thresholds.

**Why:** The per-head thresholds $q[k]$ are broadcast across all test samples and all classes simultaneously via NumPy broadcasting, keeping the implementation compact and avoiding explicit loops over heads or samples. The `np.all(..., axis=0)` computes the logical AND across all heads for each test sample — this is precisely the intersection of per-head prediction sets.

### 4.4 Head Sweep

```python
def compute_head_sweep(cal_output, test_output, cal_target, test_target, scoring_method):
    config = {'ALPHA': ALPHA, 'SCORING_METHOD': scoring_method}
    cal_scores  = np.round(compute_scores(cal_output,  config), 4)
    test_scores = np.round(compute_scores(test_output, config), 4)
    rows, last_bundle = [], None
    for nH in range(1, cal_output.shape[0] + 1):
        Dc, Dt, Rc, Rt = generate_Dcal_Dcells_sets(cal_scores[:nH], cal_target)
        cov, msz, pred_sets = main_algo(Dc, Dt, Rc, Rt, test_scores[:nH],
                                        test_target, ALPHA, config)
        rows.append({'heads': nH, 'coverage': float(cov), 'set_size': float(msz)})
        if nH == cal_output.shape[0]:
            last_bundle = (config, Dc, Dt, Rc, Rt, pred_sets)
    return pd.DataFrame(rows), last_bundle
```

**What this does:** Runs the full conformal evaluation for each prefix of heads $\{1, 2, \ldots, K\}$. At each step, the calibration and test score tensors are sliced to the first `nH` heads, thresholds are computed, and coverage plus average set size are recorded. The final-head bundle (thresholds and prediction sets) is returned for use in the full-scene mapping.

**Why:** The progressive head-sweep diagnostic is the core contribution of MultiCP: it reveals whether adding more heads consistently improves coverage or at what number of heads the method reaches its target coverage. If coverage drops below $1-\alpha$ as more heads are added, it signals that the additional heads are providing non-redundant, distinct conformal constraints — which may also reduce set sizes.

### 4.5 Binary Uncertainty Mapping for the Full Scene

```python
def build_binary_uncertainty_outputs(model, padded_x, y_raw, config, Dc, Dt, Rc, Rt):
    image_outputs = get_image_multi_head_outputs(model, padded_x, H, W, B, P_S, BATCH_SIZE)
    image_scores  = np.round(compute_scores(image_outputs, config), 4)
    ...
    cov, mset, pred_bool = main_algo(Dc, Dt, Rc, Rt, img_valid, y_valid, config['ALPHA'], config)
    set_sizes = pred_bool.sum(axis=2).mean(axis=0)
    u_valid   = set_sizes / float(NUM_CLASSES)
    thresh    = np.nanquantile(u_valid, 1 - UNCERTAIN_FRACTION)
    cp_uncertain_valid = u_valid >= thresh
    ...
```

**What this does:** Applies the model and the conformal thresholds (derived from the calibration set) to every pixel in the full $H \times W$ scene. The per-pixel average set size is normalised to $[0,1]$ and thresholded at the $(1-\xi)$ quantile (default top 10%) to produce the binary certain/uncertain mask.

**Why:** The conformal thresholds from the calibration phase are *fixed* — they are not re-computed using the scene pixels. Reusing them on the full scene provides a direct uncertainty map with the same coverage semantics as the test-set evaluation, as long as the scene pixels are from the same distribution as the calibration data.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

We use a toy dataset with $K = 3$ heads, $C = 4$ classes (Class 0–3), and $\alpha = 0.10$ (targeting 90% coverage). We have $n = 12$ calibration samples (3 per class) and 3 test samples designed to expose method differences.

**Calibration probability matrix** (softmax outputs, rows sum to 1.00; bold = true class probability)

| Sample | True $y$ | $\hat{p}_0$ | $\hat{p}_1$ | $\hat{p}_2$ | $\hat{p}_3$ |
|--------|-----------|-------------|-------------|-------------|-------------|
| 0 | 0 | **0.70** | 0.15 | 0.10 | 0.05 |
| 1 | 0 | **0.60** | 0.20 | 0.12 | 0.08 |
| 2 | 0 | **0.55** | 0.22 | 0.13 | 0.10 |
| 3 | 1 | 0.10 | **0.65** | 0.15 | 0.10 |
| 4 | 1 | 0.12 | **0.58** | 0.20 | 0.10 |
| 5 | 1 | 0.15 | **0.50** | 0.22 | 0.13 |
| 6 | 2 | 0.08 | 0.12 | **0.72** | 0.08 |
| 7 | 2 | 0.10 | 0.15 | **0.62** | 0.13 |
| 8 | 2 | 0.12 | 0.18 | **0.55** | 0.15 |
| 9 | 3 | 0.05 | 0.10 | 0.15 | **0.70** |
| 10 | 3 | 0.08 | 0.12 | 0.18 | **0.62** |
| 11 | 3 | 0.10 | 0.15 | 0.22 | **0.53** |

For simplicity, we assume the three heads yield identical score outputs in this toy example (illustrating the single-score-function path). We use the simplified *THR/LAC* score (which MultiCP uses internally via `compute_scores`): $s(x, y) = 1 - \hat{p}_y$.

**Test probability matrix**

| Sample | True $y$ | $\hat{p}_0$ | $\hat{p}_1$ | $\hat{p}_2$ | $\hat{p}_3$ | Description |
|--------|-----------|-------------|-------------|-------------|-------------|-------------|
| T0 | 2 | 0.05 | 0.08 | **0.80** | 0.07 | Easy: model clearly correct |
| T1 | 1 | 0.15 | **0.42** | 0.28 | 0.15 | Borderline: true class has moderate probability |
| T2 | 0 | **0.28** | 0.27 | 0.25 | 0.20 | Ambiguous: no class dominates |

---

### 5.1 RAPS Score Computation (Simplified, $\lambda=0$, $k_{\text{reg}}=\infty$, $u=0$)

With $\lambda = 0$ and $u = 0$, RAPS reduces to $s(x, y) = \sum_{j: \hat{p}_{(j)} > \hat{p}_y} \hat{p}_{(j)}$ — the cumulative mass of classes ranked above $y$.

#### Step A — Calibration Scores

For each calibration sample, we compute $s(x_i, y_i)$ = probability mass of classes ranked above the true class.

| Sample | True $y$ | $\hat{p}_{y_i}$ | Classes ranked above | $s_i$ |
|--------|-----------|-----------------|----------------------|--------|
| 0 | 0 | 0.70 | none | 0.00 |
| 1 | 0 | 0.60 | none | 0.00 |
| 2 | 0 | 0.55 | none | 0.00 |
| 3 | 1 | 0.65 | none | 0.00 |
| 4 | 1 | 0.58 | none | 0.00 |
| 5 | 1 | 0.50 | none | 0.00 |
| 6 | 2 | 0.72 | none | 0.00 |
| 7 | 2 | 0.62 | none | 0.00 |
| 8 | 2 | 0.55 | none | 0.00 |
| 9 | 3 | 0.70 | none | 0.00 |
| 10 | 3 | 0.62 | none | 0.00 |
| 11 | 3 | 0.53 | none | 0.00 |

All calibration true classes happen to be ranked first (the model is well-calibrated on calibration data in this toy), so all $s_i = 0.00$.

Sorted scores: $[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]$.

#### Step B — Calibration Threshold

$$q_{\text{level}} = \frac{\lceil (n+1)(1-\alpha) \rceil}{n} = \frac{\lceil 13 \times 0.90 \rceil}{12} = \frac{\lceil 11.7 \rceil}{12} = \frac{12}{12} = 1.00$$

The quantile level clips to 1.00, so $\hat{q} = \max(s_i) = 0.00$.

**Interpretation:** Because all calibration true-class scores are 0, the threshold is 0. Under RAPS, a class $y$ is in the prediction set if its cumulative mass above it does not exceed 0 — meaning only the *top-ranked* class is included for each test sample.

#### Step C — Prediction Sets for Test Samples

For each test sample, the prediction set = $\{c : s(x_{\text{test}}, c) \leq 0.00\}$ = $\{$ top-ranked class $\}$.

**Test T0** (true class = 2, $\hat{\mathbf{p}} = [0.05, 0.08, 0.80, 0.07]$):

| Class | $\hat{p}_c$ | Classes above | Score | $\leq 0$? | In set? |
|-------|-------------|---------------|-------|-----------|---------|
| 0 | 0.05 | {2,1,3} | 0.80+0.08+0.07=0.95 | No | ✗ |
| 1 | 0.08 | {2,3} | 0.80+0.07=0.87 | No | ✗ |
| **2** | **0.80** | none | 0.00 | Yes | **✓** |
| 3 | 0.07 | {2,1} | 0.80+0.08=0.88 | No | ✗ |

Prediction set: $\{2\}$. True class (2) covered: ✓

**Test T1** (true class = 1, $\hat{\mathbf{p}} = [0.15, 0.42, 0.28, 0.15]$):

| Class | $\hat{p}_c$ | Classes above | Score | $\leq 0$? | In set? |
|-------|-------------|---------------|-------|-----------|---------|
| 0 | 0.15 | {1,2,3} | 0.42+0.28+0.15=0.85 | No | ✗ |
| **1** | **0.42** | none | 0.00 | Yes | **✓** |
| 2 | 0.28 | {1} | 0.42 | No | ✗ |
| 3 | 0.15 | {1,2} | 0.42+0.28=0.70 | No | ✗ |

Prediction set: $\{1\}$. True class (1) covered: ✓

**Test T2** (true class = 0, $\hat{\mathbf{p}} = [0.28, 0.27, 0.25, 0.20]$):

| Class | $\hat{p}_c$ | Classes above | Score | $\leq 0$? | In set? |
|-------|-------------|---------------|-------|-----------|---------|
| **0** | **0.28** | none | 0.00 | Yes | **✓** |
| 1 | 0.27 | {0} | 0.28 | No | ✗ |
| 2 | 0.25 | {0,1} | 0.55 | No | ✗ |
| 3 | 0.20 | {0,1,2} | 0.80 | No | ✗ |

Prediction set: $\{0\}$. True class (0) covered: ✓

#### Step D — Summary Table (RAPS)

| Test Sample | Prediction Set | Set Size | Coverage |
|-------------|----------------|----------|----------|
| T0 (Easy) | $\{2\}$ | 1 | ✓ |
| T1 (Borderline) | $\{1\}$ | 1 | ✓ |
| T2 (Ambiguous) | $\{0\}$ | 1 | ✓ |

---

### 5.2 SAPS Score Computation ($\lambda = 0.1$, $u = 0$)

For SAPS, the score when the true class is *not* top-ranked is: $s(x,y) = \hat{p}_{\max} + (o_x(y) - 2) \times \lambda$. When the true class *is* top-ranked ($o = 1$): $s = 0$.

#### Step A — Calibration SAPS Scores

With $\lambda = 0.1$ and $u = 0$, for each calibration sample where the true class is top-ranked, $s_i = 0$. As in the calibration table above, all true classes are top-ranked, so all $s_i^{\text{SAPS}} = 0$.

Sorted scores: $[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]$.

#### Step B — SAPS Calibration Threshold

$$\hat{q}^{\text{SAPS}} = Q_{0.90}([0, \ldots, 0]) = 0.00$$

Same as RAPS in this toy (all scores are 0).

#### Step C — SAPS Prediction Sets

For test samples, SAPS includes class $c$ if $s^{\text{SAPS}}(x, c) \leq 0.00$ — this means only the top-ranked class gets score 0 (when $u=0$).

The result is identical to RAPS in this toy example. The difference between RAPS and SAPS becomes evident in more ambiguous scenarios where the softmax tail is miscalibrated: SAPS discards the full tail and relies only on the top probability scale, while RAPS accumulates the actual cumulative mass, making RAPS more sensitive to tail miscalibration.

---

### 5.3 Multi-Head Intersection: Joint Prediction Sets

Assume three heads each individually produce the singleton sets above (all three heads agree in this toy). The joint intersection:

| Test Sample | Head 1 Set | Head 2 Set | Head 3 Set | Joint Set | Coverage |
|-------------|-----------|-----------|-----------|-----------|----------|
| T0 | $\{2\}$ | $\{2\}$ | $\{2\}$ | $\{2\}$ | ✓ |
| T1 | $\{1\}$ | $\{1\}$ | $\{1\}$ | $\{1\}$ | ✓ |
| T2 | $\{0\}$ | $\{0\}$ | $\{0\}$ | $\{0\}$ | ✓ |

**Empirical joint coverage:** 3/3 = 1.00 (above the 90% target). **Average set size:** 1.00.

In a real scenario where the heads have different trained weights, each head's per-class thresholds will differ, and borderline test samples (like T1 and T2) will often be excluded from some but not all heads' sets — producing either empty sets (below the coverage target) or multi-class sets that reflect genuine ambiguity.

---

### 5.4 Normalised Uncertainty and Binary Map

For the scene, assume three test pixels with these average set sizes over 3 heads:

| Pixel | Avg set size | Normalised $u$ |
|-------|-------------|-----------------|
| P0 | 1.0 | 1.0/4 = 0.25 |
| P1 | 2.5 | 2.5/4 = 0.625 |
| P2 | 3.8 | 3.8/4 = 0.95 |

With $\xi = 0.10$ (top 10% uncertain):
$$\hat{\tau}_\xi = Q_{0.90}([0.25, 0.625, 0.95]) = 0.905$$

Pixel P2 ($u = 0.95 \geq 0.905$) is declared **uncertain**. Pixels P0 and P1 are **certain** and carry their top-ranked class label.

---

### 5.5 Cross-Method Comparison

| Test Sample | RAPS Pred. Set | SAPS Pred. Set | Joint (RAPS, K=3) |
|-------------|----------------|----------------|-------------------|
| T0 Easy | $\{2\}$ ✓ | $\{2\}$ ✓ | $\{2\}$ ✓ |
| T1 Borderline | $\{1\}$ ✓ | $\{1\}$ ✓ | $\{1\}$ ✓ |
| T2 Ambiguous | $\{0\}$ ✓ | $\{0\}$ ✓ | $\{0\}$ ✓ |

In this simple toy, all methods agree because the model is well-calibrated on calibration data and the calibration scores collapse to 0. In practice, differences emerge when:

- **RAPS vs. SAPS on ambiguous samples:** If the softmax tail is inflated (miscalibrated), RAPS accumulates this mass and may set a higher threshold, resulting in larger prediction sets that capture the ambiguity. SAPS ignores the tail and uses only the top probability, typically producing smaller sets but potentially under-covering classes whose true probability lies in the tail.
- **Head disagreement in MultiCP:** When different heads produce different top-ranked classes for an ambiguous pixel (e.g., Head 1: $\{0,1\}$, Head 2: $\{0\}$, Head 3: $\{1,2\}$), the joint set is $\{0,1\} \cap \{0\} \cap \{1,2\} = \{\}$ — an *empty* prediction set, flagging the pixel as maximally uncertain. This is precisely the desired behaviour: no class is consistently supported across all views of the model.

---

## 6. References

[1] Vovk, V., Gammerman, A., and Shafer, G. "Algorithmic Learning in a Random World." Springer, 2005.

[2] Shafer, G. and Vovk, V. "A Tutorial on Conformal Prediction." *Journal of Machine Learning Research*, 9:371–421, 2008. [Link](https://www.jmlr.org/papers/volume9/shafer08a/shafer08a.pdf)

[3] Angelopoulos, A. N., Bates, S., Malik, J., and Jordan, M. I. "Uncertainty Sets for Image Classifiers using Conformal Prediction." *ICLR*, 2021. [arXiv:2009.14193](https://arxiv.org/abs/2009.14193) *(RAPS method)*

[4] Huang, J., Xi, H., Zhang, L., Yao, H., Qiu, Y., and Wei, H. "Conformal Prediction for Deep Classifier via Label Ranking." *ICML*, 2024. [arXiv:2310.06430](https://arxiv.org/abs/2310.06430) *(SAPS method)*

[5] Rao, Y., Zhao, W., Zhu, Z., Lu, J., and Zhou, J. "Global Filter Networks for Image Classification." *NeurIPS*, 2021. [arXiv:2107.00645](https://arxiv.org/abs/2107.00645) *(GFNet backbone)*

[6] Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale." *ICLR*, 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) *(Vision Transformer backbone)*

[7] yamtawa. "Multi-CP: Multi-Head Conformal Prediction." GitHub Repository. [https://github.com/yamtawa/Multi-CP](https://github.com/yamtawa/Multi-CP) *(Score computation utilities used in this notebook)*
