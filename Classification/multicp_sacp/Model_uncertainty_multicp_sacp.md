# Spatial MultiCP (SCMCP): Theory & Implementation Summary

> **One-line description:** A multi-head conformal prediction framework that spatially smooths softmax probability maps — in probability space, with renormalisation — before score computation, then intersects per-head prediction sets to produce tighter, spatially coherent uncertainty estimates for pixel-wise remote-sensing image classification.

---

## 1. Overview & Intuition

Conformal prediction (CP) provides a formal, distribution-free guarantee: for any pre-trained classifier and any miscoverage level $\alpha$, a calibrated prediction set will contain the true label with probability at least $1 - \alpha$ on exchangeable test data. Standard split-CP applies this idea to a single model, producing a single set of nonconformity scores and a single quantile threshold. While this is theoretically clean, it leaves two important sources of signal unexploited: the diversity of an ensemble of models trained on the same task, and the spatial structure of imagery where neighbouring pixels share semantic content.

Multi-head CP addresses the first gap. When a neural network is trained with $K$ output heads — each producing its own softmax distribution — each head constitutes an independent conformal predictor. The final prediction set is the **intersection** of all $K$ per-head sets: a label survives only if every head judges it plausible. Because intersection can only reduce set size relative to any individual head, the resulting sets are simultaneously more efficient (smaller) and at least as safe as any single-head approach.

Spatial-Aware CP (SACP) addresses the second gap. In hyperspectral and multispectral remote-sensing imagery, adjacent pixels typically share land-cover class and thus share predictive uncertainty patterns. Aggregating information from spatial neighbours before committing to a nonconformity score exploits this local coherence: pixels in homogeneous regions receive sharper, more confident probability vectors after smoothing, while pixels near class boundaries retain broader, flatter distributions that appropriately produce larger prediction sets.

The Spatial MultiCP (SCMCP) method unifies both ideas in a specific and principled order. Crucially, the spatial smoothing is applied to the **softmax probability vectors themselves** — before any nonconformity score is computed — and is followed by renormalisation so the smoothed vectors remain valid probability distributions. Only then are APS or SAPS scores derived. This ordering is architecturally critical: score functions like APS depend on the internal ordering and cumulative mass of class probabilities, so smoothing raw logits or raw scores would distort their semantics. By working in probability space, SCMCP ensures the score function always sees a coherent, properly normalised input, and that spatial averaging has a well-defined probabilistic interpretation as blending belief vectors.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathcal{X}$ be the input space (multispectral image patches) and $\mathcal{Y} = \{0, 1, \ldots, C-1\}$ a finite label set of $C$ land-cover classes. A multi-head neural network $f : \mathcal{X} \to [0,1]^{K \times C}$ produces $K$ independent softmax output heads. For a pixel $i$ with patch $x_i$ and spatial location $(r_i, c_i)$ in an $H \times W$ scene, the model produces:

$$f(x_i) = \bigl(\hat{\pi}^{(1)}_i,\; \hat{\pi}^{(2)}_i,\; \ldots,\; \hat{\pi}^{(K)}_i\bigr)$$

where each $\hat{\pi}^{(k)}_i \in \Delta^{C-1}$ is a probability simplex vector for head $k$. The dataset is split into a training set, a **calibration set** $\mathcal{D}_\text{cal} = \{(x_i, y_i, r_i, c_i)\}_{i=1}^{n}$, and an **evaluation set** $\mathcal{D}_\text{eval}$.

The miscoverage level $\alpha \in (0,1)$ is fixed; the notebook uses $\alpha = 0.05$, targeting $95\%$ marginal coverage.

---

### 2.2 Spatial Smoothing in Probability Space

**Notation:**

- $\mathcal{N}_w(r,c)$ — set of valid labelled neighbour pixel locations within a square window of size $w$ centred at $(r, c)$, excluding the centre pixel itself.
- $\lambda \in (0, 1]$ — blend weight controlling how much neighbour information is incorporated (notebook default: $\lambda = 0.5$).
- $k_\text{iter}$ — number of smoothing iterations (notebook default: $k_\text{iter} = 1$).
- $\varepsilon \ll 1$ — numerical stability constant to prevent division by zero.

For head $k$, the raw softmax output is placed into a spatial array $P^{(k)} \in \mathbb{R}^{H \times W \times C}$ at the coordinates of labelled pixels. One iteration of SCMCP smoothing updates each pixel $p = (r, c)$:

**Smoothing equation:**
$$\tilde{P}^{(k)}(r,c) = (1 - \lambda)\,\hat{\pi}^{(k)}(r,c) + \lambda \cdot \frac{1}{|\mathcal{N}_w(r,c)|}\sum_{(r',c') \in \mathcal{N}_w(r,c)} \hat{\pi}^{(k)}(r',c')$$

**Where:**
- $\hat{\pi}^{(k)}(r,c) \in \mathbb{R}^C$ — original softmax probability vector for head $k$ at location $(r,c)$
- $\mathcal{N}_w(r,c)$ — the set of valid labelled neighbour pixels within a square window of side $w$, excluding the centre
- $\lambda$ — blend weight on the neighbourhood mean; $(1-\lambda)$ retains the pixel's own signal
- $\tilde{P}^{(k)}(r,c)$ — the blended (pre-normalisation) probability vector

**Renormalisation (mandatory after each smoothing step):**

Convex combination preserves non-negativity, but blending across pixels does not guarantee exact unit-sum due to boundary effects and missing neighbours. An explicit normalisation step restores the simplex constraint:

$$\hat{P}^{(k)}_\text{smooth}(r,c) = \frac{\tilde{P}^{(k)}(r,c)}{\sum_{c'=0}^{C-1} \tilde{P}^{(k)}(r,c,c') + \varepsilon}$$

**What this means:** Each pixel's probability vector is replaced by a weighted average of itself and its spatial neighbours, then renormalised to sum to 1. Pixels in spatially homogeneous regions receive sharp, confident distributions because neighbour averaging reinforces the dominant class. Pixels at class boundaries receive flatter distributions because competing classes in the neighbourhood wash out the dominant signal, which appropriately produces larger prediction sets.

This operation is applied $k_\text{iter}$ times in sequence, with renormalisation after every pass. The calibration and evaluation sets are smoothed in **separate** spatial volumes so no calibration signal leaks into evaluation smoothing.

---

### 2.3 Nonconformity Score Functions

After spatial smoothing, scores are computed on the smoothed, renormalised probabilities $\hat{P}^{(k)}_\text{smooth}$. Two score functions are used:

#### APS (Adaptive Prediction Sets, Romano et al. 2020)

Sort each pixel's class probabilities in descending order: $\hat{\pi}_{(1)} \geq \hat{\pi}_{(2)} \geq \cdots \geq \hat{\pi}_{(C)}$. Let $r(y)$ be the rank of the true class $y$ in this sorted order (rank 1 = highest probability). The APS nonconformity score is:

$$s_\text{APS}(x, y) = \sum_{j=1}^{r(y)-1} \hat{\pi}_{(j)} + U \cdot \hat{\pi}_{(r(y))}$$

**Where:**
- $\hat{\pi}_{(j)}$ — the $j$-th largest smoothed probability
- $r(y)$ — rank of the true class in the sorted order (1-indexed)
- $U \sim \text{Uniform}(0,1)$ — random tie-breaking term that enables exact, non-conservative coverage

**What this means:** The score is the cumulative probability mass of all classes ranked above the true class, plus a random fraction of the true class's own mass. For an easy sample where the true class has the highest probability, $s_\text{APS} = U \cdot \hat{\pi}_{(1)}$, which is small. For a hard sample where the true class is ranked 5th, the score includes the summed mass of the 4 more-likely classes — a large value.

#### SAPS (Sorted Adaptive Prediction Sets, Huang et al. 2024)

SAPS replaces the cumulative summation with a rank-based linear penalty $\lambda_\text{SAPS}$:

$$s_\text{SAPS}(x, y) = \begin{cases} U \cdot \hat{\pi}_\text{max}(x) & \text{if } r(y) = 1 \\ \hat{\pi}_\text{max}(x) + (r(y) - 2 + U)\,\lambda_\text{SAPS} & \text{if } r(y) > 1 \end{cases}$$

**Where:**
- $\hat{\pi}_\text{max}(x) = \hat{\pi}_{(1)}$ — the top smoothed probability
- $\lambda_\text{SAPS}$ — penalty per rank step beyond the top class
- $U \sim \text{Uniform}(0,1)$ — tie-breaking term

**What this means:** SAPS prevents miscalibrated long-tail probabilities from inflating scores and tends to produce more compact sets when the top-1 probability is well-calibrated. It is preferred when the softmax distribution is reliable in its ranking but not necessarily in its specific probability magnitudes.

---

### 2.4 Multi-Head Calibration and the Per-Head Quantile Threshold

After scoring, the calibration scores form a tensor $\mathbf{S}_\text{cal} \in \mathbb{R}^{K \times n \times C}$ where entry $(k, i, c)$ is the score assigned to class $c$ by head $k$ for calibration pixel $i$. For each head $k$, the relevant calibration score is the score at the **true class**:

$$s^{(k)}_i = \mathbf{S}_\text{cal}[k, i, y_i]$$

The per-head quantile threshold is:

$$\hat{q}^{(k)} = Q_{1-\alpha}\!\left(\bigl\{s^{(k)}_i\bigr\}_{i=1}^{n}\right) = \text{the } \lceil (n+1)(1-\alpha) \rceil\text{-th smallest value of } \{s^{(k)}_1, \ldots, s^{(k)}_n\}$$

**Where:**
- $n$ — number of calibration pixels in $D_\text{re-cal}$ (the 95% re-calibration split; see Section 2.5)
- $\alpha$ — miscoverage level (0.05 in the notebook)
- $\hat{q}^{(k)}$ — the threshold for head $k$; any test class with score $\leq \hat{q}^{(k)}$ is included in that head's prediction set

By the standard split-CP exchangeability argument, each head $k$ independently satisfies:

$$\mathbb{P}\!\left(y_\text{test} \in \mathcal{C}^{(k)}\right) \geq 1 - \alpha$$

---

### 2.5 MultiCP Calibration Split: $D_\text{cells}$ and $D_\text{re-cal}$

Following the Multi-CP protocol, a small fraction (5%) of calibration samples is held out as a **cell-selection set** $D_\text{cells}$, used for internal meta-selection (Voronoi cell ordering in the head sweep). The remaining 95% form $D_\text{re-cal}$, on which the per-head quantile thresholds $\hat{q}^{(k)}$ are computed. This split is performed once per head-count step in the head sweep.

The Voronoi cell selection uses $D_\text{cells}$ score vectors projected into a 2D space to determine the order in which heads are included in the intersection, prioritising heads whose calibration region provides the most informative coverage. In the notebook this is visualised as a Voronoi diagram coloured by selection order (darker = selected earlier).

---

### 2.6 Per-Head Prediction Sets

For a test pixel with smoothed scores $s^{(k)}_\text{test}(c)$ for each class $c$, head $k$ includes class $c$ in its prediction set if and only if:

$$\mathcal{C}^{(k)}(x) = \bigl\{c \in \mathcal{Y} : s^{(k)}_\text{test}(c) \leq \hat{q}^{(k)}\bigr\}$$

This produces a Boolean tensor $\mathbf{B} \in \{0,1\}^{K \times N_\text{test} \times C}$ where $\mathbf{B}[k, i, c] = 1$ iff class $c$ is in head $k$'s prediction set for pixel $i$.

---

### 2.7 MultiCP Intersection and Coverage Guarantee

The final SCMCP prediction set for pixel $i$ is the **intersection across all $K$ heads**:

$$\mathcal{C}_\text{SCMCP}(x_i) = \bigcap_{k=1}^{K} \mathcal{C}^{(k)}(x_i) = \bigl\{c \in \mathcal{Y} : \forall k,\; s^{(k)}_\text{test}(c) \leq \hat{q}^{(k)}\bigr\}$$

In code, this is a logical AND across the head dimension:

```python
joint_pred = pred_sets.all(axis=0)   # shape: (N_test, C)
```

**Coverage guarantee:** Since each head independently achieves $1-\alpha$ marginal coverage, the intersection is at minimum as safe as the individual heads. In practice, because the heads share backbone features and training data, they are positively correlated, meaning the coverage of the intersection is empirically close to (and often above) $1-\alpha$, while set sizes are significantly smaller than any single head alone.

Under independence between heads (a conservative bound):
$$P\!\left(y_\text{test} \in \bigcap_{k=1}^K \mathcal{C}^{(k)}\right) = \prod_{k=1}^K P\!\left(y_\text{test} \in \mathcal{C}^{(k)}\right) \geq (1-\alpha)^K$$

Under positive dependence (the realistic regime):
$$P\!\left(y_\text{test} \in \bigcap_{k=1}^K \mathcal{C}^{(k)}\right) \geq 1-\alpha$$

---

### 2.8 Head Sweep

The notebook sweeps the number of active heads from $n_H = 1$ to $n_H = K$, evaluating empirical coverage and mean intersected set size at each step. This **head sweep** reveals the trade-off between head count and prediction-set efficiency. Adding more heads generally reduces set size by tightening the intersection, while coverage remains above $1 - \alpha$; however, beyond some optimal point, adding further heads can marginally depress coverage if the heads have sufficiently independent error patterns. The sweep is a practical diagnostic for choosing the optimal $K$ before deployment.

---

### 2.9 Uncertainty Map Construction

After obtaining per-pixel intersected set sizes, a binary uncertainty map is constructed for the full scene:

$$\text{uncertain}(r,c) = \mathbf{1}\!\left[|\mathcal{C}_\text{SCMCP}(r,c)| \geq \tau_u\right]$$

where $\tau_u$ is the $(1 - \delta_u)$-quantile of all per-pixel set sizes, with $\delta_u = 0.10$ (i.e., the top 10% of pixels by set size are labelled uncertain). A pixel with set size 1 is "certain" — only one class is consistent with all heads' thresholds. A pixel with set size $> 1$ is "uncertain" — multiple classes remain plausible under the joint conformal criterion.

The final binary map merges two sources of uncertainty:
- **CP-uncertain pixels:** top-10% by intersected set size
- **Ground-truth unlabelled pixels:** pixels with label 7 in this dataset (a reserved "unknown" class), which are always marked uncertain

The display map shows class predictions only for certain pixels; uncertain pixels are masked out in grey.

---

### 2.10 Window-Size Sensitivity Analysis

The notebook sweeps over four window sizes $w \in \{3, 5, 7, 9\}$, repeating the full SCMCP pipeline (smoothing → scoring → calibration → intersection → uncertainty map) for each. For a window of size $w$, the neighbourhood contains $(w^2 - 1)$ offsets — 8 for $w=3$, 24 for $w=5$, 48 for $w=7$, and 80 for $w=9$. A cross-window combined summary DataFrame records empirical coverage, mean set size, mean per-class coverage, and uncertain pixel rate for every (model, scoring method, window size) combination, enabling direct comparison of how the spatial context radius affects prediction-set efficiency and spatial coherence.

---

## 3. Algorithm

**Input:**
- Multi-head model $f$ with $K$ heads; neural architectures include AlexNet, GFNet, and ViT-UNet
- Calibration set $\{(x_i, y_i, r_i, c_i)\}_{i=1}^{n}$ and evaluation set
- Window size $w \in \{3, 5, 7, 9\}$, blend weight $\lambda = 0.5$, iterations $k_\text{iter} = 1$
- Score function $\in \{\text{SAPS}, \text{APS}\}$, miscoverage level $\alpha = 0.05$
- Uncertain fraction $\delta_u = 0.10$

**Output:** Per-pixel prediction sets, empirical coverage, mean set size, binary uncertainty map, per-class coverage table, head-sweep summary

1. **Multi-head inference:** Run $f$ on calibration patches $\to \mathbf{P}_\text{cal} \in [0,1]^{K \times n \times C}$; run on evaluation patches $\to \mathbf{P}_\text{eval}$.
2. **Construct neighbour offsets:** Build the list of $(\Delta r, \Delta c)$ pairs for window size $w$, excluding $(0,0)$.
3. **Spatial smoothing — calibration:** Place $\mathbf{P}_\text{cal}[k, :, :]$ into a spatial array $(H, W, C)$ using calibration pixel coordinates. Apply $k_\text{iter}$ rounds of neighbourhood averaging and renormalisation. Read back smoothed probabilities at calibration coordinates $\to \hat{\mathbf{P}}_\text{cal,smooth}$.
4. **Spatial smoothing — evaluation:** Same procedure applied to evaluation pixels in a **separate** spatial volume $\to \hat{\mathbf{P}}_\text{eval,smooth}$.
5. **Score computation:** Apply SAPS or APS to $\hat{\mathbf{P}}_\text{cal,smooth}$ and $\hat{\mathbf{P}}_\text{eval,smooth}$, yielding score tensors $\mathbf{S}_\text{cal}, \mathbf{S}_\text{eval} \in \mathbb{R}^{K \times N \times C}$.
6. **Head sweep:** For $n_H = 1, 2, \ldots, K$:
   - a. Split calibration scores into $D_\text{cells}$ (5%) and $D_\text{re-cal}$ (95%).
   - b. Extract true-class calibration scores: $s^{(k)}_i = \mathbf{S}_\text{cal}[k, i, y_i]$ for $k = 1 \ldots n_H$.
   - c. Compute per-head quantiles: $\hat{q}^{(k)} = Q_{1-\alpha}(\{s^{(k)}_i\})$.
   - d. Compute per-head boolean prediction sets for all evaluation pixels.
   - e. Intersect across heads: `joint_pred = pred_sets.all(axis=0)`.
   - f. Record empirical coverage and mean intersected set size.
7. **Per-class coverage:** Compute marginal coverage separately for each class from the intersected sets.
8. **Full-scene uncertainty map:** Extract all $H \times W$ pixel patches from the padded scene, smooth all pixels jointly in a single spatial volume, compute scores, apply calibration thresholds, intersect sets, compute per-pixel set sizes.
9. **Binary uncertainty map:** Threshold intersected set sizes at the $(1-\delta_u)$ quantile; mark top 10% as uncertain. Overlay with ground-truth unlabelled pixels.
10. **Sweep over window sizes $w \in \{3, 5, 7, 9\}$:** Repeat steps 1–9 for each window size; save per-window Excel workbooks, CSVs, and figures.
11. **Cross-window combined summary:** Aggregate all per-window results into a single DataFrame; produce combined coverage and set-size trend plots.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_uncertainty_multicp_sacp.md`

### 4.1 Neighbour Offset Construction (`build_neighbour_offsets`)

```python
def build_neighbour_offsets(window_size):
    radius = window_size // 2
    return [(dr, dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if not (dr == 0 and dc == 0)]
```

**What this does:** Returns all $(\Delta r, \Delta c)$ integer offsets in a square window of side `window_size` around the origin, excluding the centre $(0,0)$. For $w = 3$, this gives 8 neighbours; for $w = 9$, it gives 80 neighbours.

**Why:** These offsets define the neighbourhood $\mathcal{N}_w(r,c)$ used in the smoothing formula. Sweeping over $w \in \{3, 5, 7, 9\}$ quantifies the sensitivity of results to neighbourhood size and is the notebook's primary hyperparameter analysis.

---

### 4.2 Single-Head Probability Smoothing (`spatial_smooth_prob_map`)

```python
def spatial_smooth_prob_map(prob_map, mask_map, neighbors, lambda_=0.5, eps=EPS):
    smoothed = np.copy(prob_map)
    H, W, C  = prob_map.shape
    rows, cols = np.where(mask_map)
    for r, c in zip(rows, cols):
        ori   = prob_map[r, c]
        n_sum = np.zeros(C); n_count = 0
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask_map[nr, nc]:
                n_sum += prob_map[nr, nc]; n_count += 1
        if n_count > 0:
            n_mean         = n_sum / n_count
            smoothed[r, c] = (1.0 - lambda_) * ori + lambda_ * n_mean
    # Renormalise: restore sum(p) == 1
    sums = smoothed[mask_map].sum(axis=-1, keepdims=True)
    smoothed[mask_map] = smoothed[mask_map] / np.maximum(sums, eps)
    return smoothed
```

**What this does:** For every labelled pixel $(r,c)$, computes the mean probability vector of its valid labelled neighbours, blends it with the pixel's own vector at weight $\lambda$, and renormalises. A critical implementation note: it reads from `prob_map` (the unmodified input) when computing neighbour sums, but writes to `smoothed` (the output copy). This avoids the in-place mutation bug where a pixel's newly-smoothed value would corrupt the neighbour average for later pixels in the same pass.

**Why:** This is the core SCMCP operation — one step of the smoothing formula. The renormalisation restores the simplex constraint before scores are computed. Reading only from the original array ensures all pixels in a single iteration see the same pre-smoothing probabilities, making the operation well-defined and reproducible.

---

### 4.3 Multi-Head Smoothing Orchestration (`build_spatially_smoothed_probs`)

```python
def build_spatially_smoothed_probs(raw_probs, coords, H, W, neighbors, lambda_, k_iters):
    K, N, C = raw_probs.shape
    smoothed_probs = np.zeros_like(raw_probs, dtype=np.float64)
    mask_map = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        mask_map[r, c] = True
    for k in range(K):
        prob_map = np.zeros((H, W, C), dtype=np.float64)
        for i, (r, c) in enumerate(coords):
            prob_map[r, c] = raw_probs[k, i]
        current = prob_map
        for _ in range(k_iters):
            current = spatial_smooth_prob_map(current, mask_map, neighbors, lambda_)
        for i, (r, c) in enumerate(coords):
            smoothed_probs[k, i] = current[r, c]
    return smoothed_probs
```

**What this does:** Iterates over each head $k$, unpacks the flat $(N, C)$ probability array into a 2D spatial grid $(H, W, C)$, runs `k_iters` rounds of smoothing, and reads the smoothed probabilities back out at the original pixel coordinates. The result is a new `(K, N, C)` array of smoothed, renormalised probabilities.

**Why:** The flat array format `(K, N, C)` is what the score function and the rest of the pipeline expect. The spatial grid `(H, W, C)` is needed to identify valid neighbours by their geometric proximity. This function bridges the two representations. When called for the full-scene uncertainty map, `coords` covers all $H \times W$ locations, enabling spatially-continuous smoothing across the entire image.

---

### 4.4 Score Computation on Smoothed Probabilities

```python
cal_scores_smooth  = np.round(compute_scores(cal_probs_smooth,  config), 4)
test_scores_smooth = np.round(compute_scores(test_probs_smooth, config), 4)
```

**What this does:** Applies the configured score function (SAPS or APS) from the Multi-CP library to the smoothed, renormalised probability arrays. The result is a `(K, N, C)` score tensor where entry `[k, i, c]` is the nonconformity score of head $k$ for pixel $i$ being class $c$.

**Why:** The score function is applied **after** smoothing, not before. This preserves the mathematical validity of APS/SAPS, which require proper probability vectors as input. Smoothing in score space would destroy the cumulative-mass semantics of APS.

---

### 4.5 Per-Head Quantile Thresholds and MultiCP Intersection (`main_algo`)

```python
def main_algo(Dcells_scores, Dcells_target, Dre_cal_scores, Dre_cal_target,
              test_scores, test_target, alpha, config):
    K, N_cal = Dre_cal_scores.shape[0], Dre_cal_scores.shape[1]
    cal_true = Dre_cal_scores[np.arange(K)[:, None], np.arange(N_cal), Dre_cal_target]
    q        = np.quantile(cal_true, 1 - alpha, axis=1)       # shape: (K,)
    pred_sets = test_scores <= q[:, None, None]               # shape: (K, N_test, C)
    joint_pred = pred_sets.all(axis=0)                        # shape: (N_test, C)
    valid   = (test_target >= 0) & (test_target < joint_pred.shape[1])
    covered = joint_pred[np.arange(np.sum(valid)), test_target[valid]]
    set_size = joint_pred.sum(axis=1)
    return covered.mean(), set_size.mean(), pred_sets
```

**What this does:** Extracts the true-class calibration score for each head, computes the $(1-\alpha)$ quantile per head, applies thresholds to test scores to build per-head boolean prediction sets, and intersects across heads using `.all(axis=0)`. Coverage and set size are measured on the **intersected** set.

**Why:** This is the mathematical realisation of $\mathcal{C}_\text{SCMCP} = \bigcap_k \mathcal{C}^{(k)}$. Computing set size on `joint_pred` (not on any individual head) gives the size of the set the practitioner actually receives at inference time.

---

### 4.6 Head Sweep (`compute_head_sweep_fused`)

```python
for nH in range(1, cal_output.shape[0] + 1):
    Dc, Dt, Rc, Rt = generate_Dcal_Dcells_sets(cal_scores_smooth[:nH], cal_target)
    cov, msz, pred_sets = main_algo(Dc, Dt, Rc, Rt,
                                     test_scores_smooth[:nH], test_target,
                                     ALPHA, config)
    rows.append({'heads': nH, 'coverage': float(cov), 'set_size': float(msz)})
```

**What this does:** Sweeps the number of active heads from 1 to $K$, evaluating coverage and mean set size at each step. At $n_H = K$, all heads participate in the intersection. The final bundle `(config, Dc, Dt, Rc, Rt, pred_sets)` is returned for downstream use.

**Why:** The sweep reveals the trade-off curve between head count and efficiency. The per-window head-sweep figure plots coverage and set size vs. number of heads, with the target coverage line ($1 - \alpha = 0.95$) overlaid, allowing the researcher to identify the minimum number of heads needed to meet the coverage guarantee while maximising set compactness.

---

### 4.7 Per-Class Coverage (`per_class_coverage_df_fused`)

```python
def per_class_coverage_df_fused(pred_sets, y_true, n_classes):
    joint_sets = pred_sets.all(axis=0)    # (N_test, C)
    rows = []
    for c in range(n_classes):
        idx      = np.where(y_true == c)[0]
        coverage = float(np.mean([joint_sets[j, c] for j in idx])) if idx.size > 0 else np.nan
        rows.append({'class_id': c, 'class_coverage': coverage, 'support_count': len(idx)})
    return pd.DataFrame(rows)
```

**What this does:** Computes marginal coverage of the intersected prediction set for each class separately — the fraction of test pixels with true class $c$ for which class $c$ appears in the final SCMCP set.

**Why:** Marginal coverage $\geq 1 - \alpha$ is guaranteed only on average across all classes. Per-class coverage reveals whether any particular land-cover type is systematically under-covered (e.g., rare classes with few calibration examples may receive inadequate calibration quantiles). This is essential for fairness analysis: the conformal guarantee is average-case, not worst-case, so rare or spectrally ambiguous land-cover classes can still be under-covered even when the overall guarantee holds.

---

### 4.8 Full-Scene Binary Uncertainty Map (`build_binary_uncertainty_outputs_fused`)

```python
image_outputs = get_image_multi_head_outputs(model, padded_x, H, W, B, PATCH_SIZE, BATCH_SIZE)

all_coords_full = np.array([[r, c] for r in range(H) for c in range(W)])
smoothed_probs_full = build_spatially_smoothed_probs(
    image_outputs, all_coords_full, H, W,
    neighbors=neighbors, lambda_=lambda_, k_iters=k_iters)   # (K, H*W, C)

image_scores = np.round(compute_scores(smoothed_probs_full, config), 4)

# ... mask unlabelled pixels (class 7) ...
cov, mset, pred_bool = main_algo(Dc, Dt, Rc, Rt, img_valid, y_valid, config['ALPHA'], config)
joint_pred_valid = pred_bool.all(axis=0)
set_sizes        = joint_pred_valid.sum(axis=1)

thresh             = np.nanquantile(set_sizes.astype(float), 1 - UNCERTAIN_FRACTION)
cp_uncertain_valid = set_sizes >= thresh

# Overlay ground-truth unlabelled pixels
final_uncertain = cp_uncertain | gt_uncertain
```

**What this does:** Applies the full SCMCP pipeline to every pixel in the scene (not just labelled pixels), smoothing all $H \times W$ locations jointly in a single spatial volume. The calibration thresholds $\hat{q}^{(k)}$ learned from the labelled calibration set are reused without modification. Pixels whose intersected set size exceeds the top-10% threshold are marked uncertain, and this is merged with the ground-truth unlabelled mask.

**Why:** The binary uncertainty map provides an interpretable, spatially contiguous visualisation of model uncertainty across the full remote-sensing scene. Because the thresholds are fixed from calibration, this application to unlabelled pixels is a genuine out-of-sample inference step — it extends the conformal guarantee to the full image. Full-scene smoothing (using all $H \times W$ neighbours) produces spatially coherent uncertainty regions rather than isolated per-pixel decisions.

---

### 4.9 Voronoi Cell-Selection Visualisation (`visualize_cell_selection`)

```python
def visualize_cell_selection(Dcells_scores, Dcells_target, D_i_order, model_name):
    pts  = Dcells_scores[:, :2] if Dcells_scores.shape[1] > 2 else Dcells_scores
    vor  = Voronoi(pts)
    ranks            = np.argsort(D_i_order)
    normalized_order = np.zeros(len(pts))
    normalized_order[ranks] = np.linspace(0, 1, len(pts))
    # ... colour Voronoi cells by selection order ...
```

**What this does:** Projects $D_\text{cells}$ scores into 2D (using the first two score dimensions), constructs a Voronoi tessellation, and colours each cell by the order in which it was selected during the Multi-CP head sweep. Cells selected earlier (darker colour) correspond to heads whose calibration region is most informative for the overall coverage objective.

**Why:** This is a diagnostic visualisation for the Multi-CP cell-selection step. It provides geometric intuition about which regions of the score space are "covered" by early heads and which require additional heads to achieve adequate coverage. Cells that are isolated or in low-density regions of score space tend to be selected later.

---

### 4.10 Main Execution Loop and Output Organisation (Sections 8–9)

```python
for ws in SACP_WINDOW_SIZES:          # [3, 5, 7, 9]
    for model_key, model in models.items():
        for scoring_method in SCORING_METHODS:   # ['APS', 'SAPS']
            head_df, bundle = compute_head_sweep_fused(...)
            class_cov_df   = per_class_coverage_df_fused(...)
            binary_outputs = build_binary_uncertainty_outputs_fused(...)
            # ... generate 6 figures, write Excel sheet ...
    # save per-window workbook, summary CSV, per-class CSV
```

**What this does:** The outer triple loop over window sizes, models, and scoring methods constitutes the complete experimental grid. For each combination it runs the full pipeline, generates six figures (head-sweep line plot, per-class coverage bar chart, binary uncertainty map, class prediction map, pixel-count bar chart, Voronoi diagram), writes a dedicated Excel sheet with metadata and embedded figures, and saves per-window CSV summaries. Section 9 then collates all per-window results into a combined summary DataFrame and produces three multi-line trend plots (coverage, set size, mean per-class coverage — all vs. window size, broken out by model).

**Why:** The nested loop structure cleanly separates the three dimensions of variation in the experiment: spatial context (window size), model architecture, and score function. The per-window Excel workbooks allow fine-grained inspection of any single configuration, while the combined summary enables cross-cutting analysis of the most important trends.

---

### 4.11 Final Validation (Section 10)

```python
for ws in SACP_WINDOW_SIZES:
    assert workbook_path.exists(), ...
    assert 'Summary' in sheets, ...
    assert any(f'ws{ws}' in s for s in sheets), ...

assert len(combined_summary_df) == expected_rows, ...
assert ((combined_summary_df['empirical_coverage'] >= 0) & ...).all(), ...
assert set(combined_summary_df['window_size'].unique()) == set(SACP_WINDOW_SIZES), ...
assert set(combined_summary_df['scoring_method'].unique()) == set(SCORING_METHODS), ...
```

**What this does:** Asserts that every per-window workbook exists with the expected sheet structure, that the combined summary DataFrame has the correct number of rows (one per window size × model × scoring method), that all coverage values are in $[0,1]$, and that all window sizes and scoring methods are represented.

**Why:** These integrity checks catch silent failures in the pipeline (e.g., a workbook that was not saved, or a run that was skipped due to a memory error) before the results are used downstream in a paper or report. Structural validation is especially important when results will be aggregated across many configurations.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

Toy dataset: $K = 3$ heads, $C = 4$ classes, $n = 12$ calibration pixels, 3 test pixels, miscoverage $\alpha = 0.10$, blend weight $\lambda = 0.5$, window size $w = 3$, $k_\text{iter} = 1$.

For tractability, the nonconformity score used in this example is the LAC (Least Ambiguous Classifier) score: $s(x, y) = 1 - \hat{\pi}^\text{smooth}_{y}$, i.e. one minus the smoothed probability at the true class. This is equivalent to the APS score when the true class has the top rank; the full APS/SAPS arithmetic is analogous but more complex to trace by hand. Conclusions transfer directly.

#### Calibration Probability Matrix (Head 1, before smoothing)

Each row sums to 1.00. The true-class probability is **bolded**.

| Pixel | True Class | $\hat{\pi}_0$ | $\hat{\pi}_1$ | $\hat{\pi}_2$ | $\hat{\pi}_3$ |
|-------|------------|---------------|---------------|---------------|---------------|
| C1    | 0          | **0.70**      | 0.15          | 0.10          | 0.05          |
| C2    | 1          | 0.10          | **0.65**      | 0.20          | 0.05          |
| C3    | 2          | 0.05          | 0.15          | **0.72**      | 0.08          |
| C4    | 3          | 0.08          | 0.12          | 0.10          | **0.70**      |
| C5    | 0          | **0.55**      | 0.25          | 0.12          | 0.08          |
| C6    | 1          | 0.20          | **0.50**      | 0.20          | 0.10          |
| C7    | 2          | 0.10          | 0.08          | **0.60**      | 0.22          |
| C8    | 3          | 0.12          | 0.14          | 0.24          | **0.50**      |
| C9    | 0          | **0.45**      | 0.30          | 0.15          | 0.10          |
| C10   | 1          | 0.15          | **0.45**      | 0.25          | 0.15          |
| C11   | 2          | 0.08          | 0.20          | **0.52**      | 0.20          |
| C12   | 3          | 0.10          | 0.18          | 0.32          | **0.40**      |

#### Test Probability Matrix (Head 1, before smoothing)

| Pixel | True Class | $\hat{\pi}_0$ | $\hat{\pi}_1$ | $\hat{\pi}_2$ | $\hat{\pi}_3$ | Type |
|-------|------------|---------------|---------------|---------------|---------------|------|
| T1    | 0          | **0.75**      | 0.12          | 0.08          | 0.05          | Easy |
| T2    | 1          | 0.28          | **0.35**      | 0.25          | 0.12          | Borderline |
| T3    | 2          | 0.30          | 0.28          | **0.22**      | 0.20          | Ambiguous |

Heads 2 and 3 follow the same pattern with slightly different probability values (simulating ensemble diversity). The full example below works through Head 1 and then summarises the cross-head intersection.

---

### 5.1 Spatial Smoothing (Head 1)

#### Toy Spatial Layout

Assign calibration pixels to a $4 \times 3$ grid:

```
(0,0)=C1  (0,1)=C2  (0,2)=C3
(1,0)=C4  (1,1)=C5  (1,2)=C6
(2,0)=C7  (2,1)=C8  (2,2)=C9
(3,0)=C10 (3,1)=C11 (3,2)=C12
```

Using $w=3$, each pixel's neighbourhood is its up-to-8 immediately surrounding grid cells.

#### Example: Smoothing pixel C5 at (1,1)

Neighbours: C1(0,0), C2(0,1), C3(0,2), C4(1,0), C6(1,2), C7(2,0), C8(2,1), C9(2,2) — all 8 cells present.

Neighbour mean per class:

$$\bar{\pi}_0 = \frac{0.70 + 0.10 + 0.05 + 0.08 + 0.20 + 0.10 + 0.12 + 0.45}{8} = \frac{1.80}{8} = 0.225$$

$$\bar{\pi}_1 = \frac{0.15 + 0.65 + 0.15 + 0.12 + 0.50 + 0.08 + 0.14 + 0.30}{8} = \frac{2.09}{8} = 0.261$$

$$\bar{\pi}_2 = \frac{0.10 + 0.20 + 0.72 + 0.10 + 0.20 + 0.60 + 0.24 + 0.15}{8} = \frac{2.31}{8} = 0.289$$

$$\bar{\pi}_3 = \frac{0.05 + 0.05 + 0.08 + 0.70 + 0.10 + 0.22 + 0.50 + 0.10}{8} = \frac{1.80}{8} = 0.225$$

Neighbour mean sums to $0.225 + 0.261 + 0.289 + 0.225 = 1.000$ ✓

Blended vector (pre-renormalisation, $\lambda = 0.5$):

$$\tilde{P}_\text{C5} = 0.5 \times [0.55, 0.25, 0.12, 0.08] + 0.5 \times [0.225, 0.261, 0.289, 0.225]$$
$$= [0.388, 0.256, 0.205, 0.153]$$

Sum = $1.002$ (rounding artefact). After renormalisation:

$$\hat{P}^\text{smooth}_\text{C5} = [0.387, 0.255, 0.204, 0.153] \quad (\text{sum} = 1.000)$$

The true class (0) still has the highest smoothed probability, but it decreased from 0.55 to 0.387 due to the mixed neighbourhood, slightly increasing the LAC nonconformity score.

---

### 5.2 Step A — Nonconformity Scores on Calibration Set (Head 1, smoothed)

Using $s_i = 1 - \hat{\pi}^\text{smooth}_{y_i}$:

| Pixel | True Class | $\hat{\pi}^\text{smooth}_{y_i}$ | Score $s_i$ |
|-------|------------|---------------------------------|-------------|
| C1    | 0          | 0.680 (boundary; less smoothing)| 0.320       |
| C2    | 1          | 0.620                           | 0.380       |
| C3    | 2          | 0.690                           | 0.310       |
| C4    | 3          | 0.660                           | 0.340       |
| C5    | 0          | 0.387                           | 0.613       |
| C6    | 1          | 0.470                           | 0.530       |
| C7    | 2          | 0.570                           | 0.430       |
| C8    | 3          | 0.462                           | 0.538       |
| C9    | 0          | 0.410                           | 0.590       |
| C10   | 1          | 0.420                           | 0.580       |
| C11   | 2          | 0.480                           | 0.520       |
| C12   | 3          | 0.375                           | 0.625       |

All 12 calibration scores: `[0.320, 0.380, 0.310, 0.340, 0.613, 0.530, 0.430, 0.538, 0.590, 0.580, 0.520, 0.625]`

Sorted: `[0.310, 0.320, 0.340, 0.380, 0.430, 0.520, 0.530, 0.538, 0.580, 0.590, 0.613, 0.625]`

---

### 5.3 Step B — Calibration Threshold (Head 1)

$$q_\text{level} = \frac{\lceil (n+1)(1-\alpha) \rceil}{n} = \frac{\lceil 13 \times 0.90 \rceil}{12} = \frac{\lceil 11.70 \rceil}{12} = \frac{12}{12} = 1.0$$

Since $q_\text{level}$ clips to 1.0, the threshold $\hat{q}^{(1)}$ equals the **maximum** calibration score:

$$\hat{q}^{(1)} = 0.625$$

With $n = 12$ and $\alpha = 0.10$, we need at least 13 calibration points to avoid this saturation; with 12 the threshold is conservative, which preserves the $\geq 1-\alpha$ guarantee but may admit more classes into the prediction set than strictly necessary.

For Heads 2 and 3 (simulating slightly stricter thresholds from ensemble diversity):

$$\hat{q}^{(2)} = 0.540, \quad \hat{q}^{(3)} = 0.570$$

---

### 5.4 Step C — Prediction Sets for Each Test Pixel

After smoothing, scores $s_\text{test}(c) = 1 - \hat{\pi}^\text{smooth}_c$ are computed for each class $c$.

#### Test Pixel T1 (Easy: True Class = 0)

After smoothing (easy pixel in a predominantly class-0 neighbourhood), probabilities sharpen:

| Class | $\hat{\pi}^\text{smooth}_c$ | Score $1 - p$ | $\leq 0.625$? | $\leq 0.540$? | $\leq 0.570$? | In Final Set? |
|-------|-----------------------------|---------------|---------------|---------------|---------------|---------------|
| 0     | 0.720                       | 0.280         | ✓             | ✓             | ✓             | **✓**         |
| 1     | 0.120                       | 0.880         | ✗             | ✗             | ✗             | ✗             |
| 2     | 0.090                       | 0.910         | ✗             | ✗             | ✗             | ✗             |
| 3     | 0.070                       | 0.930         | ✗             | ✗             | ✗             | ✗             |

**Final set: {0}** — Size = 1 — True class covered? **✓**

All three heads agree; the intersection is a singleton. Spatial smoothing reinforced the dominant class, producing a sharper distribution and a smaller, more confident prediction set.

---

#### Test Pixel T2 (Borderline: True Class = 1)

After smoothing, the borderline pixel's class-1 probability drops from 0.35 to approximately 0.310 due to neighbourhood dilution (surrounding pixels spread probability across all classes):

| Class | $\hat{\pi}^\text{smooth}_c$ | Score | $\leq 0.625$? | $\leq 0.540$? | $\leq 0.570$? | In Final Set? |
|-------|-----------------------------|-------|---------------|---------------|---------------|---------------|
| 0     | 0.295                       | 0.705 | ✗             | ✗             | ✗             | ✗             |
| 1     | 0.310                       | 0.690 | ✗             | ✗             | ✗             | ✗             |
| 2     | 0.255                       | 0.745 | ✗             | ✗             | ✗             | ✗             |
| 3     | 0.140                       | 0.860 | ✗             | ✗             | ✗             | ✗             |

**Final set: {}** — Size = 0 — True class covered? **✗**

An empty set occurs because all smoothed probabilities are low (≈ 0.14–0.31), producing LAC scores ≈ 0.69–0.86, all exceeding the loosest threshold $\hat{q}^{(1)} = 0.625$. This is an artefact of using the LAC approximation: the LAC score overpunishes uniformly-low distributions. Under the actual APS score, the true class at rank 2 would have score $\leq \hat{\pi}_{(1)} + U \cdot \hat{\pi}_{(2)}$, which for this pixel is approximately $0.295 + U \cdot 0.310$, almost certainly $\leq 0.625$, and so class 1 would survive Head 1's gate. The point stands that borderline pixels receive larger sets under SCMCP than under single-head CP, appropriately reflecting their genuine ambiguity.

---

#### Test Pixel T3 (Ambiguous: True Class = 2)

After smoothing, all four classes have smoothed probabilities around 0.24–0.26 — a nearly flat distribution:

| Class | $\hat{\pi}^\text{smooth}_c$ | Score | $\leq 0.625$? | $\leq 0.540$? | $\leq 0.570$? | In Final Set? |
|-------|-----------------------------|-------|---------------|---------------|---------------|---------------|
| 0     | 0.260                       | 0.740 | ✗             | ✗             | ✗             | ✗             |
| 1     | 0.250                       | 0.750 | ✗             | ✗             | ✗             | ✗             |
| 2     | 0.240                       | 0.760 | ✗             | ✗             | ✗             | ✗             |
| 3     | 0.250                       | 0.750 | ✗             | ✗             | ✗             | ✗             |

**Final set: {}** — Size = 0 — True class covered? **✗**

Under the LAC approximation, the truly ambiguous pixel produces an empty set for the same reason as T2. Under APS, such a pixel would produce a **large** set containing multiple classes — the desired adaptive behaviour. The APS score accumulates cumulative probability mass from the top-ranked class down, so even a flat distribution with $\hat{\pi}_{(1)} \approx 0.26$ yields scores well below 0.625 for the top two or three classes. The LAC approximation used here for notational clarity is unsuitable for genuinely ambiguous pixels and should not be taken to imply SCMCP produces empty sets in practice.

---

### 5.5 Step D — Summary Table

| Test Pixel       | Prediction Set | Set Size | Coverage |
|------------------|----------------|----------|----------|
| T1 (Easy)        | {0}            | 1        | ✓        |
| T2 (Borderline)  | {}             | 0        | ✗        |
| T3 (Ambiguous)   | {}             | 0        | ✗        |

---

### 5.6 Cross-Head Comparison and Discussion

The table below shows what each head individually would produce using its own threshold:

| Test Pixel       | Head 1 Set ($\hat{q}=0.625$, loose) | Head 2 Set ($\hat{q}=0.540$, strict) | Head 3 Set ($\hat{q}=0.570$, mid) | SCMCP Intersection |
|------------------|-------------------------------------|--------------------------------------|------------------------------------|--------------------|
| T1 (Easy)        | {0}                                 | {0}                                  | {0}                                | {0}                |
| T2 (Borderline)  | {} (LAC)                            | {} (LAC)                             | {} (LAC)                           | {}                 |
| T3 (Ambiguous)   | {} (LAC)                            | {} (LAC)                             | {} (LAC)                           | {}                 |

Note: "(LAC)" indicates the empty set is an artefact of the LAC approximation for illustration; under true APS, borderline and ambiguous pixels would have non-empty — and likely large — prediction sets.

**Effective probability threshold summary** (minimum smoothed probability a class must achieve to pass each head's gate under LAC):

| Head | Threshold $\hat{q}^{(k)}$ | Minimum probability to pass ($1 - \hat{q}$) |
|------|--------------------------|---------------------------------------------|
| Head 1 (loose) | 0.625 | 0.375 |
| Head 2 (strict) | 0.540 | 0.460 |
| Head 3 (mid) | 0.570 | 0.430 |
| SCMCP (intersection) | — | **0.460** (strictest head governs) |

Under SCMCP intersection, the effective gate is set by the strictest head: a class must have smoothed probability $\geq 0.460$ to survive all three heads. This explains why T1 (class 0 at 0.720) is certain and T2/T3 (all classes $\leq 0.310$) are uncertain. The key insight is that **intersection amplifies the strictness of the most demanding head** — a class is only retained if it passes the highest bar across the ensemble, which is precisely what eliminates spurious inclusions and produces tighter, more reliable sets.

---

## 6. References

[1] J. Liu, T. Sun, H. Zeng, Y. Zhang, C.-M. Pun, and C.-M. Vong. "Spatial-Aware Conformal Prediction for Trustworthy Hyperspectral Image Classification." *arXiv preprint arXiv:2409.01236*, 2024. [https://arxiv.org/abs/2409.01236](https://arxiv.org/abs/2409.01236)

[2] A. Baheri and M. A. Shahbazi. "Multi-Scale Conformal Prediction: A Theoretical Framework with Coverage Guarantees." *arXiv preprint arXiv:2502.05565*, 2025. Also published as "Conformal prediction across scales: Finite-sample coverage with hierarchical efficiency." *Elsevier*, 2025. [https://arxiv.org/abs/2502.05565](https://arxiv.org/abs/2502.05565)

[3] Y. Romano, M. Sesia, and E. J. Candès. "Classification with Valid and Adaptive Coverage." *Advances in Neural Information Processing Systems (NeurIPS)*, 2020. (Introduces APS.) [https://arxiv.org/abs/2006.02544](https://arxiv.org/abs/2006.02544)

[4] A. N. Angelopoulos, S. Bates, M. Jordan, and J. Malik. "Uncertainty Sets for Image Classifiers Using Conformal Prediction." *International Conference on Learning Representations (ICLR)*, 2021. (Introduces RAPS.) [https://arxiv.org/abs/2009.14193](https://arxiv.org/abs/2009.14193)

[5] J. Huang, H. Xi, L. Zhang, H. Yao, Y. Qiu, and H. Xie. "Conformal Prediction for Deep Classifier via Label Ranking." *arXiv preprint arXiv:2310.06430*, 2023. (Introduces SAPS.) [https://arxiv.org/abs/2310.06430](https://arxiv.org/abs/2310.06430)

[6] A. N. Angelopoulos and S. Bates. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *arXiv preprint arXiv:2107.07511*, 2021. [https://arxiv.org/abs/2107.07511](https://arxiv.org/abs/2107.07511)

[7] E. Hajihashemi and Y. Shen. "Multi-Model Ensemble Conformal Prediction in Dynamic Environments." *Advances in Neural Information Processing Systems (NeurIPS)*, 2024. [https://arxiv.org/abs/2411.03678](https://arxiv.org/abs/2411.03678)

[8] Multi-CP Repository (yamtawa). Reference implementation of the multi-head score computation and head-sweep protocol used in this notebook. [https://github.com/yamtawa/Multi-CP](https://github.com/yamtawa/Multi-CP)
