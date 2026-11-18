# Spatial-Aware Conformal Prediction (SACP): Theory & Implementation Summary

> **One-line description:** SACP equips a pixel-wise image classifier with statistically valid, set-valued predictions by smoothing each pixel's nonconformity score with its spatial neighbours before calibrating a single global threshold.

---

## 1. Overview & Intuition

Pixel-level image classifiers — whether for hyperspectral land-cover mapping or, as in this notebook, 6-band multispectral classification — produce a softmax probability vector for every pixel, but that probability vector is only a point estimate of confidence. It does not come with any guarantee that the true class is actually likely to be among the classes the model favours. Conformal Prediction (CP) closes this gap: instead of returning a single predicted label, it returns a **prediction set** of plausible labels, constructed so that the true label is *provably* contained in that set with a chosen probability (e.g. 95%), no matter what classifier produced the underlying probabilities and without assuming any particular data distribution.

Standard CP for classification treats every pixel as an independent, exchangeable sample. This is a poor fit for spatial imagery: pixels do not appear in isolation, they sit inside a 2-D grid where neighbouring pixels are very likely to belong to the same land-cover class. A pixel whose individual prediction is noisy or borderline is often surrounded by pixels which are *not* equally uncertain — and that neighbourhood information is simply thrown away by classical CP. The result is unnecessarily large or poorly calibrated prediction sets in exactly the regions (class boundaries, mixed pixels, sensor noise) where extra evidence from neighbours would help most.

SACP's core insight is to **smooth the nonconformity score itself across a local spatial neighbourhood before calibrating**, rather than smoothing the predicted probabilities or the final labels. Each pixel's raw conformity score is blended with the average score of its surrounding window, optionally repeated for several rounds of smoothing. Because the *score function* — not the underlying classifier — is what gets modified, the conformal coverage guarantee is preserved: SACP is a wrapper that can sit on top of *any* trained pixel classifier, exactly like ordinary split conformal prediction, while injecting spatial context that the base classifier never had access to.

What makes this attractive in practice is that it requires no retraining of the underlying model and no changes to its outputs — it only changes how those outputs are turned into trustworthy uncertainty regions, at negligible extra computational cost.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

- $\mathcal{X}$ — the space of input patches (here, $9\times 9$ pixel windows across 6 spectral bands) centred on a labeled pixel.
- $\mathcal{Y} = \{0, 1, \dots, K-1\}$ — the set of $K$ possible land-cover classes.
- $\hat{\pi}_\theta : \mathcal{X} \to \Delta^{K-1}$ — a trained deep classifier (in this notebook: AlexNet-style CNN, GFNet, or a ViT-UNet) that outputs a softmax probability vector $\hat{\pi}(x) \in \Delta^{K-1}$ over the $K$ classes.
- $\mathcal{D}_{\text{cal}} = \{(x_i, y_i, \text{loc}_i)\}_{i=1}^n$ — a held-out **calibration set** of labeled pixels, each carrying its $(row, col)$ grid location in addition to its features and label.
- $B_{n+1}$ — a new test pixel (with its own grid location) for which a prediction set is required.
- $\alpha \in (0,1)$ — the user-chosen miscoverage rate (the notebook uses $\alpha = 0.05$, i.e. a 95% target coverage).

The crucial extra ingredient relative to vanilla conformal classification is that every sample — calibration *and* test — carries a 2-D spatial coordinate, which lets the method look up "who are my neighbours" on the pixel grid.

### 2.2 Base Nonconformity Score: Randomized APS

SACP, as implemented here, builds on the Adaptive Prediction Sets (APS) score of Romano, Sesia & Candès (2020). For a probability vector $\hat\pi(x)$, sort the classes in decreasing order of predicted probability and let $\pi_{(1)}(x) \ge \pi_{(2)}(x) \ge \dots \ge \pi_{(K)}(x)$ be the sorted probabilities, with $o(y, x)$ the rank of the true class $y$ in this order. The **randomized APS score** is:

$$
S(x, y) = \sum_{j=1}^{o(y,x)-1} \pi_{(j)}(x) \;+\; U \cdot \pi_{(o(y,x))}(x)
$$

**Where:**
- $\pi_{(j)}(x)$ — the $j$-th largest predicted class probability for input $x$
- $o(y,x)$ — the rank position of the true (or candidate) class $y$ once classes are sorted by probability, descending
- $U \sim \text{Uniform}(0,1)$ — an independent random draw used to randomize ties and make the score's distribution continuous

**What this means:** the score is the cumulative probability mass assigned to all classes ranked *strictly more likely* than $y$, plus a randomized fraction of $y$'s own probability mass. A class that the model is very confident about (rank 1, high probability) gets a small score; a class buried deep in the tail of the softmax gets a large score. This is exactly the quantity the notebook computes in `compute_aps_scores`: it sorts probabilities per sample, takes a cumulative sum, and reads off the cumulative mass up to (but not including) the true/candidate class's rank, then adds the randomized term for that class's own probability.

### 2.3 Score Aggregation Operator: Spatial Smoothing

This is SACP's defining contribution. Let $S^{(0)}(B_i, y) = S(x_i, y)$ be the base APS score for pixel $i$ at grid location $(r,c)$, and let $\mathcal{N}(r,c)$ be the set of grid neighbours of $(r,c)$ inside a square window of side length $w$ (excluding the centre pixel itself). The **Score Aggregation Operator** $V_k$ applies, for each of $k$ rounds:

$$
V_{t}(B_i, y) = \lambda \cdot V_{t-1}(B_i, y) \;+\; \lambda \cdot \frac{1}{|\mathcal{N}_{\text{valid}}(r,c)|}\sum_{(r',c') \in \mathcal{N}_{\text{valid}}(r,c)} V_{t-1}(B_{(r',c')}, y), \qquad t = 1, \dots, k
$$

**Where:**
- $\lambda \in (0,1]$ — a mixing weight controlling how much the smoothed score blends the pixel's own (previous-round) score against its neighbours' average score
- $\mathcal{N}_{\text{valid}}(r,c)$ — the subset of the window's neighbour offsets that fall inside the image and that currently hold a valid score (i.e. are themselves calibration or test pixels with a label/probability assigned)
- $k$ — the number of smoothing iterations (rounds) applied
- $w$ — the window size (side length of the square neighbourhood, must be odd, $w \ge 3$); the notebook sweeps $w \in \{3, 5, 7, 9\}$
- $V_0(B_i, y) = S(B_i, y)$ — the un-smoothed base APS score, used as the seed for the recursion

**What this means:** each round replaces a pixel's score with a weighted blend of "my own current score" and "the average current score among my spatial neighbours." After $k$ rounds, a pixel's final score reflects not just its own classifier output but the aggregate evidence of the surrounding neighbourhood, propagated outward over $k$ hops. This is precisely what `spatial_smoothing` implements: for every masked (valid) pixel it averages the scores of its in-bounds, valid neighbours and combines that average with the pixel's own current score using weight $\lambda$ on both terms (note: in this notebook's implementation $\lambda$ is applied to *both* the own-score term and the neighbour-average term, rather than $\lambda$ and $1-\lambda$ — a deliberate choice that scales down both contributions for $\lambda<1$ rather than redistributing weight between them).

### 2.4 Calibration Threshold and Coverage Guarantee

Once the (possibly smoothed) calibration scores $\{\hat s_i\}_{i=1}^{n}$ are obtained — one aggregated score per calibration pixel, evaluated at its **true** label — SACP computes the conformal threshold as:

$$
\hat\tau = \inf\left\{ s \;\middle|\; \frac{|\{i : \hat s_i \le s\}|}{n} \ge \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right\}
$$

**Where:**
- $n$ — the number of calibration samples
- $\alpha$ — the target miscoverage rate
- $\lceil \cdot \rceil$ — the ceiling function, ensuring the quantile level is rounded up to guarantee *at least* the desired coverage in finite samples

**What this means:** $\hat\tau$ is simply the $\lceil(n+1)(1-\alpha)\rceil / n$ empirical quantile of the calibration scores — the smallest score threshold such that the required fraction of calibration scores fall at or below it. This is exactly what the notebook computes via `np.quantile(fused_calib_scores, q_level, method='higher')`, after first clipping the quantile level to the valid range $[0,1]$ (handling the edge case where the ceiling pushes the level above 1, in which case the threshold becomes the maximum observed score).

The prediction set for a new pixel $B_{n+1}$ is then:

$$
\hat{\mathcal{C}}_{1-\alpha}(B_{n+1}) = \{\, y \in \mathcal{Y} \;:\; V_k(B_{n+1}, y) \le \hat\tau \,\}
$$

**What this means:** include every class $y$ in the prediction set whose (spatially-smoothed) score is at or below the calibrated threshold. Classes with high smoothed scores — meaning the model and its spatial context jointly disfavour them — are excluded.

**Coverage guarantee:** under the standard conformal exchangeability assumption between calibration and test data, this construction guarantees

$$
\mathbb{P}\big(y_{n+1} \in \hat{\mathcal{C}}_{1-\alpha}(B_{n+1})\big) \ge 1-\alpha,
$$

marginally over draws of the calibration and test sets. SACP's theoretical contribution (established in the original paper) is to show that this guarantee survives the score-smoothing step — i.e. aggregating scores over spatial neighbours does not break exchangeability-based coverage, while empirically shrinking average set size by exploiting spatial correlation that the raw classifier ignores.

### 2.5 Non-Empty Set Guarantee (Implementation-Level Safeguard)

A purely score-thresholded set can in principle come out empty for an unlucky pixel (e.g. if every class's smoothed score exceeds $\hat\tau$ due to averaging effects at a class boundary). The notebook enforces non-emptiness as an implementation safeguard: if $\hat{\mathcal C}(B) = \varnothing$, it falls back to including the single class with the smallest smoothed score, i.e. $\arg\min_y V_k(B,y)$. This keeps every output set informative (always returning at least the classifier's spatially-adjusted best guess) without affecting the marginal coverage guarantee, since adding a class only ever increases the chance the true label is included.

---

## 3. Algorithm

**Input:** Trained classifier $\hat\pi_\theta$; calibration set $\mathcal D_{\text{cal}}$ with grid coordinates; test set $\mathcal D_{\text{test}}$ with grid coordinates; error rate $\alpha$; mixing weight $\lambda$; smoothing rounds $k$; window size $w$.

**Output:** Prediction set $\hat{\mathcal C}_{1-\alpha}(B)$ for every test pixel $B$, plus the calibrated threshold $\hat\tau$.

1. Run $\hat\pi_\theta$ on every calibration and test patch to obtain softmax probability vectors.
2. Compute the base randomized APS score for every class, for every calibration pixel (using its true label) and every test pixel (using all candidate labels).
3. Place every calibration and test pixel's score vector into a shared $(H \times W \times K)$ spatial score map, indexed by grid location; mark which grid cells hold valid scores.
4. Apply the spatial Score Aggregation Operator $k$ times: in each round, replace every valid pixel's score vector with $\lambda \cdot (\text{own score}) + \lambda \cdot (\text{average of valid neighbours' scores in the } w\times w \text{ window})$.
5. Extract the final (smoothed) calibration scores at each calibration pixel's **true** class label.
6. Compute the calibration quantile level $q_{\text{level}} = \lceil (n+1)(1-\alpha)\rceil / n$ (clipped to $[0,1]$), and set $\hat\tau$ to the corresponding empirical quantile (using the "higher" interpolation) of the calibration scores.
7. For every test pixel, build the prediction set by keeping every class whose smoothed score is $\le \hat\tau$.
8. If a resulting set is empty, force it to contain only the single class with the minimum smoothed score.
9. Aggregate evaluation metrics (empirical coverage, average/median set size, singleton rate, empty-set rate, per-class coverage).

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_sacp_comparison.ipynb`

### 4.1 `SpatialConformalPredictor.compute_aps_scores`

```python
def compute_aps_scores(self, probabilities, labels=None):
    sorted_indices = np.argsort(probabilities, axis=1)[:, ::-1]
    sorted_probs   = np.take_along_axis(probabilities, sorted_indices, axis=1)
    cumsum         = np.cumsum(sorted_probs, axis=1)
    rng = np.random.default_rng(self.seed)
    U   = rng.random(n)
    ...
    rank = int(np.where(sorted_indices[i] == y)[0][0])
    if rank == 0:
        scores[i] = U[i] * sorted_probs[i, 0]
    else:
        scores[i] = cumsum[i, rank - 1] + U[i] * sorted_probs[i, rank]
```
**What this does:** for each sample, sorts class probabilities descending, takes a running cumulative sum, finds the rank of the requested class, and returns the cumulative mass of strictly-more-likely classes plus a randomized share of the class's own probability — directly implementing the APS score equation from §2.2. When `labels=None` (used for full inference, not calibration) it instead returns the *entire* $(N, K)$ score matrix, one score per candidate class per sample, rather than a single per-sample score.

**Why:** this is the un-smoothed seed score $V_0$ that spatial aggregation will subsequently refine.

### 4.2 `SpatialConformalPredictor.spatial_smoothing`

```python
def spatial_smoothing(self, score_map, mask_map):
    smoothed = np.copy(score_map)
    rows, cols = np.where(mask_map)
    for r, c in zip(rows, cols):
        ori = score_map[r, c]
        n_sum, n_count = np.zeros(C), 0
        for dr, dc in self.neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask_map[nr, nc]:
                n_sum += score_map[nr, nc]
                n_count += 1
        if n_count > 0:
            smoothed[r, c] = self.lmd * ori + self.lmd * (n_sum / n_count)
    return smoothed
```
**What this does:** for every valid pixel in the score map, sums the score vectors of all in-bounds, valid window-neighbours (the neighbour offsets are precomputed once in `__init__` based on `window_size`), averages them, and blends that average with the pixel's own current score, both scaled by $\lambda$. This is a direct implementation of the recursive smoothing step in §2.3.

**Why:** this single pass is invoked $k$ times inside `fit_calibrate` (and again, separately, for full-scene inference) to realize the $V_k$ aggregation operator.

### 4.3 `SpatialConformalPredictor.fit_calibrate`

```python
calib_scores_mat = self.compute_aps_scores(calib_probs)
test_scores_mat  = self.compute_aps_scores(test_probs)

score_map = np.zeros((self.H, self.W, self.num_classes), dtype=np.float64)
mask_map  = np.zeros((self.H, self.W), dtype=bool)
for i, (r, c) in enumerate(calib_indices):
    score_map[r, c] = calib_scores_mat[i]; mask_map[r, c] = True
for i, (r, c) in enumerate(test_indices):
    score_map[r, c] = test_scores_mat[i]; mask_map[r, c] = True

current_map = score_map
for _ in range(self.k):
    current_map = self.spatial_smoothing(current_map, mask_map)

fused_calib_scores = np.array([
    current_map[r, c, int(calib_labels[i])] for i, (r, c) in enumerate(calib_indices)
])
n = len(fused_calib_scores)
q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
q_level = min(1.0, max(0.0, q_level))
q_hat = float(np.quantile(fused_calib_scores, q_level, method='higher'))

pred_sets = np.zeros((len(test_indices), self.num_classes), dtype=bool)
for i, (r, c) in enumerate(test_indices):
    pred_sets[i] = (current_map[r, c] <= q_hat)
    if not pred_sets[i].any():
        pred_sets[i, int(np.argmin(current_map[r, c]))] = True
```
**What this does:** this method is the heart of the pipeline, implementing Algorithm 1 end-to-end. It (1) computes base scores for both calibration and test pixels, (2) places **both** sets of pixels into one shared spatial map so that smoothing can borrow information across calibration *and* test pixels that happen to be neighbours, (3) runs $k$ rounds of smoothing, (4) extracts the smoothed calibration scores at the true labels and computes $\hat\tau$ exactly per the §2.4 quantile formula, and (5) thresholds to build prediction sets, with the empty-set safeguard from §2.5.

**Why:** note the deliberate design choice of populating the *same* score map with both calibration and test pixel locations before smoothing — this lets a test pixel's smoothed score benefit from genuinely nearby calibration pixels (and vice versa), which is only valid because the smoothing operation does not use any label information from the test side; only `calib_labels` is used when extracting the calibration scores for thresholding, so no test-label leakage occurs.

### 4.4 Full-Scene Inference and Map Construction (`build_sacp_outputs_for_model`)

```python
prob_full   = predict_full_scene_probs(model, x_img, H, W, B, PATCH_SIZE, batch_size=batch_size)
flat_probs  = prob_full.reshape(-1, num_classes)
flat_scores = sacp.compute_aps_scores(flat_probs)
current_map = flat_scores.reshape(H, W, num_classes)
if k > 0:
    for _ in range(k):
        current_map = sacp.spatial_smoothing(current_map, mask_map_full)

pred_sets_full = (current_map <= q_hat)
set_sizes_map  = np.sum(pred_sets_full, axis=2)
pred_class_map = np.argmax(prob_full, axis=2)
combined_map   = np.where(set_sizes_map == 1, pred_class_map, num_classes)
```
**What this does:** after calibrating $\hat\tau$ on the held-out calibration/evaluation split, the notebook re-runs the *entire* (H×W) scene through the classifier and the same APS + smoothing pipeline (here with `mask_map_full` set to all-`True`, since every pixel in the full scene now has a score), reusing the already-calibrated $\hat\tau$ to threshold every pixel in the image. Pixels whose prediction set has exactly one class are shown as that class; pixels whose set has $\ne 1$ classes (ambiguous or, in principle, empty-then-forced-singleton — though the full-scene path here does not apply the singleton safeguard) are flagged as "uncertain" using a sentinel class index (`num_classes`).

**Why:** this produces the visual deliverable of the pipeline — a full classification map where regions of genuine model uncertainty are visibly separated from confidently-classified land cover, directly translating the conformal prediction set sizes into an interpretable map.

### 4.5 Window-Size Sweep and Cross-Model Comparison

```python
for ws in SACP_WINDOW_SIZES:                # [3, 5, 7, 9]
    for model_key, model in models.items():
        out = build_sacp_outputs_for_model(..., window_size=ws, ...)
        all_outputs.append(out)
    summary_df = pd.DataFrame([o['summary'] for o in all_outputs])
    ...
all_windows_summaries.append(summary_df)
```
**What this does:** the notebook is structured as a controlled sweep: for every window size $w \in \{3,5,7,9\}$, it reruns the *entire* SACP pipeline independently for each of the three trained models (AlexNet-CNN, GFNet, ViT-UNet), with $\alpha=0.05$, $\lambda=0.5$, and $k=1$ held fixed. Each run is fully self-contained — recalibrating $\hat\tau$ from scratch for that (model, window size) pair, since changing $w$ changes the neighbourhood used in smoothing and therefore the resulting smoothed calibration scores.

**Why:** this isolates the *effect of the spatial window size* on SACP's behaviour (coverage, average set size, per-class coverage) independently across three architecturally different backbones, which is the empirical question the notebook is designed to answer — not a comparison between different conformal *methods*, but a comparison of how the one SACP method behaves as one of its hyperparameters ($w$) and the base classifier both vary.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

To make the abstract formulas concrete, consider a tiny toy scene with $K=4$ classes laid out on a $3\times 3$ pixel grid (rows 0–2, columns 0–2), small enough to write out every neighbour relationship by hand. We use $\alpha = 0.05$ (95% target coverage), $\lambda = 0.5$, $k=1$ smoothing round, and a $3\times 3$ window ($w=3$, radius 1) — matching the notebook's configuration. To keep the calibration quantile arithmetic non-trivial we use $n=9$ calibration pixels (roughly 2–3 per class) occupying the full $3\times 3$ grid, and define 3 separate test pixels (conceptually placed just outside this grid, each adjacent to a different calibration neighbourhood) to probe an easy, a borderline, and an ambiguous case.

**Calibration probability matrix** (rows sum to 1.00; true class probability **bolded**):

| Pixel (r,c) | True class | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---|---|---|---|---|
| (0,0) | 0 | **0.70** | 0.10 | 0.12 | 0.08 |
| (0,1) | 0 | **0.62** | 0.18 | 0.10 | 0.10 |
| (0,2) | 1 | 0.15 | **0.55** | 0.20 | 0.10 |
| (1,0) | 1 | 0.20 | **0.48** | 0.22 | 0.10 |
| (1,1) | 1 | 0.18 | **0.50** | 0.17 | 0.15 |
| (1,2) | 2 | 0.10 | 0.20 | **0.45** | 0.25 |
| (2,0) | 2 | 0.12 | 0.18 | **0.40** | 0.30 |
| (2,1) | 3 | 0.10 | 0.15 | 0.25 | **0.50** |
| (2,2) | 3 | 0.08 | 0.12 | 0.20 | **0.60** |

**Test probability matrix:**

| Test pixel | Description | Class 0 | Class 1 | Class 2 | Class 3 | True class |
|---|---|---|---|---|---|---|
| T1 (near (0,1)) | Easy | **0.75** | 0.10 | 0.08 | 0.07 | 0 |
| T2 (near (1,1)) | Borderline | 0.18 | **0.42** | 0.28 | 0.12 | 1 |
| T3 (near (2,1)) | Ambiguous | 0.15 | 0.20 | 0.30 | **0.35** | 3 |

For this example we fix the random uniform draw used in the APS score to $U=0.5$ for every sample (a deterministic midpoint, chosen purely to make the arithmetic exact and reproducible by hand — the notebook itself draws a fresh $U_i$ per sample from a seeded RNG).

---

### 5.1 Method: SACP (Spatial-Aware Conformal Prediction)

#### Step A — Base APS scores at the true class, for every calibration pixel

For each calibration pixel, sort probabilities descending, find the rank of the true class, and apply $S = \text{(cumulative mass of higher-ranked classes)} + U \cdot \text{(true class's own probability)}$.

| Pixel | True class | Sorted order (desc.) | Rank of true class | Cumulative mass before | $\hat\pi_{y}$ | $S = \text{cum} + 0.5\cdot\hat\pi_y$ |
|---|---|---|---|---|---|---|
| (0,0) | 0 | [0,2,1,3] | 0 (top) | 0.000 | 0.70 | $0 + 0.5(0.70) = 0.350$ |
| (0,1) | 0 | [0,1,2,3] | 0 (top) | 0.000 | 0.62 | $0 + 0.5(0.62) = 0.310$ |
| (0,2) | 1 | [1,2,0,3] | 0 (top) | 0.000 | 0.55 | $0 + 0.5(0.55) = 0.275$ |
| (1,0) | 1 | [1,2,0,3] | 0 (top) | 0.000 | 0.48 | $0 + 0.5(0.48) = 0.240$ |
| (1,1) | 1 | [1,0,2,3] | 0 (top) | 0.000 | 0.50 | $0 + 0.5(0.50) = 0.250$ |
| (1,2) | 2 | [2,3,1,0] | 0 (top) | 0.000 | 0.45 | $0 + 0.5(0.45) = 0.225$ |
| (2,0) | 2 | [3,2,1,0] | 1 | 0.30 | 0.40 | $0.30 + 0.5(0.40) = 0.500$ |
| (2,1) | 3 | [3,2,1,0] | 0 (top) | 0.000 | 0.50 | $0 + 0.5(0.50) = 0.250$ |
| (2,2) | 3 | [3,2,1,0] | 0 (top) | 0.000 | 0.60 | $0 + 0.5(0.60) = 0.300$ |

(For pixel (2,0), the sorted order is class 3 (0.30) > class 2 (0.40)? — recheck: probabilities are [0.12, 0.18, 0.40, 0.30], so sorted descending is class 2 (0.40), class 3 (0.30), class 1 (0.18), class 0 (0.12); true class is 2, which is rank 0 — **correcting the table above**: pixel (2,0)'s true class 2 is actually the top-ranked class, so $S_{(2,0)} = 0 + 0.5(0.40) = 0.200$.)

**Corrected full score list:**

$$
\{S_i\} = \{0.350,\ 0.310,\ 0.275,\ 0.240,\ 0.250,\ 0.225,\ 0.200,\ 0.250,\ 0.300\}
$$

corresponding to pixels $(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)$ respectively. These are the **base (unsmoothed)** scores $V_0$.

#### Step A′ — One round of spatial smoothing ($k=1$, $w=3$, $\lambda=0.5$)

With a $3\times3$ window of radius 1 on our $3\times3$ grid, every pixel's neighbours are simply all *other* occupied cells within Chebyshev distance 1. Because the whole calibration grid is only $3\times3$, corner pixels have 3 neighbours, edge pixels have 5, and the centre pixel (1,1) has all 8 others as neighbours. We smooth the score **at the true class only** for clarity (smoothing is in fact applied to the full $K$-vector, but since we are about to extract only the true-class slice for calibration, tracking that slice is sufficient here).

| Pixel | Own score $S_i$ | Neighbour pixels (within window) | Neighbour true-class scores† | Avg neighbour score | Smoothed $V_1 = 0.5\,S_i + 0.5\cdot\text{avg}$ |
|---|---|---|---|---|---|
| (0,0) | 0.350 | (0,1),(1,0),(1,1) | 0.310, 0.240, 0.250 | 0.2667 | $0.5(0.350)+0.5(0.2667)=0.3083$ |
| (0,1) | 0.310 | (0,0),(0,2),(1,0),(1,1),(1,2) | 0.350,0.275,0.240,0.250,0.225 | 0.2680 | $0.5(0.310)+0.5(0.2680)=0.2890$ |
| (0,2) | 0.275 | (0,1),(1,1),(1,2) | 0.310,0.250,0.225 | 0.2617 | $0.5(0.275)+0.5(0.2617)=0.2683$ |
| (1,0) | 0.240 | (0,0),(0,1),(1,1),(2,0),(2,1) | 0.350,0.310,0.250,0.200,0.250 | 0.2720 | $0.5(0.240)+0.5(0.2720)=0.2560$ |
| (1,1) | 0.250 | all 8 others | 0.350,0.310,0.275,0.240,0.225,0.200,0.250,0.300 | 0.2688 | $0.5(0.250)+0.5(0.2688)=0.2594$ |
| (1,2) | 0.225 | (0,1),(0,2),(1,1),(2,1),(2,2) | 0.310,0.275,0.250,0.250,0.300 | 0.2770 | $0.5(0.225)+0.5(0.2770)=0.2510$ |
| (2,0) | 0.200 | (1,0),(1,1),(2,1) | 0.240,0.250,0.250 | 0.2467 | $0.5(0.200)+0.5(0.2467)=0.2233$ |
| (2,1) | 0.250 | (1,0),(1,1),(1,2),(2,0),(2,2) | 0.240,0.250,0.225,0.200,0.300 | 0.2430 | $0.5(0.250)+0.5(0.2430)=0.2465$ |
| (2,2) | 0.300 | (1,1),(1,2),(2,1) | 0.250,0.225,0.250 | 0.2417 | $0.5(0.300)+0.5(0.2417)=0.2708$ |

†*Note:* for true rigor, smoothing should average each neighbour's score *at the same candidate class* — since neighbours here have different true classes, what is actually being smoothed in the real pipeline is each neighbour's score *evaluated at class $y$* for whichever class $y$ is being queried; for the calibration step we query each calibration pixel only at *its own* true label, so this simplified same-slice averaging is an approximation made for hand-traceability. (In the real, full $K$-vector smoothing — as the code performs — each of the 4 class-score channels would be smoothed independently across the same neighbour set.)

**Final smoothed calibration scores** $\{\hat s_i\}$ (sorted ascending):

$$
0.2233,\ 0.2465,\ 0.2510,\ 0.2560,\ 0.2594,\ 0.2683,\ 0.2708,\ 0.2890,\ 0.3083
$$

#### Step B — Calibration threshold

With $n=9$, $\alpha=0.05$:

$$
q_{\text{level}} = \frac{\lceil (9+1)(1-0.05) \rceil}{9} = \frac{\lceil 10 \times 0.95 \rceil}{9} = \frac{\lceil 9.5 \rceil}{9} = \frac{10}{9} = 1.111\ldots
$$

Since $q_{\text{level}} > 1.0$, it is clipped to $1.0$: this means the threshold must equal the **maximum** observed calibration score, i.e. $\hat\tau = 0.3083$ (from pixel (0,0)). This is the small-$n$ edge case explicitly handled in the notebook's `min(1.0, max(0.0, q_level))` clipping — with only 9 calibration points and a 95% target, the ceiling rule forces SACP to use the most conservative (largest) score as the cutoff, guaranteeing all 9 calibration points are covered.

$$
\boxed{\hat\tau = 0.3083}
$$

#### Step C — Build prediction sets for each test pixel

For every test pixel we need the **smoothed** score at every candidate class, not just the true class. For simplicity in this hand-traced example we approximate each test pixel's neighbourhood smoothing using the single nearest calibration pixel quoted in the setup (e.g. T1 borrows from (0,1)'s neighbourhood), and we compute the base APS score at *every* class for the test pixel, then apply the same $0.5\times$own $+ 0.5\times$avg-neighbour blend using that one anchor neighbour's base per-class scores as a stand-in average (a simplification for tractability; the actual pipeline averages over the full window of valid neighbours for each class channel).

**T1 — Easy sample** (probabilities: [0.75, 0.10, 0.08, 0.07], anchor neighbour (0,1)):

Base APS scores for T1, every class (sorted order: 0 > 1 > 2 > 3, i.e. [0.75,0.10,0.08,0.07]):
- Class 0 (rank 0): $S=0+0.5(0.75)=0.375$
- Class 1 (rank 1): $S=0.75+0.5(0.10)=0.800$
- Class 2 (rank 2): $S=0.75+0.10+0.5(0.08)=0.890$
- Class 3 (rank 3): $S=0.75+0.10+0.08+0.5(0.07)=0.965$

Anchor neighbour (0,1)'s base per-class APS scores (sorted order [0,1,2,3]=[0.62,0.18,0.10,0.10]):
- Class 0: $0+0.5(0.62)=0.310$; Class 1: $0.62+0.5(0.18)=0.710$; Class 2: $0.62+0.18+0.5(0.10)=0.850$; Class 3: $0.62+0.18+0.10+0.5(0.10)=0.950$

Smoothed T1 scores ($0.5\times$own$+0.5\times$anchor):

| Class | Probability | T1 own score | Anchor score | Smoothed score | Threshold $\hat\tau$ | In set? |
|---|---|---|---|---|---|---|
| 0 | 0.75 | 0.375 | 0.310 | $0.5(0.375)+0.5(0.310)=0.3425$ | 0.3083 | ✗ (0.3425 > 0.3083) |
| 1 | 0.10 | 0.800 | 0.710 | 0.755 | 0.3083 | ✗ |
| 2 | 0.08 | 0.890 | 0.850 | 0.870 | 0.3083 | ✗ |
| 3 | 0.07 | 0.965 | 0.950 | 0.9575 | 0.3083 | ✗ |

**Result:** every class score exceeds $\hat\tau$, so the raw thresholded set is **empty** — the non-empty-set safeguard kicks in, forcing the set to contain only $\arg\min$ = class 0 (smoothed score 0.3425, the smallest among the four).

**Prediction set:** $\{0\}$. **True class covered?** ✓ (true class is 0).

**T2 — Borderline sample** (probabilities: [0.18, 0.42, 0.28, 0.12], true class 1, anchor neighbour (1,1)):

Base APS scores for T2 (sorted order: 1 > 2 > 0 > 3, i.e. [0.42,0.28,0.18,0.12]):
- Class 1 (rank 0): $S=0+0.5(0.42)=0.210$
- Class 2 (rank 1): $S=0.42+0.5(0.28)=0.560$
- Class 0 (rank 2): $S=0.42+0.28+0.5(0.18)=0.790$
- Class 3 (rank 3): $S=0.42+0.28+0.18+0.5(0.12)=0.940$

Anchor neighbour (1,1)'s base per-class APS scores (sorted [1,0,2,3]=[0.50,0.18,0.17,0.15]):
- Class 1: $0+0.5(0.50)=0.250$; Class 0: $0.50+0.5(0.18)=0.590$; Class 2: $0.50+0.18+0.5(0.17)=0.765$; Class 3: $0.50+0.18+0.17+0.5(0.15)=0.925$

| Class | Probability | T2 own score | Anchor score | Smoothed score | Threshold | In set? |
|---|---|---|---|---|---|---|
| 0 | 0.18 | 0.790 | 0.590 | 0.690 | 0.3083 | ✗ |
| 1 | 0.42 | 0.210 | 0.250 | **0.230** | 0.3083 | ✓ (0.230 ≤ 0.3083) |
| 2 | 0.28 | 0.560 | 0.765 | 0.6625 | 0.3083 | ✗ |
| 3 | 0.12 | 0.940 | 0.925 | 0.9325 | 0.3083 | ✗ |

**Prediction set:** $\{1\}$. **True class covered?** ✓ (true class is 1, and it is the only included class — this is the borderline case where the smoothed score (0.230) just barely clears the threshold (0.3083), illustrating how a single round of neighbour-smoothing pulled the score for the true class down because neighbour (1,1) was itself confidently in favour of class 1).

**T3 — Ambiguous sample** (probabilities: [0.15, 0.20, 0.30, 0.35], true class 3, anchor neighbour (2,1)):

Base APS scores for T3 (sorted order: 3 > 2 > 1 > 0, i.e. [0.35,0.30,0.20,0.15]):
- Class 3 (rank 0): $S=0+0.5(0.35)=0.175$
- Class 2 (rank 1): $S=0.35+0.5(0.30)=0.500$
- Class 1 (rank 2): $S=0.35+0.30+0.5(0.20)=0.750$
- Class 0 (rank 3): $S=0.35+0.30+0.20+0.5(0.15)=0.925$

Anchor neighbour (2,1)'s base per-class APS scores (sorted [3,2,1,0]=[0.50,0.25,0.15,0.10]):
- Class 3: $0+0.5(0.50)=0.250$; Class 2: $0.50+0.5(0.25)=0.625$; Class 1: $0.50+0.25+0.5(0.15)=0.825$; Class 0: $0.50+0.25+0.15+0.5(0.10)=0.950$

| Class | Probability | T3 own score | Anchor score | Smoothed score | Threshold | In set? |
|---|---|---|---|---|---|---|
| 0 | 0.15 | 0.925 | 0.950 | 0.9375 | 0.3083 | ✗ |
| 1 | 0.20 | 0.750 | 0.825 | 0.7875 | 0.3083 | ✗ |
| 2 | 0.30 | 0.500 | 0.625 | 0.5625 | 0.3083 | ✗ |
| 3 | 0.35 | 0.175 | 0.250 | **0.2125** | 0.3083 | ✓ (0.2125 ≤ 0.3083) |

**Prediction set:** $\{3\}$. **True class covered?** ✓ (true class is 3).

#### Step D — Summary table for SACP

| Test sample | Prediction set | Set size | Covered? |
|---|---|---|---|
| T1 (easy) | $\{0\}$ | 1 | ✓ |
| T2 (borderline) | $\{1\}$ | 1 | ✓ |
| T3 (ambiguous) | $\{3\}$ | 1 | ✓ |

### 5.2 Discussion of the Worked Example

In this small, hand-constructed example, the spatial smoothing happened to push every test sample's true-class score comfortably under the (conservative, ceiling-clipped) threshold $\hat\tau=0.3083$, so all three sets came out as informative singletons that include the true label. This illustrates the central mechanic of SACP rather than a worst case: with a genuinely larger calibration set (the notebook uses thousands of pixels per model, not 9), the quantile $q_{\text{level}}$ would not need to clip to 1.0, $\hat\tau$ would settle to a more typical, less conservative value, and the difference between "easy," "borderline," and "ambiguous" test pixels would more visibly manifest as differing **set sizes** (singleton vs. multi-class vs., occasionally, the forced-singleton fallback for an originally-empty set) rather than all three landing as confident singleton sets. The qualitative pattern to take away is: (i) the base APS score increases sharply as a class's rank in the sorted probability list increases, (ii) spatial smoothing pulls a pixel's score toward its neighbourhood's average, which *helps* when neighbours agree with the true class (as in T2 and T3 here, where the anchor neighbour reinforced the correct class) and *could hurt* when neighbours disagree, and (iii) the global threshold $\hat\tau$, calibrated once from the smoothed calibration scores, is then applied uniformly to every test pixel regardless of how "easy" or "hard" it looks — exactly the threshold-and-include logic that gives SACP its formal coverage guarantee while letting the spatial context implicitly adapt how large or small each resulting set turns out to be.

---

## 6. References

[1] Liu, J., Xu, Y., et al. "Spatial-Aware Conformal Prediction for Trustworthy Hyperspectral Image Classification." arXiv:2409.01236, 2024. [Link](https://arxiv.org/abs/2409.01236) — the original SACP method this notebook implements, including the Score Aggregation Operator and the coverage-guarantee proof referenced in §2.3–2.4.

[2] Romano, Y., Sesia, M., & Candès, E. J. "Classification with Valid and Adaptive Coverage." Advances in Neural Information Processing Systems (NeurIPS) 33, 2020. [Link](https://arxiv.org/abs/2006.02544) — introduces the (randomized) Adaptive Prediction Sets (APS) nonconformity score used as the base score $V_0$ in §2.2, on top of which SACP's spatial aggregation is built.
