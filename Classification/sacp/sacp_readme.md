# Spatial-Aware Conformal Prediction (SACP): Theory & Implementation Summary

> **One-line description:** SACP wraps any trained hyperspectral image classifier with statistically guaranteed prediction sets by propagating and spatially smoothing APS nonconformity scores across pixel neighbourhoods before calibrating a coverage threshold.

---

## 1. Overview & Intuition

Hyperspectral image (HSI) classification assigns a land-cover label to every pixel in a scene that may contain hundreds of spectral bands. Modern deep classifiers (CNNs, Vision Transformers, GFNet-style frequency networks) have achieved impressive accuracy on this task, but their softmax outputs are poorly calibrated—a high softmax score does not reliably translate to a correspondingly high probability of being correct. For safety-critical remote sensing applications this is a serious limitation.

**Conformal Prediction (CP)** offers a distribution-free solution: given any black-box classifier and a held-out calibration set, CP produces a *prediction set*—a set of class labels—that is *guaranteed* to contain the true label with at least a user-specified probability (e.g., 95%), without any parametric assumptions about the data distribution. The only assumption needed is *exchangeability*: calibration and test samples must be drawn from the same distribution in a way that makes their ordering arbitrary.

Standard (split) conformal prediction applies a single global threshold to the entire image, ignoring the spatial structure inherent in HSI data. Adjacent pixels in an HSI are highly correlated—they typically belong to the same land-cover patch and produce similar spectral signatures. **SACP** (Spatial-Aware Conformal Prediction, Liu et al., 2024) exploits this spatial correlation by blending each pixel's nonconformity score with the scores of its neighbours before calibration. The result is a spatially coherent, smoother score map that tends to produce smaller, more informative prediction sets while still satisfying the coverage guarantee.

The notebook applies SACP with the APS (Adaptive Prediction Sets) score function to three architectures—AlexNet, GFNet, and ViT-UNet—on a 6-band multispectral image, sweeping over four spatial smoothing window sizes (3, 5, 7, 9) to study the coverage–set-size trade-off.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let $\mathcal{X} = \mathbb{R}^{P \times P \times B}$ be the space of spectral patches (patch size $P$, $B$ bands) and $\mathcal{Y} = \{1, \ldots, C\}$ the set of $C$ land-cover classes. A trained classifier $\hat{f} : \mathcal{X} \to [0,1]^C$ outputs a softmax probability vector.

The full scene is a 2-D grid of $H \times W$ pixels. Each pixel at spatial position $(r,c)$ carries a patch $x_{r,c} \in \mathcal{X}$ and a true label $y_{r,c} \in \mathcal{Y}$.

The dataset is split into three disjoint parts:
- **Training set** $\mathcal{D}_{\text{train}}$: used to fit $\hat{f}$ (handled offline; models are loaded from saved weights).
- **Calibration set** $\mathcal{D}_{\text{cal}} = \{(x_i, y_i, (r_i,c_i))\}_{i=1}^{n}$: used to learn the SACP threshold $\hat{q}$.
- **Evaluation set** $\mathcal{D}_{\text{eval}}$: used to measure empirical coverage and set size.

The miscoverage rate is $\alpha$ (here $\alpha = 0.05$, targeting $\geq 95\%$ coverage). The smoothing strength is controlled by $\lambda \in [0,1]$ and the number of smoothing iterations $k$.

### 2.2 APS Nonconformity Score

The notebook uses the **Adaptive Prediction Sets (APS)** score (Angelopoulos et al., 2021, building on Romano et al., 2020). APS accumulates softmax probabilities in descending order down to the true class, with a randomisation term $U_i \sim \text{Uniform}(0,1)$ to achieve exchangeability.

**Definition (APS score for calibration):** Given predicted probabilities $\hat{f}(x_i) \in [0,1]^C$ and true label $y_i$, let $\pi_i$ be the permutation that sorts classes by descending probability, so $\hat{f}(x_i)_{\pi_i(1)} \geq \hat{f}(x_i)_{\pi_i(2)} \geq \cdots$. Let $L_i = \mathrm{rank}(y_i; \pi_i)$ be the 0-based rank of the true class. Then:

$$s_i = \sum_{j=1}^{L_i - 1} \hat{f}(x_i)_{\pi_i(j)} \;+\; U_i \cdot \hat{f}(x_i)_{\pi_i(L_i)}$$

**Where:**
- $\hat{f}(x_i)_{\pi_i(j)}$ — the $j$-th largest softmax probability for sample $i$
- $L_i$ — the rank of the true class in the descending probability ordering (0-based; $L_i = 0$ means the classifier's top prediction is correct)
- $U_i \sim \text{Uniform}(0,1)$ — a randomisation draw, fixed by a seed for reproducibility
- The leading sum is zero when $L_i = 0$

**What this means:** $s_i$ measures how much probability mass the model places on classes it ranked *above* the true class, plus a random fraction of the true class's own probability. A low score means the model was confident and correct; a high score means the true class was buried under many higher-ranked alternatives. Crucially, under exchangeability the scores $(s_1, \ldots, s_n)$ are i.i.d., which underpins the coverage guarantee.

**Full score matrix for inference:** When labels are unknown (at test time), SACP computes a $C$-dimensional score vector for every pixel. For each class $y \in \{1,\ldots,C\}$, the score is the cumulative probability up to that class's rank:

$$s_i^{(y)} = \sum_{j=1}^{\mathrm{rank}(y;\pi_i)-1} \hat{f}(x_i)_{\pi_i(j)} \;+\; U_i \cdot \hat{f}(x_i)_{\pi_i(\mathrm{rank}(y;\pi_i))}$$

This produces the score matrix $S \in \mathbb{R}^{N \times C}$ used for spatial smoothing and set construction.

### 2.3 Spatial Score Map and Smoothing

SACP's key innovation is placing scores onto the 2-D pixel grid and mixing each pixel's score with its neighbourhood average.

**Score map:** Scores are written into a spatial array $\mathbf{M} \in \mathbb{R}^{H \times W \times C}$, where $\mathbf{M}[r,c,:] = \mathbf{s}_{r,c}$ is the $C$-dimensional score vector of the pixel at position $(r,c)$.

**One smoothing pass:**

$$\tilde{\mathbf{M}}[r,c] = \lambda \cdot \mathbf{M}[r,c] \;+\; \lambda \cdot \frac{1}{|\mathcal{N}(r,c)|} \sum_{(r',c') \in \mathcal{N}(r,c)} \mathbf{M}[r',c']$$

**Where:**
- $\mathcal{N}(r,c)$ — the set of valid neighbouring pixel positions within a $(W_s \times W_s)$ square window centred on $(r,c)$, excluding $(r,c)$ itself; $W_s \in \{3,5,7,9\}$ in the notebook
- $\lambda = 0.5$ — the blending weight (equal mixture of self and neighbour average)
- The smoothing is applied $k$ times (here $k = 1$)

**What this means:** After smoothing, each pixel's score reflects not only its own model confidence but also the confidence of its spatial neighbours. Adjacent pixels that share a land-cover class tend to have similar scores, so smoothing reduces noise in the score field and concentrates scores closer to their true local value.

### 2.4 Conformal Threshold Calibration

After $k$ smoothing passes, the calibration scores are read back from the smoothed map:

$$\tilde{s}_i = \tilde{\mathbf{M}}[r_i, c_i, y_i], \quad i = 1, \ldots, n$$

The conformal quantile is then:

$$\hat{q} = \text{Quantile}\!\left(\{\tilde{s}_1, \ldots, \tilde{s}_n\},\; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$

**Where:**
- $n$ — number of calibration samples
- $\alpha$ — miscoverage level (0.05 in this notebook)
- The slightly inflated quantile level $\lceil(n+1)(1-\alpha)\rceil / n$ ensures the finite-sample guarantee: $\Pr(Y_{\text{test}} \in \hat{C}(X_{\text{test}})) \geq 1 - \alpha$

### 2.5 Prediction Set Construction

For a test pixel at position $(r,c)$, the smoothed score vector $\tilde{\mathbf{M}}[r,c,:]$ is compared against $\hat{q}$:

$$\hat{C}(x_{r,c}) = \{y \in \mathcal{Y} : \tilde{\mathbf{M}}[r,c,y] \leq \hat{q}\}$$

If the resulting set is empty (which can happen when all smoothed scores exceed $\hat{q}$), the class with the smallest score is included to guarantee non-emptiness:

$$\hat{C}(x_{r,c}) \leftarrow \hat{C}(x_{r,c}) \cup \{\arg\min_y \tilde{\mathbf{M}}[r,c,y]\}$$

**Coverage guarantee:** Under exchangeability, the marginal coverage satisfies:

$$\Pr\!\bigl(Y_{\text{test}} \in \hat{C}(X_{\text{test}})\bigr) \geq 1 - \alpha$$

This is a finite-sample, distribution-free guarantee—it holds for every compliant model and dataset, regardless of the architecture used.

**Singleton set interpretation:** A prediction set of size 1 is called a *certain* prediction: the method is confident enough to commit to a single label. High singleton rates (close to 1.0, as seen in the results) indicate that the classifier is well-calibrated for most pixels and that spatial smoothing tightened the scores enough to make the threshold discriminative.

---

## 3. Algorithm

**Input:** Calibration set $\mathcal{D}_{\text{cal}}$ (patches, labels, pixel coordinates); evaluation set $\mathcal{D}_{\text{eval}}$; trained classifier $\hat{f}$; spatial grid dimensions $H, W$; window size $W_s$; smoothing weight $\lambda$; smoothing iterations $k$; miscoverage level $\alpha$.

**Output:** Prediction sets $\hat{C}(x)$ for each evaluation pixel; threshold $\hat{q}$; empirical coverage.

1. Run $\hat{f}$ on all calibration and evaluation patches; normalize softmax outputs.
2. Compute the full $C$-dimensional APS score matrix for every calibration and evaluation pixel using a fixed random seed.
3. Populate the spatial score map $\mathbf{M} \in \mathbb{R}^{H \times W \times C}$ by placing each pixel's score vector at its grid position.
4. Repeat $k$ times: replace $\mathbf{M}$ with one pass of spatial smoothing using window $W_s$ and weight $\lambda$.
5. Extract calibration scores $\tilde{s}_i = \tilde{\mathbf{M}}[r_i, c_i, y_i]$ from the smoothed map.
6. Compute threshold $\hat{q}$ as the $\lceil(n+1)(1-\alpha)\rceil/n$-quantile of $\{\tilde{s}_1, \ldots, \tilde{s}_n\}$.
7. For each evaluation pixel: $\hat{C} = \{y : \tilde{\mathbf{M}}[r,c,y] \leq \hat{q}\}$; if empty, add the argmin.
8. Report empirical coverage, average set size, singleton rate, and per-class coverage.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_sacp_comparison.ipynb`

### 4.1 APS Score Computation (`compute_aps_scores`)

```python
def compute_aps_scores(self, probabilities, labels=None):
    sorted_indices = np.argsort(probabilities, axis=1)[:, ::-1]
    sorted_probs   = np.take_along_axis(probabilities, sorted_indices, axis=1)
    cumsum         = np.cumsum(sorted_probs, axis=1)

    rng = np.random.default_rng(self.seed)
    U   = rng.random(n)

    if labels is not None:  # calibration mode
        for i in range(n):
            rank = int(np.where(sorted_indices[i] == y)[0][0])
            scores[i] = cumsum[i, rank - 1] + U[i] * sorted_probs[i, rank]
        return scores  # shape (n,)

    else:  # inference mode — full score matrix
        for i in range(n):
            scores_sorted[0]  = U[i] * sorted_probs[i, 0]
            scores_sorted[1:] = cumsum[i, :-1] + U[i] * sorted_probs[i, 1:]
            scores_matrix[i, sorted_indices[i]] = scores_sorted
        return scores_matrix  # shape (n, C)
```

**What this does:** Sorts each pixel's softmax vector in descending order, accumulates it cumulatively, then picks the cumulative value at the true class's rank position plus a randomised fraction of that class's probability.

**Why:** The APS score is monotone in the model's uncertainty—it equals the cumulative mass the model assigns to classes ranked higher than the true class. The randomisation term $U \cdot p_{\text{true}}$ ensures that ties are broken without bias, preserving the exchangeability argument needed for the coverage guarantee.

### 4.2 Spatial Smoothing (`spatial_smoothing`)

```python
def spatial_smoothing(self, score_map, mask_map):
    for r, c in zip(rows, cols):
        ori   = score_map[r, c]            # current pixel's C-dim score
        n_sum = np.zeros(C)
        for dr, dc in self.neighbors:      # iterate over window offsets
            if 0 <= nr < H and 0 <= nc < W and mask_map[nr, nc]:
                n_sum   += score_map[nr, nc]
                n_count += 1
        if n_count > 0:
            smoothed[r, c] = self.lmd * ori + self.lmd * (n_sum / n_count)
    return smoothed
```

**What this does:** For every labelled pixel, replaces its $C$-dimensional score vector with a weighted average of itself and the mean of its valid neighbours inside the $W_s \times W_s$ window.

**Why:** Nearby HSI pixels share spectral characteristics and class identity. Averaging their scores encourages consistency: if a pixel is on the interior of a homogeneous region its score tightens toward the regional mean; boundary pixels remain more uncertain because their neighbours may carry scores from a different class.

### 4.3 Calibration Quantile and Set Construction (`fit_calibrate`)

```python
fused_calib_scores = [current_map[r, c, int(y)] for (r,c), y in zip(coords_cal, labels_cal)]
q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
q_hat   = float(np.quantile(fused_calib_scores, q_level, method='higher'))

for i, (r, c) in enumerate(test_indices):
    pred_sets[i] = (current_map[r, c] <= q_hat)
    if not pred_sets[i].any():
        pred_sets[i, int(np.argmin(current_map[r, c]))] = True
```

**What this does:** Extracts each calibration pixel's score for its own true class from the smoothed map, then selects the $\lceil(n+1)(1-\alpha)\rceil/n$-quantile as the threshold. At test time, any class whose smoothed score falls below this threshold is included in the prediction set.

**Why:** The inflated quantile level is the standard conformal correction ensuring the finite-sample coverage inequality holds with probability at least $1 - \alpha$ over the randomness in the calibration set.

### 4.4 Full-Scene Visualisation

After calibration, the notebook regenerates predictions for every pixel in the $H \times W$ scene (including unlabelled background pixels), applies smoothing to the full spatial score map, then builds three visual outputs: a binary certain/uncertain map, a false-colour class map with uncertain pixels masked in grey, and a pixel-count bar chart.

---

## 5. Worked Numerical Example

**Setup:** 5 classes, 4 calibration pixels, 1 test pixel. Window size 3 (immediate 8-neighbour ring). $\alpha = 0.05$, $\lambda = 0.5$, $k = 1$, seed fixed.

**Calibration pixel scores (after APS, before smoothing):**

| Pixel $(r,c)$ | True class | Raw score $s_i$ |
|---|---|---|
| (1,1) | 2 | 0.31 |
| (1,2) | 2 | 0.28 |
| (2,1) | 3 | 0.45 |
| (2,2) | 2 | 0.29 |

Test pixel at (1,1) also participates in the score map.

**Step 1 — Populate map.** Scores at their grid positions.

**Step 2 — Smooth pixel (1,1).** Neighbours within a 3×3 window (on the populated grid) include (1,2) and (2,1) and (2,2). Suppose the class-2 score component is:
- own: 0.31
- neighbour average: $(0.28 + 0.45 + 0.29)/3 \approx 0.34$
- smoothed: $0.5 \times 0.31 + 0.5 \times 0.34 = 0.325$

**Step 3 — Extract calibration scores.** Read each pixel's smoothed score for its true class. Suppose the four smoothed true-class scores are $\{0.325, 0.29, 0.43, 0.30\}$.

**Step 4 — Compute $\hat{q}$.**
$$n = 4, \quad q_\text{level} = \frac{\lceil 5 \times 0.95 \rceil}{4} = \frac{5}{4} = 1.0 \;\Rightarrow\; \hat{q} = \max\{0.325, 0.29, 0.43, 0.30\} = 0.43$$

**Step 5 — Build test prediction set.** For the test pixel at (2,1), suppose its smoothed score vector across 5 classes is $[0.52,\; 0.41,\; 0.29,\; 0.38,\; 0.61]$.

Classes with score $\leq 0.43$: class 2 (0.41 ✓), class 3 (0.29 ✓), class 4 (0.38 ✓).

**Result:** $\hat{C}(x) = \{2, 3, 4\}$. The true label (class 3) is contained in the set, consistent with the 95% coverage guarantee. If the model were more accurate, only class 3 would survive the threshold (singleton set), classified as *certain*.

---

## 6. References

[1] Kangdao Liu, Tianhao Sun, Hao Zeng, Yongshan Zhang, Chi-Man Pun, Chi-Man Vong. "Spatial-Aware Conformal Prediction for Trustworthy Hyperspectral Image Classification." *arXiv:2409.01236*, 2024. [https://arxiv.org/abs/2409.01236](https://arxiv.org/abs/2409.01236)

[2] Anastasios N. Angelopoulos, Stephen Bates, Jitendra Malik, Michael I. Jordan. "Uncertainty Sets for Image Classifiers using Conformal Prediction." *ICLR 2021 Spotlight*, arXiv:2009.14193. [https://arxiv.org/abs/2009.14193](https://arxiv.org/abs/2009.14193)

[3] Anastasios N. Angelopoulos, Stephen Bates. "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." *arXiv:2107.07511*, 2021. [https://arxiv.org/abs/2107.07511](https://arxiv.org/abs/2107.07511)

[4] Yaniv Romano, Matteo Sesia, Emmanuel J. Candès. "Classification with Valid and Adaptive Coverage." *NeurIPS 2020*. (Original APS paper.)
