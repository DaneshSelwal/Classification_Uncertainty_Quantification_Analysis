# Baseline Uncertainty & Conformal Prediction Walkthrough (`Classification/baseline/Model_uncertainty_comparison.ipynb`)

## 1. Plain-English Overview ("The Big Idea")
This is the second notebook in the pipeline and introduces the core **Conformal Prediction (CP)** and **Calibration** methods. Instead of outputting a single best-guess class $\hat{y}$ that might be wrong on ambiguous pixels, Conformal Prediction outputs a **prediction set** $C(x) \subseteq \{1, \dots, K\}$ mathematically guaranteed to contain the true land-cover class at least $1 - \alpha$ (e.g., $95\%$ for $\alpha = 0.05$) of the time.

## 2. Stage-by-Stage Process & Math

### Stage 1: Auditing Calibration (ECE & Brier Score)
Before forming prediction sets, we check whether the model's confidence matches its actual accuracy:
1. **Expected Calibration Error (ECE)** across $M=15$ confidence bins $B_m$:
$$
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$
2. **B
### Stage 2: Basic Split Conformal Prediction (THR / LAC)
Split CP uses the 15% held-out validation split as a **Calibration Set ($D_{\text{calib}}$)** in 3 simple steps:
1. **Non-Conformity Score ($s_i$)**: For each calibration pixel $(x_i, y_i)$, measure how much the model doubted the true class $y_i$:
$$
s_i = 1 - \hat{p}_{i, y_i}
$$
2. **Quantile Threshold ($\hat{q}_{1-\alpha}$)**: Sort the $N_{\text{val}}$ calibration scores and take the finite-sample corrected $(1-\alpha)$ quantile:
$$
\hat{q}_{1-\alpha} = \text{Quantile}\left( \{s_1, \dots, s_{N_{\text{val}}}\}, \; \frac{\lceil (N_{\text{val}} + 1)(1 - \alpha) \rceil}{N_{\text{val}}} \right)
$$
3. **Prediction Set Construction**: For any new test pixel $x$, include every class $k$ whose score $1 - \hat{p}_k(x) \le \hat{q}_{1-\alpha}$:
$$
C_{\text{THR}}(x) = \left\{ k \in \{1, \dots, K\} : \hat{p}_k(x) \ge 1 - \hat{q}_{1-\alpha} \right\}
$$

### Stage 3: Adaptive Prediction Sets (APS) & Regularized APS (RAPS)
- **Why upgrade from THR to APS?** If a very uncertain pixel has flat probabilities $[0.25, 0.25, 0.25, 0.25]$ and $1 - \hat{q}_{1-\alpha} = 0.30$, THR outputs an **empty set** $C(x) = \emptyset$.
- **APS** sorts probabilities from highest to lowest $\pi_{(1)}(x) \ge \pi_{(2)}(x) \ge \dots \ge \pi_{(K)}(x)$ and accumulates them up to the true class rank $\tau(y)$ (with uniform tie-breaker $u \sim U(0,1)$):
$$
s_{\text{APS}}(x, y) = \sum_{j=1}^{\tau(y)-1} \pi_{(j)}(x) + u \cdot \pi_{(\tau(y))}(x)
$$
- **RAPS** adds a penalty $\lambda \max(0, \tau(y) - k_{\text{reg}})$ for ranks beyond $k_{\text{reg}}$ so tail classes do not bloat the prediction set:
$$
s_{\text{RAPS}}(x, y) = \sum_{j=1}^{\tau(y)-1} \pi_{(j)}(x) + u \cdot \pi_{(\tau(y))}(x) + \lambda \max(0, \tau(y) - k_{\text{reg}})
$$

### Stage 4: Monte Carlo (MC) Dropout
Keeps Dropout active at test time over $T=20$ stochastic passes to compute predictive mean $\bar{p}(y \mid x) = \frac{1}{T}\sum_{t=1}^T p^{(t)}(y \mid x)$ and predictive entropy $\mathcal{H}[\bar{p}] = -\sum_k \bar{p}_k \log \bar{p}_k$.

## 3. Worked Numerical Example (Split CP vs. APS)
Suppose for a test pixel $x$, sorted class probabilities are *Forest* ($0.52$), *Crop* ($0.38$), *Urban* ($0.07$), *Water* ($0.03$) and target coverage is $95\%$ ($\alpha = 0.05$):
- **Split CP (THR)** with $\hat{q}_{0.95} = 0.65$ (minimum probability $1 - 0.65 = 0.35$): Includes *Forest* ($0.52 \ge 0.35$) and *Crop* ($0.38 \ge 0.35$) $\rightarrow$ $C_{\text{THR}}(x) = \{\text{Forest}, \text{Crop}\}$.
- **APS** with cumulative threshold $\hat{q}_{0.95}^{\text{APS}} = 0.88$: Rank 1 is $0.52 < 0.88$; Rank 1 + Rank 2 is $0.52 + 0.38 = 0.90 \ge 0.88$ $\rightarrow$ $C_{\text{APS}}(x) = \{\text{Forest}, \text{Crop}\}$. Both methods tell us with 95% confidence that this boundary pixel is either *Forest* or *Crop*.rier Score (BS)**:
$$
\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (\hat{p}_{i,k} - \mathbb{I}(y_i = k))^2

$$
