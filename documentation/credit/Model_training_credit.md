# CREDIT (Credal Ensemble Distillation) Walkthrough (`Classification/credit/Model_training_credit.ipynb`)

## 1. Plain-English Overview ("The Big Idea")
A standard neural network only outputs a single probability vector $\hat{p} = (\hat{p}_1, \dots, \hat{p}_K)$. Even if $\hat{p}_{\text{Forest}} = 0.65$, we cannot tell whether the model is solidly sure the probability is $0.65$ or wildly guessing between $0.25$ and $0.95$.

**CREDIT (Credal Ensemble Distillation)** represents uncertainty as a **Credal Set**—an interval of probabilities $[\underline{p}_k(x), \overline{p}_k(x)]$ bounded by a **lower probability** and an **upper probability** for each class $k$. The width of this interval directly measures epistemic ambiguity.

## 2. Stage-by-Stage Process & Math
### Stage 1: Lower and Upper Probability Bounds (Credal Intervals)
Given mean probability $\bar{p}_k(x)$ and predictive dispersion $\sigma_k(x)$ across stochastic passes/heads, CREDIT constructs lower and upper probability bounds for every class $k$:

$$
\underline{p}_k(x) = \max\left(0, \; \bar{p}_k(x) - \gamma \cdot \sigma_k(x)\right), \quad \overline{p}_k(x) = \min\left(1, \; \bar{p}_k(x) + \gamma \cdot \sigma_k(x)\right)
$$

The induced **Credal Set** $\mathcal{Q}(x)$ is the convex set of all valid probability distributions lying inside those bounds:

$$
\mathcal{Q}(x) = \left\{ p \in \Delta^{K-1} : \underline{p}_k(x) \le p_k \le \overline{p}_k(x) \; \forall k \in \{1, \dots, K\} \right\}
$$

### Stage 2: Credal Imprecision & Conservative Conformal Score
1. **Credal Interval Width (Epistemic Imprecision)**:
$$
U_{\text{credal}}(x) = \frac{1}{K} \sum_{k=1}^K \left( \overline{p}_k(x) - \underline{p}_k(x) \right)
$$
2. **Conservative Non-Conformity Score**: Uses the worst-case lower bound $\underline{p}_y(x)$ to guarantee safe coverage:
$$
s_{\text{CREDIT}}(x, y) = 1 - \underline{p}_y(x)
$$

## 3. Worked Numerical Example
Suppose two pixels both have mean probability $\bar{p}_{\text{Forest}} = 0.65$ (with $\gamma = 2$):
- **Pixel A (Dense Forest Interior)**: Low dispersion $\sigma = 0.02 \implies [\underline{p}, \overline{p}] = [0.61, 0.69]$ (width $= 0.08$), so $s_{\text{CREDIT}} = 1 - 0.61 = 0.39$.
- **Pixel B (Noisy Shadow Edge)**: High dispersion $\sigma = 0.18 \implies [\underline{p}, \overline{p}] = [0.29, 1.00]$ (width $= 0.71$), so $s_{\text{CREDIT}} = 1 - 0.29 = 0.71$, automatically expanding the conformal prediction set on the ambiguous edge!

## 4. Key Results in `Classification/credit/results/`
- **Test Accuracy**: `0.9935` | **Test NLL**: `0.0182` | **Test ECE**: `0.0052` | **Conformal Coverage**: `94.75%` (`credit_evaluation_summary.csv`).
