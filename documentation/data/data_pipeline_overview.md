# Data Pipeline & Preprocessing Overview (`Classification/data/`)

## 1. Plain-English Overview
Before running any classifier or uncertainty algorithm, raw hyperspectral and multispectral satellite cubes (Height $\times$ Width $\times$ Bands) are converted into standardized feature tables (`data.csv`) and reference ground-truth label maps (`ref.csv`). Across the repository, experiments evaluate four benchmark datasets: **Indian Pines**, **Pavia University**, **Hisar-3 (3m)**, and **Hissar-25**.

## 2. Stage-by-Stage Process & Math
### Stage 1: Per-Band Standardization
For each pixel $i$ and spectral band $b \in \{1, \dots, B\}$, reflectance values are normalized using the training split mean $\mu_b$ and standard deviation $\sigma_b$:

$$
\tilde{x}_{i,b} = \frac{x_{i,b} - \mu_b}{\sigma_b + \epsilon}
$$

Computing $\mu_b$ and $\sigma_b$ strictly on the training set prevents data leakage into the calibration and test splits.

### Stage 2: The 3-Way Stratified Split (70% / 15% / 15%)
Why do we need **three** splits instead of just Train and Test? Because **Conformal Prediction** and **Temperature Scaling** require a held-out **Calibration Set ($D_{\text{calib}}$)** that the neural network has never seen during backpropagation:
- **Training Split (70%, $N_{\text{train}}$)**: Updates neural network weights $\theta$.
- **Validation / Calibration Split (15%, $N_{\text{val}}$)**: Used for early stopping, learning temperature $T^*$, and computing the conformal non-conformity quantile $\hat{q}_{1-\alpha}$.
- **Test Split (15%, $N_{\text{test}}$)**: Held out to evaluate final Test Accuracy, ECE, Brier Score, NLL, Empirical Coverage, and Average Prediction Set Size.
