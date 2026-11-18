# Credal Deep Ensemble (CreDE) Uncertainty Quantification: Theory & Implementation Summary

> **One-line description:** CreDE turns a homogeneous deep ensemble's disagreeing softmax outputs into a *credal set* (a per-class probability interval) for every pixel, then reads off aleatoric, epistemic, and total uncertainty directly from that interval's shape and width.

---

## 1. Overview & Intuition

The notebook performs pixel-wise classification of a six-band multispectral scene (330 × 307 pixels) using three different deep architectures — a CNN (`AlexNet_CNN`), a frequency-domain transformer (`GFNet`), and a hybrid Vision-Transformer/U-Net (`ViT_UNet`) — and, for each architecture, an ensemble of several independently trained checkpoints. The goal is not just to classify each pixel but to say *how much* to trust that classification, and *why* it might be untrustworthy: because the spectral signature genuinely sits between two land-cover classes (irreducible, aleatoric uncertainty), or because the ensemble members themselves cannot agree on an answer (reducible, epistemic uncertainty, often a sign of an out-of-distribution or under-represented pixel).

A standard softmax classifier cannot make this distinction: it returns one probability vector, and a flat vector could mean either "this pixel is inherently ambiguous" or "this model has no idea." Standard Deep Ensembles improve on this by averaging several independently trained networks and splitting the entropy of the average into an aleatoric term (the average of each member's entropy) and an epistemic term (the entropy of the average minus that average), but empirical studies have shown this decomposition can still under-represent epistemic uncertainty, since it ultimately collapses the ensemble back into a single distribution.

Credal Deep Ensembles, introduced by Wang et al. (NeurIPS 2024), address this by never collapsing the ensemble into a single distribution at all. Instead, each class is given a *lower* and an *upper* probability bound, jointly describing a convex set of admissible distributions — a credal set. In the original paper this credal set is produced by training special networks ("CreNets") with a custom two-part loss (an optimistic cross-entropy term for the upper bound, and a distributionally-robust, pessimistic term for the lower bound) and a custom "Interval Softmax" output layer. The width of the resulting interval is, by construction, a direct measure of epistemic uncertainty.

This notebook implements a lighter, **training-free variant** of that idea. Rather than training new interval-output networks, it takes an already-trained homogeneous deep ensemble (several independently trained copies of the *same* architecture) and forms the credal set directly from the **empirical envelope** of the members' ordinary softmax outputs: for each class, the lower bound is simply the smallest probability any ensemble member assigned to it, and the upper bound is the largest. This sidesteps the custom training loop and Interval-Softmax layer entirely, while still producing a valid, non-empty credal set per pixel (a property proven below) from which aleatoric, epistemic, and total uncertainty can be read off cheaply at inference time. The trade-off is that this envelope is a coarser, more heuristic stand-in for the formally-trained, distributionally-robust intervals of the original CreDE paper — it is well suited to retrofitting uncertainty quantification onto an existing ensemble of trained models, as is done here for a remote-sensing scene with three different backbone architectures.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Each pixel of the scene is represented by a spatial-spectral patch $x \in \mathbb{R}^{9\times9\times6}$ (a $9\times9$ neighbourhood across 6 spectral bands). For a chosen architecture, an ensemble of $M$ independently trained networks $f_1,\dots,f_M$ (identical architecture, different training seeds/checkpoints) is available. Each network outputs an ordinary softmax probability vector over $C$ classes,

$$q_m(x) = (q_{m,1}(x), \dots, q_{m,C}(x)) \in \Delta^{C-1}, \qquad m = 1,\dots,M,$$

where $\Delta^{C-1}$ is the probability simplex. There are $N = H\times W$ pixels in the scene; the notebook runs this for every pixel to build full uncertainty maps.

### 2.2 Credal Bounds via the Ensemble Envelope

For each class $i$, CreDE (as implemented here) defines the **lower** and **upper** probability bound as the minimum and maximum value assigned to that class across all ensemble members:

$$\underline{p}_i(x) = \min_{m=1,\dots,M} q_{m,i}(x), \qquad \overline{p}_i(x) = \max_{m=1,\dots,M} q_{m,i}(x).$$

**Where:**
- $\underline{p}_i(x)$ — lower probability bound for class $i$ at input $x$ (worst-case, most pessimistic estimate any member gave that class).
- $\overline{p}_i(x)$ — upper probability bound for class $i$ (best-case, most optimistic estimate).
- The interval $[\underline{p}_i(x), \overline{p}_i(x)]$, taken jointly over all $C$ classes, defines a credal set $Q(x) = \{q \in \Delta^{C-1} : \underline{p}_i(x) \le q_i \le \overline{p}_i(x) \,\, \forall i\}$ — the convex set of all probability distributions consistent with every member's per-class bound.

**What this means:** instead of forcing all $M$ opinions into one averaged distribution, every class keeps the full *range* of opinions the ensemble expressed about it. A class with a wide range $[\underline{p}_i,\overline{p}_i]$ is one the ensemble members strongly disagree about.

A useful property (parallel to the well-formedness condition required by the original CreDE paper) is that this envelope automatically yields a non-empty credal set, since

$$\sum_{i=1}^{C}\underline{p}_i(x) = \sum_{i=1}^{C}\min_m q_{m,i}(x) \;\le\; \sum_{i=1}^{C} q_{m_0,i}(x) = 1 \;\le\; \sum_{i=1}^{C}\max_m q_{m,i}(x) = \sum_{i=1}^{C}\overline{p}_i(x)$$

for any fixed member $m_0$ — i.e. the lower bounds can never over-shoot a valid distribution, and the upper bounds can never under-shoot one, so the credal set is guaranteed never to collapse to nothing.

### 2.3 The Interval Width and the Normalized Lower Distribution

Two quantities derived from the bounds drive everything downstream:

$$\Delta p_i(x) = \overline{p}_i(x) - \underline{p}_i(x), \qquad p^{*}_i(x) = \frac{\underline{p}_i(x)}{\sum_{j=1}^{C}\underline{p}_j(x) + \varepsilon}.$$

**Where:**
- $\Delta p_i(x)$ — the width of the credal interval for class $i$; large width means the ensemble strongly disagrees about that class.
- $p^{*}(x)$ — the lower-bound vector $\underline{p}(x)$ renormalized to sum to 1 (with a small $\varepsilon=10^{-12}$ for numerical safety, and values clipped to $[\varepsilon, 1]$ afterward). Because $\sum_i \underline{p}_i \le 1$ in general, $\underline{p}$ itself is not a valid distribution; $p^{*}$ is the closest "pessimistic" distribution that *is* valid, and acts as the single point estimate the rest of the pipeline reasons about.

**What this means:** $p^{*}$ is a deliberately conservative ("worst-case") representative of the credal set, while $\Delta p$ measures, class by class, how wide the disagreement is around that representative.

### 2.4 Uncertainty Decomposition: Aleatoric, Epistemic, Total

Given $p^{*}(x)$ and $\Delta p(x)$, the notebook defines three per-pixel uncertainty scores:

$$\mathrm{AU}(x) = -\sum_{i=1}^{C} p^{*}_i(x)\,\ln p^{*}_i(x), \qquad \mathrm{EU}(x) = \frac{1}{C}\sum_{i=1}^{C}\Delta p_i(x), \qquad \mathrm{TU}(x) = \mathrm{AU}(x) + \mathrm{EU}(x).$$

**Where:**
- $\mathrm{AU}(x)$ — aleatoric uncertainty: the (natural-log, "nats") Shannon entropy of the conservative distribution $p^{*}$. Because $p^{*}$ is computed once the ensemble's disagreement has already been absorbed into the bounds, a high $\mathrm{AU}$ reflects genuine class overlap (several classes plausible) rather than model disagreement.
- $\mathrm{EU}(x)$ — epistemic uncertainty: the average per-class interval width. This is a simple, cheap stand-in for the more rigorous *generalized-entropy* or *generalized-Hartley* epistemic measures used in the original CreDE paper (which require a constrained optimization to compute), trading some theoretical sharpness for a closed-form, instantaneous computation suitable for scoring every pixel of a large scene.
- $\mathrm{TU}(x)$ — total uncertainty, taken here simply as the additive combination $\mathrm{AU}+\mathrm{EU}$ (again a practical simplification of the formal upper-entropy quantity $\overline{H}(Q)$ used in the source paper).

**What this means:** $\mathrm{AU}$ answers "is this pixel inherently ambiguous?", $\mathrm{EU}$ answers "do the ensemble members disagree about it?", and $\mathrm{TU}$ combines both into a single "how much should I distrust this prediction?" score. Because the two sources are kept separate, a region with high $\mathrm{EU}$ but low $\mathrm{AU}$ flags model ignorance (worth more training data), whereas high $\mathrm{AU}$ with low $\mathrm{EU}$ flags a class boundary the model has already learned reliably but that is intrinsically fuzzy in spectral space.

The predicted class is the arg-max of the conservative distribution,

$$\hat{c}(x) = \arg\max_{i} p^{*}_i(x),$$

mirroring the "maximin" decision rule of the original CreDE paper (which picks the class with the highest *reachable* lower probability), but applied here to the normalized $p^{*}$ rather than the raw, unnormalized lower bounds.

### 2.5 Absolute-Threshold Masking for Spatial Visualization

To turn the continuous $\mathrm{AU}$, $\mathrm{EU}$, $\mathrm{TU}$ maps into binary "certain vs. uncertain" maps for visualization and reporting, the notebook applies fixed, dataset-specific thresholds:

$$\text{mask}_{\mathrm{AU}}(x) = \mathbb{1}[\mathrm{AU}(x) > \tau_{\mathrm{AU}}], \quad \text{mask}_{\mathrm{EU}}(x) = \mathbb{1}[\mathrm{EU}(x) > \tau_{\mathrm{EU}}], \quad \text{mask}_{\mathrm{TU}}(x) = \mathbb{1}[\mathrm{TU}(x) > \tau_{\mathrm{TU}}],$$

with $\tau_{\mathrm{AU}} = 0.5$, $\tau_{\mathrm{EU}} = 0.2$, $\tau_{\mathrm{TU}} = 0.7$ in this notebook's configuration. **What this means:** a pixel can be flagged "uncertain" by any of the three masks independently, which is exactly what makes it possible to visually separate aleatoric-driven uncertainty regions from epistemic-driven ones on the scene map.

---

## 3. Algorithm

**Input:** for a chosen architecture, $M$ trained `.keras` checkpoint paths; the full-scene patch tensor $X \in \mathbb{R}^{N\times 9\times9\times6}$ (and, separately, fixed thresholds $\tau_{\mathrm{AU}}, \tau_{\mathrm{EU}}, \tau_{\mathrm{TU}}$).
**Output:** per-pixel predicted class map, conservative distribution $p^{*}$, and the three uncertainty maps $\mathrm{AU}, \mathrm{EU}, \mathrm{TU}$, plus visualizations and a summary table.

1. For each of the $M$ checkpoints: load the model (resolving its custom layers), run a forward pass over the entire scene in batches to obtain a softmax prediction matrix $q_m \in \mathbb{R}^{N\times C}$, then discard the model and free memory before loading the next checkpoint.
2. Stack the $M$ prediction matrices into a tensor of shape $(M, N, C)$.
3. Take the elementwise minimum and maximum over the ensemble axis to obtain $\underline{p}, \overline{p} \in \mathbb{R}^{N\times C}$.
4. Compute the interval width $\Delta p = \overline{p}-\underline{p}$.
5. Normalize $\underline{p}$ per pixel (with epsilon-clipping) to obtain the conservative distribution $p^{*}$.
6. Compute $\mathrm{AU} = -\sum_i p^{*}_i\ln p^{*}_i$, $\mathrm{EU} = \mathrm{mean}_i(\Delta p_i)$, and $\mathrm{TU}=\mathrm{AU}+\mathrm{EU}$ for every pixel.
7. Predict the class map as $\hat c = \arg\max_i p^{*}_i$.
8. Reshape $\hat c$, $\mathrm{AU}$, $\mathrm{EU}$, $\mathrm{TU}$ back to the $(H,W)$ scene grid; threshold each uncertainty map at its configured value to obtain a binary certain/uncertain mask; render a standardized multi-panel figure (base prediction map, the three binary masks, three grey-overlay maps, and three pixel-count bar charts) and save it.
9. Repeat steps 1–8 independently for every target architecture; accumulate each architecture's mean $\mathrm{AU}$, $\mathrm{EU}$, $\mathrm{TU}$ and per-class pixel counts into a master table, then export the table (and the saved figures) as CSV and a formatted Excel report.

---

## 4. Implementation Walkthrough

> Based on the uploaded notebook: `Model_uncertainty_CreDE.ipynb`

### 4.1 Custom Layer Registry (needed to load the saved ensembles)

```python
CUSTOM_OBJECTS = {
    "PatchExtractor":      PatchExtractor,
    "PatchPositionEncoder": PatchPositionEncoder,
    "GlobalFilterLayer":   GlobalFilterLayer,
    "PatchEncoderWithCLS": PatchEncoderWithCLS,
}
```
**What this does:** registers the four custom Keras layers (a non-overlapping patch extractor, a learned positional encoder, an FFT-based global filter used by the `GFNet` backbone, and a CLS-token encoder used by `ViT_UNet`) so that `tf.keras.models.load_model` can correctly deserialize each saved ensemble checkpoint.
**Why:** CreDE itself is architecture-agnostic — it only needs each member's softmax output — but the ensemble members here are saved as full Keras models with these bespoke layers, so they must be resolvable before any predictions can be obtained.

### 4.2 Locating Ensemble Checkpoints

```python
def get_ensemble_paths(model_name):
    primary_pattern  = str(MODEL_DIR / f"{model_name}_ens_*_final.keras")
    paths = glob.glob(primary_pattern)
    if not paths:
        fallback_pattern = str(MODEL_DIR / "ensembles_old" / f"{model_name}_ens_*_final.keras")
        paths = glob.glob(fallback_pattern)
    return paths
```
**What this does:** for a given architecture name, finds every saved checkpoint belonging to its homogeneous ensemble (falling back to a legacy folder if none are found in the primary one).
**Why:** CreDE needs the *full list* of independently trained members $f_1,\dots,f_M$ for that architecture — this is the step that assembles them.

### 4.3 The Core CreDE Computation

```python
stacked_preds = tf.stack(all_preds, axis=0)          # (M, N, C)
p_min = tf.reduce_min(stacked_preds, axis=0)
p_max = tf.reduce_max(stacked_preds, axis=0)
delta_p = p_max - p_min

p_star = p_min / (tf.reduce_sum(p_min, axis=-1, keepdims=True) + 1e-12)
p_star = np.clip(p_star, 1e-12, 1.0)

au = -np.sum(p_star * np.log(p_star), axis=-1)   # aleatoric: entropy
eu = np.mean(delta_p, axis=-1)                    # epistemic: mean credal spread
tu = au + eu                                      # total uncertainty
pred_class = np.argmax(p_star, axis=-1)
```
**What this does:** this is the entire CreDE inference step described in Sections 2.2–2.4, written for the whole scene at once. `all_preds` (built by looping over the $M$ checkpoints, predicting, then immediately deleting the model and clearing the TensorFlow session/garbage-collecting to keep memory bounded) is stacked into the $(M,N,C)$ tensor; min/max across axis 0 gives the credal bounds; normalizing the lower bound gives $p^{*}$; and the three uncertainty scores and the predicted class follow directly from the equations above.
**Why:** this single block is what makes the method "CreDE" rather than a plain Deep Ensemble — the min/max envelope replaces the usual "average the softmax outputs" step, preserving disagreement information that averaging would have erased.

### 4.4 Spatial Uncertainty Maps

```python
au_mask = (au_map > au_thresh).astype(int)
eu_mask = (eu_map > eu_thresh).astype(int)
tu_mask = (tu_map > tu_thresh).astype(int)
combined_au = np.where(au_mask == 1, n_cls, pred_map)
...
fig, axes = plt.subplots(3, 4, figsize=(38, 26))
```
**What this does:** thresholds each uncertainty map into a binary certain/uncertain mask (Section 2.5), overlays the "uncertain" pixels in grey on top of the predicted class map, and assembles a 3×4 figure per architecture (prediction map, the three binary masks, the three grey overlays, and three pixel-count bar charts), saved as a PNG.
**Why:** this turns the abstract per-pixel $\mathrm{AU}/\mathrm{EU}/\mathrm{TU}$ scores into an interpretable spatial product — e.g. revealing whether uncertainty concentrates at class boundaries (aleatoric) or in specific spatial clusters the ensemble disagrees about (epistemic).

### 4.5 Master Evaluation Loop

```python
for model_name in architectures:
    ensemble_paths = get_ensemble_paths(model_name)
    pred_class, p_star, au, eu, tu = evaluate_homogeneous_ensemble(
        ensemble_paths, scene_pixels_scaled, batch_size=2048)
    saved_plot_path = generate_spatial_crede_maps(model_name, pred_class, p_star, au, eu, tu, ...)
    master_results.append({"Model": f"{model_name}_CreDE", "Mean_AU": float(np.mean(au)), ...})
```
**What this does:** repeats the whole CreDE pipeline independently for `AlexNet_CNN`, `GFNet`, and `ViT_UNet`, aggressively freeing memory between architectures, and accumulates each architecture's mean uncertainty scores and per-class pixel counts into a summary table that is exported to CSV and an Excel workbook (with the saved figures embedded) — the styling and `openpyxl` export code itself is a reporting convenience and not part of the CreDE method.
**Why:** lets the three backbones be compared on equal footing using the same CreDE uncertainty pipeline, since CreDE places no requirement on the underlying architecture beyond it producing a softmax output.

---

## 5. Worked Numerical Example

### 5.0 Shared Setup

Since CreDE (as implemented here) is not a calibration-based method, no separate calibration set is required — the same ensemble outputs both the prediction and the uncertainty for each pixel directly. The toy example below uses:

- $C=4$ classes: $C_0, C_1, C_2, C_3$.
- $M=3$ ensemble members, mimicking a small homogeneous ensemble of, say, `AlexNet_CNN` retrained from three different seeds.
- The notebook's actual thresholds: $\tau_{\mathrm{AU}}=0.5$, $\tau_{\mathrm{EU}}=0.2$, $\tau_{\mathrm{TU}}=0.7$.
- Three test pixels: an **easy** one (clear majority class, members agree), a **borderline** one (members disagree about which of two classes is on top, but stay close to each other in absolute terms), and an **ambiguous** one (no member is confident, and members substantially disagree).

**Easy pixel — member softmax outputs (true class bolded):**

| Member | $C_0$ | $C_1$ | $C_2$ | $C_3$ |
|---|---|---|---|---|
| $f_1$ | **0.93** | 0.03 | 0.02 | 0.02 |
| $f_2$ | **0.90** | 0.04 | 0.03 | 0.03 |
| $f_3$ | **0.94** | 0.02 | 0.02 | 0.02 |

**Borderline pixel — member softmax outputs (true class bolded):**

| Member | $C_0$ | $C_1$ | $C_2$ | $C_3$ |
|---|---|---|---|---|
| $f_1$ | 0.30 | **0.45** | 0.15 | 0.10 |
| $f_2$ | 0.42 | **0.33** | 0.15 | 0.10 |
| $f_3$ | 0.28 | **0.40** | 0.20 | 0.12 |

**Ambiguous pixel — member softmax outputs (true class bolded):**

| Member | $C_0$ | $C_1$ | $C_2$ | $C_3$ |
|---|---|---|---|---|
| $f_1$ | 0.40 | 0.15 | **0.25** | 0.20 |
| $f_2$ | 0.10 | 0.40 | **0.30** | 0.20 |
| $f_3$ | 0.12 | 0.10 | **0.40** | 0.38 |

---

### 5.1 Step A — Credal Bounds From the Ensemble

For each pixel, take the per-class minimum and maximum across the three members.

**Easy pixel:**

| Class | $\underline{p}_i$ | $\overline{p}_i$ | $\Delta p_i$ |
|---|---|---|---|
| $C_0$ | 0.90 | 0.94 | 0.04 |
| $C_1$ | 0.02 | 0.04 | 0.02 |
| $C_2$ | 0.02 | 0.03 | 0.01 |
| $C_3$ | 0.02 | 0.03 | 0.01 |
| **Sum** | **0.96** | **1.04** | — |

**Borderline pixel:**

| Class | $\underline{p}_i$ | $\overline{p}_i$ | $\Delta p_i$ |
|---|---|---|---|
| $C_0$ | 0.28 | 0.42 | 0.14 |
| $C_1$ | 0.33 | 0.45 | 0.12 |
| $C_2$ | 0.15 | 0.20 | 0.05 |
| $C_3$ | 0.10 | 0.12 | 0.02 |
| **Sum** | **0.86** | **1.19** | — |

**Ambiguous pixel:**

| Class | $\underline{p}_i$ | $\overline{p}_i$ | $\Delta p_i$ |
|---|---|---|---|
| $C_0$ | 0.10 | 0.40 | 0.30 |
| $C_1$ | 0.10 | 0.40 | 0.30 |
| $C_2$ | 0.25 | 0.40 | 0.15 |
| $C_3$ | 0.20 | 0.38 | 0.18 |
| **Sum** | **0.65** | **1.58** | — |

Note that in every case $\sum_i \underline{p}_i \le 1 \le \sum_i \overline{p}_i$, confirming (as proven in Section 2.2) that each credal set is well-formed and non-empty.

### 5.2 Step B — Normalized Conservative Distribution $p^{*}$

Each $p^{*}_i = \underline{p}_i / \sum_j \underline{p}_j$:

| Pixel | $p^{*}_{C_0}$ | $p^{*}_{C_1}$ | $p^{*}_{C_2}$ | $p^{*}_{C_3}$ |
|---|---|---|---|---|
| Easy | 0.9375 | 0.02083 | 0.02083 | 0.02083 |
| Borderline | 0.32558 | 0.38372 | 0.17442 | 0.11628 |
| Ambiguous | 0.15385 | 0.15385 | 0.38462 | 0.30769 |

For the easy pixel, dividing by $\sum\underline{p}_i = 0.96$ sharpens $C_0$ from 0.90 to 0.9375 while the three minor classes (each $0.02$) become $0.02083$. For the borderline pixel, dividing by $0.86$ raises $C_1$ from $0.33$ to $0.38372$ — now clearly the top class — while $C_0$ rises to $0.32558$, keeping the two classes close. For the ambiguous pixel, dividing by $0.65$ turns the four lower bounds $(0.10,0.10,0.25,0.20)$ into $(0.15385,0.15385,0.38462,0.30769)$, with $C_2$ and $C_3$ now the two leading (but not dominant) classes.

### 5.3 Step C — Uncertainty Decomposition, Prediction, and Masking

For each pixel: $\mathrm{AU}=-\sum_i p^{*}_i\ln p^{*}_i$, $\mathrm{EU}=\mathrm{mean}_i(\Delta p_i)$, $\mathrm{TU}=\mathrm{AU}+\mathrm{EU}$, $\hat c=\arg\max_i p^{*}_i$.

**Easy pixel:**
$$\mathrm{AU} = -\big[0.9375\ln(0.9375) + 3\times 0.02083\ln(0.02083)\big] = -(-0.06051 - 3\times0.08065) = 0.30245$$
$$\mathrm{EU} = \tfrac{1}{4}(0.04+0.02+0.01+0.01) = 0.02000, \qquad \mathrm{TU} = 0.30245+0.02000 = 0.32245$$
$\hat c = C_0$ (true class $C_0$ → **covered ✓**).

| Quantity | Value | Threshold | In set? (uncertain if value > threshold) |
|---|---|---|---|
| AU | 0.302 | 0.5 | No → **certain** |
| EU | 0.020 | 0.2 | No → **certain** |
| TU | 0.322 | 0.7 | No → **certain** |

**Borderline pixel:**
$$\mathrm{AU} = -\big[0.32558\ln(0.32558)+0.38372\ln(0.38372)+0.17442\ln(0.17442)+0.11628\ln(0.11628)\big] = 1.28768$$
$$\mathrm{EU} = \tfrac{1}{4}(0.14+0.12+0.05+0.02) = 0.08250, \qquad \mathrm{TU} = 1.28768 + 0.08250 = 1.37018$$
$\hat c = C_1$ (true class $C_1$ → **covered ✓**), even though the margin over $C_0$ in $p^{*}$ is narrow (0.384 vs. 0.326).

| Quantity | Value | Threshold | In set? |
|---|---|---|---|
| AU | 1.288 | 0.5 | Yes → **uncertain** |
| EU | 0.083 | 0.2 | No → **certain** |
| TU | 1.370 | 0.7 | Yes → **uncertain** |

This pixel is flagged uncertain by AU and TU but *not* by EU: the ensemble members do not strongly disagree on the spread (the interval widths are modest), but the resulting conservative distribution is itself spread across two plausible classes — i.e. the model has converged on a stable but inherently ambiguous (aleatoric) call.

**Ambiguous pixel:**
$$\mathrm{AU} = -\big[2\times0.15385\ln(0.15385) + 0.38462\ln(0.38462) + 0.30769\ln(0.30769)\big] = 1.30611$$
$$\mathrm{EU} = \tfrac{1}{4}(0.30+0.30+0.15+0.18) = 0.23250, \qquad \mathrm{TU} = 1.30611+0.23250 = 1.53861$$
$\hat c = C_2$ (true class $C_2$ → **covered ✓**), despite high uncertainty on all three measures.

| Quantity | Value | Threshold | In set? |
|---|---|---|---|
| AU | 1.306 | 0.5 | Yes → **uncertain** |
| EU | 0.233 | 0.2 | Yes → **uncertain** |
| TU | 1.539 | 0.7 | Yes → **uncertain** |

Here all three masks fire: the conservative distribution is spread across classes (high AU) *and* the ensemble members substantially disagree about which class is on top (high EU) — exactly the kind of pixel a practitioner would want flagged for manual review or for prioritizing additional training data.

### 5.4 Step D — Summary Table

| Pixel | Predicted class | True class | Covered? | AU | EU | TU | au_mask | eu_mask | tu_mask |
|---|---|---|---|---|---|---|---|---|---|
| Easy | $C_0$ | $C_0$ | ✓ | 0.302 | 0.020 | 0.322 | 0 (certain) | 0 (certain) | 0 (certain) |
| Borderline | $C_1$ | $C_1$ | ✓ | 1.288 | 0.083 | 1.370 | 1 (uncertain) | 0 (certain) | 1 (uncertain) |
| Ambiguous | $C_2$ | $C_2$ | ✓ | 1.306 | 0.233 | 1.539 | 1 (uncertain) | 1 (uncertain) | 1 (uncertain) |

All three pixels are correctly classified ($p^{*}$'s arg-max matches the true class in every case), which illustrates a key feature of credal-set uncertainty: a prediction can be *correct* while still being flagged as *uncertain* — the uncertainty maps are diagnostic of confidence, not of correctness, and the borderline/ambiguous cases show how AU and EU can disagree about *why* a pixel is hard, even when the final class call is right.

---

## 6. References

[1] Wang, K., Cuzzolin, F., Manchingal, S. K., Shariatmadar, K., Moens, D., Hallez, H. "Credal Deep Ensembles for Uncertainty Quantification." *Advances in Neural Information Processing Systems (NeurIPS)*, 2024. [Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/911fc798523e7d4c2e9587129fcf88fc-Paper-Conference.pdf)

[2] Wang, K., Shariatmadar, K., Manchingal, S. K., Cuzzolin, F., Moens, D., Hallez, H. "CreINNs: Credal-Set Interval Neural Networks for Uncertainty Estimation in Classification Tasks." *arXiv preprint arXiv:2401.05043*, 2024. [Link](https://arxiv.org/abs/2401.05043)

[3] Hüllermeier, E., Waegeman, W. "Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods." *Machine Learning*, 110(3):457–506, 2021.

[4] Lakshminarayanan, B., Pritzel, A., Blundell, C. "Simple and scalable predictive uncertainty estimation using deep ensembles." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 2017.

[5] Rao, Y., Zhao, W., Zhu, Z., Lu, J., Zhou, J. "Global Filter Networks for Image Classification." *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 2021. (architectural reference for the `GFNet`/`GlobalFilterLayer` backbone used as one of the three ensembles evaluated with CreDE in this notebook.)
