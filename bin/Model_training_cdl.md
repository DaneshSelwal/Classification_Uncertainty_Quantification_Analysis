# Multispectral Patch-Based Classification Pipeline: Conformalized Deep Learning (CDL)

> **One-line description:** A classification pipeline that trains AlexNet-CNN, GFNet, and ViT-UNet on spatially extracted multispectral patches using Conformalized Deep Learning (CDL) loss to produce natively uncertainty-aware probability distributions.

---

## 1. Overview & Intuition

Traditional neural network classifiers trained with standard cross-entropy tend to be overconfident and poorly calibrated. The **Conformalized Deep Learning (CDL)** method modifies the training objective to produce models whose outputs inherently align with the properties required for conformal prediction. 

This notebook processes 9 × 9 × 6 multispectral patches and trains three diverse architectures: **AlexNet** (a local CNN), **GFNet** (global frequency-domain filtering), and **ViT-UNet** (transformer with multi-scale skip connections). However, instead of standard cross-entropy, these models are trained with a composite CDL loss. CDL introduces a differentiable conformity score (a soft Adaptive Prediction Set metric) and penalises the network if the order statistics of these scores deviate from a uniform distribution, naturally encouraging the network to output probabilities that provide reliable distribution-free coverage.

---

## 2. Mathematical Framework

### 2.1 Problem Setup and Patch Extraction
Let the multispectral scene be $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$. As in the baseline approach, patches $\mathbf{X}^{(i)} \in \mathbb{R}^{P \times P \times B}$ are extracted around each labelled pixel after per-band min-max normalisation. 

### 2.2 Differentiable Conformity Scores (Soft APS)
The standard APS conformity score sums the probabilities of all classes more likely than the true class. To incorporate this into a neural network loss, CDL makes this thresholding operation differentiable using a scaled sigmoid function:
$$ s_i = \sum_{c=1}^K p_c \cdot \sigma\left(\frac{p_c - p_{y_i}}{\tau}\right) \cdot (1 - \mathbf{1}[c = y_i]) + u_i \cdot p_{y_i} $$
**Where:**
- $p_c$ is the predicted probability for class $c$.
- $p_{y_i}$ is the predicted probability for the true class.
- $\sigma(\cdot)$ is the sigmoid function.
- $\tau = 0.05$ is the temperature/softness parameter.
- $u_i \sim \text{Uniform}(0,1)$ is a random tie-breaker.

### 2.3 Uniform Matching Loss
Conformal prediction theory indicates that well-calibrated conformity scores should follow a uniform distribution. The uniform matching loss computes the Mean Squared Error (MSE) between the sorted scores in a mini-batch and theoretical uniform quantiles:
$$ \mathcal{L}_{\text{matching}} = \frac{1}{N} \sum_{j=1}^{N} \left( s_{(j)} - \frac{j}{N+1} \right)^2 $$
**Where:**
- $s_{(j)}$ is the $j$-th smallest conformity score in the batch of size $N$.

### 2.4 Total CDL Objective
The final training loss is a convex combination of standard Cross-Entropy (CE) and the uniform matching loss:
$$ \mathcal{L}_{\text{CDL}} = \mathcal{L}_{\text{CE}} + \mu \cdot \mathcal{L}_{\text{matching}} $$
Where $\mu = 0.1$ controls the regularisation strength.

---

## 3. Algorithm

1. **Pre-process Data**: Normalise 6-band multispectral data per band to $[0, 1]$.
2. **Patch Extraction**: Extract 9 × 9 patches around every labelled pixel.
3. **Data Splits**: Create Train/Val/Test splits. (AlexNet uses a legacy random split without a validation set).
4. **Model Instantiation**: Build AlexNet, GFNet, or ViT-UNet.
5. **Compile**: 
   - Compile models using the custom `categorical_cdl_loss` (or sparse equivalent for AlexNet).
   - Optimizers: Adagrad (AlexNet) or AdamW with Cosine Decay (GFNet, ViT-UNet).
6. **Training Loop**: Train for 100 epochs. Automatically fallback to smaller architectural parameters (`capacity_tag="fallback"`) if `ResourceExhaustedError` (OOM) occurs.
7. **Inference & Evaluation**: Compute test set accuracy, F1 scores, ECE, and Multiclass Brier scores. 
8. **Dense Inference**: Predict the entire $H \times W$ scene row-by-row and save classification artifacts.

---

## 4. Implementation Walkthrough

- **`aps_scores_soft`**: Calculates the differentiable APS nonconformity score utilizing `tf.sigmoid` for the pairwise class comparisons.
- **`uniform_matching_loss`**: Uses `tf.sort(scores)` to rank the batch's conformity scores, generating an exact gradient, and compares it to `tf.range(1, N+1) / (N+1)`.
- **`categorical_cdl_loss`**: A closure returning a Keras-compatible loss function that sums standard Cross Entropy and $\mu \times$ `uniform_matching_loss`.
