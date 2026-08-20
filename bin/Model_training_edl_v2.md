# Multispectral Patch-Based Classification Pipeline with Evidential Deep Learning (EDL v2)

> **One-line description:** A comparative deep-learning framework that trains three architectures (AlexNet-CNN, GFNet, ViT-UNet) using Evidential Deep Learning (EDL) to predict Dirichlet concentration parameters instead of standard softmax probabilities, enabling rigorous uncertainty quantification.

---

## 1. Overview & Intuition

Unlike standard classifiers that output a single probability distribution over classes via a softmax function, Evidential Deep Learning (EDL) places a Dirichlet distribution over the class probability simplex. This allows the model to distinguish between *data uncertainty* (aleatoric) and *model uncertainty* (epistemic). 

This pipeline adapts the baseline patch-based multispectral classification approach. Instead of softmax, the final layer uses an `evidence_activation` function to output non-negative evidence for each class. These evidence values parameterize a Dirichlet distribution. The loss function is modified to an expected cross-entropy combined with a properly bounded KL divergence term that penalizes the model for accumulating evidence on incorrect classes. This "v2" formulation addresses earlier collapse issues by using a proper Dirichlet KL divergence and a damping factor (`KL_LAMBDA`).

---

## 2. Mathematical Framework

### 2.1 Problem Setup
Let the multispectral scene be $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$. Patches of $9 \times 9 \times 6$ are extracted. The model outputs evidence vector $\mathbf{e}^{(i)} \geq 0$ for each patch.

### 2.2 Evidential Activation and Dirichlet Distribution
Instead of a softmax, the network outputs logits $z_k$, which are transformed into evidence $e_k$ using a softplus activation:
$$e_k = \text{softplus}(z_k) = \log(1 + \exp(z_k))$$
The Dirichlet concentration parameters are $\alpha_k = e_k + 1$. The total evidence is $S = \sum_{k=1}^K \alpha_k$. The expected probability for class $k$ is $\hat{p}_k = \alpha_k / S$.

### 2.3 EDL Loss Function
The loss consists of two parts: Expected Cross-Entropy (ECE) and a proper KL divergence penalty.
$$\mathcal{L}_{CE} = \sum_{k=1}^K y_k (\log(S) - \log(\alpha_k))$$
To regularize, the KL divergence between the predicted Dirichlet distribution (with target class evidence removed) and a uniform Dirichlet distribution is added:
$$\mathcal{L}_{KL} = \text{KL}(\text{Dir}(\tilde{\boldsymbol{\alpha}}) || \text{Dir}(\mathbf{1}))$$
where $\tilde{\alpha}_k = y_k + (1 - y_k) \alpha_k$.
The total loss is:
$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_{KL} \cdot \eta(\text{epoch}) \cdot \mathcal{L}_{KL}$$
where $\eta(\text{epoch})$ is an annealing coefficient that grows from 0 to 1 over the first half of training, and $\lambda_{KL} = 0.1$ is a damping factor to prevent evidence collapse.

### 2.4 The Three Architecture Families
The architectures (AlexNet, GFNet, ViT-UNet) remain identical to the baseline in their feature extraction stages. The only change is the final dense layer, which replaces `softmax` with the custom `evidence_activation`.

---

## 3. Algorithm

1. **Preprocess:** Min-max normalize 6-band data, extract $9 \times 9$ patches.
2. **Build Model:** Instantiate AlexNet, GFNet, or ViT-UNet with `evidence_activation`.
3. **Compile:** Use `make_edl_loss` combining expected cross-entropy and annealed KL divergence.
4. **Train:** Train for 100 epochs. Pass an `EDLAnnealer` callback to update the epoch variable for KL annealing.
5. **Evaluate:** Convert predicted $\alpha$ to probabilities $\hat{p} = \alpha / S$. Calculate Brier score, ECE, and macro/weighted F1.
6. **Dense Scene Inference:** Slide the window across the full scene to produce Dirichlet parameters for every pixel.

---

## 4. Implementation Walkthrough

### 4.1 Evidential Activation
```python
@tf.keras.utils.register_keras_serializable()
def evidence_activation(logits):
    return tf.math.softplus(logits) + 1.0
```
This guarantees non-negative evidence. Adding 1 gives the Dirichlet $\alpha$ parameter.

### 4.2 EDL Loss with Proper KL Divergence
```python
def make_edl_loss(epoch_variable, total_epochs, num_classes, sparse=False, kl_lambda=0.1):
    # ...
    # alpha_tilde 'removes' the true class's evidence
    alpha_tilde = y_true_oh + (1.0 - y_true_oh) * alpha
    # Computes proper KL divergence bounded by kl_lambda and annealing_coef
```
The annealing coefficient $\eta$ ensures the KL penalty does not dominate early training.

---

## 5. References
[1] Sensoy, M., Kaplan, L., & Kandemir, M. (2018). "Evidential Deep Learning to Quantify Classification Uncertainty." *NeurIPS*.
