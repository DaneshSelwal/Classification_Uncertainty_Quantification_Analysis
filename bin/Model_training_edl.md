# Multispectral Patch-Based Classification Pipeline: Evidential Deep Learning (EDL)

> **One-line description:** A comparative deep-learning framework that trains three architectures — AlexNet-CNN, Global Filter Network (GFNet), and a ViT with U-Net skip connections — using Evidential Deep Learning (EDL) to produce Dirichlet concentration parameters for pixel-wise land-cover classification and uncertainty quantification.

---

## 1. Overview & Intuition

This pipeline adapts standard patch-based classification to incorporate Evidential Deep Learning (EDL). While standard models output a softmax probability distribution that can often be overconfident, EDL models are trained to output the parameters of a Dirichlet distribution over the class probability simplex. This allows the model to inherently quantify its uncertainty by associating high "evidence" with confident predictions and low evidence with uncertain ones.

The same three architectures from the baseline are used: **AlexNet**, **GFNet**, and **ViT-UNet**. However, instead of a final softmax layer, the network produces raw logits which are passed through a `softplus(logits) + 1` activation to compute the Dirichlet concentration parameters, $\alpha$.

---

## 2. Mathematical Framework

### 2.1 Problem Setup

Let the multispectral scene be represented as a tensor $\mathcal{X} \in \mathbb{R}^{H \times W \times B}$, where $H = 330$, $W = 307$, and $B = 6$ (spectral bands). Each band is normalised independently to $[0, 1]$ via min-max scaling.

### 2.2 Evidential Output and Dirichlet Distribution

Instead of a standard softmax probability vector $\hat{\mathbf{p}}$, the model outputs a vector of Dirichlet concentration parameters $\boldsymbol{\alpha} = [\alpha_1, \ldots, \alpha_K] \in \mathbb{R}^K$. These parameters represent the "evidence" collected for each class, where $\alpha_k \geq 1$.

The activation function applied to the final dense layer logits is:
$$ \alpha_k = \text{softplus}(z_k) + 1 = \ln(1 + e^{z_k}) + 1 $$

The expected probability for class $k$ is given by:
$$ \hat{p}_k = \frac{\alpha_k}{S} \quad \text{where} \quad S = \sum_{j=1}^K \alpha_j $$
Here, $S$ is the total Dirichlet strength (or total evidence). Higher values of $S$ indicate higher overall confidence.

### 2.3 The Evidential Loss Function

The models are trained using a specialized EDL loss function, which consists of two main components:
1. **Expected Cross-Entropy:** Pushes the $\alpha$ of the true class upward.
2. **KL-Divergence Regularizer:** Penalizes the accumulation of evidence on incorrect classes.

Given a one-hot target vector $\mathbf{y}$, the loss is computed as:
$$ \mathcal{L} = \sum_{k=1}^K y_k \left( \ln(S) - \ln(\alpha_k) \right) + \lambda_t \text{KL}\left[ \text{Dir}(\boldsymbol{\tilde{\alpha}}) \,\|\, \text{Dir}(\mathbf{1}) \right] $$

Where:
- $\boldsymbol{\tilde{\alpha}} = \mathbf{y} + (1 - \mathbf{y}) \odot \boldsymbol{\alpha}$ sets the true class evidence parameter to 1, ensuring only non-target evidence is penalized.
- $\lambda_t = \min\left(1.0, \frac{t}{T/2}\right)$ is an epoch-based annealing coefficient that gradually introduces the KL penalty during training, where $t$ is the current epoch and $T$ is the total number of epochs.

---

## 3. Implementation Walkthrough

The notebook `Model_training_edl.ipynb` builds heavily on the baseline architecture with key modifications for the EDL formulation:

### 3.1 Evidential Output Activation
A custom Keras layer replaces the softmax activation:
```python
@tf.keras.utils.register_keras_serializable()
def evidence_activation(logits):
    return tf.math.softplus(logits) + 1.0
```

### 3.2 The EDL Loss Closure
The loss function requires tracking the current epoch to compute the KL annealing coefficient. This is achieved using a custom `EDLAnnealer` callback that updates a `tf.Variable`:
```python
class EDLAnnealer(keras.callbacks.Callback):
    def __init__(self, epoch_variable):
        super().__init__()
        self.epoch_variable = epoch_variable

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_variable.assign(float(epoch))
```
The loss function closure accesses this variable to compute $\lambda_t$ dynamically.

### 3.3 Probability Extraction
For downstream evaluation metrics like Expected Calibration Error (ECE) and Brier Score, the Dirichlet parameters $\boldsymbol{\alpha}$ are converted to expected probabilities before scoring:
```python
def alpha_to_probs(alpha):
    return alpha / alpha.sum(axis=-1, keepdims=True)
```
