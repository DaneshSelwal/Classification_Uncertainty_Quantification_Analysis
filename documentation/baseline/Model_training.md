# Baseline Model Training Walkthrough (`Classification/baseline/Model_training.ipynb`)

## 1. Plain-English Overview ("The Big Idea")
This is the first notebook in the execution order. It trains the three foundational neural network backbones on the 70% training split (with 15% validation and 15% testing) so we have baseline softmax probability predictions before applying uncertainty algorithms:
1. **AlexNet_CNN**: 5-layer convolutional encoder + 3 dense layers with Dropout ($p=0.5$) for local spatial-spectral patches.
2. **GFNet (Global Filter Network)**: Uses a 2D Discrete Fourier Transform (DFT) to mix spatial tokens globally in the frequency domain via element-wise learnable filters $W$:
$$
\hat{X} = \mathcal{F}^{-1}\left(\mathcal{F}(X) \odot W\right)
$$
3. **ViT_UNet**: Hybrid Vision Transformer encoder with U-Net skip connections for sharp boundary preservation.
4.
## 2. Stage-by-Stage Process & Math
### Stage 1: From Raw Logits to Softmax Probabilities
For each pixel $i$, the network outputs raw scores (logits) $z_i = (z_{i,1}, \dots, z_{i,K})$ across $K$ land-cover classes. Softmax converts these logits into probabilities summing to $1$:

$$
\hat{p}_{i,k} = \frac{\exp(z_{i,k})}{\sum_{j=1}^K \exp(z_{i,j})}
$$

### Stage 2: Cross-Entropy Training Loss
The network weights $\theta$ are optimized by minimizing Categorical Cross-Entropy (Negative Log-Likelihood) on the training set:

$$
\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} \log \hat{p}_{i, y_i}
$$

where $y_i$ is the true class of pixel $i$.
## 3. Simple Numerical Example (Why Softmax Alone Is Not Enough)
Suppose a boundary pixel between *Forest* and *Crop* has logits $z = [-1.2, 2.1, -0.5, 1.9]$ across 4 classes (*Water*, *Forest*, *Urban*, *Crop*):
- Exponentiated values: $\exp(z) = [0.301, 8.166, 0.607, 6.686]$, sum $= 15.760$.
- Softmax probabilities: $\hat{p}_{\text{Water}} = 1.9\%$, $\hat{p}_{\text{Forest}} = 51.8\%$, $\hat{p}_{\text{Urban}} = 3.9\%$, $\hat{p}_{\text{Crop}} = 42.4\%$.
- A standard classifier outputs only $\arg\max_k \hat{p}_k = \text{Forest}$ and hides the $42.4\%$ chance of *Crop*. This motivates `Model_uncertainty_comparison.ipynb`.

## 4. Saved Outputs in `Classification/baseline/results/`
- `scene_visualizations/AlexNet_CNN_initial_classification.png`, `GFNet_initial_classification.png`, `ViT_UNet_initial_classification.png`
- `scene_visualizations/initial_classification_maps.xlsx`
