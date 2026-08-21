# 1.0 — Setup & Imports
Mount Google Drive and import all required standard, third-party, and ML libraries.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```text
Mounted at /content/drive
```

```python
# Standard library
import io
import json
import os
import random
import time
from pathlib import Path

# Third-party – data / viz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as XLImage
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

sns.set_style("whitegrid")
print("TensorFlow:", tf.__version__)
```

```text
TensorFlow: 2.20.0
```

# 2.0 — Configuration
All hyper-parameters, paths, and architecture configs are defined here. Change values in this single cell to reconfigure the whole notebook.

```python
# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────
MODEL_DIR   = Path("/content/drive/My Drive/Classification/spatialgcn/models")
RESULTS_DIR = Path("/content/drive/My Drive/Classification/spatialgcn/results")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR    = RESULTS_DIR / "training_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR     = RESULTS_DIR / "scene_visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR   = Path("/content/drive/My Drive/Classification/data")
DATA_FILE  = DATA_DIR / "data.csv"
LABEL_FILE = DATA_DIR / "ref.csv"

# ── Dataset geometry ──────────────────────────────────────────────────
H, W, B    = 330, 307, 6          # height, width, bands
PATCH_SIZE  = 9
INNER_PATCH = 3                   # unused by SpatialGCN; kept for compatibility with shared helpers

# ── Train / val / test splits ─────────────────────────────────────────
TRAIN_PERCENT          = 0.75
VAL_SPLIT_FROM_TRAIN   = 0.20

# ── Training ──────────────────────────────────────────────────────────
BATCH_SIZE    = 128
EPOCHS        = 100
LEARNING_RATE = 3e-4
DROPOUT_RATE  = 0.25
CAPACITY_PRESET = "spatialgcn_arch"

TRAIN_CFG = {
    "label_smoothing": 0.05,
    "weight_decay":    1e-4,
    "clipnorm":        1.0,
    "cosine_alpha":    0.05,
}

# ── SpatialGCN architecture config ──────────────────────────────────────
# Treats each 9x9 patch as a fixed regular-grid graph (one node per pixel,
# 4- or 8-connected neighbours) and runs graph convolutions over it, in the
# spirit of the power-grid graph convolution used by graph-based foundation
# models for spatial-temporal data (e.g. GridFM, Hamann et al., 2024),
# applied here to a pixel-adjacency grid rather than an electrical network.
GCN_CFG = {
    "connectivity": 4,                    # 4- or 8-connected pixel neighbours
    "gcn_units":    [64, 128, 128],        # one GraphConvLayer per entry
    "dense_units":  [128, 64],             # classifier head after pooling
}

# Fallback config used when Colab runs out of memory
GCN_FALLBACK_CFG = {
    "connectivity": 4,
    "gcn_units":    [48, 96],
    "dense_units":  [96, 48],
}

# ── Visualisation palette ─────────────────────────────────────────────
CLASS_COLOR_BASE = [
    "#0000FF", "#00FF00", "#FF0000", "#00FFFF", "#FF00FF",
    "#FFFF00", "#A52A2A", "#FFA500", "#7FFF00", "#8A2BE2",
]
BACKGROUND_COLOR  = "#000000"
VIS_EXCEL_PATH    = VIS_DIR / "initial_classification_maps.xlsx"

print("Data file:    ", DATA_FILE)
print("Label file:   ", LABEL_FILE)
print("Model dir:    ", MODEL_DIR)
print("Results dir:  ", RESULTS_DIR)
print("Plot dir:     ", PLOT_DIR)
print("Arch preset:  ", CAPACITY_PRESET)
```

```text
Data file:     /content/drive/My Drive/Classification/data/data.csv
Label file:    /content/drive/My Drive/Classification/data/ref.csv
Model dir:     /content/drive/My Drive/Classification/spatialgcn/models
Results dir:   /content/drive/My Drive/Classification/spatialgcn/results
Plot dir:      /content/drive/My Drive/Classification/spatialgcn/results/training_plots
Arch preset:   spatialgcn_arch
```

# 3.0 — Data Loading & Preprocessing
Load the raw multispectral CSV files, apply per-band min-max normalisation, extract labelled patches, and create stratified train / val / test splits.

## 3.1 — Loading and patch extraction helpers

```python
def load_multispectral_6band(data_path, label_path, height, width, bands):
    """Load multispectral data and reference labels; normalise each band to [0, 1]."""
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(height, width, bands)
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(height, width)

    x_norm = np.empty_like(x, dtype=np.float32)
    for b in range(bands):
        band     = x[:, :, b]
        band_min = np.min(band)
        band_max = np.max(band)
        denom    = max(band_max - band_min, 1e-8)
        x_norm[:, :, b] = (band - band_min) / denom

    return x_norm, y


def extract_labeled_patches(x, y, patch_size=9):
    """Extract (patch_size × patch_size) spatial patches around every labelled pixel."""
    pad   = patch_size // 2
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

    coords  = np.argwhere(y > 0)
    patches = np.empty((coords.shape[0], patch_size, patch_size, x.shape[-1]), dtype=np.float32)
    labels  = np.empty((coords.shape[0],), dtype=np.int32)

    for i, (r, c) in enumerate(coords):
        patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
        labels[i]  = int(y[r, c]) - 1  # convert class IDs to 0..C-1

    return patches, labels, coords
```

## 3.2 — Load data and create splits

```python
x_img, y_img = load_multispectral_6band(DATA_FILE, LABEL_FILE, H, W, B)
X, y, coords = extract_labeled_patches(x_img, y_img, PATCH_SIZE)

num_classes = int(np.unique(y).size)
input_shape = (PATCH_SIZE, PATCH_SIZE, B)

print("x_img:", x_img.shape, "  y_img:", y_img.shape)
print("Labelled samples:", X.shape[0])
print("Patch tensor:    ", X.shape)
print("Num classes:     ", num_classes)
```

```text
x_img: (330, 307, 6)   y_img: (330, 307)
Labelled samples: 17239
Patch tensor:     (17239, 9, 9, 6)
Num classes:      7
```

```python
# Stratified split used by SpatialGCN
x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y,
    train_size=TRAIN_PERCENT,
    random_state=SEED,
    stratify=y,
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=VAL_SPLIT_FROM_TRAIN,
    random_state=SEED,
    stratify=y_train_full,
)

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat   = keras.utils.to_categorical(y_val,   num_classes)
y_test_cat  = keras.utils.to_categorical(y_test,  num_classes)

print("Train:", x_train.shape, y_train.shape)
print("Val:  ", x_val.shape,   y_val.shape)
print("Test: ", x_test.shape,  y_test.shape)
```

```text
Train: (10343, 9, 9, 6) (10343,)
Val:   (2586, 9, 9, 6) (2586,)
Test:  (4310, 9, 9, 6) (4310,)
```

# 4.0 — Model Definitions
Defines SpatialGCN: a graph-convolutional classifier over the fixed pixel-adjacency grid of each patch. A custom Keras layer is registered for serialisation.

## 4.1 — Shared custom Keras layer: GraphConvLayer

```python
@tf.keras.utils.register_keras_serializable()
class GraphConvLayer(layers.Layer):
    """Graph convolution over a fixed pixel-adjacency grid graph.

    Each pixel in the (patch_size x patch_size) patch is treated as a graph
    node, connected to its 4- or 8-connected spatial neighbours. Since every
    patch has the identical regular-grid topology, the (symmetrically
    normalised, Kipf & Welling, 2017-style) adjacency matrix is a fixed
    constant computed once at build time -- not a per-sample input. This
    mirrors the graph-convolution formulation used by graph-based foundation
    models for spatial(-temporal) data (e.g. GridFM's power-grid convolution,
    Hamann et al., 2024, Eq. 36), applied here to a pixel grid instead of an
    electrical network.
    """

    def __init__(self, patch_size, units, connectivity=4, activation="gelu", **kwargs):
        super().__init__(**kwargs)
        self.patch_size   = patch_size
        self.units        = units
        self.connectivity = connectivity
        self.activation   = keras.activations.get(activation)

    def build(self, input_shape):
        n = self.patch_size * self.patch_size
        adj = np.zeros((n, n), dtype="float32")
        for r in range(self.patch_size):
            for c in range(self.patch_size):
                idx = r * self.patch_size + c
                neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
                if self.connectivity == 8:
                    neighbors += [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
                for nr, nc in neighbors:
                    if 0 <= nr < self.patch_size and 0 <= nc < self.patch_size:
                        adj[idx, nr * self.patch_size + nc] = 1.0
        adj = adj + np.eye(n, dtype="float32")                    # self-loops
        deg = adj.sum(axis=1)
        d_inv_sqrt = np.zeros_like(deg)
        nonzero = deg > 0
        d_inv_sqrt[nonzero] = np.power(deg[nonzero], -0.5)
        norm_adj = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]  # symmetric normalisation

        self.norm_adj = tf.constant(norm_adj, dtype=tf.float32)     # (n, n), fixed
        self.kernel = layers.Dense(self.units, use_bias=True, name=f"{self.name}_kernel")
        super().build(input_shape)

    def call(self, node_features):
        # node_features: (batch, n_nodes, in_dim)
        aggregated = tf.einsum("ij,bjd->bid", self.norm_adj, node_features)
        return self.activation(self.kernel(aggregated))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "patch_size": self.patch_size, "units": self.units,
            "connectivity": self.connectivity,
            "activation": keras.activations.serialize(self.activation),
        })
        return cfg
```

## 4.2 — SpatialGCN: graph convolution over the pixel-adjacency grid

Each 9x9 patch is treated as a graph with one node per pixel and edges to its 4-connected neighbours (`GCN_CFG["connectivity"]`). A stack of `GraphConvLayer`s aggregates spectral information across this fixed grid graph, followed by global average pooling over all nodes and a small MLP classifier head.

This is a genuinely different computational paradigm from the earlier spectral-hypernetwork, hierarchical-transformer, and CLIP-style models: no attention, no wavelength conditioning, no text tower — just local graph message-passing, which is a natural fit for the strong spatial correlation between neighbouring pixels in a land-cover patch. Because the graph topology is a fixed regular grid (not a pretrained/external structure), this model is trained entirely from scratch with no external-knowledge caveat, unlike the DOFA/GeoRSCLIP-style layers built previously.

```python
def build_spatial_gcn(input_shape, num_classes, dropout_rate=0.25, cfg=None):
    """Build the SpatialGCN classifier: graph convolutions over the fixed
    pixel-adjacency grid of a patch, followed by global average pooling and
    an MLP head.
    """
    cfg = cfg or GCN_CFG
    patch_h, patch_w, bands = input_shape
    assert patch_h == patch_w, "SpatialGCN assumes a square patch."

    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape((patch_h * patch_w, bands), name="gcn_flatten_nodes")(inputs)  # (B, n_nodes, bands)

    for i, units in enumerate(cfg["gcn_units"], start=1):
        x = GraphConvLayer(patch_h, units, connectivity=cfg["connectivity"], name=f"gcn_layer{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"gcn_drop{i}")(x)

    x = layers.GlobalAveragePooling1D(name="gcn_gap")(x)
    for i, units in enumerate(cfg["dense_units"], start=1):
        x = layers.Dense(units, activation="gelu", name=f"gcn_fc{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"gcn_fc_drop{i}")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="gcn_logits")(x)
    return keras.Model(inputs, outputs, name="SpatialGCN")
```

# 5.0 — Training & Evaluation Helpers
Optimiser factory, learning-rate schedule, calibration metrics (Brier score, ECE), and the main `train_save_evaluate` function.

## 5.1 — Calibration metrics

```python
def multiclass_brier_score(y_onehot, y_prob):
    """Compute the multiclass Brier score (mean squared error over probability vectors)."""
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Compute Expected Calibration Error (ECE) with equal-width confidence bins."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct     = (predictions == y_true).astype(np.float32)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece         = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        prop   = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(correct[in_bin]) - np.mean(confidences[in_bin])) * prop

    return float(ece)
```

## 5.2 — Optimiser factory

```python
def make_optimizer(num_train_samples):
    """Build an AdamW optimiser with a cosine-decay learning-rate schedule."""
    steps_per_epoch = int(np.ceil(num_train_samples / BATCH_SIZE))
    decay_steps     = max(1, steps_per_epoch * EPOCHS)
    lr_schedule     = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=decay_steps,
        alpha=TRAIN_CFG["cosine_alpha"],
    )
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=TRAIN_CFG["weight_decay"],
        clipnorm=TRAIN_CFG["clipnorm"],
    )
```

## 5.3 — Main training, saving, and evaluation function

```python
def train_save_evaluate(model_name, model_builder, capacity_tag="max"):
    """Compile, train, save (best + final), and evaluate a single-head model.

    Returns a metrics dict, per-class classification report, confusion matrix,
    and the raw Keras history dict.
    """
    tf.keras.backend.clear_session()
    model      = model_builder()
    best_path  = MODEL_DIR / f"{model_name}_best.keras"
    final_path = MODEL_DIR / f"{model_name}_final.keras"

    # AdamW + cosine decay with label smoothing.
    model.compile(
        optimizer=make_optimizer(len(x_train)),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=TRAIN_CFG["label_smoothing"]),
        metrics=["accuracy"],
    )
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path), monitor="val_loss",
            mode="min", save_best_only=True, verbose=1,
        ),
    ]
    x_tr, y_tr           = x_train, y_train_cat
    x_va, y_va           = x_val,   y_val_cat
    x_te, y_te, y_te_cat = x_test, y_test, y_test_cat
    x_eval, y_eval, y_eval_cat = x_val, y_val, y_val_cat

    train_start = time.perf_counter()
    history_obj = model.fit(
        x_tr, y_tr,
        validation_data=(x_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )
    train_time_sec = float(time.perf_counter() - train_start)

    epochs_ran = int(len(history_obj.history.get("loss", [])))
    if epochs_ran != EPOCHS:
        print(f"WARNING: {model_name} ran {epochs_ran} epochs, expected {EPOCHS}.")
    else:
        print(f"{model_name} completed full training: {epochs_ran}/{EPOCHS} epochs.")

    model.save(final_path)

    y_eval_prob = model.predict(x_eval, batch_size=BATCH_SIZE, verbose=0)
    y_test_prob = model.predict(x_te,   batch_size=BATCH_SIZE, verbose=0)
    y_test_pred = np.argmax(y_test_prob, axis=1)

    report = classification_report(y_te, y_test_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_te, y_test_pred)

    row = {
        "model":            model_name,
        "capacity_tag":     capacity_tag,
        "test_accuracy":    float(accuracy_score(y_te, y_test_pred)),
        "kappa":            float(cohen_kappa_score(y_te, y_test_pred)),
        "macro_f1":         float(f1_score(y_te, y_test_pred, average="macro")),
        "weighted_f1":      float(f1_score(y_te, y_test_pred, average="weighted")),
        "val_nll":          float(log_loss(y_eval, y_eval_prob, labels=np.arange(num_classes))),
        "test_nll":         float(log_loss(y_te,   y_test_prob, labels=np.arange(num_classes))),
        "val_brier":        multiclass_brier_score(y_eval_cat, y_eval_prob),
        "test_brier":       multiclass_brier_score(y_te_cat,   y_test_prob),
        "test_ece_15bin":   expected_calibration_error(y_te, y_test_prob, n_bins=15),
        "epochs_configured": int(EPOCHS),
        "epochs_ran":       epochs_ran,
        "train_time_sec":   train_time_sec,
        "best_model_path":  str(best_path),
        "final_model_path": str(final_path),
    }
    return row, report, cm, history_obj.history
```

# 6.0 — Model Training
Builds and trains SpatialGCN. An OOM error automatically retries with a reduced fallback config.

```python
def build_gcn_with_cfg(cfg):
    """Instantiate SpatialGCN from a config dict."""
    return build_spatial_gcn(
        input_shape, num_classes,
        dropout_rate=DROPOUT_RATE,
        cfg=cfg,
    )


model_builders = {
    "SpatialGCN": lambda: build_gcn_with_cfg(GCN_CFG),
}
```

```python
results_rows   = []
model_artifacts = {}

for model_name, builder in model_builders.items():
    print(f"\n{'=' * 25} Training {model_name} ({CAPACITY_PRESET}) {'=' * 25}")

    try:
        row, report, cm, history = train_save_evaluate(model_name, builder,
                                                       capacity_tag=CAPACITY_PRESET)
    except tf.errors.ResourceExhaustedError:
        # Automatically retry with a smaller fallback config to handle Colab OOM
        if model_name == "SpatialGCN":
            print("OOM on SpatialGCN default config. Retrying with fallback config.")
            tf.keras.backend.clear_session()
            row, report, cm, history = train_save_evaluate(
                model_name, lambda: build_gcn_with_cfg(GCN_FALLBACK_CFG),
                capacity_tag="fallback",
            )
        else:
            raise

    results_rows.append(row)
    model_artifacts[model_name] = {"report": report, "confusion_matrix": cm, "history": history}

summary_df = (
    pd.DataFrame(results_rows)
    .sort_values("test_accuracy", ascending=False)
    .reset_index(drop=True)
)
summary_df
```

```text

========================= Training SpatialGCN (spatialgcn_arch) =========================
Epoch 1/100
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - accuracy: 0.4039 - loss: 1.8510
Epoch 1: val_loss improved from None to 1.54879, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 1: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 107ms/step - accuracy: 0.4190 - loss: 1.7404 - val_accuracy: 0.4242 - val_loss: 1.5488
Epoch 2/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 5ms/step - accuracy: 0.4325 - loss: 1.5093
Epoch 2: val_loss improved from 1.54879 to 1.27814, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 2: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - accuracy: 0.4531 - loss: 1.4435 - val_accuracy: 0.5251 - val_loss: 1.2781
Epoch 3/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 5ms/step - accuracy: 0.5393 - loss: 1.2576
Epoch 3: val_loss improved from 1.27814 to 1.18151, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 3: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.5436 - loss: 1.2394 - val_accuracy: 0.5518 - val_loss: 1.1815
Epoch 4/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 6ms/step - accuracy: 0.5464 - loss: 1.1974
Epoch 4: val_loss improved from 1.18151 to 1.16362, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 4: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.5508 - loss: 1.1990 - val_accuracy: 0.5541 - val_loss: 1.1636
Epoch 5/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 6ms/step - accuracy: 0.5523 - loss: 1.1765
Epoch 5: val_loss improved from 1.16362 to 1.15222, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 5: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.5552 - loss: 1.1832 - val_accuracy: 0.5584 - val_loss: 1.1522
Epoch 6/100
[1m71/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 6ms/step - accuracy: 0.5619 - loss: 1.1568
Epoch 6: val_loss improved from 1.15222 to 1.13888, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 6: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.5611 - loss: 1.1666 - val_accuracy: 0.5623 - val_loss: 1.1389
Epoch 7/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 5ms/step - accuracy: 0.5637 - loss: 1.1476
Epoch 7: val_loss improved from 1.13888 to 1.12563, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 7: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.5648 - loss: 1.1532 - val_accuracy: 0.5673 - val_loss: 1.1256
Epoch 8/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.5765 - loss: 1.1281
Epoch 8: val_loss improved from 1.12563 to 1.10905, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 8: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.5764 - loss: 1.1352 - val_accuracy: 0.5754 - val_loss: 1.1090
Epoch 9/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.5863 - loss: 1.1124
Epoch 9: val_loss improved from 1.10905 to 1.08675, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 9: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.5874 - loss: 1.1149 - val_accuracy: 0.5986 - val_loss: 1.0868
Epoch 10/100
[1m75/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.5946 - loss: 1.0867
Epoch 10: val_loss improved from 1.08675 to 1.05522, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 10: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.6040 - loss: 1.0901 - val_accuracy: 0.6292 - val_loss: 1.0552
Epoch 11/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.6277 - loss: 1.0518
Epoch 11: val_loss improved from 1.05522 to 1.01246, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 11: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.6284 - loss: 1.0539 - val_accuracy: 0.6508 - val_loss: 1.0125
Epoch 12/100
[1m80/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.6437 - loss: 1.0144
Epoch 12: val_loss improved from 1.01246 to 0.96804, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 12: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.6478 - loss: 1.0155 - val_accuracy: 0.6736 - val_loss: 0.9680
Epoch 13/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.6662 - loss: 0.9740
Epoch 13: val_loss improved from 0.96804 to 0.92078, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 13: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.6687 - loss: 0.9747 - val_accuracy: 0.7015 - val_loss: 0.9208
Epoch 14/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.6892 - loss: 0.9281
Epoch 14: val_loss improved from 0.92078 to 0.87895, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 14: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.6933 - loss: 0.9287 - val_accuracy: 0.7177 - val_loss: 0.8789
Epoch 15/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.7131 - loss: 0.8918
Epoch 15: val_loss improved from 0.87895 to 0.84655, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 15: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7078 - loss: 0.8972 - val_accuracy: 0.7258 - val_loss: 0.8466
Epoch 16/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.7270 - loss: 0.8666
Epoch 16: val_loss improved from 0.84655 to 0.83055, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 16: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7231 - loss: 0.8733 - val_accuracy: 0.7301 - val_loss: 0.8305
Epoch 17/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.7378 - loss: 0.8479
Epoch 17: val_loss improved from 0.83055 to 0.82006, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 17: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7352 - loss: 0.8549 - val_accuracy: 0.7324 - val_loss: 0.8201
Epoch 18/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.7423 - loss: 0.8358
Epoch 18: val_loss improved from 0.82006 to 0.80731, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 18: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7359 - loss: 0.8446 - val_accuracy: 0.7413 - val_loss: 0.8073
Epoch 19/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.7464 - loss: 0.8231
Epoch 19: val_loss improved from 0.80731 to 0.79849, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 19: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7431 - loss: 0.8314 - val_accuracy: 0.7452 - val_loss: 0.7985
Epoch 20/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.7596 - loss: 0.8157
Epoch 20: val_loss improved from 0.79849 to 0.79097, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 20: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7504 - loss: 0.8280 - val_accuracy: 0.7498 - val_loss: 0.7910
Epoch 21/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.7614 - loss: 0.8055
Epoch 21: val_loss improved from 0.79097 to 0.78729, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 21: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7560 - loss: 0.8179 - val_accuracy: 0.7548 - val_loss: 0.7873
Epoch 22/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.7639 - loss: 0.8012
Epoch 22: val_loss improved from 0.78729 to 0.77717, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 22: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7585 - loss: 0.8081 - val_accuracy: 0.7626 - val_loss: 0.7772
Epoch 23/100
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.7677 - loss: 0.7958
Epoch 23: val_loss improved from 0.77717 to 0.76859, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 23: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7626 - loss: 0.8046 - val_accuracy: 0.7676 - val_loss: 0.7686
Epoch 24/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.7731 - loss: 0.7806
Epoch 24: val_loss improved from 0.76859 to 0.76313, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 24: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7655 - loss: 0.7956 - val_accuracy: 0.7707 - val_loss: 0.7631
Epoch 25/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 5ms/step - accuracy: 0.7801 - loss: 0.7788
Epoch 25: val_loss improved from 0.76313 to 0.75776, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 25: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.7698 - loss: 0.7913 - val_accuracy: 0.7749 - val_loss: 0.7578
Epoch 26/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 5ms/step - accuracy: 0.7794 - loss: 0.7712
Epoch 26: val_loss improved from 0.75776 to 0.75378, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 26: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - accuracy: 0.7728 - loss: 0.7846 - val_accuracy: 0.7757 - val_loss: 0.7538
Epoch 27/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 5ms/step - accuracy: 0.7834 - loss: 0.7710
Epoch 27: val_loss improved from 0.75378 to 0.74748, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 27: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - accuracy: 0.7758 - loss: 0.7811 - val_accuracy: 0.7796 - val_loss: 0.7475
Epoch 28/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 6ms/step - accuracy: 0.7863 - loss: 0.7641
Epoch 28: val_loss improved from 0.74748 to 0.74268, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 28: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.7783 - loss: 0.7742 - val_accuracy: 0.7831 - val_loss: 0.7427
Epoch 29/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 5ms/step - accuracy: 0.7848 - loss: 0.7615
Epoch 29: val_loss improved from 0.74268 to 0.73809, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 29: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7797 - loss: 0.7736 - val_accuracy: 0.7846 - val_loss: 0.7381
Epoch 30/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.7959 - loss: 0.7513
Epoch 30: val_loss improved from 0.73809 to 0.73423, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 30: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7842 - loss: 0.7657 - val_accuracy: 0.7889 - val_loss: 0.7342
Epoch 31/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.7925 - loss: 0.7501
Epoch 31: val_loss improved from 0.73423 to 0.73024, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 31: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7859 - loss: 0.7622 - val_accuracy: 0.7923 - val_loss: 0.7302
Epoch 32/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.7923 - loss: 0.7493
Epoch 32: val_loss improved from 0.73024 to 0.72804, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 32: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7867 - loss: 0.7603 - val_accuracy: 0.7923 - val_loss: 0.7280
Epoch 33/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.7947 - loss: 0.7454
Epoch 33: val_loss improved from 0.72804 to 0.72480, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 33: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7886 - loss: 0.7561 - val_accuracy: 0.7951 - val_loss: 0.7248
Epoch 34/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.7937 - loss: 0.7444
Epoch 34: val_loss improved from 0.72480 to 0.72143, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 34: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7879 - loss: 0.7530 - val_accuracy: 0.8009 - val_loss: 0.7214
Epoch 35/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8026 - loss: 0.7366
Epoch 35: val_loss improved from 0.72143 to 0.71999, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 35: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7936 - loss: 0.7490 - val_accuracy: 0.7997 - val_loss: 0.7200
Epoch 36/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.8003 - loss: 0.7334
Epoch 36: val_loss improved from 0.71999 to 0.71735, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 36: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7924 - loss: 0.7473 - val_accuracy: 0.8032 - val_loss: 0.7174
Epoch 37/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.7975 - loss: 0.7370
Epoch 37: val_loss improved from 0.71735 to 0.71300, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 37: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7918 - loss: 0.7473 - val_accuracy: 0.8039 - val_loss: 0.7130
Epoch 38/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8008 - loss: 0.7306
Epoch 38: val_loss improved from 0.71300 to 0.71184, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 38: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7959 - loss: 0.7425 - val_accuracy: 0.8047 - val_loss: 0.7118
Epoch 39/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8040 - loss: 0.7208
Epoch 39: val_loss improved from 0.71184 to 0.70973, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 39: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7964 - loss: 0.7328 - val_accuracy: 0.8039 - val_loss: 0.7097
Epoch 40/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.8041 - loss: 0.7212
Epoch 40: val_loss improved from 0.70973 to 0.70612, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 40: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.7974 - loss: 0.7358 - val_accuracy: 0.8059 - val_loss: 0.7061
Epoch 41/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8065 - loss: 0.7201
Epoch 41: val_loss improved from 0.70612 to 0.70598, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 41: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7984 - loss: 0.7326 - val_accuracy: 0.8039 - val_loss: 0.7060
Epoch 42/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8039 - loss: 0.7233
Epoch 42: val_loss improved from 0.70598 to 0.70222, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 42: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7999 - loss: 0.7321 - val_accuracy: 0.8090 - val_loss: 0.7022
Epoch 43/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8049 - loss: 0.7188
Epoch 43: val_loss improved from 0.70222 to 0.69908, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 43: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.7979 - loss: 0.7311 - val_accuracy: 0.8121 - val_loss: 0.6991
Epoch 44/100
[1m75/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8084 - loss: 0.7165
Epoch 44: val_loss improved from 0.69908 to 0.69818, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 44: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8016 - loss: 0.7281 - val_accuracy: 0.8109 - val_loss: 0.6982
Epoch 45/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8059 - loss: 0.7121
Epoch 45: val_loss improved from 0.69818 to 0.69456, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 45: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8003 - loss: 0.7252 - val_accuracy: 0.8140 - val_loss: 0.6946
Epoch 46/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8097 - loss: 0.7092
Epoch 46: val_loss improved from 0.69456 to 0.69404, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 46: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - accuracy: 0.8043 - loss: 0.7222 - val_accuracy: 0.8121 - val_loss: 0.6940
Epoch 47/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 5ms/step - accuracy: 0.8113 - loss: 0.7062
Epoch 47: val_loss improved from 0.69404 to 0.69178, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 47: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.8045 - loss: 0.7186 - val_accuracy: 0.8121 - val_loss: 0.6918
Epoch 48/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 6ms/step - accuracy: 0.8151 - loss: 0.7062
Epoch 48: val_loss improved from 0.69178 to 0.68982, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 48: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.8073 - loss: 0.7191 - val_accuracy: 0.8105 - val_loss: 0.6898
Epoch 49/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 6ms/step - accuracy: 0.8133 - loss: 0.6988
Epoch 49: val_loss improved from 0.68982 to 0.68675, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 49: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.8046 - loss: 0.7150 - val_accuracy: 0.8140 - val_loss: 0.6867
Epoch 50/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 6ms/step - accuracy: 0.8152 - loss: 0.6980
Epoch 50: val_loss improved from 0.68675 to 0.68617, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 50: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - accuracy: 0.8100 - loss: 0.7084 - val_accuracy: 0.8136 - val_loss: 0.6862
Epoch 51/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 5ms/step - accuracy: 0.8117 - loss: 0.7037
Epoch 51: val_loss improved from 0.68617 to 0.68344, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 51: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8058 - loss: 0.7163 - val_accuracy: 0.8140 - val_loss: 0.6834
Epoch 52/100
[1m69/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 5ms/step - accuracy: 0.8127 - loss: 0.6960
Epoch 52: val_loss improved from 0.68344 to 0.68114, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 52: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8044 - loss: 0.7105 - val_accuracy: 0.8128 - val_loss: 0.6811
Epoch 53/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8176 - loss: 0.6966
Epoch 53: val_loss improved from 0.68114 to 0.67917, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 53: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8129 - loss: 0.7076 - val_accuracy: 0.8144 - val_loss: 0.6792
Epoch 54/100
[1m75/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8195 - loss: 0.6932
Epoch 54: val_loss improved from 0.67917 to 0.67725, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 54: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8146 - loss: 0.7057 - val_accuracy: 0.8167 - val_loss: 0.6773
Epoch 55/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8182 - loss: 0.6967
Epoch 55: val_loss improved from 0.67725 to 0.67559, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 55: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8123 - loss: 0.7073 - val_accuracy: 0.8155 - val_loss: 0.6756
Epoch 56/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8198 - loss: 0.6875
Epoch 56: val_loss improved from 0.67559 to 0.67369, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 56: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8148 - loss: 0.6984 - val_accuracy: 0.8159 - val_loss: 0.6737
Epoch 57/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8245 - loss: 0.6874
Epoch 57: val_loss improved from 0.67369 to 0.67215, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 57: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8168 - loss: 0.6986 - val_accuracy: 0.8175 - val_loss: 0.6722
Epoch 58/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8264 - loss: 0.6893
Epoch 58: val_loss improved from 0.67215 to 0.67084, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 58: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8149 - loss: 0.7016 - val_accuracy: 0.8206 - val_loss: 0.6708
Epoch 59/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8221 - loss: 0.6896
Epoch 59: val_loss improved from 0.67084 to 0.66911, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 59: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8171 - loss: 0.6992 - val_accuracy: 0.8210 - val_loss: 0.6691
Epoch 60/100
[1m74/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8265 - loss: 0.6808
Epoch 60: val_loss improved from 0.66911 to 0.66804, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 60: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8188 - loss: 0.6934 - val_accuracy: 0.8221 - val_loss: 0.6680
Epoch 61/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8254 - loss: 0.6825
Epoch 61: val_loss improved from 0.66804 to 0.66579, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 61: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8195 - loss: 0.6928 - val_accuracy: 0.8221 - val_loss: 0.6658
Epoch 62/100
[1m75/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8252 - loss: 0.6780
Epoch 62: val_loss improved from 0.66579 to 0.66448, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 62: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8183 - loss: 0.6908 - val_accuracy: 0.8248 - val_loss: 0.6645
Epoch 63/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8249 - loss: 0.6830
Epoch 63: val_loss improved from 0.66448 to 0.66311, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 63: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8181 - loss: 0.6929 - val_accuracy: 0.8260 - val_loss: 0.6631
Epoch 64/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.8269 - loss: 0.6813
Epoch 64: val_loss improved from 0.66311 to 0.66188, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 64: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8204 - loss: 0.6940 - val_accuracy: 0.8256 - val_loss: 0.6619
Epoch 65/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8284 - loss: 0.6758
Epoch 65: val_loss improved from 0.66188 to 0.66034, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 65: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8203 - loss: 0.6877 - val_accuracy: 0.8260 - val_loss: 0.6603
Epoch 66/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8257 - loss: 0.6724
Epoch 66: val_loss improved from 0.66034 to 0.65833, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 66: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8223 - loss: 0.6834 - val_accuracy: 0.8264 - val_loss: 0.6583
Epoch 67/100
[1m74/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 6ms/step - accuracy: 0.8313 - loss: 0.6720
Epoch 67: val_loss did not improve from 0.65833
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - accuracy: 0.8232 - loss: 0.6868 - val_accuracy: 0.8279 - val_loss: 0.6586
Epoch 68/100
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8272 - loss: 0.6715
Epoch 68: val_loss improved from 0.65833 to 0.65678, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 68: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.8219 - loss: 0.6844 - val_accuracy: 0.8283 - val_loss: 0.6568
Epoch 69/100
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8306 - loss: 0.6704
Epoch 69: val_loss improved from 0.65678 to 0.65606, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 69: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.8254 - loss: 0.6826 - val_accuracy: 0.8291 - val_loss: 0.6561
Epoch 70/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 7ms/step - accuracy: 0.8320 - loss: 0.6682
Epoch 70: val_loss improved from 0.65606 to 0.65473, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 70: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.8265 - loss: 0.6810 - val_accuracy: 0.8333 - val_loss: 0.6547
Epoch 71/100
[1m75/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 6ms/step - accuracy: 0.8283 - loss: 0.6706
Epoch 71: val_loss improved from 0.65473 to 0.65423, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 71: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - accuracy: 0.8239 - loss: 0.6814 - val_accuracy: 0.8326 - val_loss: 0.6542
Epoch 72/100
[1m70/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 5ms/step - accuracy: 0.8258 - loss: 0.6661
Epoch 72: val_loss improved from 0.65423 to 0.65151, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 72: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8240 - loss: 0.6789 - val_accuracy: 0.8329 - val_loss: 0.6515
Epoch 73/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.8341 - loss: 0.6626
Epoch 73: val_loss improved from 0.65151 to 0.64996, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 73: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8283 - loss: 0.6759 - val_accuracy: 0.8345 - val_loss: 0.6500
Epoch 74/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.8306 - loss: 0.6699
Epoch 74: val_loss improved from 0.64996 to 0.64993, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 74: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8256 - loss: 0.6797 - val_accuracy: 0.8337 - val_loss: 0.6499
Epoch 75/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8407 - loss: 0.6616
Epoch 75: val_loss improved from 0.64993 to 0.64817, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 75: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8334 - loss: 0.6752 - val_accuracy: 0.8353 - val_loss: 0.6482
Epoch 76/100
[1m74/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8365 - loss: 0.6644
Epoch 76: val_loss improved from 0.64817 to 0.64719, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 76: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8285 - loss: 0.6774 - val_accuracy: 0.8345 - val_loss: 0.6472
Epoch 77/100
[1m75/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8341 - loss: 0.6596
Epoch 77: val_loss improved from 0.64719 to 0.64628, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 77: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8303 - loss: 0.6730 - val_accuracy: 0.8353 - val_loss: 0.6463
Epoch 78/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.8341 - loss: 0.6605
Epoch 78: val_loss improved from 0.64628 to 0.64551, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 78: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8289 - loss: 0.6734 - val_accuracy: 0.8349 - val_loss: 0.6455
Epoch 79/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8351 - loss: 0.6595
Epoch 79: val_loss improved from 0.64551 to 0.64475, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 79: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8294 - loss: 0.6720 - val_accuracy: 0.8357 - val_loss: 0.6447
Epoch 80/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8380 - loss: 0.6598
Epoch 80: val_loss improved from 0.64475 to 0.64390, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 80: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8325 - loss: 0.6731 - val_accuracy: 0.8357 - val_loss: 0.6439
Epoch 81/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8376 - loss: 0.6562
Epoch 81: val_loss improved from 0.64390 to 0.64341, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 81: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8312 - loss: 0.6694 - val_accuracy: 0.8357 - val_loss: 0.6434
Epoch 82/100
[1m77/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8357 - loss: 0.6580
Epoch 82: val_loss improved from 0.64341 to 0.64261, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 82: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8302 - loss: 0.6736 - val_accuracy: 0.8384 - val_loss: 0.6426
Epoch 83/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8380 - loss: 0.6569
Epoch 83: val_loss improved from 0.64261 to 0.64201, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 83: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8320 - loss: 0.6698 - val_accuracy: 0.8376 - val_loss: 0.6420
Epoch 84/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8358 - loss: 0.6590
Epoch 84: val_loss improved from 0.64201 to 0.64147, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 84: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8301 - loss: 0.6714 - val_accuracy: 0.8372 - val_loss: 0.6415
Epoch 85/100
[1m75/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8380 - loss: 0.6558
Epoch 85: val_loss improved from 0.64147 to 0.64110, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 85: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8305 - loss: 0.6691 - val_accuracy: 0.8391 - val_loss: 0.6411
Epoch 86/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8332 - loss: 0.6576
Epoch 86: val_loss improved from 0.64110 to 0.64098, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 86: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8291 - loss: 0.6690 - val_accuracy: 0.8384 - val_loss: 0.6410
Epoch 87/100
[1m74/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8379 - loss: 0.6569
Epoch 87: val_loss improved from 0.64098 to 0.64007, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 87: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8319 - loss: 0.6708 - val_accuracy: 0.8380 - val_loss: 0.6401
Epoch 88/100
[1m75/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8354 - loss: 0.6557
Epoch 88: val_loss improved from 0.64007 to 0.64002, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 88: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8304 - loss: 0.6701 - val_accuracy: 0.8391 - val_loss: 0.6400
Epoch 89/100
[1m79/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 6ms/step - accuracy: 0.8367 - loss: 0.6546
Epoch 89: val_loss improved from 0.64002 to 0.63952, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 89: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.8298 - loss: 0.6672 - val_accuracy: 0.8384 - val_loss: 0.6395
Epoch 90/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 6ms/step - accuracy: 0.8354 - loss: 0.6495
Epoch 90: val_loss improved from 0.63952 to 0.63900, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 90: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.8338 - loss: 0.6648 - val_accuracy: 0.8380 - val_loss: 0.6390
Epoch 91/100
[1m80/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 6ms/step - accuracy: 0.8334 - loss: 0.6550
Epoch 91: val_loss did not improve from 0.63900
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - accuracy: 0.8297 - loss: 0.6676 - val_accuracy: 0.8387 - val_loss: 0.6391
Epoch 92/100
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8343 - loss: 0.6532
Epoch 92: val_loss improved from 0.63900 to 0.63839, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 92: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.8311 - loss: 0.6656 - val_accuracy: 0.8384 - val_loss: 0.6384
Epoch 93/100
[1m75/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 7ms/step - accuracy: 0.8385 - loss: 0.6524
Epoch 93: val_loss improved from 0.63839 to 0.63822, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 93: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - accuracy: 0.8331 - loss: 0.6670 - val_accuracy: 0.8391 - val_loss: 0.6382
Epoch 94/100
[1m73/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 5ms/step - accuracy: 0.8381 - loss: 0.6545
Epoch 94: val_loss improved from 0.63822 to 0.63775, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 94: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8309 - loss: 0.6676 - val_accuracy: 0.8391 - val_loss: 0.6378
Epoch 95/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.8335 - loss: 0.6521
Epoch 95: val_loss improved from 0.63775 to 0.63771, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 95: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8294 - loss: 0.6673 - val_accuracy: 0.8384 - val_loss: 0.6377
Epoch 96/100
[1m70/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.8354 - loss: 0.6486
Epoch 96: val_loss improved from 0.63771 to 0.63751, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 96: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8314 - loss: 0.6643 - val_accuracy: 0.8391 - val_loss: 0.6375
Epoch 97/100
[1m76/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8398 - loss: 0.6480
Epoch 97: val_loss improved from 0.63751 to 0.63715, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 97: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8310 - loss: 0.6651 - val_accuracy: 0.8395 - val_loss: 0.6371
Epoch 98/100
[1m74/81[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 4ms/step - accuracy: 0.8413 - loss: 0.6518
Epoch 98: val_loss improved from 0.63715 to 0.63666, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 98: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.8344 - loss: 0.6652 - val_accuracy: 0.8395 - val_loss: 0.6367
Epoch 99/100
[1m78/81[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 4ms/step - accuracy: 0.8352 - loss: 0.6548
Epoch 99: val_loss improved from 0.63666 to 0.63639, saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras

Epoch 99: finished saving model to /content/drive/My Drive/Classification/spatialgcn/models/SpatialGCN_best.keras
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - accuracy: 0.8315 - loss: 0.6652 - val_accuracy: 0.8399 - val_loss: 0.6364
Epoch 100/100
[1m72/81[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 4ms/step - accuracy: 0.8378 - loss: 0.6490
Epoch 100: val_loss did not improve from 0.63639
[1m81/81[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8309 - loss: 0.6640 - val_accuracy: 0.8395 - val_loss: 0.6364
SpatialGCN completed full training: 100/100 epochs.
```

```text
        model     capacity_tag  test_accuracy     kappa  macro_f1  \
0  SpatialGCN  spatialgcn_arch       0.845708  0.788338  0.782528   

   weighted_f1   val_nll  test_nll  val_brier  test_brier  test_ece_15bin  \
0     0.843014  0.466311  0.453401   0.232072    0.224204        0.059742   

   epochs_configured  epochs_ran  train_time_sec  \
0                100         100       82.785414   

                                     best_model_path  \
0  /content/drive/My Drive/Classification/spatial...   

                                    final_model_path  
0  /content/drive/My Drive/Classification/spatial...  
```

# 7.0 — Results & Metrics
Save classification summaries to CSV / JSON, then plot training curves, cross-model bar charts, calibration proxies, and confusion matrices.

## 7.1 — Save summary CSV and per-model JSON reports

```python
summary_path = RESULTS_DIR / "classification_summary.csv"
summary_df.to_csv(summary_path, index=False)
print("Saved summary:", summary_path)

for model_name, artifact in model_artifacts.items():
    report_path = RESULTS_DIR / f"{model_name}_classification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(artifact["report"], f, indent=2)

print("Per-model classification reports saved to:", RESULTS_DIR)
summary_df
```

```text
Saved summary: /content/drive/My Drive/Classification/spatialgcn/results/classification_summary.csv
Per-model classification reports saved to: /content/drive/My Drive/Classification/spatialgcn/results
```

```text
        model     capacity_tag  test_accuracy     kappa  macro_f1  \
0  SpatialGCN  spatialgcn_arch       0.845708  0.788338  0.782528   

   weighted_f1   val_nll  test_nll  val_brier  test_brier  test_ece_15bin  \
0     0.843014  0.466311  0.453401   0.232072    0.224204        0.059742   

   epochs_configured  epochs_ran  train_time_sec  \
0                100         100       82.785414   

                                     best_model_path  \
0  /content/drive/My Drive/Classification/spatial...   

                                    final_model_path  
0  /content/drive/My Drive/Classification/spatial...  
```

## 7.2 — Training curves (accuracy and loss per epoch)

```python
for model_name, artifact in model_artifacts.items():
    hist   = artifact["history"]
    epochs = np.arange(1, len(hist["loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(epochs, hist["accuracy"],     color="#1f77b4", linewidth=2,   label="Train Accuracy")
    ax1.plot(epochs, hist["val_accuracy"], color="#2ca02c", linewidth=2, linestyle="--", label="Val Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, hist["loss"],     color="#111111", linewidth=1.8,   label="Train Loss")
    ax2.plot(epochs, hist["val_loss"], color="#d62728", linewidth=1.8, linestyle="--", label="Val Loss")
    ax2.set_ylabel("Loss")

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)
    plt.title(f"{model_name}: Accuracy / Loss vs Epoch")
    plt.tight_layout()

    curve_path = PLOT_DIR / f"{model_name}_training_curves.png"
    plt.savefig(curve_path, dpi=200, bbox_inches="tight")
    plt.show()
```

## 7.3 — Cross-model comparison bar charts

```python
plot_df = summary_df[[
    "model", "test_accuracy", "macro_f1", "kappa", "train_time_sec",
    "test_nll", "test_brier", "test_ece_15bin",
]].copy()

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes    = axes.flatten()
palette = ["#4c72b0", "#55a868", "#c44e52"]

metric_specs = [
    ("test_accuracy",  "Test Accuracy",   (0.0, 1.0), "{:.3f}"),
    ("macro_f1",       "Macro F1",        (0.0, 1.0), "{:.3f}"),
    ("kappa",          "Cohen Kappa",     (0.0, 1.0), "{:.3f}"),
    ("train_time_sec", "Train Time (sec)", None,       "{:.1f}"),
]

for ax, (col, title, ylim, fmt) in zip(axes, metric_specs):
    local = plot_df[["model", col]].copy()
    bars  = ax.bar(local["model"], local[col], color=palette[:len(local)])
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    for patch in bars:
        height = patch.get_height()
        xpos   = patch.get_x() + patch.get_width() / 2
        offset = (ylim[1] - ylim[0]) * 0.02 if ylim else max(local[col].max() * 0.02, 0.05)
        ax.text(xpos, height + offset, fmt.format(height), ha="center", va="bottom", fontsize=9)

plt.suptitle("Single-Head Model Comparison", y=1.02, fontsize=14)
plt.tight_layout()
comparison_path = PLOT_DIR / "model_comparison_metrics.png"
plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
plt.show()
```

## 7.4 — Calibration proxy metrics and confusion matrices

```python
# Calibration proxy chart (lower is better for all three metrics)
fig, ax     = plt.subplots(figsize=(12, 5))
calib_cols  = ["test_nll", "test_brier", "test_ece_15bin"]
calib_df    = plot_df[["model"] + calib_cols].set_index("model")
calib_df.plot(kind="bar", ax=ax, rot=0)
ax.set_title("Uncertainty Proxy Metrics (Lower is Better)")
ax.set_ylabel("Metric Value")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
calib_path = PLOT_DIR / "uncertainty_proxy_metrics.png"
plt.savefig(calib_path, dpi=200, bbox_inches="tight")
plt.show()

# Side-by-side confusion matrices
class_ticks = [str(i + 1) for i in range(num_classes)]
fig, axes   = plt.subplots(1, len(model_artifacts), figsize=(7 * len(model_artifacts), 5.5))
if len(model_artifacts) == 1:
    axes = [axes]

for ax, (model_name, artifact) in zip(axes, model_artifacts.items()):
    sns.heatmap(
        artifact["confusion_matrix"],
        annot=True, fmt="d", cmap="Blues",
        xticklabels=class_ticks, yticklabels=class_ticks,
        cbar=False, ax=ax,
    )
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

plt.tight_layout()
cm_path = PLOT_DIR / "confusion_matrices_side_by_side.png"
plt.savefig(cm_path, dpi=220, bbox_inches="tight")
plt.show()
print("Saved confusion matrix figure:", cm_path)
print("All training plots saved to:", PLOT_DIR)
```

```text
Saved confusion matrix figure: /content/drive/My Drive/Classification/spatialgcn/results/training_plots/confusion_matrices_side_by_side.png
All training plots saved to: /content/drive/My Drive/Classification/spatialgcn/results/training_plots
```

# 8.0 — Scene Visualisation
Load the saved best models, run dense patch-by-patch inference over the entire scene, and save classified maps as PNGs and an Excel workbook.

## 8.1 — Visualisation helpers

```python
CUSTOM_OBJECTS = {
    "GraphConvLayer": GraphConvLayer,
}


def get_scene_rgb(x_img, bands):
    """Select three bands as an approximate RGB composite; clip to [0, 1]."""
    if bands >= 6:
        rgb_idx = [bands // 2 - 1, bands // 2, bands // 2 + 1]
    elif bands >= 3:
        rgb_idx = [0, 1, 2]
    else:
        raise ValueError(f"Need at least 3 bands to form RGB, got {bands}.")
    return np.clip(x_img[:, :, rgb_idx], 0.0, 1.0)


def get_display_cmap(num_classes):
    """Build a ListedColormap: index 0 = background black, 1..K = class colours."""
    return ListedColormap([BACKGROUND_COLOR] + CLASS_COLOR_BASE[:num_classes])


def predict_full_scene_labels(model, x_img, patch_size=9, batch_size=256):
    """Run dense sliding-window inference row-by-row; returns a (H, W) label map."""
    pad   = patch_size // 2
    x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    preds = np.zeros((x_img.shape[0], x_img.shape[1]), dtype=np.int32)

    for r in range(x_img.shape[0]):
        row_patches = np.empty(
            (x_img.shape[1], patch_size, patch_size, x_img.shape[-1]),
            dtype=np.float32,
        )
        for c in range(x_img.shape[1]):
            row_patches[c] = x_pad[r:r + patch_size, c:c + patch_size, :]

        row_prob  = model.predict(row_patches, batch_size=batch_size, verbose=0)
        preds[r]  = np.argmax(row_prob, axis=1) + 1  # back to 1..K

    return preds


def load_saved_models_for_visualization(model_dir, custom_objects):
    """Attempt to load all three best models; return dicts of loaded and missing."""
    model_files = {
        "SpatialGCN": model_dir / "SpatialGCN_best.keras",
    }
    loaded, missing = {}, {}
    for model_key, model_path in model_files.items():
        if not model_path.exists():
            missing[model_key] = f"Missing file: {model_path}"
            continue
        try:
            loaded[model_key] = keras.models.load_model(
                model_path, custom_objects=custom_objects,
                compile=False, safe_mode=False,
            )
        except Exception as exc:
            missing[model_key] = f"Load failed: {exc}"
    return loaded, missing


def save_single_panel(image, title, save_path, cmap=None, vmin=None, vmax=None):
    """Save a single image panel as a high-DPI PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if cmap is None:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved:", save_path)


def save_combined_overview(rgb_image, y_true, pred_maps, missing_info, num_classes, save_path):
    """Save a five-panel side-by-side overview: RGB, GT, and one panel per model."""
    cmap   = get_display_cmap(num_classes)
    panels = [
        ("Approximate RGB",                  rgb_image,                    "rgb"),
        ("True Label Map",                   y_true,                       "label"),
        ("SpatialGCN Initial Classification", pred_maps.get("SpatialGCN"), "label"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(30, 7), sharex=True, sharey=True)
    for ax, (title, img, kind) in zip(axes, panels):
        ax.set_title(title, fontsize=13)
        if kind == "rgb":
            ax.imshow(img)
        elif img is None:
            ax.imshow(np.zeros_like(y_true), cmap=cmap, vmin=0, vmax=num_classes)
            key = {"SpatialGCN": "SpatialGCN"}.get(
                next((k for k in ("SpatialGCN",) if k in title), None)
            )
            ax.text(0.5, 0.5, missing_info.get(key, "Unavailable"),
                    ha="center", va="center", fontsize=10, wrap=True, transform=ax.transAxes)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=num_classes)
        ax.axis("off")

    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKGROUND_COLOR, label="Background")]
    for i in range(num_classes):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR_BASE[i], label=f"Class {i+1}"))
    fig.legend(handles=handles, loc="lower center", ncol=min(num_classes + 1, 8),
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved:", save_path)


def save_images_to_excel(excel_path, image_paths, sheet_name="Initial_Classification_Maps"):
    """Embed PNG images into an Excel workbook at fixed anchor positions."""
    if excel_path.exists():
        wb = load_workbook(excel_path)
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
        ws = wb.create_sheet(title=sheet_name)
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name

    anchors = ["A1", "J1", "A25", "J25", "A49", "J49"]
    for path, anchor in zip(image_paths, anchors):
        if path.exists():
            img        = XLImage(str(path))
            img.anchor = anchor
            ws.add_image(img)

    wb.save(excel_path)
    print("Saved Excel workbook:", excel_path)
```

## 8.2 — Generate and save all scene maps

```python
scene_rgb    = get_scene_rgb(x_img, B)
display_cmap = get_display_cmap(num_classes)

loaded_models, missing_models = load_saved_models_for_visualization(MODEL_DIR, CUSTOM_OBJECTS)
prediction_maps = {}

for model_key, loaded_model in loaded_models.items():
    print(f"Generating full-scene classified image for {model_key}...")
    prediction_maps[model_key] = predict_full_scene_labels(
        loaded_model, x_img, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE,
    )
    print(f"{model_key} map shape:", prediction_maps[model_key].shape)

if missing_models:
    print("Some models could not be loaded:")
    for model_key, reason in missing_models.items():
        print(f"  - {model_key}: {reason}")

# Individual PNG paths
rgb_png      = VIS_DIR / "scene_rgb.png"
gt_png       = VIS_DIR / "ground_truth_label_map.png"
gcn_png      = VIS_DIR / "SpatialGCN_initial_classification.png"
overview_png = VIS_DIR / "combined_initial_classification_overview.png"

save_single_panel(scene_rgb, "Approximate RGB", rgb_png)
save_single_panel(y_img, "Ground Truth Label Map", gt_png, cmap=display_cmap, vmin=0, vmax=num_classes)

if prediction_maps.get("SpatialGCN") is not None:
    save_single_panel(prediction_maps["SpatialGCN"], "SpatialGCN Initial Classification",
                      gcn_png, cmap=display_cmap, vmin=0, vmax=num_classes)

save_combined_overview(
    rgb_image=scene_rgb, y_true=y_img, pred_maps=prediction_maps,
    missing_info=missing_models, num_classes=num_classes, save_path=overview_png,
)

image_paths_for_excel = [rgb_png, gt_png, gcn_png, overview_png]
save_images_to_excel(VIS_EXCEL_PATH, image_paths_for_excel, sheet_name="Initial_Classification_Maps")

print("All visualisation outputs saved to:", VIS_DIR)
```

```text
Generating full-scene classified image for SpatialGCN...
SpatialGCN map shape: (330, 307)
```

```text
Saved: /content/drive/My Drive/Classification/spatialgcn/results/scene_visualizations/scene_rgb.png
```

```text
Saved: /content/drive/My Drive/Classification/spatialgcn/results/scene_visualizations/ground_truth_label_map.png
```

```text
Saved: /content/drive/My Drive/Classification/spatialgcn/results/scene_visualizations/SpatialGCN_initial_classification.png
```

```text
Saved: /content/drive/My Drive/Classification/spatialgcn/results/scene_visualizations/combined_initial_classification_overview.png
Saved Excel workbook: /content/drive/My Drive/Classification/spatialgcn/results/scene_visualizations/initial_classification_maps.xlsx
All visualisation outputs saved to: /content/drive/My Drive/Classification/spatialgcn/results/scene_visualizations
```

