# 1.0 — Setup & Imports

Mount Google Drive and import all required standard, scientific, and deep-learning libraries.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
# ── Standard library ──────────────────────────────────────────────────────────
import os
import glob
import json
import time
import random
from pathlib import Path

# ── Scientific / data ─────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualisation ─────────────────────────────────────────────────────────────
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ── Deep learning ─────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── ML utilities ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix,
    classification_report, f1_score, log_loss
)

# ── Excel export ──────────────────────────────────────────────────────────────
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage

sns.set_style('whitegrid')
print('TensorFlow:', tf.__version__)
```

# 2.0 — Configuration

All paths, seeds, dataset geometry, training hyperparameters, and per-architecture configs in one place. **Edit only this section between experiments.**

```python
# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Directory layout ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/content/drive/My Drive/Classification')
DATA_DIR     = PROJECT_ROOT / 'data'
MODEL_DIR    = PROJECT_ROOT / 'credit' / 'models'
RESULTS_DIR  = PROJECT_ROOT / 'credit' / 'results'
ENSEMBLE_DIR = PROJECT_ROOT / 'ensemble' / 'models' / 'ensembles'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = RESULTS_DIR / 'training_plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE  = DATA_DIR / 'data.csv'
LABEL_FILE = DATA_DIR / 'ref.csv'

# ── Dataset geometry ──────────────────────────────────────────────────────────
H, W, B     = 330, 307, 6   # height, width, spectral bands
PATCH_SIZE  = 9
INNER_PATCH = 3             # tokenisation patch size for GFNet / ViT

# ── Split ratios ──────────────────────────────────────────────────────────────
TRAIN_PERCENT        = 0.75
VAL_SPLIT_FROM_TRAIN = 0.20

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE    = 128
EPOCHS        = 100
LEARNING_RATE = 3e-4
DROPOUT_RATE  = 0.25

TRAIN_CFG = {
    'label_smoothing': 0.05,
    'weight_decay':    1e-4,
    'clipnorm':        1.0,
    'cosine_alpha':    0.05,
}

# ── Architecture configs ──────────────────────────────────────────────────────
ALEXNET_CFG = {
    'conv_filters': [96, 256, 384, 384, 256],
    'dense_units':  [4096, 1024, 256, 32],
}
GFNET_CFG = {
    'hidden_dim': 512,
    'num_blocks': 5,
    'mlp_ratio':  4,
}
VIT_CFG = {
    'projection_dim':     256,
    'num_heads':          4,
    'transformer_layers': 12,
    'mlp_multiplier':     2,
    'head_units':         [512, 256, 128, 64],
}

# ── AlexNet legacy recipe (keeps uncertainty recovery reproducible) ────────────
ALEXNET_LEGACY_SPLIT_SEED    = 10
ALEXNET_LEGACY_TRAIN_PERCENT = 0.75
ALEXNET_LR_START = 0.01
ALEXNET_LR_MAX   = 0.02
ALEXNET_LR_MIN   = 0.005

# ── Sanity print ──────────────────────────────────────────────────────────────
print('Data   :', DATA_FILE)
print('Labels :', LABEL_FILE)
print('Models :', MODEL_DIR)
print('Results:', RESULTS_DIR)
```

# 3.0 — Data Loading & Patch Extraction

Reads the multispectral CSV rasters, applies per-band min-max normalisation, extracts fixed-size spatial patches centred on every labelled pixel, and builds both the standard and AlexNet-legacy train/test splits.

## 3.1 — Helper Functions

```python
def load_multispectral_6band(data_path, label_path, height, width, bands):
    """Read CSV rasters, reshape to (H, W, B), and apply per-band min-max normalisation."""
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(height, width, bands)
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(height, width)

    x_norm = np.empty_like(x, dtype=np.float32)
    for b in range(bands):
        band = x[:, :, b]
        b_min, b_max = np.min(band), np.max(band)
        x_norm[:, :, b] = (band - b_min) / max(b_max - b_min, 1e-8)
    return x_norm, y


def extract_labeled_patches(x, y, patch_size=9):
    """Extract spatial patches centred on every labelled (y > 0) pixel."""
    pad   = patch_size // 2
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

    coords  = np.argwhere(y > 0)
    patches = np.empty((len(coords), patch_size, patch_size, x.shape[-1]), dtype=np.float32)
    labels  = np.empty((len(coords),), dtype=np.int32)

    for i, (r, c) in enumerate(coords):
        patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
        labels[i]  = int(y[r, c]) - 1   # 1-indexed → 0-indexed

    return patches, labels, coords
```

## 3.2 — Load Image & Extract Patches

```python
print('Loading multispectral image...')
x_img, y_img = load_multispectral_6band(DATA_FILE, LABEL_FILE, H, W, B)

print('Extracting patches...')
X, y, coords = extract_labeled_patches(x_img, y_img, PATCH_SIZE)

num_classes = int(np.unique(y).size)
input_shape = (PATCH_SIZE, PATCH_SIZE, B)

print(f'Image  : {x_img.shape}  |  Labels : {y_img.shape}')
print(f'Patches: {X.shape}  |  Classes: {num_classes}')
```

## 3.3 — Train / Validation / Test Split

Two independent splits are created: a **standard split** (shared by GFNet and ViT) and an **AlexNet legacy split** (separate seed to preserve reproducibility of uncertainty recovery).

```python
# ── Standard split (GFNet & ViT) ──────────────────────────────────────────────
x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y, train_size=TRAIN_PERCENT, random_state=SEED, stratify=y
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=VAL_SPLIT_FROM_TRAIN, random_state=SEED, stratify=y_train_full
)

y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_val_cat   = keras.utils.to_categorical(y_val,   num_classes)
y_test_cat  = keras.utils.to_categorical(y_test,  num_classes)

print(f'Train : {x_train.shape}  Val : {x_val.shape}  Test : {x_test.shape}')

# ── AlexNet legacy split (separate seed for uncertainty recovery) ──────────────
x_train_alex, x_test_alex, y_train_alex, y_test_alex = train_test_split(
    X, y,
    train_size=ALEXNET_LEGACY_TRAIN_PERCENT,
    random_state=ALEXNET_LEGACY_SPLIT_SEED,
    stratify=y
)
y_train_alex_cat = keras.utils.to_categorical(y_train_alex, num_classes)
y_test_alex_cat  = keras.utils.to_categorical(y_test_alex,  num_classes)

print(f'AlexNet  Train : {x_train_alex.shape}  Test : {x_test_alex.shape}')
```

# 4.0 — Model Definitions

Defines all three single-head base architectures (AlexNet CNN, GFNet, ViT with U-Net skip connections) plus the shared custom Keras layers used by GFNet and ViT.

## 4.1 — Shared Custom Keras Layers

Four serialisable custom layers: patch extraction, positional encoding (with and without CLS token), and the frequency-domain Global Filter layer used in GFNet.

```python
@tf.keras.utils.register_keras_serializable()
class PatchExtractor(layers.Layer):
    """Extract non-overlapping patches from a spatial image tensor."""

    def __init__(self, patch_size=3, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        batch       = tf.shape(images)[0]
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
        patch_dim   = tf.shape(patches)[-1]
        return tf.reshape(patches, [batch, num_patches, patch_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'patch_size': self.patch_size})
        return cfg


@tf.keras.utils.register_keras_serializable()
class PatchPositionEncoder(layers.Layer):
    """Linear projection + learnable position embedding (no CLS token)."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection         = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return cfg


@tf.keras.utils.register_keras_serializable()
class PatchEncoderWithCLS(layers.Layer):
    """Linear projection + learnable position embedding WITH a prepended CLS token."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection         = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches + 1, output_dim=projection_dim
        )

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name='cls_token', shape=(1, 1, self.projection_dim),
            initializer='zeros', trainable=True
        )
        super().build(input_shape)

    def call(self, patches):
        batch      = tf.shape(patches)[0]
        patch_proj = self.projection(patches)
        cls_tokens = tf.repeat(self.cls_token, repeats=batch, axis=0)
        x          = tf.concat([cls_tokens, patch_proj], axis=1)
        positions  = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        return x + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return cfg


@tf.keras.utils.register_keras_serializable()
class GlobalFilterLayer(layers.Layer):
    """Learnable complex-valued filter applied in the 2-D frequency domain."""

    def __init__(self, token_side, **kwargs):
        super().__init__(**kwargs)
        self.token_side = token_side

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.w_real = self.add_weight(
            name='w_real', shape=(self.token_side, self.token_side, channels),
            initializer='glorot_uniform', trainable=True
        )
        self.w_imag = self.add_weight(
            name='w_imag', shape=(self.token_side, self.token_side, channels),
            initializer='zeros', trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        batch    = tf.shape(x)[0]
        channels = tf.shape(x)[-1]
        x_2d       = tf.reshape(x, [batch, self.token_side, self.token_side, channels])
        x_fft      = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
        w_complex  = tf.complex(self.w_real, self.w_imag)
        x_filtered = x_fft * w_complex
        x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
        return tf.reshape(x_spatial, [batch, self.token_side * self.token_side, channels])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'token_side': self.token_side})
        return cfg
```

## 4.2 — AlexNet CNN

```python
def build_alexnet(input_shape, num_classes, dropout_rate=0.25, cfg=None):
    """Build a single-head AlexNet-style CNN with configurable conv and dense units."""
    cfg = cfg or ALEXNET_CFG

    inputs = keras.Input(shape=input_shape)
    x = inputs

    for i, filters in enumerate(cfg['conv_filters'], start=1):
        x = layers.Conv2D(filters, (3, 3), activation='relu',
                          padding='same', name=f'alex_conv_{i}')(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                            padding='same', name='alex_pool')(x)
    x = layers.Flatten(name='alex_flatten')(x)

    fc_names   = ['alex_fc1', 'alex_fc2', 'alex_fc3', 'alex_fc4']
    drop_names = ['TRAIN_DROPOUT_1', 'TRAIN_DROPOUT_2', 'TRAIN_DROPOUT_3', None]

    for units, fc_name, drop_name in zip(cfg['dense_units'], fc_names, drop_names):
        x = layers.Dense(units, activation='relu', name=fc_name)(x)
        if drop_name:
            x = layers.Dropout(dropout_rate, name=drop_name)(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='alex_logits')(x)
    return keras.Model(inputs, outputs, name='AlexNet_SingleHead')
```

## 4.3 — Global Filter Network (GFNet)

```python
def gf_block(x, token_side, dim, mlp_ratio=4, dropout_rate=0.25, name_prefix='gf'):
    """One GFNet block: Global Filter → LayerNorm → MLP → residual add."""
    y = layers.LayerNormalization(name=f'{name_prefix}_ln1')(x)
    y = GlobalFilterLayer(token_side, name=f'{name_prefix}_gfilter')(y)
    y = layers.LayerNormalization(name=f'{name_prefix}_ln2')(y)
    y = layers.Dense(dim * mlp_ratio, activation=tf.keras.activations.gelu,
                     name=f'{name_prefix}_mlp1')(y)
    y = layers.Dropout(dropout_rate, name=f'{name_prefix}_drop1')(y)
    y = layers.Dense(dim, activation=tf.keras.activations.gelu,
                     name=f'{name_prefix}_mlp2')(y)
    y = layers.Dropout(dropout_rate, name=f'{name_prefix}_drop2')(y)
    return layers.Add(name=f'{name_prefix}_add')([x, y])


def build_gfnet(input_shape, num_classes, inner_patch=3,
                hidden_dim=512, num_blocks=5, mlp_ratio=4, dropout_rate=0.25):
    """Build a single-head Global Filter Network over tokenised image patches."""
    num_patches = (input_shape[0] // inner_patch) * (input_shape[1] // inner_patch)
    token_side  = int(np.sqrt(num_patches))

    inputs = keras.Input(shape=input_shape)
    x = PatchExtractor(inner_patch, name='gf_patch_extractor')(inputs)
    x = PatchPositionEncoder(num_patches, hidden_dim, name='gf_patch_encoder')(x)
    x = layers.Dropout(dropout_rate, name='TRAIN_DROPOUT_1')(x)

    for i in range(num_blocks):
        x = gf_block(x, token_side, hidden_dim, mlp_ratio,
                     dropout_rate, name_prefix=f'gf_block_{i+1}')

    x = layers.Dropout(dropout_rate, name='TRAIN_DROPOUT_2')(x)
    x = layers.LayerNormalization(name='gf_final_ln')(x)
    x = layers.GlobalAveragePooling1D(name='gf_gap')(x)
    x = layers.Flatten(name='gf_flatten')(x)
    x = layers.Dropout(dropout_rate, name='TRAIN_DROPOUT_3')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='gf_logits')(x)

    return keras.Model(inputs, outputs, name='GFNet_SingleHead')
```

## 4.4 — Vision Transformer with U-Net Skip Connections

```python
def transformer_block(x, num_heads, projection_dim, mlp_dim, dropout_rate, name_prefix):
    """Standard pre-LN transformer block: Multi-Head Attention + MLP with residual adds."""
    y = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln1')(x)
    y = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim,
        dropout=dropout_rate, name=f'{name_prefix}_mha'
    )(y, y)
    x = layers.Add(name=f'{name_prefix}_add1')([y, x])

    y = layers.LayerNormalization(epsilon=1e-6, name=f'{name_prefix}_ln2')(x)
    y = layers.Dense(mlp_dim, activation=tf.keras.activations.gelu,
                     name=f'{name_prefix}_mlp1')(y)
    y = layers.Dropout(dropout_rate, name=f'{name_prefix}_drop1')(y)
    y = layers.Dense(projection_dim, activation=tf.keras.activations.gelu,
                     name=f'{name_prefix}_mlp2')(y)
    y = layers.Dropout(dropout_rate, name=f'{name_prefix}_drop2')(y)
    return layers.Add(name=f'{name_prefix}_add2')([y, x])


def build_vit_unet_singlehead(
    input_shape, num_classes, inner_patch=3,
    projection_dim=256, num_heads=4, transformer_layers=12,
    mlp_multiplier=2, dropout_rate=0.25, head_units=(512, 256, 128, 64)
):
    """Build a single-head ViT with encoder-decoder skip connections (U-Net style)."""
    num_patches = (input_shape[0] // inner_patch) * (input_shape[1] // inner_patch)

    inputs = keras.Input(shape=input_shape)
    x = PatchExtractor(inner_patch, name='vit_patch_extractor')(inputs)
    x = PatchEncoderWithCLS(num_patches, projection_dim, name='vit_patch_encoder')(x)

    block_list = []
    for i in range(transformer_layers):
        x = transformer_block(
            x, num_heads=num_heads, projection_dim=projection_dim,
            mlp_dim=projection_dim * mlp_multiplier,
            dropout_rate=0.1, name_prefix=f'vit_block_{i+1}'
        )
        if i <= transformer_layers // 2:
            block_list.append(x)
        else:
            x = layers.Add(name=f'vit_skip_add_{i+1}')(
                [x, block_list[transformer_layers - i - 1]]
            )

    x         = layers.Dropout(dropout_rate, name='TRAIN_DROPOUT_1')(x)
    x         = layers.LayerNormalization(epsilon=1e-6, name='vit_cls_norm')(x)
    cls_token = layers.Lambda(lambda t: t[:, 0, :], name='vit_cls_token')(x)

    drop_slots = ['TRAIN_DROPOUT_3', None, 'TRAIN_DROPOUT_5', 'TRAIN_DROPOUT_6']
    head_names = ['vit_head_1', 'vit_head_2', 'vit_head_3', 'vit_head_4']
    y = cls_token
    for units, hname, dname in zip(head_units, head_names, drop_slots):
        y = layers.Dense(units, activation=tf.keras.activations.gelu, name=hname)(y)
        if dname:
            y = layers.Dropout(dropout_rate, name=dname)(y)

    outputs = layers.Dense(num_classes, activation='softmax', name='vit_logits')(y)
    return keras.Model(inputs, outputs, name='ViT_UNet_SingleHead')
```

# 5.0 — Training Helpers

Utility functions for calibration metrics, optimiser construction, and the formatted Excel report writer.

## 5.1 — Calibration Metrics

```python
def multiclass_brier_score(y_onehot, y_prob):
    """Compute the multiclass Brier score (mean squared probability error)."""
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """Compute Expected Calibration Error (ECE) using equal-width confidence bins."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    correct     = (predictions == y_true).astype(np.float32)
    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences >= lo) & (
            confidences <= hi if i == n_bins - 1 else confidences < hi
        )
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(correct[in_bin]) - np.mean(confidences[in_bin])) * prop
    return float(ece)
```

## 5.2 — Optimiser & LR Schedule

```python
def make_adamw_optimizer(num_train_samples):
    """Build a cosine-decay AdamW optimiser (used for GFNet and ViT)."""
    steps_per_epoch = int(np.ceil(num_train_samples / BATCH_SIZE))
    decay_steps     = max(1, steps_per_epoch * EPOCHS)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=decay_steps,
        alpha=TRAIN_CFG['cosine_alpha'],
    )
    return keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=TRAIN_CFG['weight_decay'],
        clipnorm=TRAIN_CFG['clipnorm'],
    )


def _alexnet_legacy_lr(epoch):
    """Cosine LR schedule oscillating between ALEXNET_LR_MIN and ALEXNET_LR_MAX."""
    if EPOCHS <= 1:
        return ALEXNET_LR_START
    phase        = np.pi * epoch / (EPOCHS - 1)
    cosine_decay = 0.5 * (1.0 + np.cos(phase))
    return float((ALEXNET_LR_MAX - ALEXNET_LR_MIN) * cosine_decay + ALEXNET_LR_MIN)
```

## 5.3 — Excel Report Helpers

Style constants and helper functions for writing DataFrames and plots into a formatted `.xlsx` workbook.

```python
# ── Style constants ───────────────────────────────────────────────────────────
_HDR_FILL   = PatternFill('solid', start_color='1F4E79')
_ALT_FILL   = PatternFill('solid', start_color='D6E4F0')
_HDR_FONT   = Font(name='Arial', bold=True, color='FFFFFF', size=11)
_BODY_FONT  = Font(name='Arial', size=10)
_TITLE_FONT = Font(name='Arial', bold=True, size=13)
_CENTER     = Alignment(horizontal='center', vertical='center', wrap_text=True)
_LEFT       = Alignment(horizontal='left',   vertical='center')
_THIN_SIDE  = Side(style='thin', color='AAAAAA')
_THIN_BORDER = Border(
    left=_THIN_SIDE, right=_THIN_SIDE, top=_THIN_SIDE, bottom=_THIN_SIDE
)


def _style_header_row(ws, row, col_start, col_end):
    """Apply dark-blue header styling to a worksheet row range."""
    for c in range(col_start, col_end + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill      = _HDR_FILL
        cell.font      = _HDR_FONT
        cell.alignment = _CENTER
        cell.border    = _THIN_BORDER


def _style_data_rows(ws, row_start, row_end, col_start, col_end):
    """Apply alternating-row fill and border to a worksheet data range."""
    for r in range(row_start, row_end + 1):
        fill = _ALT_FILL if r % 2 == 0 else PatternFill()
        for c in range(col_start, col_end + 1):
            cell = ws.cell(row=r, column=c)
            cell.font      = _BODY_FONT
            cell.fill      = fill
            cell.alignment = _LEFT
            cell.border    = _THIN_BORDER


def _write_df_to_sheet(ws, df, start_row=1, start_col=1, title=None):
    """Write a DataFrame into a worksheet with headers, alternating rows, and auto-width columns."""
    r = start_row
    if title:
        cell = ws.cell(row=r, column=start_col, value=title)
        cell.font      = _TITLE_FONT
        cell.alignment = _LEFT
        ws.merge_cells(
            start_row=r, start_column=start_col,
            end_row=r, end_column=start_col + len(df.columns) - 1
        )
        r += 1

    # Header row
    for j, col_name in enumerate(df.columns, start=start_col):
        ws.cell(row=r, column=j, value=col_name)
    _style_header_row(ws, r, start_col, start_col + len(df.columns) - 1)
    r += 1

    # Data rows
    data_start = r
    for _, row_data in df.iterrows():
        for j, val in enumerate(row_data, start=start_col):
            ws.cell(row=r, column=j,
                    value=round(float(val), 6)
                    if isinstance(val, (float, np.floating)) else val)
        r += 1
    _style_data_rows(ws, data_start, r - 1, start_col, start_col + len(df.columns) - 1)

    # Auto column width
    for j, col_name in enumerate(df.columns, start=start_col):
        max_len = max(
            len(str(col_name)),
            max((len(str(ws.cell(row=row, column=j).value or ''))
                 for row in range(data_start, r)), default=0)
        )
        ws.column_dimensions[get_column_letter(j)].width = min(max_len + 4, 40)

    return r  # next free row
```

## 5.4 — Excel Report Builder

Assembles the full `CREDIT_Results.xlsx` workbook with four sheets: Training Summary, Evaluation Summary, Confusion Matrices, and Plots.

```python
def create_credit_excel_report(
    results_dir,
    train_summary_df,
    eval_summary_df,
    credit_artifacts_dict,
    plot_paths           # list of (label_str, image_path_str)
):
    """
    Create CREDIT_Results.xlsx with four sheets:
      Sheet 1 — Training Summary
      Sheet 2 — Evaluation Summary
      Sheet 3 — Confusion Matrices (per model, side by side)
      Sheet 4 — Plots (one image per row)
    """
    xlsx_path = Path(results_dir) / 'CREDIT_Results.xlsx'
    wb = Workbook()

    # ── Sheet 1: Training Summary ──────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = 'Training Summary'
    ws1.row_dimensions[1].height = 30
    _write_df_to_sheet(ws1, train_summary_df,
                       start_row=1, title='CREDIT — Training Uncertainty Summary')

    # ── Sheet 2: Evaluation Summary ────────────────────────────────────────────
    ws2 = wb.create_sheet('Evaluation Summary')
    _write_df_to_sheet(ws2, eval_summary_df,
                       start_row=1, title='CREDIT — Full Evaluation Metrics')

    # ── Sheet 3: Confusion Matrices ────────────────────────────────────────────
    ws3 = wb.create_sheet('Confusion Matrices')
    col_cursor = 1
    for mname, art in credit_artifacts_dict.items():
        cm    = art['confusion_matrix']
        rep   = art['report']
        n_cls = cm.shape[0]

        # Title
        cell = ws3.cell(row=1, column=col_cursor,
                        value=f'{mname} — Confusion Matrix')
        cell.font = _TITLE_FONT
        ws3.merge_cells(start_row=1, start_column=col_cursor,
                        end_row=1, end_column=col_cursor + n_cls)

        # Column headers (Predicted)
        ws3.cell(row=2, column=col_cursor,
                 value='True \\ Pred').font = Font(bold=True, name='Arial')
        for j in range(n_cls):
            c = ws3.cell(row=2, column=col_cursor + 1 + j, value=f'C{j+1}')
            c.font      = Font(bold=True, name='Arial', color='FFFFFF')
            c.fill      = _HDR_FILL
            c.alignment = _CENTER

        # Data rows
        for i in range(n_cls):
            ws3.cell(row=3 + i, column=col_cursor,
                     value=f'C{i+1}').font = Font(bold=True, name='Arial')
            for j in range(n_cls):
                val  = int(cm[i, j])
                cell = ws3.cell(row=3 + i, column=col_cursor + 1 + j, value=val)
                cell.alignment = _CENTER
                cell.border    = _THIN_BORDER
                if i == j:  # diagonal → green
                    cell.fill = PatternFill('solid', start_color='C6EFCE')
                    cell.font = Font(name='Arial', bold=True, color='276221')
                else:
                    cell.font = _BODY_FONT

        # Per-class F1 from classification report
        f1_row = 3 + n_cls + 1
        ws3.cell(row=f1_row, column=col_cursor,
                 value='Per-Class F1').font = Font(bold=True, name='Arial')
        for j in range(n_cls):
            key = str(j)
            f1  = rep.get(key, {}).get('f1-score', '')
            ws3.cell(row=f1_row, column=col_cursor + 1 + j,
                     value=round(f1, 4) if f1 != '' else '').font = _BODY_FONT

        col_cursor += n_cls + 3   # gap between model tables

    # ── Sheet 4: Plots ─────────────────────────────────────────────────────────
    ws4 = wb.create_sheet('Plots')
    title_cell = ws4.cell(row=1, column=1, value='CREDIT — All Output Plots')
    title_cell.font = Font(name='Arial', bold=True, size=14)

    img_row = 3
    for label, img_path in plot_paths:
        if not Path(img_path).exists():
            continue
        ws4.cell(row=img_row, column=1,
                 value=label).font = Font(name='Arial', bold=True, size=11)
        img_row += 1

        xl_img = XLImage(img_path)
        orig_w, orig_h = xl_img.width, xl_img.height
        target_w = 900
        scale         = target_w / orig_w if orig_w > 0 else 1
        xl_img.width  = int(orig_w * scale)
        xl_img.height = int(orig_h * scale)

        cell_addr = f'A{img_row}'
        ws4.add_image(xl_img, cell_addr)
        # openpyxl row height is in points; 1pt ≈ 0.75px
        ws4.row_dimensions[img_row].height = xl_img.height * 0.75
        img_row += int(xl_img.height / 15) + 3

    wb.save(xlsx_path)
    print(f'\n✅ Excel report saved → {xlsx_path}')
    return str(xlsx_path)
```

# 6.0 — CREDIT Distillation

Generates soft targets from the pre-trained ensemble teachers, builds a dual-head student for each architecture, and trains it via KL divergence (aleatoric head) + MSE (epistemic head).

## 6.1 — Soft Target Generation

Loads each ensemble teacher, runs inference over the training data, and derives CREDIT targets: `p_star` (normalised per-class minimum — aleatoric proxy) and `delta_p` (per-class prediction spread — epistemic proxy).

```python
def get_ensemble_paths(model_name):
    """Locate ensemble teacher weights using three fallback search patterns."""
    # Primary: standardised location
    pattern = str(ENSEMBLE_DIR / f'{model_name}_ens_*_final.keras')
    paths   = glob.glob(pattern)

    # Fallback 1: sub-folder style
    if not paths:
        pattern = str(
            ENSEMBLE_DIR / f'ensembles_{model_name}' / f'{model_name}_ens_*_final.keras'
        )
        paths = glob.glob(pattern)

    # Fallback 2: old Credit/models location
    if not paths:
        pattern = str(MODEL_DIR / f'{model_name}_ens_*_final.keras')
        paths   = glob.glob(pattern)

    return sorted(paths)


def generate_credit_targets(ensemble_paths, x_data, batch_size=128):
    """
    Run each teacher over x_data and derive CREDIT interval targets.

    Returns:
        p_star_true  : (N, C) normalised per-class minimum  (aleatoric proxy)
        delta_p_true : (N, C) per-class prediction spread   (epistemic proxy)
    """
    all_preds = []
    for path in ensemble_paths:
        print(f'  Loading teacher: {path}')
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        all_preds.append(model.predict(x_data, batch_size=batch_size, verbose=1))
        del model
        tf.keras.backend.clear_session()

    stacked = tf.stack(all_preds, axis=0)   # (M, N, C)
    p_min   = tf.reduce_min(stacked, axis=0)  # (N, C)
    p_max   = tf.reduce_max(stacked, axis=0)  # (N, C)

    delta_p_true = p_max - p_min
    p_star_true  = p_min / (tf.reduce_sum(p_min, axis=-1, keepdims=True) + 1e-12)
    return p_star_true, delta_p_true
```

## 6.2 — CREDIT Student (Dual-Head)

Wraps any single-head base architecture with two output heads: a softmax head for the aleatoric belief (`p_star`) and a sigmoid head for the epistemic spread (`delta_p`).

```python
def build_credit_student(base_builder_func, num_classes):
    """
    Attach CREDIT dual-output heads to any single-head base architecture.

    Args:
        base_builder_func: Zero-argument callable that returns a Keras model.
        num_classes:       Number of target classes.

    Returns:
        A Keras model with outputs [p_star (softmax), delta_p (sigmoid)].
    """
    base_model = base_builder_func()
    features   = base_model.layers[-2].output   # penultimate feature tensor

    p_star  = layers.Dense(num_classes, activation='softmax',  name='p_star' )(features)
    delta_p = layers.Dense(num_classes, activation='sigmoid',  name='delta_p')(features)

    return tf.keras.Model(
        inputs=base_model.input,
        outputs=[p_star, delta_p],
        name='CREDIT_Student'
    )
```

## 6.3 — Model Builder Registry

Centralised registry mapping model names to zero-argument builder lambdas.

```python
model_builders = {
    'AlexNet_CNN': lambda: build_alexnet(
        input_shape, num_classes,
        dropout_rate=DROPOUT_RATE, cfg=ALEXNET_CFG
    ),
    'GFNet': lambda: build_gfnet(
        input_shape, num_classes,
        inner_patch=INNER_PATCH,
        hidden_dim=GFNET_CFG['hidden_dim'],
        num_blocks=GFNET_CFG['num_blocks'],
        mlp_ratio=GFNET_CFG['mlp_ratio'],
        dropout_rate=DROPOUT_RATE
    ),
    'ViT_UNet': lambda: build_vit_unet_singlehead(
        input_shape, num_classes,
        inner_patch=INNER_PATCH,
        projection_dim=VIT_CFG['projection_dim'],
        num_heads=VIT_CFG['num_heads'],
        transformer_layers=VIT_CFG['transformer_layers'],
        mlp_multiplier=VIT_CFG['mlp_multiplier'],
        dropout_rate=DROPOUT_RATE,
        head_units=tuple(VIT_CFG['head_units'])
    ),
}
```

## 6.4 — Training Loop

For each architecture: locate ensemble teachers → generate soft targets → build `tf.data` pipelines → build and compile the CREDIT student → train with `ModelCheckpoint` → compute a quick post-training uncertainty summary.

```python
train_results = []

for model_name, builder in model_builders.items():
    print(f"\n{'='*25} CREDIT Distillation: {model_name} {'='*25}")

    # ── 1. Locate ensemble teachers ────────────────────────────────────────────
    ensemble_paths = get_ensemble_paths(model_name)
    if len(ensemble_paths) == 0:
        print('  No ensemble models found — skipping.')
        continue
    if len(ensemble_paths) < 5:
        print(f'  Warning: only {len(ensemble_paths)} ensemble models found (expected 5).')

    # ── 2. Select correct data split ───────────────────────────────────────────
    x_tr, x_te = (
        (x_train_alex, x_test_alex) if model_name == 'AlexNet_CNN'
        else (x_train, x_test)
    )

    # ── 3. Generate soft targets ───────────────────────────────────────────────
    print('  Generating training targets...')
    p_star_tr, delta_p_tr = generate_credit_targets(ensemble_paths, x_tr)

    print('  Generating test targets...')
    p_star_te, delta_p_te = generate_credit_targets(ensemble_paths, x_te)

    # ── 4. Build tf.data pipelines ─────────────────────────────────────────────
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((x_tr, (p_star_tr, delta_p_tr)))
        .shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset
        .from_tensor_slices((x_te, (p_star_te, delta_p_te)))
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )

    # ── 5. Build & compile student ─────────────────────────────────────────────
    tf.keras.backend.clear_session()
    student = build_credit_student(builder, num_classes)

    optimizer = (
        keras.optimizers.Adagrad(learning_rate=ALEXNET_LR_START)
        if model_name == 'AlexNet_CNN'
        else keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    )

    student.compile(
        optimizer=optimizer,
        loss={
            'p_star':  tf.keras.losses.KLDivergence(),
            'delta_p': tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={'p_star': 1.0, 'delta_p': 0.5},   # lambda = 0.5
    )

    # ── 6. Train ───────────────────────────────────────────────────────────────
    best_path = MODEL_DIR / f'{model_name}_CREDIT_best.keras'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path), monitor='val_loss',
            mode='min', save_best_only=True, verbose=1
        )
    ]

    t0 = time.perf_counter()
    student.fit(
        train_ds, validation_data=test_ds,
        epochs=EPOCHS, callbacks=callbacks, verbose=1
    )
    train_time = time.perf_counter() - t0

    # ── 7. Post-training uncertainty summary ───────────────────────────────────
    student.load_weights(best_path)
    p_star_pred, delta_p_pred = student.predict(x_te, batch_size=BATCH_SIZE)

    au = -np.sum(p_star_pred * np.log(p_star_pred + 1e-12), axis=-1)
    eu =  np.mean(delta_p_pred, axis=-1)
    tu =  au + eu

    train_results.append({
        'Model':          model_name,
        'Mean_AU':        np.mean(au),
        'Mean_EU':        np.mean(eu),
        'Mean_TU':        np.mean(tu),
        'Train_Time_sec': train_time,
    })

# ── Save training uncertainty summary ─────────────────────────────────────────
summary_df   = pd.DataFrame(train_results)
summary_path = RESULTS_DIR / 'credit_train_uncertainty_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f'\nSaved training summary → {summary_path}')
print(summary_df)

# Stash for the final Excel report
_credit_train_summary_df = summary_df.copy()
```

# 7.0 — Evaluation

Loads the best saved CREDIT student weights for each architecture and computes a full suite of classification and calibration metrics — no re-training.

## 7.1 — Load, Predict & Compute Metrics

```python
eval_results     = []
credit_artifacts = {}

for model_name, builder in model_builders.items():
    best_path = MODEL_DIR / f'{model_name}_CREDIT_best.keras'

    if not best_path.exists():
        print(f'  [{model_name}] No saved weights at {best_path} — skipping.')
        continue

    print(f"\n{'='*25} Evaluating: {model_name} {'='*25}")
    tf.keras.backend.clear_session()

    student = build_credit_student(builder, num_classes)
    student.load_weights(best_path)

    # Select the correct test split for this architecture
    if model_name == 'AlexNet_CNN':
        x_te, y_te_true, y_te_cat = x_test_alex, y_test_alex, y_test_alex_cat
    else:
        x_te, y_te_true, y_te_cat = x_test, y_test, y_test_cat

    p_star_pred, delta_p_pred = student.predict(x_te, batch_size=BATCH_SIZE)

    au = -np.sum(p_star_pred * np.log(p_star_pred + 1e-12), axis=-1)
    eu =  np.mean(delta_p_pred, axis=-1)
    tu =  au + eu

    y_pred = np.argmax(p_star_pred, axis=-1)

    credit_artifacts[model_name] = {
        'confusion_matrix': confusion_matrix(y_te_true, y_pred),
        'report':           classification_report(
            y_te_true, y_pred, output_dict=True, zero_division=0
        ),
    }

    eval_results.append({
        'Model':         model_name,
        'Test_Accuracy': float(accuracy_score(y_te_true, y_pred)),
        'Macro_F1':      float(f1_score(y_te_true, y_pred, average='macro')),
        'Cohen_Kappa':   float(cohen_kappa_score(y_te_true, y_pred)),
        'Test_NLL':      float(log_loss(y_te_true, p_star_pred,
                                        labels=np.arange(num_classes))),
        'Test_Brier':    float(multiclass_brier_score(y_te_cat, p_star_pred)),
        'Test_ECE':      float(expected_calibration_error(
            y_te_true, p_star_pred, n_bins=15)),
        'Mean_AU':       float(np.mean(au)),
        'Mean_EU':       float(np.mean(eu)),
        'Mean_TU':       float(np.mean(tu)),
    })

eval_df = pd.DataFrame(eval_results)
eval_df.to_csv(RESULTS_DIR / 'credit_evaluation_summary.csv', index=False)
print('\n--- CREDIT Evaluation Summary ---')
print(eval_df.to_string(index=False))

# Stash for the final Excel report
_credit_eval_df        = eval_df.copy()
_credit_artifacts_dict = credit_artifacts
```

## 7.2 — Confusion Matrices

Renders and saves a side-by-side confusion matrix heatmap for every evaluated model.

```python
if credit_artifacts:
    class_ticks = [str(i + 1) for i in range(num_classes)]
    n_models    = len(credit_artifacts)
    fig, axes   = plt.subplots(1, n_models, figsize=(7 * n_models, 5.5))
    if n_models == 1:
        axes = [axes]

    for ax, (mname, art) in zip(axes, credit_artifacts.items()):
        sns.heatmap(
            art['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=class_ticks, yticklabels=class_ticks, cbar=False, ax=ax
        )
        ax.set_title(f'{mname} — CREDIT\nConfusion Matrix', fontsize=13)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')

    plt.tight_layout()
    save_path = RESULTS_DIR / 'credit_confusion_matrices.png'
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    print(f'Saved → {save_path}')
    plt.show()
```

# 8.0 — Spatial Uncertainty Mapping

Runs full-scene inference for each CREDIT student and produces a 3×4 figure showing: the base prediction map, binary certain/uncertain masks, grey-overlay maps, and per-class pixel-count bar charts — all for the aleatoric, epistemic, and total uncertainty channels.

## 8.1 — Extract Full-Scene Patch Array

```python
print('Extracting full-scene patch array (all H×W pixels)...')
pad   = PATCH_SIZE // 2
x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')

scene_patches = np.empty((H * W, PATCH_SIZE, PATCH_SIZE, B), dtype=np.float32)
for idx, (r, c) in enumerate(
    [(r, c) for r in range(H) for c in range(W)]
):
    scene_patches[idx] = x_pad[r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]

print(f'Scene patch tensor: {scene_patches.shape}')
```

## 8.2 — Spatial Map Generator

Defines the function that produces and saves the 3×4 uncertainty figure for a given student model.

```python
CLASS_COLORS = [
    '#0000FF', '#00FF00', '#FF0000', '#00FFFF',
    '#FF00FF', '#FFFF00', '#A52A2A', '#FFA500',
    '#7FFF00', '#8A2BE2',
]


def generate_spatial_credit_maps(
    model_name, student_model, scene_pixels,
    H, W, au_thresh=0.5, eu_thresh=0.2, tu_thresh=0.7
):
    """
    Run full-scene inference and save a 3×4 spatial uncertainty figure.

    Panel layout (rows × cols):
      Row 0: Base prediction | AU binary | EU binary | TU binary
      Row 1: AU grey overlay | EU grey overlay | TU grey overlay | (blank)
      Row 2: AU bar chart   | EU bar chart   | TU bar chart   | (blank)
    """
    print(f'  Inferring full scene for {model_name}...')
    p_star_scene, delta_p_scene = student_model.predict(
        scene_pixels, batch_size=2048, verbose=1
    )
    p_star_scene = np.clip(p_star_scene, 1e-12, 1.0)
    n_cls = p_star_scene.shape[-1]

    au_scene = -np.sum(p_star_scene * np.log(p_star_scene), axis=-1)
    eu_scene =  np.mean(delta_p_scene, axis=-1)
    tu_scene =  au_scene + eu_scene

    pred_map = np.argmax(p_star_scene, axis=-1).reshape(H, W)
    au_mask  = (au_scene.reshape(H, W) > au_thresh).astype(int)
    eu_mask  = (eu_scene.reshape(H, W) > eu_thresh).astype(int)
    tu_mask  = (tu_scene.reshape(H, W) > tu_thresh).astype(int)

    combined_au = np.where(au_mask == 1, n_cls, pred_map)
    combined_eu = np.where(eu_mask == 1, n_cls, pred_map)
    combined_tu = np.where(tu_mask == 1, n_cls, pred_map)

    cmap_base   = ListedColormap(CLASS_COLORS[:n_cls])
    cmap_unc    = ListedColormap(CLASS_COLORS[:n_cls] + ['#808080'])
    cmap_binary = ListedColormap(['#FFFF00', '#001F3F'])

    bar_lbls = [f'Class {i}' for i in range(n_cls)] + ['Uncertain']
    bar_cols = CLASS_COLORS[:n_cls] + ['#808080']

    fig, axes = plt.subplots(3, 4, figsize=(38, 26))
    fig.suptitle(
        f'{model_name} — CREDIT Uncertainty Maps (Absolute Thresholds)',
        fontsize=24, fontweight='bold', y=0.99
    )

    # ── Row 0: Base prediction + 3 binary maps ─────────────────────────────────
    axes[0, 0].imshow(pred_map, cmap=cmap_base, vmin=0, vmax=n_cls - 1)
    axes[0, 0].set_title('Base Prediction Map', fontsize=15)
    axes[0, 0].axis('off')

    binary_specs = [
        (axes[0, 1], au_mask, f'Aleatoric (AU > {au_thresh})'),
        (axes[0, 2], eu_mask, f'Epistemic (EU > {eu_thresh})'),
        (axes[0, 3], tu_mask, f'Total     (TU > {tu_thresh})'),
    ]
    for ax, mask, label in binary_specs:
        ax.imshow(mask, cmap=cmap_binary, vmin=0, vmax=1)
        ax.set_title(f'Certain vs Uncertain\n{label}', fontsize=15, pad=10)
        ax.axis('off')
        ax.legend(
            handles=[
                Patch(facecolor='#FFFF00', label='Certain'),
                Patch(facecolor='#001F3F', label='Uncertain'),
            ],
            loc='upper left',
            bbox_to_anchor=(0.0, -0.02),
            borderaxespad=0,
            fontsize=11,
            framealpha=0.9,
            ncol=2
        )

    # ── Row 1: Grey overlay maps ────────────────────────────────────────────────
    overlay_specs = [
        (axes[1, 0], combined_au, f'Aleatoric (AU > {au_thresh})'),
        (axes[1, 1], combined_eu, f'Epistemic (EU > {eu_thresh})'),
        (axes[1, 2], combined_tu, f'Total     (TU > {tu_thresh})'),
    ]
    for ax, combined, label in overlay_specs:
        ax.imshow(combined, cmap=cmap_unc, vmin=0, vmax=n_cls)
        ax.set_title(f'Grey Overlay — {label}', fontsize=15, pad=10)
        ax.axis('off')
    axes[1, 3].axis('off')

    # ── Row 2: Pixel-count bar charts ───────────────────────────────────────────
    bar_specs = [
        (axes[2, 0], combined_au, f'Aleatoric (AU > {au_thresh})'),
        (axes[2, 1], combined_eu, f'Epistemic (EU > {eu_thresh})'),
        (axes[2, 2], combined_tu, f'Total     (TU > {tu_thresh})'),
    ]
    for ax, combined, label in bar_specs:
        uniq, cnt = np.unique(combined, return_counts=True)
        c_dict    = {int(k): int(v) for k, v in zip(uniq, cnt)}
        bar_vals  = [c_dict.get(i, 0) for i in range(n_cls + 1)]
        ax.bar(bar_lbls, bar_vals, color=bar_cols, edgecolor='black')
        ax.set_title(f'Pixel Counts — {label}', fontsize=15, pad=10)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.set_ylabel('Pixel Count', fontsize=12)
        max_val = max(bar_vals, default=1)
        for i, v in enumerate(bar_vals):
            ax.text(i, v + max_val * 0.01,
                    f'{v:,}', ha='center', va='bottom',
                    fontweight='bold', fontsize=9)
        ax.set_ylim(0, max_val * 1.12)
    axes[2, 3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_path = RESULTS_DIR / f'{model_name}_CREDIT_spatial_maps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'  Saved → {save_path}')
    plt.show()
```

## 8.3 — Run Spatial Mapping for All Models

```python
for model_name, builder in model_builders.items():
    best_path = MODEL_DIR / f'{model_name}_CREDIT_best.keras'
    if not best_path.exists():
        print(f'[{model_name}] No weights found — skipping spatial maps.')
        continue

    tf.keras.backend.clear_session()
    student = build_credit_student(builder, num_classes)
    student.load_weights(best_path)

    generate_spatial_credit_maps(
        model_name, student, scene_patches, H, W,
        au_thresh=0.5, eu_thresh=0.2, tu_thresh=0.7
    )
```

# 9.0 — Excel Report Export

Collects all saved plot paths (confusion matrices, spatial uncertainty maps, and any training-curve PNGs) and writes the final `CREDIT_Results.xlsx` workbook.

```python
# ── Collect all plot paths ─────────────────────────────────────────────────────
_plot_entries = []

# Confusion matrix image
_cm_path = str(RESULTS_DIR / 'credit_confusion_matrices.png')
_plot_entries.append(('Confusion Matrices (all models)', _cm_path))

# Per-model spatial uncertainty maps
for _mname in model_builders:
    _sp_path = str(RESULTS_DIR / f'{_mname}_CREDIT_spatial_maps.png')
    _plot_entries.append((f'{_mname} — Spatial Uncertainty Maps', _sp_path))

# Any additional training-curve PNGs saved to PLOT_DIR
for _p in sorted(PLOT_DIR.glob('*.png')):
    _plot_entries.append((_p.stem.replace('_', ' ').title(), str(_p)))

# ── Write the workbook ─────────────────────────────────────────────────────────
create_credit_excel_report(
    results_dir           = RESULTS_DIR,
    train_summary_df      = _credit_train_summary_df,
    eval_summary_df       = _credit_eval_df,
    credit_artifacts_dict = _credit_artifacts_dict,
    plot_paths            = _plot_entries,
)
```

