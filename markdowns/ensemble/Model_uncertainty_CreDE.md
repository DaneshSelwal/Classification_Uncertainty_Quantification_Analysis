# 1.0 — Setup & Imports

Mount Google Drive and import all required standard-library, third-party, and framework packages.

## 1.1 — Drive Mount

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 1.2 — Library Imports

Groups: standard library → numerical/data → visualisation → deep learning.

```python
# Standard library
import os
import gc
import glob
from pathlib import Path

# Numerical & data
import numpy as np
import pandas as pd
import seaborn as sns

# Visualisation
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Deep learning
import tensorflow as tf
from tensorflow.keras import layers

# Metrics & Splitting
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

sns.set_style("whitegrid")
print("TensorFlow:", tf.__version__)
```

# 2.0 — Configuration

All tunable constants and filesystem paths live here. Edit this section before running
the notebook on a new dataset or experiment.

## 2.1 — Constants & Paths

```python
# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Filesystem
ROOT_DIR = Path("/content/drive/My Drive/Classification")
MODEL_DIR = Path("/content/drive/My Drive/Classification/ensemble/models/ensembles")
DATA_DIR  = Path("/content/drive/My Drive/Classification/data")
DATA_FILE  = DATA_DIR / "data.csv"
LABEL_FILE = DATA_DIR / "ref.csv"

# Scene geometry
H, W, B = 330, 307, 6   # height, width, spectral bands
PATCH_SIZE = 9

# Uncertainty thresholds used for masking
AU_THRESH = 0.5
EU_THRESH = 0.2
TU_THRESH = 0.7
```

## 2.2 — Output Directory Setup

Creates a dedicated subfolder inside `MODEL_DIR` to hold all CreDE outputs (PNGs and CSV).

```python
CREDE_OUT_DIR = ROOT_DIR/ "ensemble" / "results"
CREDE_OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"CreDE output directory: {CREDE_OUT_DIR}")
```

# 3.0 — Data Loading & Preprocessing

Reads the multispectral image and label rasters from CSV, normalises each band
independently to [0, 1], then extracts every spatial patch needed for full-scene inference.

## 3.1 — Multispectral Data Loader

```python
def load_multispectral_6band(data_path, label_path, height, width, bands):
    """
    Load a flat-CSV multispectral image and its reference label raster.

    Each band is min-max normalised independently to [0, 1].

    Parameters
    ----------
    data_path  : path-like  Path to the pixel-value CSV.
    label_path : path-like  Path to the integer label CSV.
    height, width, bands : int  Scene dimensions.

    Returns
    -------
    x_norm : np.ndarray  Shape (H, W, B), dtype float32, values in [0, 1].
    y      : np.ndarray  Shape (H, W),    dtype int32.
    """
    x = pd.read_csv(data_path).to_numpy(dtype=np.float32).reshape(height, width, bands)
    y = pd.read_csv(label_path).to_numpy(dtype=np.int32).reshape(height, width)

    x_norm = np.empty_like(x, dtype=np.float32)
    for b in range(bands):
        band = x[:, :, b]
        band_min, band_max = np.min(band), np.max(band)
        x_norm[:, :, b] = (band - band_min) / max(band_max - band_min, 1e-8)

    return x_norm, y


print("Loading multispectral data...")
x_img, y_img = load_multispectral_6band(DATA_FILE, LABEL_FILE, H, W, B)
num_classes = int(np.unique(y_img[y_img > 0]).size)
print(f"Scene shape: {x_img.shape}  |  Classes: {num_classes}")
```

## 3.2 — Spatial Patch Extraction

Pads the scene and slides a `PATCH_SIZE × PATCH_SIZE` window over every pixel,
producing a flat array of patches ready for batch inference.

```python
print("Extracting all spatial patches for full-scene inference...")

pad   = PATCH_SIZE // 2
x_pad = np.pad(x_img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

scene_pixels_scaled = np.empty((H * W, PATCH_SIZE, PATCH_SIZE, B), dtype=np.float32)

idx = 0
for r in range(H):
    for c in range(W):
        scene_pixels_scaled[idx] = x_pad[r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
        idx += 1

print(f"scene_pixels_scaled shape: {scene_pixels_scaled.shape}")
```

```python
# -----------------------------
# 3.3 — Labeled Patch Extraction & Splitting
# -----------------------------
print("Extracting labeled patches for metric evaluation...")

def extract_labeled_patches(x, y, patch_size=9):
    pad = patch_size // 2
    x_pad = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    coords = np.argwhere(y > 0)
    patches = np.empty((coords.shape[0], patch_size, patch_size, x.shape[-1]), dtype=np.float32)
    labels = np.empty((coords.shape[0],), dtype=np.int32)
    for i, (r, c) in enumerate(coords):
        patches[i] = x_pad[r:r + patch_size, c:c + patch_size, :]
        labels[i] = int(y[r, c]) - 1
    return patches, labels, coords

X, y_labels, coords = extract_labeled_patches(x_img, y_img, PATCH_SIZE)

# Standard Split
TRAIN_PERCENT = 0.75
x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y_labels, train_size=TRAIN_PERCENT, random_state=SEED, stratify=y_labels
)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"Standard Test Set Extracted: {x_test.shape}")
```

# 4.0 — Custom Keras Layers

Registers the four custom `tf.keras` layers required to deserialise the saved `.keras` models.
All four are decorated with `@register_keras_serializable()` so they survive `load_model`.
**Do not alter any layer internals** — these must match the definitions used during training.

## 4.1 — PatchExtractor

```python
@tf.keras.utils.register_keras_serializable()
class PatchExtractor(layers.Layer):
    """Extracts non-overlapping image patches and flattens them into a sequence."""

    def __init__(self, patch_size=3, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        batch, patch_dim = tf.shape(images)[0], tf.shape(patches)[-1]
        num_patches = tf.shape(patches)[1] * tf.shape(patches)[2]
        return tf.reshape(patches, [batch, num_patches, patch_dim])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg
```

## 4.2 — PatchPositionEncoder

```python
@tf.keras.utils.register_keras_serializable()
class PatchPositionEncoder(layers.Layer):
    """Projects patches to `projection_dim` and adds learned positional embeddings."""

    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches    = num_patches
        self.projection_dim = projection_dim
        self.projection         = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return cfg
```

## 4.3 — GlobalFilterLayer

```python
@tf.keras.utils.register_keras_serializable()
class GlobalFilterLayer(layers.Layer):
    """
    Applies a learnable complex-valued filter in the 2-D frequency domain.

    The token sequence is reshaped to a square grid, transformed with FFT2D,
    element-wise multiplied by trainable complex weights, then inverse-transformed
    back to the spatial sequence.
    """

    def __init__(self, token_side, **kwargs):
        super().__init__(**kwargs)
        self.token_side = token_side

    def build(self, input_shape):
        channels = int(input_shape[-1])
        self.w_real = self.add_weight(
            name="w_real", shape=(self.token_side, self.token_side, channels),
            initializer="glorot_uniform", trainable=True
        )
        self.w_imag = self.add_weight(
            name="w_imag", shape=(self.token_side, self.token_side, channels),
            initializer="zeros", trainable=True
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
        cfg.update({"token_side": self.token_side})
        return cfg
```

## 4.4 — PatchEncoderWithCLS

```python
@tf.keras.utils.register_keras_serializable()
class PatchEncoderWithCLS(layers.Layer):
    """
    Projects patches, prepends a learnable [CLS] token, and adds positional embeddings.

    Used by the ViT-UNet architecture; the CLS token aggregates global context
    for the final classification head.
    """

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
            name="cls_token", shape=(1, 1, self.projection_dim),
            initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, patches):
        batch      = tf.shape(patches)[0]
        patch_proj = self.projection(patches)
        cls_tokens = tf.repeat(self.cls_token, repeats=batch, axis=0)  # broadcast CLS to batch
        x          = tf.concat([cls_tokens, patch_proj], axis=1)
        positions  = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        return x + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return cfg
```

## 4.5 — Custom Objects Registry

Collects all custom layer classes into a single dict passed to `tf.keras.models.load_model`.

```python
CUSTOM_OBJECTS = {
    "PatchExtractor":      PatchExtractor,
    "PatchPositionEncoder": PatchPositionEncoder,
    "GlobalFilterLayer":   GlobalFilterLayer,
    "PatchEncoderWithCLS": PatchEncoderWithCLS,
}
```

# 5.0 — Core Ensemble Functions

Functions for discovering saved ensemble checkpoints and running the CreDE
credal-bound computation across all ensemble members.

## 5.1 — Ensemble Path Discovery

```python
def get_ensemble_paths(model_name):
    """
    Locate all saved `.keras` checkpoint files for a given architecture.

    Searches `MODEL_DIR/ensembles_<model_name>/` first, then falls back to
    `MODEL_DIR/ensembles_old/` if the primary directory is empty.

    Parameters
    ----------
    model_name : str  Architecture identifier (e.g. 'AlexNet_CNN').

    Returns
    -------
    list[str]  Absolute paths to matching checkpoint files.
    """
    primary_pattern  = str(MODEL_DIR / f"{model_name}_ens_*_final.keras")
    paths = glob.glob(primary_pattern)

    if not paths:  # Fallback to legacy directory
        fallback_pattern = str(MODEL_DIR / "ensembles_old" / f"{model_name}_ens_*_final.keras")
        paths = glob.glob(fallback_pattern)

    return paths
```

## 5.2 — Homogeneous Ensemble Evaluation (CreDE)

Loads each ensemble member sequentially, predicts on the full scene,
then computes credal bounds (p_min / p_max) to derive aleatoric uncertainty (AU),
epistemic uncertainty (EU), and total uncertainty (TU).
Each model is deleted immediately after prediction to minimise peak RAM usage.

```python
def evaluate_homogeneous_ensemble(model_paths, input_data, batch_size=2048):
    """
    Run CreDE inference over a list of ensemble checkpoints.

    For each model: predict → stack → compute credal bounds → derive uncertainty measures.

    Parameters
    ----------
    model_paths : list[str]   Paths to `.keras` checkpoint files.
    input_data  : np.ndarray  Full-scene patch array, shape (N, P, P, B).
    batch_size  : int         Inference batch size (default 2048).

    Returns
    -------
    pred_class : np.ndarray  Shape (N,)   — argmax of credal mean probabilities.
    p_star     : np.ndarray  Shape (N, C) — normalised lower credal probabilities.
    au         : np.ndarray  Shape (N,)   — aleatoric uncertainty (entropy of p_star).
    eu         : np.ndarray  Shape (N,)   — epistemic uncertainty (mean credal spread).
    tu         : np.ndarray  Shape (N,)   — total uncertainty (au + eu).
    """
    all_preds = []

    for path in model_paths:
        print(f"  -> Loading & predicting: {Path(path).name}")
        model = tf.keras.models.load_model(
            path, compile=False, custom_objects=CUSTOM_OBJECTS, safe_mode=False
        )
        preds = model.predict(input_data, batch_size=batch_size, verbose=1)
        all_preds.append(preds)

        # Delete model immediately to free RAM before loading the next member
        del model
        tf.keras.backend.clear_session()
        gc.collect()

    print("  -> Computing credal bounds...")
    stacked_preds = tf.stack(all_preds, axis=0)  # (M, N, C)
    p_min = tf.reduce_min(stacked_preds, axis=0)
    p_max = tf.reduce_max(stacked_preds, axis=0)

    delta_p = p_max - p_min

    # Normalise lower credal probabilities to form p_star
    p_star = p_min / (tf.reduce_sum(p_min, axis=-1, keepdims=True) + 1e-12)
    p_star = np.clip(p_star, 1e-12, 1.0)

    # Uncertainty decomposition
    au = -np.sum(p_star * np.log(p_star), axis=-1)   # aleatoric: entropy
    eu = np.mean(delta_p, axis=-1)                    # epistemic: mean credal spread
    tu = au + eu                                      # total uncertainty

    pred_class = np.argmax(p_star, axis=-1)

    # Coerce any TF tensors to numpy
    to_np = lambda t: t.numpy() if hasattr(t, 'numpy') else t
    return to_np(pred_class), to_np(p_star), to_np(au), to_np(eu), to_np(tu)
```

# 6.0 — Visualisation Engine

Generates and saves a standardised 6-panel figure for each architecture:
base prediction map, binary certain/uncertain mask, pixel-count bar chart,
and three grey-overlay maps (AU, EU, TU).

## 6.1 — 6-Panel Spatial Mapping Function

```python
# Shared colour palette (up to 10 classes + grey for uncertain pixels)
CLASS_COLORS = [
    '#0000FF', '#00FF00', '#FF0000', '#00FFFF', '#FF00FF',
    '#FFFF00', '#A52A2A', '#FFA500', '#7FFF00', '#8A2BE2'
]

def generate_spatial_crede_maps(
    model_name, pred_class_scene, p_star_scene, au_scene, eu_scene, tu_scene,
    H=330, W=307, au_thresh=0.5, eu_thresh=0.2, tu_thresh=0.7
):
    """Produce and save a 3x4 spatial uncertainty figure for CreDE."""
    print(f'  -> Generating 3x4 spatial maps for {model_name}...')
    n_cls = p_star_scene.shape[-1]

    au_map         = au_scene.reshape((H, W))
    eu_map         = eu_scene.reshape((H, W))
    tu_map         = tu_scene.reshape((H, W))
    pred_map       = pred_class_scene.reshape((H, W))

    au_mask = (au_map > au_thresh).astype(int)
    eu_mask = (eu_map > eu_thresh).astype(int)
    tu_mask = (tu_map > tu_thresh).astype(int)

    combined_au = np.where(au_mask == 1, n_cls, pred_map)
    combined_eu = np.where(eu_mask == 1, n_cls, pred_map)
    combined_tu = np.where(tu_mask == 1, n_cls, pred_map)

    cmap_base   = ListedColormap(CLASS_COLORS[:n_cls])
    cmap_unc    = ListedColormap(CLASS_COLORS[:n_cls] + ['#808080'])
    cmap_binary = ListedColormap(['#FFFF00', '#001F3F'])

    bar_lbls = [f'Class {i}' for i in range(n_cls)] + ['Uncertain']
    bar_cols = CLASS_COLORS[:n_cls] + ['#808080']

    fig, axes = plt.subplots(3, 4, figsize=(38, 26))
    fig.suptitle(f'{model_name} — CreDE Uncertainty Maps (Absolute Thresholds)',
                 fontsize=24, fontweight='bold', y=0.99)

    # ── Row 0: Base prediction + 3 binary maps
    axes[0, 0].imshow(pred_map, cmap=cmap_base, vmin=0, vmax=n_cls - 1)
    axes[0, 0].set_title('Base Prediction Map', fontsize=15)
    axes[0, 0].axis('off')

    binary_specs = [
        (axes[0, 1], au_mask, f'Aleatoric (AU > {au_thresh})'),
        (axes[0, 2], eu_mask, f'Epistemic (EU > {eu_thresh})'),
        (axes[0, 3], tu_mask, f'Total      (TU > {tu_thresh})'),
    ]
    for ax, mask, label in binary_specs:
        ax.imshow(mask, cmap=cmap_binary, vmin=0, vmax=1)
        ax.set_title(f'Certain vs Uncertain\n{label}', fontsize=15, pad=10)
        ax.axis('off')
        ax.legend(
            handles=[Patch(facecolor='#FFFF00', label='Certain'),
                     Patch(facecolor='#001F3F', label='Uncertain')],
            loc='upper left', bbox_to_anchor=(0.0, -0.02), borderaxespad=0,
            fontsize=11, framealpha=0.9, ncol=2
        )

    # ── Row 1: Grey overlay maps (AU, EU, TU) + blank
    overlay_specs = [
        (axes[1, 0], combined_au, f'Aleatoric (AU > {au_thresh})'),
        (axes[1, 1], combined_eu, f'Epistemic (EU > {eu_thresh})'),
        (axes[1, 2], combined_tu, f'Total      (TU > {tu_thresh})'),
    ]
    for ax, combined, label in overlay_specs:
        ax.imshow(combined, cmap=cmap_unc, vmin=0, vmax=n_cls)
        ax.set_title(f'Grey Overlay — {label}', fontsize=15, pad=10)
        ax.axis('off')

    axes[1, 3].axis('off')

    # ── Row 2: Bar charts (AU, EU, TU) + blank
    bar_specs = [
        (axes[2, 0], combined_au, f'Aleatoric (AU > {au_thresh})'),
        (axes[2, 1], combined_eu, f'Epistemic (EU > {eu_thresh})'),
        (axes[2, 2], combined_tu, f'Total      (TU > {tu_thresh})'),
    ]
    for ax, combined, label in bar_specs:
        uniq, cnt = np.unique(combined, return_counts=True)
        c_dict   = {int(k): int(v) for k, v in zip(uniq, cnt)}
        bar_vals = [c_dict.get(i, 0) for i in range(n_cls + 1)]
        ax.bar(bar_lbls, bar_vals, color=bar_cols, edgecolor='black')
        ax.set_title(f'Pixel Counts — {label}', fontsize=15, pad=10)
        ax.tick_params(axis='x', rotation=45, labelsize=11)
        ax.set_ylabel('Pixel Count', fontsize=12)
        max_val = max(bar_vals, default=1)
        for i, v in enumerate(bar_vals):
            ax.text(i, v + max_val * 0.01,
                    f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.set_ylim(0, max_val * 1.12)

    axes[2, 3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_path = CREDE_OUT_DIR / f'{model_name}_CreDE_spatial_maps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'  -> Saved: {save_path}')
    return str(save_path)
```

# 7.0 — Main Execution

Iterates over each target architecture, runs CreDE inference, generates spatial maps,
and accumulates per-architecture metrics into a master summary CSV.
RAM is wiped aggressively after every architecture to prevent OOM errors.

```python
# ── Excel Export Helpers ──────────────────────────────────────────────────────
_HDR_FILL   = PatternFill('solid', start_color='1F4E79')
_ALT_FILL   = PatternFill('solid', start_color='D6E4F0')
_HDR_FONT   = Font(name='Arial', bold=True, color='FFFFFF', size=11)
_BODY_FONT  = Font(name='Arial', size=10)
_TITLE_FONT = Font(name='Arial', bold=True, size=13)
_CENTER     = Alignment(horizontal='center', vertical='center', wrap_text=True)
_LEFT       = Alignment(horizontal='left',   vertical='center')
_THIN_SIDE  = Side(style='thin', color='AAAAAA')
_THIN_BORDER= Border(left=_THIN_SIDE, right=_THIN_SIDE,
                     top=_THIN_SIDE,  bottom=_THIN_SIDE)

def _style_header_row(ws, row, col_start, col_end):
    for c in range(col_start, col_end + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill   = _HDR_FILL
        cell.font   = _HDR_FONT
        cell.alignment = _CENTER
        cell.border = _THIN_BORDER

def _style_data_rows(ws, row_start, row_end, col_start, col_end):
    for r in range(row_start, row_end + 1):
        fill = _ALT_FILL if r % 2 == 0 else PatternFill()
        for c in range(col_start, col_end + 1):
            cell = ws.cell(row=r, column=c)
            cell.font      = _BODY_FONT
            cell.fill      = fill
            cell.alignment = _LEFT
            cell.border    = _THIN_BORDER

def _write_df_to_sheet(ws, df, start_row=1, start_col=1, title=None):
    r = start_row
    if title:
        cell = ws.cell(row=r, column=start_col, value=title)
        cell.font      = _TITLE_FONT
        cell.alignment = _LEFT
        ws.merge_cells(start_row=r, start_column=start_col,
                       end_row=r, end_column=start_col + len(df.columns) - 1)
        r += 1

    for j, col_name in enumerate(df.columns, start=start_col):
        ws.cell(row=r, column=j, value=col_name)
    _style_header_row(ws, r, start_col, start_col + len(df.columns) - 1)
    r += 1

    data_start = r
    for _, row_data in df.iterrows():
        for j, val in enumerate(row_data, start=start_col):
            ws.cell(row=r, column=j, value=round(float(val), 6)
                    if isinstance(val, (float, np.floating)) else val)
        r += 1
    _style_data_rows(ws, data_start, r - 1, start_col, start_col + len(df.columns) - 1)

    for j, col_name in enumerate(df.columns, start=start_col):
        max_len = max(len(str(col_name)),
                      max((len(str(ws.cell(row=row, column=j).value or ''))
                           for row in range(data_start, r)), default=0))
        ws.column_dimensions[get_column_letter(j)].width = min(max_len + 4, 40)
    return r

def create_crede_excel_report(out_dir, summary_df, plot_paths):
    """Auto-creates CreDE_Results.xlsx with Summary and Plots."""
    xlsx_path = Path(out_dir) / 'CreDE_Results.xlsx'
    wb = Workbook()

    # ── Sheet 1: Summary ──────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = 'CreDE Summary'
    ws1.row_dimensions[1].height = 30
    _write_df_to_sheet(ws1, summary_df, start_row=1, title='CreDE — Inference Summary')

    # ── Sheet 2: Plots ────────────────────────────────────────────────────────
    if plot_paths:
        ws2 = wb.create_sheet('Plots')
        title_cell = ws2.cell(row=1, column=1, value='CreDE — Spatial Uncertainty Maps')
        title_cell.font = Font(name='Arial', bold=True, size=14)

        img_row = 3
        for label, img_path in plot_paths:
            if not Path(img_path).exists():
                continue
            ws2.cell(row=img_row, column=1, value=label).font = Font(name='Arial', bold=True, size=11)
            img_row += 1

            xl_img = XLImage(img_path)
            orig_w, orig_h = xl_img.width, xl_img.height
            target_w = 900
            scale    = target_w / orig_w if orig_w > 0 else 1
            xl_img.width  = int(orig_w * scale)
            xl_img.height = int(orig_h * scale)

            ws2.add_image(xl_img, f'A{img_row}')
            ws2.row_dimensions[img_row].height = xl_img.height * 0.75
            img_row += int(xl_img.height / 15) + 3

    wb.save(xlsx_path)
    print(f'\n✅ Excel report saved → {xlsx_path}')
    return str(xlsx_path)
```

```python
# -----------------------------
# Metric Helpers
# -----------------------------
def multiclass_brier_score(y_onehot, y_prob):
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

def expected_calibration_error(y_true, y_prob, n_bins=15):
    confidences, predictions = np.max(y_prob, axis=1), np.argmax(y_prob, axis=1)
    correct = (predictions == y_true).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i+1] if i == n_bins - 1 else confidences < bin_edges[i+1])
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(correct[in_bin]) - np.mean(confidences[in_bin])) * prop
    return float(ece)
```

## 7.1 — Master Evaluation Loop & Summary Export

```python
master_results = []
plot_entries = []

# Assuming you want to loop over these architectures (edit if you have more)
architectures = ['AlexNet_CNN', 'GFNet', 'ViT_UNet']

for model_name in architectures:
    print(f"\n{'='*60}\n  Evaluating CreDE: {model_name}\n{'='*60}")

    ensemble_paths = get_ensemble_paths(model_name)
    if not ensemble_paths:
        print(f"  -> No ensemble models found for {model_name}. Skipping.")
        continue

    # Step 1: Run homogenous ensemble evaluation
    pred_class, p_star, au, eu, tu = evaluate_homogeneous_ensemble(
        ensemble_paths, scene_pixels_scaled, batch_size=2048
    )

    # Step 2: Generate 3x4 spatial maps
    saved_plot_path = generate_spatial_crede_maps(
        model_name, pred_class, p_star, au, eu, tu,
        H=H, W=W, au_thresh=AU_THRESH, eu_thresh=EU_THRESH, tu_thresh=TU_THRESH
    )
    plot_entries.append((f'{model_name} — Spatial Uncertainty Maps', saved_plot_path))

    # Step 3: Accumulate summary metrics
    unique, counts = np.unique(pred_class, return_counts=True)
    pixel_counts   = dict(zip(unique, counts))

    master_results.append({
        "Model":    f"{model_name}_CreDE",
        "Mean_AU":  float(np.mean(au)),
        "Mean_EU":  float(np.mean(eu)),
        "Mean_TU":  float(np.mean(tu)),
        **{f"Class_{int(k)}_Pixels": int(v) for k, v in pixel_counts.items()}
    })

    # Step 4: Aggressively free RAM before next architecture
    del pred_class, p_star, au, eu, tu
    tf.keras.backend.clear_session()
    gc.collect()

# --- Export master summary & Excel Report ---
if master_results:
    df_summary = pd.DataFrame(master_results)

    # Save CSV
    csv_path = CREDE_OUT_DIR / "CreDE_Master_Summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved CSV summary → {csv_path}")
    print(df_summary.to_string(index=False))

    # Save Excel Report
    create_crede_excel_report(
        out_dir=CREDE_OUT_DIR,
        summary_df=df_summary,
        plot_paths=plot_entries
    )
```

