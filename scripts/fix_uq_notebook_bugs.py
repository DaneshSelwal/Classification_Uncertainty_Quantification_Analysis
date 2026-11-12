#!/usr/bin/env python3
"""Fix the systematic NameError / wrong-path bugs in the UQ notebooks across all 4 target datasets.
Each fix inserts a definition cell immediately BEFORE the first cell that uses the symbol,
located by a unique anchor substring (robust across the structurally-identical dataset clones).
Idempotent: skips a notebook if the fix marker is already present.
"""
import json, sys, glob, os, re

MARK = "# [auto-fix: hoisted definition so it exists before first use]"

DATASETS = [
    'indian_pines_uncertainty_quantification',
    'pavia_university_uncertainty_quantification',
    'hisar_3mts_uncertainty_quantification',
    'hissar_25_uncertainty_quantification',
]

def new_code_cell(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
            "source":[l if l.endswith('\n') else l+'\n' for l in src.splitlines()]}

def load(path):
    with open(path) as f: return json.load(f)
def save(path,nb):
    with open(path,'w') as f: json.dump(nb,f,indent=1,ensure_ascii=False)

def insert_before_anchor(nb, anchor, def_src, label):
    """Insert a def cell before the first code cell containing `anchor`. Returns status str."""
    cells = nb['cells']
    # idempotency: already have this exact hoist?
    for c in cells:
        if c['cell_type']=='code' and MARK in ''.join(c['source']) and label in ''.join(c['source']):
            return 'skip(already-fixed)'
    for i,c in enumerate(cells):
        if c['cell_type']!='code': continue
        if anchor in ''.join(c['source']):
            full = f"{MARK}  <{label}>\n" + def_src
            cells.insert(i, new_code_cell(full))
            return f'inserted-before-cell{i}'
    return 'ANCHOR-NOT-FOUND'

def replace_in_cells(nb, old, new):
    n=0
    for c in nb['cells']:
        if c['cell_type']!='code': continue
        src=''.join(c['source'])
        if old in src:
            src=src.replace(old,new)
            c['source']=[l if l.endswith('\n') else l+'\n' for l in src.splitlines()]
            n+=1
    return n

# (relative path, anchor substring, definition source, label)
FIXES = [
 ('sacp/model_sacp_comparison.ipynb',
  'extract_labeled_patches_with_coords(x_img',
  'x_img, y_img, H, W, B = load_scene_from_mat(DATA_FILE, LABEL_FILE, DATA_KEY, LABEL_KEY)\n'
  'print("Scene loaded:", x_img.shape, "labels:", y_img.shape)\n',
  'x_img'),
 ('credit/model_training_credit.ipynb',
  'test_size=VAL_SPLIT_FROM_TRAIN',
  'VAL_SPLIT_FROM_TRAIN = 0.20\n',
  'VAL_SPLIT_FROM_TRAIN'),
 ('multi_cp/model_training_multihead.ipynb',
  'load_scene_from_mat(DATA_FILE, LABEL_FILE)',
  'PERCENTILE_CLIP = (2.0, 98.0)\n',
  'PERCENTILE_CLIP'),
 ('dapm/model_training_dapm_full.ipynb',
  'MAX_TARGET_UNLABELED',
  'MAX_TARGET_UNLABELED = 20000\n',
  'MAX_TARGET_UNLABELED'),
 ('dapm/model_uncertainty_dapm_full.ipynb',
  'MODEL_NAME_MAP.get(',
  "MODEL_NAME_MAP = {\n    'AlexNet_CNN': 'AlexNet',\n    'GFNet':       'GFNet',\n    'ViT_UNet':    'ViT',\n}\n",
  'MODEL_NAME_MAP'),
 ('ensemble/model_uncertainty_credal_ensemble.ipynb',
  'CREDE_OUT_DIR',
  'from pathlib import Path as _P\n'
  'try:\n    ROOT_DIR\nexcept NameError:\n    ROOT_DIR = _P.cwd()\n'
  'CREDE_OUT_DIR = ROOT_DIR / "ensemble" / "results"\n',
  'CREDE_OUT_DIR'),
 ('multicp_sacp/Model_uncertainty_multicp_sacp.ipynb',
  'train_size=TRAIN_PERCENT',
  'TRAIN_PERCENT = 0.75\n',
  'TRAIN_PERCENT'),
]

base_dir = sys.argv[1]
summary={}
for ds in DATASETS:
    print(f"\n===== {ds} =====")
    # 1) constant/data hoist fixes
    for rel, anchor, defsrc, label in FIXES:
        path=os.path.join(base_dir,ds,rel)
        if not os.path.exists(path):
            print(f"  {rel:52s} MISSING FILE"); continue
        nb=load(path)
        status=insert_before_anchor(nb,anchor,defsrc,label)
        if status.startswith('inserted'): save(path,nb)
        print(f"  {rel:52s} {label:22s} {status}")
    # 2) baseline/comparison model-path fix
    bpath=os.path.join(base_dir,ds,'baseline/model_uncertainty_comparison.ipynb')
    if os.path.exists(bpath):
        nb=load(bpath)
        old='local_model_dirs = [Path("models"), Path("results/models"), Path("../models")]'
        new='local_model_dirs = [Path("/kaggle/working/models"), Path("models"), Path("results/models"), Path("../models")]'
        n=replace_in_cells(nb,old,new)
        if n: save(bpath,nb)
        print(f"  baseline/model_uncertainty_comparison.ipynb         model-path fix: {n} cell(s)")
print("\nDONE")
