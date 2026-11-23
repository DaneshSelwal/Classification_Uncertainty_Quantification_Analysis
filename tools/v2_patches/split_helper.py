# FIX C1/C3/C4 — shared train / val / cal / eval split (identical code in every notebook) ──
# SPLIT_MODE = "spatial_block": whole spatial blocks go to one role and training
#   pixels whose 9x9 patch would contain a cal/eval pixel are dropped, so no
#   labelled test pixel is ever seen inside a training patch.
# SPLIT_MODE = "random": legacy stratified pixel-wise split (spatially leaky).
import numpy as _np_split
SPLIT_MODE        = globals().get("SPLIT_MODE", "spatial_block")
SPLIT_SEED        = 42
SPLIT_BLOCK       = 10
SPLIT_TRAIN_FRAC  = 0.75   # fraction of labelled pixels assigned to train+val
SPLIT_VAL_FRAC    = 0.20   # fraction of train+val used for validation / checkpointing
SPLIT_CAL_FRAC    = 0.50   # fraction of the held-out pool used for conformal calibration


def make_shared_split(y_img, patch_size=9, mode=None, seed=SPLIT_SEED,
                      train_frac=SPLIT_TRAIN_FRAC, val_frac=SPLIT_VAL_FRAC,
                      cal_frac=SPLIT_CAL_FRAC, block=SPLIT_BLOCK):
    """Return dict of index arrays (into np.argwhere(y_img > 0) order):
    train, val, cal, eval, test (= cal U eval)."""
    mode   = mode or SPLIT_MODE
    coords = _np_split.argwhere(y_img > 0)
    labels = y_img[y_img > 0].astype(int) - 1
    n, K   = len(labels), int(labels.max()) + 1
    rng    = _np_split.random.default_rng(seed)
    roles  = _np_split.full(n, -1)                       # 0 train, 1 val, 2 cal, 3 eval
    held   = 1.0 - train_frac
    frac   = _np_split.array([train_frac * (1 - val_frac), train_frac * val_frac,
                              held * cal_frac, held * (1 - cal_frac)])
    if mode == "random":
        from sklearn.model_selection import train_test_split as _tts
        idx = _np_split.arange(n)
        def _strat(a):
            cnt = _np_split.bincount(labels[a], minlength=K)
            return labels[a] if cnt[cnt > 0].min() >= 2 else None
        tv, te = _tts(idx, train_size=train_frac, random_state=seed, stratify=_strat(idx))
        tr, va = _tts(tv, test_size=val_frac, random_state=seed, stratify=_strat(tv))
        ca, ev = _tts(te, test_size=1 - cal_frac, random_state=seed, stratify=_strat(te))
        for r, a in enumerate([tr, va, ca, ev]):
            roles[a] = r
    else:
        bid = (coords[:, 0] // block) * 100000 + (coords[:, 1] // block)
        ub, inv = _np_split.unique(bid, return_inverse=True)
        cnt = _np_split.zeros((len(ub), K))
        _np_split.add.at(cnt, (inv, labels), 1)
        target = frac[None, :] * cnt.sum(0)[:, None]          # (K, 4)
        have   = _np_split.zeros((K, 4))
        order  = rng.permutation(len(ub))
        # rare classes first so each role receives some of them
        rarity = (cnt / _np_split.maximum(cnt.sum(0, keepdims=True), 1)).max(1)
        order  = order[_np_split.argsort(-rarity[order], kind="stable")]
        for b in order:
            w    = cnt[b]
            need = ((target - have) / _np_split.maximum(target, 1)) * w[:, None]
            role = int(_np_split.argmax(need.sum(0)))
            have[:, role] += w
            roles[inv == b] = role
        # drop training/validation pixels whose patch would contain a held-out pixel
        pad  = patch_size // 2
        grid = _np_split.zeros(y_img.shape, dtype=bool)
        hc   = coords[roles >= 2]
        grid[hc[:, 0], hc[:, 1]] = True
        from scipy.ndimage import maximum_filter as _mf
        near = _mf(grid.astype(_np_split.uint8), size=2 * pad + 1) > 0
        drop = (roles <= 1) & near[coords[:, 0], coords[:, 1]]
        roles[drop] = -1
    out = {k: _np_split.where(roles == r)[0] for r, k in enumerate(["train", "val", "cal", "eval"])}
    out["test"] = _np_split.sort(_np_split.concatenate([out["cal"], out["eval"]]))
    out["coords"], out["labels"], out["mode"] = coords, labels, mode
    return out


def describe_split(split, num_classes):
    import pandas as _pd
    rows = {k: _np_split.bincount(split["labels"][split[k]], minlength=num_classes)
            for k in ["train", "val", "cal", "eval"]}
    df = _pd.DataFrame(rows); df.index.name = "class"
    print(f"Split mode: {split['mode']}  |  sizes:", {k: int(v.sum()) for k, v in rows.items()})
    return df
