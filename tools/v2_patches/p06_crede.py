import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
nb.rep('''x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y_labels, train_size=TRAIN_PERCENT, random_state=SEED, stratify=y_labels
)''', R("split_helper.py") + '''

# FIX C1: shared split -> same held-out pixels for every model
SPLIT  = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_test, y_test = X[SPLIT["test"]], y_labels[SPLIT["test"]]''', tag="C1 shared split")
if nb.has('''_, x_test_alex, _, y_test_alex = train_test_split('''): nb.rep('''_, x_test_alex, _, y_test_alex = train_test_split(
    X, y_labels,
    train_size=ALEXNET_LEGACY_TRAIN_PERCENT,
    random_state=ALEXNET_LEGACY_SPLIT_SEED,
    stratify=y_labels
)''', '''x_test_alex, y_test_alex = x_test, y_test   # FIX C1: AlexNet members use the shared split''', tag="C1 AlexNet")
import re as _re
_NEW = """    # FIX H14: standard ensemble decomposition (normalised by ln C so all values lie in [0, 1]):
    #   TU = H(mean_m p_m),  AU = mean_m H(p_m),  EU = TU - AU  (mutual information)
    P      = np.clip(np.asarray(stacked_preds, dtype=np.float64), 1e-12, 1.0)   # (M, N, C)
    lnC    = np.log(P.shape[-1])
    p_star = P.mean(axis=0)                                    # ensemble mean (credal centre)
    tu = -np.sum(p_star * np.log(p_star), axis=-1) / lnC
    au = np.mean(-np.sum(P * np.log(P), axis=-1), axis=0) / lnC
    eu = np.clip(tu - au, 0.0, None)

    pred_class = np.argmax(p_star, axis=-1)"""
_pat = _re.compile(r"    p_min = tf\.reduce_min\(stacked_preds, axis=0\).*?pred_class = np\.argmax\(p_star, axis=-1\)", _re.S)
_hit = 0
for _c in nb._cells():
    _s = "".join(_c["source"]); _s2, _n = _pat.subn(lambda m: _NEW, _s)
    if _n: _c["source"] = _s2.splitlines(keepends=True); _hit += _n
if _hit != 1: raise SystemExit(f"[{nb.path.name}] H14 CreDE decomposition: {_hit} matches")
nb.log.append("H14 CreDE decomposition")
if nb.has("    p_star     : np.ndarray  Shape (N, C) — normalised"): nb.rep("    p_star     : np.ndarray  Shape (N, C) — normalised lower credal probabilities.",
       "    p_star     : np.ndarray  Shape (N, C) — ensemble mean probabilities.", tag="doc")
if nb.has("    au         : np.ndarray  Shape (N,)   — aleatoric uncertainty (entropy of p_star)."): nb.rep("    au         : np.ndarray  Shape (N,)   — aleatoric uncertainty (entropy of p_star).\n    eu         : np.ndarray  Shape (N,)   — epistemic uncertainty (mean credal spread).\n    tu         : np.ndarray  Shape (N,)   — total uncertainty (au + eu).",
       "    au         : np.ndarray  Shape (N,)   — aleatoric: mean member entropy / ln C.\n    eu         : np.ndarray  Shape (N,)   — epistemic: mutual information / ln C.\n    tu         : np.ndarray  Shape (N,)   — total: entropy of the mean / ln C (= au + eu).", tag="doc2")
common_fixes(nb, F)
print(nb.save())
