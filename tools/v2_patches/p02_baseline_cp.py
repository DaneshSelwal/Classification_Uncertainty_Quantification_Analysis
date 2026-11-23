import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, rep_tail, fix_gfnet
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
SPLIT = R("split_helper.py")
if not nb.has('''all_idx = np.arange(len(y_all))'''):
    from nbpatch import generic_split_fallback; generic_split_fallback(nb, F)
else: nb.rep('''all_idx = np.arange(len(y_all))
try:
    train_idx, test_pool_idx = train_test_split(
        all_idx, train_size=TRAIN_PERCENT, random_state=SEED, stratify=y_all
    )
except ValueError:
    train_idx, test_pool_idx = train_test_split(
        all_idx, train_size=TRAIN_PERCENT, random_state=SEED, stratify=None
    )

test_y_pool = y_all[test_pool_idx]
try:
    cal_sub, eval_sub = train_test_split(
        np.arange(len(test_pool_idx)), test_size=(1.0 - CALIB_FRACTION_OF_TEST),
        random_state=SEED, stratify=test_y_pool
    )
except ValueError:
    cal_sub, eval_sub = train_test_split(
        np.arange(len(test_pool_idx)), test_size=(1.0 - CALIB_FRACTION_OF_TEST),
        random_state=SEED, stratify=None
    )

cal_idx = test_pool_idx[cal_sub]
eval_idx = test_pool_idx[eval_sub]
''', SPLIT + '''

# FIX C1: every model (incl. AlexNet) is calibrated / evaluated on the same held-out pixels
SPLIT    = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
cal_idx  = SPLIT["cal"]
eval_idx = SPLIT["eval"]
''', tag="C1 shared split")
rep_tail_marker = None
# H6 quantile
old_q = '''def conformal_qhat(scores, alpha):
    """Compute the conformal quantile threshold q_hat for a given alpha."""
    n       = len(scores)
    if n == 0:
        return 1.0
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return safe_quantile(scores, q_level)
'''
nb.rep(old_q, R("qhat_fix.txt"), tag="H6 qhat")
# H8 RC3P: choose the mixing parameter on calibration data, not on the evaluation set
nb.rep('''    q_hats, suit_indices = compute_rc3p_qhats_and_sets(
        prob_cal, y_cal, prob_eval, alpha, truncated_gap,
    )''', '''    # FIX H8: the set-size criterion is evaluated on calibration data (was: on the eval set)
    q_hats, suit_indices = compute_rc3p_qhats_and_sets(
        prob_cal, y_cal, prob_cal, alpha, truncated_gap,
    )''', tag="H8 RC3P")
# M19 ClCP: separate clustering and calibration halves
nb.rep('''    # Cluster classes by mean embedding
    global_mean = emb_cal.mean(axis=0)
    class_means = []
    for c in range(n_classes):
        m = (y_cal == c)
        class_means.append(emb_cal[m].mean(axis=0) if m.sum() > 0 else global_mean)''',
'''    # FIX M19: fit clusters on one half of the calibration data and compute the
    # conformal quantiles on the other half (Ding et al., 2023)
    _perm   = np.random.default_rng(SEED).permutation(len(y_cal))
    _clu    = np.zeros(len(y_cal), dtype=bool); _clu[_perm[: len(y_cal) // 2]] = True
    global_mean = emb_cal[_clu].mean(axis=0)
    class_means = []
    for c in range(n_classes):
        m = (y_cal == c) & _clu
        class_means.append(emb_cal[m].mean(axis=0) if m.sum() > 0 else global_mean)''', tag="M19 ClCP a")
nb.rep('''        cls  = np.where(cluster_assignments == cluster_id)[0]
        mask = np.isin(y_cal, cls)''', '''        cls  = np.where(cluster_assignments == cluster_id)[0]
        mask = np.isin(y_cal, cls) & ~_clu''', tag="M19 ClCP b")
# H7 RAPS
old_raps_start = 'def raps_score_single(prob_row, true_label, lam=0.01, k_reg=1):'
old_raps_end = '''    # Guarantee at least the top-1 class is included
    if not pred_set.any():
        pred_set[order[0]] = True

    return pred_set
'''
src_cells = [c for c in nb._cells() if old_raps_start in "".join(c["source"])]
s = "".join(src_cells[0]["source"]); i = s.index(old_raps_start); j = s.index(old_raps_end) + len(old_raps_end)
src_cells[0]["source"] = (s[:i] + R("raps_fix.txt") + s[j:]).splitlines(keepends=True); nb.log.append("H7 RAPS")
# H9 spatial (dead code in this notebook, replaced by the correct shared implementation)
if nb.has("def build_sacp_outputs_for_model("): rep_tail(nb, "def build_sacp_outputs_for_model(", R("spatial_fix.txt"), tag="H9 spatial")
assert fix_gfnet(nb, R("gfnet_fix.txt"))
print(nb.save())
