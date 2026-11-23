import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes, fix_gfnet_rfft
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
def rep_block(start, end, new, tag):
    c = [c for c in nb._cells() if start in "".join(c["source"])][0]; s = "".join(c["source"])
    i = s.index(start); j = s.index(end, i) + len(end)
    c["source"] = (s[:i] + new + s[j:]).splitlines(keepends=True); nb.log.append(tag)
nb.rep('''_, x_test_pool, _, y_test_pool, _, coords_test_pool = train_test_split(
    X_all, y_all, coords_all,
    train_size=TRAIN_PERCENT, random_state=SEED, stratify=y_all)

x_cal, x_eval, y_cal, y_eval, coords_cal, coords_eval = split_calib_eval_with_coords(
    x_test_pool, y_test_pool, coords_test_pool,
    seed=SEED, calib_fraction=CALIB_FRACTION_OF_TEST)''', R("split_helper.py") + '''

# FIX C1: shared split (same cal / eval pixels in every notebook)
SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_cal,  y_cal,  coords_cal  = X_all[SPLIT["cal"]],  y_all[SPLIT["cal"]],  coords_all[SPLIT["cal"]]
x_eval, y_eval, coords_eval = X_all[SPLIT["eval"]], y_all[SPLIT["eval"]], coords_all[SPLIT["eval"]]
_EVAL_MASK = np.zeros(y_img.shape, dtype=bool)
_EVAL_MASK[tuple(SPLIT["coords"][SPLIT["eval"]].T)] = True''', tag="C1 shared split")
rep_block("def main_algo(Dcells_scores, Dcells_target, Dre_cal_scores, Dre_cal_target,",
          "    return covered.mean(), set_size.mean(), pred_sets",
          R("qhat_fix.txt") + "\n\n" + R("multicp_algo.txt"), "H11 main_algo")
rep_block("def build_spatially_smoothed_probs(", "    return smoothed_probs", '''def build_spatially_smoothed_probs(
    raw_probs,    # (K, N, C) — multi-head softmax probabilities for N pixels
    coords,       # (N, 2) — spatial coordinates of those pixels
    H, W,
    neighbors,
    lambda_=0.5,
    k_iters=1,
):
    """
    Vectorised SCMCP probability smoothing (same maths as spatial_smooth_prob_map):
    p <- (1 - lambda) * p + lambda * mean(neighbours present in the map), then renormalise,
    repeated k_iters times. Pixels outside `coords` are ignored as neighbours.
    """
    import scipy.ndimage as ndi
    K, N, C = raw_probs.shape
    w = int(round(np.sqrt(len(neighbors) + 1)))
    m = np.zeros((H, W), dtype=np.float64); m[coords[:, 0], coords[:, 1]] = 1.0
    cnt = ndi.uniform_filter(m, size=w, mode='constant') * w * w - m
    out = np.zeros_like(raw_probs, dtype=np.float64)
    for k in range(K):
        P = np.zeros((H, W, C), dtype=np.float64); P[coords[:, 0], coords[:, 1]] = raw_probs[k]
        for _ in range(k_iters):
            Pm  = P * m[..., None]
            num = ndi.uniform_filter(Pm, size=(w, w, 1), mode='constant') * w * w - Pm
            nb_mean = np.where(cnt[..., None] > 0, num / np.maximum(cnt[..., None], 1e-12), P)
            P = (1.0 - lambda_) * P + lambda_ * nb_mean
            P = P / np.maximum(P.sum(axis=-1, keepdims=True), EPS)
        out[k] = P[coords[:, 0], coords[:, 1]]
    return out

def full_scene_smoothed_probs(model, neighbors, lambda_, k_iters):
    """FIX H10: smooth the FULL-scene probability volume once; calibration, evaluation and
    the scene map all read their scores from this same map (one calibrated procedure)."""
    image_outputs = get_image_multi_head_outputs(model, padded_x, H, W, B, PATCH_SIZE, BATCH_SIZE)
    all_coords = np.argwhere(np.ones((H, W), dtype=bool))
    return image_outputs, build_spatially_smoothed_probs(
        image_outputs, all_coords, H, W, neighbors=neighbors, lambda_=lambda_, k_iters=k_iters)
''', "H10 vectorised smoothing")
nb.rep('''    lambda_=0.5,
    k_iters=1,
):
    """
    Fused Spatial MultiCP (SCMCP) head sweep''', '''    lambda_=0.5,
    k_iters=1,
    smoothed_full=None,
):
    """
    Fused Spatial MultiCP (SCMCP) head sweep''', tag="H10 sig")
nb.rep('''    # ── Probability-space smoothing — calibration and evaluation kept separate ──
    cal_probs_smooth = build_spatially_smoothed_probs(
        cal_output, coords_cal, H, W,
        neighbors=neighbors, lambda_=lambda_, k_iters=k_iters)   # (K, N_cal, C)

    test_probs_smooth = build_spatially_smoothed_probs(
        test_output, coords_eval, H, W,
        neighbors=neighbors, lambda_=lambda_, k_iters=k_iters)   # (K, N_eval, C)''', '''    # FIX H10: read cal / eval probabilities from the single full-scene smoothed map
    cal_probs_smooth  = smoothed_full[:, coords_cal[:, 0]  * W + coords_cal[:, 1],  :]
    test_probs_smooth = smoothed_full[:, coords_eval[:, 0] * W + coords_eval[:, 1], :]''', tag="H10 sweep")
nb.rep('''    neighbors, lambda_, k_iters,
):''', '''    neighbors, lambda_, k_iters,
    image_outputs=None, smoothed_full=None,
):''', tag="H10 sig2")
nb.rep('''    image_outputs = get_image_multi_head_outputs(
        model, padded_x, H, W, B, PATCH_SIZE, BATCH_SIZE)   # (K, H*W, C) — softmax probs

    # ── Full-scene probability-space smoothing + renormalisation per head ───────
    all_coords_full = np.array([[r, c] for r in range(H) for c in range(W)])
    smoothed_probs_full = build_spatially_smoothed_probs(
        image_outputs, all_coords_full, H, W,
        neighbors=neighbors, lambda_=lambda_, k_iters=k_iters)   # (K, H*W, C)
''', '''    if smoothed_full is None:
        image_outputs, smoothed_full = full_scene_smoothed_probs(model, neighbors, lambda_, k_iters)
    smoothed_probs_full = smoothed_full   # (K, H*W, C), same map used for calibration (FIX H10)
''', tag="H10 map")
nb.rep('''    gt_uncertain   = (y_flat == 7) & orig_mask_flat
    cp_valid       = orig_mask_flat & (~gt_uncertain)

    img_valid  = image_scores[:, cp_valid, :]
    y_valid    = y_flat[cp_valid] - 1''', '''    # FIX H12: no hard-coded class is forced to "uncertain"; coverage on eval pixels only
    gt_uncertain   = np.zeros_like(orig_mask_flat)
    cp_valid       = orig_mask_flat

    img_valid  = image_scores[:, cp_valid, :]
    y_valid    = np.where(_EVAL_MASK.ravel()[cp_valid], y_flat[cp_valid].astype(int) - 1, -1)''', tag="H12")
nb.rep('''        cal_output  = get_multihead_outputs(model, x_cal,  BATCH_SIZE)
        eval_output = get_multihead_outputs(model, x_eval, BATCH_SIZE)
''', '''        cal_output  = get_multihead_outputs(model, x_cal,  BATCH_SIZE)
        eval_output = get_multihead_outputs(model, x_eval, BATCH_SIZE)
        _img_out, _smooth_full = full_scene_smoothed_probs(model, neighbors, SACP_LAMBDA, SACP_K)
''', tag="H10 once per model")
nb.rep('''                lambda_=SACP_LAMBDA,
                k_iters=SACP_K,
            )
            config, Dc, Dt, Rc, Rt, pred_sets = bundle''', '''                lambda_=SACP_LAMBDA,
                k_iters=SACP_K,
                smoothed_full=_smooth_full,
            )
            config, Dc, Dt, Rc, Rt, pred_sets = bundle''', tag="H10 call1")
nb.rep('''                neighbors=neighbors, lambda_=SACP_LAMBDA, k_iters=SACP_K,
            )''', '''                neighbors=neighbors, lambda_=SACP_LAMBDA, k_iters=SACP_K,
                image_outputs=_img_out, smoothed_full=_smooth_full,
            )''', tag="H10 call2")
nb.rep('''        ['git', 'clone', 'https://github.com/yamtawa/Multi-CP.git', str(repo_path)],
        check=True,
    )''', '''        ['git', 'clone', 'https://github.com/yamtawa/Multi-CP.git', str(repo_path)],
        check=True,
    )
    # FIX M20: pin the external code to a fixed commit for reproducibility
    subprocess.run(['git', '-C', str(repo_path), 'checkout', '-q',
                    'fc252502d0bd2e2311f872672e3b70f7f64bcb74'], check=True)''', tag="M20 pin")
assert fix_gfnet_rfft(nb)
common_fixes(nb, F)
print(nb.save())
