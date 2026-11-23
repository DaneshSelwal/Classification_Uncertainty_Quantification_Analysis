import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
nb.rep('''_, x_test_pool, _, y_test_pool, _, coords_test_pool = train_test_split(
    X_all, y_all, coords_all,
    train_size=TRAIN_PERCENT, random_state=SEED, stratify=y_all,
)

x_cal, x_eval, y_cal, y_eval, coords_cal, coords_eval = split_calib_eval_with_coords(
    x_test_pool, y_test_pool, coords_test_pool,
    seed=SEED, calib_fraction=CALIB_FRACTION_OF_TEST,
)
''', R("split_helper.py") + '''

# FIX C1: shared split -> identical cal / eval pixels for every model and notebook
SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_cal,  y_cal,  coords_cal  = X_all[SPLIT["cal"]],  y_all[SPLIT["cal"]],  coords_all[SPLIT["cal"]]
x_eval, y_eval, coords_eval = X_all[SPLIT["eval"]], y_all[SPLIT["eval"]], coords_all[SPLIT["eval"]]
''', tag="C1 shared split")
nb.rep('''                smoothed[r, c] = self.lmd * ori + self.lmd * (n_sum / n_count)''',
       '''                # FIX H10: weights must sum to 1 -> (1 - lambda) * own + lambda * neighbours
                smoothed[r, c] = (1.0 - self.lmd) * ori + self.lmd * (n_sum / n_count)''', tag="H10 lambda")
old_start = "    # ── 1. Inference on calibration & evaluation splits"
old_end = "    pixel_counts_df = build_pixel_counts_df(combined_map, num_classes)\n"
cells = [c for c in nb._cells() if old_start in "".join(c["source"])]
s = "".join(cells[0]["source"]); i = s.index(old_start); j = s.index(old_end, i) + len(old_end)
new_body = '''    # FIX H10: calibration, evaluation and the map all use ONE full-scene smoothed APS score
    # map (smoothing uses predictions only, never labels), so the map is the calibrated rule.
    print(f'Generating full-scene probabilities for {model_name} ...')
    prob_full = predict_full_scene_probs(model, x_img, H, W, B, PATCH_SIZE, batch_size=batch_size)
    current_map = aps_score_map(prob_full, seed=SEED)
    for _ in range(max(int(k), 0)):
        current_map = (1.0 - lambda_) * current_map + lambda_ * _neighbour_mean(current_map, window_size)

    cr, cc = coords_cal[:, 0], coords_cal[:, 1]
    er, ec = coords_eval[:, 0], coords_eval[:, 1]
    q_hat  = conformal_qhat(current_map[cr, cc, np.asarray(y_cal, dtype=int)], alpha)

    pred_sets_eval = current_map[er, ec, :] <= q_hat
    avg_size = float(pred_sets_eval.sum(axis=1).mean())
    metrics  = compute_set_metrics(pred_sets_eval, y_eval)
    per_cls  = per_class_coverage_df(pred_sets_eval, y_eval, num_classes)

    pred_sets_full  = (current_map <= q_hat)
    set_sizes_map   = np.sum(pred_sets_full, axis=2)
    pred_class_map  = np.argmax(prob_full, axis=2)
    combined_map    = np.where(set_sizes_map == 1, pred_class_map, num_classes)
    pixel_counts_df = build_pixel_counts_df(combined_map, num_classes)
'''
cells[0]["source"] = (s[:i] + new_body + s[j:]).splitlines(keepends=True); nb.log.append("H10 SACP full-map")
nb.rep('''def build_sacp_outputs_for_model(''', R("spatial_helpers.txt") + "\n\ndef build_sacp_outputs_for_model(", tag="H10 helpers")
nb.rep("title='SACP: Per-Class Coverage (Full Image)'", "title='SACP: Per-Class Coverage (evaluation pixels)'", tag="nit title")
common_fixes(nb, F)
print(nb.save())
