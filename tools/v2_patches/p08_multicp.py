import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes, fix_gfnet_rfft
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
if not nb.has('''try:
    x_train, x_test, y_train, y_test = train_test_split('''):
    from nbpatch import generic_split_fallback; generic_split_fallback(nb, F)
else: nb.rep('''try:
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, train_size=TRAIN_PERCENT, stratify=Y, random_state=10
    )
except ValueError:
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, train_size=TRAIN_PERCENT, stratify=None, random_state=10
    )
try:
    x_cal, x_test_new, y_cal, y_test_new = train_test_split(
        x_test, y_test, test_size=0.5, stratify=y_test, random_state=10
    )
except ValueError:
    x_cal, x_test_new, y_cal, y_test_new = train_test_split(
        x_test, y_test, test_size=0.5, stratify=None, random_state=10
    )''', R("split_helper.py") + '''

# FIX C1: shared split (same cal / eval pixels as every other notebook)
SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_cal,      y_cal      = X[SPLIT["cal"]],  Y[SPLIT["cal"]]
x_test_new, y_test_new = X[SPLIT["eval"]], Y[SPLIT["eval"]]
_EVAL_MASK = np.zeros(y_img.shape, dtype=bool)
_EVAL_MASK[tuple(SPLIT["coords"][SPLIT["eval"]].T)] = True''', tag="C1 shared split")
s_start = "def main_algo(Dcells_scores, Dcells_target, Dre_cal_scores, Dre_cal_target,"
s_end = "    return covered.mean(), pred_sets.sum(axis=2).mean(), pred_sets\n"
c = [c for c in nb._cells() if s_start in "".join(c["source"])][0]; s = "".join(c["source"])
i = s.index(s_start); j = s.index(s_end) + len(s_end)
c["source"] = (s[:i] + R("qhat_fix.txt") + "\n\n" + R("multicp_algo.txt") + s[j:]).splitlines(keepends=True); nb.log.append("H11 main_algo")
nb.rep('''    gt_uncertain   = (y_flat == 7) & orig_mask_flat
    cp_valid       = orig_mask_flat & (~gt_uncertain)

    img_valid  = image_scores[:, cp_valid, :]
    y_valid    = y_flat[cp_valid] - 1
    cov, mset, pred_bool = main_algo(Dc, Dt, Rc, Rt, img_valid, y_valid, config['ALPHA'], config)
    set_sizes  = pred_bool.sum(axis=2).mean(axis=0)''', '''    # FIX H12: no hard-coded class id is forced to "uncertain"; full-image coverage is
    # measured only on the held-out evaluation pixels (background / train pixels -> -1).
    gt_uncertain   = np.zeros_like(orig_mask_flat)
    cp_valid       = orig_mask_flat

    img_valid  = image_scores[:, cp_valid, :]
    y_valid    = np.where(_EVAL_MASK.ravel()[cp_valid], y_flat[cp_valid].astype(int) - 1, -1)
    cov, mset, pred_bool = main_algo(Dc, Dt, Rc, Rt, img_valid, y_valid, config['ALPHA'], config)
    set_sizes  = pred_bool.all(axis=0).sum(axis=1)   # FIX H11: joint set size''', tag="H12 binary map")
nb.rep("    Ground-truth class-7 pixels (unlabelled) are also marked uncertain.\n", "", tag="doc")
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
