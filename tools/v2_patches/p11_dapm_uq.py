import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
nb.rep('''coords_labeled = np.argwhere(y_img > 0)                      # only labelled pixels
y_labeled      = np.array(
    [int(y_img[r, c]) - 1 for r, c in coords_labeled], dtype=np.int32
)
num_classes = int(np.unique(y_img[y_img > 0]).size)''', '''num_classes = int(np.unique(y_img[y_img > 0]).size)
''' + R("split_helper.py") + '''

# FIX C4: accuracy / uncertainty metrics are reported on held-out pixels only
# (previously on ALL labelled pixels, i.e. mostly training pixels)
SPLIT          = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
coords_labeled = SPLIT["coords"][SPLIT["test"]]
y_labeled      = SPLIT["labels"][SPLIT["test"]].astype(np.int32)''', tag="C4 held-out metrics")
s_start = "def compute_dapm_ttest_uncertainty_chunk(probs_samples, p_thresh=0.05):"
s_end = '''        uncertain_mask[i] = bool(pval > p_thresh)
'''
c = [c for c in nb._cells() if s_start in "".join(c["source"])][0]; s = "".join(c["source"])
i = s.index(s_start); j = s.index(s_end) + len(s_end)
new = '''def compute_dapm_ttest_uncertainty_chunk(probs_samples, p_thresh=0.05):
    """
    FIX H15: per-pixel uncertainty from the diffusion draws as a *flip rate*.

    For each pixel the winning class c1 is the argmax of the mean probability. The flip rate
    is the fraction of the n_samples draws whose own argmax is NOT c1, i.e. a Monte-Carlo
    estimate of P(prediction changes). A pixel is uncertain when flip_rate > p_thresh.
    Unlike the previous Welch t-test (independent-samples test on paired, dependent draws),
    this does not drift towards "certain" as n_samples grows.
    The 'p_values' key holds the flip rate so downstream tables keep their columns.
    """
    probs_samples                 = np.asarray(probs_samples, dtype=np.float32)
    n_samples, n_points, n_classes = probs_samples.shape

    mean_prob  = np.mean(probs_samples, axis=0)
    order      = np.argsort(-mean_prob, axis=1)
    c1, c2     = order[:, 0], order[:, 1]
    draw_top   = np.argmax(probs_samples, axis=2)                    # (S, N)
    flip_rate  = np.mean(draw_top != c1[None, :], axis=0).astype(np.float32)
    pts        = np.arange(n_points)
    mean_gap   = (mean_prob[pts, c1] - mean_prob[pts, c2]).astype(np.float32)
    pred_class = c1.astype(np.int32)
    p_values   = flip_rate
    uncertain_mask = flip_rate > p_thresh
'''
c["source"] = (s[:i] + new + s[j:]).splitlines(keepends=True); nb.log.append("H15 flip rate")
nb.rep("(DAPM t-test, p={P_THRESH})", "(DAPM flip rate > {P_THRESH})", tag="label")
if nb.has("'DAPM_TTest'"): nb.rep_all("'DAPM_TTest'", "'DAPM_FlipRate'", tag="label flip rate")
common_fixes(nb, F)
print(nb.save())
