"""Tiny notebook patch engine: exact-match replacements inside code cells."""
import json, sys
from pathlib import Path

class NB:
    def __init__(self, path):
        self.path = Path(path); self.nb = json.loads(self.path.read_text())
        self.log = []
    def _cells(self):
        return [c for c in self.nb["cells"] if c["cell_type"] == "code"]
    def rep(self, old, new, count=1, tag=""):
        hits = [(c, "".join(c["source"]).count(old)) for c in self._cells()]
        total = sum(h for _, h in hits)
        if total != count:
            raise SystemExit(f"[{self.path.name}] {tag}: expected {count} match(es), found {total}\n---\n{old[:300]}")
        for c, h in hits:
            if h:
                src = "".join(c["source"]).replace(old, new)
                c["source"] = src.splitlines(keepends=True)
        self.log.append(tag)
    def rep_all(self, old, new, tag=""):
        n = sum("".join(c["source"]).count(old) for c in self._cells())
        if n == 0:
            raise SystemExit(f"[{self.path.name}] {tag}: no match\n{old[:200]}")
        self.rep(old, new, count=n, tag=f"{tag} (x{n})")
    def has(self, s):
        return any(s in "".join(c["source"]) for c in self._cells())
    def save(self):
        self.path.write_text(json.dumps(self.nb, indent=1, ensure_ascii=False) + "\n")
        return self.log


def rep_tail(nb, start_marker, new, tag=""):
    """Replace everything from start_marker to the end of the (single) cell containing it."""
    cells = [c for c in nb._cells() if start_marker in "".join(c["source"])]
    if len(cells) != 1:
        raise SystemExit(f"[{nb.path.name}] {tag}: marker in {len(cells)} cells")
    src = "".join(cells[0]["source"]); i = src.index(start_marker)
    cells[0]["source"] = (src[:i] + new).splitlines(keepends=True); nb.log.append(tag)


GF_VARIANTS = [
 ('''        x_fft      = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
        w_complex  = tf.complex(self.w_real, self.w_imag)
        x_filtered = x_fft * w_complex
        x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
''', None),
 ('''        x_fft     = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
        w_complex  = tf.complex(self.w_real, self.w_imag)
        x_filtered = x_fft * w_complex
        x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
''', None),
 ('''        x_fft    = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
        w_complex  = tf.complex(self.w_real, self.w_imag)
        x_filtered = x_fft * w_complex
        x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
''', None),
 ('''        x_fft = tf.signal.fft2d(tf.cast(x2, tf.complex64))
        w  = tf.complex(self.w_real, self.w_imag)
        x_f = x_fft * w
        x_i = tf.math.real(tf.signal.ifft2d(x_f))
''', '''        # FIX C5: fft2d acts on the two innermost axes -> put channels first so the
        # filter runs over the (token_row, token_col) grid.
        x2  = tf.transpose(x2, [0, 3, 1, 2])
        x_fft = tf.signal.fft2d(tf.cast(x2, tf.complex64))
        w  = tf.transpose(tf.complex(self.w_real, self.w_imag), [2, 0, 1])
        x_f = x_fft * w
        x_i = tf.transpose(tf.math.real(tf.signal.ifft2d(x_f)), [0, 2, 3, 1])
'''),
 ('''        x_fft = tf.signal.fft2d(tf.cast(x2, tf.complex64))
        w  = tf.complex(self.w_real, self.w_imag)
        x_i = tf.math.real(tf.signal.ifft2d(x_fft * w))
''', '''        # FIX C5: fft2d acts on the two innermost axes -> put channels first so the
        # filter runs over the (token_row, token_col) grid.
        x2  = tf.transpose(x2, [0, 3, 1, 2])
        x_fft = tf.signal.fft2d(tf.cast(x2, tf.complex64))
        w  = tf.transpose(tf.complex(self.w_real, self.w_imag), [2, 0, 1])
        x_i = tf.transpose(tf.math.real(tf.signal.ifft2d(x_fft * w)), [0, 2, 3, 1])
'''),
]


def fix_gfnet(nb, fix_text):
    for old, new in GF_VARIANTS:
        if nb.has(old):
            nb.rep_all(old, new if new is not None else fix_text, tag="C5 GFNet FFT")
            return True
    return False


def common_fixes(nb, F):
    """Generic fixes applied to every notebook that contains the pattern."""
    R = lambda f: open(f"{F}/{f}").read()
    for old in ["engine='xlsxwriter')", 'engine="xlsxwriter")']:
        if nb.has(old):
            nb.rep_all(old, old[:-1] + ", engine_kwargs={'options': {'nan_inf_to_errors': True}})",
                       tag="K1 xlsx inf")
    fix_gfnet(nb, R("gfnet_fix.txt"))
    add_split_check(nb)
    fix_runtime_generic(nb)


GF_R_OLD_BUILD = '''            shape=(self.patch_size, self.patch_size, input_shape[-1] // 2 + 1, 2),'''
GF_R_NEW_BUILD = '''            # FIX C5: weights live on the (token_row, token_col) frequency grid per channel
            shape=(input_shape[-1], self.patch_size, self.patch_size // 2 + 1, 2),'''
GF_R_OLD_CALL = '''        x       = tf.reshape(x, [-1, a, b, C])
        x       = tf.signal.rfft2d(x)
        weight  = tf.dtypes.complex(self.complex_weight[:, :, :, 0], self.complex_weight[:, :, :, -1])
        x       = x * weight
        x       = tf.signal.irfft2d(x)
        return tf.reshape(x, [-1, N, C])'''
GF_R_NEW_CALL = '''        x       = tf.reshape(x, [-1, a, b, C])
        # FIX C5: rfft2d acts on the two innermost axes -> channels first, then FFT over (a, b)
        x       = tf.transpose(x, [0, 3, 1, 2])
        x       = tf.signal.rfft2d(x)
        weight  = tf.dtypes.complex(self.complex_weight[:, :, :, 0], self.complex_weight[:, :, :, -1])
        x       = x * weight
        x       = tf.signal.irfft2d(x, fft_length=[a, b])
        x       = tf.transpose(x, [0, 2, 3, 1])
        return tf.reshape(x, [-1, N, C])'''


def fix_gfnet_rfft(nb):
    """Multi-head GFNet (rfft2d variant). Call text differs slightly between notebooks."""
    import re
    done = False
    for c in nb._cells():
        s = "".join(c["source"])
        if "class GF_GlobalFilter" not in s or "FIX C5" in s:
            continue
        s2 = s.replace(GF_R_OLD_BUILD, GF_R_NEW_BUILD)
        pat = re.compile(r"(\n(\s+)x\s*=\s*tf\.signal\.rfft2d\(x\)\n)(.*?)(\n\s+x\s*=\s*tf\.signal\.irfft2d\(x\)\n)", re.S)
        m = pat.search(s2)
        if not m or s2 == s:
            raise SystemExit(f"[{nb.path.name}] GF rfft pattern not found")
        ind = m.group(2)
        new = (f"\n{ind}# FIX C5: rfft2d acts on the two innermost axes -> channels first, FFT over the token grid\n"
               f"{ind}_a, _b = int(x.shape[1]), int(x.shape[2])\n{ind}x = tf.transpose(x, [0, 3, 1, 2])\n"
               f"{ind}x = tf.signal.rfft2d(x)\n" + m.group(3) +
               f"\n{ind}x = tf.signal.irfft2d(x, fft_length=[_a, _b])\n{ind}x = tf.transpose(x, [0, 2, 3, 1])\n")
        s2 = s2[:m.start()] + new + s2[m.end():]
        c["source"] = s2.splitlines(keepends=True); done = True
    if done:
        nb.log.append("C5 GFNet rfft")
    return done


QHAT_OLD = '''def conformal_qhat(scores, alpha):
    """Compute the conformal quantile threshold q_hat for a given alpha."""
    n       = len(scores)
    if n == 0:
        return 1.0
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return safe_quantile(scores, q_level)
'''


def fix_cp_methods(nb, F):
    """H6 / H7 / H8 / M19 for notebooks that carry the split-CP method family."""
    R = lambda f: open(f"{F}/{f}").read()
    if nb.has(QHAT_OLD):
        nb.rep(QHAT_OLD, R("qhat_fix.txt"), tag="H6 qhat")
    old = '''    q_hats, suit_indices = compute_rc3p_qhats_and_sets(
        prob_cal, y_cal, prob_eval, alpha, truncated_gap,
    )'''
    if nb.has(old):
        nb.rep(old, '''    # FIX H8: the set-size criterion is evaluated on calibration data (was: on the eval set)
    q_hats, suit_indices = compute_rc3p_qhats_and_sets(
        prob_cal, y_cal, prob_cal, alpha, truncated_gap,
    )''', tag="H8 RC3P")
    old = '''    # Cluster classes by mean embedding
    global_mean = emb_cal.mean(axis=0)
    class_means = []
    for c in range(n_classes):
        m = (y_cal == c)
        class_means.append(emb_cal[m].mean(axis=0) if m.sum() > 0 else global_mean)'''
    if nb.has(old):
        nb.rep(old, '''    # FIX M19: fit clusters on one half of the calibration data and compute the
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
    a = 'def raps_score_single(prob_row, true_label, lam=0.01, k_reg=1):'
    b = '''    # Guarantee at least the top-1 class is included
    if not pred_set.any():
        pred_set[order[0]] = True

    return pred_set
'''
    if nb.has(a):
        c = [c for c in nb._cells() if a in "".join(c["source"])][0]; s = "".join(c["source"])
        i = s.index(a); j = s.index(b) + len(b)
        c["source"] = (s[:i] + R("raps_fix.txt") + s[j:]).splitlines(keepends=True); nb.log.append("H7 RAPS")


SPLIT_CHECK_DEF = '''
def check_split_alignment(split, ns):
    """Fail loudly if the notebook's labelled-patch order differs from the shared split order
    (row-major over y > 0). Any 1-D label vector of the same length is compared up to relabelling."""
    lab = split["labels"]; n = len(lab); checked = []
    for name in ("y", "Y", "y_all", "y_labels", "y_labeled"):
        v = ns.get(name)
        if isinstance(v, _np_split.ndarray) and v.ndim == 1 and len(v) == n:
            pairs = set(zip(lab.tolist(), _np_split.asarray(v).astype(int).tolist()))
            if len(pairs) != len(set(lab.tolist())) or len({b for _, b in pairs}) != len(pairs):
                raise RuntimeError(f"shared split is not aligned with label vector '{name}'")
            checked.append(name)
    print("split alignment verified against:", checked or "nothing (no 1-D label vector in scope)")
'''


def add_split_check(nb):
    import re
    n = 0
    for c in nb._cells():
        s = "".join(c["source"])
        if "make_shared_split(" not in s or "check_split_alignment(SPLIT" in s:
            continue
        s2, k = re.subn(r"^(SPLIT\s*=\s*make_shared_split\([^\n]*\)\n)",
                        lambda m: m.group(1) + "check_split_alignment(SPLIT, globals())\n", s, flags=re.M)
        if k and "def check_split_alignment" not in s2:
            s2 = s2.replace("\ndef describe_split(", SPLIT_CHECK_DEF + "\n\ndef describe_split(", 1)
        if k:
            c["source"] = s2.splitlines(keepends=True); n += k
    if n:
        nb.log.append(f"split alignment check (x{n})")


def generic_split_fallback(nb, F, tag="C1 shared split (variant)"):
    """Variant split blocks: cal/eval pool form and the multicp two-step form."""
    import re
    R = lambda f: open(f"{F}/{f}").read()
    forms = [
        (r"^_, x_test_pool, _, y_test_pool = train_test_split\(\n(?:.*\n)*?\)\n\s*\nx_cal, x_eval, y_cal, y_eval = split_calib_eval\(\n(?:.*\n)*?\)",
         lambda m: R("split_helper.py") + '''

# FIX C1: shared split -> identical cal / eval pixels for every model and notebook
SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_cal,  y_cal  = X_all[SPLIT["cal"]],  y_all[SPLIT["cal"]]
x_eval, y_eval = X_all[SPLIT["eval"]], y_all[SPLIT["eval"]]'''),
        (r"^x_train, x_test, y_train, y_test = train_test_split\(\n.*?\)\nx_cal, x_test_new, y_cal, y_test_new = train_test_split\(\n.*?\)",
         lambda m: R("split_helper.py") + '''

# FIX C1: shared split (same cal / eval pixels as every other notebook)
SPLIT = make_shared_split(y, patch_size=P_S)
display(describe_split(SPLIT, int(np.unique(Y).size)))
x_train,    y_train    = X[SPLIT["train"]], Y[SPLIT["train"]]
x_test,     y_test     = X[SPLIT["test"]],  Y[SPLIT["test"]]
x_cal,      y_cal      = X[SPLIT["cal"]],   Y[SPLIT["cal"]]
x_test_new, y_test_new = X[SPLIT["eval"]],  Y[SPLIT["eval"]]
_EVAL_MASK = np.zeros(y.shape, dtype=bool)
_EVAL_MASK[tuple(SPLIT["coords"][SPLIT["eval"]].T)] = True'''),
    ]
    for pat, new in forms:
        for c in nb._cells():
            s = "".join(c["source"]); s2, k = re.subn(pat, new, s, count=1, flags=re.M | re.S)
            if k:
                c["source"] = s2.splitlines(keepends=True); nb.log.append(tag); return True
    raise SystemExit(f"[{nb.path.name}] {tag}: no known split form found")


def fix_runtime_generic(nb):
    """Non-maths robustness fixes (K2, K3) found in the Indian Pines GPU smoke run."""
    import re
    n_cm = n_cr = n_ws = 0
    for c in nb._cells():
        s = "".join(c["source"]); o = s
        # K2: held-out sets may miss tiny classes under the spatial split -> fix the label set
        s, k = re.subn(r"\bconfusion_matrix\(\s*([\w\[\]\.]+),\s*([\w\[\]\.]+)\s*\)",
                       r"confusion_matrix(\1, \2, labels=np.arange(num_classes))", s); n_cm += k
        s, k = re.subn(r"\bclassification_report\(\s*([\w\[\]\.]+),\s*([\w\[\]\.]+),(?![^)]*labels=)",
                       r"classification_report(\1, \2, labels=np.arange(num_classes),", s); n_cr += k
        # K3: row-count checks follow the configured window list instead of a hard-coded 4
        if "SACP_WINDOW_SIZES" in "".join("".join(x["source"]) for x in nb._cells()):
            s, k = re.subn(r"expected_rows = 4 \* 3 \* 2[^\n]*", "expected_rows = len(SACP_WINDOW_SIZES) * 3 * 2   # windows * models * scores", s); n_ws += k
            s = s.replace("assert expected_rows == 24, f'Expected 24 total rows, got {expected_rows}'",
                          "assert expected_rows == len(SACP_WINDOW_SIZES) * 3 * 2")
            s = s.replace("assert len(combined_summary_df) == 24, (", "assert len(combined_summary_df) == expected_rows, (")
            s = s.replace("f'Expected 24 rows in combined_summary_df, got", "f'Expected {expected_rows} rows in combined_summary_df, got")
            s = re.sub(r"expected_per_class_rows = 24 \* num_classes[^\n]*", "expected_per_class_rows = len(SACP_WINDOW_SIZES) * 3 * 2 * num_classes", s)
        if s != o: c["source"] = s.splitlines(keepends=True)
    for tag, k in [("K2 confusion labels", n_cm), ("K2 report labels", n_cr), ("K3 window count", n_ws)]:
        if k: nb.log.append(f"{tag} (x{k})")
