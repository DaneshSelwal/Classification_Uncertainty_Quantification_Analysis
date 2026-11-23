"""Generic CREDIT fix (C1/C3/H13/H14) for suite variants; regex rules + leftover checks."""
import sys, re; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
cells = lambda: [c for c in nb._cells()]
def sub(pat, repl, tag, min_n=1, max_n=99, flags=re.M):
    n = 0
    for c in cells():
        s = "".join(c["source"]); s2, k = re.subn(pat, repl, s, flags=flags)
        if k: c["source"] = s2.splitlines(keepends=True); n += k
    if not (min_n <= n <= max_n): raise SystemExit(f"[{nb.path.name}] {tag}: {n} matches")
    nb.log.append(f"{tag} (x{n})")

# ---- C1 shared split (same anchors as Indian Pines; tolerant whitespace)
sub(r"^x_train_full, x_test, y_train_full, y_test = train_test_split\(\n.*?\n\)\nx_train, x_val, y_train, y_val = train_test_split\(\n.*?\n\)\n",
    lambda m: R("split_helper.py") + '''

SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_train, y_train = X[SPLIT["train"]], y[SPLIT["train"]]
x_val,   y_val   = X[SPLIT["val"]],   y[SPLIT["val"]]
x_test,  y_test  = X[SPLIT["test"]],  y[SPLIT["test"]]
''', "C1 shared split", 1, 1, re.M | re.S)
sub(r"^x_train_alex, x_test_alex, y_train_alex, y_test_alex = train_test_split\(\n.*?\n\)",
    "# FIX C1/H13: AlexNet shares the split of the other backbones (legacy split constants are unused)\n"
    "x_train_alex, x_test_alex, y_train_alex, y_test_alex = x_train, x_test, y_train, y_test",
    "C1/H13 AlexNet", 1, 1, re.M | re.S)
# ---- H14 teacher targets
sub(r"^([ \t]+)delta_p_true\s*=\s*p_max - p_min\n[ \t]+p_star_true\s*=\s*p_min / \(tf\.reduce_sum\(p_min, axis=-1, keepdims=True\) \+ 1e-12\)\n[ \t]+return p_star_true, delta_p_true",
    lambda m: m.group(1) + '''# FIX H14: proper targets -> mean probability, lower bound, interval width, aleatoric part
    n_cls   = int(stacked.shape[-1])
    p_mean  = tf.reduce_mean(stacked, axis=0)
    ent_m   = -tf.reduce_sum(stacked * tf.math.log(tf.clip_by_value(stacked, 1e-12, 1.0)), axis=-1)
    au_t    = tf.reduce_mean(ent_m, axis=0)[:, None] / np.log(n_cls)
    return {'p_star': p_mean, 'p_low': p_min, 'delta_p': p_max - p_min,
            'au': tf.clip_by_value(au_t, 0.0, 1.0)}


def credit_uncertainty(outputs, num_classes):
    """FIX H14: TU = H(p_star)/ln C, AU = predicted mean member entropy (<= TU), EU = TU - AU.
    Credal bounds: lower = p_low, upper = min(1, p_low + delta_p)."""
    p_star, p_low, delta_p, au_hat = [np.asarray(o, dtype=np.float64) for o in outputs]
    p_star = np.clip(p_star, 1e-12, 1.0)
    tu = -np.sum(p_star * np.log(p_star), axis=-1) / np.log(num_classes)
    au = np.minimum(au_hat.reshape(-1), tu)
    eu = tu - au
    lower = np.clip(p_low, 0.0, 1.0)
    upper = np.clip(lower + delta_p, 0.0, 1.0)
    return p_star, lower, upper, au, eu, tu


def credit_predict(model, x, **kw):
    """Student inference -> (p_star, packed) with packed = [delta_p | p_low | au] (N, 2C+1),
    so existing two-value unpacking and chunk concatenation keep working."""
    o = model.predict(x, **kw)
    return np.asarray(o['p_star']), np.concatenate(
        [np.asarray(o['delta_p']), np.asarray(o['p_low']), np.asarray(o['au'])], axis=-1)


def credit_uncertainty_packed(p_star, packed):
    C = np.asarray(p_star).shape[-1]
    packed = np.asarray(packed)
    return credit_uncertainty([p_star, packed[:, C:2 * C], packed[:, :C], packed[:, 2 * C:]], C)''',
    "H14 targets", 1, 1)
# ---- H14 student heads
sub(r"^([ \t]+)delta_p = layers\.Dense\(num_classes, activation='sigmoid',\s*name='delta_p'\s*\)\(features\)\n",
    lambda m: m.group(0) + f"{m.group(1)}p_low   = layers.Dense(num_classes, activation='sigmoid', name='p_low')(features)   # FIX H14\n"
                           f"{m.group(1)}au      = layers.Dense(1, activation='sigmoid', name='au')(features)               # FIX H14\n",
    "H14 heads", 1, 1)
sub(r"outputs=\[p_star, delta_p\],", "outputs={'p_star': p_star, 'p_low': p_low, 'delta_p': delta_p, 'au': au},", "H14 outputs", 1, 1)
# ---- targets / data (C3: student validated on the validation split)
sub(r"^([ \t]+)p_star_tr, delta_p_tr = generate_credit_targets\((\w+), (\w+)\)", r"\1tgt_tr = generate_credit_targets(\2, \3)", "H14 tgt tr", 1, 1)
sub(r"^([ \t]+)p_star_te, delta_p_te = generate_credit_targets\((\w+), (\w+)\)",
    r"\1tgt_te = generate_credit_targets(\2, x_val)   # FIX C3: validation split, not test", "C3 tgt val", 1, 1)
sub(r"\(x_tr, \(p_star_tr, delta_p_tr\)\)", "(x_tr, tgt_tr)", "H14 ds tr", 1, 1)
sub(r"\((\w+), \(p_star_te, delta_p_te\)\)", "(x_val, tgt_te)", "C3 ds val", 1, 1)
sub(r"^([ \t]+)'delta_p': tf\.keras\.losses\.MeanSquaredError\(\),",
    r"\1'delta_p': tf.keras.losses.MeanSquaredError(),\n\1'p_low':   tf.keras.losses.MeanSquaredError(),\n\1'au':      tf.keras.losses.MeanSquaredError(),",
    "H14 losses", 1, 1)
sub(r"loss_weights=\{'p_star': 1\.0, 'delta_p': 0\.5\}", "loss_weights={'p_star': 1.0, 'p_low': 0.5, 'delta_p': 0.5, 'au': 0.5}", "H14 weights", 1, 1)
# ---- inference + maths
sub(r"^([ \t]*)(\w+), (\w+) = (student\w*)\.predict\(", r"\1\2, \3 = credit_predict(\4, ", "H14 predict", 2, 6)
sub(r"^([ \t]*)(\w+)\s*=\s*-np\.sum\((\w+) \* np\.log\(\3(?: \+ 1e-12)?\), axis=-1\)\n[ \t]*(\w+)\s*=\s*np\.mean\((\w+), axis=-1\)\n[ \t]*(\w+)\s*=\s*\2 \+ \4",
    r"\1_, _c_lo, _c_up, \2, \4, \6 = credit_uncertainty_packed(\3, \5)   # FIX H14 (normalised, AU+EU=TU)",
    "H14 maths", 2, 6)
sub(r"lower\s*=\s*(\w+),(\s*)upper\s*=\s*np\.minimum\(1\.0,\s*\1\s*\+\s*\w+\),", r"lower=_c_lo,\2upper=_c_up,", "H14 bounds", 0, 3)
sub(r"lower = p\* \(student softmax head\), upper = min\(1, p\* \+ delta_p\) \(student sigmoid head\)\.",
    "lower = p_low (student head, target = min over members), upper = min(1, p_low + delta_p).", "H14 note", 0, 3)
# ---- leftovers that would mean old maths survived
allsrc = "\n".join("".join(c["source"]) for c in cells())
for bad in [r"np\.mean\(delta_p\w*, axis=-1\)", r"p_star_true\s*=\s*p_min", r"= (student\w*)\.predict\(", r"p_star_\w+ \+ delta_p_\w+"]:
    m = re.search(bad, allsrc)
    if m: raise SystemExit(f"[{nb.path.name}] leftover old CREDIT code: {m.group(0)!r}")
common_fixes(nb, F)
print(nb.save())
