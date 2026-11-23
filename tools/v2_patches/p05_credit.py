import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
if nb.has('''# AlexNet legacy split constants
ALEXNET_LEGACY_TRAIN_PERCENT = 0.20'''): nb.rep('''# AlexNet legacy split constants
ALEXNET_LEGACY_TRAIN_PERCENT = 0.20
ALEXNET_LEGACY_SPLIT_SEED = 42
''', '''# FIX H13: the silent override of the AlexNet split (20 % / seed 42) is removed;
# every model now uses the shared split below.
''', tag="H13 remove override")
nb.rep('''x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y, train_size=TRAIN_PERCENT, random_state=SEED, stratify=y
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=VAL_SPLIT_FROM_TRAIN, random_state=SEED, stratify=y_train_full
)
''', R("split_helper.py") + '''

SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_train, y_train = X[SPLIT["train"]], y[SPLIT["train"]]
x_val,   y_val   = X[SPLIT["val"]],   y[SPLIT["val"]]
x_test,  y_test  = X[SPLIT["test"]],  y[SPLIT["test"]]
''', tag="C1 shared split")
nb.rep('''x_train_alex, x_test_alex, y_train_alex, y_test_alex = train_test_split(
    X, y,
    train_size=ALEXNET_LEGACY_TRAIN_PERCENT,
    random_state=ALEXNET_LEGACY_SPLIT_SEED,
    stratify=y
)''', '''# FIX C1/H13: AlexNet shares the split of the other backbones
x_train_alex, x_test_alex, y_train_alex, y_test_alex = x_train, x_test, y_train, y_test''', tag="C1 AlexNet")

# H14: teacher targets
if nb.has('''    Run each teacher over x_data and derive CREDIT interval targets.'''): nb.rep('''def generate_credit_targets(ensemble_paths, x_data, batch_size=128):
    """
    Run each teacher over x_data and derive CREDIT interval targets.

    Returns:
        p_star_true  : (N, C) normalised per-class minimum  (aleatoric proxy)
        delta_p_true : (N, C) per-class prediction spread   (epistemic proxy)
    """''', '''def generate_credit_targets(ensemble_paths, x_data, batch_size=128):
    """
    Run each teacher over x_data and derive CREDIT targets (FIX H14).

    Returns a dict of targets for the student heads:
        p_star  : (N, C) ensemble mean probability (a proper distribution)
        p_low   : (N, C) per-class lower credal bound  = min over members
        delta_p : (N, C) credal interval width          = max - min over members
        au      : (N, 1) aleatoric part = mean member entropy / ln(C)   (in [0, 1])
    """''', tag="H14 doc")
nb.rep('''    delta_p_true = p_max - p_min
    p_star_true  = p_min / (tf.reduce_sum(p_min, axis=-1, keepdims=True) + 1e-12)
    return p_star_true, delta_p_true''', '''    n_cls   = int(stacked.shape[-1])
    p_mean  = tf.reduce_mean(stacked, axis=0)
    ent_m   = -tf.reduce_sum(stacked * tf.math.log(tf.clip_by_value(stacked, 1e-12, 1.0)), axis=-1)
    au_t    = tf.reduce_mean(ent_m, axis=0, keepdims=False)[:, None] / np.log(n_cls)
    return {'p_star': p_mean, 'p_low': p_min, 'delta_p': p_max - p_min,
            'au': tf.clip_by_value(au_t, 0.0, 1.0)}


def credit_uncertainty(outputs, num_classes):
    """FIX H14: coherent CREDIT uncertainty from the student's four heads (all in [0, 1]).

    TU = H(p_star) / ln C ; AU = predicted mean member entropy (capped at TU) ; EU = TU - AU
    (mutual-information decomposition, so AU + EU = TU holds exactly).
    Credal bounds: lower = p_low, upper = min(1, p_low + delta_p).
    """
    p_star, p_low, delta_p, au_hat = [np.asarray(o, dtype=np.float64) for o in outputs]
    p_star = np.clip(p_star, 1e-12, 1.0)
    tu = -np.sum(p_star * np.log(p_star), axis=-1) / np.log(num_classes)
    au = np.minimum(au_hat.reshape(-1), tu)
    eu = tu - au
    lower = np.clip(p_low, 0.0, 1.0)
    upper = np.clip(lower + delta_p, 0.0, 1.0)
    return p_star, lower, upper, au, eu, tu''', tag="H14 targets")
nb.rep('''    p_star  = layers.Dense(num_classes, activation='softmax',  name='p_star' )(features)
    delta_p = layers.Dense(num_classes, activation='sigmoid',  name='delta_p')(features)

    return tf.keras.Model(
        inputs=base_model.input,
        outputs=[p_star, delta_p],
        name='CREDIT_Student'
    )''', '''    p_star  = layers.Dense(num_classes, activation='softmax',  name='p_star' )(features)
    p_low   = layers.Dense(num_classes, activation='sigmoid',  name='p_low'  )(features)   # FIX H14
    delta_p = layers.Dense(num_classes, activation='sigmoid',  name='delta_p')(features)
    au      = layers.Dense(1,           activation='sigmoid',  name='au'     )(features)   # FIX H14

    return tf.keras.Model(
        inputs=base_model.input,
        outputs={'p_star': p_star, 'p_low': p_low, 'delta_p': delta_p, 'au': au},
        name='CREDIT_Student'
    )''', tag="H14 student heads")
nb.rep('''    x_tr, x_te = (
        (x_train_alex, x_test_alex) if model_name == 'AlexNet_CNN'
        else (x_train, x_test)
    )

    # ── 3. Generate soft targets ───────────────────────────────────────────────
    print('  Generating training targets...')
    p_star_tr, delta_p_tr = generate_credit_targets(ensemble_paths, x_tr)

    print('  Generating test targets...')
    p_star_te, delta_p_te = generate_credit_targets(ensemble_paths, x_te)

    # ── 4. Build tf.data pipelines ─────────────────────────────────────────────
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((x_tr, (p_star_tr, delta_p_tr)))
        .shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset
        .from_tensor_slices((x_te, (p_star_te, delta_p_te)))
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )''', '''    # FIX C3: the student is checkpointed on the validation split, the test split stays unseen
    x_tr, x_va, x_te = x_train, x_val, x_test

    # ── 3. Generate soft targets ───────────────────────────────────────────────
    print('  Generating training targets...')
    tgt_tr = generate_credit_targets(ensemble_paths, x_tr)

    print('  Generating validation targets...')
    tgt_va = generate_credit_targets(ensemble_paths, x_va)

    # ── 4. Build tf.data pipelines ─────────────────────────────────────────────
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((x_tr, tgt_tr))
        .shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset
        .from_tensor_slices((x_va, tgt_va))
        .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )''', tag="C3/H14 data")
nb.rep('''        loss={
            'p_star':  tf.keras.losses.KLDivergence(),
            'delta_p': tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={'p_star': 1.0, 'delta_p': 0.5},   # lambda = 0.5''', '''        loss={
            'p_star':  tf.keras.losses.KLDivergence(),
            'p_low':   tf.keras.losses.MeanSquaredError(),
            'delta_p': tf.keras.losses.MeanSquaredError(),
            'au':      tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={'p_star': 1.0, 'p_low': 0.5, 'delta_p': 0.5, 'au': 0.5},   # lambda = 0.5''', tag="H14 losses")
nb.rep('''    student.load_weights(best_path)
    p_star_pred, delta_p_pred = student.predict(x_te, batch_size=BATCH_SIZE)

    au = -np.sum(p_star_pred * np.log(p_star_pred + 1e-12), axis=-1)
    eu =  np.mean(delta_p_pred, axis=-1)
    tu =  au + eu

    train_results.append({''', '''    student.load_weights(best_path)
    _o = student.predict(x_va, batch_size=BATCH_SIZE)
    _, _, _, au, eu, tu = credit_uncertainty([_o['p_star'], _o['p_low'], _o['delta_p'], _o['au']], num_classes)

    train_results.append({''', tag="H14 train summary")
nb.rep('''    p_star_pred, delta_p_pred = student.predict(x_te, batch_size=BATCH_SIZE)

    au = -np.sum(p_star_pred * np.log(p_star_pred + 1e-12), axis=-1)
    eu =  np.mean(delta_p_pred, axis=-1)
    tu =  au + eu

    y_pred = np.argmax(p_star_pred, axis=-1)

    # Credal prediction sets: lower = p*, upper = min(1, p* + delta_p)
    credal_m = credal_set_metrics(
        lower=p_star_pred,
        upper=np.minimum(1.0, p_star_pred + delta_p_pred),
        y_true=y_te_true, num_classes=num_classes
    )''', '''    _o = student.predict(x_te, batch_size=BATCH_SIZE)
    p_star_pred, c_lower, c_upper, au, eu, tu = credit_uncertainty(
        [_o['p_star'], _o['p_low'], _o['delta_p'], _o['au']], num_classes)

    y_pred = np.argmax(p_star_pred, axis=-1)

    # FIX H14: credal prediction sets from proper bounds [p_low, p_low + delta_p]
    credal_m = credal_set_metrics(
        lower=c_lower, upper=c_upper,
        y_true=y_te_true, num_classes=num_classes
    )''', tag="H14 eval")
nb.rep('''    p_star_scene, delta_p_scene = student_model.predict(
        scene_pixels, batch_size=2048, verbose=1
    )
    p_star_scene = np.clip(p_star_scene, 1e-12, 1.0)
    n_cls = p_star_scene.shape[-1]

    au_scene = -np.sum(p_star_scene * np.log(p_star_scene), axis=-1)
    eu_scene =  np.mean(delta_p_scene, axis=-1)
    tu_scene =  au_scene + eu_scene
''', '''    _o = student_model.predict(scene_pixels, batch_size=2048, verbose=1)
    n_cls = _o['p_star'].shape[-1]
    # FIX H14 / M22: normalised AU / EU / TU in [0, 1] (comparable across datasets)
    p_star_scene, _, _, au_scene, eu_scene, tu_scene = credit_uncertainty(
        [_o['p_star'], _o['p_low'], _o['delta_p'], _o['au']], n_cls)
''', tag="H14 scene")
nb.rep("lower = p* (student softmax head), upper = min(1, p* + delta_p) (student sigmoid head).",
       "lower = p_low (student head, target = min over members), upper = min(1, p_low + delta_p).", tag="H14 note")
common_fixes(nb, F)
print(nb.save())
