import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB
F = sys.argv[2]; nb = NB(sys.argv[1])
SPLIT = open(f"{F}/split_helper.py").read()

# C1/C3/C4: one shared spatial split for all models; AlexNet uses it too, validated on val (not test)
nb.rep('''# Primary stratified split used by GFNet and ViT
x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y,
    train_size=TRAIN_PERCENT,
    random_state=SEED,
    stratify=y,
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=VAL_SPLIT_FROM_TRAIN,
    random_state=SEED,
    stratify=y_train_full,
)
''', SPLIT + '''

SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_train, y_train = X[SPLIT["train"]], y[SPLIT["train"]]
x_val,   y_val   = X[SPLIT["val"]],   y[SPLIT["val"]]
x_test,  y_test  = X[SPLIT["test"]],  y[SPLIT["test"]]   # held-out pool = cal + eval
''', tag="C1/C3/C4 shared split")

nb.rep('''# AlexNet legacy split — matches the original single-head notebook for
# uncertainty recovery (different seed and no val split)
x_train_alex, x_test_alex, y_train_alex, y_test_alex = train_test_split(
    X, y,
    train_size=ALEXNET_LEGACY_TRAIN_PERCENT,
    random_state=ALEXNET_LEGACY_SPLIT_SEED,
    stratify=y,
)
y_train_alex_cat = keras.utils.to_categorical(y_train_alex, num_classes)
y_test_alex_cat  = keras.utils.to_categorical(y_test_alex,  num_classes)
''', '''# FIX C1: AlexNet now uses the same shared split as GFNet / ViT (no separate seed-10 split),
# so its held-out pixels are never part of its training data in later UQ notebooks.
x_train_alex, x_test_alex, y_train_alex, y_test_alex = x_train, x_test, y_train, y_test
y_train_alex_cat = keras.utils.to_categorical(y_train_alex, num_classes)
y_test_alex_cat  = keras.utils.to_categorical(y_test_alex,  num_classes)
''', tag="C1 AlexNet split")

# C3: AlexNet validated/checkpointed on the validation set, not on the test set
nb.rep('''        x_tr, y_tr         = x_train_alex, y_train_alex
        x_va, y_va         = x_test_alex,  y_test_alex
        x_te, y_te, y_te_cat = x_test_alex, y_test_alex, y_test_alex_cat
        x_eval, y_eval, y_eval_cat = x_test_alex, y_test_alex, y_test_alex_cat''',
'''        # FIX C3: checkpoint selection on the validation split (was: on the test split)
        x_tr, y_tr         = x_train_alex, y_train_alex
        x_va, y_va         = x_val,        y_val
        x_te, y_te, y_te_cat = x_test_alex, y_test_alex, y_test_alex_cat
        x_eval, y_eval, y_eval_cat = x_val, y_val, y_val_cat''', tag="C3 AlexNet val")

# M16: evaluate the best checkpoint (the one every later notebook loads)
nb.rep('''
    y_eval_prob = model.predict(''', '''
    # FIX M16: report metrics for the *best* checkpoint, which is the file later notebooks load
    if best_path.exists():
        model = keras.models.load_model(best_path, custom_objects=CUSTOM_OBJECTS_TRAIN,
                                        compile=False, safe_mode=False)

    y_eval_prob = model.predict(''', tag="M16 eval best")
nb.rep('''def train_save_evaluate(model_name, model_builder, capacity_tag="max"):''',
'''CUSTOM_OBJECTS_TRAIN = {"PatchExtractor": PatchExtractor, "PatchPositionEncoder": PatchPositionEncoder,
                        "GlobalFilterLayer": GlobalFilterLayer, "PatchEncoderWithCLS": PatchEncoderWithCLS}
try:
    keras.config.enable_unsafe_deserialization()
except Exception:
    pass


def train_save_evaluate(model_name, model_builder, capacity_tag="max"):''', tag="M16 custom objs")

# C5: GFNet FFT axes
nb.rep('''        x_fft      = tf.signal.fft2d(tf.cast(x_2d, tf.complex64))
        w_complex  = tf.complex(self.w_real, self.w_imag)
        x_filtered = x_fft * w_complex
        x_spatial  = tf.math.real(tf.signal.ifft2d(x_filtered))
''', open(f"{F}/gfnet_fix.txt").read(), tag="C5 GFNet FFT")
print(nb.save())
