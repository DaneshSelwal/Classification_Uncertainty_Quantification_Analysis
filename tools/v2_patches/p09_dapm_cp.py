import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes, fix_cp_methods
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
nb.rep('''_, x_test_pool, _, y_test_pool = train_test_split(
    X_all, y_all,
    train_size=TRAIN_PERCENT,
    random_state=SEED,
    stratify=y_all,
)

x_cal, x_eval, y_cal, y_eval = split_calib_eval(
    x_test_pool, y_test_pool,
    seed=SEED,
    calib_fraction=CALIB_FRACTION_OF_TEST,
)''', R("split_helper.py") + '''

# FIX C1: shared split -> identical cal / eval pixels for every model and notebook
SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_cal,  y_cal  = X_all[SPLIT["cal"]],  y_all[SPLIT["cal"]]
x_eval, y_eval = X_all[SPLIT["eval"]], y_all[SPLIT["eval"]]''', tag="C1 shared split")
fix_cp_methods(nb, F)
common_fixes(nb, F)
print(nb.save())
