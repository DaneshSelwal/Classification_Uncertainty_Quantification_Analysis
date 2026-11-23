import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
nb.rep('''x_train_full, x_test, y_train_full, y_test, coords_train_full, coords_test = train_test_split(
    X_all, y_all, coords_all,
    train_size=TRAIN_PERCENT, random_state=SEED, stratify=y_all,
)
# Further split train → train / val (20 % of train)
x_train, x_val, y_train, y_val, coords_train, coords_val = train_test_split(
    x_train_full, y_train_full, coords_train_full,
    test_size=VAL_SPLIT_FROM_TRAIN, random_state=SEED, stratify=y_train_full,
)''', R("split_helper.py") + '''

# FIX C1/C4: shared split (identical in every notebook)
SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_train, y_train, coords_train = X_all[SPLIT["train"]], y_all[SPLIT["train"]], coords_all[SPLIT["train"]]
x_val,   y_val,   coords_val   = X_all[SPLIT["val"]],   y_all[SPLIT["val"]],   coords_all[SPLIT["val"]]
x_test,  y_test,  coords_test  = X_all[SPLIT["test"]],  y_all[SPLIT["test"]],  coords_all[SPLIT["test"]]''', tag="C1 shared split")
common_fixes(nb, F)
print(nb.save())
