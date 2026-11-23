import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes, fix_gfnet_rfft
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
nb.rep('''x_train, x_test, y_train, y_test = train_test_split(
    X, Y, train_size=(train_percent / 100), stratify=Y, random_state=10
)''', R("split_helper.py") + '''

# FIX C1/C3: shared split; the multi-head models are checkpointed on the validation split
SPLIT = make_shared_split(y_img, patch_size=P_S)
display(describe_split(SPLIT, num_classes))
x_train, y_train = X[SPLIT["train"]], Y[SPLIT["train"]]
x_val,   y_val   = X[SPLIT["val"]],   Y[SPLIT["val"]]
x_test,  y_test  = X[SPLIT["test"]],  Y[SPLIT["test"]]''', tag="C1/C3 shared split")
nb.rep("        validation_data=(x_test, [y_test] * 7),", "        validation_data=(x_val, [y_val] * 7),   # FIX C3 (was: test split)", tag="C3 val")
assert fix_gfnet_rfft(nb)
common_fixes(nb, F)
print(nb.save())
