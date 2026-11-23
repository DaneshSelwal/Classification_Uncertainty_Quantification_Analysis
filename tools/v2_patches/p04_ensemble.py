import sys; sys.path.insert(0, sys.argv[2]); from nbpatch import NB, common_fixes
F = sys.argv[2]; nb = NB(sys.argv[1]); R = lambda f: open(f"{F}/{f}").read()
nb.rep('''# Primary split (GFNet / ViT)
x_train_full, x_test, y_train_full, y_test = train_test_split(
    X, y, train_size=TRAIN_PERCENT, random_state=SEED, stratify=y,
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full,
    test_size=VAL_SPLIT_FROM_TRAIN, random_state=SEED, stratify=y_train_full,
)
''', R("split_helper.py") + '''

SPLIT = make_shared_split(y_img, patch_size=PATCH_SIZE)
display(describe_split(SPLIT, num_classes))
x_train, y_train = X[SPLIT["train"]], y[SPLIT["train"]]
x_val,   y_val   = X[SPLIT["val"]],   y[SPLIT["val"]]
x_test,  y_test  = X[SPLIT["test"]],  y[SPLIT["test"]]
''', tag="C1/C4 shared split")
nb.rep('''# AlexNet legacy split (matches old single-head script behaviour)
x_train_alex, x_test_alex, y_train_alex, y_test_alex = train_test_split(
    X, y,
    train_size=ALEXNET_LEGACY_TRAIN_PERCENT,
    random_state=ALEXNET_LEGACY_SPLIT_SEED,
    stratify=y,
)''', '''# FIX C1: AlexNet members use the same shared split as GFNet / ViT
x_train_alex, x_test_alex, y_train_alex, y_test_alex = x_train, x_test, y_train, y_test''', tag="C1 AlexNet")
nb.rep('''            x_tr, y_tr                                         = x_train_alex, y_train_alex
            x_va, y_va                                         = x_test_alex,  y_test_alex''',
'''            # FIX C3: monitor on the validation split, never on the test split
            x_tr, y_tr                                         = x_train_alex, y_train_alex
            x_va, y_va                                         = x_val,        y_val''', tag="C3 AlexNet val")
if nb.has('''            history_obj = type('H', (), {'history': {
                'loss': [0.0]'''): nb.rep('''            history_obj = type('H', (), {'history': {
                'loss': [0.0], 'val_loss': [0.0],
                'accuracy': [0.0], 'val_accuracy': [0.0],
            }})()
            train_time_sec = 0.0''', '''            # FIX M18: a reused member has no training history -> NaN, not fake zeros
            history_obj = type('H', (), {'history': {
                'loss': [np.nan], 'val_loss': [np.nan],
                'accuracy': [np.nan], 'val_accuracy': [np.nan],
            }})()
            train_time_sec = np.nan''', tag="M18 reused history")
if nb.has('''        # Mirror member 1 to MODEL_DIR'''): nb.rep('''        # Mirror member 1 to MODEL_DIR for downstream visualization and baseline compatibility
        if i == 1:
            shutil.copy(final_path, MODEL_DIR / f'{model_name}_best.keras')
            shutil.copy(final_path, MODEL_DIR / f'{model_name}_final.keras')
''', '''        # FIX C2: do NOT overwrite the baseline {model}_best/_final.keras with ensemble member 1;
        # later notebooks (DAPM, ...) must keep using the baseline backbone.
''', tag="C2 no overwrite")
if nb.has("if (i == 1) else str(final_path),"): nb.rep("            'best_model_path':    str(MODEL_DIR / f'{model_name}_best.keras') if (i == 1) else str(final_path),",
       "            'best_model_path':    str(final_path),", tag="C2 path col")
_vis_old = """        candidates = [
            model_dir / f'{model_key}_best.keras',
            model_dir / f'{model_key}_final.keras',
            model_dir / 'ensembles' / f'{model_key}_ens_1_final.keras',
            model_dir / 'ensembles' / f'{model_key}_ens_1_best.keras',
            ensemble_dir / f'{model_key}_ens_1_final.keras',
            ensemble_dir / f'{model_key}_ens_1_best.keras',"""
if nb.has(_vis_old):
    nb.rep(_vis_old, """        # FIX C2: visualise ensemble member 1 first (baseline files are no longer overwritten)
        candidates = [
            ensemble_dir / f'{model_key}_ens_1_final.keras',
            ensemble_dir / f'{model_key}_ens_1_best.keras',
            model_dir / 'ensembles' / f'{model_key}_ens_1_final.keras',
            model_dir / 'ensembles' / f'{model_key}_ens_1_best.keras',
            model_dir / f'{model_key}_best.keras',
            model_dir / f'{model_key}_final.keras',""", tag="C2 vis order")
common_fixes(nb, F)
print(nb.save())
