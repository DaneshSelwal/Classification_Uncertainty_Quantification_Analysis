import json, re, sys, glob, os
SIG = {
 "split not shared": r"^(?![ \t]*#).*\btrain_test_split\(",
 "old qhat": r"q_level = min\(1\.0, np\.ceil\(\(n \+ 1\)",
 "multicp no correction": r"np\.quantile\(cal_true, 1 - alpha",
 "class-7 hack": r"y_flat == 7",
 "old credal p_star": r"p_min / \(tf\.reduce_sum\(p_min",
 "old CREDIT eu": r"np\.mean\(delta_p\w*, axis=-1\)",
 "t-test uq": r"_safe_ttest_ind\(\w+",
 "val on test": r"validation_data=\(x_test|x_va, y_va\s*=\s*x_test_alex",
 "member1 overwrite": r"shutil\.copy\(final_path, MODEL_DIR / f'\{model_name\}_best",
 "rc3p on eval": r"compute_rc3p_qhats_and_sets\(\s*prob_cal, y_cal, prob_eval",
 "fake zero history": r"'loss': \[0\.0\]",
 "old sacp lambda": r"\(1 - lambda_?\) \* ori \+ neigh|ori \+ lambda_? \* neigh(?!.*1)",
}
def scan(p):
    hits = []
    for i, c in enumerate(json.load(open(p))["cells"]):
        if c["cell_type"] != "code": continue
        s = "".join(c["source"])
        # train_test_split inside helper defs is fine (split_calib_eval etc. now unused) -> only top-level calls count
        for k, pat in SIG.items():
            for m in re.finditer(pat, s, re.M):
                line = s[s.rfind("\n", 0, m.start()) + 1: s.find("\n", m.start())]
                if k == "split not shared" and (line.startswith((" ", "\t")) or "_np_split" in line): continue
                hits.append((k, line.strip()[:90]))
        if ("fft2d(" in s and "FIX C5" not in s): hits.append(("GFNet axes", "fft2d without FIX C5"))
    return hits
roots = sys.argv[1:]
for root in roots:
    for p in sorted(glob.glob(f"{root}/**/*.ipynb", recursive=True)):
        h = scan(p)
        if h: print(p.replace(root, "")[:80]); [print("    ", k, "|", l) for k, l in sorted(set(h))]
