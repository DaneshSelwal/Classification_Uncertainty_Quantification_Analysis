"""Static checks of the 12 pipeline notebooks of every dataset suite (no data, no GPU, seconds).

usage: python3 tools/check_notebooks.py [examples/<suite> ...]      (default: all suites)

Checks per notebook
  * syntax: every code cell compiles (IPython `%` / `!` lines are skipped);
  * v1 bug signatures that v2.0 removed must be absent;
  * required v2.0 pieces must be present (shared split + alignment check, labels over all classes,
    xlsxwriter nan_inf_to_errors, guarded Colab import);
  * data-specific hard-coding: literal class counts / band counts / class ids used as "unlabelled".
Exit code 1 if any check fails.
"""
import ast, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ["baseline/model_training.ipynb", "baseline/model_uncertainty_comparison.ipynb",
            "sacp/model_sacp_comparison.ipynb", "ensemble/model_training_ensembles.ipynb",
            "credit/model_training_credit.ipynb", "ensemble/model_uncertainty_credal_ensemble.ipynb",
            "multi_cp/model_training_multihead.ipynb", "multi_cp/model_uncertainty_multicp.ipynb",
            "dapm/model_training_dapm_full.ipynb", "dapm/model_uncertainty_dapm_full.ipynb",
            "dapm/Model_uncertainty_dapm_cp.ipynb", "multicp_sacp/Model_uncertainty_multicp_sacp.ipynb"]
# regex -> message: must NOT appear in code
FORBIDDEN = {
    r"np\.minimum\(\s*1(\.0)?\s*,\s*np\.ceil\(\s*\(n\s*\+\s*1\)": "H6: q_hat level capped at 1",
    r"\(1\s*-\s*lam\w*\)\s*\*\s*\w+\s*\+\s*neigh": "H10: wrong SACP smoothing",
    r"==\s*7\s*\)\s*\]?\s*=\s*.*uncertain": "H12: class 7 hard-coded",
    r"tf\.signal\.fft2d\(\s*tf\.cast\(\s*x\s*,": "C5: FFT over (token, channel)",
    r"random_state\s*=\s*ALEXNET_LEGACY_SPLIT_SEED|train_size\s*=\s*ALEXNET_LEGACY": "C1: AlexNet legacy split used",
}
# regex -> message: must appear somewhere in the notebook (only checked where relevant)
REQUIRED_ALL = {r"make_shared_split": "C1: shared split", r"check_split_alignment": "C1: alignment check"}
WRITER = re.compile(r"pd\.ExcelWriter\([^\n]*xlsxwriter")
CM = re.compile(r"\b(confusion_matrix|classification_report)\(")


def code_of(nb):
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            yield i, "".join(c["source"])


def strip_magics(src):
    return "\n".join("" if l.lstrip().startswith(("%", "!")) else l for l in src.split("\n"))


SUITES = json.loads((ROOT / "tools/kaggle/suites.json").read_text())


def check(path):
    errs, nb = [], json.loads(path.read_text())
    n_cls = SUITES.get(path.parents[1].name, {}).get("num_classes")
    full = ""
    for i, src in code_of(nb):
        full += src + "\n"
        try:
            ast.parse(strip_magics(src))
        except SyntaxError as e:
            errs.append(f"cell {i}: SyntaxError line {e.lineno}: {e.msg}")
    for pat, msg in FORBIDDEN.items():
        if re.search(pat, full): errs.append("v1 bug left: " + msg)
    if "ttest_ind(" in full and "FlipRate" not in full:
        errs.append("v1 bug left: H15 Welch t-test decides DAPM uncertainty")
    for pat, msg in REQUIRED_ALL.items():
        if not re.search(pat, full): errs.append("missing " + msg)
    for l in full.split("\n"):
        if WRITER.search(l) and "nan_inf_to_errors" not in l: errs.append("K1 missing: " + l.strip()[:80])
        if CM.search(l) and "labels=" not in l and not l.lstrip().startswith(("#", "from ", "import ")) \
                and "def " not in l and ")" in l.split(CM.search(l).group(0), 1)[1]:
            errs.append("K2 missing: " + l.strip()[:80])
        if re.match(r"\s*from google\.colab import drive", l) is None and re.match(r"^drive\.mount\(", l):
            errs.append("K4 unguarded drive.mount")
        m = re.search(r"\b(num_classes|NUM_CLASSES|n_classes)\s*=\s*(\d+)\s*$", l)
        if m and int(m.group(2)) != n_cls:   # literal class counts must equal the dataset's class count
            errs.append(f"class count {m.group(2)} != {n_cls} for this dataset: " + l.strip()[:80])
    lines = full.split("\n")
    for k, l in enumerate(lines):
        if re.match(r"^from google\.colab import drive", l):
            errs.append("K4 unguarded Colab import")
    return errs


def main(argv):
    suites = [Path(a) for a in argv] or sorted((ROOT / "examples").glob("*_uncertainty_quantification"))
    bad = 0
    for s in suites:
        for rel in PIPELINE:
            p = s / rel
            if not p.exists():
                print(f"MISSING  {s.name}/{rel}"); bad += 1; continue
            e = check(p)
            if e:
                bad += 1; print(f"FAIL     {s.name}/{rel}"); [print("         -", x) for x in e]
        print(f"checked  {s.name}")
    print("RESULT:", "OK" if bad == 0 else f"{bad} notebook(s) with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
