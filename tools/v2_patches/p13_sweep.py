# Generic consistency sweep over the 12 pipeline notebooks of every suite.
#  K1: every xlsxwriter ExcelWriter gets nan_inf_to_errors (q_hat can be +inf).
#  K4: unguarded `from google.colab import drive` / `drive.mount(...)` (DIAS, 6-Band) made safe outside Colab.
# usage: python3 -I p13_sweep.py <repo_root>
import json, re, sys
from pathlib import Path

root = Path(sys.argv[1])
PIPE = re.compile(r"/(baseline|sacp|ensemble|credit|multi_cp|dapm|multicp_sacp)/")
XL = re.compile(r"(pd\.ExcelWriter\([^()\n]*?engine\s*=\s*(['\"])xlsxwriter\2)\)")
IMP = re.compile(r"^(\s*)from google\.colab import drive\s*$")
MNT = re.compile(r"^(\s*)drive\.mount\((.*)\)\s*$")
tot = {"K1": 0, "K4": 0}
for p in sorted(root.glob("examples/*_uncertainty_quantification/*/*.ipynb")):
    if not PIPE.search(str(p)): continue
    nb = json.loads(p.read_text()); n = {"K1": 0, "K4": 0}
    for c in nb["cells"]:
        if c["cell_type"] != "code": continue
        out = []
        lines = "".join(c["source"]).split("\n")
        for i, l in enumerate(lines):
            if "nan_inf_to_errors" not in l and XL.search(l):
                l = XL.sub(r"\1, engine_kwargs={'options': {'nan_inf_to_errors': True}})", l)
                l += "  # FIX K1"; n["K1"] += 1
            m = IMP.match(l)
            prev = lines[i - 1].strip() if i else ""
            if m and prev != "try:  # FIX K4: also runs outside Colab (Kaggle / local)":
                ind = m.group(1)
                out += [f"{ind}try:  # FIX K4: also runs outside Colab (Kaggle / local)",
                        f"{ind}    from google.colab import drive", f"{ind}except ImportError:", f"{ind}    drive = None"]
                n["K4"] += 1; continue
            m = MNT.match(l)
            if m:
                l = f"{m.group(1)}if drive is not None: drive.mount({m.group(2)})  # FIX K4"; n["K4"] += 1
            out.append(l)
        s = "\n".join(out)
        if s != "".join(c["source"]):
            c["source"] = s.splitlines(keepends=True)
    if any(n.values()):
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print(p.relative_to(root), n)
    for k in tot: tot[k] += n[k]
print("TOTAL", tot)
