"""Local data dry run (CPU, no TensorFlow needed, about a minute per suite).

Executes every pipeline notebook of a suite from the top up to the shared split (`SPLIT = make_shared_split`)
plus the following cells until the first cell that trains or loads a model. TensorFlow / Keras are stubbed, so
only the data path logic, data loading, label handling, shared split and split-alignment check really run.
This catches data-specific errors (file names, shapes, class counts, label conventions) before a GPU run.

usage: python3 tools/dry_run_data.py examples/<suite> <input_root> [--fake-models] [notebook ...]
  --fake-models : place empty model files where the training stages save them (models/, results/models/,
                  results/models/ensembles/, Classification/dapm/dapm_full_artifacts/) to test that the
                  later stages find the models of the same run (FIX K5).
  <input_root> plays the role of /kaggle/input: it must contain datasets/<owner>/<slug>/<files>, e.g. made with
  `kaggle datasets download <owner>/<slug> -p <input_root>/datasets/<owner>/<slug> --unzip`.
  Real-name links from tools/kaggle/suites.json are created next to the token files (as the Kaggle runner does).
"""
import contextlib, importlib.abc, importlib.machinery, io, json, os, re, sys, tempfile, time, traceback, types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "kaggle"))
PIPELINE = [rel for _, rel in json.loads(json.dumps(__import__("ast").literal_eval(
    re.search(r"PIPELINE = (\[.*?\])\n", (ROOT / "tools/kaggle/make_suite_kernel.py").read_text(), re.S).group(1))))]
STUB_PREFIXES = ("tensorflow", "keras", "tf_keras", "google.colab")
STOP = re.compile(r"\.fit\(|load_model\(|\.predict\(|keras\.models\.|build_\w+_model\(|train_\w+\(|"
                  r"for\s+\w+\s+in\s+.*MODEL|ensemble_member|\.load_weights\(")


class _Meta(type):
    def __getattr__(cls, n):
        if n.startswith("__"): raise AttributeError(n)
        return _Any


class _Any(metaclass=_Meta):
    """Stand-in for any TensorFlow / Keras object: callable, subscriptable, subclassable."""
    def __init__(self, *a, **k): pass
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return _Any()
    def __call__(self, *a, **k): return _Any()
    def __getitem__(self, k): return _Any()
    def __iter__(self): return iter(())
    def __len__(self): return 0
    def __bool__(self): return False
    def __repr__(self): return "<stub>"
    __add__ = __radd__ = __sub__ = __mul__ = __rmul__ = __truediv__ = __call__


class _StubModule(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        return _Any


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(STUB_PREFIXES):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = _StubModule(spec.name); m.__path__ = []; m.__spec__ = spec; m.__version__ = "2.17.0"
        return m

    def exec_module(self, module):
        pass


def prepare_inputs(suite, input_root):
    cfg = json.loads((ROOT / "tools/kaggle/suites.json").read_text())[suite]
    for p in list(Path(input_root).rglob("*")):
        if p.is_file() and p.name in cfg["files"]:
            q = p.parent / cfg["files"][p.name]
            if not q.exists(): q.symlink_to(p.name)
    return cfg


def transform(src, input_root, work, colab_root):
    s = src.replace("/kaggle/input", str(input_root)).replace("/kaggle/working", str(work))
    s = re.sub(r"(['\"])/kaggle\1", lambda m: m.group(1) + str(work) + m.group(1), s)   # Kaggle detection
    s = s.replace("/content/drive/My Drive/Classification", str(colab_root)).replace("/content/", str(work) + "/content_")
    s = s.replace("from google.colab import drive", "drive = None")
    s = s.replace("if drive is not None: drive.mount", "if False: drive.mount")
    return "\n".join("" if l.lstrip().startswith(("%", "!")) else l for l in s.split("\n"))


def run_notebook(path, input_root, work, colab_root, timeout_s=600):
    nb = json.loads(path.read_text())
    try:
        from IPython.display import display, HTML, Markdown
    except Exception:
        display = print; HTML = Markdown = str
    ns = {"__name__": "__main__", "display": display, "HTML": HTML, "Markdown": Markdown}
    split_seen, log, t0 = False, [], time.time()
    cwd = os.getcwd(); os.chdir(work)
    try:
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] != "code": continue
            src = "".join(c["source"])
            if split_seen and STOP.search(src):
                return "OK", f"stopped before model cell {i}", ns
            if time.time() - t0 > timeout_s:
                return "TIMEOUT", f"cell {i}", ns
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(transform(src, input_root, work, colab_root), f"{path.name}[{i}]", "exec"), ns)
            except Exception as e:
                if split_seen and re.search(r"(?i)\.keras|\.h5|weights|model file|checkpoint", str(e)):
                    return "OK", f"stopped at cell {i}: needs model files of an earlier stage", ns
                if not split_seen and re.search(r"(?i)\.keras|\.h5|weights|model file|checkpoint", str(e)):
                    return "NEEDS_MODELS", f"cell {i}: model files of an earlier stage are loaded before the split", ns
                tb = traceback.format_exc().strip().split("\n")
                where = next((l for l in reversed(tb) if f"{path.name}[" in l), "")
                return "FAIL", f"cell {i}: {type(e).__name__}: {str(e)[:300]} | {where.strip()[:160]}", ns
            out = buf.getvalue()
            log += [l for l in out.split("\n") if "split alignment" in l or "Data file" in l]
            if re.search(r"^\s*SPLIT\s*=\s*make_shared_split\(", src, re.M):
                split_seen = True
        return ("OK" if split_seen else "NO_SPLIT"), "end of notebook", ns
    finally:
        os.chdir(cwd)
        import matplotlib.pyplot as plt; plt.close("all")


def main():
    suite_dir, input_root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    fake = "--fake-models" in sys.argv
    only = [a for a in sys.argv[3:] if a != "--fake-models"]
    suite = suite_dir.name
    cfg = prepare_inputs(suite, input_root)
    sys.meta_path.insert(0, _StubFinder())
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; plt.show = lambda *a, **k: None
    work = Path(tempfile.mkdtemp(prefix="dryrun_")); colab = work; (colab / "data").mkdir(parents=True)   # as the runner
    for p in input_root.rglob("*"):
        if (p.is_file() or p.is_symlink()) and not (colab / "data" / p.name).exists():
            (colab / "data" / p.name).symlink_to(p.resolve())
    if fake:
        M = ("AlexNet_CNN", "GFNet", "ViT_UNet")
        files = [f"models/{m}_{k}.keras" for m in M for k in ("best", "final")]
        files += [f"results/models/ensembles/{m}_ens_{i}_final.keras" for m in M for i in range(1, 6)]
        files += [f"results/models/{m}_MultiHead_{k}.keras" for m in M for k in ("best", "final")]
        files += [f"Classification/dapm/dapm_full_artifacts/{m}_dapm_full_{w}.weights.h5" for m in M
                  for w in ("encoder", "source_decoder", "target_decoder", "classifier", "discriminator", "diffusion")]
        files += [f"Classification/dapm/dapm_full_artifacts/{m}_dapm_full_config.json" for m in M]
        # Colab-style layout (6-Band; PROJECT_ROOT = /kaggle/working through the runner's Colab mapping)
        files += [f"baseline/models/{m}_{k}.keras" for m in M for k in ("best", "final")]
        files += [f"ensemble/models/ensembles/{m}_ens_{i}_final.keras" for m in M for i in range(1, 6)]
        files += [f"multicp/models/{m}_MultiHead_{k}.keras" for m in M for k in ("best", "final")]
        files += [f"dapm/dapm_full_artifacts/{m}_dapm_full_{w}" for m in M
                  for w in ("encoder.weights.h5", "source_decoder.weights.h5", "target_decoder.weights.h5",
                            "classifier.weights.h5", "discriminator.weights.h5", "diffusion.weights.h5", "config.json")]
        for f in files:
            (work / f).parent.mkdir(parents=True, exist_ok=True)
            (work / f).write_text("{}") if f.endswith(".json") else (work / f).touch()
    bad = 0
    for rel in PIPELINE:
        if only and not any(o in rel for o in only): continue
        status, info, ns = run_notebook(suite_dir / rel, input_root, work, colab)
        split = ns.get("SPLIT")
        extra = ""
        if split is not None:
            try:
                roles = {k: int(len(split[k])) for k in ("train", "val", "cal", "eval") if k in split}
                extra = f" | split {roles}"
                lab = split["labels"]; nc0 = int(lab.max()) + 1
                empty = [c for c in range(nc0) if (lab[split["train"]] == c).sum() and not (lab[split["eval"]] == c).sum()]
                if empty: extra += f" | classes with no eval pixel: {empty}"
            except Exception:
                extra = " | split built"
            ncls = ns.get("num_classes", ns.get("NUM_CLASSES", ns.get("n_classes")))
            if isinstance(ncls, int) and ncls != cfg.get("num_classes"):
                status, info = "FAIL", f"num_classes = {ncls}, expected {cfg.get('num_classes')}"
            extra += f" | classes {ncls}"
        bad += status not in ("OK", "NEEDS_MODELS")
        print(f"{status:8s} {rel:52s} {info[:200]}{extra}", flush=True)
    print("RESULT:", "OK" if not bad else f"{bad} notebook(s) failed", "| work dir", work)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
