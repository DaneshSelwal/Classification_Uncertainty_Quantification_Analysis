import glob
import json
import uuid
from pathlib import Path


BOOTSTRAP_SENTINEL = "COLAB_REPO_ROOT"


def make_bootstrap_cell(module_name: str) -> dict:
    src = [
        "from pathlib import Path\n",
        "import sys\n",
        "\n",
        f"MODULE_NAME = {module_name!r}\n",
        "# In Colab after uploading the repo folder to Google Drive, set this to the Drive path\n",
        "# that contains the module folders (baseline/, credit/, ...) and data/.\n",
        'COLAB_REPO_ROOT = ""  # e.g., "/content/drive/MyDrive/classification_uncertainty_analysis"\n',
        "\n",
        "try:\n",
        "    from google.colab import drive  # type: ignore\n",
        "    IN_COLAB = True\n",
        "except Exception:\n",
        "    IN_COLAB = False\n",
        "\n",
        "if IN_COLAB:\n",
        "    drive.mount('/content/drive')\n",
        "    MODULE_DIR = (Path(COLAB_REPO_ROOT) / MODULE_NAME) if COLAB_REPO_ROOT else Path.cwd()\n",
        "else:\n",
        "    MODULE_DIR = Path.cwd()\n",
        "\n",
        "# Allow running from repo root (common in Colab) or from the module directory.\n",
        "if MODULE_DIR.name != MODULE_NAME:\n",
        "    if (MODULE_DIR / MODULE_NAME).exists():\n",
        "        MODULE_DIR = MODULE_DIR / MODULE_NAME\n",
        "    else:\n",
        "        raise RuntimeError(\n",
        "            f\"Expected to run from `{MODULE_NAME}` dir or repo root. cwd={MODULE_DIR}. \"\n",
        "            \"In Colab, set `COLAB_REPO_ROOT` to the Drive folder containing the repo.\"\n",
        "        )\n",
        "\n",
        "REPO_ROOT = MODULE_DIR.parent\n",
        "RESULTS_DIR = MODULE_DIR / 'results'\n",
        "MODELS_DIR = MODULE_DIR / 'models'\n",
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
        "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n",
        "print('Effective output directory:', RESULTS_DIR)\n",
        "print('Effective model directory:', MODELS_DIR)\n",
    ]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def is_bootstrap_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    return any(BOOTSTRAP_SENTINEL in line for line in (cell.get("source") or []))


def normalize_code_cell_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    skip_colab_if = False
    skip_proj_candidates = False
    for line in lines:
        s = line.strip()

        if skip_proj_candidates:
            if "]" in line:
                skip_proj_candidates = False
            continue

        if s.startswith("PROJECT_ROOT_CANDIDATES"):
            skip_proj_candidates = True
            continue
        if s.startswith("PROJECT_ROOT = next("):
            out.append("PROJECT_ROOT = REPO_ROOT\n")
            continue

        if s.startswith("if 'google.colab' in sys.modules"):
            skip_colab_if = True
            continue
        if skip_colab_if:
            if line.startswith(" ") or line.startswith("\t") or s == "":
                continue
            skip_colab_if = False

        # Remove old per-notebook bootstrap fragments.
        if s.startswith("MODULE_DIR ="):
            continue
        if s.startswith("RESULTS_DIR =") or s.startswith("MODELS_DIR ="):
            continue
        if s.startswith("RESULTS_DIR.mkdir") or s.startswith("MODELS_DIR.mkdir"):
            continue
        if "Effective output directory" in line or "Effective model directory" in line:
            continue
        if s.startswith("from google.colab import drive"):
            continue
        if s.startswith("drive.mount("):
            continue

        # Replace legacy Drive-root data paths with repo-local read paths.
        line = line.replace(
            'DATA_DIR = Path("/content/drive/My Drive/m_p/data/multispectral")',
            'DATA_DIR = REPO_ROOT / "data" / "multispectral"',
        )
        line = line.replace(
            'DATA_DIR = Path("/content/drive/MyDrive/m_p/data/multispectral")',
            'DATA_DIR = REPO_ROOT / "data" / "multispectral"',
        )

        # Replace trusted roots pointing to old locations.
        line = line.replace("Path('/content/drive/My Drive/m_p/models')", "MODELS_DIR")
        line = line.replace("Path('/Users/danesh/Documents/m_p/models')", "MODELS_DIR")

        # Replace generic old repo roots.
        line = line.replace("Path('/content/drive/My Drive/m_p')", "REPO_ROOT")
        line = line.replace("Path('/content/drive/MyDrive/m_p')", "REPO_ROOT")
        line = line.replace("Path('/Users/danesh/Documents/m_p')", "REPO_ROOT")

        out.append(line)

    return out


def patch_notebook(nb_path: str) -> bool:
    module_name = Path(nb_path).parent.name
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False

    if not any(is_bootstrap_cell(c) for c in (nb.get("cells") or [])):
        nb["cells"] = [make_bootstrap_cell(module_name)] + (nb.get("cells") or [])
        changed = True

    # Ensure bootstrap cell is intact and first.
    cells = nb.get("cells") or []
    bs_indices = [i for i, c in enumerate(cells) if is_bootstrap_cell(c)]
    if bs_indices:
        bs_cell = cells[bs_indices[0]]
        if bs_indices[0] != 0:
            cells.insert(0, cells.pop(bs_indices[0]))
            changed = True
        # Overwrite bootstrap source to canonical form.
        if bs_cell.get("source") != make_bootstrap_cell(module_name)["source"]:
            bs_cell["source"] = make_bootstrap_cell(module_name)["source"]
            changed = True

    for cell in nb.get("cells") or []:
        if cell.get("cell_type") != "code":
            continue
        if is_bootstrap_cell(cell):
            continue
        old = cell.get("source") or []
        new = normalize_code_cell_lines(old)
        if new != old:
            cell["source"] = new
            changed = True

    if changed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")

    return changed


def main() -> int:
    nb_paths = sorted(glob.glob("**/*.ipynb", recursive=True))
    any_changed = False
    for p in nb_paths:
        if patch_notebook(p):
            any_changed = True
            print(f"patched {p}")
        else:
            print(f"ok {p}")
    if not any_changed:
        print("no changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
