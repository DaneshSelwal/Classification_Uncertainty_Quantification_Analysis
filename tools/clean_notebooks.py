import json
import os
import glob
from pathlib import Path

def clean_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    module_name = Path(nb_path).parent.name
    cells = nb.get('cells', [])
    if not cells:
        return
    
    # 1. Identify the first code cell for path setup consolidation
    first_code_idx = -1
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            first_code_idx = i
            break
    
    if first_code_idx == -1:
        return

    # Professional Bootstrap Cell
    setup_code = [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "\n",
        "from pathlib import Path\n",
        "import os\n",
        "import sys\n",
        "\n",
        f"MODULE_NAME = {module_name!r}\n",
        'REPO_ROOT = Path("/content/drive/MyDrive/classification_uncertainty_analysis")\n',
        "MODULE_DIR = REPO_ROOT / MODULE_NAME\n",
        "RESULTS_DIR = MODULE_DIR / 'results'\n",
        "MODELS_DIR = MODULE_DIR / 'models'\n",
        "\n",
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n",
        "MODELS_DIR.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "print(f'Module: {MODULE_NAME}')\n",
        "print(f'Output Directory: {RESULTS_DIR}')\n"
    ]
    
    # Replace first code cell source
    cells[first_code_idx]['source'] = setup_code
    
    # 2. Remove redundant mount or path setup cells that might follow
    # and consolidate imports if they appear immediately after
    new_cells = []
    seen_setup = False
    
    # We'll also collect all imports to move them to a secondary "Imports" cell if they are scattered
    # but for now let's just remove the obvious redundancies.
    
    indices_to_remove = set()
    for i in range(first_code_idx + 1, len(cells)):
        cell = cells[i]
        src_text = "".join(cell.get('source', [])).lower()
        
        # Redundant mount calls
        if "drive.mount" in src_text or "google.colab" in src_text:
            indices_to_remove.add(i)
            continue
            
        # Redundant path variables
        if "module_dir =" in src_text or "repo_root =" in src_text or "results_dir =" in src_text:
            # If the cell only contains these, remove it. 
            # If it has more, we might need to be careful.
            # For simplicity, if it's a small cell with these, remove.
            if len(cell.get('source', [])) < 10:
                indices_to_remove.add(i)
                continue
                
        # Empty cells
        if not src_text.strip():
            indices_to_remove.add(i)
            continue

    # Filter out marked cells
    nb['cells'] = [c for i, c in enumerate(cells) if i not in indices_to_remove]
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Cleaned {nb_path}")

def main():
    target_dir = "classification_uncertainty_analysis"
    nb_files = glob.glob(f"{target_dir}/**/*.ipynb", recursive=True)
    for nb_file in nb_files:
        clean_notebook(nb_file)

if __name__ == "__main__":
    main()
