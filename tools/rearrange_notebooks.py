import json
import glob
from pathlib import Path

def rearrange_notebook(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    if not cells:
        return

    # Find the first code cell (likely my setup cell)
    setup_idx = -1
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            setup_idx = i
            break
    
    if setup_idx == -1:
        return

    setup_cell = cells.pop(setup_idx)
    
    # Find the first contiguous block of markdown cells at the new beginning
    # and insert the setup cell after them.
    insert_pos = 0
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            insert_pos = i + 1
        else:
            break
            
    cells.insert(insert_pos, setup_cell)
    nb['cells'] = cells
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Rearranged {nb_path}")

def main():
    target_dir = "classification_uncertainty_analysis"
    nb_files = glob.glob(f"{target_dir}/**/*.ipynb", recursive=True)
    for nb_file in nb_files:
        rearrange_notebook(nb_file)

if __name__ == "__main__":
    main()
