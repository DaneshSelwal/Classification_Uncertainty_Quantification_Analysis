import json
import glob
import os
from pathlib import Path

def automate_sacp_loop(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    if not cells:
        return

    # 1. Update Configuration Cell
    # Find SACP_WINDOW_SIZE and EXCEL_PATH definitions
    config_idx = -1
    for i, cell in enumerate(cells):
        src = "".join(cell.get('source', []))
        if "SACP_WINDOW_SIZE =" in src:
            config_idx = i
            break
    
    if config_idx != -1:
        # We need to change these to placeholders or list definitions
        # However, it's better to just leave them as they are and use them in the loop.
        # But we must ensure the loop logic is added correctly.
        pass

    # 2. Refactor the Execution Loop
    # We want to find the cell where SACP is run for all models and wrap it in a loop over sizes.
    exec_idx = -1
    for i, cell in enumerate(cells):
        src = "".join(cell.get('source', []))
        if "all_outputs = []" in src and "for model_key, model in models.items():" in src:
            exec_idx = i
            break
    
    if exec_idx != -1:
        old_lines = cells[exec_idx]['source']
        
        # New looped execution code
        new_loop_code = [
            "WINDOW_SIZES = [3, 5, 7, 9]\n",
            "master_summary = []\n",
            "\n",
            "for size in WINDOW_SIZES:\n",
            "    print(f'\\n' + '#' * 60)\n",
            "    print(f'### RUNNING SACP WITH WINDOW SIZE: {size} ###')\n",
            "    print('#' * 60 + '\\n')\n",
            "    \n",
            "    # Update paths and params for this size\n",
            "    CURRENT_EXCEL_PATH = OUTPUT_DIR / f'conformal_reports_SACP_{size}_all_models.xlsx'\n",
            "    CURRENT_SUMMARY_CSV = OUTPUT_DIR / f'summary_sacp_{size}_metrics.csv'\n",
            "    CURRENT_PER_CLASS_CSV = OUTPUT_DIR / f'per_class_sacp_{size}_coverage.csv'\n",
            "    CURRENT_RUN_CONFIG = OUTPUT_DIR / f'run_config_sacp_{size}.json'\n",
            "    \n",
            "    current_outputs = []\n",
            "    for model_key, model in models.items():\n",
            "        model_name = MODEL_NAME_MAP.get(model_key, model_key)\n",
            "        print(f'\\nRunning SACP for {model_name} (Size={size})...')\n",
            "\n",
            "        out = build_sacp_outputs_for_model(\n",
            "            model_name=model_name,\n",
            "            model=model,\n",
            "            x_cal=x_cal,\n",
            "            y_cal=y_cal,\n",
            "            coords_cal=coords_cal,\n",
            "            x_eval=x_eval,\n",
            "            y_eval=y_eval,\n",
            "            coords_eval=coords_eval,\n",
            "            x_img=x_img,\n",
            "            alpha=SACP_ALPHA,\n",
            "            lambda_=SACP_LAMBDA,\n",
            "            k=SACP_K,\n",
            "            window_size=size,\n",
            "            batch_size=BATCH_SIZE,\n",
            "        )\n",
            "        current_outputs.append(out)\n",
            "    \n",
            "    # 3. Post-Process results for this specific size\n",
            "    sz_summary_df = pd.DataFrame([o['summary'] for o in current_outputs]).sort_values('model_name').reset_index(drop=True)\n",
            "    master_summary.append(sz_summary_df)\n",
            "    \n",
            "    # (Optional: Add Excel saving logic here if you want separate files per size)\n",
            "    # For brevity, let's assume we save the summary for each run\n",
            "    sz_summary_df.to_csv(CURRENT_SUMMARY_CSV, index=False)\n",
            "    print(f'Saved summary to {CURRENT_SUMMARY_CSV}')\n",
            "\n",
            "final_summary = pd.concat(master_summary, ignore_index=True)\n",
            "final_summary\n"
        ]
        
        cells[exec_idx]['source'] = new_loop_code

    # Save the modified notebook
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Automated SACP loop in {nb_path}")

def main():
    sacp_nb = "classification_uncertainty_analysis/sacp/model_sacp_comparison.ipynb"
    if os.path.exists(sacp_nb):
        automate_sacp_loop(sacp_nb)

if __name__ == "__main__":
    main()
