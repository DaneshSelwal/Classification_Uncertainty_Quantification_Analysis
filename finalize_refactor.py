import os
import glob
import json
import shutil

repo_root = os.getcwd()

# 1. Move the 6-band multispectral data into the new folder
multi_src = os.path.join(repo_root, "data", "multispectral")
multi_dst = os.path.join(repo_root, "6_band_uncertainty_quantification", "data")
if os.path.exists(multi_src):
    os.makedirs(multi_dst, exist_ok=True)
    for f in os.listdir(multi_src):
        src_path = os.path.join(multi_src, f)
        dst_path = os.path.join(multi_dst, f)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {f} to {multi_dst}")

# 2. Patch all Notebooks
def patch_notebooks(directory, folder_name):
    print(f"Patching notebooks in {directory}...")
    for nb_path in glob.glob(os.path.join(directory, "**/*.ipynb"), recursive=True):
        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            changed = False
            for cell in nb.get('cells', []):
                if cell['cell_type'] == 'code':
                    src = cell['source']
                    for i, line in enumerate(src):
                        # Update REPO_ROOT
                        if 'REPO_ROOT = Path(' in line:
                            new_line = f'REPO_ROOT = Path("/content/drive/MyDrive/{folder_name}")\n'
                            if src[i] != new_line:
                                src[i] = new_line
                                changed = True
                        
                        # Handle 6-band specific data files
                        if folder_name == "6_band_uncertainty_quantification":
                            if 'DATA_FILE = DATA_DIR /' in line and '"data.csv"' not in line:
                                src[i] = 'DATA_FILE = DATA_DIR / "data.csv"\n'
                                changed = True
                            if 'LABEL_FILE = DATA_DIR /' in line and '"ref.csv"' not in line:
                                src[i] = 'LABEL_FILE = DATA_DIR / "ref.csv"\n'
                                changed = True
                    cell['source'] = src
            
            if changed:
                with open(nb_path, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=1, ensure_ascii=False)
                    f.write("\n")
                print(f"  Patched {os.path.relpath(nb_path, repo_root)}")
        except Exception as e:
            print(f"  Error patching {nb_path}: {e}")

# Run patching for 6-band framework
patch_notebooks(os.path.join(repo_root, "6_band_uncertainty_quantification"), "6_band_uncertainty_quantification")

# Run patching for Examples
examples_path = os.path.join(repo_root, "Examples")
if os.path.exists(examples_path):
    for suite in os.listdir(examples_path):
        suite_path = os.path.join(examples_path, suite)
        if os.path.isdir(suite_path):
            patch_notebooks(suite_path, suite)

# 3. Update .gitignore
gitignore_path = os.path.join(repo_root, ".gitignore")
with open(gitignore_path, 'w', encoding='utf-8') as f:
    f.write("# Ignore large model artifact directories\n")
    f.write("**/models/\n")
    f.write("**/models/**\n\n")
    f.write("# Ignore local tools and original data\n")
    f.write("tools/\n")
    f.write("data/\n")
    f.write("legacy_and_temp_files/\n\n")
    f.write("# Ignore localized data in suites (prevent pushing large files)\n")
    f.write("6_band_uncertainty_quantification/data/\n")
    f.write("Examples/*/data/\n\n")
    f.write("# Notebook checkpoints\n")
    f.write(".ipynb_checkpoints/\n")
    f.write("**/.ipynb_checkpoints/\n")

print("Cleanup and patching complete.")
