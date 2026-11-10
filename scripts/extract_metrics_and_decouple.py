import json
import os
import re
import shutil
from pathlib import Path
from huggingface_hub import HfApi

REPO_ROOT = Path("Classification_Uncertainty_Quantification_Analysis")
HF_TOKEN = (Path.home() / ".cache/huggingface/token").read_text().strip()
api = HfApi(token=HF_TOKEN)

SUITE_MAPPING = {
    "ip-uncertainty-k01": {
        "suite_dir": REPO_ROOT / "examples/indian_pines_uncertainty_quantification",
        "dataset_name": "Indian Pines",
        "hf_repo": "DaneshSelwal/Classification-IndianPines-Models",
        "bands": 220,
        "classes": 16,
        "geometry": "(145, 145, 220)"
    },
    "pavia-uncertainty-k02": {
        "suite_dir": REPO_ROOT / "examples/pavia_university_uncertainty_quantification",
        "dataset_name": "Pavia University",
        "hf_repo": "DaneshSelwal/Classification-PaviaUniversity-Models",
        "bands": 103,
        "classes": 9,
        "geometry": "(610, 340, 103)"
    },
    "hisar3-uncertainty-k03": {
        "suite_dir": REPO_ROOT / "examples/hisar_3mts_uncertainty_quantification",
        "dataset_name": "Hisar-3",
        "hf_repo": "DaneshSelwal/Classification-Hisar3-Models",
        "bands": 4,
        "classes": 8,
        "geometry": "(1733, 2647, 4)"
    },
    "hissar25-uncertainty-k04": {
        "suite_dir": REPO_ROOT / "examples/hissar_25_uncertainty_quantification",
        "dataset_name": "Hissar-25",
        "hf_repo": "DaneshSelwal/Classification-Hissar25-Models",
        "bands": 4,
        "classes": 8,
        "geometry": "(722, 1014, 4)"
    }
}

def process_kernel_output(kernel_id: str, download_dir: Path):
    info = SUITE_MAPPING[kernel_id]
    suite_dir = info["suite_dir"]
    logs_dir = suite_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing output for {kernel_id} -> {suite_dir.name}")
    
    # 1. Search for output files
    all_files = list(download_dir.rglob("*"))
    print(f"Downloaded files count: {len(all_files)}")
    
    # 2. Check for .keras models to decouple to Hugging Face
    keras_files = list(download_dir.rglob("*.keras"))
    if keras_files:
        print(f"Found {len(keras_files)} local .keras models; decoupling to {info['hf_repo']}...")
        for kf in keras_files:
            print(f"  Uploading {kf.name} to Hugging Face...")
            api.upload_file(
                path_or_fileobj=str(kf),
                path_in_repo=f"models/{kf.name}",
                repo_id=info["hf_repo"],
                repo_type="model"
            )
            print(f"  Unlinking local {kf.name} to preserve local disk constraint...")
            kf.unlink()
        print("Model weights 100% decoupled to Hugging Face Hub (0 bytes on local Mac disk).")
        
    # 3. Copy results (CSVs, JSONs, PNGs, logs) into examples/<suite>/
    results_dir = suite_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for src in download_dir.rglob("*"):
        if src.is_file() and not src.suffix in [".keras", ".h5"]:
            # If it's a log or summary
            if src.suffix in [".csv", ".json", ".xlsx", ".txt", ".log"]:
                dst = logs_dir / src.name
                shutil.copy2(src, dst)
                print(f"  Copied log file: {dst.name} -> {logs_dir}")
            if src.suffix in [".png", ".jpg", ".csv", ".xlsx"]:
                dst2 = results_dir / src.name
                shutil.copy2(src, dst2)
                
    # 4. Parse execution log / metrics if available
    metrics = []
    log_files = list(download_dir.rglob("*.log")) + list(download_dir.rglob("*.txt"))
    for lf in log_files:
        text = lf.read_text(errors="ignore")
        for line in text.splitlines():
            if "[METRIC_REPORT]" in line:
                metrics.append(line.strip())
                
    if metrics:
        print(f"Extracted {len(metrics)} metric reports from execution logs.")
        with open(logs_dir / "extracted_metrics.txt", "w") as f:
            f.write("\n".join(metrics) + "\n")
            
    return info["dataset_name"], metrics

if __name__ == "__main__":
    print("Metrics extraction & decoupling helper initialized.")
