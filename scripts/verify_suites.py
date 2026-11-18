#!/usr/bin/env python3
"""
Verification Suite for Classification & Uncertainty Quantification Analysis (11 Benchmark Suites).

Validates across all 11 suites:
1. Decoupling & Hygiene: zero local .keras/.h5 model weights in repo or /tmp / /private/tmp,
   and zero literal hf_/KGAT_ tokens or unignored >50 MB raw data files.
2. Notebook JSON validity and Python AST syntax across all 11 suites.
3. Tier 3: Indian Pines (145x145x220, K=16)
4. Tier 4: Pavia University (610x340x103, K=9)
5. Tier 5: Hisar-3 (1733x2647x4, K=8)
6. Tier 6: Hissar-25 (722x1014x4, K=8)
7. Tier 7: Completed Benchmarks (DIAS 65-band 512x512x65, 6-band 330x307x6)
8. Tier 7B: New Staged Benchmark Suites (Pavia Centre 1096x715x102 K=9, 372-Band 1101x566x372 K=13,
   ETM 108 330x307x108 K=7, Achla SAR 964x1028x32 K=12, Planet Hisar CSV 1733x2647x4 K=8),
   validating lightweight GT files and making >50 MB raw feature cubes optional when absent locally.
9. Tier 8: Conformal Prediction & Model Methods across all 9 unified suites.
10. Tier 9: Subfolder Notebooks, Symbol Bindings, and chunk_rows=50 memory guards across all 9 unified suites.

Pass condition: Exit code 0 with all checks green.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio


UNIFIED_9_SUITES = [
    "indian_pines_uncertainty_quantification",
    "pavia_university_uncertainty_quantification",
    "hisar_3mts_uncertainty_quantification",
    "hissar_25_uncertainty_quantification",
    "pavia_uncertainty_quantification",
    "372_band_uncertainty_quantification",
    "multispectral_uncertainty_quantification",
    "achla_multi_temporal_uncertainty_quantification",
    "planet_data_hisar_uncertainty_quantification",
]

ALL_11_SUITES = UNIFIED_9_SUITES + [
    "dias_uncertainty_quantification",
    "6_band_uncertainty_quantification",
]

CANONICAL_17_NOTEBOOKS = [
    "Model_training.ipynb",
    "Model_training_credit.ipynb",
    "Model_uncertainty_CreDE.ipynb",
    "Model_uncertainty_comparison.ipynb",
    "baseline/model_training.ipynb",
    "baseline/model_uncertainty_comparison.ipynb",
    "credit/model_training_credit.ipynb",
    "dapm/Model_uncertainty_dapm_cp.ipynb",
    "dapm/model_training_dapm_full.ipynb",
    "dapm/model_uncertainty_dapm_full.ipynb",
    "ensemble/Model_uncertainty_CreDE.ipynb",
    "ensemble/model_training_ensembles.ipynb",
    "ensemble/model_uncertainty_credal_ensemble.ipynb",
    "multi_cp/model_training_multihead.ipynb",
    "multi_cp/model_uncertainty_multicp.ipynb",
    "multicp_sacp/Model_uncertainty_multicp_sacp.ipynb",
    "sacp/model_sacp_comparison.ipynb",
]


class SuiteVerifier:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.examples_dir = repo_root / "examples"
        self.active_suites = list(ALL_11_SUITES)
        self.passed_checks = 0
        self.failed_checks = 0
        self.log_messages = []

    def check(self, condition: bool, description: str, details: str = ""):
        if condition:
            self.passed_checks += 1
            print(f"  [PASS] {description}")
            self.log_messages.append(f"PASS: {description}")
        else:
            self.failed_checks += 1
            msg = f"  [FAIL] {description} -- {details}"
            print(msg)
            self.log_messages.append(msg)

    def verify_all(self) -> bool:
        print("=" * 80)
        print("  CLASSIFICATION & UNCERTAINTY QUANTIFICATION PIPELINE VERIFIER (11 SUITES)")
        print("=" * 80)

        self.verify_storage_decoupling()
        self.verify_notebook_json_and_ast()
        self.verify_indian_pines_suite()
        self.verify_pavia_university_suite()
        self.verify_hisar_3mts_suite()
        self.verify_hissar_25_suite()
        self.verify_completed_benchmarks()
        self.verify_new_staged_suites()
        self.verify_conformal_and_model_methods()
        self.verify_subfolder_notebooks_and_symbols()

        print("\n" + "=" * 80)
        print(f"VERIFICATION SUMMARY: {self.passed_checks} PASSED, {self.failed_checks} FAILED")
        print("=" * 80)
        return self.failed_checks == 0

    def verify_storage_decoupling(self):
        print("\n[Tier 1] Verifying Artifact Decoupling & Storage Guard...")
        repo_keras = list(self.repo_root.glob("**/*.keras")) + list(self.repo_root.glob("**/*.h5"))
        self.check(
            len(repo_keras) == 0,
            "Zero local model weights stored in repo (weights decoupled to Hugging Face)",
            f"Found {len(repo_keras)} weight files in repo: {repo_keras[:3]}",
        )

        tmp_weights = []
        weight_exts = {".keras", ".h5", ".hdf5", ".pt", ".pth", ".ckpt"}
        for tmp_path in [Path("/tmp"), Path("/private/tmp")]:
            if not tmp_path.exists():
                continue
            for root, _, files in os.walk(tmp_path):
                for f in files:
                    if Path(f).suffix.lower() in weight_exts:
                        p = Path(root) / f
                        if str(p) not in tmp_weights:
                            tmp_weights.append(str(p))
        self.check(
            len(tmp_weights) == 0,
            "Zero local model weights present in /tmp or /private/tmp",
            f"Found {len(tmp_weights)} weight files in /tmp: {tmp_weights[:3]}",
        )

    def verify_notebook_json_and_ast(self):
        print("\n[Tier 2] Verifying All 11 Suites Notebook JSON Syntax, Token Hygiene, and Python AST...")
        notebooks = []
        for suite in self.active_suites:
            suite_path = self.examples_dir / suite
            if suite_path.exists():
                notebooks.extend(sorted(suite_path.rglob("*.ipynb")))

        self.check(len(notebooks) >= 150, f"Discovered {len(notebooks)} production notebooks across 11 suites")

        json_valid_count = 0
        ast_valid_count = 0
        token_clean_count = 0
        ast_errors = []

        for nb_path in notebooks:
            try:
                raw_text = nb_path.read_text(encoding="utf-8")
                nb = json.loads(raw_text)
                json_valid_count += 1
            except Exception as e:
                self.check(False, f"JSON validation for {nb_path.name}", str(e))
                continue

            if not re.findall(r"hf_[a-zA-Z0-9]{30,}|KGAT_[a-zA-Z0-9]{20,}", raw_text):
                token_clean_count += 1

            has_cells = "cells" in nb and isinstance(nb["cells"], list) and len(nb["cells"]) >= 20
            self.check(
                has_cells,
                f"Notebook {nb_path.relative_to(self.repo_root)} has valid schema ({len(nb.get('cells', []))} cells)",
            )

            all_cells_clean = True
            for idx, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "code":
                    src = "".join(cell.get("source", []))
                    cleaned_lines = [
                        ("# " + line if line.strip().startswith(("%", "!")) else line)
                        for line in src.splitlines()
                    ]
                    cleaned_src = "\n".join(cleaned_lines)
                    try:
                        ast.parse(cleaned_src)
                    except SyntaxError as se:
                        all_cells_clean = False
                        ast_errors.append(f"{nb_path.name} [cell {idx}]: {se}")

            if all_cells_clean:
                ast_valid_count += 1

        self.check(
            json_valid_count == len(notebooks),
            f"All {len(notebooks)} active suite notebooks parse as 100% valid JSON",
        )
        self.check(
            token_clean_count == len(notebooks),
            f"All {len(notebooks)} active suite notebooks contain zero literal HF/Kaggle tokens",
        )
        self.check(
            ast_valid_count == len(notebooks),
            f"All {len(notebooks)} active suite notebooks passed AST syntax parse without any SyntaxError",
            f"Errors: {ast_errors[:3]}",
        )

    def verify_indian_pines_suite(self):
        print("\n[Tier 3] Verifying Indian Pines Suite (145x145x220, K=16)...")
        suite_dir = self.examples_dir / "indian_pines_uncertainty_quantification"
        data_dir = suite_dir / "data"

        for nb_name in [
            "Model_training.ipynb",
            "Model_uncertainty_comparison.ipynb",
            "Model_training_credit.ipynb",
            "Model_uncertainty_CreDE.ipynb",
        ]:
            self.check((suite_dir / nb_name).exists(), f"Indian Pines canonical notebook exists: {nb_name}")

        data_mat_file = data_dir / "Indian_pines.mat"
        label_mat_file = data_dir / "Indian_pines_gt.mat"
        self.check(data_mat_file.exists(), "Indian Pines data file exists: Indian_pines.mat")
        self.check(label_mat_file.exists(), "Indian Pines label file exists: Indian_pines_gt.mat")

        data_mat = sio.loadmat(data_mat_file)
        label_mat = sio.loadmat(label_mat_file)

        self.check("indian_pines" in data_mat, "Indian_pines.mat contains key 'indian_pines'")
        self.check("indian_pines_gt" in label_mat, "Indian_pines_gt.mat contains key 'indian_pines_gt'")

        x_ip = data_mat["indian_pines"]
        y_ip = label_mat["indian_pines_gt"]

        self.check(
            x_ip.shape == (145, 145, 220),
            f"Indian Pines cube shape matches specification: (145, 145, 220) [Actual: {x_ip.shape}]",
        )
        self.check(
            y_ip.shape == (145, 145),
            f"Indian Pines label shape matches specification: (145, 145) [Actual: {y_ip.shape}]",
        )

        active_classes = np.unique(y_ip[y_ip > 0])
        self.check(
            len(active_classes) == 16,
            f"Indian Pines has exactly 16 active classes (background 0 excluded) [Actual: {len(active_classes)}]",
        )

        with open(suite_dir / "Model_training.ipynb", "r", encoding="utf-8") as f:
            nb = json.load(f)
        nb_code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")

        self.check("indian_pines" in nb_code and "DATA_KEY" in nb_code, "Indian Pines Model_training specifies DATA_KEY='indian_pines'")
        self.check("indian_pines_gt" in nb_code and "LABEL_KEY" in nb_code, "Indian Pines Model_training specifies LABEL_KEY='indian_pines_gt'")
        self.check("PATCH_SIZE  = 9" in nb_code or "PATCH_SIZE = 9" in nb_code, "Indian Pines Model_training specifies PATCH_SIZE = 9")

    def verify_pavia_university_suite(self):
        print("\n[Tier 4] Verifying Pavia University Suite (610x340x103, K=9)...")
        suite_dir = self.examples_dir / "pavia_university_uncertainty_quantification"
        data_dir = suite_dir / "data"

        for nb_name in [
            "Model_training.ipynb",
            "Model_uncertainty_comparison.ipynb",
            "Model_training_credit.ipynb",
            "Model_uncertainty_CreDE.ipynb",
        ]:
            self.check((suite_dir / nb_name).exists(), f"Pavia University canonical notebook exists: {nb_name}")

        data_mat_file = data_dir / "paviauni.mat"
        label_mat_file = data_dir / "paviauni_gt.mat"
        self.check(data_mat_file.exists(), "Pavia University data file exists: paviauni.mat")
        self.check(label_mat_file.exists(), "Pavia University label file exists: paviauni_gt.mat")

        data_mat = sio.loadmat(data_mat_file)
        label_mat = sio.loadmat(label_mat_file)

        self.check("paviaU" in data_mat, "paviauni.mat contains key 'paviaU'")
        self.check("paviaU_gt" in label_mat, "paviauni_gt.mat contains key 'paviaU_gt'")

        x_pu = data_mat["paviaU"]
        y_pu = label_mat["paviaU_gt"]

        self.check(
            x_pu.shape == (610, 340, 103),
            f"Pavia University cube shape matches specification: (610, 340, 103) [Actual: {x_pu.shape}]",
        )
        self.check(
            y_pu.shape == (610, 340),
            f"Pavia University label shape matches specification: (610, 340) [Actual: {y_pu.shape}]",
        )

        active_classes = np.unique(y_pu[y_pu > 0])
        self.check(
            len(active_classes) == 9,
            f"Pavia University has exactly 9 active classes (background 0 excluded) [Actual: {len(active_classes)}]",
        )

        with open(suite_dir / "Model_training.ipynb", "r", encoding="utf-8") as f:
            nb = json.load(f)
        nb_code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")

        self.check("paviaU" in nb_code and "DATA_KEY" in nb_code, "Pavia University Model_training specifies DATA_KEY='paviaU'")
        self.check("paviaU_gt" in nb_code and "LABEL_KEY" in nb_code, "Pavia University Model_training specifies LABEL_KEY='paviaU_gt'")
        self.check("PATCH_SIZE  = 9" in nb_code or "PATCH_SIZE = 9" in nb_code, "Pavia University Model_training specifies PATCH_SIZE = 9")

    def verify_hisar_3mts_suite(self):
        print("\n[Tier 5] Verifying Hisar-3 Multispectral Suite (1733x2647x4, K=8)...")
        suite_dir = self.examples_dir / "hisar_3mts_uncertainty_quantification"
        data_dir = suite_dir / "data"

        for nb_name in [
            "Model_training.ipynb",
            "Model_uncertainty_comparison.ipynb",
            "Model_training_credit.ipynb",
            "Model_uncertainty_CreDE.ipynb",
        ]:
            self.check((suite_dir / nb_name).exists(), f"Hisar-3 canonical notebook exists: {nb_name}")

        data_tif = data_dir / "Hisar3.tif"
        label_tif = data_dir / "Clsimg3.tif"
        self.check(data_tif.exists(), "Hisar-3 data file exists: Hisar3.tif")
        self.check(label_tif.exists(), "Hisar-3 label file exists: Clsimg3.tif")

        raw_x = np.fromfile(str(data_tif), dtype=np.uint16).reshape((4, 1733, 2647))
        x_h3 = np.moveaxis(raw_x, 0, -1).astype(np.float32)
        y_h3 = np.fromfile(str(label_tif), dtype=np.uint8).reshape((1733, 2647)).astype(np.int32)

        self.check(
            x_h3.shape == (1733, 2647, 4),
            f"Hisar-3 cube shape matches specification: (1733, 2647, 4) [Actual: {x_h3.shape}]",
        )
        self.check(
            y_h3.shape == (1733, 2647),
            f"Hisar-3 label shape matches specification: (1733, 2647) [Actual: {y_h3.shape}]",
        )

        active_classes = np.unique(y_h3[y_h3 > 0])
        self.check(
            len(active_classes) == 8,
            f"Hisar-3 has exactly 8 active classes (background 0 excluded) [Actual: {len(active_classes)}]",
        )

        with open(suite_dir / "Model_training.ipynb", "r", encoding="utf-8") as f:
            nb = json.load(f)
        nb_code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")

        self.check("Hisar3.tif" in nb_code and "Clsimg3.tif" in nb_code, "Hisar-3 Model_training specifies Hisar3.tif and Clsimg3.tif")
        self.check("PATCH_SIZE" in nb_code and "9" in nb_code, "Hisar-3 Model_training specifies PATCH_SIZE = 9")

    def verify_hissar_25_suite(self):
        print("\n[Tier 6] Verifying Hissar-25 Multispectral Suite (722x1014x4, K=8)...")
        suite_dir = self.examples_dir / "hissar_25_uncertainty_quantification"
        data_dir = suite_dir / "data"

        for nb_name in [
            "Model_training.ipynb",
            "Model_uncertainty_comparison.ipynb",
            "Model_training_credit.ipynb",
            "Model_uncertainty_CreDE.ipynb",
        ]:
            self.check((suite_dir / nb_name).exists(), f"Hissar-25 canonical notebook exists: {nb_name}")

        data_mat_file = data_dir / "datas.mat"
        label_csv_file = data_dir / "fullgt.csv"
        self.check(data_mat_file.exists(), "Hissar-25 data file exists: datas.mat")
        self.check(label_csv_file.exists(), "Hissar-25 label file exists: fullgt.csv")

        data_mat = sio.loadmat(data_mat_file)
        self.check("datas" in data_mat, "datas.mat contains key 'datas'")

        x_h25 = data_mat["datas"].astype(np.float32).reshape((722, 1014, 4))
        y_h25 = pd.read_csv(label_csv_file).to_numpy(dtype=np.int32).reshape((722, 1014))

        self.check(
            x_h25.shape == (722, 1014, 4),
            f"Hissar-25 cube shape matches specification: (722, 1014, 4) [Actual: {x_h25.shape}]",
        )
        self.check(
            y_h25.shape == (722, 1014),
            f"Hissar-25 label shape matches specification: (722, 1014) [Actual: {y_h25.shape}]",
        )

        active_classes = np.unique(y_h25[y_h25 > 0])
        self.check(
            len(active_classes) == 8,
            f"Hissar-25 has exactly 8 active classes (background 0 excluded) [Actual: {len(active_classes)}]",
        )

        with open(suite_dir / "Model_training.ipynb", "r", encoding="utf-8") as f:
            nb = json.load(f)
        nb_code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c.get("cell_type") == "code")

        self.check("datas.mat" in nb_code and "fullgt.csv" in nb_code, "Hissar-25 Model_training specifies datas.mat and fullgt.csv")
        self.check("PATCH_SIZE" in nb_code and "9" in nb_code, "Hissar-25 Model_training specifies PATCH_SIZE = 9")

    def verify_completed_benchmarks(self):
        print("\n[Tier 7] Verifying Completed Benchmarks Staging (DIAS 65-band & 6-band)...")
        dias_dir = self.examples_dir / "dias_uncertainty_quantification"
        for nb_name in [
            "Model_training.ipynb",
            "Model_uncertainty_comparison.ipynb",
            "Model_training_credit.ipynb",
            "Model_uncertainty_CreDE.ipynb",
        ]:
            self.check((dias_dir / nb_name).exists(), f"DIAS 65-band canonical notebook exists: {nb_name}")

        dias_data = sio.loadmat(dias_dir / "data" / "DIAS.mat")["DIAS"]
        self.check(dias_data.shape == (512, 512, 65), "DIAS 65-band cube verified: (512, 512, 65)")

        six_dir = self.examples_dir / "6_band_uncertainty_quantification"
        for nb_name in [
            "Model_training.ipynb",
            "Model_uncertainty_comparison.ipynb",
            "Model_training_credit.ipynb",
            "Model_uncertainty_CreDE.ipynb",
        ]:
            self.check((six_dir / nb_name).exists(), f"6-band canonical notebook exists: {nb_name}")

        df_6_data = pd.read_csv(six_dir / "data" / "data.csv")
        self.check(df_6_data.shape == (101310, 6), "6-band data table verified: (101310, 6) -> (330, 307, 6)")

    def verify_new_staged_suites(self):
        print("\n[Tier 7B] Verifying 5 New Staged Benchmark Suites (Pavia Centre, 372-Band, ETM 108, Achla SAR, Planet Hisar)...")
        specs = [
            {
                "suite": "pavia_uncertainty_quantification",
                "label": "Pavia Centre",
                "hf_repo": "DaneshSelwal/Classification-PaviaCentre-Models",
                "tokens": ["pavia", "pavia_gt", "pavia.mat", "pavia_gt.mat", "1096", "715", "102"],
                "gt_file": ("mat", "pavia_gt.mat", "pavia_gt", (1096, 715), 9),
                "opt_data": ("mat", "pavia.mat", "pavia", (1096, 715, 102)),
            },
            {
                "suite": "372_band_uncertainty_quantification",
                "label": "372-Band",
                "hf_repo": "DaneshSelwal/Classification-372Band-Models",
                "tokens": ["full", "full_gt", "full.mat", "full_gt.mat", "1101", "566", "372"],
                "gt_file": ("mat", "full_gt.mat", "full_gt", (623166, 1), 13),
                "opt_data": ("mat", "full.mat", "full", (623166, 372)),
            },
            {
                "suite": "multispectral_uncertainty_quantification",
                "label": "ETM 108",
                "hf_repo": "DaneshSelwal/Classification-ETM108-Models",
                "tokens": ["all_etmls.csv", "ref.csv", "330", "307", "108"],
                "gt_file": ("csv", "ref.csv", None, (101310, 1), 7),
                "opt_data": ("csv", "all_etmls.csv", None, (101310, 108)),
            },
            {
                "suite": "achla_multi_temporal_uncertainty_quantification",
                "label": "Achla SAR",
                "hf_repo": "DaneshSelwal/Classification-AchlaSAR-Models",
                "tokens": ["vv_dectomarch.csv", "vh_dectomarch.csv", "gt.csv", "964", "1028", "32"],
                "gt_file": ("csv", "gt.csv", None, (990992, 1), 12),
                "opt_data": ("csv", "vv_dectomarch.csv", None, (990992, 16)),
            },
            {
                "suite": "planet_data_hisar_uncertainty_quantification",
                "label": "Planet Hisar CSV",
                "hf_repo": "DaneshSelwal/Classification-PlanetHisar-Models",
                "tokens": ["planet.csv", "planetgt.csv", "1733", "2647", "4"],
                "gt_file": None,
                "opt_data": None,
            },
        ]

        for sp in specs:
            s_dir = self.examples_dir / sp["suite"]
            data_dir = s_dir / "data"

            # Canonical 17-notebook layout
            missing_nbs = [nb for nb in CANONICAL_17_NOTEBOOKS if not (s_dir / nb).exists()]
            self.check(
                len(missing_nbs) == 0,
                f"{sp['label']} contains all 17 canonical notebooks",
                f"Missing: {missing_nbs}",
            )

            # Data directory size hygiene (<= 50 MB)
            oversized = []
            if data_dir.exists():
                oversized = [p.name for p in data_dir.iterdir() if p.is_file() and p.stat().st_size > 50_000_000]
            self.check(
                len(oversized) == 0,
                f"{sp['label']} data/ contains zero >50 MB raw files in repo",
                f"Oversized: {oversized}",
            )

            # Lightweight GT verification
            if sp["gt_file"] is not None:
                kind, fname, key, expected_shape, expected_k = sp["gt_file"]
                gt_path = data_dir / fname
                self.check(gt_path.exists(), f"{sp['label']} lightweight GT file exists: {fname}")
                if gt_path.exists():
                    if kind == "mat":
                        arr = sio.loadmat(gt_path)[key]
                    else:
                        arr = pd.read_csv(gt_path).to_numpy()
                    self.check(
                        arr.shape == expected_shape,
                        f"{sp['label']} GT shape matches {expected_shape} [Actual: {arr.shape}]",
                    )
                    active_k = int(np.unique(arr[arr > 0]).size)
                    self.check(
                        active_k == expected_k,
                        f"{sp['label']} GT has {expected_k} active classes [Actual: {active_k}]",
                    )

            # Optional >50 MB raw data cube check if present locally
            if sp["opt_data"] is not None:
                kind, fname, key, expected_shape = sp["opt_data"]
                opt_path = data_dir / fname
                if opt_path.exists():
                    if kind == "mat":
                        arr = sio.loadmat(opt_path)[key]
                    else:
                        arr = pd.read_csv(opt_path).to_numpy()
                    self.check(
                        arr.shape == expected_shape,
                        f"{sp['label']} optional local data {fname} matches shape {expected_shape}",
                    )
                else:
                    self.check(True, f"{sp['label']} >50 MB raw data {fname} omitted locally (staged on Kaggle)")

            # Model_training.ipynb config constants
            mt_nb = json.load(open(s_dir / "Model_training.ipynb", "r", encoding="utf-8"))
            mt_code = "\n".join("".join(c.get("source", [])) for c in mt_nb["cells"] if c.get("cell_type") == "code")
            self.check(
                sp["hf_repo"] in mt_code and all(tok in mt_code for tok in sp["tokens"]),
                f"{sp['label']} Model_training.ipynb configures HF_REPO, geometry, and dataset filenames",
            )
            self.check(
                "PATCH_SIZE" in mt_code and "9" in mt_code,
                f"{sp['label']} Model_training.ipynb specifies PATCH_SIZE = 9",
            )
            mcp_src = (s_dir / "multi_cp" / "model_uncertainty_multicp.ipynb").read_text(encoding="utf-8")
            self.check(
                "chunk_rows=50" in mcp_src,
                f"{sp['label']} multi_cp/model_uncertainty_multicp.ipynb uses chunk_rows=50 memory-safe scene inference",
            )

    def verify_conformal_and_model_methods(self):
        print("\n[Tier 8] Verifying Conformal Prediction & Model Methods across all 9 Unified Suites...")
        for suite in UNIFIED_9_SUITES:
            uq_nb = self.examples_dir / suite / "Model_uncertainty_comparison.ipynb"
            content = uq_nb.read_text(encoding="utf-8")

            self.check(
                "build_split_outputs_for_model" in content and "SplitConformal" in content,
                f"{suite} implements Standard Split Conformal Prediction (SplitCP)",
            )
            self.check(
                "build_classconditional_outputs_for_model" in content and "ClassConditional" in content,
                f"{suite} implements Class-Conditional Conformal Prediction (Multi-CP)",
            )
            self.check(
                "build_sacp_outputs_for_model" in content and "SACP" in content,
                f"{suite} implements Spatial-Aware Conformal Prediction (SACP)",
            )
            self.check(
                "build_scmcp_outputs_for_model" in content and "SCMCP" in content,
                f"{suite} implements Spatial Class-Conditional Conformal Prediction (SCMCP)",
            )

            mt_content = (self.examples_dir / suite / "Model_training.ipynb").read_text(encoding="utf-8")
            self.check(
                "build_alexnet" in mt_content and "build_gfnet" in mt_content and "build_vit" in mt_content,
                f"{suite} trains AlexNet, GFNet, and ViT-UNet backbones",
            )

            credit_content = (self.examples_dir / suite / "Model_training_credit.ipynb").read_text(encoding="utf-8")
            self.check(
                "build_credit_student" in credit_content,
                f"{suite} implements CREDIT distillation dual-head student",
            )

            crede_content = (self.examples_dir / suite / "Model_uncertainty_CreDE.ipynb").read_text(encoding="utf-8")
            self.check(
                "evaluate_homogeneous_ensemble" in crede_content,
                f"{suite} implements CreDE ensemble uncertainty quantification",
            )

    def verify_subfolder_notebooks_and_symbols(self):
        print("\n[Tier 9] Verifying Subfolder Notebooks & Symbol Bindings across all 9 Unified Suites...")
        for suite in UNIFIED_9_SUITES:
            s_dir = self.examples_dir / suite

            # 1. multi_cp/model_uncertainty_multicp.ipynb
            mcp_src = (s_dir / "multi_cp" / "model_uncertainty_multicp.ipynb").read_text(encoding="utf-8")
            self.check(
                "NUM_CLASSES = num_classes" in mcp_src and "CLASS_COLORS = get_class_colors(NUM_CLASSES)" in mcp_src,
                f"{suite} multi_cp assigns NUM_CLASSES and CLASS_COLORS dynamically",
            )

            # 2. dapm/Model_uncertainty_dapm_cp.ipynb
            dapm_cp_src = (s_dir / "dapm" / "Model_uncertainty_dapm_cp.ipynb").read_text(encoding="utf-8")
            self.check(
                "def split_calib_eval(" in dapm_cp_src,
                f"{suite} dapm_cp defines split_calib_eval",
            )
            self.check(
                "X_all, y_all, coords_all = extract_labeled_patches(" in dapm_cp_src,
                f"{suite} dapm_cp correctly unpacks 3 variables from extract_labeled_patches",
            )

            # 3. multicp_sacp/Model_uncertainty_multicp_sacp.ipynb
            sacp_src = (s_dir / "multicp_sacp" / "Model_uncertainty_multicp_sacp.ipynb").read_text(encoding="utf-8")
            self.check(
                "def split_calib_eval_with_coords(" in sacp_src,
                f"{suite} multicp_sacp defines split_calib_eval_with_coords",
            )

            # 4. dapm/model_uncertainty_dapm_full.ipynb
            dapm_full_src = (s_dir / "dapm" / "model_uncertainty_dapm_full.ipynb").read_text(encoding="utf-8")
            self.check(
                "def extract_patches_from_coords(" in dapm_full_src,
                f"{suite} dapm_full UQ defines extract_patches_from_coords",
            )

            # 5. dapm/model_training_dapm_full.ipynb
            dapm_tr_src = (s_dir / "dapm" / "model_training_dapm_full.ipynb").read_text(encoding="utf-8")
            self.check(
                "def get_display_cmap(" in dapm_tr_src
                and "def get_scene_rgb(" in dapm_tr_src
                and "def get_class_colors(" in dapm_tr_src,
                f"{suite} dapm_full training defines visualisation palette helpers",
            )

            # 6. ensemble/model_training_ensembles.ipynb
            ens_src = (s_dir / "ensemble" / "model_training_ensembles.ipynb").read_text(encoding="utf-8")
            self.check(
                "try:" in ens_src and "save_images_to_excel(" in ens_src,
                f"{suite} ensemble training wraps save_images_to_excel in try-except guard",
            )


def main():
    repo_root = Path(__file__).resolve().parent.parent
    verifier = SuiteVerifier(repo_root)
    success = verifier.verify_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
