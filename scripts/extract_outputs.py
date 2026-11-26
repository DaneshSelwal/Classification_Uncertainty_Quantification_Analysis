#!/usr/bin/env python3
"""
Enhanced Multi-Account & 11-Suite Benchmark Runner Output Extractor & Monitor.

Monitors Kaggle GPU & CPU runners across all 9 Kaggle-staged benchmark suites
(alongside 2 locally completed suites: DIAS 65-band and 6-band):
- indian_pines_uncertainty_quantification (k11 / k15)
- pavia_university_uncertainty_quantification (k12 / k16)
- hisar_3mts_uncertainty_quantification (k13 / k17)
- hissar_25_uncertainty_quantification (k14 / k18)
- pavia_uncertainty_quantification (k19 / k24)
- 372_band_uncertainty_quantification (k20 / k25)
- multispectral_uncertainty_quantification (k21 / k26)
- achla_multi_temporal_uncertainty_quantification (k22 / k27)
- planet_data_hisar_uncertainty_quantification (k23 / k28)

Downloads all valid non-empty CSV/XLSX/JSON/PNG outputs and logs into:
- examples/<suite>/results/
- examples/<suite>/logs/<runner>/

Strict R2 Filtering:
- Skips blank CSVs (<= 2 bytes, or stripped length < 5 bytes).
- Skips header-only placeholder .xlsx (max_row <= 1 and size < 6000 bytes).
- Skips fullgt.csv (case-insensitive).
- Skips forbidden prefixes: .cache/, Multi-CP/, .git/, data/, catboost_info/, __pycache__/, suite_notebooks/.
- Enforces zero model weight extraction (never extracts .keras, .h5, etc.).
"""

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import openpyxl
from huggingface_hub import HfApi
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_EXTS = {".csv", ".xlsx", ".json", ".png"}
BAD_PREFIXES = (
    ".cache/",
    "Multi-CP/",
    ".git/",
    "data/",
    "catboost_info/",
    "__pycache__/",
    "models/ensembles/",
    "suite_notebooks/",
)

SUITES: Dict[str, dict] = {
    "indian_pines": {
        "dir_name": "indian_pines_uncertainty_quantification",
        "hf_repo": "DaneshSelwal/Classification-IndianPines-Models",
        "runners": [
            {"slug": "ip-suite-runner-k11", "hw": "GPU", "role": "primary"},
            {"slug": "ip-cpu-runner-k15", "hw": "CPU", "role": "fallback"},
        ],
    },
    "pavia_university": {
        "dir_name": "pavia_university_uncertainty_quantification",
        "hf_repo": "DaneshSelwal/Classification-PaviaUniversity-Models",
        "runners": [
            {"slug": "pavia-suite-runner-k12", "hw": "GPU", "role": "primary"},
            {"slug": "pavia-cpu-runner-k16", "hw": "CPU", "role": "fallback"},
        ],
    },
    "hisar_3mts": {
        "dir_name": "hisar_3mts_uncertainty_quantification",
        "hf_repo": "DaneshSelwal/Classification-Hisar3-Models",
        "runners": [
            {"slug": "hisar3-suite-runner-k13", "hw": "GPU", "role": "primary"},
            {"slug": "hisar3-cpu-runner-k17", "hw": "CPU", "role": "fallback"},
            {"slug": "hisar3-cpu-runner-k17-s1", "hw": "CPU", "role": "fallback"},
            {"slug": "hisar3-cpu-runner-k17-s2", "hw": "CPU", "role": "fallback"},
        ],
    },
    "hissar_25": {
        "dir_name": "hissar_25_uncertainty_quantification",
        "hf_repo": "DaneshSelwal/Classification-Hissar25-Models",
        "runners": [
            {"slug": "hissar25-suite-runner-k14", "hw": "GPU", "role": "primary"},
            {"slug": "hissar25-cpu-runner-k18", "hw": "CPU", "role": "fallback"},
        ],
    },
    "pavia_centre": {
        "dir_name": "pavia_uncertainty_quantification",
        "hf_repo": "DaneshSelwal/Classification-PaviaCentre-Models",
        "runners": [
            {"slug": "pavia-suite-runner-k19", "hw": "GPU", "role": "primary"},
            {"slug": "pavia-cpu-runner-k24", "hw": "CPU", "role": "fallback"},
            {"slug": "pavia-cpu-runner-k24-s1", "hw": "CPU", "role": "fallback"},
            {"slug": "pavia-cpu-runner-k24-s2", "hw": "CPU", "role": "fallback"},
        ],
    },
    "band372": {
        "dir_name": "372_band_uncertainty_quantification",
        "hf_repo": "DaneshSelwal/Classification-372Band-Models",
        "runners": [
            {"slug": "band372-suite-runner-k20", "hw": "GPU", "role": "primary"},
            {"slug": "band372-cpu-runner-k25", "hw": "CPU", "role": "fallback"},
        ],
    },
    "etm108": {
        "dir_name": "multispectral_uncertainty_quantification",
        "hf_repo": "DaneshSelwal/Classification-ETM108-Models",
        "runners": [
            {"slug": "etm108-suite-runner-k21", "hw": "GPU", "role": "primary"},
            {"slug": "etm108-cpu-runner-k26", "hw": "CPU", "role": "fallback"},
            {"slug": "etm108-cpu-runner-k26-s1", "hw": "CPU", "role": "fallback"},
        ],
    },
    "achla": {
        "dir_name": "achla_multi_temporal_uncertainty_quantification",
        "hf_repo": "DaneshSelwal/Classification-AchlaSAR-Models",
        "runners": [
            {"slug": "achla-suite-runner-k22", "hw": "GPU", "role": "primary"},
            {"slug": "achla-cpu-runner-k27", "hw": "CPU", "role": "fallback"},
            {"slug": "achla-cpu-runner-k27-s2", "hw": "CPU", "role": "fallback"},
        ],
    },
    "planethisar": {
        "dir_name": "planet_data_hisar_uncertainty_quantification",
        "hf_repo": "DaneshSelwal/Classification-PlanetHisar-Models",
        "runners": [
            {"slug": "planethisar-suite-runner-k23", "hw": "GPU", "role": "primary"},
            {"slug": "cla21-11-2026-23-02", "hw": "GPU", "role": "primary"},
            {"slug": "planethisar-cpu-runner-k28", "hw": "CPU", "role": "fallback"},
            {"slug": "planethisar-cpu-runner-k28-s1", "hw": "CPU", "role": "fallback"},
            {"slug": "planethisar-cpu-runner-k28-s2", "hw": "CPU", "role": "fallback"},
        ],
    },
}


def list_kernel_files(api: KaggleApi, slug: str, kaggle_user: str = "daneshselwal") -> List:
    """Paginate and return all output files from a Kaggle kernel session."""
    files = []
    token = None
    try:
        with api.build_kaggle_client() as client:
            while True:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = kaggle_user
                req.kernel_slug = slug
                if token:
                    req.page_token = token
                resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
                if not resp or not hasattr(resp, "files") or not resp.files:
                    break
                files.extend(resp.files)
                token = getattr(resp, "next_page_token", None)
                if not token:
                    break
    except Exception as exc:
        print(f"  [WARN] Failed to list files for {kaggle_user}/{slug}: {exc}", file=sys.stderr)
    return files


def should_skip_name(name: str) -> bool:
    """Filter out forbidden directories, fullgt.csv, model weights, and unsupported files."""
    path_parts = set(Path(name).parts)
    if any(bad in path_parts for bad in {".cache", "Multi-CP", ".git", "catboost_info", "__pycache__", "suite_notebooks"}):
        return True
    if name.startswith(BAD_PREFIXES):
        return True
    base = Path(name).name.lower()
    if base in {
        "fullgt.csv",
        "all_etmls.csv",
        "ref.csv",
        "vv_dectomarch.csv",
        "vh_dectomarch.csv",
        "gt.csv",
        "planet.csv",
        "planetgt.csv",
    }:
        return True
    ext = Path(name).suffix.lower()
    if ext not in VALID_EXTS:
        return True
    # Strictly forbid any model weights
    if ext in {".keras", ".h5", ".hdf5", ".ckpt", ".pt", ".pth", ".safetensors"}:
        return True
    return False


def get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face API token from environment, cache, or Kaggle secrets."""
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        return token.strip()
    cache_token = Path.home() / ".cache/huggingface/token"
    if cache_token.exists():
        val = cache_token.read_text().strip()
        if val:
            return val
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    return None


def stream_decouple_runner_models(
    api: Optional[KaggleApi],
    slug: str,
    repo_id: str,
    files: Optional[List] = None,
    allow_partial: bool = False,
    dry_run: bool = False,
    kaggle_user: str = "daneshselwal",
) -> bool:
    """Stream-decouple .keras/.h5 model weights from a Kaggle runner to HF Hub via ephemeral Xet scratch (0 repo/tmp persistence)."""
    hf_token = get_hf_token()
    if not hf_token:
        print(f"  [ERROR] Cannot decouple {slug} weights: HF_TOKEN not found.", file=sys.stderr)
        return False

    hf_api = HfApi(token=hf_token)
    if api is None:
        api = KaggleApi()
        api.authenticate()

    if files is None:
        files = list_kernel_files(api, slug, kaggle_user=kaggle_user)

    weight_files = [
        f for f in files
        if (f.file_name.endswith(".keras") or f.file_name.endswith(".h5"))
        and not any(p in f.file_name for p in (".cache/", "Multi-CP/", "suite_notebooks/"))
    ]
    if not weight_files:
        return False

    try:
        existing_hf = set(hf_api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception:
        existing_hf = set()

    # Deduplicate by target_path, preferring canonical models/... sources
    by_target: Dict[str, Tuple] = {}
    for f in sorted(weight_files, key=lambda x: (0 if x.file_name.startswith("models/") else 1, x.file_name)):
        target_path = f.file_name if f.file_name.startswith("models/") else f"models/{Path(f.file_name).name}"
        if target_path in existing_hf or Path(f.file_name).name in existing_hf:
            continue
        if target_path not in by_target:
            by_target[target_path] = (f, target_path)
    pending = [by_target[k] for k in sorted(by_target)]

    if not pending:
        print(f"  [HF Decouple] All {len(weight_files)} weight files from {slug} already live on {repo_id}.")
        return True

    print("\n" + "=" * 80)
    print(f"  STREAMING WEIGHT DECOUPLING: {kaggle_user}/{slug} -> HF Hub ({repo_id})")
    print("=" * 80)
    print(f"  Discovered {len(pending)} unique new model weights (out of {len(weight_files)} in session output).")

    if dry_run:
        for f, target_path in pending:
            print(f"    - [DRY-RUN] {f.file_name} -> {target_path}")
        return True

    uploaded = 0
    for f, target_path in pending:
        print(f"  -> Streaming {f.file_name} -> HF Hub ({repo_id}/{target_path})...")
        req_dl = urllib.request.Request(f.url, headers={"User-Agent": "Mozilla/5.0"})
        with tempfile.TemporaryDirectory(prefix="hf_xet_") as tmpdir:
            tmp_file = Path(tmpdir) / Path(target_path).name
            with urllib.request.urlopen(req_dl, timeout=300) as r, open(tmp_file, "wb") as out_fp:
                shutil.copyfileobj(r, out_fp, length=4 * 1024 * 1024)
            fsize = tmp_file.stat().st_size
            hf_api.upload_file(
                path_or_fileobj=str(tmp_file),
                path_in_repo=target_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Decouple {target_path} from {slug}",
            )
            print(f"     [SUCCESS] Decoupled {target_path} ({fsize:,} bytes) to HF Hub.")
        uploaded += 1

    print(f"  Decoupling complete: {uploaded}/{len(pending)} model weights uploaded to {repo_id}.\n")
    return True


def stream_decouple_hisar3(
    api: Optional[KaggleApi] = None,
    allow_partial: bool = False,
    dry_run: bool = False,
    kaggle_user: str = "daneshselwal",
) -> bool:
    """Stream-decouple ensemble weights from hisar3-suite-runner-k13 to HF Hub in-memory (0 disk writes)."""
    return stream_decouple_runner_models(
        api=api,
        slug="hisar3-suite-runner-k13",
        repo_id="DaneshSelwal/Classification-Hisar3-Models",
        allow_partial=allow_partial,
        dry_run=dry_run,
        kaggle_user=kaggle_user,
    )


def is_valid_payload(ext: str, data: bytes) -> Tuple[bool, str]:
    """Validate payload content against blank CSVs and placeholder XLSX files."""
    if not data:
        return False, "empty bytes"
    if len(data) > 49 * 1024 * 1024:
        return False, f"oversize ({len(data) / 1048576:.1f} MB > 49 MB git limit; host on HF Hub instead)"

    if ext == ".csv":
        stripped = data.strip()
        if len(stripped) < 5 or len(data) <= 2:
            return False, f"blank csv ({len(data)}B)"
        return True, f"csv ({len(data)}B)"

    if ext == ".json":
        try:
            obj = json.loads(data.decode("utf-8"))
            if not obj and obj != 0 and obj is not False:
                return False, "empty json"
            return True, f"json ({len(data)}B)"
        except Exception as exc:
            return False, f"invalid json: {exc}"

    if ext == ".xlsx":
        try:
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
            max_rows = max((wb[s].max_row or 0) for s in wb.sheetnames) if wb.sheetnames else 0
            wb.close()
            # If max_row <= 1 and size < 6KB, it is an empty header-only placeholder
            if max_rows <= 1 and len(data) < 6000:
                return False, f"header-only placeholder xlsx (max_row={max_rows}, {len(data)}B)"
            return True, f"xlsx ({len(data)}B, max_row={max_rows})"
        except Exception as exc:
            return False, f"invalid xlsx: {exc}"

    if ext == ".png":
        if len(data) < 100 or not data.startswith(b"\x89PNG"):
            return False, f"invalid png ({len(data)}B)"
        return True, f"png ({len(data)}B)"

    return False, f"unsupported ext {ext}"


MAX_LOCAL_BYTES = 48 * 1024 * 1024


def download_and_save(item: Tuple[str, Path, str, bool]) -> Tuple[str, str, str, str]:
    """Download item from URL, validate payload, and save to destination path."""
    url, dest_path, rel_src, dry_run = item
    ext = Path(rel_src).suffix.lower()

    if dry_run:
        return "planned", rel_src, str(dest_path), f"dry-run {ext}"

    if dest_path.exists() and dest_path.stat().st_size > 0:
        try:
            existing_bytes = dest_path.read_bytes()
            ok_cached, reason_cached = is_valid_payload(ext, existing_bytes)
            if ok_cached and len(existing_bytes) <= MAX_LOCAL_BYTES:
                return "saved", rel_src, str(dest_path), f"cached {reason_cached}"
        except Exception:
            pass

    attempts = 4
    data = None
    last_err = None
    for attempt in range(attempts):
        try:
            from urllib.parse import urlsplit, urlunsplit, quote
            parts = urlsplit(url)
            clean_path = quote(parts.path, safe="/%")
            clean_url = urlunsplit((parts.scheme, parts.netloc, clean_path, parts.query, parts.fragment))
            req = urllib.request.Request(clean_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                cl = int(resp.headers.get("Content-Length") or 0)
                if cl > MAX_LOCAL_BYTES:
                    return "skipped", rel_src, str(dest_path), f"skipped >48MB for Git ({cl}B)"
                data = resp.read()
            break
        except Exception as exc:
            last_err = exc
            if attempt < attempts - 1:
                import time
                time.sleep(1.0)
    if data is None:
        return "error", rel_src, str(dest_path), f"download failed: {last_err}"

    if len(data) > MAX_LOCAL_BYTES:
        return "skipped", rel_src, str(dest_path), f"skipped >48MB for Git ({len(data)}B)"

    ok, reason = is_valid_payload(ext, data)
    if not ok:
        return "skipped", rel_src, str(dest_path), reason

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(data)
        return "saved", rel_src, str(dest_path), reason
    except Exception as exc:
        return "error", rel_src, str(dest_path), f"write error: {exc}"


def plan_downloads_for_runner(
    runner_slug: str,
    suite_dir: Path,
    files: List,
    dry_run: bool = False,
    results_target: bool = True,
) -> List[Tuple[str, Path, str, bool]]:
    """Map output files from a runner into examples/<suite>/results and logs."""
    jobs = []
    seen_dest: Set[str] = set()

    def add_job(url: str, dest_path: Path, rel_src: str):
        key = str(dest_path.resolve())
        if key not in seen_dest:
            seen_dest.add(key)
            jobs.append((url, dest_path, rel_src, dry_run))

    results_dir = suite_dir / "results"
    runner_log_dir = suite_dir / "logs" / runner_slug

    filtered_files = {f.file_name: f.url for f in files if not should_skip_name(f.file_name)}

    for name, url in sorted(filtered_files.items()):
        rel_src = f"{runner_slug}:{name}"
        ext = Path(name).suffix.lower()
        base_name = Path(name).name

        # 1. Pipeline Execution Manifest
        if base_name == "pipeline_execution_manifest.csv":
            add_job(url, runner_log_dir / "pipeline_execution_manifest.csv", rel_src)
            if results_target:
                add_job(url, results_dir / "pipeline_execution_manifest.csv", rel_src)
            continue

        # 2. Output prefix
        if name.startswith("output/"):
            rel = name[len("output/"):]
            if results_target:
                add_job(url, results_dir / rel, rel_src)
            if ext in {".csv", ".xlsx"} and "/" not in rel:
                add_job(url, runner_log_dir / rel, rel_src)
            continue

        # 3. Results prefix
        if name.startswith("results/"):
            rel = name[len("results/"):]
            if results_target:
                add_job(url, results_dir / rel, rel_src)
            if ext in {".csv", ".xlsx"} and "/" not in rel:
                add_job(url, runner_log_dir / rel, rel_src)
            continue

        # 4. Ensemble results prefix
        if name.startswith("ensemble/results/"):
            rel = name[len("ensemble/results/"):]
            if results_target:
                add_job(url, results_dir / "ensemble" / rel, rel_src)
            if ext in {".csv", ".xlsx"}:
                add_job(url, runner_log_dir / base_name, rel_src)
            continue

        # 5. CREDIT results prefix
        if name.startswith("credit/results/"):
            rel = name[len("credit/results/"):]
            if results_target:
                add_job(url, results_dir / "credit" / rel, rel_src)
            if ext in {".csv", ".xlsx"}:
                add_job(url, runner_log_dir / base_name, rel_src)
            continue

        # 6. Models prefix (metrics, json reports, classification tables)
        if name.startswith("models/"):
            rel = name[len("models/"):]
            if results_target:
                add_job(url, results_dir / "models" / rel, rel_src)
            if ext in {".csv", ".xlsx"}:
                add_job(url, runner_log_dir / base_name, rel_src)
            continue

        # 7. Root-level tables or visualizations
        if results_target:
            add_job(url, results_dir / name, rel_src)
        if ext in {".csv", ".xlsx"}:
            add_job(url, runner_log_dir / base_name, rel_src)

    return jobs


def check_and_extract(
    api: KaggleApi,
    suite_key: Optional[str] = None,
    runner_slug: Optional[str] = None,
    dry_run: bool = False,
    workers: int = 12,
    status_only: bool = False,
    allow_partial: bool = False,
    skip_weights: bool = False,
    kaggle_user: str = "daneshselwal",
):
    """Check all runner statuses and extract valid outputs."""
    suites_to_process = (
        {suite_key: SUITES[suite_key]} if suite_key and suite_key in SUITES else SUITES
    )

    print("=" * 80)
    print(f"  KAGGLE BENCHMARK RUNNERS STATUS & OUTPUT EXTRACTION ({kaggle_user})")
    print("=" * 80)

    all_jobs: List[Tuple[str, Path, str, bool]] = []
    runner_summary: List[Tuple[str, str, str, str, int]] = []

    for s_name, s_meta in suites_to_process.items():
        if runner_slug:
            runners_for_suite = [
                {**r, "slug": runner_slug}
                for r in s_meta["runners"]
                if r["slug"] == runner_slug or runner_slug.startswith(r["slug"] + "-")
            ]
        else:
            runners_for_suite = s_meta["runners"]
        if not runners_for_suite:
            continue

        suite_dir = REPO_ROOT / "examples" / s_meta["dir_name"]
        print(f"\n[Suite] {s_name.upper()} ({s_meta['dir_name']})")

        # First probe runner statuses to check if primary is complete
        primary_complete = False
        runner_statuses = {}
        for runner in runners_for_suite:
            slug = runner["slug"]
            try:
                status_obj = api.kernels_status(f"{kaggle_user}/{slug}")
                status_str = getattr(status_obj, "status", str(status_obj))
            except Exception as exc:
                status_str = f"ERROR ({exc})"
            runner_statuses[slug] = status_str
            if not runner_slug and runner["role"] == "primary" and "COMPLETE" in str(status_str).upper():
                primary_complete = True

        for runner in runners_for_suite:
            slug = runner["slug"]
            hw = runner["hw"]
            role = runner["role"]
            status_str = runner_statuses[slug]

            files = []
            if not status_only:
                files = list_kernel_files(api, slug, kaggle_user=kaggle_user)

            runner_summary.append((s_name, slug, hw, str(status_str), len(files)))
            print(f"  • {slug:28s} [{hw:3s} | {role:8s}]: status={status_str} | session_files={len(files)}")

            if files and not status_only:
                # If runner is COMPLETE (or allow_partial and CANCEL_ACKNOWLEDGED), stream-decouple any .keras/.h5 weights to HF Hub
                hf_repo = s_meta.get("hf_repo")
                if not skip_weights and hf_repo and (
                    "COMPLETE" in str(status_str).upper()
                    or (allow_partial and "CANCEL_ACKNOWLEDGED" in str(status_str).upper())
                ):
                    stream_decouple_runner_models(
                        api,
                        slug=slug,
                        repo_id=hf_repo,
                        files=files,
                        allow_partial=allow_partial,
                        dry_run=dry_run,
                        kaggle_user=kaggle_user,
                    )

                target_results = not (role == "fallback" and primary_complete)
                jobs = plan_downloads_for_runner(
                    slug, suite_dir, files, dry_run=dry_run, results_target=target_results
                )
                all_jobs.extend(jobs)
                print(f"    -> Planned {len(jobs)} valid download jobs from {slug}")

    if status_only:
        print("\n[Status Only] Finished querying runner statuses.")
        return

    print("\n" + "=" * 80)
    print(f"Executing {len(all_jobs)} download jobs (dry_run={dry_run}, workers={workers})...")
    print("=" * 80)

    saved = 0
    skipped: List[Tuple[str, str]] = []
    errors: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_and_save, j) for j in all_jobs]
        for fut in as_completed(futures):
            try:
                status, src, dest, reason = fut.result()
                if status in {"saved", "planned"}:
                    saved += 1
                elif status == "skipped":
                    skipped.append((src, reason))
                else:
                    errors.append((src, reason))
            except Exception as exc:
                errors.append(("unknown", str(exc)))

    print(f"\nExtraction Results: saved={saved}, skipped={len(skipped)}, errors={len(errors)}")

    if skipped:
        print(f"\nSkipped {len(skipped)} blank/placeholder/invalid files:")
        for src, reason in sorted(skipped)[:25]:
            print(f"  - {src}: {reason}")
        if len(skipped) > 25:
            print(f"  ... and {len(skipped) - 25} more skipped files.")

    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for src, reason in sorted(errors)[:20]:
            print(f"  - {src}: {reason}")

    print("\n" + "=" * 80)
    print("RUNNER SUMMARY TABLE:")
    print(f"{'Suite':18s} | {'Runner':25s} | {'HW':3s} | {'Status':20s} | {'Files':5s}")
    print("-" * 80)
    for s_name, slug, hw, st, cnt in runner_summary:
        print(f"{s_name:18s} | {slug:25s} | {hw:3s} | {st:20s} | {cnt:<5d}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Kaggle benchmark runners and extract valid outputs into examples/."
    )
    parser.add_argument(
        "--suite",
        choices=list(SUITES.keys()) + ["all"],
        default="all",
        help="Target benchmark suite (default: all)",
    )
    parser.add_argument(
        "--runner",
        default=None,
        help="Optional single runner slug filter (e.g. hissar25-cpu-runner-k18)",
    )
    parser.add_argument(
        "--kaggle-user",
        default="daneshselwal",
        help="Kaggle account username owning the kernels (default: daneshselwal)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only display status table without downloading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned downloads without saving files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of concurrent download threads (default: 12)",
    )
    parser.add_argument(
        "--decouple-hisar3",
        action="store_true",
        help="Stream-decouple Hisar-3 ensemble model weights directly to Hugging Face Hub in-memory",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow decoupling and extraction from CANCEL_ACKNOWLEDGED / timed-out runners with partial outputs",
    )
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip HF Hub weight decoupling during extraction (useful when running decoupling asynchronously)",
    )
    args = parser.parse_args()

    user_config_dirs = {
        "daneshjangra": "~/.kaggle_alt",
        "geminicli": "~/.kaggle_alt3",
        "daneshselwal1": "~/.kaggle_alt4",
    }
    if args.kaggle_user in user_config_dirs and "KAGGLE_CONFIG_DIR" not in os.environ:
        alt_dir = Path(user_config_dirs[args.kaggle_user]).expanduser()
        if (alt_dir / "access_token").exists():
            os.environ["KAGGLE_CONFIG_DIR"] = str(alt_dir)

    cfg_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle")).expanduser()
    tok_file = cfg_dir / "access_token"
    if tok_file.exists() and not os.environ.get("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_API_TOKEN"] = tok_file.read_text(encoding="utf-8").strip()

    api = KaggleApi()
    api.authenticate()

    if args.decouple_hisar3:
        stream_decouple_hisar3(
            api,
            allow_partial=args.allow_partial,
            dry_run=args.dry_run,
            kaggle_user=args.kaggle_user,
        )
        if args.status_only or "--suite" not in sys.argv:
            return

    suite_arg = None if args.suite == "all" else args.suite
    check_and_extract(
        api=api,
        suite_key=suite_arg,
        runner_slug=args.runner,
        dry_run=args.dry_run,
        workers=args.workers,
        status_only=args.status_only,
        allow_partial=args.allow_partial,
        skip_weights=args.skip_weights,
        kaggle_user=args.kaggle_user,
    )


if __name__ == "__main__":
    main()
