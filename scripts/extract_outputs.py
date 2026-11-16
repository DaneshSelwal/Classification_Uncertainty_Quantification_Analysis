#!/usr/bin/env python3
"""
Enhanced 4-Suite Benchmark Runner Output Extractor & Monitor.

Monitors Kaggle GPU & CPU runners across all 4 benchmark suites:
- indian_pines_uncertainty_quantification (k11 / k15)
- pavia_university_uncertainty_quantification (k12 / k16)
- hisar_3mts_uncertainty_quantification (k13 / k17)
- hissar_25_uncertainty_quantification (k14 / k18)

Downloads all valid non-empty CSV/XLSX/JSON/PNG outputs and logs into:
- examples/<suite>/results/
- examples/<suite>/logs/<runner>/

Strict R2 Filtering:
- Skips blank CSVs (<= 2 bytes, or stripped length < 5 bytes).
- Skips header-only placeholder .xlsx (max_row <= 1 and size < 6000 bytes).
- Skips fullgt.csv (case-insensitive).
- Skips forbidden prefixes: .cache/, Multi-CP/, .git/, data/, catboost_info/, __pycache__/.
- Enforces zero model weight extraction (never extracts .keras, .h5, etc.).
"""

import argparse
import io
import json
import os
import sys
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
)

SUITES: Dict[str, dict] = {
    "indian_pines": {
        "dir_name": "indian_pines_uncertainty_quantification",
        "runners": [
            {"slug": "ip-suite-runner-k11", "hw": "GPU", "role": "primary"},
            {"slug": "ip-cpu-runner-k15", "hw": "CPU", "role": "fallback"},
        ],
    },
    "pavia_university": {
        "dir_name": "pavia_university_uncertainty_quantification",
        "runners": [
            {"slug": "pavia-suite-runner-k12", "hw": "GPU", "role": "primary"},
            {"slug": "pavia-cpu-runner-k16", "hw": "CPU", "role": "fallback"},
        ],
    },
    "hisar_3mts": {
        "dir_name": "hisar_3mts_uncertainty_quantification",
        "runners": [
            {"slug": "hisar3-suite-runner-k13", "hw": "GPU", "role": "primary"},
            {"slug": "hisar3-cpu-runner-k17", "hw": "CPU", "role": "fallback"},
        ],
    },
    "hissar_25": {
        "dir_name": "hissar_25_uncertainty_quantification",
        "runners": [
            {"slug": "hissar25-suite-runner-k14", "hw": "GPU", "role": "primary"},
            {"slug": "hissar25-cpu-runner-k18", "hw": "CPU", "role": "fallback"},
        ],
    },
}


def list_kernel_files(api: KaggleApi, slug: str) -> List:
    """Paginate and return all output files from a Kaggle kernel session."""
    files = []
    token = None
    try:
        with api.build_kaggle_client() as client:
            while True:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = "daneshselwal"
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
        print(f"  [WARN] Failed to list files for {slug}: {exc}", file=sys.stderr)
    return files


def should_skip_name(name: str) -> bool:
    """Filter out forbidden directories, fullgt.csv, model weights, and unsupported files."""
    path_parts = set(Path(name).parts)
    if any(bad in path_parts for bad in {".cache", "Multi-CP", ".git", "catboost_info", "__pycache__"}):
        return True
    if name.startswith(BAD_PREFIXES):
        return True
    base = Path(name).name.lower()
    if base == "fullgt.csv":
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


def stream_decouple_hisar3(
    api: Optional[KaggleApi] = None,
    allow_partial: bool = False,
    dry_run: bool = False,
) -> bool:
    """Stream-decouple ensemble weights from hisar3-suite-runner-k13 to HF Hub in-memory (0 disk writes)."""
    hf_token = get_hf_token()
    if not hf_token:
        print("  [ERROR] Cannot decouple Hisar-3 weights: HF_TOKEN not found.", file=sys.stderr)
        return False

    hf_api = HfApi(token=hf_token)
    repo_id = "DaneshSelwal/Classification-Hisar3-Models"
    slug = "hisar3-suite-runner-k13"

    if api is None:
        api = KaggleApi()
        api.authenticate()

    print("\n" + "=" * 80)
    print(f"  STREAMING WEIGHT DECOUPLING: {slug} -> HF Hub ({repo_id})")
    print("=" * 80)

    try:
        status_obj = api.kernels_status("daneshselwal/" + slug)
        status_str = getattr(status_obj, "status", str(status_obj))
    except Exception as exc:
        print(f"  [ERROR] Failed to query status for {slug}: {exc}", file=sys.stderr)
        return False

    print(f"  Runner status: {status_str}")
    allowed_terms = ("COMPLETE", "CANCEL_ACKNOWLEDGED") if allow_partial else ("COMPLETE",)
    if not any(st in str(status_str).upper() for st in allowed_terms):
        print(f"  Runner {slug} is not COMPLETE yet (current status: {status_str}). Ensembles cannot be decoupled until execution finishes.")
        return False

    files = list_kernel_files(api, slug)
    ensemble_files = [
        f for f in files
        if f.file_name.startswith("models/ensembles/") and f.file_name.endswith(".keras")
    ]
    print(f"  Discovered {len(ensemble_files)} ensemble model weights in session output.")
    if not ensemble_files:
        print(f"  [WARN] No models/ensembles/*.keras found in {slug} session output.")
        return False

    if dry_run:
        print(f"  [DRY-RUN] Would decouple {len(ensemble_files)} ensemble model weights to HF Hub ({repo_id}):")
        for f in sorted(ensemble_files, key=lambda x: x.file_name):
            print(f"    - {f.file_name}")
        return True

    uploaded = 0
    for f in sorted(ensemble_files, key=lambda x: x.file_name):
        print(f"  -> Streaming {f.file_name} directly to HF Hub ({repo_id})...")
        req_dl = urllib.request.Request(f.url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_dl, timeout=120) as r:
            data = r.read()

        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(data),
            path_in_repo=f.file_name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Decouple {f.file_name} from {slug}",
        )
        print(f"     [SUCCESS] Decoupled {f.file_name} ({len(data):,} bytes) to HF Hub.")
        del data  # Keep heap bounded to ~260MB, zero disk persistence
        uploaded += 1

    print(f"  Decoupling complete: {uploaded}/{len(ensemble_files)} ensembles uploaded to {repo_id}.\n")
    return True


def is_valid_payload(ext: str, data: bytes) -> Tuple[bool, str]:
    """Validate payload content against blank CSVs and placeholder XLSX files."""
    if not data:
        return False, "empty bytes"

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


def download_and_save(item: Tuple[str, Path, str, bool]) -> Tuple[str, str, str, str]:
    """Download item from URL, validate payload, and save to destination path."""
    url, dest_path, rel_src, dry_run = item
    ext = Path(rel_src).suffix.lower()

    if dry_run:
        return "planned", rel_src, str(dest_path), f"dry-run {ext}"

    attempts = 3
    data = None
    last_err = None
    for attempt in range(attempts):
        try:
            from urllib.parse import urlsplit, urlunsplit, quote
            parts = urlsplit(url)
            clean_path = quote(parts.path, safe="/%")
            clean_url = urlunsplit((parts.scheme, parts.netloc, clean_path, parts.query, parts.fragment))
            req = urllib.request.Request(clean_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
            break
        except Exception as exc:
            last_err = exc
            if attempt < attempts - 1:
                import time
                time.sleep(1.0)
    if data is None:
        return "error", rel_src, str(dest_path), f"download failed: {last_err}"

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
    dry_run: bool = False,
    workers: int = 12,
    status_only: bool = False,
    allow_partial: bool = False,
):
    """Check all runner statuses and extract valid outputs."""
    suites_to_process = (
        {suite_key: SUITES[suite_key]} if suite_key and suite_key in SUITES else SUITES
    )

    print("=" * 80)
    print("  KAGGLE BENCHMARK RUNNERS STATUS & OUTPUT EXTRACTION")
    print("=" * 80)

    all_jobs: List[Tuple[str, Path, str, bool]] = []
    runner_summary: List[Tuple[str, str, str, str, int]] = []

    for s_name, s_meta in suites_to_process.items():
        suite_dir = REPO_ROOT / "examples" / s_meta["dir_name"]
        print(f"\n[Suite] {s_name.upper()} ({s_meta['dir_name']})")

        # First probe runner statuses to check if primary is complete
        primary_complete = False
        runner_statuses = {}
        for runner in s_meta["runners"]:
            slug = runner["slug"]
            try:
                status_obj = api.kernels_status("daneshselwal/" + slug)
                status_str = getattr(status_obj, "status", str(status_obj))
            except Exception as exc:
                status_str = f"ERROR ({exc})"
            runner_statuses[slug] = status_str
            if runner["role"] == "primary" and "COMPLETE" in str(status_str).upper():
                primary_complete = True

        for runner in s_meta["runners"]:
            slug = runner["slug"]
            hw = runner["hw"]
            role = runner["role"]
            status_str = runner_statuses[slug]

            files = []
            if not status_only:
                files = list_kernel_files(api, slug)

            runner_summary.append((s_name, slug, hw, str(status_str), len(files)))
            print(f"  • {slug:25s} [{hw:3s} | {role:7s}]: status={status_str} | session_files={len(files)}")

            if files and not status_only:
                # If hisar3-suite-runner-k13 is COMPLETE (or allow_partial and CANCEL_ACKNOWLEDGED), stream-decouple its ensemble weights to HF Hub
                if slug == "hisar3-suite-runner-k13" and (
                    "COMPLETE" in str(status_str).upper()
                    or (allow_partial and "CANCEL_ACKNOWLEDGED" in str(status_str).upper())
                ):
                    stream_decouple_hisar3(api, allow_partial=allow_partial, dry_run=dry_run)

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
        choices=["indian_pines", "pavia_university", "hisar_3mts", "hissar_25", "all"],
        default="all",
        help="Target benchmark suite (default: all)",
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
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    if args.decouple_hisar3:
        stream_decouple_hisar3(api, allow_partial=args.allow_partial, dry_run=args.dry_run)
        if args.status_only or "--suite" not in sys.argv:
            return

    suite_arg = None if args.suite == "all" else args.suite
    check_and_extract(
        api=api,
        suite_key=suite_arg,
        dry_run=args.dry_run,
        workers=args.workers,
        status_only=args.status_only,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
