import glob
import json


FORBIDDEN_SUBSTRINGS = [
    "/content/drive/My Drive/m_p/",
    "/content/drive/MyDrive/m_p/",
    "/content/drive/My Drive/m_p",
    "C:\\Users\\",
    "uncertainty_results",
    "/saved_models",
]

WRITE_CALL_HINTS = [
    ".to_csv(",
    ".to_excel(",
    ".savefig(",
    "np.save(",
    "np.savez(",
    "open(",
    ".save(",
    "model.save(",
]


def iter_code_cells(nb: dict):
    for cell in nb.get("cells") or []:
        if cell.get("cell_type") == "code":
            yield cell.get("source") or []


def compile_cells(nb_path: str, nb: dict) -> list[str]:
    errors: list[str] = []
    for idx, src_lines in enumerate(iter_code_cells(nb), start=1):
        code = "".join(src_lines)
        try:
            compile(code, f"{nb_path}::cell{idx}", "exec")
        except Exception as e:
            errors.append(f"{nb_path}: code cell {idx}: {e}")
    return errors


def scan_forbidden(nb_path: str, nb: dict) -> list[str]:
    hits: list[str] = []
    for idx, src_lines in enumerate(iter_code_cells(nb), start=1):
        text = "".join(src_lines)
        for s in FORBIDDEN_SUBSTRINGS:
            if s in text:
                hits.append(f"{nb_path}: code cell {idx}: contains {s!r}")
    return hits


def scan_write_destinations(nb_path: str, nb: dict) -> list[str]:
    hits: list[str] = []
    for idx, src_lines in enumerate(iter_code_cells(nb), start=1):
        text = "".join(src_lines)
        if not any(h in text for h in WRITE_CALL_HINTS):
            continue
        for s in FORBIDDEN_SUBSTRINGS:
            if s in text:
                hits.append(
                    f"{nb_path}: code cell {idx}: write-related cell contains forbidden root {s!r}"
                )
    return hits


def validate_json(nb_path: str) -> list[str]:
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            json.load(f)
        return []
    except Exception as e:
        return [f"{nb_path}: invalid JSON: {e}"]


def main() -> int:
    nb_paths = sorted(glob.glob("**/*.ipynb", recursive=True))
    if not nb_paths:
        print("No notebooks found.")
        return 0

    all_errors: list[str] = []

    for nb_path in nb_paths:
        all_errors.extend(validate_json(nb_path))
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        all_errors.extend(compile_cells(nb_path, nb))
        all_errors.extend(scan_forbidden(nb_path, nb))
        all_errors.extend(scan_write_destinations(nb_path, nb))

    if all_errors:
        print("NOTEBOOK_VERIFY_FAILED")
        for e in all_errors:
            print(e)
        return 1

    print("NOTEBOOK_VERIFY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
