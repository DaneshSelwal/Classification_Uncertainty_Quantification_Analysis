"""Unattended runner for the v2.0 Kaggle plan (docs/run_plan.tsv).

usage: python3 tools/kaggle/run_plan.py            # loop until the plan is finished (run it in the background)
       python3 tools/kaggle/run_plan.py --once     # one cycle (poll, collect, push), then exit
       python3 tools/kaggle/run_plan.py --status   # print the plan
       add --no-push to only poll / collect (no new kernels)

Each cycle
  1. polls every RUNNING row; when a kernel has finished it downloads stage_status.json, the log, the tables and
     the executed notebooks, writes DONE_OK / DONE_FAILED (+ failed stages) into the plan and the run registry;
       smoke -> docs/smoke_runs/<CODE>/      full -> examples/<suite>/logs/v2.0_<CODE>/
  2. fills free GPU sessions (max 2 per account) with the next PENDING row it may start:
       * a `full` row only after the `smoke` row of the same suite is DONE_OK;
       * the account column may name one account or `*` (any);
       * pushes go through tools/kaggle/queue_push.sh (naming CLA..., one push per minute, shared lock);
       * "Maximum weekly GPU quota" -> that account is marked exhausted for the rest of this run;
  3. appends the GPU usage of both accounts to docs/gpu_ledger.tsv (every hour) and sleeps 5 minutes.
It stops when no row is PENDING or RUNNING, or when nothing can start because every account is exhausted.
A DONE_FAILED smoke blocks its full run: fix the notebooks, set the smoke row back to PENDING, restart.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/run_plan.tsv"
REG = ROOT / "docs/run_registry.tsv"
JOBS = ROOT / ".kaggle_jobs"                     # generated kernels (git-ignored)
ACCOUNTS = {                                     # account -> directory holding its access_token
    "daneshselwal": Path.home() / ".kaggle",
    "daneshjangra": Path.home() / ".kaggle_alt",
}
MAX_SESSIONS = 2
POLL_S = 300
COLS = ["suite", "kind", "account", "code_name", "slug", "status", "note"]
KAGGLE = os.environ.get("KAGGLE_BIN") or subprocess.run(["zsh", "-lc", "command -v kaggle"], capture_output=True,
                                                         text=True).stdout.strip() or "kaggle"


def log(*a):
    print(time.strftime("%m-%d %H:%M"), *a, flush=True)


def load():
    rows = []
    for l in PLAN.read_text().splitlines()[1:]:
        if l.strip():
            f = l.split("\t"); f += [""] * (len(COLS) - len(f)); rows.append(dict(zip(COLS, f)))
    return rows


def save(rows):
    PLAN.write_text("\t".join(COLS) + "\n" + "".join("\t".join(r[c] for c in COLS) + "\n" for r in rows))


def env_for(acct):
    e = dict(os.environ)
    e["KAGGLE_API_TOKEN"] = (ACCOUNTS[acct] / "access_token").read_text().strip()
    e["KAGGLE_TOKEN_FILE"] = str(ACCOUNTS[acct] / "access_token"); e["KAGGLE_USER"] = acct
    return e


def kaggle(acct, *args, timeout=900):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, env=env_for(acct), timeout=timeout)
    return "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "outdated" not in l)


def kernel_status(acct, slug):
    out = kaggle(acct, "kernels", "status", slug, timeout=120)
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)
    return m.group(1) if m else "UNKNOWN"


def registry_set(code, status):
    if not REG.exists(): return
    L = REG.read_text().split("\n")
    for i, l in enumerate(L):
        if l.startswith(code + "\t"):
            f = l.split("\t"); f[-1] = status; L[i] = "\t".join(f)
    REG.write_text("\n".join(L))


def collect(r):
    acct, code = r["slug"].split("/")[0], r["code_name"]
    dest = ROOT / ("docs/smoke_runs" if r["kind"] == "smoke" else f"examples/{r['suite']}/logs") / (
        code if r["kind"] == "smoke" else f"v2.0_{code}")
    dest.mkdir(parents=True, exist_ok=True)
    pat = r"output/stage_status\.json|.*\.log" if r["kind"] == "smoke" else \
        r"output/.*\.(ipynb|json)|.*\.csv|.*\.json|.*\.log"
    kaggle(acct, "kernels", "output", r["slug"], "-p", str(dest), "--file-pattern", pat, timeout=3600)
    st = next(dest.rglob("stage_status.json"), None)
    if st is None:
        return "DONE_FAILED", "no stage_status.json (kernel crashed or timed out)"
    s = json.loads(st.read_text())
    bad = [str(x["stage"]) for x in s["stages"] if not x["ok"]]
    n = len(s["stages"])
    if bad or n < 12:
        if r["kind"] == "smoke":   # keep the executed notebooks of a failed smoke for debugging
            kaggle(acct, "kernels", "output", r["slug"], "-p", str(dest), "--file-pattern", r"output/.*\.ipynb",
                   timeout=3600)
        return "DONE_FAILED", f"{s['total_minutes']:.0f} min; failed stages {','.join(bad) or '-'}; ran {n}/12"
    return "DONE_OK", f"{s['total_minutes']:.0f} min; all 12 stages OK"


def may_start(r, rows):
    if r["status"] != "PENDING": return False
    if r["kind"] == "full":
        return any(x["suite"] == r["suite"] and x["kind"] == "smoke" and x["status"] == "DONE_OK" for x in rows)
    return True


def push(r, acct):
    job = JOBS / f"{r['suite']}_{r['kind']}_{acct}"
    cmd = [sys.executable, str(ROOT / "tools/kaggle/make_suite_kernel.py"), str(ROOT / "examples" / r["suite"]),
           str(job), acct] + (["--smoke"] if r["kind"] == "smoke" else [])
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    note = f"{'SMOKE' if r['kind'] == 'smoke' else 'FULL'} {r['suite']} v2.0 (run_plan)"
    out = subprocess.run(["zsh", str(ROOT / "tools/kaggle/queue_push.sh"), note, str(job)], capture_output=True,
                         text=True, env=env_for(acct), timeout=7 * 3600).stdout
    m = re.search(r"(CLA\d\d_\d\d_\d{4}_\d\d_\d\d) \(.*\) pushed: https://www.kaggle.com/code/([\w-]+/[\w-]+)", out)
    if m: return "PUSHED", m.group(1), m.group(2)
    if "weekly GPU quota" in out: return "QUOTA", "", out.strip()[-200:]
    return "ERROR", "", out.strip()[-300:]


def cycle(state):
    rows = load()
    for r in rows:                                   # 1. poll / collect
        if r["status"] != "RUNNING": continue
        acct = r["slug"].split("/")[0]
        st = kernel_status(acct, r["slug"])
        if st in ("RUNNING", "QUEUED", "UNKNOWN"): continue
        res, note = collect(r)
        r["status"], r["note"] = res, f"{st}: {note}"
        registry_set(r["code_name"], f"{res} {r['kind']} {note}")
        log(r["code_name"], r["suite"], r["kind"], res, note); save(rows)
    for acct in ACCOUNTS:                            # 2. fill free sessions
        while acct not in state["exhausted"] and "--no-push" not in sys.argv:
            running = sum(1 for x in rows if x["status"] == "RUNNING" and x["slug"].startswith(acct + "/"))
            if running >= MAX_SESSIONS: break
            cand = next((x for x in rows if may_start(x, rows) and x["account"] in ("*", acct)), None)
            if cand is None: break
            res, code, info = push(cand, acct)
            if res == "PUSHED":
                cand.update(status="RUNNING", code_name=code, slug=info, note=f"pushed {time.strftime('%m-%d %H:%M')}")
                log("pushed", code, cand["suite"], cand["kind"], "on", acct)
            elif res == "QUOTA":
                state["exhausted"].add(acct); log(acct, "weekly GPU quota reached -> no more pushes there")
            else:
                cand.update(status="DONE_FAILED", note="push error: " + info.replace("\t", " ").replace("\n", " "))
                log("push error", cand["suite"], cand["kind"], info)
            save(rows)
    if time.time() - state["ledger"] > 3600:         # 3. GPU ledger
        subprocess.run(["zsh", str(ROOT / "tools/kaggle/gpu_ledger.sh")], capture_output=True)
        state["ledger"] = time.time()
    active = [x for x in rows if x["status"] == "RUNNING"]
    startable = [x for x in rows if may_start(x, rows) and
                 any(a not in state["exhausted"] and x["account"] in ("*", a) for a in ACCOUNTS)]
    return bool(active or startable)


def main():
    if "--status" in sys.argv:
        for r in load(): print(f"{r['suite'][:34]:34s} {r['kind']:5s} {r['account']:12s} {r['code_name']:20s} {r['status']:12s} {r['note'][:70]}")
        return
    JOBS.mkdir(exist_ok=True)
    state = {"exhausted": set(), "ledger": 0}
    while True:
        more = cycle(state)
        if "--once" in sys.argv or not more:
            break
        time.sleep(POLL_S)
    log("plan finished or blocked:"); [print("  ", r["suite"], r["kind"], r["status"], r["note"][:90]) for r in load()]


if __name__ == "__main__":
    main()
