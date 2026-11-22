#!/bin/zsh
# usage: [KAGGLE_TOKEN_FILE=~/.kaggle_alt/access_token KAGGLE_USER=daneshjangra] queue_push.sh "<note>" dir1 [dir2 ...]
# Naming rule: every classification run is named CLA<DD>_<MM>_<YYYY>_<HH>_<MM> = local push time
# (Kaggle slug cla<DD>-<MM>-<YYYY>-<HH>-<MM>). At most ONE push per minute across this project AND the
# regression project: the regression session's lock directory is shared and both registries are checked.
# Retries while Kaggle's GPU session limit is reached. Token is read from a file and never printed.
export KAGGLE_API_TOKEN="$(cat "${KAGGLE_TOKEN_FILE:-$HOME/.kaggle/access_token}")"
KAGGLE_BIN="${KAGGLE_BIN:-$(command -v kaggle || echo /opt/anaconda3/bin/kaggle)}"
KUSER="${KAGGLE_USER:-daneshselwal}"
REG="${RUN_REGISTRY:-$(cd "$(dirname "$0")/../.." && pwd)/docs/run_registry.tsv}"
OTHER_REG="${OTHER_REGISTRY:-/Users/danesh/Desktop/Regression_check/Regression_Uncertainty_Quantification_Analysis/docs/run_registry.tsv}"
LOCK="${OTHER_REG}.lock"; [[ -d "$(dirname "$OTHER_REG")" ]] || LOCK="$REG.lock"
[[ -f "$REG" ]] || printf "code_name\tslug\tsuite\tjob_dir\tpushed_at\tkind\tstatus\n" > "$REG"
NOTE="$1"; shift
for d in "$@"; do
  while true; do
    until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
    now="$(date '+%Y-%m-%d %H:%M')"; code="CLA$(date +%d_%m_%Y_%H_%M)"
    # one push per minute over both projects
    while grep -q "^$code	" "$REG" || { [[ -f "$OTHER_REG" ]] && grep -q "	$now	" "$OTHER_REG"; }; do
      sleep 15; now="$(date '+%Y-%m-%d %H:%M')"; code="CLA$(date +%d_%m_%Y_%H_%M)"; done
    slug=$(echo "$code" | tr 'A-Z_' 'a-z-')
    python3 -I - "$d/kernel-metadata.json" "$KUSER/$slug" "$code" <<'PY'
import json, sys
p, kid, title = sys.argv[1:4]
m = json.load(open(p)); m["id"] = kid; m["title"] = title
json.dump(m, open(p, "w"), indent=1)
PY
    suite=$(head -1 "$d/run.py" | sed 's/^# v2.0 suite runner: //; s/ |.*//')
    out=$("$KAGGLE_BIN" kernels push -p "$d" 2>&1 | grep -v outdated)
    if echo "$out" | grep -q "successfully pushed"; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$code" "$KUSER/$slug" "$suite" "$(basename "$d")" "$now" "gpu" "RUNNING $NOTE" >> "$REG"
      rmdir "$LOCK"; echo "$(date +%H:%M) $code ($suite) pushed: https://www.kaggle.com/code/$KUSER/$slug"; break
    fi
    rmdir "$LOCK"
    if ! echo "$out" | grep -qi "Maximum.*session"; then echo "$(date +%H:%M) ERROR $d: $out"; break; fi
    sleep 120
  done
done
echo QUEUE_DONE
