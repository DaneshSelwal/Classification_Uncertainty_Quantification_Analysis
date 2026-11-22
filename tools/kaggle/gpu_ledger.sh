#!/bin/zsh
# usage: gpu_ledger.sh  -> appends one line per account to docs/gpu_ledger.tsv and prints it
# columns: checked_at, account, gpu_used_h, gpu_allowed_h(api), refresh, running CLA kernels
L="$(cd "$(dirname "$0")/../.." && pwd)/docs/gpu_ledger.tsv"
[[ -f "$L" ]] || printf "checked_at\taccount\tgpu_used_h\tgpu_allowed_h_api\trefresh\trunning_cla\n" > "$L"
for pair in "daneshselwal:$HOME/.kaggle" "daneshjangra:$HOME/.kaggle_alt"; do
  acct=${pair%%:*}; dir=${pair#*:}
  for try in 1 2 3; do
  line=$(KAGGLE_API_TOKEN="$(cat $dir/access_token)" python3 -I - "$acct" 2>/dev/null <<'PY'

import re, sys, time, warnings; warnings.filterwarnings("ignore")
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kc:
    q = kc.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
import json; j = json.loads(q.to_json()); g = j["gpuQuota"]
sec = lambda s: float(re.match(r"[\d.]+?(?=\.\d+s$|s$)|[\d.]+", s.rstrip("s")).group(0)) if s else 0.0
used = float(g["timeUsed"].rstrip("s").split(".")[0]) + (float("0." + g["timeUsed"].rstrip("s").split(".")[1]) if g["timeUsed"].count(".") >= 1 else 0)
run = []
for k in api.kernels_list(mine=True, sort_by="dateRun", page_size=20):
    if str(k.ref).split("/")[-1].startswith("cla"):
        try: st = str(api.kernels_status(k.ref).status)
        except Exception: st = ""
        if "RUNNING" in st or "QUEUED" in st: run.append(str(k.ref).split("/")[-1])
print(f"{time.strftime('%Y-%m-%d %H:%M')}\t{sys.argv[1]}\t{used/3600:.2f}\t{float(g['totalTimeAllowed'].rstrip('s'))/3600:.1f}\t{j['quotaRefreshTime'][:10]}\t{','.join(run) or '-'}")
PY
)
  line=$(echo "$line" | grep '^20')
  [[ -n "$line" ]] && { echo "$line" >> "$L"; break; }; sleep 10; done
done
tail -2 "$L"
