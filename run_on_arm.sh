#!/usr/bin/env bash
# Run the cost bench on a remote ARM host and bring the results back.
#
# WHY: every cost figure in the paper comes from x86. The model
#   T = N * t_pass(N)
# claims an engineer can measure t_pass on their own hardware, count the passes,
# and predict the runtime without repeating this bench. A second ARCHITECTURE is
# what tests that claim: the pass COUNTS must come back identical (they are a
# property of the method) while the milliseconds must differ (they are a
# property of the machine) -- and the model must predict the new milliseconds
# from the t_pass measured there.
#
# NOTE FOR THE PAPER: an AWS Graviton is a SERVER-CLASS ARM part with large
# caches and many cores. It is NOT automotive silicon. Report it as "a second
# architecture", never as "embedded hardware".
#
# Usage:
#   ./run_on_arm.sh ubuntu@<ip> ~/.ssh/<key>.pem
#
# Requires on the remote: Ubuntu 22.04/24.04 arm64, ~20 GB disk, python3.
set -euo pipefail

HOST="${1:?usage: $0 user@host /path/to/key.pem}"
KEY="${2:?usage: $0 user@host /path/to/key.pem}"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
SCP="scp -i $KEY -o StrictHostKeyChecking=accept-new"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/results/arm-$STAMP"

echo "== 1/5 remote architecture =="
$SSH "$HOST" 'uname -m; nproc; grep -m1 "^model name\|^Model" /proc/cpuinfo || true; free -g | sed -n 2p'

echo "== 2/5 shipping the repository (no data, no venv) =="
$SSH "$HOST" 'rm -rf ~/xai-bench && mkdir -p ~/xai-bench'
$SCP -q "$HERE"/*.py "$HERE"/requirements.txt "$HOST":~/xai-bench/

echo "== 3/5 installing (this is the slow part: torch is ~2 GB) =="
$SSH "$HOST" 'cd ~/xai-bench && \
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv >/dev/null && \
  python3 -m venv .venv && \
  .venv/bin/pip install -q --upgrade pip && \
  .venv/bin/pip install -q -r requirements.txt && \
  .venv/bin/python -c "import shap, lime, captum, xgboost, torch; print(\"imports ok\")"'

echo "== 4/5 running the cost bench =="
# Only the cost scripts: they need no dataset. The CAN experiments would need
# ~4 GB of captures shipped over, and their numbers are architecture-independent
# anyway (F1 and attribution, not milliseconds).
#
# nohup + setsid on the REMOTE side: without it the benchmark takes SIGHUP and
# dies the moment the ssh connection drops, which is exactly what happened on
# the first run -- 40 minutes of install, then the measurement killed by a
# network blip. The run must outlive its own ssh session.
$SSH "$HOST" 'cd ~/xai-bench && mkdir -p results && \
  setsid nohup .venv/bin/python -u xai_cost.py --repeats 40 --kernel-repeats 3 \
      --out results/tabular_arm.json > results/cost_arm.log 2>&1 < /dev/null & \
  echo started'
echo "  waiting for the cost bench..."
until $SSH "$HOST" 'test -f ~/xai-bench/results/tabular_arm.json' 2>/dev/null; do sleep 20; done
$SSH "$HOST" 'cd ~/xai-bench && \
  setsid nohup .venv/bin/python -u validate_model.py > results/validate_arm.log 2>&1 < /dev/null & \
  echo started'
echo "  waiting for the model validation (this one is long)..."
until $SSH "$HOST" 'grep -q "model_validation.json\|Traceback" ~/xai-bench/results/validate_arm.log' 2>/dev/null; do sleep 30; done
$SSH "$HOST" 'cat ~/xai-bench/results/validate_arm.log'

echo "== 5/5 bringing results back =="
mkdir -p "$OUT"
$SCP -q "$HOST":'~/xai-bench/results/*.json' "$OUT"/ || true
echo "-> $OUT"
ls -la "$OUT"

cat <<'NOTE'

NEXT: compare against the x86 baseline.

  .venv/bin/python - <<'PY'
  import json, glob
  arm = json.load(open(sorted(glob.glob('results/arm-*/tabular_arm.json'))[-1]))
  x86 = json.load(open('results/tabular_canonical.json'))
  # the pass counts must match exactly; the milliseconds must not
  PY

What confirms the claim: identical pass counts, different milliseconds, and the
cost model predicting the ARM runtimes from the t_pass measured on ARM.
NOTE
