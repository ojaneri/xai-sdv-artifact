#!/usr/bin/env bash
# Run build/xai_st-iotnode.hex on one FIT IoT-LAB st-iotnode (Saclay) and capture
# its serial console until the firmware's '#' end marker.
#
# Flow adapted from an earlier submission script of ours.
# Differences: the console is retried until the node's serial port is up; the node is reset
# AFTER the console is attached, so the log starts at boot; the experiment is
# stopped as soon as '#' arrives instead of holding the node.
# Auth: ~/.iotlabrc (iotlab-auth) and ~/.ssh/iotlab_ed25519. Nothing secret here.
#
#   ./iotlab_run.sh [duration_min] [name] [node_id]
# The .hex is made from the .bin with --change-addresses 0x08000000, as
# submit_cycles.sh does. Exp 450898 (ELF-derived hex, node 10) failed to deploy;
# whether the hex or the node was at fault was not isolated.
set -euo pipefail
DUR="${1:-40}"
NAME="${2:-xai-st-iotnode-r1}"
NODE_ID="${3:-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN=/opt/iotlab-venv/bin
HEX="${FW:-$HERE/build/xai_st-iotnode.hex}"
OUT="$HERE/results/$NAME"
KEY="$HOME/.ssh/iotlab_ed25519"
IOTLAB_USER="${IOTLAB_USER:?set IOTLAB_USER to your FIT IoT-LAB login}"
mkdir -p "$OUT"

SUB=$("$BIN/iotlab-experiment" submit -n "$NAME" -d "$DUR" \
      -l "saclay,st-iotnode,$NODE_ID,$HEX")
ID=$(echo "$SUB" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "experiment $ID"
"$BIN/iotlab-experiment" wait -i "$ID" --state Running --timeout 1800
NODE="st-iotnode-$NODE_ID"
echo "node $NODE"
DEPLOY=$("$BIN/iotlab-experiment" get -i "$ID" -d)
echo "deployment $DEPLOY"
if ! echo "$DEPLOY" | python3 -c 'import sys,json; sys.exit(0 if "0" in json.load(sys.stdin) else 1)'; then
  echo "firmware deployment failed"; "$BIN/iotlab-experiment" stop -i "$ID" >/dev/null 2>&1 || true; exit 1
fi
cat > "$OUT/experiment.json" <<EOF
{"experiment_id": $ID, "site": "saclay", "node": "$NODE", "archi": "st-iotnode",
 "mcu": "STM32L475VG @ 16 MHz (CLOCK_BENCHMARK)", "duration_min": $DUR,
 "firmware": "$(basename "$HEX")", "firmware_sha256": "$(sha256sum "$HEX" | cut -d' ' -f1)",
 "submitted_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

# console first, then reset, so the capture starts at boot
( timeout $(( DUR * 60 - 60 )) ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
    "$IOTLAB_USER"@saclay.iot-lab.info "for i in \$(seq 40); do nc $NODE 20000 && break; sleep 5; done" \
    > "$OUT/raw_console.log" 2>"$OUT/ssh.err" ) &
CAP=$!
sleep 45
"$BIN/iotlab-node" --reset -i "$ID" > "$OUT/reset.json" 2>&1 || true
while kill -0 $CAP 2>/dev/null; do
  if grep -q '^#' "$OUT/raw_console.log" 2>/dev/null; then kill $CAP 2>/dev/null || true; break; fi
  sleep 20
done
wait $CAP 2>/dev/null || true
"$BIN/iotlab-experiment" stop -i "$ID" > /dev/null 2>&1 || true
sha256sum "$OUT/raw_console.log" > "$OUT/manifest.sha256"
echo "lines: $(wc -l < "$OUT/raw_console.log")  end-marker: $(grep -c '^#' "$OUT/raw_console.log" || true)"
