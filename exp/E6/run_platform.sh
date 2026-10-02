#!/usr/bin/env bash
# E6: measure the cost of the E1/E2 explanation methods on one AWS platform, then terminate.
#
#   exp/E6/run_platform.sh {g5g|c7g|g4dn} [--dry-run]
#
# Runs from the repo root in Git Bash or WSL on the desktop (needs: aws CLI v2 configured,
# ssh, scp, tar, curl). Environment:
#   KEY_NAME    EC2 key-pair name             (required)
#   KEY_FILE    path to its private .pem      (required)
#   WEIGHTS     detector weights from E1      (required; stays out of git)
#   AWS_REGION  default us-east-1
#   MAX_MIN     dead-man switch, minutes until the instance powers itself off (default 180)
#   KEEP=1      do not terminate at the end (debug only; you pay while it runs)
#   AMI_ID      override the AMI lookup
#
# Cost safety, in three layers:
#  1. trap EXIT terminates the instance whatever happens to this script;
#  2. instance-initiated shutdown = terminate, plus `shutdown -h +MAX_MIN` in user-data,
#     so a lost laptop or a killed shell still ends the bill;
#  3. every resource is tagged Project=xai-sdv-e6. List leftovers with:
#       aws ec2 describe-instances --filters Name=tag:Project,Values=xai-sdv-e6 \
#         Name=instance-state-name,Values=pending,running
set -euo pipefail

PLATFORM="${1:-}"
[[ -n "$PLATFORM" ]] || { echo "usage: $0 g5g|c7g|g4dn [--dry-run]" >&2; exit 2; }
DRY=0; [[ "${2:-}" == "--dry-run" ]] && DRY=1
REGION="${AWS_REGION:-us-east-1}"
MAX_MIN="${MAX_MIN:-180}"
: "${KEY_NAME:?set KEY_NAME}" "${KEY_FILE:?set KEY_FILE}" "${WEIGHTS:?set WEIGHTS}"
[[ -f "$KEY_FILE" ]] || { echo "KEY_FILE not found: $KEY_FILE" >&2; exit 1; }
[[ -f "$WEIGHTS" ]]  || { echo "WEIGHTS not found: $WEIGHTS" >&2; exit 1; }
[[ -f exp/E1/cost.py ]] || { echo "exp/E1/cost.py missing: run E1 step 7 first" >&2; exit 1; }

# Platform table. The AMI parameters are AWS-published SSM paths. If a lookup fails, the
# script prints the available paths and stops; it never guesses an AMI.
case "$PLATFORM" in
  g5g)  ITYPE=g5g.xlarge;  DEVICE=cuda; ARCH=arm64
        SSM=/aws/service/deeplearning/ami/arm64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id
        TORCH_INDEX=https://download.pytorch.org/whl/cu128 ;;
  g4dn) ITYPE=g4dn.xlarge; DEVICE=cuda; ARCH=x86_64
        SSM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id
        TORCH_INDEX=https://download.pytorch.org/whl/cu128 ;;
  c7g)  ITYPE=c7g.2xlarge; DEVICE=cpu;  ARCH=arm64
        SSM=/aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id
        TORCH_INDEX=https://download.pytorch.org/whl/cpu ;;
  *) echo "unknown platform: $PLATFORM" >&2; exit 2 ;;
esac

aws() { command aws --region "$REGION" "$@"; }

if [[ -z "${AMI_ID:-}" ]]; then
  AMI_ID=$(aws ssm get-parameter --name "$SSM" --query Parameter.Value --output text 2>/dev/null || true)
  if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    echo "AMI lookup failed for $SSM. Available DLAMI paths:" >&2
    aws ssm get-parameters-by-path --path "/aws/service/deeplearning/ami/$ARCH" \
        --recursive --query 'Parameters[].Name' --output text 2>/dev/null | tr '\t' '\n' | head -40 >&2 || true
    echo "Pick one and rerun with AMI_ID=ami-..." >&2; exit 3
  fi
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="exp/E6/results/$PLATFORM/$STAMP"
echo "platform=$PLATFORM type=$ITYPE ami=$AMI_ID region=$REGION device=$DEVICE out=$OUT"
if (( DRY )); then echo "dry run: nothing launched"; exit 0; fi

# Security group: SSH from this machine's public IP only.
MYIP=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=xai-sdv-e6 \
       --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [[ -z "$SG" || "$SG" == "None" ]]; then
  SG=$(aws ec2 create-security-group --group-name xai-sdv-e6 \
         --description "XAI-SDV E6 cost runs (SSH only)" --query GroupId --output text)
  aws ec2 create-tags --resources "$SG" --tags Key=Project,Value=xai-sdv-e6
fi
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
    --cidr "$MYIP/32" >/dev/null 2>&1 || true   # already present is fine

USERDATA=$(printf '#!/bin/bash\nshutdown -h +%s\n' "$MAX_MIN")
IID=$(aws ec2 run-instances --image-id "$AMI_ID" --instance-type "$ITYPE" \
        --key-name "$KEY_NAME" --security-group-ids "$SG" \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=80,VolumeType=gp3,DeleteOnTermination=true}' \
        --user-data "$USERDATA" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=xai-sdv-e6},{Key=Name,Value=e6-$PLATFORM}]" \
        --query 'Instances[0].InstanceId' --output text)
echo "launched $IID"

cleanup() {
  if [[ "${KEEP:-0}" == "1" ]]; then
    echo "KEEP=1: NOT terminating $IID (it still self-terminates after $MAX_MIN min)"
  else
    echo "terminating $IID"; aws ec2 terminate-instances --instance-ids "$IID" >/dev/null || true
  fi
}
trap cleanup EXIT

aws ec2 wait instance-status-ok --instance-ids "$IID"
HOST=$(aws ec2 describe-instances --instance-ids "$IID" \
         --query 'Reservations[0].Instances[0].PublicDnsName' --output text)
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 "ubuntu@$HOST")
for i in $(seq 1 30); do "${SSH[@]}" true 2>/dev/null && break; sleep 10; done

# Ship code (not data) and the weights.
tar czf /tmp/e6_code.tgz --exclude='work' --exclude='.venv' --exclude='results' \
    exp/E1 exp/E2 xai_cost.py xai_cost_vision.py requirements.txt
scp -i "$KEY_FILE" /tmp/e6_code.tgz "$WEIGHTS" "ubuntu@$HOST:~/"
WNAME=$(basename "$WEIGHTS")

"${SSH[@]}" bash -s -- "$DEVICE" "$TORCH_INDEX" "$WNAME" <<'REMOTE'
set -euo pipefail
DEVICE=$1; TORCH_INDEX=$2; WNAME=$3
mkdir -p run && tar xzf e6_code.tgz -C run && cd run
sudo apt-get -qq update >/dev/null && sudo apt-get -qq install -y python3-venv >/dev/null
python3 -m venv .venv && . .venv/bin/activate
pip -q install --upgrade pip
pip -q install torch torchvision --index-url "$TORCH_INDEX"
[ -f exp/E1/requirements-cost.txt ] && pip -q install -r exp/E1/requirements-cost.txt
pip -q install captum numpy scipy
mkdir -p out/env
{ uname -a; lscpu; } > out/env/platform.txt 2>&1
command -v nvidia-smi >/dev/null && nvidia-smi > out/env/nvidia-smi.txt 2>&1 || true
pip freeze > out/env/pip-freeze.txt
# Verify the device with code, not pip's exit code.
python - "$DEVICE" <<'PY' | tee out/env/device-check.txt
import sys, torch
want = sys.argv[1]
ok = (want == "cpu") or torch.cuda.is_available()
print({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
       "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       "requested": want, "ok": ok})
sys.exit(0 if ok else 4)
PY
python exp/E1/cost.py --device "$DEVICE" --weights "$HOME/$WNAME" --repeats 100 --runs 4 --out out/e1
# Continuity with the published CPU figures (ResNet-18 harness, 1 thread).
python xai_cost_vision.py --out out/legacy_vision.json 2>&1 | tail -5 || echo "legacy harness failed" > out/legacy_FAILED.txt
[ -f exp/E2/cost.py ] && python exp/E2/cost.py --device "$DEVICE" --out out/e2 || true
tar czf ~/e6_out.tgz out
REMOTE

mkdir -p "$OUT"
scp -i "$KEY_FILE" "ubuntu@$HOST:~/e6_out.tgz" "$OUT/"
tar xzf "$OUT/e6_out.tgz" -C "$OUT" && rm "$OUT/e6_out.tgz"
printf '{"platform":"%s","instance_type":"%s","ami":"%s","region":"%s","instance_id":"%s","utc":"%s"}\n' \
  "$PLATFORM" "$ITYPE" "$AMI_ID" "$REGION" "$IID" "$STAMP" > "$OUT/run.json"
echo "results in $OUT"
