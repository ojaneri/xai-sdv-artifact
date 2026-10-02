# E6 — Cost of the perception explanations across platforms (AWS stand-in for an embedded GPU)

| Platform | Role | Device |
|---|---|---|
| RTX 3060 (desktop, E1 step 7) | workstation / central-compute proxy | cuda |
| `g5g.xlarge` (Graviton2 + NVIDIA T4G) | closest AWS analogue to an ARM+GPU automotive SoC | cuda |
| `c7g.2xlarge` (Graviton3) | ARM CPU without an accelerator (continues the published runs) | cpu |
| `g4dn.xlarge` (x86 + T4) | fallback if CUDA on aarch64 fails on g5g | cuda |

## Before the first run

1. Credits arrived, or you accept paying (about US$ 11 for the whole E6, per the shared
   calculator estimate).
2. Service Quotas → "Running On-Demand G and VT instances" ≥ 4 vCPUs in us-east-1.
3. Billing → a budget alert at US$ 20.
4. Desktop: AWS CLI v2 (`aws configure`), an EC2 key pair (`KEY_NAME`, `KEY_FILE`), and Git
   Bash or WSL.
5. E1 step 7 done, so that `exp/E1/cost.py` and the detector weights exist.

## Run (from the repo root)

```bash
export KEY_NAME=xai-e6 KEY_FILE=~/.ssh/xai-e6.pem WEIGHTS=exp/E1/work/<detector-weights>
exp/E6/run_platform.sh g5g --dry-run   # prints AMI and plan; launches nothing
exp/E6/run_platform.sh g5g
exp/E6/run_platform.sh c7g
# only if g5g reports cuda_available=False:
exp/E6/run_platform.sh g4dn
```

Each run launches one instance, ships the code and weights, installs PyTorch, and checks the
device **in code**. It then runs `exp/E1/cost.py` (and `exp/E2/cost.py` if present), plus
the legacy ResNet-18 harness for continuity, copies the results into
`exp/E6/results/<platform>/<UTC>/`, and **terminates** the instance. The instance also
powers itself off after `MAX_MIN` minutes (default 180), even if your shell dies.

If `device-check.txt` says `"ok": false`, the run stops: the CUDA build did not see the GPU.
Record it in `NOTES.md` and use `g4dn`. Do not report CPU numbers as GPU numbers.

Leftover check:

```bash
aws ec2 describe-instances --filters Name=tag:Project,Values=xai-sdv-e6 \
  Name=instance-state-name,Values=pending,running --query 'Reservations[].Instances[].InstanceId'
```

## Return

Commit `exp/E6/results/` (small JSON and text files only) and push. Same rules as E1:
nothing invented, failures written down.
