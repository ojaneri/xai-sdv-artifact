# Desktop (RTX 3060) — how to run the experiments and return the results

Server: `ssh -p 2222 root@janeri2.janeri.com.br`.

Where the instructions live:
- On the server: `/root/xai-sdv-artifact/exp/DESKTOP.md` (a clone of branch
  `exp/e1-perception`; the runbooks are under `/root/xai-sdv-artifact/exp/`).
- On GitHub: https://github.com/ojaneri/xai-sdv-artifact/blob/exp/e1-perception/exp/DESKTOP.md
- To clone from the server instead of GitHub:
  `git clone -b exp/e1-perception ssh://root@janeri2.janeri.com.br:2222/root/xai-sdv-artifact`
 Results go to `/root/xai-results/`, which is
outside the web root on purpose. Only small files go there: JSON, CSV, logs, plots. No
datasets, weights or videos.

## 0. One-time setup (PowerShell)

```powershell
ssh-keygen -t ed25519                     # if ~/.ssh/id_ed25519 does not exist yet
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh -p 2222 root@janeri2.janeri.com.br "cat >> ~/.ssh/authorized_keys"
ssh -p 2222 root@janeri2.janeri.com.br "echo ok"   # must print ok without a password
git clone -b exp/e1-perception https://github.com/ojaneri/xai-sdv-artifact.git
cd xai-sdv-artifact
```

## 1. Order of work

| Step | What | Instructions | Hardware |
|---|---|---|---|
| E1 | SMIRK perception + patch + saliency/OOD + cost | `exp/E1/RUNBOOK.md` | 3060 |
| E2 | CARLA camera+LiDAR fusion under attack | `exp/E2/RUNBOOK.md` | 3060 |
| E5 | CAN bench, IDS + TreeSHAP under GPU load | `exp/E5/README.md` | 3060 + CANable + Nucleo/ESP32 |
| E6 | Cost on AWS ARM / ARM+GPU | `exp/E6/README.md` | AWS (after credits) |

Each runbook has the same rules: no invented numbers, failures go into `NOTES.md`, and ask
before spending money or installing system software.

## 2. Return the results

Preferred: commit `exp/<E>/results/` and `NOTES.md`, then `git push`. If push is not
possible, or as an extra copy:

```powershell
$TS = Get-Date -Format yyyyMMddTHHmmss
ssh -p 2222 root@janeri2.janeri.com.br "mkdir -p /root/xai-results/E1/$TS"
scp -P 2222 -r exp/E1/results exp/E1/NOTES.md "root@janeri2.janeri.com.br:/root/xai-results/E1/$TS/"
```

Note the flags: `ssh -p 2222` (lowercase) and `scp -P 2222` (uppercase). Replace `E1` with
the experiment you ran. Check the copy with
`ssh -p 2222 root@janeri2.janeri.com.br "ls -R /root/xai-results"`.
