# E1 — AEB perception under camera attack (desktop RTX 3060)

Runbook for a Claude Code session on the author's desktop (Windows, RTX 3060 12 GB).
It turns the illustrative perception rows of the paper's AEB table into measurements.

Context: the OJVT desk reject (OJVT-2026-09-0870) said the AEB component is "illustrative
rather than experimentally evaluated". E1 is the first experiment that answers it.

## Hard rules

1. **Never invent a number.** Every value in `results/` comes from a run you executed in this
   session. If a step fails, record the failure in `NOTES.md`. Do not paper over it.
2. **Never fabricate data or labels.** If the SMIRK data cannot be obtained, stop at step 2
   and report. Do not substitute another dataset without asking the user.
3. **Datasets, weights and videos stay on the desktop.** Only small result files (JSON, CSV,
   PNG figures, Markdown) are committed. The `.gitignore` in this folder enforces that.
4. Ask the user before anything costs money, installs system-wide drivers, or deletes data
   outside this folder.
5. Commit as the user (`Osvaldo Janeri Filho <janeri@gmail.com>`), with no Co-Authored-By
   lines.

## Step 0 — Workspace and GPU check

- Work in `exp/E1/` of this repo (branch `exp/e1-perception`). Put heavy files in
  `exp/E1/work/` (ignored by git). Check free disk first and report it: the plan needs
  about 30–50 GB.
- Python 3.11 or 3.12 virtualenv in `exp/E1/.venv`. Install PyTorch with CUDA from
  https://pytorch.org (pick the CUDA build that matches `nvidia-smi`).
- Verify with code, not with pip's exit code: `torch.cuda.is_available()` is True and
  `torch.cuda.get_device_name(0)` names the RTX 3060. Save `nvidia-smi` output and
  `pip freeze` to `results/env/`.

## Step 1 — SMIRK system

- Clone https://github.com/RI-SE/smirk into `work/smirk`. Read its README and docs to find:
  the pedestrian detector (YOLO-based), the OOD/anomaly "safety cage" (autoencoder), the
  trained weights, and the test data.
- Record in `NOTES.md`: the SMIRK commit hash, where the weights and data came from, and
  their licences (SMIRK itself is GPL-3.0).

## Step 2 — Data and baseline reproduction

- Obtain the SMIRK test data, following SMIRK's own instructions. If it needs a request or
  login, **stop and tell the user what to request**.
- Run the detector on the test set and report precision, recall and false positives per
  image within 80 m. SMIRK's own requirements, used in the paper: accuracy ≥ 93% within 80 m,
  FNR < 7% within 50 m, ≤ 0.1% false positives per image within 80 m.
- Compare with the numbers SMIRK reports. A mismatch is a finding: write it down, don't
  tune it away. Output: `results/e1_baseline.json`.

## Step 3 — Camera adversarial patch (the attack)

- Optimize a printable adversarial patch against the SMIRK detector that suppresses
  pedestrian detection. Use expectation over transformation: random scale, position near
  the pedestrian or roadside, rotation, and brightness.
- Train the patch on the train/val split only; evaluate on the test split.
- Report the attack success rate (pedestrian missed at the operating threshold), the
  confidence drop, and results by distance band (< 50 m, 50–80 m).
- Controls: a random-noise patch and a grey patch of the same size, so the effect is shown
  to come from the optimization rather than occlusion.
- Output: `results/e1_attack.json`, the patch PNG, 3 example frames.

## Step 4 — Explanation evidence under attack

- Saliency for the detector: Grad-CAM, or EigenCAM if Grad-CAM is ill-defined for the
  detection head. Use https://github.com/jacobgil/pytorch-grad-cam and record which method
  you used and why.
- Per frame, clean vs. attacked:
  (a) the share of saliency mass inside the pedestrian box;
  (b) the share inside the patch region;
  (c) the saliency-map IoU between clean and attacked.
- Report the distributions (median and IQR) and whether a threshold on (a) or (b) detects
  the attack. Give AUROC with a bootstrap 95% CI, and the false-alarm rate on clean frames.
- Output: `results/e1_saliency.json`, a per-frame CSV, and a figure.

## Step 5 — OOD / safety cage under attack

- Run SMIRK's autoencoder (or its documented OOD monitor) on clean, attacked, and
  control-patch frames. Report the AUROC of attacked vs. clean, with a bootstrap CI.
- Optional, only if the time is there: MC-dropout or a 3-seed ensemble uncertainty on the
  detector.
- Output: `results/e1_ood.json`.

## Step 6 — Combined monitor

- Combine the saliency signal (Step 4) and the OOD score (Step 5) with a simple OR rule,
  thresholds set on validation data only.
- Report the detection rate and false alarms on test data versus each signal alone. This is
  the first piece of the "integrated" evaluation the editor asked for.
- Output: `results/e1_combined.json`.

## Step 7 — Cost on the RTX 3060

- Reuse the measurement discipline of `xai_cost_vision.py` (adaptive warmup, GC disabled
  in the timed window, count images not calls) but on the real SMIRK detector, on GPU,
  with `torch.cuda.synchronize()` around each timed call.
- Methods: inference only, Grad-CAM/EigenCAM, Integrated Gradients (50 steps), occlusion.
- Run n = 400 for the saliency method and n ≥ 100 for the rest, over 4 runs. Report the
  median and p99 with a bootstrap CI.
- Output: `results/e1_cost_rtx3060.json`. Keep the timing script standalone: the same
  script will later run on AWS (g5g ARM+T4G, c7g) for E6.

## Every JSON result must contain

`git` commit of this repo, SMIRK commit, seeds, GPU name, driver/CUDA/torch versions, the
exact command line, the start/end timestamps, and `n` for every statistic.

## Returning results

1. Update `exp/E1/NOTES.md` with what ran, what failed, the deviations from this plan, and
   the open questions.
2. Write `exp/E1/SUMMARY.md`: one table per step with the measured values only, each with
   the path of the JSON it came from.
3. `git add exp/E1` (check that nothing from `work/` or `.venv/` is staged, and that no file
   is over 5 MB), commit, and `git push origin exp/e1-perception`.
4. If the push fails (no GitHub credentials), zip `exp/E1/results`, `NOTES.md` and
   `SUMMARY.md` into `E1-results.zip`, and tell the user where it is.

Stop after any step whose acceptance check fails, and report. Partial, honest results are
better than complete, doubtful ones.
