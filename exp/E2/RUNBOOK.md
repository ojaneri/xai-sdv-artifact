# E2 — Camera+LiDAR fusion under single- and dual-modality attack (CARLA, desktop RTX 3060)

Runbook for a Claude Code session on the author's desktop. It replaces the illustrative
"Fusion: Camera 65% / LiDAR 35% → 92% / 8%" row of the paper's AEB table with measurements,
and adds the modality-attribution evidence that the framework claims.

Run it **after E1**. It reuses E1's virtualenv conventions, cost discipline and
result-file format.

## Hard rules

The rules of `exp/E1/RUNBOOK.md` apply unchanged:
- no invented numbers;
- heavy files stay in `exp/E2/work/` (ignored by git);
- ask before paying, installing system-wide or deleting data outside this folder;
- commit as the user, with no Co-Authored-By lines.

One addition: **the LiDAR attacks here are simulated.** Every result file and `SUMMARY.md`
must say so; the paper discusses this as a threat to validity.

## Step 0 — CARLA

- Install a CARLA 0.9.x release (Unreal Engine 4) for Windows from
  https://github.com/carla-simulator/carla/releases. The UE5 0.10 line is heavier on VRAM.
  Record the exact version, and install the matching `carla` Python package in the venv.
- Check: start the server off-screen (`CarlaUE4.exe -RenderOffScreen`), connect from
  Python, and spawn a vehicle in synchronous mode. Disk: about 20 GB for CARLA, plus the
  data in Step 1.
- If CARLA does not run on this machine, stop and report the error. Do not switch simulators
  without asking.

## Step 1 — Scenario generation (the SMIRK case, in CARLA)

- Scenario: ego vehicle at 50 km/h on a rural single-lane road; a pedestrian enters from the
  shoulder, partly occluded at entry. Same assumptions as the paper's Section V: SMIRK
  hands over at TTC < 4 s and braking commits at TTC < 3 s.
- Sensors, synchronous mode, fixed Δt = 0.05 s (20 Hz, above SMIRK's ≥ 10 FPS): one front
  RGB camera, and one LiDAR (32 or 64 channels; record which), with known extrinsics.
- Vary weather, lighting (day, dusk, night), pedestrian blueprint, entry distance and
  occlusion. Include **no-pedestrian runs** for false-alarm statistics.
- Split by scenario seed: train / val / test = 60 / 20 / 20. No frame of a test scenario may
  appear in training. Save the per-frame ground truth: pedestrian box, distance and TTC.
- Target: enough scenarios that the test split has ≥ 100 pedestrian approaches. Record the
  actual count.

## Step 2 — Perception models

- **Camera:** a YOLO pedestrian detector fine-tuned on the train split.
- **LiDAR:** points cropped to the ego corridor, clustered (e.g. DBSCAN), with a small
  classifier (pedestrian vs. not) on per-cluster features (size, point count, height,
  intensity statistics).
- **Fusion:** late fusion that associates camera boxes and LiDAR clusters (projected with the
  extrinsics), then a gradient-boosted classifier deciding "pedestrian in path" from
  **named** camera features (confidence, box size and position) and **named** LiDAR features
  (cluster size, points, range, height). Keeping the features named and tree-based keeps the
  modality attribution exact (TreeSHAP), as in the paper's CAN experiments.
- Report the per-modality and fused precision/recall on clean test data, by distance band.

## Step 3 — Attacks

| ID | Attack | How it is simulated |
|---|---|---|
| A-cam | Camera adversarial patch | Re-optimize the E1 method against the Step-2 camera detector (EOT); place it in the scene as a texture or composite it onto the frames. Record which. |
| A-lid-rm | LiDAR point removal (hiding) | Drop a fraction of the points on the pedestrian (sweep 25/50/75/100 %). |
| A-lid-inj | LiDAR spoofed obstacle | Inject a pedestrian-shaped point cluster in the corridor with no camera counterpart (false-brake attack). |
| A-both | Joint camera + LiDAR | A-cam together with A-lid-rm on the same frames (the multi-sensor-fusion attack class of the paper's ref. Cao 2021). |

Apply the attacks only to test scenarios. Keep the clean copy of every frame for paired
comparison.

## Step 4 — Evidence under attack

Per frame, clean vs. attacked:
- (a) **Modality attribution:** the share of |SHAP| mass on camera features vs. LiDAR
  features in the fusion classifier.
- (b) **Cross-modal disagreement:** the camera says pedestrian XOR LiDAR says pedestrian.
- (c) The **OOD score** of the E1 monitor, applied to the camera frames.

Report, per attack and as a bootstrap 95% CI:
- the fused-decision failure rate (missed pedestrian, or false brake for A-lid-inj);
- the AUROC of (a), (b) and (c) for separating attacked from clean frames, and the
  false-alarm rate on no-pedestrian runs.

## Step 5 — Decision-window test (feeds E5)

For each test approach, find the frames whose TTC lies between 4 s and 3 s (the decision
window). Report:
- the fraction of attacked approaches in which the combined monitor ((a) OR (b) OR (c),
  thresholds set on val only) flags **inside** the window;
- the time from window start to the first flag;
- the fraction of clean approaches with a flag (false fallback).

## Step 6 — Cost

Time the fusion classifier, TreeSHAP on it, and the clustering, on the RTX 3060 and the CPU.
Use the E1 cost discipline, with n ≥ 400 for TreeSHAP over 4 runs. The script must accept
`--device` so that E6 can rerun it on AWS.

## Outputs

`results/e2_dataset.json` (counts, splits, versions), `e2_models.json`, `e2_attacks.json`,
`e2_evidence.json`, `e2_window.json`, `e2_cost.json`, plus per-frame CSVs (< 5 MB each;
split or summarize if larger) and 3–5 example figures.

Every JSON records the git commit, CARLA version, seeds, GPU/driver/torch versions, the
command line, timestamps, and `n` per statistic.

Return the results the same way as E1: `NOTES.md`, `SUMMARY.md`, then commit and push.
If the push fails, make `E2-results.zip`.
