# XAI-SDV-E — measurement code

Code and lab notebook for the empirical sections of

> **XAI-SDV: Explainable AI for Safety, Security, and Resilience in
> Software-Defined Vehicles** (under anonymous review; authors withheld)

Two questions are measured here, both of which the paper otherwise only asserts:

1. **What does evidence cost?** The framework claims that online and offline
   explanation cannot share one computational budget. That is an engineering
   claim, so we measured it in two regimes and derived a portable cost model.
2. **Does explanation detect what performance metrics miss?** The framework
   proposes an *explanation drift* check. We tested it against a real detector
   failure on real CAN intrusion data — and it does not detect it.

`RESULTS.md` is the lab notebook and the authoritative record. **Read it before
changing any number**: it separates what is established from what was refuted,
including three hypotheses of our own that did not survive.

## Setup

The host is PEP 668 externally-managed; a bare `pip install` fails — silently,
if `--quiet` hides it inside a pipe. Use a virtualenv and verify by importing,
never by pip's exit code.

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import shap, lime, captum, xgboost, torch; print('ok')"
```

## Data

The ROAD dataset is **not redistributed here**. Fetch it from Zenodo
(557 MB compressed, 3.0 GB extracted, CC BY 4.0):

```bash
.venv/bin/python download_road.py          # download, verify, extract
.venv/bin/python download_road.py --check   # verify an existing copy
```

Verification is by byte count against the Zenodo record: a truncated archive
that still extracts would silently produce short captures and a wrong result.

Cite the dataset if you use it — Verma et al., *PLoS ONE* 19(1):e0296879, 2024,
DOI 10.1371/journal.pone.0296879.

The HCRL Car-Hacking dataset (cross-vehicle experiment) is not redistributed
either; `download_hcrl.py` fetches it from its publisher.

## Running

```bash
.venv/bin/python test_smoke.py        # 9 checks, ~1 min — run this first
.venv/bin/python xai_cost.py          # cost, tabular regime
.venv/bin/python xai_cost_vision.py   # cost, perception regime
.venv/bin/python validate_model.py    # cost model: calibrate once, predict
.venv/bin/python drift_experiment.py  # collapse + attribution (main experiment)
.venv/bin/python collapse_full.py     # all 13 matched pairs, both labellings
.venv/bin/python sanity_masq.py       # is the collapse real or a broken pipe?
.venv/bin/python cross_vehicle.py     # HCRL <-> ROAD transfer (needs both datasets)
.venv/bin/python bus_budget.py        # per-frame CAN budgets from benign traffic
```

Everything writes JSON into `results/`. The CAN scripts need the dataset; the
cost scripts do not.

## Claim → script → result

| Paper claim | Script | Output |
|---|---|---|
| Fastest periodic CAN ID every 10 ms; bus gap 0.42 ms (ROAD), 0.51 ms (HCRL), 0.23 ms peak | `bus_budget.py` | `results/bus_budget.json` |
| TreeSHAP 8.2% of one ID's 10 ms budget, 0 model calls; KernelSHAP 209,601 calls, 633 ms | `xai_cost.py` | `results/tabular_canonical.json` |
| Pass counts identical on ARM Neoverse-V1 / N1, latencies move 11–20% | `run_on_arm.sh` | `results/arm-c7g/`, `results/arm/` |
| Inference alone consumes 63% of the perception frame | `xai_cost_vision.py` | `results/vision.json` |
| Grad-CAM p99 sits at 82% of the frame (n=400, bootstrap CI) | `gradcam_boundary.py` | `results/gradcam_boundary.json` |
| `T = N · t_pass(N)` predicts within 4.1% over a 12.5× sweep | `validate_model.py` | `results/model_validation.json` |
| IG has no work of its own (ratio 0.98–1.03) | `ig_overhead.py` | `results/ig_overhead.json` |
| Batch effect is ~10% up to batch 10, not measurable above | `batch_paired.py` | `results/batch_paired.json` |
| Window labels are 50% benign under fabrication | `test_smoke.py` | asserted in the test suite |
| F1 1.000 → 0.000 across modality; L1 drift 0.229 | `drift_experiment.py` | `results/drift.json` |
| Collapse on 6/7 held-out pairs, both labellings compared | `collapse_full.py` | `results/collapse_full.json` |
| The signal exists: F1 0.966 trained on masquerade | `sanity_masq.py` | `results/sanity_masq.json` |
| Blindness score B is refuted (margin −0.644) | `blindness_check.py` | `results/blindness.json` |
| Cross-vehicle: F1 1.000/0.999 within, 0.000 and 0.198 across; FPR 17.9% | `cross_vehicle.py` | `results/cross_vehicle.json` |
| MCU: TreeSHAP 166 ms (nRF52840, 64 MHz), 520 ms (STM32L4, 16 MHz); KernelSHAP 138 s / 318 s | `mcu/` | `mcu/results/*/analysis.json` |
| MCU: Eq. (1) calibrated within KernelSHAP predicts 10× and 100× N to −1.3% / −1.4% | `mcu/analyze.py` | `mcu/results/*/analysis.json` (`eq1.same_method`) |
| STM32L475 and STM32L476 agree within 100 cycles | `mcu/analyze.py` | `mcu/results/xai-st-iotnode-r1`, `mcu/results/xai-nucleo-l476rg` |

## Layout

```
road_common.py        loading, labelling, model, guarded attribution
can_features.py       CAN parser and 16 causal features
download_road.py      fetch + verify the dataset from Zenodo
test_smoke.py         9 checks over the pipeline

xai_cost.py           cost, tabular regime
xai_cost_vision.py    cost, perception regime
validate_model.py     cost-model validation by prediction
ig_overhead.py        does IG do work beyond its passes?
batch_paired.py       batch effect, paired measurement
gradcam_boundary.py   Grad-CAM p99 with a bootstrap CI

drift_experiment.py   the main CAN experiment
collapse_full.py      all 13 matched pairs
sanity_masq.py        three checks that the collapse is real
blindness_check.py    a refuted proposal, kept as the record of why
```

## Microcontrollers (`mcu/`)

The tabular model and its explainers, ported to C and run bare-metal on Cortex-M4.

```
export_model.py   rebuilds the exact model of xai_cost.py -> model.h + ref.json (shap reference)
xai_core.c/.h     inference, exact TreeSHAP, KernelSHAP, LIME-style surrogate, occlusion
host_check.c      host validation against ref.json (gcc -O2 host_check.c xai_core.c -lm)
xai_fw.c          firmware: nRF52840 bare metal, or STM32L4 via the pqm4 libopencm3 HAL (-DXAI_OCM3)
shepherd_run.py   submit/download on the Shepherd Nova testbed (nRF52840, calibrated V/I)
iotlab_run.sh     run on a FIT IoT-LAB st-iotnode (STM32L475); submit the .elf, one serial connection per node
analyze.py        correctness (C lines vs ref.json), cycles/energy per block, Eq. (1) checks
build/            the exact binaries that ran (ELF debug sections stripped; loadable content unchanged)
results/          analysis.json per board, console logs, experiment metadata
```

`model.h` is generated (`export_model.py`, ~390 KB) and not committed. The
nRF52840 HDF5 trace (439 MB) is not redistributed; its SHA-256 is in
`results/xai-nrf52840-r1/experiment.json`. Build flags: `arm-none-eabi-gcc` 13.2.1,
`-O2 -mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16`.

## Two things worth knowing before you trust a number

**Labels.** Most CAN IDS work marks every frame in the injection interval as an
attack. Under fabrication that is 50% benign traffic, measured. A model trained
that way learns *this window is under attack*, not *this frame is forged*. We
label per frame from ROAD's injection mask and report the window label
alongside, because the difference changes which features the model learns.

**Measurement.** `RESULTS.md` section D lists eleven rules, each one born from a
mistake made here — among them: compare only within one run; pair the
measurement when the effect is the size of the noise; never run `tracemalloc`
in the pass that measures time; a ratio only cancels noise when numerator and
denominator share its source.

## Licence

Code: MIT. Documentation: CC BY 4.0. The ROAD dataset is CC BY 4.0 and belongs
to Oak Ridge National Laboratory; it is downloaded, not redistributed.
