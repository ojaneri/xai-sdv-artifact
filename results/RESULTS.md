# Results — what is established and what is not

Lab notebook for the empirical sections of the XAI-SDV-E paper. It deliberately
separates **measured fact** from **proposed explanation**: over the course of
this work three plausible physical explanations of our own were built on top of
platform noise or on a mislabelled dataset, and survived several rounds because
they were plausible. Section B exists so that the wrong version does not end up
looking as solid as the right one.

Every number below is reproduced by a named script (see `../README.md`).

---

## A. Established

### A1. Cost per explanation — tabular regime

> **Superseded budget (2026-09-25).** The "% of 50 ms budget" column below uses the
> original, uncited 50 ms. The paper now uses budgets derived from benign bus traffic
> (`bus_budget.py` → `results/bus_budget.json`): 10 ms for one CAN ID's deadline
> (TreeSHAP 8.2 %, KernelSHAP 63×) and 0.42 ms mean / 0.23 ms peak per frame for the
> whole bus. Latencies and pass counts below are unchanged.

XGBoost 200×6, 24 features, AMD EPYC, single thread. Medians over four runs.

| method | median | p99 | % of 50 ms budget | model calls |
|---|---|---|---|---|
| inference only | 0.28 ms | 0.46 ms | 0.6% | 1 |
| SHAP (Tree, exact) | 0.82 ms | 0.98 ms | **1.6%** | **0** |
| LIME | 61.6 ms | 71.4 ms | 123% | 5,000 |
| SHAP (Kernel) | 633 ms | 622 ms | **1,265%** | 209,601 |

TreeSHAP consumes **zero** model calls: it reads the tree structure rather than
probing it. The engineering consequence is not that one method is better — it is
that **cheap explanation requires white-box access to model structure.** An
opaque architecture chosen at ECU design time silently commits the programme to
offline-only explanation, and that is a safety decision taken by default.

### A2. Cost per explanation — perception regime
ResNet18, 2 classes, 224×224, single thread.

| method | median | p99 | % of 100 ms frame | images |
|---|---|---|---|---|
| inference only | 63.3 ms | 85.6 ms | 63% | 1 |
| Grad-CAM | 78.5 ms | 81.8 ms | **78%** | 1 |
| Integrated Gradients (50) | 7,934 ms | 8,324 ms | 7,934% | 50 |
| Occlusion (stride 16) | 10,834 ms | 11,193 ms | 10,834% | 170 |

**Bare inference already consumes 63% of the frame.** The budget available for
online explanation is not the frame; it is the residue after inference, and it is
thin. Grad-CAM fits, and consumes most of what is left.

### A3. Cost model, validated by prediction
`T = N × t_pass`, with `t_pass` measured once in isolation and **no free
parameters**. Varying the occlusion stride changes only the count:

| stride | passes | measured | predicted | error |
|---|---|---|---|---|
| 32 | 50 | 3,158 ms | 3,117 ms | 1.3% |
| 16 | 170 | 10,945 ms | 10,599 ms | 3.3% |
| 8 | 626 | 40,626 ms | 39,030 ms | 4.1% |

A 12.5× sweep. The residual is **smaller than the platform's own run-to-run
variation** (16%).

### A4. Integrated Gradients has no work of its own
Paired against the pure cost of the passes it performs, at the same batch:

| n_steps | pure passes | full IG | ratio |
|---|---|---|---|
| 5 | 621 ms | 638 ms | 1.03 |
| 10 | 1,245 ms | 1,263 ms | 0.98 |
| 20 | 2,636 ms | 2,634 ms | 1.00 |
| 50 | 7,607 ms | 7,667 ms | 0.99 |

Path interpolation, the Riemann sum and allocation cost ≈0. IG **is** its passes.

### A5. Per-pass cost grows with batch when gradients are involved

| batch | ms per image (forward + input gradient) |
|---|---|
| 5 | 124.2 |
| 50 | **152.1** |

**+22%.** Cause: back-propagating N copies retains the intermediate activations
of all N simultaneously. In the **forward-only** regime (`no_grad`, no
retention) paired measurement finds ~10% savings up to batch 10 and nothing
measurable above — a different effect with the opposite sign, because there are
no retained activations.

### A6. Final cost model: `T = N × t_pass(N)`
The unit cost is a **function of batch size**, not a constant.
- Occlusion issues N calls at batch 1 ⇒ `t_pass` constant ⇒ error ≤4.1% (A3).
- IG issues one call at batch N ⇒ `t_pass` grows ⇒ correct only if the unit cost
  is measured at the production batch size.

Actionable rule: **calibrate `t_pass` at the batch size used in production.**
Calibrating at batch 1 and extrapolating *underestimates*, which is the
dangerous direction for a safety argument.

### A7. Input gradient costs ~2.1× a forward pass
`autograd.grad(output, input)` with parameters at `requires_grad=False`. Stable
at 1.96–2.10 for batches 1–20. This is a **ratio measured within one window**,
which is why it survives the machine's drift.

Distinction that belongs in the paper: `.backward()` (gradients of 11 M
parameters) and `autograd.grad(out, x)` (input gradient) are different
operations, ~60% apart in cost. When instructing an engineer to measure `t_bwd`
on their hardware, it is necessary to say **which one**.

### A8. Grad-CAM is cheap because it stops the backward pass early
Integrated Gradients back-propagates to the input; Grad-CAM stops at the last
convolutional layer. Hence 1.24× instead of ~2.1×. This is not "being simpler" —
it is a structural, measurable property that generalises to other architectures.

### A9. Label imprecision in the CAN experiment — measured
ROAD publishes the injected payload as a byte mask (e.g. `XXXXXXXXXXFFXXXX`),
which separates forged frames from legitimate ones.

| capture | in window | forged | legitimate | mislabelled |
|---|---|---|---|---|
| max_speedometer_3 | 12,214 | 6,107 | 6,107 | **50.0%** |
| reverse_light_off_3 | 4,868 | 2,434 | 2,434 | **50.0%** |
| reverse_light_on_3 | 4,701 | 2,350 | 2,351 | **50.0%** |
| *(all three masquerade)* | — | 100% | 0 | **0.0%** |

Exactly 50% in all three. Under fabrication the attacker injects one frame per
legitimate one — the rate doubles and half the "positives" are benign traffic.
Under masquerade the legitimate frame is suppressed and replaced, so the window
label is exact.

**A model trained on window labels learns "this window is under attack", not
"this frame is forged".**

### A10. Cross-modality collapse (ROAD, per-frame labels)
Trained on six fabrication captures, evaluated held-out.

| held-out group | n | F1 fabrication | F1 masquerade |
|---|---|---|---|
| same CAN ID as training | 3 | **1.000** | **0.000** |
| CAN ID never seen | 4 | 0.000 | 0.011 |
| *one-rule baseline* | | *0.504* | *0.000* |

Two frames flagged out of 20,895, against a 52.1% true attack rate, median
attack probability 0.0000. Collapse on **6/7** held-out pairs.

The clean comparison is the same-ID group: on the unseen-ID pairs the model does
not work under fabrication either, so their collapse says nothing about
modality. Under window labels the unseen-ID group reports F1 0.999 — that figure
measured the trivial rule, not generalisation.

### A11. The mechanism, measured
The model places **59.0%** of its attribution on `bytes_changed`, and 71.2% on
payload features overall. That feature discriminates only because fabrication
interleaves forged and legitimate frames:

| | label alternation between consecutive frames |
|---|---|
| fabrication | **100.0%** (12,213 of 12,213) |
| masquerade | **0.0%** (0 of 6,106) |

And `bytes_changed` follows: under fabrication it separates forged (median 1)
from legitimate (3); under masquerade both are 2. The learned rule is not wrong
about fabrication — it is a rule about an interleaving that masquerade removes.

### A12. The signal exists — it is a transfer failure
| model | F1 on masquerade | timing mass | payload mass |
|---|---|---|---|
| trained on fabrication | 0.000 | 28.8% | 71.2% |
| trained on masquerade | **0.966** | **0.3%** | **99.7%** |

This rules out the alternative reading, "the features do not cover masquerade".
They do.

### A13. Explanation drift does NOT detect this failure
L1 distance between the attribution distributions: **0.229** (max 2.0), on
vectors of comparable mass (0.97×), so the comparison is between two real
distributions rather than between one distribution and noise.

The check proposed in the paper's OTA module **would pass while the detector is
entirely blind.** A check that grants false assurance is worse than no check.

### A14. A second online/offline boundary: label availability
Attribution diagnoses the mechanism precisely, but the check that tried to
detect it **in service, without labels** is refuted (B3). The reason is
structural: diagnosing this failure requires knowing where the attack lives, and
that requires labels.

TreeSHAP consumes 1.6% of the in-vehicle budget and zero model calls (A1) —
comfortably online by budget — yet *this use of it* cannot run online.

**Cost is necessary but not sufficient** for assigning an artifact to the online
path. The evidence model should record, for each artifact, both its cost class
and its label dependence.

### A15. Cross-vehicle transfer fails too, and asymmetrically
HCRL Car-Hacking (Hyundai Sonata, OBD-II, per-frame R/T labels) against ROAD
(Ford Expedition, dynamometer). Both supply FABRICATION attacks, so modality is
held fixed and the vehicle is the variable. Whole bus, no target-ID filter, so
attack prevalence is the realistic 1.5-2.9% rather than the ~30% the filter
produces.

| transfer | condition | F1 | recall | FPR |
|---|---|---|---|---|
| HCRL to HCRL (held-out) | same vehicle, same modality | 1.000 | 1.000 | 0.0000 |
| ROAD to ROAD | same vehicle, same modality | 0.999 | 1.000 | 0.0000 |
| **HCRL to ROAD** | diff. vehicle, same modality | **0.000** | 0.000 | **0.1792** |
| **ROAD to HCRL** | diff. vehicle, same modality | **0.198** | 0.110 | 0.0001 |
| HCRL to ROAD masquerade | diff. vehicle, diff. modality | 0.003 | 0.012 | 0.1831 |

**The 2x2 now closes:**

| | same modality | different modality |
|---|---|---|
| same vehicle | 0.999-1.000 | **0.000** (A10) |
| different vehicle | **0.000-0.198** | **0.003** |

Neither dimension transfers. The detector works only on the vehicle and the
modality it was trained on.

**The asymmetry is the operationally useful part.** The two directions fail in
opposite ways:

- HCRL to ROAD: **FPR 17.9%** -- the model does not go quiet, it fires on nearly
  a fifth of clean traffic. Against the 0.1% ceiling the Hamid review sets as
  already producing thousands of false alarms per hour in a fleet, that is
  **179x** over. Such an IDS would be switched off in its first week.
- ROAD to HCRL: **FPR 0.0001, recall 0.110** -- silent instead. Almost never
  wrong, almost never right.

F1 conflates the two failures; FPR separates them.

### A16. Attribution explains the cross-vehicle failure
| model | timing mass | dominant features |
|---|---|---|
| trained on HCRL | **51.0%** | `id_count_win` (50%), `bytes_changed` (46%) |
| trained on ROAD | **19.9%** | `bytes_changed` (78%), `id_count_win` (19%) |

Facing the *same attack type* on different vehicles, the two models learn
different strategies. Consistent with what the data shows: HCRL's DoS floods the
bus (`id_count_win` 7.3x, payload entropy to **zero**), so a frame-count rule
works there; ROAD's physically-verified stealthy injection has no such coarse
signature.

The attribution diagnoses the transfer failure **before any test on the target
vehicle** -- which is the paper's thesis, now in a second dimension.

### A17. Feature design is what makes cross-vehicle comparison possible
The two datasets share **3 CAN IDs out of 27 and 106** (11% of the smaller). Any
feature indexing an ID by value would transfer nothing. None does:
`delta_same_id` and `id_count_win` are computed relative to whichever ID the
frame carries, never to its numeric value.

Equally load-bearing: HCRL labels every frame (R/T), while ROAD's per-frame
label had to be reconstructed from the injection mask (A9). Without that
reconstruction this experiment would compare a window label against a frame
label and call the difference generalisation.

### A18. The cost model holds on a second architecture
Re-run on AWS Graviton3 (`c7g.large`, Neoverse-V1, dedicated cores,
single-threaded). Every figure below was produced by the same scripts, with the
platform string **detected** rather than hardcoded.

| method | ARM Neoverse-V1 | ARM Neoverse-N1 | x86 AMD EPYC | passes |
|---|---|---|---|---|
| inference only | 0.25 ms | 0.35 ms | 0.28 ms | **1** |
| SHAP (Tree) | 0.70 ms | 0.98 ms | 0.82 ms | **0** |
| LIME | 49.0 ms | 77.3 ms | 61.6 ms | **5,000** |
| SHAP (Kernel) | 534.7 ms | 846.9 ms | 632.7 ms | **209,601** |

**Latencies span 58% across machines; the pass counts are identical to the
digit on all three.** That is the separation the model asserts, measured across
two ISAs and three microarchitectures.

Model validation on Neoverse-V1, `t_fwd` = 84.72 ms measured in isolation:

| method | configuration | measured | predicted | error |
|---|---|---|---|---|
| Occlusion | stride 32, 50 passes | 4,222 ms | 4,236 ms | **−0.3%** |
| Occlusion | stride 16, 170 passes | 14,796 ms | 14,403 ms | 2.7% |
| Occlusion | stride 8, 626 passes | 53,253 ms | 53,038 ms | **0.4%** |
| Integrated Gradients | n=10 | 1,672 ms | 1,746 ms | −4.3% |
| Integrated Gradients | n=20 | 3,249 ms | 3,492 ms | −7.0% |
| Integrated Gradients | n=50 | 8,306 ms | 8,730 ms | −4.9% |

Median error on the **six non-calibrated** points: **3.5%**, max 7.0%.

### A19. The batch dependence is real but its magnitude is machine-specific
On x86 the Integrated Gradients error reached **+22.7% and +42.5%** at the
larger batches (A4), which is what exposed the batch dependence and forced the
model into its `t_pass(N)` form. On Neoverse-V1 the same points err by at most
**7.0%**, and the sign inverts: the model now slightly **over**estimates.

The Graviton3 memory hierarchy absorbs the cost of retaining N copies'
activations better than the EPYC does. So the batch effect exists on both, but
its size is a property of the machine — one more quantity the engineer must
measure on the target rather than inherit from our table.

Note the safety-relevant asymmetry: on x86 the naive model **under**estimates,
which is the dangerous direction; here it overestimates, which is conservative.
The rule stands either way — calibrate `t_pass` at the production batch size.

### A20. Burstable instances cannot be used to measure latency
The first ARM attempt used a `t4g.medium`, which is burstable. The same `t_fwd`
measurement returned **98 ms, then 281 ms, then 265 ms** on the same machine,
and under sustained load the instance stopped answering SSH entirely. One model
prediction blew out to **63.9%** error, not because the model failed but because
`t_fwd` and the measurement it feeds were taken under different contention.

Two compounding causes, both ours to avoid: CPU-credit exhaustion, and three
concurrent processes that `pgrep -c <pattern>` reported as zero because the
pattern did not match the real command line. Verify by `%cpu` and `loadavg`.

The `xai_cost.py` run from that instance is still usable — it ran single-process
before the contention — and its pass counts are invariant by construction. Its
timings are reported here for completeness but are not the ARM figures of record.

---

## B. Refuted / withdrawn

### B1. "Memory pressure raises the per-image cost" — RETRACTION REVERSED
The full history, because the path is the lesson:

1. Proposed memory pressure (activation retention) as the cause of the IG error.
2. Tested it by measuring **forward-only** cost and found nothing. Withdrew it.
3. Unpaired measurement then gave OPPOSITE trends (+37% and −29%) between runs.
   Concluded the effect was noise.
4. Paired measurement: forward-only shows ~10% savings up to batch 10, nothing
   measurable above.
5. Measured **in the gradient regime**: +22% from batch 5 to 50.
   **The original hypothesis was right** (A5).

The error at step 2: activation retention was tested under `torch.no_grad()`,
which is precisely the instruction *not to retain activations*. The instrument
could not see the effect being looked for, and the null result caused a correct
claim to be withdrawn.

Still invalid from step 3: the "U curve" and the "minimum at batch 10" in the
forward-only regime — those were platform noise measured without pairing.

### B2. "Pass count is the portable variable" — CORRECTED IN A6
Incomplete as stated. What is portable is the pair
**(pass count, batch size)**. The count alone predicts the cost only when the
batch is fixed.

### B3. Structural-blindness score B — REFUTED
Proposed from 3 pairs: `B = Σ attr_j·(1 − mad_obs_j/mad_ref_j)₊`, with an
apparent separation of 0.064 (fabrication) vs 0.823 (masquerade).

It fell in three stages:
1. On 13 pairs instead of 3, the margin dropped to +0.047, and the three
   `correlated_signal` captures (CAN ID 0x6e0, unseen in training) produced
   false positives.
2. With the reference taken per-ID from **ambient** traffic instead of from
   fabrication captures, the margin inverted to −0.163.
3. Re-run against the per-frame-label attribution vector (payload-dominant
   rather than timing-dominant): margin **−0.644**. The refutation holds and is
   stronger.

**Where the separation came from:** the original reference was calibrated on
fabrication captures, recorded *under injection*, so its dispersion was
inflated. Against that swollen baseline, masquerade — which by construction
preserves a normal-looking bus — appeared collapsed. The score was detecting
fabrication and calling it blindness.

Same pattern as B1 and B2: **the number looks significant because the
denominator is wrong.**

### B4. "Transfers across channels at F1 = 0.999" — WITHDRAWN
Measured under window labels, where detecting a doubled frame rate works on any
CAN ID. Under per-frame labels the median is **0.000** on unseen IDs. It was
measuring the trivial rule, not generalisation (A10).

---

## C. Open

1. **n = 3** in the clean same-ID held-out group. Small.
2. ~~Cross-vehicle generalisation~~ — **done** (A15–A17). HCRL Car-Hacking is in
   fact a direct download; the "request required" reading of its page was wrong.
   Two vehicles now, both directions measured.
3. **False-alarm rate of the drift thresholds in service** — not measured.
4. **Automotive silicon** — every figure comes from server-class parts (two
   ISAs, three microarchitectures). Ratios transfer, as A18 shows; a Cortex-R
   with a small cache and no SMP is a different regime and remains untested.
5. **Batch-effect magnitude between sessions** — +39% in one session, +22% in
   another (batch 5→50). Direction always the same, magnitude variable. The
   conservative value is the larger.

---

## D. Measurement rules these rounds produced

1. **Compare only within one run.** `t_fwd` drifted 62 → 73 → 62 ms (16%)
   between runs. Comparisons across windows measure the server.
2. **Pair the measurement** when the effect is the size of the noise: A, B, A,
   and take the ratio within the pair.
3. **Never run `tracemalloc` in the pass that measures time** — it distorts the
   time being measured.
4. **Warmup proportional to cost**, not fixed: 50 warmups of Occlusion is 9,800
   passes spent warming up.
5. **Separate counting from timing.** Counting is deterministic and needs one
   call; together they cost 20 minutes where 20 seconds sufficed.
6. **Count images processed, not module calls.** captum stacks 50 IG steps into
   one call; counting calls understates by a factor equal to the batch.
7. **Do not characterise a scale effect over half a decade of sweep.** One extra
   point inverted the batch conclusion.
8. **The proxy must perform the same operation as the target.** `.backward()`
   instead of `autograd.grad(out, x)` measured 11 M gradients the target never
   asks for.
9. **Testing a hypothesis requires a regime where the effect can appear.**
   Testing activation retention under `no_grad()` guarantees a null result by
   construction.
10. **Pair against the target, not a substitute.** IG was only explained when
    measured side by side with `grad_input` at the same batch.
11. **A ratio only cancels noise when numerator and denominator share its
    source.** If the sources are independent, the ratio *amplifies*. Measured:
    in the tabular regime the absolute varies 8–13% between runs while the ratio
    varies 35–58%. Where the ratio does not hold, report the absolute against
    the budget — which is how an ECU engineer reasons anyway.
