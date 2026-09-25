#!/usr/bin/env python3
"""Structural-blindness score B -- REFUTED, kept as the record of why.

Proposed after seeing 3 pairs: weight each feature's attribution by how much its
observed dispersion collapsed relative to a reference,

    B = sum_j attr_j * (1 - mad_obs_j / mad_ref_j)_+

The intuition: a model betting its weight on features that do not vary here
cannot detect anything, so its silence is not evidence of absence.

It fell in three stages, all reproduced by this script:
  1. On 13 pairs instead of 3 the margin dropped from 6.7x to +0.047.
  2. With the reference taken per-ID from AMBIENT traffic (rather than from
     fabrication captures, which are recorded *under injection* and therefore
     have inflated dispersion) the margin inverted to -0.163.
  3. V5 of the adversarial review re-runs it against the per-frame-label
     attribution, which is payload-dominant rather than timing-dominant. If the
     refutation held only under the old attribution vector, it would have to be
     revisited.

Run it to see the refutation, not to use the score.
"""
import json

import numpy as np

import can_features as cf
import road_common as rc

TRAIN = ['max_speedometer_attack_1', 'max_speedometer_attack_2',
         'reverse_light_off_attack_1', 'reverse_light_off_attack_2',
         'reverse_light_on_attack_1', 'reverse_light_on_attack_2']


def mad(X):
    """Median absolute deviation per column.

    Robust to the outliers the attack itself introduces, which a variance would
    absorb into the reference it is supposed to be measured against.
    """
    return np.median(np.abs(X - np.median(X, axis=0)), axis=0) + 1e-9


def ambient_reference(target_ids, n_captures=3):
    """Per-ID dispersion from ambient traffic -- what a vehicle can self-calibrate."""
    logs = sorted(rc.AMBIENT.glob('*.log'))[:n_captures]
    per_id = {t: [] for t in target_ids}
    for p in logs:
        msgs = cf.parse_log(p)
        X, _ = cf.extract(msgs)
        ids = np.array([m[1] for m in msgs])
        for t in target_ids:
            sel = ids == t
            if sel.sum() >= 50:
                per_id[t].append(X[sel])
    return {t: mad(np.vstack(v)) for t, v in per_id.items() if v}


def blindness(name, attr, ref):
    """Blindness score B for one capture, or None when it is not applicable."""
    tid = rc.target_id(name)
    if tid is None or tid not in ref:
        return None
    X, _ = rc.load(name)
    if X is None:
        return None
    return float((attr * np.clip(1.0 - mad(X) / ref[tid], 0, 1)).sum())


def main():
    """Reproduce the refutation and write results/blindness.json."""
    drift = json.load(open(rc.ROOT / 'results' / 'drift.json'))
    attr = np.array(drift['attr_fab'])
    tmass = float(attr[rc.TIMING_IDX].sum())
    print(f"attribution vector in use: timing {tmass*100:.1f}%, "
          f"payload {(1-tmass)*100:.1f}%")

    pairs = rc.matched_pairs()
    tids = {rc.target_id(b) for b in pairs} - {None}
    print(f"building per-ID ambient reference for {len(tids)} ids...", flush=True)
    ref = ambient_reference(tids)

    rows = []
    print(f"\n{'base':<34}{'id':>7}{'B fab':>9}{'B masq':>9}{'ratio':>8}")
    print("-" * 70)
    for base in pairs:
        a = blindness(base, attr, ref)
        b = blindness(base + '_masquerade', attr, ref)
        if a is None or b is None:
            continue
        rows.append({'base': base, 'id': hex(rc.target_id(base)), 'fab': a,
                     'mas': b, 'train': base in TRAIN})
        print(f"{base:<34}{hex(rc.target_id(base)):>7}{a:>9.3f}{b:>9.3f}"
              f"{b/max(a,1e-9):>7.1f}x{'  <- train' if base in TRAIN else ''}")

    held = [r for r in rows if not r['train']]
    fab = [r['fab'] for r in held]
    mas = [r['mas'] for r in held]
    margin = min(mas) - max(fab)
    print(f"\n--- held-out ({len(held)} pairs) ---")
    print(f"B fab : median {np.median(fab):.3f}  range [{min(fab):.3f},{max(fab):.3f}]")
    print(f"B masq: median {np.median(mas):.3f}  range [{min(mas):.3f},{max(mas):.3f}]")
    print(f"margin = min(masq) - max(fab) = {margin:+.3f}")
    print("REFUTED: no usable separation" if margin <= 0.05
          else f"separation of {margin:+.3f} -- REVISIT the refutation")

    p = rc.ROOT / 'results' / 'blindness.json'
    p.write_text(json.dumps({'attr_timing_mass': tmass, 'rows': rows,
                             'margin': float(margin),
                             'refuted': bool(margin <= 0.05)}, indent=2))
    print(f"-> {p.relative_to(rc.ROOT)}")


if __name__ == '__main__':
    main()
