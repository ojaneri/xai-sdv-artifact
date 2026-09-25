#!/usr/bin/env python3
"""Cross-vehicle transfer: HCRL (Hyundai Sonata) vs ROAD (Ford Expedition).

The modality experiment held the vehicle fixed and varied the attack modality.
This one holds the modality fixed (both datasets supply FABRICATION attacks) and
varies the vehicle. Together they fill a 2x2:

                        same modality        different modality
    same vehicle        F1 = 1.000           F1 = 0.000   (measured earlier)
    different vehicle   this experiment      this experiment

The two datasets share only 3 CAN IDs out of 27 and 106, so nothing that indexes
an ID by value could transfer. Nothing does: `delta_same_id` and `id_count_win`
are relative to whichever ID the frame carries, never to its numeric value.

HCRL labels every frame (R/T). For ROAD the per-frame label is reconstructed
from the injection mask -- without that reconstruction the comparison would put
a window label against a frame label and call the difference generalisation.
"""
import json

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

import can_features as cf
import road_common as rc

HCRL = rc.ROOT / 'data' / 'hcrl'
HCRL_SETS = ['DoS', 'Fuzzy', 'gear', 'RPM']
ROAD_FAB = ['max_speedometer_attack_1', 'max_speedometer_attack_2',
            'reverse_light_off_attack_1', 'reverse_light_off_attack_2',
            'reverse_light_on_attack_1', 'reverse_light_on_attack_2']
ROAD_TEST_FAB = ['max_speedometer_attack_3', 'reverse_light_off_attack_3',
                 'reverse_light_on_attack_3']
ROAD_TEST_MAS = [n + '_masquerade' for n in ROAD_TEST_FAB]
LIMIT = 600_000       # rows per HCRL capture; the full files are ~3.6 M


def load_hcrl(names, limit=LIMIT):
    """Features and per-frame labels for a list of HCRL attack sets."""
    Xs, ys = [], []
    for n in names:
        msgs, y, st = cf.parse_hcrl_csv(HCRL / f'{n}_dataset.csv',
                                        limit=limit, want_stats=True)
        if st['rejected_frac'] > 0.01:
            raise ValueError(f"{n}: {st['rejected_frac']*100:.1f}% unparsed")
        X, _ = cf.extract(msgs)
        Xs.append(X)
        ys.append(y)
        print(f"   HCRL {n:<8} {len(X):>8} rows  {y.mean()*100:5.1f}% attack",
              flush=True)
    return np.vstack(Xs), np.concatenate(ys)


def load_road(names, only_target_id=False):
    """Features and per-frame labels for ROAD captures (whole bus by default)."""
    X, y = rc.load_many(names, only_target_id=only_target_id, verbose=True)
    return X, y


def perf(model, X, y):
    """F1, recall, precision and FPR for one evaluation set."""
    p = model.predict(X)
    return {'f1': float(f1_score(y, p, zero_division=0)),
            'recall': float(recall_score(y, p, zero_division=0)),
            'precision': float(precision_score(y, p, zero_division=0)),
            'fpr': float(((p == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))}


def timing_mass(model):
    """Fraction of gain-based importance the model puts on timing features."""
    imp = model.feature_importances_
    imp = imp / max(imp.sum(), 1e-9)
    return float(imp[rc.TIMING_IDX].sum()), imp


def main():
    out = {}
    print("Loading HCRL (Hyundai Sonata, fabrication only):")
    Xh, yh = load_hcrl(HCRL_SETS)
    print("Loading ROAD training (Ford Expedition, fabrication):")
    Xr, yr = load_road(ROAD_FAB)
    print("Loading ROAD held-out:")
    Xrf, yrf = load_road(ROAD_TEST_FAB)
    Xrm, yrm = load_road(ROAD_TEST_MAS)

    print("\nTraining two detectors...", flush=True)
    mh = rc.make_model().fit(Xh, yh)
    mr = rc.make_model().fit(Xr, yr)

    # held-out split within HCRL so "same vehicle" is not trained-on data
    n = len(Xh)
    idx = np.random.default_rng(rc.SEED).permutation(n)
    hold = idx[: n // 5]
    Xhh, yhh = Xh[hold], yh[hold]

    rows = [
        ('HCRL -> HCRL (held-out)', 'same vehicle, same modality',
         perf(mh, Xhh, yhh)),
        ('HCRL -> ROAD fabrication', 'diff vehicle, same modality',
         perf(mh, Xrf, yrf)),
        ('HCRL -> ROAD masquerade', 'diff vehicle, diff modality',
         perf(mh, Xrm, yrm)),
        ('ROAD -> ROAD fabrication', 'same vehicle, same modality',
         perf(mr, Xrf, yrf)),
        ('ROAD -> HCRL', 'diff vehicle, same modality',
         perf(mr, Xhh, yhh)),
    ]
    print(f"\n{'transfer':<28}{'condition':<30}{'F1':>8}{'recall':>9}{'FPR':>9}")
    print("-" * 86)
    for tag, cond, r in rows:
        print(f"{tag:<28}{cond:<30}{r['f1']:>8.3f}{r['recall']:>9.3f}"
              f"{r['fpr']:>9.4f}")
        out[tag] = {'condition': cond, **r}

    th, imph = timing_mass(mh)
    tr, impr = timing_mass(mr)
    print(f"\ntiming attribution mass -- HCRL model {th*100:.1f}%, "
          f"ROAD model {tr*100:.1f}%")
    top = lambda imp: ", ".join(f"{cf.FEATS[k]}({imp[k]*100:.0f}%)"
                                for k in np.argsort(-imp)[:3])
    print(f"  HCRL model looks at: {top(imph)}")
    print(f"  ROAD model looks at: {top(impr)}")
    out['timing_mass'] = {'hcrl_model': th, 'road_model': tr}

    p = rc.ROOT / 'results' / 'cross_vehicle.json'
    p.write_text(json.dumps(out, indent=2))
    print(f"-> {p.relative_to(rc.ROOT)}")


if __name__ == '__main__':
    main()
