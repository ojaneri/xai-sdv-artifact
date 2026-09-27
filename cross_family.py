#!/usr/bin/env python3
"""Is the cross-vehicle failure a vehicle effect or an attack-family effect?

cross_vehicle.py trains on all four HCRL sets (DoS, Fuzzy, gear, RPM) and tests on
ROAD's targeted fabrication, so "vehicle" is confounded with "attack family"
(review item CR-03). Here the family is held fixed across vehicles:

  spoofing  HCRL gear + RPM (one ID spoofed at high rate)  <->  ROAD targeted
            fabrication (one ID's frames added, per-frame labels from the mask)
  fuzzing   HCRL Fuzzy (random IDs/payloads)               <->  ROAD fuzzing
            (every ID injected with payload FF..FF inside the injection interval)

Same features, model and seed as the rest of the bench; whole bus; per-frame
labels. Within-vehicle references use a time split of each capture (first 70 %
train, last 30 % test) so they are not trivially in-sample.

ROAD fuzzing has no injection mask in the metadata, only injection_data_str
FF..FF and an interval; a frame is labelled attack if it is inside the interval
and its payload is all 0xFF. The script reports how many all-FF frames occur
OUTSIDE the interval: if that is not ~0, the label is unsafe and the fuzzing
rows should not be used.

    .venv/bin/python cross_family.py        # writes results/cross_family.json
"""
import json
from pathlib import Path

import numpy as np

import can_features as cf
import road_common as rc
from cross_vehicle import load_hcrl, perf, timing_mass, ROAD_FAB, ROAD_TEST_FAB

OUT = Path(__file__).resolve().parent / 'results' / 'cross_family.json'
ROAD_FUZZ = ['fuzzing_attack_1', 'fuzzing_attack_2', 'fuzzing_attack_3']


def time_split(X, y, frac=0.7):
    k = int(len(X) * frac)
    return (X[:k], y[:k]), (X[k:], y[k:])


def load_road_fuzz(names):
    Xs, ys, audit = [], [], []
    for n in names:
        msgs, X, ts, ids = rc._parsed(n)
        lo, hi = rc.META[n]['injection_interval']
        ff = np.array([len(m[2]) > 0 and all(b == 0xFF for b in m[2]) for m in msgs])
        rel = ts - ts[0]                      # interval is relative to capture start
        inside = (rel >= lo) & (rel <= hi)
        y = (ff & inside).astype(np.int8)
        audit.append({'capture': n, 'frames': int(len(y)), 'attack_frac': float(y.mean()),
                      'ff_inside': int((ff & inside).sum()), 'ff_outside': int((ff & ~inside).sum())})
        print(f"   {n:<20} {len(y):>8} rows  {y.mean()*100:5.2f}% attack  "
              f"all-FF outside interval: {audit[-1]['ff_outside']}", flush=True)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys), audit


def fit(X, y):
    return rc.make_model().fit(X, y)


def main():
    out = {'rows': []}

    def row(family, train, test, cond, m, X, y):
        r = perf(m, X, y)
        r.update(family=family, train=train, test=test, condition=cond,
                 prevalence=float(y.mean()), n=int(len(y)))
        out['rows'].append(r)
        print(f"{family:9s} {train:>22} -> {test:<22} {cond:<26} F1={r['f1']:.3f} "
              f"rec={r['recall']:.3f} FPR={r['fpr']:.4f}", flush=True)

    # ---------------- spoofing ----------------
    print("HCRL spoofing (gear, RPM):")
    Xh, yh = load_hcrl(['gear', 'RPM'])
    (Xh_tr, yh_tr), (Xh_te, yh_te) = time_split(Xh, yh)
    print("ROAD targeted fabrication:")
    Xr_tr, yr_tr = rc.load_many(ROAD_FAB, only_target_id=False)
    Xr_te, yr_te = rc.load_many(ROAD_TEST_FAB, only_target_id=False)
    m_h = fit(Xh_tr, yh_tr)
    m_r = fit(Xr_tr, yr_tr)
    row('spoofing', 'HCRL gear+RPM', 'HCRL (time split)', 'same vehicle, same family', m_h, Xh_te, yh_te)
    row('spoofing', 'ROAD targeted', 'ROAD (held-out)', 'same vehicle, same family', m_r, Xr_te, yr_te)
    row('spoofing', 'HCRL gear+RPM', 'ROAD targeted', 'diff vehicle, same family', m_h, Xr_te, yr_te)
    row('spoofing', 'ROAD targeted', 'HCRL gear+RPM', 'diff vehicle, same family', m_r, Xh_te, yh_te)
    out['timing_mass'] = {'hcrl_spoof': timing_mass(m_h)[0], 'road_spoof': timing_mass(m_r)[0]}

    # ---------------- fuzzing ----------------
    print("HCRL fuzzing:")
    Xf, yf = load_hcrl(['Fuzzy'])
    (Xf_tr, yf_tr), (Xf_te, yf_te) = time_split(Xf, yf)
    print("ROAD fuzzing:")
    Xg, yg, audit = load_road_fuzz(ROAD_FUZZ)
    out['road_fuzz_label_audit'] = audit
    ff_out = sum(a['ff_outside'] for a in audit)
    ff_in = sum(a['ff_inside'] for a in audit)
    out['road_fuzz_label_safe'] = bool(ff_out <= 0.01 * max(ff_in, 1))
    Xg_tr, yg_tr, _ = load_road_fuzz(ROAD_FUZZ[:2])
    Xg_te, yg_te, _ = load_road_fuzz(ROAD_FUZZ[2:])
    m_f = fit(Xf_tr, yf_tr)
    m_g = fit(Xg_tr, yg_tr)
    row('fuzzing', 'HCRL Fuzzy', 'HCRL (time split)', 'same vehicle, same family', m_f, Xf_te, yf_te)
    row('fuzzing', 'ROAD fuzzing 1-2', 'ROAD fuzzing 3', 'same vehicle, same family', m_g, Xg_te, yg_te)
    row('fuzzing', 'HCRL Fuzzy', 'ROAD fuzzing', 'diff vehicle, same family', m_f, Xg, yg)
    row('fuzzing', 'ROAD fuzzing 1-2', 'HCRL Fuzzy', 'diff vehicle, same family', m_g, Xf_te, yf_te)
    out['timing_mass'].update(hcrl_fuzz=timing_mass(m_f)[0], road_fuzz=timing_mass(m_g)[0])
    print('timing mass:', {k: round(v, 3) for k, v in out['timing_mass'].items()})
    print('ROAD fuzzing label safe:', out['road_fuzz_label_safe'], f'(FF outside {ff_out}, inside {ff_in})')
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
