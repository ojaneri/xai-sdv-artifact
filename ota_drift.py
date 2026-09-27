#!/usr/bin/env python3
"""OTA explanation drift between MODEL VERSIONS (review item CR-04).

Section VII measured drift of one model across two input regimes. The OTA check of
the framework compares two model versions on the same validation inputs. This
script runs that check on three versions of the CAN detector:

  v1   certified: fabrication captures, per-frame labels (the paper's detector)
  v2a  "coverage update": fabrication + masquerade captures, per-frame labels
  v1_retrain_a/b  retraining floor: v1's recipe on 90 % resamples (two seeds);
       the drift an unchanged recipe produces, i.e. the scale for the others
  v2b  "shortcut update": fabrication captures, WINDOW labels -- the imprecise
       label most published work uses, which teaches "this window is under
       attack" and pushes the model onto frame counts

Validation inputs = the held-out fabrication captures (the certified regime).
For each update: performance drift (F1 under per-frame labels, and under window
labels as a team using window labels would measure it), plus explanation drift
between versions on the same inputs: aggregate L1 of normalized mean |SHAP|,
1 - Kendall tau, and whether the dominant feature changes. Masquerade F1 is
reported to show what each update does to the blind spot.

    .venv/bin/python ota_drift.py        # writes results/ota_drift.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import can_features as cf
import road_common as rc
from drift_experiment import TRAIN, TEST_FAB, TEST_MAS, perf

OUT = Path(__file__).resolve().parent / 'results' / 'ota_drift.json'
TRAIN_MAS = [n + '_masquerade' for n in TRAIN]


def mean_abs_shap(model, X, n=3000):
    import shap
    idx = np.random.default_rng(rc.SEED).choice(len(X), size=min(n, len(X)), replace=False)
    sv = np.abs(shap.TreeExplainer(model).shap_values(X[idx])).mean(0)
    return sv / sv.sum()


def main():
    Xf, yf = rc.load_many(TRAIN)
    Xm, ym = rc.load_many(TRAIN_MAS)
    Xw, yw = rc.load_many(TRAIN, precise=False)
    Xv, yv = rc.load_many(TEST_FAB)                  # validation, per-frame labels
    _, yv_win = rc.load_many(TEST_FAB, precise=False)  # same frames, window labels
    Xq, yq = rc.load_many(TEST_MAS)
    assert len(yv) == len(yv_win)

    def resample(seed):
        idx = np.random.default_rng(seed).choice(len(Xf), size=int(0.9 * len(Xf)), replace=False)
        return rc.make_model().fit(Xf[idx], yf[idx])
    models = {'v1': rc.make_model().fit(Xf, yf),
              # retraining floor: same data and labels, 90 % resamples, two seeds
              'v1_retrain_a': resample(rc.SEED + 1),
              'v1_retrain_b': resample(rc.SEED + 2),
              'v2a': rc.make_model().fit(np.vstack([Xf, Xm]), np.r_[yf, ym]),
              'v2b': rc.make_model().fit(Xw, yw)}
    att = {k: mean_abs_shap(m, Xv) for k, m in models.items()}
    out = {'versions': {}}
    for k, m in models.items():
        a = att[k]
        top = [cf.FEATS[j] for j in np.argsort(-a)[:3]]
        r = {'f1_val_precise': perf(m, Xv, yv)['f1'],
             'f1_val_window': perf(m, Xv, yv_win)['f1'],
             'f1_masquerade': perf(m, Xq, yq)['f1'],
             'top3': top, 'top3_share': [float(a[cf.FEATS.index(t)]) for t in top],
             'timing_share': float(a[rc.TIMING_IDX].sum())}
        if k != 'v1':
            r['expl_drift_l1_vs_v1'] = float(np.abs(a - att['v1']).sum())
            r['rank_dist_vs_v1'] = float(1 - kendalltau(a, att['v1']).statistic)
            r['top1_changed'] = bool(np.argmax(a) != np.argmax(att['v1']))
            for lab in ('precise', 'window'):
                key = f'f1_val_{lab}'
                r[f'perf_drift_{lab}'] = r[key] - out['versions']['v1'][key]
        out['versions'][k] = r
        print(k, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}))
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
