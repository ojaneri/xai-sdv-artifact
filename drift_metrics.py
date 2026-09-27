#!/usr/bin/env python3
"""Which form of explanation drift, if any, sees the masquerade collapse?

drift_experiment.py found that the aggregate check -- L1 between normalized
mean-|SHAP| vectors -- barely registers a collapse from F1 = 1.000 to 0.000.
That tests one form of the check. This script tests the others the paper
declared untested, all computed WITHOUT target labels:

  agg_l1      aggregate L1 of normalized mean |SHAP| (the paper's check)
  rank_tau    1 - Kendall tau between feature rankings by mean |SHAP|
  ks_max      max over features of the two-sample KS statistic, per-sample SHAP
  mmd         MMD^2 (RBF, median heuristic) between per-sample SHAP vectors
  c2st_shap   classifier two-sample test on per-sample SHAP vectors (CV AUC)
and, as baselines that use no explanation at all:
  c2st_input  the same classifier test on the raw input features
  ks_score    KS statistic on the model's output probability

Design. Reference = attributions on the TRAINING captures (the validated
regime). Null conditions = the three held-out FABRICATION captures, one at a
time (same modality, new capture), AND the twelve AMBIENT captures (benign
driving, no attack, frames of the same target ID): a monitor must stay quiet on
both, and ambient traffic is what it sees most of the time in service. Target
conditions = the three MASQUERADE captures (the collapse: it must fire here).
A metric SEPARATES the regimes only if every target value exceeds every null
value; the margin (min target - max null) is reported, together with
permutation p-values for ks/mmd against the reference.

    .venv/bin/python drift_metrics.py      # writes results/drift_metrics.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import can_features as cf
import road_common as rc
from drift_experiment import TRAIN, TEST_FAB, TEST_MAS, perf

import sys
REF_MODE = sys.argv[1] if len(sys.argv) > 1 else 'train'   # 'train' or 'ambient'
OUT = Path(__file__).resolve().parent / 'results' / (
    'drift_metrics.json' if REF_MODE == 'train' else 'drift_metrics_ambient_ref.json')
N_SAMPLE = 1500          # per-sample SHAP rows per condition
N_PERM = 200             # permutations for the MMD p-value
AMB_LINES = 300_000      # lines parsed per ambient capture (large logs)
TARGET_ID = 0xd0         # the ID every TRAIN capture attacks
RNG = np.random.default_rng(rc.SEED)


def sample(X, n=N_SAMPLE):
    idx = RNG.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx]


def mmd2(A, B, gamma):
    def k(P, Q):
        d = (P * P).sum(1)[:, None] + (Q * Q).sum(1)[None, :] - 2 * P @ Q.T
        return np.exp(-gamma * np.maximum(d, 0))
    return float(k(A, A).mean() + k(B, B).mean() - 2 * k(A, B).mean())


def mmd_test(A, B):
    Z = np.vstack([A, B])
    sub = Z[RNG.choice(len(Z), size=min(1000, len(Z)), replace=False)]
    d = np.sqrt(np.maximum(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1), 0))
    med = np.median(d[d > 0])
    gamma = 1.0 / (2 * med * med)
    stat = mmd2(A, B, gamma)
    n = len(A)
    null = []
    for _ in range(N_PERM):
        p = RNG.permutation(len(Z))
        null.append(mmd2(Z[p[:n]], Z[p[n:]], gamma))
    return stat, float((np.sum(np.array(null) >= stat) + 1) / (N_PERM + 1))


def c2st(A, B):
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(cross_val_score(clf, X, y, cv=5, scoring='roc_auc').mean())


def metrics(ref_shap, ref_X, ref_p, S, X, p):
    ma, mb = np.abs(ref_shap).mean(0), np.abs(S).mean(0)
    na, nb = ma / ma.sum(), mb / mb.sum()
    tau = kendalltau(na, nb).statistic
    ks = [ks_2samp(ref_shap[:, j], S[:, j]) for j in range(S.shape[1])]
    ks_stat = [k.statistic for k in ks]
    j = int(np.argmax(ks_stat))
    mmd, mmd_p = mmd_test(ref_shap, S)
    return {
        'agg_l1': float(np.abs(na - nb).sum()),
        'top1_same': bool(np.argmax(na) == np.argmax(nb)),
        'rank_tau': float(1 - tau),
        'ks_max': float(ks_stat[j]), 'ks_max_feature': cf.FEATS[j],
        'ks_max_p_bonf': float(min(1.0, ks[j].pvalue * len(ks))),
        'mmd': mmd, 'mmd_p': mmd_p,
        'c2st_shap': c2st(ref_shap, S),
        'c2st_input': c2st(ref_X, X),
        'ks_score': float(ks_2samp(ref_p, p).statistic),
    }


def main():
    import shap
    Xtr, ytr = rc.load_many(TRAIN)
    model = rc.make_model().fit(Xtr, ytr)
    expl = shap.TreeExplainer(model)

    def prep(X):
        Xs = sample(X)
        return Xs, np.asarray(expl.shap_values(Xs)), model.predict_proba(Xs)[:, 1]

    amb_logs = sorted(rc.AMBIENT.glob('*.log'))
    amb_cache = {}

    def ambient_X(log):
        if log not in amb_cache:
            msgs = cf.parse_log(log, limit=AMB_LINES)
            X, _ = cf.extract(msgs)
            ids = np.array([m[1] for m in msgs])
            amb_cache[log] = X[ids == TARGET_ID]
        return amb_cache[log]

    if REF_MODE == 'ambient':
        # benign reference: even-indexed ambient captures; odd ones are the null.
        # A monitor in service is calibrated on benign traffic, not on attacks.
        ref_logs, null_logs = amb_logs[0::2], amb_logs[1::2]
        rX, rS, rp = prep(np.vstack([ambient_X(l) for l in ref_logs]))
    else:
        null_logs = amb_logs
        rX, rS, rp = prep(Xtr)
    rows = []
    for log in null_logs:
        X = ambient_X(log)
        if len(X) < 200:
            print(f"skip {log.stem}: {len(X)} frames of 0x{TARGET_ID:x}")
            continue
        Xs, S, p = prep(X)
        m = metrics(rS, rX, rp, S, Xs, p)
        m.update(condition='ambient', capture=log.stem, f1=None,
                 flagged_rate=float((model.predict(X) == 1).mean()))
        rows.append(m)
        print(f"ambient {log.stem:<41} flag={m['flagged_rate']:.4f} L1={m['agg_l1']:.3f} "
              f"KS={m['ks_max']:.3f} MMD={m['mmd']:.4f} C2ST shap={m['c2st_shap']:.3f} "
              f"input={m['c2st_input']:.3f} KSscore={m['ks_score']:.3f}", flush=True)
    for tag, names in (('null', TEST_FAB), ('target', TEST_MAS)):
        for n in names:
            X, y = rc.load_many([n])
            Xs, S, p = prep(X)
            m = metrics(rS, rX, rp, S, Xs, p)
            m.update(condition=tag, capture=n, f1=perf(model, X, y)['f1'])
            rows.append(m)
            print(f"{tag:6s} {n:<42} F1={m['f1']:.3f}  L1={m['agg_l1']:.3f} "
                  f"1-tau={m['rank_tau']:.3f} KS={m['ks_max']:.3f}({m['ks_max_feature']}) "
                  f"MMD={m['mmd']:.4f}(p={m['mmd_p']:.3f}) "
                  f"C2ST shap={m['c2st_shap']:.3f} input={m['c2st_input']:.3f} "
                  f"KSscore={m['ks_score']:.3f}")

    keys = ['agg_l1', 'rank_tau', 'ks_max', 'mmd', 'c2st_shap', 'c2st_input', 'ks_score']
    summary = {}
    for k in keys:
        nul = [r[k] for r in rows if r['condition'] == 'null']
        amb = [r[k] for r in rows if r['condition'] == 'ambient']
        tgt = [r[k] for r in rows if r['condition'] == 'target']
        allnull = nul + amb
        summary[k] = {'null_fab_max': max(nul), 'null_amb_max': max(amb) if amb else None,
                      'target_min': min(tgt),
                      'margin_fab': min(tgt) - max(nul),
                      'margin_all': min(tgt) - max(allnull),
                      'separates_fab': min(tgt) > max(nul),
                      'separates_all': min(tgt) > max(allnull),
                      'null': nul, 'ambient': amb, 'target': tgt}
    print('\nmetric       fab-null max  ambient max  target min  separates(fab) separates(all)')
    for k, s in summary.items():
        print(f"{k:<12} {s['null_fab_max']:12.4f} {s['null_amb_max']:11.4f} {s['target_min']:11.4f}"
              f"   {s['separates_fab']!s:>12} {s['separates_all']!s:>13}")
    OUT.write_text(json.dumps({'ref_mode': REF_MODE, 'n_sample': N_SAMPLE, 'n_perm': N_PERM,
                               'seed': rc.SEED, 'amb_lines': AMB_LINES,
                               'rows': rows, 'summary': summary}, indent=1))


if __name__ == '__main__':
    main()
