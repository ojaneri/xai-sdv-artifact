#!/usr/bin/env python3
"""Is the masquerade collapse a real transfer failure, or a broken pipeline?

F1 exactly 0.000 with recall 0.000 is the kind of number that is usually a bug.
Three checks before the paper is allowed to state it:

  1. Does the model predict anything at all? (prediction and probability spread)
  2. Are the masquerade features healthy, or degenerate/NaN?
  3. Can a model trained ON masquerade detect it?
     -> YES: the signal exists and the failure is one of transfer -- a real
        result about what the fabrication-trained model learned.
     -> NO: the signal is absent from the features, the result says nothing
        about XAI, and it must not be used.

Check 3 is decisive. V3 of the adversarial review reruns all three under
per-frame labels; the original run used window labels, under which 50% of the
fabrication positives were benign traffic.
"""
import json

import numpy as np
from sklearn.metrics import f1_score, recall_score

import can_features as cf
import road_common as rc

TRAIN = ['max_speedometer_attack_1', 'max_speedometer_attack_2',
         'reverse_light_off_attack_1', 'reverse_light_off_attack_2',
         'reverse_light_on_attack_1', 'reverse_light_on_attack_2']
TEST = ['max_speedometer_attack_3', 'reverse_light_off_attack_3',
        'reverse_light_on_attack_3']


def main():
    """Run the three checks and write results/sanity_masq.json.

    Exits through an exception if the features are unusable; prints an explicit
    DO NOT USE verdict if check 3 shows the signal is absent, because in that
    case the collapse says nothing about explanation and must not reach the
    paper.
    """
    out = {}
    Xtr, ytr = rc.load_many(TRAIN)
    Xm, ym = rc.load_many([n + '_masquerade' for n in TEST])
    model = rc.make_model().fit(Xtr, ytr)
    p = model.predict(Xm)
    pr = model.predict_proba(Xm)[:, 1]

    print("1) WHAT THE FABRICATION-TRAINED MODEL PREDICTS ON MASQUERADE")
    print(f"   flagged as attack: {int(p.sum())} of {len(p)} ({p.mean()*100:.3f}%)")
    print(f"   P(attack): min={pr.min():.4f} median={np.median(pr):.4f} max={pr.max():.4f}")
    print(f"   true attack rate: {int(ym.sum())} ({ym.mean()*100:.1f}%)")
    out['prediction'] = {'flagged': int(p.sum()), 'n': int(len(p)),
                         'prob_max': float(pr.max()),
                         'true_rate': float(ym.mean())}

    print("\n2) ARE THE MASQUERADE FEATURES HEALTHY?")
    nan, inf = int(np.isnan(Xm).sum()), int(np.isinf(Xm).sum())
    const = int((Xm.std(0) == 0).sum())
    print(f"   NaN: {nan}   inf: {inf}   constant columns: {const} of {Xm.shape[1]}")
    print(f"   global range: [{Xm.min():.3f}, {Xm.max():.3f}]")
    out['features'] = {'nan': nan, 'inf': inf, 'constant_cols': const}
    if nan or inf:
        raise ValueError("masquerade features contain NaN/inf — pipeline is broken")

    print("\n3) TRAINING ON MASQUERADE ITSELF (the decisive check)")
    Xmt, ymt = rc.load_many([n + '_masquerade' for n in TRAIN])
    m2 = rc.make_model().fit(Xmt, ymt)
    p2 = m2.predict(Xm)
    f1_2 = float(f1_score(ym, p2, zero_division=0))
    rec2 = float(recall_score(ym, p2, zero_division=0))
    imp = m2.feature_importances_
    imp = imp / max(imp.sum(), 1e-9)
    top = np.argsort(-imp)[:5]
    print(f"   F1={f1_2:.3f}  recall={rec2:.3f}")
    print("   features it uses: " + ", ".join(
        f"{cf.FEATS[k]}({imp[k]*100:.0f}%)" for k in top))
    tmass = float(imp[rc.TIMING_IDX].sum())
    print(f"   mass on TIMING: {tmass*100:.1f}%   on PAYLOAD: {(1-tmass)*100:.1f}%")
    out['masq_trained'] = {'f1': f1_2, 'recall': rec2, 'timing_mass': tmass,
                           'top': [cf.FEATS[k] for k in top]}

    print()
    if f1_2 > 0.5:
        print("=> THE SIGNAL EXISTS in the features. The fabrication-trained model")
        print("   does not see it. Transfer failure: a real result.")
    else:
        print("=> The signal is ABSENT from the features. The result says nothing")
        print("   about XAI, only that these features do not cover masquerade.")
        print("   DO NOT use it in the paper.")
    out['verdict'] = 'transfer_failure' if f1_2 > 0.5 else 'features_inadequate'

    p_out = rc.ROOT / 'results' / 'sanity_masq.json'
    p_out.write_text(json.dumps(out, indent=2))
    print(f"-> {p_out.relative_to(rc.ROOT)}")


if __name__ == '__main__':
    main()
