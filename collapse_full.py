#!/usr/bin/env python3
"""Held-out transfer across attack modality, all 13 ROAD matched pairs.

The 7/7 collapse was first measured with WINDOW labels, under which 50% of the
fabrication positives are benign traffic (V2 of the adversarial review). This
version uses per-frame labels from ROAD's injection mask, which all 13 pairs
supply, and reports both so the difference is visible rather than assumed.

Stratified by CAN ID, because ids are channels with their own physics:
  0xd0    -- the training id  -> transfer across MODALITY only
  others  -- unseen id        -> transfer across modality AND channel
"""
import json

import numpy as np
from sklearn.metrics import f1_score, recall_score

import road_common as rc

TRAIN = ['max_speedometer_attack_1', 'max_speedometer_attack_2',
         'reverse_light_off_attack_1', 'reverse_light_off_attack_2',
         'reverse_light_on_attack_1', 'reverse_light_on_attack_2']


def run(precise):
    """Train on fabrication, score every matched pair in both modalities.

    `precise` selects per-frame labels (from ROAD's injection mask) or the
    window labels most published CAN IDS work uses. Both are run so the paper
    can state what the imprecise label costs rather than depend on it silently.
    """
    tag = 'per-frame' if precise else 'window'
    print(f"[{tag}] training...", flush=True)
    Xtr, ytr = rc.load_many(TRAIN, precise=precise)
    model = rc.make_model().fit(Xtr, ytr)
    train_id = rc.target_id(TRAIN[0])
    rows = []
    pairs = rc.matched_pairs()
    for i, base in enumerate(pairs, 1):
        print(f"[{tag}] {i}/{len(pairs)} {base}", flush=True)
        tid = rc.target_id(base)
        r = {'base': base, 'id': hex(tid) if tid else None,
             'train': base in TRAIN, 'same_id': tid == train_id}
        ok = True
        for kind, name in (('fab', base), ('mas', base + '_masquerade')):
            X, y = rc.load(name, precise=precise)
            if X is None:
                ok = False
                break
            p = model.predict(X)
            r[kind] = float(f1_score(y, p, zero_division=0))
            if kind == 'mas':
                r['rec_mas'] = float(recall_score(y, p, zero_division=0))
        if ok:
            rows.append(r)
    return rows


def summarize(rows, label):
    """Group held-out pairs by whether the CAN ID was seen in training.

    The split matters: on an unseen ID the model does not work under
    fabrication either, so its masquerade collapse says nothing about modality.
    Only the same-ID group isolates the variable.
    """
    held = [r for r in rows if not r['train']]
    same = [r for r in held if r['same_id']]
    new = [r for r in held if not r['same_id']]
    print(f"\n--- {label}: held-out ({len(held)} pairs) ---")
    for grp, name in ((same, 'same CAN ID as training'), (new, 'CAN ID never seen')):
        if not grp:
            continue
        f = [r['fab'] for r in grp]
        m = [r['mas'] for r in grp]
        print(f"  {name:<24} n={len(grp)}  F1 fab median {np.median(f):.3f} "
              f"[{min(f):.3f},{max(f):.3f}]   F1 masq median {np.median(m):.3f} "
              f"[{min(m):.3f},{max(m):.3f}]")
    collapsed = sum(1 for r in held if r['mas'] < 0.05)
    print(f"  collapse (F1 masq < 0.05): {collapsed}/{len(held)} held-out pairs")
    return {'held_out': len(held), 'collapsed': collapsed,
            'median_fab': float(np.median([r['fab'] for r in held])),
            'median_mas': float(np.median([r['mas'] for r in held]))}


def main():
    """Run both labelling modes and write results/collapse_full.json."""
    out = {}
    for precise, label in ((True, 'per-frame labels'), (False, 'window labels')):
        rows = run(precise)
        print(f"\n{'base':<34}{'id':>7}{'F1 fab':>9}{'F1 masq':>10}"
              f"{'rec masq':>10}  ({label})")
        print("-" * 88)
        for r in rows:
            note = ('train' if r['train'] else
                    ('held-out, training id' if r['same_id'] else 'held-out, NEW id'))
            print(f"{r['base']:<34}{r['id']:>7}{r['fab']:>9.3f}{r['mas']:>10.3f}"
                  f"{r['rec_mas']:>10.3f}  {note}")
        key = 'precise' if precise else 'window'
        out[key] = {'rows': rows, 'summary': summarize(rows, label)}

    p = rc.ROOT / 'results' / 'collapse_full.json'
    p.write_text(json.dumps(out, indent=2))
    print(f"\n-> {p.relative_to(rc.ROOT)}")


if __name__ == '__main__':
    main()
