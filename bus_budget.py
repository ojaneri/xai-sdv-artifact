#!/usr/bin/env python3
"""Per-frame time budget on the CAN bus, derived from benign traffic.

The cost section compares explanation latency against a 50 ms "detect-and-respond"
budget that the paper does not derive. This script replaces that constant with
budgets measured from the buses the detectors were trained on: ROAD ambient
captures (one vehicle, dynamometer + road) and the HCRL normal run (another
vehicle). Only benign traffic is used, so injected frames do not inflate the rate.

Three budgets per capture, each a different reading of "online":
  bus_mean_ms   1 / mean frame rate: time per frame for ONE core to explain
                every frame on the bus and keep up on average.
  bus_peak_ms   10 ms / max frames in any 10 ms window: the same, at the
                busiest 10 ms of the capture (the queue must not grow there).
  id_fast_ms    median period of the fastest periodic CAN ID: time to explain
                a frame before the next frame of the same ID arrives.

No explanation method is run here; the cost numbers are the medians of
results/tabular_canonical.json, and the comparison is arithmetic.

    python bus_budget.py            # writes results/bus_budget.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
AMBIENT = ROOT / 'data' / 'road' / 'ambient'
HCRL_NORMAL = ROOT / 'data' / 'hcrl' / 'normal_run_data' / 'normal_run_data.txt'
COSTS = ROOT / 'results' / 'tabular_canonical.json'
OUT = ROOT / 'results' / 'bus_budget.json'
WINDOW_S = 0.010
MIN_ID_FRAMES = 20   # an ID seen fewer times has no meaningful period
# An ID is periodic when (p95 - p5) / median of its inter-arrival is below this.
# ROAD 0xFFF fails it (p5 ~1 us, p95 ~17 ms): bursty, not a periodic deadline.
MAX_SPREAD = 1.0


def read_road(path):
    """ROAD candump line: (ts) can0 ID#DATA -> (ts, id)."""
    ts, ids = [], []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            try:
                e = line.index(')')
                t = float(line[1:e])
                cid = line[e + 1:].split()[1].split('#', 1)[0]
                ts.append(t)
                ids.append(int(cid, 16))
            except (ValueError, IndexError):
                continue
    return np.array(ts), np.array(ids)


def read_hcrl_normal(path):
    """HCRL normal run: 'Timestamp: T  ID: XXXX  000  DLC: n ...' -> (ts, id)."""
    ts, ids = [], []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            p = line.split()
            try:
                t = float(p[p.index('Timestamp:') + 1])
                cid = int(p[p.index('ID:') + 1], 16)
            except (ValueError, IndexError):
                continue
            ts.append(t)
            ids.append(cid)
    return np.array(ts), np.array(ids)


def budgets(ts, ids):
    """The three budgets for one capture, in milliseconds."""
    order = np.argsort(ts, kind='stable')
    ts, ids = ts[order], ids[order]
    dur = ts[-1] - ts[0]
    n = len(ts)
    rate = (n - 1) / dur
    bins = np.floor((ts - ts[0]) / WINDOW_S).astype(np.int64)
    peak = np.bincount(bins).max()
    periods = {}
    by_id = defaultdict(list)
    for i, c in enumerate(ids):
        by_id[c].append(i)
    aperiodic = []
    for c, idx in by_id.items():
        if len(idx) < MIN_ID_FRAMES:
            continue
        d = np.diff(ts[idx])
        med = float(np.median(d))
        spread = (np.percentile(d, 95) - np.percentile(d, 5)) / med if med > 0 else np.inf
        if spread < MAX_SPREAD:
            periods[c] = med
        else:
            aperiodic.append(hex(int(c)))
    fastest = min(periods, key=periods.get)
    return {
        'frames': int(n), 'duration_s': round(float(dur), 3),
        'ids': len(by_id), 'ids_periodic': len(periods),
        'rate_fps': round(float(rate), 1),
        'peak_frames_per_10ms': int(peak),
        'bus_mean_ms': round(float(1000 / rate), 4),
        'bus_peak_ms': round(float(1000 * WINDOW_S / peak), 4),
        'id_fast_ms': round(float(1000 * periods[fastest]), 4),
        'id_fast': hex(int(fastest)),
        'ids_aperiodic': aperiodic,
    }


def main():
    caps = {}
    for p in sorted(AMBIENT.glob('*.log')):
        caps['ROAD/' + p.stem] = budgets(*read_road(p))
        print(p.stem, caps['ROAD/' + p.stem])
    caps['HCRL/normal_run'] = budgets(*read_hcrl_normal(HCRL_NORMAL))
    print('HCRL', caps['HCRL/normal_run'])

    costs = json.load(open(COSTS))
    med = {k: v['median_of_medians'] for k, v in costs.items()}
    road = [v for k, v in caps.items() if k.startswith('ROAD/')]
    summary = {}
    for key in ('bus_mean_ms', 'bus_peak_ms', 'id_fast_ms'):
        summary[key] = {
            'ROAD_min': min(r[key] for r in road),
            'ROAD_median': float(np.median([r[key] for r in road])),
            'HCRL': caps['HCRL/normal_run'][key],
        }
    # cost / budget, against the tightest ROAD capture and against HCRL
    ratios = {}
    for m, c in med.items():
        ratios[m] = {f'{key}/{src}': round(c / summary[key][src], 3)
                     for key in summary for src in ('ROAD_min', 'HCRL')}
    out = {'window_s': WINDOW_S, 'min_id_frames': MIN_ID_FRAMES,
           'captures': caps, 'summary': summary,
           'cost_median_ms': med, 'cost_over_budget': ratios}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({'summary': summary, 'cost_over_budget': ratios}, indent=1))


if __name__ == '__main__':
    main()
