#!/usr/bin/env python3
"""Turn an MCU run of xai_fw.c into per-method cost, energy and an Eq. (1) check.

Input is either a Shepherd Nova result directory (HDF5 with calibrated V/I,
GPIO and UART) or a plain serial console log (IoT-LAB, NUCLEO capture.py).

  - `C` lines are checked against ref.json (the `shap` package's margins and
    exact TreeSHAP values): the port is validated on the hardware itself.
  - `B` lines give cycles and model passes per block. On Shepherd the n-th
    gate window (GPIO2 high) is matched to the n-th `B` line, and the match is
    verified by comparing the window's duration with cycles / f_cpu.
  - Eq. (1), T = N * t_pass: t_pass is calibrated once, from `infer_bg`
    (bare inference on background rows, the rows sampling methods perturb
    towards), and used to predict every sampling method with no free
    parameter.

    python analyze.py results/xai-nrf52840-r1 --hz 64e6
    python analyze.py results/xai-st-iotnode-r1/raw_console.log --hz 16e6
"""
import argparse
import json
import re
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B_RE = re.compile(r'B r=(\d+) op=(\S+) reps=(\d+) passes=(\d+) cyc=(\d+)')
C_RE = re.compile(r'C i=(\d+) m=(-?\d+) phi=([-\d,]+)')
N_FEAT = 24


def parse_lines(text):
    blocks, checks, dropped = [], {}, []
    for line in text.splitlines():
        m = B_RE.search(line)
        if m:
            r, op, reps, passes, cyc = m.groups()
            blocks.append(dict(round=int(r), op=op, reps=int(reps),
                               passes=int(passes), cyc=int(cyc)))
            continue
        m = C_RE.search(line)
        if m:
            i, mg, phi = m.groups()
            vals = [v for v in phi.split(',') if v]
            if len(vals) != N_FEAT or len(C_RE.findall(line)) != 1:
                dropped.append(line[:80])   # truncated or merged on the wire
                continue
            checks[int(i)] = (int(mg) / 1e6, [int(v) / 1e6 for v in vals])
    return blocks, checks, dropped


def check_correctness(checks):
    if not checks:
        return {'checked': 0}
    ref = json.load(open(HERE / 'ref.json'))
    em, ep = 0.0, 0.0
    for i, (mg, phi) in checks.items():
        em = max(em, abs(mg - ref['margin'][i]))
        ep = max(ep, float(np.max(np.abs(np.array(phi) - np.array(ref['phi'][i])))))
    # values are printed rounded to 1e-6, so ~1e-6 of the error is printing
    return {'checked': len(checks), 'max_abs_err_margin': em, 'max_abs_err_treeshap': ep,
            'ok': em < 1e-4 and ep < 1e-4}


def shepherd_load(d):
    import h5py
    files = sorted(list(Path(d).rglob('*.h5')) + list(Path(d).rglob('*.hdf5')))
    if not files:
        raise SystemExit(f'no HDF5 under {d}')
    cal = lambda ds: ds[:].astype(np.float64) * ds.attrs.get('gain', 1.0) + ds.attrs.get('offset', 0.0)
    f = h5py.File(files[0], 'r')
    t = cal(f['data/time']); V = cal(f['data/voltage']); I = cal(f['data/current'])
    gt = cal(f['gpio/time']); gv = f['gpio/value'][:]
    msgs = f['uart/message'][:] if 'uart' in f else []
    text = b''.join(m if isinstance(m, bytes) else bytes(m) for m in msgs).decode('ascii', 'replace')
    return files[0], t, V, I, gt, gv, text


def gate_windows(gt, gv, t_start, pin=2, skip_s=1.5):
    bit = ((gv >> pin) & 1).astype(np.int8)
    d = np.diff(bit)
    rise = gt[1:][d == 1]
    fall = gt[1:][d == -1]
    out, j = [], 0
    for r in rise:
        if r - t_start < skip_s:
            continue
        while j < len(fall) and fall[j] <= r:
            j += 1
        if j < len(fall):
            out.append((r, fall[j]))
            j += 1
    return out


def summarize(blocks, hz, energy=None):
    ops = {}
    for k, b in enumerate(blocks):
        o = ops.setdefault(b['op'], {'cyc_per_rep': [], 'passes_per_rep': set(), 'uJ_per_rep': [],
                                     'mW': [], 'dur_err_pct': []})
        o['cyc_per_rep'].append(b['cyc'] / b['reps'])
        o['passes_per_rep'].add(b['passes'] / b['reps'])
        if energy is not None and k < len(energy):
            e, dur = energy[k]
            o['uJ_per_rep'].append(e / b['reps'] * 1e6)
            o['mW'].append(e / dur * 1e3)
            o['dur_err_pct'].append(100 * (dur - b['cyc'] / hz) / (b['cyc'] / hz))
    res = {}
    for op, o in ops.items():
        c = o['cyc_per_rep']
        r = {'n': len(c), 'passes_per_rep': sorted(o['passes_per_rep']),
             'cyc_median': st.median(c), 'cyc_cv_pct': (100 * st.pstdev(c) / st.mean(c)) if len(c) > 1 else 0.0,
             'ms_median': st.median(c) / hz * 1e3}
        if o['uJ_per_rep']:
            r.update(uJ_median=st.median(o['uJ_per_rep']), mW_median=st.median(o['mW']),
                     window_vs_cycles_err_pct_max=max(abs(x) for x in o['dur_err_pct']))
        res[op] = r
    return res


def eq1(res, hz):
    """Predict each sampling method from t_pass = infer_bg cycles, no free parameter."""
    if 'infer_bg' not in res:
        return {}
    tp = res['infer_bg']['cyc_median']
    out = {'t_pass_cyc': tp, 't_pass_us': tp / hz * 1e6}
    for op, r in res.items():
        n = r['passes_per_rep'][0] if r['passes_per_rep'] else 0
        if op.startswith(('kernel', 'lime', 'occlusion')) and n > 0:
            pred = n * tp
            out[op] = {'N': n, 'measured_cyc': r['cyc_median'], 'predicted_cyc': pred,
                       'err_pct': 100 * (r['cyc_median'] - pred) / pred}
    # same-method calibration: t_pass taken from the smallest-N run of KernelSHAP
    # (kernel21), then used to predict the larger N -- the procedure of the
    # paper's ARM validation ("calibrate at one point, predict the others").
    if 'kernel21' in res:
        k = res['kernel21']
        tpk = k['cyc_median'] / k['passes_per_rep'][0]
        out['same_method'] = {'t_pass_cyc': tpk}
        for op in ('kernel210', 'kernel2096'):
            if op in res:
                n = res[op]['passes_per_rep'][0]
                pred = n * tpk
                out['same_method'][op] = {'N': n, 'measured_cyc': res[op]['cyc_median'],
                                          'predicted_cyc': pred,
                                          'err_pct': 100 * (res[op]['cyc_median'] - pred) / pred}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--hz', type=float, required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    src = Path(a.src)
    out = {'source': str(src), 'hz': a.hz}
    energy = None
    if src.is_dir():
        h5, t, V, I, gt, gv, text = shepherd_load(src)
        out['hdf5'] = str(h5)
        blocks, checks, dropped = parse_lines(text)
        wins = gate_windows(gt, gv, t[0])
        P = V * I
        energy = []
        for (a0, a1) in wins:
            m = (t >= a0) & (t < a1)
            energy.append((float(np.trapezoid(P[m], t[m])) if m.sum() > 1 else float('nan'), a1 - a0))
        out['gate_windows'] = len(wins)
        out['mean_current_mA'] = float(I.mean() * 1e3)
    else:
        text = src.read_text(errors='replace')
        blocks, checks, dropped = parse_lines(text)
    out['blocks'] = len(blocks)
    out['correctness'] = check_correctness(checks)
    out['correctness']['dropped_incomplete_lines'] = len(dropped)
    out['ops'] = summarize(blocks, a.hz, energy)
    out['eq1'] = eq1(out['ops'], a.hz)
    dst = Path(a.out) if a.out else (src if src.is_dir() else src.parent) / 'analysis.json'
    dst.write_text(json.dumps(out, indent=1, default=list))
    print(json.dumps(out, indent=1, default=list))


if __name__ == '__main__':
    main()
