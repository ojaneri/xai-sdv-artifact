#!/usr/bin/env python3
"""CAN log parser and feature extractor for the ROAD dataset.

Line format:  (1020000000.000000) can0 0D0#3A4E04644102B200

Features deliberately span both regimes, because the experiment compares
attacks that live in different ones:

  TIMING  -- inter-arrival delta for the SAME id, relative deviation from that
             id's nominal period, id count in the window, bus rate.
             This is where FABRICATION lives: injecting extra frames raises the
             id's frequency and collapses the interval.

  PAYLOAD -- byte entropy, Hamming distance to the previous frame of the same
             id, number of changed bytes, the raw bytes.
             This is where MASQUERADE lives: the legitimate frame is suppressed
             and replaced, so frequency does NOT change and timing shows
             nothing. Only content betrays it.

Without payload features the model could not detect masquerade even in
principle, and the experiment would have nothing to measure.

V1 (adversarial review) changes:
  - parse_log reports how many lines it rejected instead of silently dropping
    them. Swallowing parse errors means a format change yields an empty or
    truncated dataset with no warning.
  - label() validates that the injection interval falls inside the capture,
    which catches a metadata/timebase mismatch instead of producing all-zero
    labels that look like a legitimate result.
  - sliding windows use deque, not list.pop(0), which was O(n) per message.
  - entropy() documents that it is an 8-sample estimator, i.e. heavily biased
    low; it is usable as a relative signal, not as an entropy measurement.
"""
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def parse_log(path, limit=None, want_stats=False):
    """Parse a ROAD .log into [(ts, can_id, payload_bytes)].

    Returns (messages, stats) when want_stats, else messages. `stats` carries
    the rejected-line count so callers can refuse to proceed on a log the
    parser does not understand — silence here previously meant a malformed
    file would produce an empty feature matrix and a meaningless experiment.
    """
    out = []
    total = rejected = 0
    with open(path, 'r', errors='ignore') as f:
        for n, line in enumerate(f):
            if limit and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                ts_end = line.index(')')
                ts = float(line[1:ts_end])
                rest = line[ts_end + 1:].split()
                frame = rest[1]
                cid, data = frame.split('#', 1)
                out.append((ts, int(cid, 16), bytes.fromhex(data) if data else b''))
            except (ValueError, IndexError):
                rejected += 1
    stats = {'total': total, 'rejected': rejected,
             'rejected_frac': rejected / total if total else 1.0}
    return (out, stats) if want_stats else out


def parse_hcrl_csv(path, limit=None, want_stats=False):
    """Parse an HCRL Car-Hacking CSV into [(ts, can_id, payload)], labels.

    Line format:  timestamp,ID,DLC,byte0,...,byte(DLC-1),flag
    so the field count is DLC + 4 and varies per line (DLC is 2, 5 or 8 in
    practice). `flag` is R for a normal frame and T for an injected one.

    Unlike ROAD, HCRL labels EVERY FRAME individually, so no label has to be
    derived from an injection interval or payload mask. That is precisely the
    per-frame ground truth this work had to reconstruct for ROAD, and it makes
    the two datasets directly comparable once the features are computed.

    Returns (messages, labels) or (messages, labels, stats) when want_stats.
    """
    msgs, ys = [], []
    total = rejected = 0
    with open(path, 'r', errors='ignore') as f:
        for n, line in enumerate(f):
            if limit and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split(',')
            try:
                dlc = int(parts[2])
                if len(parts) != dlc + 4:
                    rejected += 1
                    continue
                ts = float(parts[0])
                cid = int(parts[1], 16)
                payload = bytes(int(b, 16) for b in parts[3:3 + dlc])
                flag = parts[-1].strip().upper()
                if flag not in ('R', 'T'):
                    rejected += 1
                    continue
                msgs.append((ts, cid, payload))
                ys.append(1 if flag == 'T' else 0)
            except (ValueError, IndexError):
                rejected += 1
    y = np.array(ys, dtype=np.int8)
    stats = {'total': total, 'rejected': rejected,
             'rejected_frac': rejected / total if total else 1.0}
    return (msgs, y, stats) if want_stats else (msgs, y)


def entropy(b):
    """Shannon entropy of the byte values in one frame.

    NOTE: a CAN frame carries at most 8 bytes, so this is an 8-sample plug-in
    estimator, strongly biased low and capped at 3 bits. It is meaningful only
    as a relative signal between frames of the same id, never as an absolute
    entropy figure.
    """
    if not b:
        return 0.0
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


FEATS = (["delta_same_id", "period_dev", "id_count_win", "bus_rate_win"]
         + ["payload_entropy", "hamming_prev", "bytes_changed", "payload_len"]
         + [f"byte{i}" for i in range(8)])

PERIOD_HISTORY = 50


def extract(msgs, window=0.5):
    """One feature row per message. Causal: only looks backwards.

    Every value is computable online from the frames seen so far, which is the
    condition for the cost figures in the paper to mean anything: a feature
    that needed future frames could not run in the vehicle at all.
    """
    last_ts, last_pl = {}, {}
    deltas = defaultdict(lambda: deque(maxlen=PERIOD_HISTORY))
    win = deque()
    id_win = defaultdict(deque)
    X = np.zeros((len(msgs), len(FEATS)), dtype=np.float32)
    ts_arr = np.zeros(len(msgs), dtype=np.float64)
    for i, (ts, cid, pl) in enumerate(msgs):
        ts_arr[i] = ts
        d = ts - last_ts[cid] if cid in last_ts else 0.0
        hist = deltas[cid]
        # Nominal period is the median of recent deltas for this id. During a
        # long fabrication attack the history itself becomes contaminated and
        # the deviation shrinks; that is a known limitation of a causal
        # estimator, not a bug.
        nominal = float(np.median(hist)) if len(hist) >= 5 else 0.0
        dev = (d - nominal) / nominal if nominal > 1e-9 else 0.0
        while win and ts - win[0] > window:
            win.popleft()
        win.append(ts)
        iw = id_win[cid]
        while iw and ts - iw[0] > window:
            iw.popleft()
        iw.append(ts)
        prev = last_pl.get(cid)
        if prev is not None and len(prev) == len(pl):
            xor = [a ^ b for a, b in zip(prev, pl)]
            ham = sum(bin(v).count('1') for v in xor)
            chg = sum(1 for v in xor if v)
        else:
            ham = chg = 0
        X[i] = ([d, dev, len(iw), len(win) / window,
                 entropy(pl), ham, chg, len(pl)]
                + [pl[k] if k < len(pl) else 0 for k in range(8)])
        last_ts[cid] = ts
        last_pl[cid] = pl
        hist.append(d)
    return X, ts_arr


def injection_mask(injection_data_str):
    """Parse ROAD's `injection_data_str` into fixed (byte_index, value) pairs.

    ROAD publishes the injected payload as a mask, e.g. "XXXXXXXXXXFFXXXX":
    'XX' means the byte varies, a hex pair means the attacker forces that value.
    Returns [] when the capture has no usable mask.
    """
    if not injection_data_str:
        return []
    s = injection_data_str.strip()
    pairs = []
    for b in range(len(s) // 2):
        pair = s[2 * b:2 * b + 2]
        if pair.upper() != 'XX':
            try:
                pairs.append((b, int(pair, 16)))
            except ValueError:
                return []
    return pairs


def label(ts_arr, interval, capture=None):
    """WINDOW labels: every frame inside the injection interval is positive.

    KNOWN IMPRECISION, quantified in V2 of the adversarial review: under
    FABRICATION the attacker adds forged frames while the legitimate ones keep
    flowing, so exactly 50% of the frames inside the window are benign traffic
    labelled as attack. Under MASQUERADE the legitimate frame is suppressed and
    replaced, so the window label is exact (0% benign).

    A model trained on this label learns "this window is under attack", not
    "this frame is malicious" -- which is why id_count_win dominates its
    attribution. Prefer label_precise() where a mask is available; this
    function is kept because most published CAN IDS work labels this way, and
    it is the comparison point.
    """
    if len(ts_arr) == 0:
        raise ValueError(f"{capture or 'capture'}: no messages parsed")
    lo, hi = interval
    if hi <= lo:
        raise ValueError(f"{capture or 'capture'}: empty injection interval {interval}")
    span = ts_arr[-1] - ts_arr[0]
    if lo > span:
        raise ValueError(
            f"{capture or 'capture'}: injection starts at {lo:.1f}s but the "
            f"capture is only {span:.1f}s long -- timebase mismatch")
    y = ((ts_arr - ts_arr[0] >= lo) & (ts_arr - ts_arr[0] <= hi)).astype(np.int8)
    if y.sum() == 0:
        raise ValueError(
            f"{capture or 'capture'}: interval {interval} labelled zero messages")
    return y


def label_precise(msgs, ts_arr, interval, target_id, mask, capture=None):
    """PER-FRAME labels: positive only when the payload matches the mask.

    This is the label the experiment should use where ROAD supplies a mask. It
    separates forged frames from the legitimate traffic that continues during a
    fabrication attack, which the window label cannot do.
    """
    if not mask:
        raise ValueError(f"{capture or 'capture'}: no injection mask available")
    lo, hi = interval
    rel = ts_arr - ts_arr[0]
    inwin = (rel >= lo) & (rel <= hi)
    y = np.zeros(len(msgs), dtype=np.int8)
    for i, (_, cid, pl) in enumerate(msgs):
        if inwin[i] and cid == target_id and all(
                b < len(pl) and pl[b] == v for b, v in mask):
            y[i] = 1
    if y.sum() == 0:
        raise ValueError(
            f"{capture or 'capture'}: mask matched zero frames -- the mask or "
            "the target id is wrong for this capture")
    return y


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    meta = json.load(open(root / 'data/road/attacks/capture_metadata.json'))
    name = sys.argv[1] if len(sys.argv) > 1 else 'max_speedometer_attack_1'
    msgs, stats = parse_log(root / 'data/road/attacks' / f'{name}.log', want_stats=True)
    X, ts = extract(msgs)
    y = label(ts, meta[name]['injection_interval'], capture=name)
    print(f"{name}: {len(msgs)} msgs, {X.shape[1]} features, "
          f"{y.sum()} labelled attack ({y.mean()*100:.1f}%)")
    print(f"duration: {ts[-1]-ts[0]:.1f}s | distinct ids: {len({m[1] for m in msgs})} "
          f"| lines rejected: {stats['rejected']} ({stats['rejected_frac']*100:.3f}%)")
