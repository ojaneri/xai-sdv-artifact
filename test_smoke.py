#!/usr/bin/env python3
"""Smoke tests for the ROAD pipeline.

Written in V5 of the adversarial review, which flagged that no script had any
check at all: every number in the paper came from code that had never been
verified except by reading its output. These tests are cheap and catch the
failure modes this review actually hit.

Run: .venv/bin/python test_smoke.py
"""
import sys
import traceback

import numpy as np

import can_features as cf
import road_common as rc

CAP = 'max_speedometer_attack_1'
FAILS = []
SKIPPED = []
HAS_DATA = rc.META.available()


def check(name, fn, needs_data=True):
    """Run one test, record the failure instead of aborting the suite.

    Tests that need the ROAD dataset are SKIPPED, not failed, when it is
    absent: a fresh clone has no data yet, and a wall of failures would hide
    whether the installation itself is sound.
    """
    if needs_data and not HAS_DATA:
        SKIPPED.append(name)
        print(f"  SKIP  {name} (dataset absent)")
        return
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        FAILS.append((name, e))
        print(f"  FAIL  {name}: {e}")


def t_parser_reports_rejections():
    """Parser reports rejections."""
    msgs, st = cf.parse_log(rc.ATTACKS / f'{CAP}.log', want_stats=True, limit=5000)
    assert st['total'] > 0, "no lines counted"
    assert st['rejected_frac'] < 0.01, f"{st['rejected_frac']:.2%} rejected"
    assert len(msgs) > 0


def t_features_are_finite():
    """Features are finite."""
    X, _ = rc.load(CAP)
    assert not np.isnan(X).any(), "NaN in features"
    assert not np.isinf(X).any(), "inf in features"


def t_labels_are_not_degenerate():
    """Labels are not degenerate."""
    _, y = rc.load(CAP)
    assert 0 < y.mean() < 1, f"degenerate label distribution: {y.mean()}"


def t_label_rejects_bad_timebase():
    """Label rejects bad timebase."""
    _, _, ts, _ = rc._parsed(CAP)
    try:
        cf.label(ts, (1e9, 2e9), capture='synthetic')
    except ValueError:
        return
    raise AssertionError("label() accepted an interval outside the capture")


def t_precise_label_is_subset_of_window():
    """Every per-frame positive must fall inside the injection window."""
    msgs, _, ts, _ = rc._parsed(CAP)
    iv = rc.META[CAP]['injection_interval']
    yw = cf.label(ts, iv, capture=CAP)
    yp = cf.label_precise(msgs, ts, iv, rc.target_id(CAP),
                          cf.injection_mask(rc.META[CAP]['injection_data_str']))
    assert (yp <= yw).all(), "a per-frame positive lies outside the window"
    assert yp.sum() < yw.sum(), "per-frame labels should be strictly fewer"


def t_fabrication_window_label_is_half_forged():
    """The 50% figure the paper reports must hold, or the claim is stale."""
    msgs, _, ts, ids = rc._parsed('max_speedometer_attack_3')
    iv = rc.META['max_speedometer_attack_3']['injection_interval']
    tid = rc.target_id('max_speedometer_attack_3')
    mask = cf.injection_mask(rc.META['max_speedometer_attack_3']['injection_data_str'])
    yp = cf.label_precise(msgs, ts, iv, tid, mask)
    rel = ts - ts[0]
    inwin = (rel >= iv[0]) & (rel <= iv[1]) & (ids == tid)
    frac = yp[inwin].sum() / inwin.sum()
    assert 0.45 < frac < 0.55, f"forged fraction {frac:.3f}, expected ~0.50"


def t_load_many_refuses_unusable_captures():
    """Load many refuses unusable captures."""
    try:
        rc.load_many(['fuzzing_attack_1'], strict=True)
    except ValueError as e:
        assert 'unusable' in str(e)
        return
    raise AssertionError("load_many silently dropped an unusable capture")


def t_attribution_guard_rejects_zero_mass():
    """Attribution guard rejects zero mass."""
    class Zero:
        def shap_values(self, X):
            """Stub explainer returning zero attribution, for the guard test."""
            return np.zeros_like(X)
    X, _ = rc.load(CAP)
    try:
        rc.normalized_attribution(Zero(), X[:100], n=100)
    except ValueError:
        return
    raise AssertionError("normalized_attribution accepted ~0 mass")


def t_extract_is_causal():
    """Truncating the stream must not change earlier rows."""
    msgs = cf.parse_log(rc.ATTACKS / f'{CAP}.log', limit=4000)
    full, _ = cf.extract(msgs)
    part, _ = cf.extract(msgs[:2000])
    assert np.allclose(full[:2000], part, equal_nan=True), \
        "feature rows depend on future messages"


if __name__ == '__main__':
    print("ROAD pipeline smoke tests")
    if not HAS_DATA:
        print("  (ROAD dataset not found -- data-dependent tests are skipped;\n"
              "   fetch it with: python download_road.py)\n")
    for name, fn in sorted(globals().items()):
        if name.startswith('t_'):
            check(name[2:], fn)
    print(f"\n{len(FAILS)} failure(s), {len(SKIPPED)} skipped")
    sys.exit(1 if FAILS else 0)
