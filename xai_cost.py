#!/usr/bin/env python3
"""Cost per explanation, tabular regime (dimensional profile of a CAN IDS).

Measures, per method and per explained sample:
  - median and p99 latency (the p99 is what breaks a real-time budget)
  - number of model inferences consumed per explanation
  - peak memory

The inference count is the portable variable: it does not depend on the machine,
and it is what lets an engineer estimate the cost on another platform without
repeating this bench.

METHODOLOGICAL NOTE: this measures COMPUTATIONAL COST, not explanation quality.
The cost of KernelSHAP/LIME depends on the feature count, the model size and the
background sample size -- not on the semantics of the data. A tabular dataset
with the dimensional profile of CAN traffic (a few dozen numeric features,
imbalanced classes) is therefore sufficient for the cost measurement. DETECTION
validation is a different matter and needs the real datasets (see the ROAD
experiments in this repository).
"""
import json, time, tracemalloc, argparse, gc, statistics as st
from pathlib import Path
import numpy as np

def platform_string():
    """Describe the machine this ran on, detected rather than hardcoded.

    This field used to be the literal string "AMD EPYC, 8 vCPU, n_jobs=1",
    written when only one machine existed. It travelled to an ARM instance and
    produced a Graviton result labelled as x86 -- a metadata lie that would have
    reached the paper if nobody had read the JSON.
    """
    import platform as _p
    import os
    model = ''
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith(('model name', 'Model')):
                    model = line.split(':', 1)[1].strip()
                    break
                if line.startswith('CPU part'):
                    model = 'ARM CPU part ' + line.split(':', 1)[1].strip()
    except OSError:
        pass
    return (f"{_p.machine()} | {model or _p.processor() or 'unknown CPU'} | "
            f"{os.cpu_count()} vCPU | single-threaded measurement")


SEED = 20260823
rng = np.random.default_rng(SEED)


class CountingModel:
    """Wraps the model and counts the inferences each explanation consumes."""

    def __init__(self, fn):
        """Set up counters and register the hook."""
        self.fn = fn
        self.calls = 0      # call count
        self.rows = 0       # rows inferred -- this is what actually costs

    def __call__(self, X):
        """Predict and record how many rows the explainer asked for."""
        X = np.asarray(X)
        self.calls += 1
        self.rows += X.shape[0] if X.ndim > 1 else 1
        return self.fn(X)

    def reset(self):
        """Zero the counters between measurements."""
        self.calls = 0
        self.rows = 0


def timed(fn, repeats, warmup=50, disable_gc=True):
    """Run fn() `repeats` times and return latencies in ms.

    Two lessons from the first round (see results/NOTE-P99.md):
      - a warmup of 3 produced a false p99; 50 stabilises it;
      - CPython's GC produced 145 ms outliers on a method whose median is
        0.75 ms. The tail was the collector, not the algorithm. Measuring with
        the GC disabled isolates the method; the runtime tail is reported apart.
    NEVER call this with tracemalloc active: it distorts the very time measured.
    """
    for _ in range(warmup):
        fn()
    if disable_gc:
        gc.collect(); gc.disable()
    lat = []
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            lat.append((time.perf_counter() - t0) * 1000.0)
    finally:
        if disable_gc:
            gc.enable()
    return lat


def peak_memory(fn):
    """SEPARATE pass for memory -- tracemalloc during the timing pass lies."""
    gc.collect()
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1] // 1024
    tracemalloc.stop()
    return peak


def summarize(name, lat, rows_per_expl, peak_kb, extra=None):
    """Assemble one result row with latency quantiles and pass counts."""
    lat_sorted = sorted(lat)
    p99 = lat_sorted[min(len(lat_sorted) - 1, int(round(0.99 * (len(lat_sorted) - 1))))]
    out = {
        "method": name,
        "n": len(lat),
        "median_ms": round(st.median(lat), 4),
        "p99_ms": round(p99, 4),
        "min_ms": round(min(lat), 4),
        "max_ms": round(max(lat), 4),
        "model_rows_per_explanation": rows_per_expl,
        "peak_kb": peak_kb,
    }
    if extra:
        out.update(extra)
    return out


def make_data(n=6000, n_features=24, n_informative=10):
    """Dimensional profile of a CAN IDS: dozens of numeric features,
    minority attack class (~5%)."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n, n_features=n_features, n_informative=n_informative,
        n_redundant=4, n_classes=2, weights=[0.95, 0.05],
        flip_y=0.01, class_sep=1.2, random_state=SEED)
    return X.astype(np.float32), y


def main():
    """Run the experiment and write results/drift.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--kernel-repeats", type=int, default=5)
    ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--lime-samples", type=int, default=5000)
    ap.add_argument("--out", default="results/tabular.json")
    args = ap.parse_args()

    import xgboost as xgb
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.model_selection import train_test_split

    X, y = make_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=SEED)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method="hist", n_jobs=1, random_state=SEED,
        eval_metric="logloss")
    model.fit(Xtr, ytr)

    def raw_predict(A):
        """Model probability output, uncounted."""
        return model.predict_proba(np.asarray(A, dtype=np.float32))

    counted = CountingModel(raw_predict)
    x1 = Xte[:1]
    results = []

    # ---- baseline: bare inference, no explanation at all ----
    lat = timed(lambda: raw_predict(x1), args.repeats * 5)
    peak = peak_memory(lambda: raw_predict(x1))
    results.append(summarize("inference only (no XAI)", lat, 1, peak))
    base_median = results[0]["median_ms"]

    # ---- TreeSHAP: exact, polynomial, exists only for tree models ----
    tree_expl = shap.TreeExplainer(model)
    lat = timed(lambda: tree_expl.shap_values(x1), args.repeats * 20)
    peak = peak_memory(lambda: tree_expl.shap_values(x1))
    results.append(summarize("SHAP (Tree, exact)", lat, 0, peak,
                             {"note": "exact algorithm; never calls the model"}))

    # ---- KernelSHAP: model-agnostic, approximated by sampling ----
    bg = shap.sample(Xtr, args.background, random_state=SEED)
    kern = shap.KernelExplainer(counted, bg)
    counted.reset()
    kern.shap_values(x1, nsamples="auto", silent=True)
    rows_kernel = counted.rows
    lat = timed(lambda: kern.shap_values(x1, nsamples="auto", silent=True),
                args.kernel_repeats, warmup=2)
    peak = peak_memory(lambda: kern.shap_values(x1, nsamples="auto", silent=True))
    results.append(summarize("SHAP (Kernel, model-agnostic)", lat, rows_kernel, peak,
                             {"background": args.background}))

    # ---- LIME: local linear model over N perturbations ----
    lime_expl = LimeTabularExplainer(
        Xtr, mode="classification", discretize_continuous=True,
        random_state=SEED)
    counted.reset()
    lime_expl.explain_instance(Xte[0], counted, num_features=10,
                               num_samples=args.lime_samples)
    rows_lime = counted.rows
    lat = timed(lambda: lime_expl.explain_instance(
        Xte[0], raw_predict, num_features=10, num_samples=args.lime_samples),
        args.kernel_repeats, warmup=2)
    peak = peak_memory(lambda: lime_expl.explain_instance(
        Xte[0], raw_predict, num_features=10, num_samples=args.lime_samples))
    results.append(summarize("LIME", lat, rows_lime, peak,
                             {"lime_samples": args.lime_samples}))

    for r in results:
        r["x_vs_inference"] = round(r["median_ms"] / base_median, 1)

    meta = {
        "seed": SEED, "n_features": int(X.shape[1]),
        "model": "XGBClassifier(n_estimators=200, max_depth=6)",
        "platform": platform_string(),
        "budget_ms": {"can_ids_detect_respond": 50, "smirk_frame_10fps": 100},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    w = max(len(r["method"]) for r in results) + 2
    print(f"{'method':<{w}}{'median':>10}{'p99':>10}{'x infer':>9}{'inferences':>13}")
    print("-" * (w + 42))
    for r in results:
        print(f"{r['method']:<{w}}{r['median_ms']:>9.2f}ms{r['p99_ms']:>9.2f}ms"
              f"{r['x_vs_inference']:>8}x{r['model_rows_per_explanation']:>13}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
