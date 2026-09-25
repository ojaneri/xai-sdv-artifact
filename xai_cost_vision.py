#!/usr/bin/env python3
"""Cost per explanation, perception regime (the AEB walkthrough of the paper).

Compares the artifacts the framework assigns to the perception layer:

  Grad-CAM             1 forward + a backward that stops at the last conv layer
  Integrated Gradients an N-step path from a baseline to the input: N forwards
                       and N backwards
  Occlusion            slides an occluding window: one forward per position

Lessons from the tabular round, applied here:
  - warmup is proportional to the method's cost, not fixed. A fixed 50 is cheap
    for Grad-CAM (2 passes) and absurd for Occlusion (196 forwards per call).
  - tracemalloc NEVER runs in the pass that measures time; it distorts the time.
  - the GC is disabled inside the timed window: CPython's collector, not the
    method, produced the tail in an earlier measurement.
  - images processed are counted, not module calls. captum stacks the N steps of
    Integrated Gradients into ONE batched call, so counting calls understates
    the cost by a factor equal to the batch -- and destroys the portability that
    is the whole reason for counting.

Random weights are deliberate: cost depends on architecture and input size, not
on weight values. This says nothing about explanation quality.
"""
import time, json, gc, argparse, statistics as st
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from xai_cost import platform_string

SEED = 20260823
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(1)          # an ECU offers one core, not eight


class ForwardCounter:
    """Counts images processed, not module calls.

    FIXED BUG (2026-08-23): the first version counted calls (`self.n += 1`) and
    reported forwards=1 for every method -- including Integrated Gradients with
    50 steps, which took 7.9 s. captum stacks the n_steps into ONE batch, so a
    single call carried 50 images. Counting calls understated the cost by a
    factor equal to the batch size and destroyed the portability of the number,
    which is the whole reason for counting it.

    The right unit is the one the tabular harness uses: rows/images inferred.
    """
    def __init__(self, model):
        """Set up counters and register the hook."""
        self.images = 0     # images processed -- the portable variable
        self.calls = 0      # chamadas do modulo (diagnostico)
        self.h = model.register_forward_hook(self._hit)
    def _hit(self, module, inp, out):
        """Forward hook: accumulate the batch size of each module call."""
        self.calls += 1
        try:
            self.images += inp[0].shape[0] if inp and hasattr(inp[0], "shape") else 1
        except Exception:
            self.images += 1
    def reset(self):
        """Zero the counters between measurements."""
        self.images = 0
        self.calls = 0
    def close(self):
        """Remove the forward hook."""
        self.h.remove()


def quantiles(lat):
    """Median, p95, p99 and max of a latency sample, in ms."""
    s = sorted(lat)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    return {"median_ms": round(st.median(s), 3), "p95_ms": round(q(.95), 3),
            "p99_ms": round(q(.99), 3), "max_ms": round(max(s), 3)}


def adaptive_warmup(fn, budget_ms=400.0, lo=2, hi=50):
    """Warmup proportional to the cost of the method.

    A fixed warmup of 50 is cheap for Grad-CAM (2 passes) and absurd for
    Occlusion (196 forwards per call): that would be 50*196 = 9,800 forwards
    just to warm up. Here one call is measured and at most `budget_ms` is spent
    warming, with a floor of 2 and a ceiling of 50.
    """
    t0 = time.perf_counter()
    fn()
    cost = (time.perf_counter() - t0) * 1000.0
    n = int(budget_ms / max(cost, 1e-6))
    return max(lo, min(hi, n)), cost


def measure(fn, repeats, warmup=None, disable_gc=True):
    """Latencies in ms, warmed up and with the GC held off."""
    if warmup is None:
        warmup, _ = adaptive_warmup(fn)
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


def main():
    """Measure every perception method and write results/vision.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--slow-repeats", type=int, default=5)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--ig-steps", type=int, default=50)
    ap.add_argument("--lime-samples", type=int, default=1000)
    ap.add_argument("--out", default="results/vision.json")
    args = ap.parse_args()

    from torchvision.models import resnet18
    from captum.attr import LayerGradCam, IntegratedGradients, Occlusion

    model = resnet18(weights=None, num_classes=2).eval()   # pedestrian / non-pedestrian
    for p in model.parameters():
        p.requires_grad_(True)
    x = torch.randn(1, 3, args.size, args.size)
    counter = ForwardCounter(model)
    results = []

    def run(name, fn, repeats, note=None):
        """Time one method, record its pass count, and print a line."""
        counter.reset(); fn(); imgs, calls = counter.images, counter.calls
        wu, first_ms = adaptive_warmup(fn)
        lat = measure(fn, repeats, warmup=wu)
        r = {"method": name, "n": len(lat), "images_per_explanation": imgs,
             "module_calls": calls, "warmup": wu, **quantiles(lat)}
        if note:
            r["note"] = note
        results.append(r)
        print(f"  {name:<34}{r['median_ms']:>9.2f}ms  p99={r['p99_ms']:>9.2f}ms"
              f"  images={imgs:>4}")
        return r

    print("perception regime -- ResNet18 (2 classes), 1 thread, "
          f"input {args.size}x{args.size}")
    base = run("inference only (no XAI)", lambda: model(x), args.repeats * 3)

    gc_cam = LayerGradCam(model, model.layer4)
    run("Grad-CAM", lambda: gc_cam.attribute(x, target=0), args.repeats)

    ig = IntegratedGradients(model)
    run(f"Integrated Gradients ({args.ig_steps} steps)",
        lambda: ig.attribute(x, target=0, n_steps=args.ig_steps), args.slow_repeats)

    occ = Occlusion(model)
    run("Occlusion (32x32 stride 16)",
        lambda: occ.attribute(x, target=0, sliding_window_shapes=(3, 32, 32),
                              strides=(3, 16, 16)), args.slow_repeats)

    counter.close()
    med = {r["method"]: r["median_ms"] for r in results}
    b = base["median_ms"]
    for r in results:
        r["x_vs_inference"] = round(r["median_ms"] / b, 1)

    meta = {"seed": SEED, "model": "resnet18 (2 classes, random init)",
            "input": f"1x3x{args.size}x{args.size}", "threads": 1,
            "platform": platform_string(),
            "budget_ms": {"smirk_frame_10fps": 100, "can_ids_detect_respond": 50},
            "method_note": "random weights: this measures cost, not explanation "
                           "quality. Cost depends on the architecture and the input "
                           "size, not on the values of the weights."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print("\nnote: for expensive methods n is deliberately small -- the p99 of "
          "those\nrows is indicative, not a statistic. See the 'n' field of each row.")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
