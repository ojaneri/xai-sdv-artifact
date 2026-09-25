"""Cost-model validation: calibrate on ONE point, PREDICT the rest.

In the previous round the coefficients were fitted to the same data that was
then used to report the error. That is fitting, not modelling. The rule here:

  1. measure t_fwd (one pass) and calibrate k_bwd on the CHEAPEST point of each
     family;
  2. with those two numbers FROZEN, predict the remaining points;
  3. report the error. A model that only lands where it was calibrated is not a
     model.

What is varied is whatever changes the pass count without changing the method:
  Integrated Gradients -> n_steps  (5, 10, 20, 50)
  Occlusion            -> stride   (32, 16, 8)  => 49, 169, 625 positions
                          (captum requires stride <= window)
"""
import time, json, gc, statistics as st
from pathlib import Path
import torch
torch.manual_seed(20260823); torch.set_num_threads(1)
from torchvision.models import resnet18
from captum.attr import IntegratedGradients, Occlusion

class Counter:
    """Counts IMAGES processed by the model, not module calls.

    captum stacks the n_steps of Integrated Gradients into one batched call, so
    counting calls understates the cost by a factor equal to the batch -- and
    destroys the portability that is the whole reason for counting.
    """

    def __init__(self, model):
        """Register the forward hook on the model."""
        self.images = 0
        model.register_forward_hook(self._hit)

    def _hit(self, module, inp, out):
        """Forward hook: accumulate the batch size of each module call."""
        self.images += inp[0].shape[0] if inp and hasattr(inp[0], 'shape') else 1

    def reset(self):
        """Zero the image counter between measurements."""
        self.images = 0


def timeit(fn, reps=3):
    """Median latency in ms over `reps` calls, GC disabled inside the window."""
    for _ in range(2): fn()
    gc.collect(); gc.disable()
    try:
        lat=[]
        for _ in range(reps):
            t=time.perf_counter(); fn(); lat.append((time.perf_counter()-t)*1000)
    finally: gc.enable()
    return st.median(lat)

m=resnet18(weights=None,num_classes=2).eval()
for p in m.parameters(): p.requires_grad_(True)
x=torch.randn(1,3,224,224); c=Counter(m)

t_fwd = timeit(lambda: m(x), 30)
print(f"t_fwd calibrated = {t_fwd:.2f} ms  (1 pass, ResNet18/224/1 thread)\n")

def probe(fn):
    """(images processed, median latency) for one attribution call."""
    c.reset(); fn(); return c.images, timeit(fn)

# ---------- Occlusion family: forward only, ZERO free coefficients ----------
occ=Occlusion(m); rows=[]
print("OCCLUSION — model: T = N_img * t_fwd   (no fitted parameter)")
print(f"{'stride':>7}{'images':>9}{'measured':>11}{'predicted':>11}{'error':>8}")
for s in (32,16,8):
    n,ms = probe(lambda s=s: occ.attribute(x,target=0,
                 sliding_window_shapes=(3,32,32),strides=(3,s,s)))
    pred=n*t_fwd; err=(ms-pred)/pred*100
    rows.append({"family":"occlusion","param":s,"images":n,"measured_ms":round(ms,1),
                 "predicted_ms":round(pred,1),"error_pct":round(err,1)})
    print(f"{s:>7}{n:>9}{ms:>10.0f}ms{pred:>10.0f}ms{err:>7.1f}%")

# ---------- familia IG: calibra k_bwd no ponto MAIS BARATO, preve o resto ----
ig=IntegratedGradients(m)
print("\nINTEGRATED GRADIENTS — model: T = N_img * t_fwd * k")
n0,ms0 = probe(lambda: ig.attribute(x,target=0,n_steps=5))
k = ms0/(n0*t_fwd)
print(f"k calibrated ONLY on n_steps=5:  k = {k:.3f}\n")
print(f"{'n_steps':>8}{'images':>9}{'measured':>11}{'predicted':>11}{'error':>8}")
for ns in (5,10,20,50):
    n,ms = probe(lambda ns=ns: ig.attribute(x,target=0,n_steps=ns))
    pred=n*t_fwd*k; err=(ms-pred)/pred*100
    tag=" (calibration)" if ns==5 else ""
    rows.append({"family":"integrated_gradients","param":ns,"images":n,
                 "measured_ms":round(ms,1),"predicted_ms":round(pred,1),
                 "error_pct":round(err,1),"calibration":ns==5})
    print(f"{ns:>8}{n:>9}{ms:>10.0f}ms{pred:>10.0f}ms{err:>7.1f}%{tag}")

pred_only=[r for r in rows if not r.get("calibration")]
errs=[abs(r["error_pct"]) for r in pred_only]
print(f"\nabsolute error on the {len(pred_only)} NON-calibrated points: "
      f"median {st.median(errs):.1f}%  max {max(errs):.1f}%")
Path("results/model_validation.json").write_text(json.dumps(
    {"t_fwd_ms":round(t_fwd,3),"k_bwd":round(k,4),"rows":rows},indent=2))
print("-> results/model_validation.json")
