"""Is Integrated Gradients just N passes, or does it do work of its own?

The cost model as a LOWER BOUND:  T >= N_passes * t_pass
The gap measures work the count cannot see -- path interpolation, the Riemann
sum, allocating the copies.

Paired design, which is what makes this measurable: for each n_steps, measure
SIDE BY SIDE in the same time window
   (a) grad_input at batch = n  -> the pure cost of the n passes IG performs
   (b) IG with n_steps = n      -> the complete method
and report the ratio (b)/(a) within the pair. Using the same batch in (a)
eliminates the batch effect entirely.

A near-constant ratio means IG is passes plus a proportional cost, and the
model holds as a lower bound with a known factor. A ratio that grows with n
would mean superlinear work and a second model term.
"""
import time, json, gc, statistics as st
from pathlib import Path
import torch
torch.manual_seed(20260823); torch.set_num_threads(1)
from torchvision.models import resnet18
from captum.attr import IntegratedGradients
m=resnet18(weights=None,num_classes=2).eval()
for p in m.parameters(): p.requires_grad_(False)
ig=IntegratedGradients(m)
x=torch.randn(1,3,224,224)

def t_grad(n):
    """A closure computing d(output)/d(input) for a batch of n, as IG does."""
    xb=torch.randn(n,3,224,224,requires_grad=True)
    def go():
        """Run the model and take the input gradient, discarding the result."""
        torch.autograd.grad(m(xb)[:,0].sum(), xb)
    return go
def t_ig(n):
    """A closure running Integrated Gradients with n_steps=n."""
    return lambda: ig.attribute(x,target=0,n_steps=n)

def once(fn):
    """One call of fn, in milliseconds."""
    t=time.perf_counter(); fn(); return (time.perf_counter()-t)*1000

STEPS=[5,10,20,50]; PAIRS=5
for n in (5,): once(t_grad(n)); once(t_ig(n))          # warmup
rows=[]
print(f"{'n_steps':>8}{'pure passes':>17}{'full IG':>14}{'ratio':>9}{'excess':>12}")
print("-"*62)
gc.collect(); gc.disable()
try:
    for n in STEPS:
        g,i=t_grad(n),t_ig(n)
        for _ in range(2): g(); i()                     # per-point warmup
        rs,gs,is_=[],[],[]
        for _ in range(PAIRS):
            a=once(g); b=once(i); c=once(g)
            base=(a+c)/2
            rs.append(b/base); gs.append(base); is_.append(b)
        r=st.median(rs)
        rows.append({"n_steps":n,"pure_passes_ms":round(st.median(gs),1),
                     "ig_ms":round(st.median(is_),1),"ratio":round(r,3),
                     "excess_ms":round(st.median(is_)-st.median(gs),1)})
        print(f"{n:>8}{st.median(gs):>16.0f}ms{st.median(is_):>13.0f}ms"
              f"{r:>9.2f}{st.median(is_)-st.median(gs):>11.0f}ms")
finally: gc.enable()

r=[x["ratio"] for x in rows]
print(f"\nIG/passes ratio: {min(r):.2f} to {max(r):.2f}"
      f"  (spread {(max(r)/min(r)-1)*100:+.0f}%)")
if max(r)/min(r) < 1.25:
    print("=> ratio ~constant: IG is N passes times a fixed factor.")
    print("   O modelo vale como LIMITE INFERIOR com fator conhecido.")
else:
    print("=> the ratio grows with n: superlinear work beyond the passes.")
ex=[x["excess_ms"] for x in rows]
print(f"excess per step: " + ", ".join(
    f"n={x['n_steps']}: {x['excess_ms']/x['n_steps']:.1f}ms" for x in rows))
Path("results/ig_overhead.json").write_text(json.dumps(rows,indent=2))
print("-> results/ig_overhead.json")
