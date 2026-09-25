"""Does Grad-CAM really exceed the 100 ms frame, or is that noise?

The paper's most quotable claim came from ONE run: p99 = 100.33 ms against a
100 ms budget. Exceeding by 0.3% on a platform whose unit cost varies 16%
between runs is not a finding, it is a coin toss.

Here: large n, and a bootstrap confidence interval on the p99. If the interval
crosses 100 ms, the honest statement is "sits at the boundary", not "exceeds".
"""
import time, json, gc, statistics as st, random
from pathlib import Path
import torch
torch.manual_seed(20260823); random.seed(20260823); torch.set_num_threads(1)
from torchvision.models import resnet18
from captum.attr import LayerGradCam

m=resnet18(weights=None,num_classes=2).eval()
for p in m.parameters(): p.requires_grad_(True)
x=torch.randn(1,3,224,224)
cam=LayerGradCam(m,m.layer4)

def once():
    """One timed Grad-CAM attribution, in milliseconds."""
    t=time.perf_counter(); cam.attribute(x,target=0)
    return (time.perf_counter()-t)*1000

N=400
for _ in range(50): once()
gc.collect(); gc.disable()
try: lat=[once() for _ in range(N)]
finally: gc.enable()

s=sorted(lat); q=lambda v,p: v[min(len(v)-1,int(round(p*(len(v)-1))))]
def boot(p, B=2000):
    """Bootstrap 95% CI for the p-th quantile of the latency sample.

    Resampling with replacement, because a single p99 over n=100 carries no
    information about where the true p99 sits.
    """
    out=[]
    for _ in range(B):
        samp=sorted(random.choices(lat,k=len(lat)))
        out.append(q(samp,p))
    out.sort()
    return q(out,0.025), q(out,0.975)

BUDGET=100.0
print(f"Grad-CAM, n={N}, 1 thread, budget {BUDGET:.0f} ms\n")
print(f"{'statistic':>12}{'value':>11}{'95% CI':>22}{'verdict':>26}")
print("-"*72)
for name,p in (("median",.50),("p95",.95),("p99",.99)):
    v=q(s,p); lo,hi=boot(p)
    if hi<BUDGET:   verd="fits (whole CI below)"
    elif lo>BUDGET: verd="EXCEEDS (whole CI above)"
    else:           verd="BOUNDARY (CI crosses)"
    print(f"{name:>12}{v:>10.2f}ms   [{lo:>7.2f}, {hi:>7.2f}]{verd:>26}")
frac=sum(1 for v in lat if v>BUDGET)/len(lat)*100
print(f"\nsamples above {BUDGET:.0f} ms: {frac:.1f}%  (min {min(lat):.1f}, max {max(lat):.1f})")
Path("results/gradcam_boundary.json").write_text(json.dumps(
  {"n":N,"budget_ms":BUDGET,"median":round(st.median(lat),2),
   "p95":round(q(s,.95),2),"p99":round(q(s,.99),2),
   "pct_over_budget":round(frac,2)},indent=2))
print("-> results/gradcam_boundary.json")
