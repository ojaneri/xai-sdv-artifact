"""Batch effect, measured with PAIRED/INTERLEAVED sampling.

Two runs of the same unpaired experiment produced OPPOSITE trends (+37% and
-29%): this machine's load variation is of the same order as the effect being
looked for. Comparing A (measured now) against B (measured three minutes from
now) measures the difference between two moments of the server, not between A
and B.

Correct design: in each round, measure batch 1 and batch b SIDE BY SIDE and keep
the RATIO of that pair. Slow noise affects both members of a pair almost
equally and cancels out. Report the median ratio and the interval -- if the
interval crosses 1.0, the effect is not measurable here, and saying so is the
result.
"""
import time, json, gc, statistics as st
from pathlib import Path
import torch
torch.manual_seed(20260823); torch.set_num_threads(1)
from torchvision.models import resnet18
m=resnet18(weights=None,num_classes=2).eval()
for p in m.parameters(): p.requires_grad_(False)

def once(x):
    """One forward pass under no_grad, in milliseconds."""
    t=time.perf_counter()
    with torch.no_grad(): m(x)
    return (time.perf_counter()-t)*1000

BATCHES=[2,5,10,20,50]
ROUNDS=15
x1=torch.randn(1,3,224,224)
xs={b:torch.randn(b,3,224,224) for b in BATCHES}
for _ in range(10):                      # shared warmup
    once(x1); once(xs[50])

ratios={b:[] for b in BATCHES}
gc.collect(); gc.disable()
try:
    for r in range(ROUNDS):
        for b in BATCHES:
            a=once(x1)/1                  # ms per image at batch 1
            c=once(xs[b])/b               # ms per image at batch b
            d=once(x1)/1                  # batch 1 again: sandwich the measurement
            base=(a+d)/2                  # temporal mean around the measured point
            ratios[b].append(c/base)
finally: gc.enable()

print(f"batch effect, {ROUNDS} interleaved pairs per point")
print(f"{'batch':>6}{'median ratio':>16}{'p10':>9}{'p90':>9}   verdict")
print("-"*62)
rows=[]
for b in BATCHES:
    v=sorted(ratios[b]); q=lambda p: v[min(len(v)-1,int(p*(len(v)-1)))]
    med,lo,hi=st.median(v),q(.10),q(.90)
    cruza = lo<=1.0<=hi
    verdict = "not measurable (interval crosses 1.0)" if cruza else \
              ("CHEAPER per image" if med<1 else "MORE EXPENSIVE per image")
    rows.append({"batch":b,"median_ratio":round(med,3),"p10":round(lo,3),
                 "p90":round(hi,3),"significant":not cruza})
    print(f"{b:>6}{med:>15.3f}{lo:>9.3f}{hi:>9.3f}   {verdict}")

sig=[r for r in rows if r["significant"]]
print(f"\npoints where the effect clears the noise: {len(sig)}/{len(rows)}")
if sig:
    print("  " + ", ".join(f"batch {r['batch']}: x{r['median_ratio']}" for r in sig))
else:
    print("  none -- on this platform the batch effect sits BELOW the noise.")
Path("results/batch_paired.json").write_text(json.dumps(rows,indent=2))
print("-> results/batch_paired.json")
