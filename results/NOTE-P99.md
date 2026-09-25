# The TreeSHAP p99 was an artifact — record

First measurement: TreeSHAP with a median of 1.44 ms and a **p99 of 126 ms**.
That would have been a strong finding — a method that looks online by its median
and breaks both budgets at the tail. It did not survive verification.

Re-measured with a warmup of 50 (was 3), 2000 repetitions (were 30), and without
`tracemalloc` running during the timed pass:

| condition | median | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| GC enabled | 0.745 ms | 0.985 | 1.253 | 1.447 | **145.1** |
| GC disabled | 0.708 ms | 0.910 | 1.096 | 1.141 | **1.181** |

**The tail was CPython's garbage collector, not TreeSHAP.** With the GC off the
maximum drops from 145 ms to 1.18 ms — the algorithm is stable.

Three consequences:

1. The 126 ms p99 of the first round came from insufficient warmup plus an
   active `tracemalloc` during the measurement. **Never run tracemalloc in the
   same pass that measures time.**
2. TreeSHAP is ~1.3× bare inference, not 2.5×. (Superseded: the tabular ratio is
   itself unstable — see RESULTS.md rule 11. The paper reports % of budget.)
3. Worth stating in the paper: in online explanation the latency tail can come
   from the *runtime*, not the algorithm. On a real ECU the runtime is not
   CPython, but the methodological lesson stands — measuring a p99 without
   controlling the runtime measures the runtime.
