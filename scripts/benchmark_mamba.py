"""One configuration per process, with three repetitions and peak RSS."""
import json
import sys
import psutil
from mamba_cpu import inputs, bench

heads, variant, rank, simd = map(int, sys.argv[1:])
p, n = 128, 128
values = inputs(heads, p, n, rank)
warm = bench(values, heads, p, n, rank, variant, simd, 10)
iterations = max(10, int(0.35 / warm))
rows = []
for repeat in range(3):
    values = inputs(heads, p, n, rank)
    seconds = bench(values, heads, p, n, rank, variant, simd, iterations)
    mem = psutil.Process().memory_info()
    # Logical traffic lower bound. Fused SIMD reads/writes the state once.
    # Includes carried K/X/angle and projection/readout arrays. Scalar does an
    # additional state read for readout, which this lower bound excludes.
    state_bytes = heads * p * n * 4
    arrays_bytes = 4 * (4 * rank * heads * n + 2 * heads * p + 4 * heads
                       + 3 * rank * heads * p + 2 * heads * (n // 4) + heads * p)
    # Persistent state: S + previous K + previous X + cumulative rotary angles.
    persistent = state_bytes + 4 * (rank * heads * n + heads * p + heads * (n // 4))
    # Per state element: decay + two rank outer products + rank readout dots.
    flops = heads * p * n * (3 + 6 * rank) if variant == 3 else heads * p * n * 5
    rows.append(dict(heads=heads, p=p, n=n, variant=variant, rank=rank, simd=simd,
                     repeat=repeat, iterations=iterations, seconds_step=seconds,
                     tokens_s=1 / seconds, state_bytes=state_bytes,
                     persistent_bytes=persistent, logical_bytes_step=2*state_bytes+arrays_bytes,
                     dominant_flops_step=flops,
                     intensity=flops/(2*state_bytes+arrays_bytes),
                     peak_rss_bytes=getattr(mem, "peak_wset", mem.rss)))
print(json.dumps(rows))
