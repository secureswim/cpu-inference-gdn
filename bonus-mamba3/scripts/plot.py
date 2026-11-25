"""Plot measured AVX2 latency, with min/max across three repeats."""
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
with (root/"results/latency.csv").open() as stream:
    rows = list(csv.DictReader(stream))
fig, ax = plt.subplots(figsize=(7,4))
for index, (variant, rank, label) in enumerate([(2,1,"Mamba-2"),(3,1,"Mamba-3 SISO"),(3,4,"Mamba-3 MIMO R4")]):
    values = [[float(r["us_per_layer_step"]) for r in rows
               if (int(r["layers"]), int(r["variant"]), int(r["rank"]), int(r["simd"])) ==
                  (layers, variant, rank, 1)] for layers in (1,64)]
    avg = [mean(v) for v in values]
    ax.bar([x+(index-1)*0.25 for x in (0,1)], avg, width=0.24, label=label,
           yerr=[[a-min(v) for a,v in zip(avg,values)], [max(v)-a for a,v in zip(avg,values)]],
           capsize=3)
ax.set_xticks([0,1], ["1 layer (1 MiB main state)", "64 layers (64 MiB main state)"])
ax.set_ylabel("Microseconds per layer decode step")
ax.set_title("Batch-one CPU recurrence, AVX2/FMA")
ax.legend()
fig.tight_layout()
fig.savefig(root/"results/latency.png", dpi=170)
