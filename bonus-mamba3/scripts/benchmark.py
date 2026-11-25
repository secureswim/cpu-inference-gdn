"""One CPU thread, matched layer shape, randomized order, native timed loop."""
import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import psutil
from mamba_cpu import LIB, inputs, pointer_table

ROOT = Path(__file__).resolve().parents[1]
HEADS, WIDTH, STATE = 16, 128, 128


def measure(layers, variant, rank, simd, repeat):
    h, p, n = HEADS, WIDTH, STATE
    buffers = [inputs(h, p, n, rank, seed=531+i) for i in range(layers)]
    pointers = pointer_table(buffers)

    def run(rounds):
        latency = LIB.mamba_ring(rounds, layers, h, p, n, rank, variant, simd, pointers)
        if latency <= 0:
            raise RuntimeError("Invalid native timing")
        return latency

    run(2)
    trial = run(max(1, 32 // layers))
    rounds = max(2, round(0.4 / (trial * layers)))
    latency = run(rounds)
    assert all(np.isfinite(v["output"]).all() for v in buffers)
    elements = h*p*n
    main = 4*elements
    if variant == 2:
        carry = 0
        extra = 2*h*n + 3*h*p + 3*h
        flops = 5*elements
    else:
        carry = 4*(rank*h*n + h*p + h*(n//4))
        extra = 4*rank*h*n + 5*h*p + 4*h + 3*rank*h*p + 3*h*(n//4)
        flops = (3+6*rank)*elements
    logical_bytes = 2*main + 4*extra
    return dict(repeat=repeat, layers=layers, heads=h, width=p, state_dim=n,
                variant=variant, rank=rank, simd=simd, rounds=rounds,
                us_per_layer_step=latency*1e6, main_bytes_per_layer=main,
                auxiliary_carry_bytes_per_layer=carry, logical_bytes_per_step=logical_bytes,
                dominant_flops=flops, estimated_flops_per_byte=flops/logical_bytes,
                allocated_array_bytes=sum(a.nbytes for v in buffers for a in v.values()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, help="logical CPU ID for single-thread affinity")
    args = parser.parse_args()
    process = psutil.Process()
    if args.cpu is not None:
        process.cpu_affinity([args.cpu])
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    env = dict(os=platform.platform(), processor=platform.processor(), cpu_affinity=process.cpu_affinity(),
               started_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), ram_bytes=psutil.virtual_memory().total,
               free_at_start=psutil.virtual_memory().available, numpy=np.__version__,
               power_plugged=getattr(psutil.sensors_battery(), "power_plugged", None))
    (out/"environment.json").write_text(json.dumps(env, indent=2))
    cases = [(layers, variant, rank, simd) for layers in (1,64)
             for variant, rank in ((2,1),(3,1),(3,4)) for simd in (0,1)]
    rows = []
    for repeat in range(3):
        random.Random(225+repeat).shuffle(cases)
        for case in cases:
            rows.append(measure(*case, repeat))
    with (out/"latency.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
