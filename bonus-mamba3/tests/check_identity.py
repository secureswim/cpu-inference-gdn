"""Embed SISO into rank four by activating one identity projection lane."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from mamba_cpu import inputs, step


def check(simd):
    h, p, n = 2, 7, 30
    single = inputs(h,p,n,1)
    multi = inputs(h,p,n,4)
    # SISO rotates adjacent pairs. MIMO pairs across the two state halves.
    half = n//2
    pairs = n//4
    perm = [index for i in range(pairs) for index in (i,half+i)]
    perm.extend(sorted(set(range(n))-set(perm)))
    multi["state"][..., perm] = single["state"]
    for name in ("a","dt","skip","trap"):
        multi[name][:] = single[name]
    for name in ("xproj","oproj","zproj"):
        multi[name].fill(0)
        multi[name][0].fill(1)
    rng = np.random.default_rng(974)
    for _ in range(1000):
        for name in ("x","z","angle_proj","b","c"):
            single[name][:] = rng.normal(0,0.3,single[name].shape)
            if name in ("b","c"):
                multi[name][0][...,perm] = single[name][0]
            else:
                multi[name][:] = single[name]
        step(single,h,p,n,1,3,simd)
        step(multi,h,p,n,4,3,simd)
        for actual, expected in [(multi["state"][...,perm],single["state"]),
                                 (multi["output"],single["output"])]:
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4, equal_nan=False)


if __name__ == "__main__":
    for simd in (0,1):
        check(simd)
        print(f"identity lane: 1000 steps passed, SIMD={simd}")
