"""Execute the unmodified official pure-PyTorch functions on CPU.

AST extraction avoids importing CUDA-only CuTe/Triton modules. The executed
function bodies remain identical to the pinned upstream files in vendor/.
"""
import ast
import json
import math
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mamba_cpu import inputs, step


def official_function(filename, name):
    tree = ast.parse((ROOT / "vendor" / filename).read_text(encoding="utf-8"))
    selected = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "math": math, "Optional": Optional}
    exec(compile(ast.Module(body=[selected], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


def check_case(rank, simd, n, seed, variant=3):
    heads, p = 2, 13  # Non-square state and SIMD tails catch indexing mistakes.
    v = inputs(heads, p, n, rank, seed)
    state = torch.from_numpy(v["state"].copy()).double()[None]
    prev_b = torch.from_numpy(v["bprev"].copy())[None]
    prev_x = torch.from_numpy(v["xprev"].copy())[None]
    angle = torch.from_numpy(v["angle"].copy())[None]
    recurrence = official_function("mamba3_step_fn.py", "selective_state_update_fused_ref_v2")
    rotary = official_function("mamba3_mimo_rotary_step.py", "apply_rotary_qk_inference_reference")
    rng = np.random.default_rng(seed + 11)
    maximum = {key: 0.0 for key in ("state", "output", "bprev", "angle")}
    for t in range(1000):
        for key in ("b", "c", "x", "z", "angle_proj"):
            v[key][:] = rng.normal(0, 0.3, v[key].shape)
        v["dt"][:] = rng.uniform(0.01, 0.3, heads)
        v["trap"][:] = rng.uniform(0.05, 0.95, heads)
        b, c, x, z = [torch.from_numpy(v[key])[None] for key in ("b", "c", "x", "z")]
        dt, a, trap = [torch.from_numpy(v[key])[None] for key in ("dt", "a", "trap")]
        if variant == 3:
            c, b, angle = rotary(c, b, angle, torch.from_numpy(v["angle_proj"])[None], dt,
                                  rotate_pairwise=(rank == 1))
        output, state = recurrence(state, a, b, c, torch.from_numpy(v["xproj"]), x,
                                   torch.from_numpy(v["zproj"]), z, dt, prev_b, prev_x,
                                   trap if variant == 3 else torch.ones_like(trap),
                                   torch.from_numpy(v["skip"]), torch.from_numpy(v["oproj"]),
                                   compute_dtype=torch.float64)
        # Euler baseline has no previous-input contribution because trap=1.
        step(v, heads, p, n, rank, variant, simd)
        expected = {"state": state[0].numpy(), "output": output[0].numpy(),
                    "bprev": b[0].numpy(), "angle": angle[0].numpy()}
        for key, value in expected.items():
            maximum[key] = max(maximum[key], float(np.max(np.abs(v[key] - value))))
            np.testing.assert_allclose(v[key], value, atol=2e-5, rtol=2e-4,
                                       err_msg=f"rank={rank} simd={simd} n={n} step={t} {key}")
        np.testing.assert_array_equal(v["xprev"], v["x"])
        prev_b, prev_x = b.clone(), x.clone()
    return {"rank": rank, "simd": simd, "n": n, "seed": seed, "variant": variant,
            "steps": 1000, "atol": 2e-5, "rtol": 2e-4, "max_abs": maximum}


if __name__ == "__main__":
    torch.set_num_threads(1)
    results = []
    for n in (16, 30, 128):
        for rank in (1, 4):
            for simd in (0, 1):
                row = check_case(rank, simd, n, 97 + n)
                results.append(row)
                print(json.dumps(row), flush=True)
    for simd in (0, 1):
        results.append(check_case(1, simd, 128, 123, variant=2))
    (ROOT / "results/mamba_correctness.json").write_text(json.dumps(results, indent=2))
