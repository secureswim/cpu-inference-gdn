"""Contiguous FP32 C ABI for the CPU recurrence, independent of PyTorch."""
import ctypes
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_dll_directories = []
if os.name == "nt":
    for folder in (Path("C:/Strawberry/c/bin"), ROOT / "build/cmake"):
        if folder.exists():
            _dll_directories.append(os.add_dll_directory(str(folder)))
LIB = ctypes.CDLL(str(ROOT / "build/cmake" / ("libmamba3_decode.dll" if os.name == "nt" else "libmamba3_decode.so")))
LIB.mamba_step.argtypes = [ctypes.c_int] * 6 + [ctypes.c_void_p] * 18
LIB.mamba_step.restype = ctypes.c_int
LIB.mamba_ring.argtypes = [ctypes.c_int] * 8 + [ctypes.POINTER(ctypes.c_void_p)]
LIB.mamba_ring.restype = ctypes.c_double
NAMES = ("state", "bprev", "xprev", "angle", "b", "c", "x", "a", "dt", "trap",
         "xproj", "oproj", "z", "zproj", "skip", "angle_proj", "output", "scratch")


def inputs(heads, p, n, rank, seed=2023522):
    rng = np.random.default_rng(seed)
    shapes = {
        "state": (heads, p, n), "bprev": (rank, heads, n), "xprev": (heads, p),
        "angle": (heads, n // 4), "b": (rank, heads, n), "c": (rank, heads, n),
        "x": (heads, p), "a": (heads,), "dt": (heads,), "trap": (heads,),
        "xproj": (rank, heads, p), "oproj": (rank, heads, p), "z": (heads, p),
        "zproj": (rank, heads, p), "skip": (heads,), "angle_proj": (heads, n // 4),
        "output": (heads, p), "scratch": (heads * 3 * rank * n,),
    }
    values = {key: (rng.normal(0, 0.3, shape)).astype(np.float32) for key, shape in shapes.items()}
    values["a"] = -rng.uniform(0.1, 2, heads).astype(np.float32)
    values["dt"] = rng.uniform(0.01, 0.3, heads).astype(np.float32)
    values["trap"] = rng.uniform(0.1, 0.9, heads).astype(np.float32)
    if rank == 1:
        for key in ("xproj", "oproj", "zproj"):
            values[key].fill(1)
    for key in ("bprev", "xprev", "angle"):
        values[key].fill(0)
    return values


def arguments(values):
    return [value.ctypes.data for value in (values[name] for name in NAMES)]


def step(values, heads, p, n, rank, variant, simd):
    result = LIB.mamba_step(heads, p, n, rank, variant, simd, *arguments(values))
    if result:
        raise ValueError("Unsupported shape or variant")


def pointer_table(layers):
    addresses = [address for layer in layers for address in arguments(layer)]
    return (ctypes.c_void_p * len(addresses))(*addresses)
