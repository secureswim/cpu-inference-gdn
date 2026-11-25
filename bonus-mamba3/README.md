# Mamba-3 CPU decode — Project 1 bonus

This experiment asks how much rank-4 MIMO costs per decoded token in one recurrent layer. It implements batch-one FP32 Mamba-3 SISO and MIMO, with a Mamba-2 selective-state-update baseline. Each has plain C++ and AVX2/FMA paths. The target platform is x86-64 with GCC, AVX2 and FMA.

## Reproduce

Run from this directory. Install CMake, Ninja, GCC and Python with NumPy, CPU PyTorch, psutil and matplotlib. The recorded run used GCC 13.2, Python 3.11, NumPy 2.4.6, PyTorch 2.9.1+cpu, psutil 7.2.2 and matplotlib 3.11.2. On Windows, put MinGW GCC and its runtime DLLs on PATH. The existing Project 1 Python environment already contains these packages.

| Command | Purpose |
|---|---|
| `python run.py --cpu 0` | Build, validate, benchmark and plot. Choose a performance-core logical ID appropriate to your machine. |
| `python run.py --cpu 0 --ggml` | Also build and check the optional ggml graph using the main project's prepared ggml source. All build outputs go under this bonus directory. |
| `python tests/check_mamba.py` | With the library already built, check the native implementation against upstream PyTorch at every step. |
| `python tests/check_identity.py` | Check the rank-one reduction using one active lane and the corresponding rotary-coordinate permutation. |

The stretch build expects the ggml source from llama.cpp revision `8bb3cf0d216a5ae48360179f2931daf284be4d01` under `../build/vendor/ggml`. The default build needs no ggml, model download, GPU or network connection once dependencies are installed.

## What one step measures

Inputs start after dense projection, B/C normalization and bias, and activation of the time step and trapezoid gate. The kernel advances the rotary phase, rotates B and C over half the real-valued state coordinates, updates the recurrent matrix, reads it out, applies the skip and SiLU gates, and mixes rank outputs. SISO rotates adjacent coordinates. MIMO pairs coordinates from opposite halves. The CPU test runs the pinned upstream rotary and recurrence reference functions via AST extraction to avoid loading GPU modules. See [upstream provenance](vendor/README.md).

For each head let α = exp(A Δt), β = (1 − λ) Δt α, and γ = λ Δt. Each rank lane projects the current and previous value vector using the same learned input mixing weights. The recurrent update scales the old matrix by α and adds two sums of outer products. The current values and rotated keys carry coefficient γ, and the previous values and previous rotated keys carry coefficient β. Each output lane reads this common updated matrix through its rotated query. Increasing rank adds outer products and readouts while keeping the matrix dimensions fixed. It also increases the size of the previous-key buffer. The Mamba-2 baseline uses its single current outer product and skips the rotary and previous-token work.

## Method and interpretation

- Deterministically seeded synthetic FP32 inputs. Correctness checks vary inputs at each of 1,000 consecutive steps, use a float64 upstream recurrence with the upstream rotary reference, and check output, recurrent matrix, previous key/value and phase where applicable. Tolerances are absolute 2e−5 and relative 2e−4. State sizes 16, 30 and 128 cover both aligned vectors and scalar tails.
- The benchmark fixes 16 heads, 128 value channels and 128 state coordinates. One recurrent matrix takes 1 MiB. The large working set cycles through 64 independent layers, each with its own state, projected inputs and scratch. Timings use fixed per-layer inputs and evolving states. These are operator measurements; whole-model throughput would include projections and other layers.
- Native timed loops use a single thread, two warm-up rounds, a calibration run, and approximately 0.4 seconds per sample. Three repeats use a deterministic shuffled case order. Initialization, Python dispatch and allocation are outside timed loops. The plot shows mean and min/max.
- Estimated arithmetic intensity divides dominant state-update/readout FLOPs by logical tensor bytes for a step. For H heads, P channels, N coordinates and rank R, the FLOP model is HPN(3+6R) for Mamba-3 and 5HPN for Mamba-2. It excludes elementwise setup, nonlinearities and mixing. Bytes include a read and write of the state plus logical inputs, auxiliary carry and output. Scratch, register/cache reuse, actual memory-controller traffic and instruction counts are outside this estimate.
- The ggml stretch uses a custom CPU graph node with packed tensors. Its 1,000-step test at each rank compares the full packed result with direct calls. The graph's packing and copy overhead are excluded from native benchmark timings.

See [measured results](RESULTS.md), [raw timings](results/latency.csv), [reference errors](results/mamba_correctness.json) and [graph checks](results/ggml.txt). Kernel and validation code build on the earlier local Project 1 experiment at commit `b83079a`. The separate bonus build adds a dedicated Mamba-2 path, a native rotating-layer benchmark, and the rank-reduction check.
