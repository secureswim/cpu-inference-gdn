# Gated DeltaNet CPU Inference

Project 1, Option 5. This study measures the fused `GATED_DELTA_NET` operator in a pinned llama.cpp/ggml CPU backend using synthetic Qwen3-Next layer shapes at batch size one.

[Report](output/pdf/Project_1_Report.pdf) · [Slides](output/Project_1_Presentation.pptx) · [Ten-minute talk script](output/Talk_Script.md) · [Plots](output/plots/) · [Benchmark data](results/)

## Results

- In the two-second timing run, one decode step at 1,024 tokens of starting history took **454 µs on one P core** and **429 µs on six P cores**. Six-core prefill at 4,096 tokens was **3.52×** as fast as one core. [Timing data](results/gdn_long_repeats.csv)
- One to 36 independent states changed mean latency from **563 to 535 µs** on one P core and **595 to 622 µs** on six. The latter difference was smaller than the 36-state repeat SD of 52 µs. [Working-set measurements](output/plots/working_sets.pdf)
- The recurrent state occupies **2 MiB** per layer. The Qwen3-Next F16 attention KV cache reaches **64 MiB** per attention layer at 32,768 tokens. [Storage comparison](output/plots/state_vs_kv.pdf)
- The correctness check covers 1,000 changing-input steps against an independent FP64 implementation and checks multi-token equivalence. [Correctness results](results/gdn_correctness.json)

## Windows reproduction

The measured machine is an Intel Core Ultra 9 185H, with six physical performance cores. The driver queries Windows topology and uses logical processors 0, 10, 12, 14, 16, and 18, one per P core. It checks actual topology at runtime. AVX2 and FMA are required by this x86 build.

Install Python 3.11+, CMake 3.20+, Ninja, Git, and a modern MinGW-w64 GCC toolchain on PATH. The original build uses GCC 13.2. Run from PowerShell at the repository root.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/prepare_ggml.py
cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/cmake -j 6
ctest --test-dir build/cmake --output-on-failure
```

When an existing llama.cpp checkout has the required commit, use `scripts/prepare_ggml.py --local-source PATH`. The script exports the committed ggml source and excludes unrelated local changes. It applies a compatibility guard for older Windows headers and optional timing instrumentation. The source SHA is fixed in the script. Build output remains outside Git.

Connect the charger, select Windows Best Performance, and exit browsers and IDEs. Run the build, correctness checks, interleaved measurements, plots, and report from PowerShell.

```powershell
./scripts/run.ps1
```

`scripts/run_benchmarks.py --clean-rerun --output build/reproduction` runs the measurements into a separate directory after the initial build. `scripts/run.ps1 -AllowBattery` saves provisional raw measurements separately under `build/`. The final run verified AC power and the effective Windows performance overlay. Chrome and Edge processes were absent at startup, and the chat remained open.

Run `scripts/run_benchmarks.py --long-repeats --output build/long-reproduction` for the two-second timing run. It uses three interleaved repeats across ten decode and prefill settings. The full thread-scaling plot uses a 0.35-second minimum at each worker count. [Timing data](results/gdn_long_repeats.csv) · [Run environment](results/long_repeats_environment.json)

## Measurement protocol

- **GDN shape** uses FP32 Q/K `[128,16,T,1]`, V `[128,32,T,1]`, scalar decay and beta gates, and initial state `[128,128,32,1]`. GGML's tiled head broadcast maps value head `h` to Q/K head `h % 16`.
- **Prefill** processes 512, 4,096, and 32,768 synthetic tokens in chunks of 128. The seeded block repeats across chunks, bounding buffers and making the state-history length explicit. Every complete prompt starts from zero state.
- **Decode** starts from a state advanced through 1,024, 8,192, or 32,768 tokens. Every repeat restores that initial state. All three histories end at the same phase of a repeating input block, and the measured maximum elementwise state difference is `5.96e-8`. The timed generation loop advances the recurrent state after every call. [History comparison](results/history_states.csv)
- **Thread scaling** covers 1 through all six physical P cores, with one hardware thread per core and no E cores in the main sweep. A persistent ggml thread pool uses zero polling. All configurations run serially.
- **Repetitions** include one graph warmup and three independent timed repeats of at least 0.35 seconds. Long prefill prompts can exceed this minimum. Raw CSVs retain elapsed time and token count. Analysis reports mean and sample standard deviation plus min/max.
- **Memory** uses Windows peak process working set and free system RAM. Each configuration gets a separate process. Within a configuration, the peak is cumulative across the three repeats, giving an upper bound for each repeat. The runner aborts above 4.5 GB process RSS or below 1 GB free RAM. This check does not directly count hard page faults.
- **STREAM** uses the official unmodified 5.10 C source, 20 million doubles per array, and 20 passes per execution. Each 160 MB array exceeds four times this machine's 24 MiB LLC, as the STREAM sizing rule requires. The three arrays occupy 480 MB. Three independent executions run at every thread count. The summary averages the best Triad bandwidth from those executions. Complete text outputs and a separate timestamped STREAM environment record accompany the CSV. Use `scripts/run_benchmarks.py --stream-only` to repeat this portion alone.
- **Profiling** combines single-worker scoped Windows timings with supplementary Ubuntu/WSL2 `perf` sampling. The scoped percentages exclude the serial driver copy and graph scheduling. The working-set diagnostic times the driver copy separately from the graph. Head ranges split ggml's initial copy across workers. [Profiling methods and evidence](results/profiles/README.md)
- **Own experiment** cycles 1, 4, 12, or 36 independent states through a single graph. It holds the operator's work per step fixed and reports latency and separate graph/copy diagnostics. The 36-state round models a 72 MiB recurrent footprint. The affinity comparison is secondary.

## Roofline and memory interpretation

The logical lower bound counts state read/write, the driver carry, and per-token inputs and outputs. The scalar-gated operator does about 3.67 MFLOP per step, counting FMA as two operations and excluding exponentials. The resulting logical arithmetic intensity is about 0.435 FLOP/byte. The roofline uses copy rates at fixed 1 MiB and 16 MiB combined buffer footprints, large-array STREAM Triad, and a measured AVX2 FP32 FMA rate. These are empirical reference ceilings. Actual traffic at each memory level remains unmeasured. [Bandwidth sweep](output/plots/cache_bandwidth.pdf) · [Cache and compute roofline](output/plots/cache_roofline.pdf)

The analytical attention comparison uses F16 KV and Qwen3-Next's two KV heads of width 256. KV memory per attention layer is `2 * T * 2 * 256 * 2` bytes. GDN's main state is `32 * 128 * 128 * 4` bytes, exactly 2 MiB. The storage comparison excludes input/output buffers, convolution state, and model weights. [Qwen3-Next configuration](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)

## Shared PARAM Rudra reference

The four-thread run, job **186891**, measured **92.77 ± 0.16 combined tokens/s** and **1.33 GiB peak RSS** on `rdcn20` (Xeon Gold 6240R). It used Llama 3.2 1B Instruct Q4_K_M, a 512-token prompt, a 128-token decode, and three repeats. The wrapper measures the benchmark child's peak RSS with Linux `getrusage`. [Run data and logs](results/param-rudra/186891/)

Archived measurements are available for the [earlier PARAM Rudra job](results/param-rudra/186881/) and the [earlier laptop session](https://github.com/secureswim/cpu-inference-gdn-mamba3/tree/3bb2993/results).

Use `scripts/run_shared_reference.sh` inside a CPU allocation with `LLAMA_SOURCE`, `MODEL`, and `THREADS` set. The original GCC 8.5 build linked with `-DCMAKE_CXX_STANDARD_LIBRARIES:STRING=-lstdc++fs`.

## Validation

`tests/validate_results.py` checks the 120 GDN rows, 72 STREAM rows, working-set and ceiling data, memory bounds, four-page report, and ten-minute talk plan.

## Sources

- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp/tree/8bb3cf0d216a5ae48360179f2931daf284be4d01)
- [STREAM source and run rules](https://www.cs.virginia.edu/stream/)

See `vendor/README.md` for upstream provenance.
