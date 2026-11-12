# CPU inference project

Option 5 Gated DeltaNet CPU operator study.

Author and Git commit identity are Siddharth Kumar, `siddharthatdmk@gmail.com`.

## Assignment scope

This project selects **Option 5**. It benchmarks the actual fused `GATED_DELTA_NET` operation from a pinned llama.cpp/ggml CPU backend using the specified Qwen3-Next shapes. It measures a batch-one recurrent layer with synthetic tensors. A full language model, model quantization sweep, and perplexity do not apply to this operator option. Operator throughput cannot establish full-model inference throughput.

## Status

Correctness checks pass. The main GDN check covers 1,000 changing-input steps against an independent FP64 implementation and tests multi-token equivalence.

The AC-powered sweeps, plots, four-page report, and talk materials are complete. The effective Windows overlay was Best Performance, while its base scheme reports Balanced. A separate power-confirmation record verifies the overlay. Desktop applications remained open, and this limitation is stated in the report.

GDN uses 2 MiB persistent main state per layer, compared with an analytical 512 MiB FP32 attention KV cache at 32k tokens with 16 KV heads.

The raw data contain 120 GDN rows and 72 STREAM rows. `tests/validate_results.py` audits full sweep coverage, repeat counts, rates, memory bounds, correctness records, report length, and talk duration. External requirements appear in `SUBMISSION_STATUS.md`, and the shared PARAM Rudra result is recorded below.

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

Connect the charger, choose performance power mode, close other applications where possible, then run the full measurement and analysis sequence.

```powershell
.venv/Scripts/python.exe scripts/run_benchmarks.py
.venv/Scripts/python.exe scripts/analyze.py
.venv/Scripts/python.exe scripts/make_report.py
```

`scripts/run.ps1` combines setup, correctness, measurements, plots, and report generation. It fails when the laptop reports battery power. An explicit `-AllowBattery` produces provisional measurements with that condition recorded in `results/environment.json`.

## Measurement protocol

- **GDN shape** uses FP32 Q/K `[128,16,T,1]`, V `[128,32,T,1]`, scalar decay and beta gates, and initial state `[128,128,32,1]`. GGML's tiled head broadcast maps value head `h` to Q/K head `h % 16`.
- **Prefill** processes 512, 4,096, and 32,768 synthetic tokens in chunks of 128. The seeded block repeats across chunks, bounding buffers and making the state-history length explicit. Every complete prompt starts from zero state.
- **Decode** starts from a state advanced through 1,024, 8,192, or 32,768 tokens. Every repeat restores that initial state. The timed generation loop advances the recurrent state after every call. Its storage stays fixed even as history grows.
- **Thread scaling** covers 1 through all six physical P cores, with one hardware thread per core and no E cores in the main sweep. A persistent ggml thread pool uses zero polling. All configurations run serially.
- **Repetitions** include one graph warmup and three independent timed repeats of at least 0.35 seconds. Long prefill prompts can exceed this minimum. Raw CSVs retain elapsed time and token count. Analysis reports mean and sample standard deviation plus min/max.
- **Memory** uses Windows peak process working set and free system RAM. Each configuration gets a separate process. Within a configuration, the peak is cumulative across the three repeats, giving an upper bound for each repeat. The runner aborts above 4.5 GB process RSS or below 1 GB free RAM. This check does not directly count hard page faults.
- **STREAM** uses the official unmodified 5.10 C source, 20 million doubles per array, and 20 passes per execution. Each 160 MB array exceeds four times this machine's 24 MiB LLC, as the STREAM sizing rule requires. The three arrays occupy 480 MB. Three independent executions run at every thread count. The summary averages the best Triad bandwidth from those executions. Complete text outputs and a separate timestamped STREAM environment record accompany the CSV. Use `scripts/run_benchmarks.py --stream-only` to repeat this portion alone.
- **Profiling** adds runtime-selectable scoped timers to the actual upstream GDN loop. Separate single-worker prefill/decode runs measure initial state copy, decay, delta dot products, state update, and output dot products. Percentages exclude driver copy and scheduling. Timer overhead affects only the instrumented runs. Throughput comes from uninstrumented runs. `scripts/profile_linux.sh` supplies sampled `perf` commands. A separate VTune/uProf trace remains a submission requirement on Windows.
- **Own experiment** compares P-core pinning with unrestricted OS placement using the same worker count and workload. It records mean and spread. Run order and background processes can confound small differences.

## Roofline and memory interpretation

The driver counts a logical lower bound comprising state read/write, the output-to-input state carry copy, and per-token Q/K/V, gates, and output. It divides measured STREAM bandwidth by those bytes. The upstream implementation makes additional passes over state. Its 2 MiB state can remain in cache, so a DRAM prediction can underestimate throughput even though repeated cache traffic creates other limits. The report distinguishes these assumptions and avoids assigning a measured gap to throttling or cache misses without hardware-counter evidence.

The analytical attention comparison uses FP32 KV, 16 KV heads, and head size 128. KV memory per layer is `2 * T * 16 * 128 * 4` bytes. GDN's main state is `32 * 128 * 128 * 4` bytes, exactly 2 MiB. Input/output buffers, model weights, and full-model layers lie outside that storage comparison.

## Files and deliverables

| Location | Contents |
| --- | --- |
| `src/gdn.cpp` | Actual ggml operator driver and independent FP64 check |
| `scripts/` | Pinned setup, sweeps, analysis, report, and external-node helpers |
| `results/` | Hardware, environment, correctness evidence, raw CSVs, and summaries |
| `output/plots/` | Required plots in PNG and vector PDF |
| `output/pdf/Project_1_Report.pdf` | Four-page report generated from measurements |
| `output/Project_1_Presentation.pptx` | Presentation with editable data charts and speaker notes |
| `output/Talk_Script.md` | Script for a ten-minute delivery |

The course template, team details, and a student-recorded talk require course/user input. Windows blocked supplemental CPU sampling with profiling-policy error `0xc5585011`. The report includes phase timings measured within the actual upstream operator, and `results/profiles/README.md` records the sampling limitation. `scripts/run_shared_reference.sh` runs the exact combined 512-token prompt / 128-token decode test with CPU-only Llama-3.2-1B Q4_K_M. It requires the allocated thread count and model path.

PARAM Rudra job **186881** completed on `rdcn20` (Xeon Gold 6240R). Llama 3.2 1B Instruct Q4_K_M with four CPU threads averaged **93.92 ± 0.07 tokens/s** (mean ± SD across three repeats) for the combined 512-token prompt and 128-token decode. [Raw benchmark and Slurm evidence](results/param-rudra/186881/). The GCC 8.5 build linked with `-DCMAKE_CXX_STANDARD_LIBRARIES:STRING=-lstdc++fs`. The compute node lacked GNU time and Slurm reported zero for MaxRSS, so peak RSS is unavailable.

The talk has twelve slides, editable charts, speaker notes, and a 600-second delivery plan. `scripts/make_deck.mjs` supplies its source. Rebuilding that deck uses the Codex bundled Artifact Tool runtime, with `RUNTIME_NODE_MODULES`, `RUNTIME_PYTHON`, and `PRESENTATIONS_SKILL_DIR` set to the installed runtime/skill locations. Set `DECK_OUTPUT` to a new output filename when rebuilding an existing deck. This authoring dependency is separate from the experiment's Python/C++ dependencies. The committed PowerPoint file is directly editable in PowerPoint.


## Sources

- [llama.cpp and ggml](https://github.com/ggml-org/llama.cpp/tree/8bb3cf0d216a5ae48360179f2931daf284be4d01)
- [STREAM source and run rules](https://www.cs.virginia.edu/stream/)

See `vendor/README.md` for upstream provenance. Results belong to the recorded CPU, build, input distribution, and power conditions. No measured values are invented.
