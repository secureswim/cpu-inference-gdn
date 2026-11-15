# Gated DeltaNet CPU Inference — Option 5

CPU-only study of llama.cpp's `GATED_DELTA_NET` operator on synthetic Qwen3-Next tensors: **16 query/key heads, 32 value heads, head size 128**, FP32 inputs/state. The questions are the cost of one decode step and how recurrent state compares with attention KV storage at 1k–32k context. Option 5 requires no full model, downloaded weights, or model-level perplexity test. No bonus implementation is included.

## Submitted work

| Required item | Location |
| --- | --- |
| Four-page report | [Project_1_Report.pdf](output/pdf/Project_1_Report.pdf) |
| Presentation | [Project_1_Presentation.pptx](output/Project_1_Presentation.pptx) |
| Implementation and correctness | `src/`, `CMakeLists.txt`, `tests/` |
| Build and benchmark scripts | `scripts/configure.py`, `scripts/run.py`, `scripts/run_benchmarks.py` and their supporting scripts |
| Raw results and hardware/build records | [results/](results/) |
| Plotting code and figures | `scripts/analyze.py`, [output/plots/](output/plots/) |
| Required own experiment | `results/working_sets.csv`, `history_states.csv`, `ceilings.csv`: rotate 1/4/12/36 independent states at fixed per-step work |
| Shared PARAM Rudra test | [results/param-rudra/186891/](results/param-rudra/186891/), `scripts/run_shared_reference.sh` |

**Recording still required:** the PPTX contains 13 slides but no embedded narration/video. Submit the recorded ten-minute talk separately using the course's common slide template or equivalent. Repository checks do not verify its duration.

## 1. Configure and check the submission

Install **64-bit Python 3.11+** and open a terminal in the repository root.

Linux / macOS:

```sh
python3 scripts/configure.py
python3 tests/validate_results.py
```

Windows PowerShell:

```powershell
python scripts/configure.py
python tests/validate_results.py
```

Configuration is mandatory. It detects the host, creates `.venv`, installs dependencies/CMake/Ninja, selects a native C/C++ OpenMP toolchain, prepares the pinned CPU backend, builds it, and runs correctness checks. The validator checks submitted CSV coverage/repeats, memory/timing bounds, correctness, STREAM checksum, PARAM Rudra evidence, figures, four report pages, and PPTX structure. Both commands should exit successfully. Python workflow/validation commands automatically use `.venv`; activation is optional.

Supported targets: Windows, Linux, and macOS on x86-64 or ARM64, including Apple Silicon. Missing tools are installed through winget, supported Linux package managers, or Homebrew/Apple Command Line Tools as appropriate. Downloads need network access; system installers may need administrator access or a restart. Python must include `venv`/`ensurepip`; setup attempts the matching venv package on Debian/Ubuntu. If automatic installation is unavailable, provision the missing tools and rerun.

The remaining examples use `python3`; use `python` on Windows.

## 2. Reproduce analysis from the submitted data

After configuration:

```sh
python3 scripts/analyze.py --results results --output build/submitted-replot/plots --summary build/submitted-replot/summary.json
```

This regenerates all nine figures and their summary without collecting new timings or modifying the submitted files. The summary provides means, sample SD, min/max, repeat counts, profiling shares, and working-set/roofline calculations for checking the report. Rendering can vary with plotting versions.

The submitted `results/`, report, slides and figures describe the original **Intel Core Ultra 9 185H Windows run**, with six P cores, 24 MiB L3, and GCC 13.2 AVX2/FMA. Hardware, compiler, Python packages and measurement conditions are recorded under `results/`. These timings are not predictions for another machine. The longer study supplies the submitted length-sweep plot; the standard sweep supplies thread scaling.

## 3. Collect a new complete run

```sh
python3 scripts/run.py --output build/my-run
python3 tests/validate_local_run.py --run build/my-run
```

The runner performs correctness, inference sweeps, STREAM, history/working-set/compute diagnostics, analysis, plotting, and PDF generation. The validator checks the new run's coverage, metrics, figures and report. Windows can alternatively use `./scripts/run.ps1 -Output build/my-run` after configuration.

| New output | Location |
| --- | --- |
| Raw data, correctness, environment and summary | `build/my-run/results/` |
| Nine figures, PNG and PDF | `build/my-run/output/plots/` |
| Four-page report for this host | `build/my-run/output/pdf/Project_1_Report.pdf` |
| Setup configuration and package versions | `build/system.json`, `build/python-lock.txt` |

New runs are ignored by Git. Keep their full run folder and setup/package records when sharing results. Use a fresh folder for each attempt; `run.py` refuses folders already containing measurements, including failed partial runs. Do not replace submitted evidence with another host's results. The new PDF describes that host and does not regenerate the submitted presentation.

Connect the charger, select performance power mode, close resource-heavy applications, and check available memory before timing. The submitted Windows run verified AC power and Best Performance. Portable runs check available battery telemetry and record Windows power settings; they do not change those settings or verify closure of every background application. Missing battery telemetry is recorded as unknown.

```sh
python3 scripts/run.py --output build/provisional --allow-battery
python3 scripts/run.py --output build/long-run --long-repeats
```

Battery measurements are provisional. The optional two-second study cannot be combined with `--allow-battery`; the full scaling sweep retains its standard timing interval. Sweeps can be slow, especially at 32k prefill or with many workers.

## Build controls and platform limits

```sh
python3 scripts/configure.py --check
python3 scripts/configure.py --threads 2 --jobs 2
python3 scripts/configure.py --local-source /path/to/llama.cpp
```

`--check` only inspects and does not unlock an unconfigured checkout. `--threads` limits benchmark workers; `--jobs` limits compilation parallelism; both must be positive. `--local-source` exports committed ggml at the pinned revision and excludes uncommitted changes; Python/tool dependencies can still require network access. Set both `CC` and `CXX` only to select an already OpenMP-capable custom toolchain.

Windows selects one logical processor per highest-efficiency-class physical core in processor group 0. Linux respects allowed affinity and selects one logical processor per physical core; hybrid CPUs may include P and E cores. The submitted Intel run used Windows P cores only. macOS uses the reported performance-core count, falling back to physical cores, with OS-managed scheduling; affinity pinning is unavailable. All builds disable GPU backends, including Metal.

Rerun configuration after changing host/toolchain, checkout location, allowed Linux CPU set, source/build requirements, or deleting required build outputs. Pass worker limits again. Documentation-only edits do not invalidate setup. Rebuild native binaries on their execution host. Missing/stale/failed configuration errors give the setup command to rerun.

## Measurement protocol and required plots

- **Prefill/decode:** FP32 Q/K `[128,16,T,1]`, V `[128,32,T,1]`, scalar decay/beta, state `[128,128,32,1]`. Prefill lengths 512/4,096/32,768, seeded 128-token chunks, zero initial state. Decode histories 1,024/8,192/32,768; restore starting state per repeat and carry state after every call. Histories end at the same repeated-block phase, so decode work is fixed.
- **Correctness/repeats:** independent FP64 reference over 1,000 changing-input steps and multi-token/recurrent equivalence; absolute error limit 2e-5. Warmup, three repeats, minimum 0.35 s each; longer study minimum 2 s. Report mean and sample SD.
- **Memory:** Windows peak working set, Linux peak RSS converted from KiB to bytes, macOS peak RSS already in bytes. Completed GDN measurements above 4.5 GB RSS or below 1 GB available RAM are rejected. These checks do not continuously enforce a cap or count hard page faults; flag and rerun any swapping measurement.
- **STREAM:** at least 20 million doubles per array, increased to exceed four times reported total L3 capacity; 20 passes and three executions per worker count. Setup checks total capacity for three arrays plus 1 GiB; runtime checks availability for three arrays plus 256 MiB. Unknown L3 remains unknown; STREAM RSS is sampled.
- **Profiling/interpretation:** scoped GDN phase timings exclude graph scheduling and serial carry. Supplementary WSL2 perf captures are in `results/profiles/`; Linux CPU sample shares do not directly measure Windows wall time or hardware cache misses. Logical byte counts/fixed copy footprints do not establish actual cache/DRAM traffic or residency.
- **Own experiment/storage:** rotate 1/4/12/36 independent states at fixed work. State is 2 MiB per layer; analytical Qwen3-Next F16 KV reaches 64 MiB per attention layer at 32,768 tokens. This storage comparison is not F16 model inference.

| Required plot | Figure |
| --- | --- |
| Decode/context and prefill/prompt | [length_sweeps](output/plots/length_sweeps.png) |
| Thread scaling | [thread_scaling](output/plots/thread_scaling.png) |
| Operator time breakdown | [operator_breakdown](output/plots/operator_breakdown.png) |
| Predicted versus measured decode | [roofline](output/plots/roofline.png) |

State, working-set, bandwidth and cache-roofline figures support the required interpretation/own experiment. PNGs support the report; PDF figures are retained because the slides cite them.

## Individual stages and checks

```sh
python3 scripts/run_benchmarks.py --clean-rerun --output build/manual/results
python3 scripts/analyze.py --results build/manual/results --output build/manual/output/plots
python3 scripts/make_report.py --results build/manual/results --output build/manual/output
python3 tests/validate_local_run.py --run build/manual
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Use a fresh folder for manual measurements; only the complete runner checks for an existing run folder. `--stream-only` produces bandwidth evidence alone, not the full report inputs. For low memory or interrupted runs, close resource-heavy applications and retry in a new folder.

Configuration runs CTest automatically. Repeat with `.venv/bin/ctest --test-dir build/cmake --output-on-failure` on Linux/macOS or `.venv/Scripts/ctest.exe --test-dir build/cmake --output-on-failure` on Windows.

Optional Linux sampling needs perf/event permissions. After setup, use `.venv/bin/cmake -S . -B build/cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo`, build with `.venv/bin/cmake --build build/cmake`, then run `bash scripts/profile_linux.sh`. Captures go to `build/perf/` as `decode-1`, `decode-max`, `prefill-1`; `PERF`, `PERF_EVENT`, `PROFILE_OUTPUT`, `PYTHON`, `GDN` provide overrides. Keep sampling separate from throughput runs and rerun configuration to restore Release before timing.

## Shared PARAM Rudra reference

Job 186891 used **Llama-3.2-1B Q4_K_M**, a 512-token prompt, 128-token decode, four threads and three repeats: **92.77 ± 0.16 combined tokens/s**, **1.33 GiB peak RSS**, Xeon Gold 6240R. Source revision, model checksum, hardware, benchmark output and RSS record are retained in `results/param-rudra/186891/`.

To repeat this separate full-model test, provision the pinned external llama.cpp checkout, licensed GGUF, Python 3 and CMake in a CPU allocation, then run from this repository:

```sh
LLAMA_SOURCE=/absolute/path/to/llama.cpp MODEL=/absolute/path/to/Llama-3.2-1B-Instruct-Q4_K_M.gguf THREADS=4 bash scripts/run_shared_reference.sh
```

The wrapper checks the source revision and records the model checksum. Output is `build/shared-reference/JOB_ID/` (`local` outside Slurm); `REFERENCE_OUTPUT` overrides it. The original GCC 8.5 build also passed `-DCMAKE_CXX_STANDARD_LIBRARIES:STRING=-lstdc++fs`. No model weights are committed.

## Sources and repository scope

llama.cpp/ggml is pinned to [8bb3cf0d216a5ae48360179f2931daf284be4d01](https://github.com/ggml-org/llama.cpp/tree/8bb3cf0d216a5ae48360179f2931daf284be4d01), retaining its MIT license in fetched source. Preparation adds an older-Windows-header guard and scoped timers without changing GDN arithmetic. CUDA/Vulkan/Metal/BLAS/Accelerate are disabled; SIMD adapts to the host, with a scalar compute-reference fallback where needed.

`vendor/stream.c` is unmodified [STREAM 5.10](https://www.cs.virginia.edu/stream/FTP/Code/stream.c), retaining its license/publication conditions; `results/source-provenance.json` records its SHA-256. MSVC compatibility headers in `src/compat/msvc/` supply timer/types without changing STREAM kernels. The storage calculation uses the [Qwen3-Next configuration](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json).

Commit the remaining source, scripts, tests, root build files/README, selected results, report, presentation and referenced plots. Local `build/`, `.venv`, fetched dependencies, weights and scratch runs remain ignored. No extra guides, talk script, or bonus files are required.
