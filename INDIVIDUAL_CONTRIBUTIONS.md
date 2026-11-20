# Individual Contributions

Project 1, Option 5 — Gated DeltaNet CPU Inference

Group 4

| Team member | Student ID | Share of overall project work |
| --- | --- | --- |
| Siddharth Kumar | 2023522 | 50% |
| Vaishnavi Srivastava | 2023574 | 50% |

The team reports equal overall contributions across implementation, experiments, reproducibility, analysis, and preparation of the submission.

## Siddharth Kumar

- Developed the C++ driver around the pinned ggml Gated DeltaNet operator, including recurrent state handling, synthetic inputs, and numerical checks against an independent FP64 reference.
- Implemented and collected the laptop prefill, decode, thread-scaling, and STREAM measurements, retaining raw samples and hardware records.
- Developed the history-state comparison, rotating-state experiment, cache-sized bandwidth measurements, and FP32 compute-ceiling benchmark.
- Collected scoped operator timings and supplementary WSL2 perf profiles, and ran the shared PARAM Rudra benchmark with peak-memory measurement.
- Produced the initial analysis, figures, report, and presentation, and revised the interpretation of decode latency, state memory, and thread scaling.

## Vaishnavi Srivastava

- Extended the project setup and build workflow for Windows, Linux, and macOS, including toolchain selection, dependency setup, OpenMP support, and host SIMD configuration.
- Added platform compatibility code and improved CPU topology, memory, and system-configuration handling for reproducible execution across supported machines.
- Developed the portable run workflow and validation for system setup and locally generated results, and strengthened checks on the submitted evidence.
- Updated the benchmark, analysis, and report workflows so new measurements generate their own results and deliverables with the recorded host configuration.
- Revised the report and presentation, expanded the reproduction instructions and requirement coverage in the README, and cleaned the repository for submission.

## Supporting work in the repository

The implementation and experiments are recorded in `src/`, `scripts/`, `results/`, and the submitted figures. The portable setup and validation work is recorded in `scripts/configure.py`, `scripts/system_config.py`, `scripts/run.py`, the platform compatibility code, and `tests/`. The report and presentation are in `output/`.

Representative implementation and experiment commits include `7d105c7` and `b3dfc2e`. Portability, validation, and submission revisions are recorded in `a02e2ed` and `4f263cd`.
