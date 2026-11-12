# Profiling evidence

Scoped phase timings come from the actual upstream GDN CPU implementation. They appear in `../gdn_raw.csv` in `profile` and `profile_prefill` rows. Their percentages and interpretation appear in the report. The instrumentation source is reproducible through `scripts/prepare_ggml.py`.

Windows Performance Recorder blocked CPU sampling with policy error `0xc5585011`. An independent Ubuntu/WSL2 build of the same pinned operator collected three software-event `perf` profiles using `cpu-clock:u` at a target of 499 Hz and DWARF call stacks. Each workload ran for at least eight seconds. The Linux process used guest scheduling. Its CPU sample percentages do not directly measure Windows wall time or hardware cache misses.

- **One-worker decode:** 41.51% of samples landed at a libc address. Its caller stacks attribute about 20.76% to the driver carry and 20.74% to the initial copy inside ggml. [Decode report](decode-1.txt)
- **Six-worker decode:** 43.54% of CPU samples landed in the OpenMP runtime and 19.41% at the same libc address. The OpenMP percentage does not resolve individual runtime operations. [Decode report](decode-6.txt)
- **One-worker prefill:** 49.72% landed in the fused ggml operator symbol, with additional samples in its vector kernels. [Prefill report](prefill-1.txt)

```bash
cmake -S . -B build/linux-profile -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/linux-profile --target gdn
GDN=build/linux-profile/gdn bash scripts/profile_linux.sh
```

Use `PERF` to select a working perf executable when the distribution's default wrapper expects a different kernel package. The full local sample captures are under `build/wsl-samples/`. Keep sampled-profiler runs separate from throughput sweeps because stack collection can perturb timing.
