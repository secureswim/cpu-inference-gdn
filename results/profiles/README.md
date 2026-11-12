# Profiling evidence

Scoped phase timings come from the actual upstream GDN CPU implementation. They appear in `../gdn_raw.csv` in `profile` and `profile_prefill` rows. Their percentages and interpretation appear in the report. The instrumentation source is reproducible through `scripts/prepare_ggml.py`.

Windows Performance Recorder blocked CPU sampling with policy error `0xc5585011`, "Failed to enable the policy to profile system performance." The measured profiles use scoped timers.

On Linux with perf permissions, `scripts/profile_linux.sh` collects prefill and decode samples from the same operator driver. On Windows with a compatible sampling profiler, use these workloads.

```powershell
build/cmake/gdn.exe bench 1 decode 1024 1
build/cmake/gdn.exe bench 1 prefill 32768 1
```

Keep sampled-profiler runs separate from the raw throughput sweeps because sampling and stack collection can perturb timing.
