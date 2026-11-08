# Profiling evidence

Scoped phase timings come from the actual upstream GDN CPU implementation. They appear in `../gdn_raw.csv` in `profile` and `profile_prefill` rows. Their percentages and interpretation appear in the report. The instrumentation source is reproducible through `scripts/prepare_ggml.py`.

A supplemental Windows Performance Recorder sampling capture was attempted twice, including under the signed-in user's execution context. Windows rejected the CPU capture with error `0xc5585011`, "Failed to enable the policy to profile system performance." No sampling capture started and no ETL trace was produced. This is an operating-system profiling-policy failure.

VTune and AMD uProf are not installed in the available environment. A separate sampled trace remains outstanding. On a Linux machine where perf is permitted, `scripts/profile_linux.sh` collects prefill and decode profiles of the same operator driver. On Windows, install VTune/uProf and collect CPU hotspots for these commands.

```powershell
build/cmake/gdn.exe bench 1 decode 1024 1
build/cmake/gdn.exe bench 1 prefill 32768 1
```

Keep sampled-profiler runs separate from the raw throughput sweeps because sampling and stack collection can perturb timing.
