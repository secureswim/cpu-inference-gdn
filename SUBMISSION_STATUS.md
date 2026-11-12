# Submission status

## Completed and checked

- [x] Option 5 Gated DeltaNet implemented through the actual pinned ggml CPU operator.
- [x] CPU-only AVX2/FMA build with automatic physical P-core affinity.
- [x] Independent FP64 GDN reference over 1,000 changing-input steps.
- [x] Multi-token GDN equivalence check.
- [x] Scripted repeated sweeps, STREAM, profiling, memory logging, plots, and report generator.

## Measurements and deliverables

- [x] AC power and performance mode confirmed before final timing sweeps.
- [x] Full raw GDN and STREAM benchmark CSVs generated and reviewed.
- [x] Required plots generated from those CSVs.
- [x] Four-page numerical report rendered and checked.
- [x] Talk deck and ten-minute speaker script generated and checked.
- [x] Shared PARAM Rudra reference run completed on the actual course node (job 186881).
- [ ] Separate sampled VTune/uProf or perf trace collected.
- [ ] Course common slide template applied.
- [ ] Team-member names and allocation confirmed.
- [ ] Student records the ten-minute talk in their own voice, as requested.

The controlled measurements ran after the user confirmed the charger and performance mode. Windows reports AC power and an effective Best Performance overlay. Its underlying power scheme still reports Balanced. Desktop applications remained open, which limits control of background interference. GDN single-token decode has visible timing spread and dispatch overhead, documented in the report.

PARAM Rudra job 186881 measured **93.92 ± 0.07 tokens/s** for combined prefill and decode on four CPU threads. Peak RSS is unavailable. Raw evidence is in `results/param-rudra/186881/`.

 Windows rejected supplemental sampling with profiling-policy error `0xc5585011`. Scoped timings from the actual upstream GDN operator are complete. A separate VTune/uProf or perf trace still requires a machine/session where sampling is permitted.
