# Submission status

## Completed and checked

- [x] Option 5 Gated DeltaNet implemented through the actual pinned ggml CPU operator.
- [x] CPU-only AVX2/FMA build with automatic physical P-core affinity.
- [x] Independent FP64 GDN reference over 1,000 changing-input steps.
- [x] Multi-token GDN equivalence check.
- [x] Mamba-2 Euler, Mamba-3 SISO, and rank-4 MIMO scalar and AVX2 kernels.
- [x] Official pure PyTorch recurrence and rotary references executed on CPU.
- [x] Fourteen correctness cases, each checked over 1,000 steps with stated tolerance.
- [x] Optional ggml graph wrapper, both ranks checked over 1,000 steps.
- [x] Scripted repeated sweeps, STREAM, profiling, memory logging, plots, and report generator.

## Measurements and deliverables

- [ ] AC power and performance mode confirmed before final timing sweeps.
- [ ] Full raw GDN, STREAM, and Mamba benchmark CSVs generated and reviewed.
- [ ] Required plots generated from those CSVs.
- [ ] Four-page numerical report rendered and checked.
- [ ] Talk deck and ten-minute speaker script generated and checked.
- [ ] Shared PARAM Rudra reference run completed on the actual course node.
- [ ] Separate sampled VTune/uProf or perf trace collected.
- [ ] Course common slide template applied.
- [ ] Team-member names and allocation confirmed.
- [ ] Student records the ten-minute talk.

The local machine initially reported battery power and the Balanced power scheme. Correctness work is complete. The numerical submission remains dependent on the controlled measurements and the external course requirements above.
