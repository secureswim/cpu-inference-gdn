# Ten-minute talk script

The slide timings total 600 seconds. Rehearse with the plots and pauses. Record your own delivery in PowerPoint.

## Slide 1 - Recurrent inference on a laptop CPU

0:00 to 0:30

This project studies the cost of recurrent sequence-model operators on a laptop processor. The main experiment uses the actual Gated DeltaNet operation in llama.cpp. The bonus implements Mamba-3 SISO and rank-four MIMO in C++ and AVX2. I will explain the correctness checks, measured throughput and memory, the bandwidth prediction, and whether MIMO's extra arithmetic has a small cost on this machine. These are single-layer kernel measurements. Full-model generation has additional work.

## Slide 2 - Gated DeltaNet and fixed-size state

0:30 to 1:10

Gated DeltaNet stores a matrix for each value head. At each token it decays that matrix, predicts the current value from the key, and uses a gated correction to update the matrix. It then reads the output using the query. We use the assignment's dimensions, sixteen query and key heads, thirty-two value heads, and head size one hundred and twenty-eight. The resulting main state is exactly two mebibytes. The state summarizes earlier tokens, so its storage has no context-length dimension. The attention comparison instead stores every earlier key and value.

## Slide 3 - Hardware and measurement protocol

1:10 to 1:55

The CPU is an Intel Core Ultra nine, eighteen-five H. Windows reports six performance cores, and the benchmark allows one logical thread per performance core. It excludes the efficiency cores from the main sweep. The machine supports AVX2 and FMA, and the build explicitly selects those instructions. Every configuration gets three repeats, each lasting at least zero point three five seconds after warmup. Long prompts can run longer. We retain raw timings and peak working set. The charger is connected. The recorded base power scheme and background applications remain relevant limits, even with the selected performance mode.

## Slide 4 - Decode and prefill length sweeps

1:55 to 2:45

The left chart shows decode after different amounts of history. Before timing, the driver actually advances the recurrent state through the labelled context length. Every timed repeat restores that starting state. The right chart processes complete synthetic prompts in chunks of one hundred and twenty-eight tokens. The input block repeats across chunks, which makes the experiment reproducible with bounded buffers. Both charts show one and six performance cores. At one thousand twenty-four context tokens, six-core decode reaches 1,816 operator tokens per second. These points describe a recurrent operator whose state allocation stays fixed. The saved plots show sample-standard-deviation error bars, and the CSVs retain each of the three repeats.

## Slide 5 - Scaling across performance cores

2:45 to 3:25

This chart holds the workload constant while changing the number of workers from one to all six physical performance cores. Decode begins at a one-thousand-token context and prefill processes four thousand ninety-six tokens. I normalize each series to its own one-core rate. Decode reaches 0.84 times the one-core rate, while prefill reaches 2.81 times. The operator partitions work across value heads. Parallel workers also incur synchronization and share resources. Departures from ideal scaling need to be interpreted with those costs and the measurement spread.

## Slide 6 - Time inside the upstream operator

3:25 to 4:15

These profiles use scoped timers inserted into the pinned upstream CPU operator. They measure the initial state copy, decay, delta dot products, rank-one update, and final output dot products. The profile uses one worker so its phase percentages are easy to interpret. Prefill copies the initial state once for a chunk and then processes many tokens. Decode copies it for every one-token graph call. This changes the relative importance of copying. The percentages normalize to the five measured phases. They exclude graph scheduling, the driver's state carry copy, and other work outside those scopes. Instrumented runs supply these percentages, while separate uninstrumented runs supply the headline throughput. A sampled VTune or perf trace would provide an additional check.

## Slide 7 - A DRAM-bandwidth prediction

4:15 to 5:05

STREAM uses three arrays totalling four hundred eighty megabytes, which exceeds the twenty-four mebibyte last-level cache. At six workers its mean best Triad bandwidth is 80.95 gigabytes per second. Dividing bandwidth by our logical traffic bound predicts 9,594 tokens per second, compared with 1,816 measured. The key assumption is that every counted byte reaches DRAM on every step. Our main recurrent state is only two mebibytes and can remain in cache, so that assumption may fail. The actual upstream implementation also makes separate passes over state. Those passes create extra cache traffic. This comparison motivates a cache-aware explanation of the measured gap, while hardware counters would be needed to quantify cache misses.

## Slide 8 - Own experiment - thread placement

5:05 to 5:45

For the additional experiment I change only the allowed processor placement. The worker count and operator workload remain the same. One configuration pins execution to the physical performance cores. The other lets the operating system place those workers across the available logical processors. At one-thousand-token decode, the change is 4.1 percent in this run. Prefill provides a second workload. Each point still has three repeats. Small differences should be interpreted against their spread. The configurations ran at different times, so background activity and power behavior remain potential confounders. This experiment supports a local observation about placement rather than a universal claim about affinity.

## Slide 9 - Mamba-3 recurrence and complex state

5:45 to 6:45

Mamba-3 changes the Mamba-2 Euler update in two important mathematical ways before we introduce MIMO. First, exponential-trapezoidal discretization mixes contributions from both the current input and the previous input. The previous contribution carries the decay factor. Omitting it would implement a different recurrence. Second, the complex transition can be represented with real coordinates and accumulated rotary angles. The implementation follows the official pairing conventions, pairwise for SISO and halved for MIMO. Rank four adds four projected input outer products and four readout contributions while keeping the main state matrix dimensions fixed. The previous-key buffer still grows with rank. Dense layer projections and BC normalization sit outside our kernel boundary.

## Slide 10 - Correctness against the official CPU reference

6:45 to 7:30

The numerical reference comes from the official repository. Its surrounding modules import GPU dependencies, so the test extracts the two pure PyTorch function bodies using Python's abstract syntax tree and executes them unchanged on CPU. We compare every output and every recurrent state across one thousand changing-input steps. We also check previous keys, previous inputs, and rotary angles. The cases cover scalar and AVX2 kernels, SISO and rank-four MIMO, state dimensions sixteen, thirty, and one hundred twenty-eight, and four cases at the main benchmark shape. The thirty-dimensional case exercises SIMD tails. The largest state error is 1.20e-6, within the stated tolerance. A separate ggml custom-node check matches direct execution exactly for both ranks.

## Slide 11 - The measured cost of rank-4 MIMO

7:30 to 9:20

The left chart holds the main recurrent state at two mebibytes and uses one performance core. AVX2 Mamba-two takes 355.0 microseconds per step. Mamba-three SISO takes 439.0, and rank-four MIMO takes 1394.1. That makes MIMO 3.18 times the SISO latency at this state size. The right chart repeats the AVX2 comparison with a thirty-two-mebibyte state, larger than the shared last-level cache. Here the ratio is 3.38 times. The scalar comparison shows what explicit SIMD and fused readout achieve for these implementations. The AVX2 loop updates one state vector and immediately uses it for output reductions before moving on. This reduces redundant state loads, but rank four still requires extra multiply-adds and readout accumulators. Dominant main-state arithmetic intensity rises from about one point one two five FLOPs per byte for SISO to three point three seven five for MIMO. Full logical-traffic estimates, including side arrays, appear in the CSV. These estimates exclude transcendental operations. The evidence answers the assignment's MIMO question for this processor, these dimensions, and this kernel implementation. Extrapolating to a GPU or full model would require new measurements.

## Slide 12 - Reproducibility and remaining course steps

9:20 to 10:00

The repository includes the exact source commits, the CPU-only build, the official reference functions, correctness evidence, raw repeated measurements, and the code that produces the figures and report. The peak working set remains within the assignment's budget. The main conclusions concern recurrent operator costs and the measured cost of MIMO on this CPU. The PARAM Rudra shared reference run is deferred until course-node access is available. The course template and student recording also need to be completed. A separate sampled profiler trace would complement the scoped operator breakdown. Those remaining requirements are recorded explicitly so the submission can be finished without confusing missing evidence with a completed experiment.
