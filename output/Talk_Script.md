# Ten-minute talk script

Speaker notes for the twelve-slide presentation.

## Slide 1 - Recurrent inference on a laptop CPU

0:00 to 0:30

This project studies the cost of recurrent sequence-model operators on a laptop processor. The main experiment uses the actual Gated DeltaNet operation in llama.cpp. I will explain the correctness checks, measured throughput and memory, the bandwidth prediction, and the shared Rudra result. These are single-layer kernel measurements. Full-model generation has additional work.

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

## Slide 9 - Checking the Gated DeltaNet implementation

5:45 to 6:45

The driver calls the actual upstream fused operator and tests it against an independent FP64 calculation. At each of one thousand steps, the inputs change and the check compares both the output and the carried recurrent state. Checking state matters because a small error may persist and influence subsequent steps. The multi-token check processes seventeen tokens as one graph and compares that result with seventeen single-token calls. The results matched exactly. These checks establish agreement for the tested shape and synthetic inputs. They make it possible to interpret throughput as the cost of the intended operator. The repository records the numerical errors and includes a command to repeat the check before benchmarking.

## Slide 10 - Recurrent state and attention cache

6:45 to 7:30

The recurrent state contains thirty-two matrices, each with one hundred twenty-eight by one hundred twenty-eight FP32 entries. That gives two mebibytes per layer. The comparison uses an attention layer with sixteen key-value heads. Each cached token contributes both a key and a value, so the storage grows with the number of cached tokens. At thirty-two thousand tokens this is five hundred twelve mebibytes per layer. This is a comparison of persistent operator state. Model weights, temporary buffers, and process overhead belong to a separate memory accounting. That distinction explains why the resident set recorded during the experiment exceeds the recurrent state size.

## Slide 11 - What explains the measured CPU cost

7:30 to 9:20

Single-token decode issues one graph call for each token. At that granularity the cost of dispatch and carrying state between calls can matter alongside the arithmetic. Prefill handles chunks of one hundred twenty-eight tokens and spreads those fixed costs across a larger amount of work. That helps explain the stronger scaling we observe for prefill. The roofline prediction assumes that counted bytes travel to DRAM on every step, and STREAM measures sustained sequential memory traffic. The real operator can reuse a portion of its state in cache. It also revisits that state across the five measured phases. In addition, it pays for synchronization and graph overhead. The gap therefore reflects several costs and cannot be assigned to a single cause from this experiment. For the affinity experiment we compared the same worker count and workload with pinning enabled or with operating system placement. The observed difference in decode was small relative to the spread across repeats. The raw runs preserve that variation. Together these experiments show why a fixed-size recurrent state is useful for capacity while the measured CPU time still depends on how the operator is called and scheduled.

## Slide 12 - Shared PARAM Rudra reference

9:20 to 10:00

On PARAM Rudra, four CPU threads ran Llama 3.2 1B Instruct Q4_K_M with a 512-token prompt and a 128-token decode. The combined rate was 93.92 ± 0.07 tokens/s across three repeats. Slurm returned zero for peak RSS, so that measurement is unavailable. This shared reference measures a complete quantized model on a Xeon CPU. Our Option 5 study measures a single FP32 Gated DeltaNet operator on the laptop, so the two throughput values describe different workloads. In the operator study, the recurrent state stayed at two mebibytes across the tested history lengths. Prefill scaled more strongly with additional cores than single-token decode. The repository contains the source revisions, raw samples, correctness checks, and scripts to reproduce the figures.
