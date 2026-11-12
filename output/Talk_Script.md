# Ten-minute talk script

## Slide 1 - Gated DeltaNet on a laptop CPU

0:00 to 0:25

We measured one Gated DeltaNet layer on a laptop CPU. The experiment asks how much one decode step costs and how its recurrent state compares with the memory used by an attention layer. I will also show the prefill and thread sweeps, explain the bandwidth prediction, and close with the shared PARAM Rudra test.

## Slide 2 - The Gated DeltaNet recurrence

0:25 to 1:30

Read these equations for one value head and one token. M sub t minus one is the incoming state. First we multiply the old state by the exponential of the log-decay gate. The resulting matrix, M bar, predicts the value from the current key. The difference between the actual value and that prediction is the residual. Beta scales that residual. Its outer product with the key adds a rank-one correction to the state. Finally the updated state multiplies the query, and we divide by the square root of the head dimension. The sixteen query and key heads are shared across thirty-two value heads using the value head index modulo sixteen. The state shape is fixed as the sequence grows.

## Slide 3 - Laptop setup and timing protocol

1:30 to 2:15

The laptop has six performance cores and eight efficiency-class cores reported by Windows. We pin one worker per physical performance core and sweep one through six workers. The fused operator runs in a CPU-only ggml build using AVX2 and FMA. Each configuration gets a warmup followed by three repeats. Prefill processes chunks of one hundred twenty-eight tokens, while decode restores its starting history before every repeat. We record elapsed time and peak working set. The laptop is plugged in with Best Performance selected, and desktop applications remained open. We treat that activity as a possible source of variation.

## Slide 4 - Correctness against an FP64 reference

2:15 to 3:10

We checked the same fused ggml operator used in the timing runs. The scalar reference implements the recurrence independently in double precision. Across a thousand recurrent steps, the query, key and value inputs change while the state carries forward. We compare both the output and the state after each step because errors in the state can affect future tokens. The largest absolute differences are about one times ten to the minus seven in state and about one times ten to the minus eight in output. We also compare one graph processing seventeen tokens with seventeen single-token graphs. The comparison returns zero difference for the tested input. These checks take the place of a model-level sanity check for this operator-only option.

## Slide 5 - Decode speed vs. starting context

3:10 to 4:00

Each point measures one-token calls after advancing the state through the context shown on the horizontal axis. The timer includes the driver’s state carry. On one core, the mean remains between about two thousand one hundred and two thousand two hundred tokens per second across the three contexts. At one thousand twenty-four tokens the reciprocal of the mean throughput is 0.47 milliseconds per token. At six cores that same configuration reaches 1,816 ± 304 tokens per second. The six-core mean falls at thirty-two thousand seven hundred sixty-eight tokens even though the state allocation and work per step stay fixed. These runs cannot isolate the cause of the fall.

## Slide 6 - Prefill speed vs. prompt length

4:00 to 4:45

Prefill processes complete prompts of five hundred twelve, four thousand ninety-six and thirty-two thousand seven hundred sixty-eight tokens. The driver uses fixed-size chunks of one hundred twenty-eight tokens and starts each full prompt from zero state. The same seeded input block repeats across chunks, keeping the buffers bounded. At four thousand ninety-six tokens, six P cores reach 16,076 ± 313 tokens per second across three repeats. At the longest prompt the mean is 14,338. That is a lower rate, with considerable repeat variation. The next slide holds prompt length constant to isolate the effect of worker count.

## Slide 7 - Thread scaling from one to six P cores

4:45 to 5:30

Here each series is normalized to its own one-core rate. The decode series starts from a one thousand twenty-four token history, and the prefill series uses a four thousand ninety-six token prompt. At six cores, prefill is 2.81 times its one-core rate. Decode is 0.84 times its one-core rate. The backend divides work across heads. Each single-token call also pays dispatch, synchronization and state carry costs. Prefill spreads fixed call costs over one hundred twenty-eight tokens. These costs help explain the different scaling patterns.

## Slide 8 - Time inside the fused operator

5:30 to 6:30

These are scoped timer measurements from inside the fused operator with one worker. In decode, the three largest shares are initial state copy, decay, and output dot products. The rank-one update is almost tied with output dot products. In prefill the top three are output dot products, decay, and delta dot products. The same initial state copy serves the whole chunk during prefill, but decode performs it on every graph call. That changes which part of the operator dominates. The percentages normalize to the five timed phases and exclude graph scheduling and the separate state carry in the driver. We used uninstrumented runs for throughput. Windows blocked supplemental CPU sampling, so this breakdown uses scoped timers.

## Slide 9 - Predicted vs. measured decode speed

6:30 to 7:35

The roofline check divides STREAM Triad bandwidth by logical bytes moved per decoded token. At six cores, STREAM gives 80.95 gigabytes per second. The logical traffic count includes reading and writing the operator state, the driver’s state carry copy, and the small inputs and output. That is 8.05 mebibytes per token. The resulting prediction is 9,594 tokens per second. At the same core count the measured rate is 1,816. The prediction charges all counted traffic to DRAM. The two mebibyte state can fit in the shared cache, so some of that traffic may be served there. The kernel also revisits state in multiple phases, and dispatch and synchronization take time. Without hardware-counter data we cannot assign a share of the gap to each cause.

## Slide 10 - State memory vs. attention KV cache

7:35 to 8:25

The Gated DeltaNet state holds thirty-two matrices, each one hundred twenty-eight by one hundred twenty-eight FP32 elements. This gives two mebibytes per layer at every context length. For attention we assume sixteen KV heads with the same head width and FP32 keys and values. Each extra token therefore adds sixteen kibibytes per layer, giving sixteen mebibytes at one thousand twenty-four tokens and five hundred twelve mebibytes at thirty-two thousand seven hundred sixty-eight. This chart compares persistent state only. Separately, the largest measured laptop process working set was 22.5 mebibytes. The raw results record peak working set for every configuration, including temporary buffers and runtime overhead.

## Slide 11 - Thread placement experiment

8:25 to 9:15

For the additional experiment we held the worker count and workload fixed and changed the processor placement policy. In the first configuration, workers are restricted to the physical P cores. In the second, the operating system can choose their placement. At one thousand twenty-four context, the pinned decode mean is 4.1 percent higher, a difference smaller than either sample standard deviation. Prefill shows a larger observed gain of 17.5 percent. We ran three repeats per setting, and the configurations ran at different times. Run order and background activity could affect the size of the prefill gain.

## Slide 12 - Shared PARAM Rudra CPU reference

9:15 to 10:00

The course reference used Llama 3.2 1B Instruct Q4 K M on a Xeon Gold CPU with four threads. For a five hundred twelve token prompt followed by one hundred twenty-eight generated tokens, the combined rate was 93.92 tokens per second, with a sample standard deviation of 0.07 across three repeats. Peak RSS is unavailable because the cluster returned zero and the compute node lacked GNU time. The reference is full-model inference. Our Option 5 operator measurements show a fixed two mebibyte recurrent state and stronger thread scaling in prefill than in single-token decode. Raw results, build notes, and scripts are in the repository.
