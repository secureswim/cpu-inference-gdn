# Ten-minute talk script

## Slide 1 - Gated DeltaNet on a laptop CPU

0:00 to 0:25

At one thousand twenty-four tokens of starting history, one decode step took about 454 microseconds on one performance core and 429 on six. Each step includes the state copy.

## Slide 2 - The Gated DeltaNet recurrence

0:25 to 1:20

Read these equations for one value head and one token. M sub t minus one is the incoming state. First we multiply the old state by the exponential of the log-decay gate. The resulting matrix, M bar, predicts the value from the current key. The difference between the actual value and that prediction is the residual. Beta scales that residual. Its outer product with the key adds a rank-one correction to the state. Finally the updated state multiplies the query, and we divide by the square root of the head dimension. The sixteen query and key heads are shared across thirty-two value heads using the value head index modulo sixteen. The state shape is fixed as the sequence grows.

## Slide 3 - Laptop setup and timing protocol

1:20 to 2:00

The laptop has six performance cores, eight efficiency cores and two low-power efficiency cores. The affinity mask uses one hardware thread per performance core. We sweep one through six workers with a CPU-only AVX2 and FMA build. Each configuration has a warmup and three timed repeats. A seeded order interleaves configurations. Prefill processes chunks of one hundred twenty-eight tokens, and decode restores its starting history before timing. The laptop used AC power and Best Performance with the browser and IDE windows closed.

## Slide 4 - Correctness against an FP64 reference

2:00 to 2:45

We checked the same fused ggml operator used in the timing runs. The scalar reference implements the recurrence independently in double precision. Across a thousand recurrent steps, the query, key and value inputs change while the state carries forward. We compare both the output and the state after each step because errors in the state can affect future tokens. The largest absolute differences are about one times ten to the minus seven in state and about one times ten to the minus eight in output. We also compare one graph processing seventeen tokens with seventeen single-token graphs. The comparison returns zero difference for the tested input. These checks take the place of a model-level sanity check for this operator-only option.

## Slide 5 - Decode speed vs. starting context

2:45 to 3:30

Each point times single-token calls after preparing the context shown. Every history uses the same seeded block of one hundred twenty-eight tokens and ends at the same block boundary. The maximum difference between the prepared states was 5.96e-8. The tensor sizes and work per step stay fixed. At one thousand twenty-four history, the one-core measurement is 454 microseconds per step. Six cores reached 2,330 ± 404 tokens per second.

## Slide 6 - Prefill speed vs. prompt length

3:30 to 4:10

Prefill processes complete prompts of five hundred twelve, four thousand ninety-six and thirty-two thousand seven hundred sixty-eight tokens. The driver starts each full prompt from zero state and uses fixed chunks of one hundred twenty-eight. The same seeded block repeats across chunks. At four thousand ninety-six tokens, six P cores reach 17,607 ± 2,249 tokens per second across three repeats. At the longest prompt the six-core mean is 18,326.

## Slide 7 - Thread scaling from one to six P cores

4:10 to 4:50

The chart shows the full thread sweep, using a zero point three five second minimum at every worker count. Each series is normalized to its one-core rate. At the one- and six-core settings, two-second timing gives speedups of 3.52 for prefill and 1.06 for decode. Prefill spreads call costs across one hundred twenty-eight tokens.

## Slide 8 - Two state copies with different costs

4:50 to 5:45

The chart shows one-worker times inside ggml. The initial copy is parallelized by head. From one to six workers, the whole graph falls from 433 to 412 microseconds, about a five-percent gain. The separate serial driver copy rises from 121 to 202 microseconds, offsetting that gain. So the graph already scales weakly before the driver copy. On the non-NUMA path, workers claim two-head chunks after a barrier. In the WSL2 profile, 43.5 percent of six-worker CPU samples landed in the OpenMP runtime. That is sample share on Linux, while these timings are Windows wall time.

## Slide 9 - Predicted vs. measured decode speed

5:45 to 6:30

The assignment comparison divides STREAM Triad bandwidth by our logical byte count per step. At six cores, STREAM gives 60.05 gigabytes per second. The logical count includes two state copies and the small input and output tensors. That gives 7,116 tokens per second compared with 2,076 measured. The estimate treats the counted bytes as DRAM traffic. A single state can fit in the shared cache, and graph scheduling, arithmetic, and extra state passes also take time.

## Slide 10 - State memory vs. attention KV cache

6:30 to 7:15

The Gated DeltaNet state holds thirty-two matrices of one hundred twenty-eight by one hundred twenty-eight FP32 values. This is two mebibytes per layer. Qwen3-Next full attention uses two key-value heads of width two hundred fifty-six. With F16 keys and values, each token adds two kibibytes per attention layer. At thirty-two thousand seven hundred sixty-eight tokens that gives sixty-four mebibytes. These figures describe persistent recurrence and attention KV storage. Process working set includes buffers and runtime overhead.

## Slide 11 - Own experiment on working-set size

7:15 to 8:20

The added experiment cycles through one, four, twelve and thirty-six independent states using one reusable graph. Each step performs the same arithmetic on the same input shapes. Thirty-six states represent seventy-two mebibytes of recurrent memory, compared with twenty-four mebibytes of shared L3. On one core latency changes from 563 to 535 microseconds per layer step. On six it changes from 595 to 622. One round over thirty-six states takes 19.3 milliseconds on one core and 22.4 on six. The footprint corresponds to the model’s thirty-six recurrent layers. The benchmark isolates the recurrent operation and its state carry. A complete model also has projections, attention, and expert layers.

## Slide 12 - Cache and compute ceilings

8:20 to 9:20

The code performs about 3.67 million floating-point operations per step, excluding exponentials. Dividing by logical bytes gives 0.435 FLOP per byte. Both decode points sit well below the cache-copy references. On one core, thirty-six states imply about sixteen gigabytes per second of logical traffic, compared with about thirty-one and a half in the large-buffer copy test. With no consistent slowdown as the footprint grows, the evidence points to the kernel loops, repeated state passes and two copies as the main costs. The logical byte count cannot tell us how much traffic actually reached DRAM.

## Slide 13 - Shared PARAM Rudra CPU reference

9:20 to 10:00

The PARAM Rudra run used Llama 3.2 1B Instruct Q4 K M on four Xeon Gold CPU threads. It delivered 92.77 combined tokens per second for a five hundred twelve token prompt and one hundred twenty-eight decode tokens. The benchmark process reached 1.33 gibibytes of peak resident memory. The full-model reference has a different scope from our layer study. The laptop results show stable state size and stronger scaling in prefill than in single-token decode.
