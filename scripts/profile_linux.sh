#!/usr/bin/env bash
set -euo pipefail
# Supplemental sampled profile. Scoped upstream timers run on every platform.
mkdir -p results/perf
perf record -g -o results/perf/decode.data -- build/cmake/gdn bench 1 decode 1024 1
perf report --stdio -i results/perf/decode.data > results/perf/decode.txt
perf record -g -o results/perf/prefill.data -- build/cmake/gdn bench 1 prefill 32768 1
perf report --stdio -i results/perf/prefill.data > results/perf/prefill.txt
