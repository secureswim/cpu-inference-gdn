#!/usr/bin/env bash
set -euo pipefail
# Run inside a PARAM Rudra CPU allocation.
: "${LLAMA_SOURCE:?Set LLAMA_SOURCE to the pinned llama.cpp checkout}"
: "${MODEL:?Set MODEL to Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
: "${THREADS:?Set THREADS to the allocated physical CPU cores}"
[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || { echo 'THREADS must be a positive integer' >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "Model file not found: $MODEL" >&2; exit 1; }
expected_commit='8bb3cf0d216a5ae48360179f2931daf284be4d01'
actual_commit=$(git -C "$LLAMA_SOURCE" rev-parse HEAD)
[[ "$actual_commit" == "$expected_commit" ]] || {
  echo "Expected llama.cpp $expected_commit; found $actual_commit" >&2
  exit 1
}
out="${REFERENCE_OUTPUT:-build/shared-reference/${SLURM_JOB_ID:-local}}"
mkdir -p "$out"
cmake -S "$LLAMA_SOURCE" -B "$LLAMA_SOURCE/build-cpu" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF "$@"
cmake --build "$LLAMA_SOURCE/build-cpu" --target llama-bench -j "$THREADS"
lscpu > "$out/hardware.txt"
sha256sum "$MODEL" > "$out/model-sha256.txt"
git -C "$LLAMA_SOURCE" rev-parse HEAD > "$out/llama-commit.txt"
# Only the combined 512-token prefill plus 128-token generation test.
python3 "$(dirname "$0")/measure_peak_rss.py" "$out/peak-rss.json" "$LLAMA_SOURCE/build-cpu/bin/llama-bench" \
  -m "$MODEL" -ngl 0 -t "$THREADS" -p 0 -n 0 -pg 512,128 -r 3 -o json \
  > "$out/reference.json" 2> "$out/reference-stderr.txt"
