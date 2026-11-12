#!/usr/bin/env bash
set -euo pipefail
# Run inside a PARAM Rudra CPU allocation.
: "${LLAMA_SOURCE:?Set LLAMA_SOURCE to the pinned llama.cpp checkout}"
: "${MODEL:?Set MODEL to Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
: "${THREADS:?Set THREADS to the allocated physical CPU cores}"
mkdir -p results/param-rudra
cmake -S "$LLAMA_SOURCE" -B "$LLAMA_SOURCE/build-cpu" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF "$@"
cmake --build "$LLAMA_SOURCE/build-cpu" --target llama-bench -j "$THREADS"
lscpu > results/param-rudra/hardware.txt
sha256sum "$MODEL" > results/param-rudra/model-sha256.txt
git -C "$LLAMA_SOURCE" rev-parse HEAD > results/param-rudra/llama-commit.txt
# Only the combined 512-token prefill plus 128-token generation test.
timer=()
if [[ -x /usr/bin/time ]]; then
  timer=(/usr/bin/time -v)
fi
"${timer[@]}" "$LLAMA_SOURCE/build-cpu/bin/llama-bench" \
  -m "$MODEL" -ngl 0 -t "$THREADS" -p 0 -n 0 -pg 512,128 -r 3 -o json \
  > results/param-rudra/reference.json 2> results/param-rudra/reference-stderr.txt
