#!/bin/bash
#SBATCH --job-name=p1-llama-rss
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=rudra-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
out="results/$SLURM_JOB_ID"
model="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
: "${LLAMA_COMMIT:?Set LLAMA_COMMIT to the source revision used to build llama-bench}"
mkdir -p "$out"
hostname > "$out/hardware.txt"
lscpu >> "$out/hardware.txt"
sha256sum "$model" > "$out/model-sha256.txt"
printf '%s\n' "$LLAMA_COMMIT" > "$out/llama-commit.txt"
srun --cpu-bind=verbose python3 ./measure_peak_rss.py "$out/peak-rss.json" ./build-cpu/bin/llama-bench \
  -m "$model" -ngl 0 -t "$SLURM_CPUS_PER_TASK" -p 0 -n 0 -pg 512,128 -r 3 -o json \
  > "$out/reference.json" 2> "$out/reference-stderr.txt"
