#!/usr/bin/env bash
set -euo pipefail
# Use a Linux build with debug symbols. PERF can select a WSL-compatible perf binary.
perf_bin="${PERF:-perf}"
gdn="${GDN:-build/cmake/gdn}"
out="${PROFILE_OUTPUT:-build/perf}"
event="${PERF_EVENT:-cpu-clock:u}"
mkdir -p "$out"
uname -r > "$out/kernel.txt"
"$perf_bin" --version > "$out/perf-version.txt"
capture() {
  local name="$1" threads="$2" mode="$3" length="$4"
  "$perf_bin" record -e "$event" -F 499 --call-graph dwarf,8192 -o "$out/$name.data" -- \
    "$gdn" bench "$threads" "$mode" "$length" 0 1 8 > "$out/$name.csv"
  "$perf_bin" report --stdio --no-children --percent-limit 0.5 -i "$out/$name.data" > "$out/$name.txt"
}
capture decode-1 1 decode 1024
capture decode-6 6 decode 1024
capture prefill-1 1 prefill 4096
