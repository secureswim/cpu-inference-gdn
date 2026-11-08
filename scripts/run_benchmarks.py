"""Run serial, repeated CPU sweeps and retain every measurement as CSV."""
import argparse
import contextlib
import csv
import ctypes
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build/cmake"
SUFFIX = ".exe" if os.name == "nt" else ""
RESULTS = ROOT / "results"
GDN_COLUMNS = ("mode,length,threads,pinned,repeat,tokens,seconds,tokens_s,peak_rss_bytes,"
               "free_ram_bytes,copy_s,decay_s,delta_s,update_s,readout_s").split(",")


def execute(args, env=None):
    result = subprocess.run([str(x) for x in args], cwd=ROOT, env=env, text=True,
                            capture_output=True, check=True)
    if result.stderr:
        with (RESULTS / "stderr.log").open("a", encoding="utf-8") as file:
            file.write(result.stderr)
    return result.stdout


@contextlib.contextmanager
def affinity(cpus):
    process = psutil.Process()
    previous = process.cpu_affinity()
    process.cpu_affinity(cpus)
    try:
        yield
    finally:
        process.cpu_affinity(previous)


def sweep(allow_battery=False):
    RESULTS.mkdir(exist_ok=True)
    hardware = execute([BIN / f"machine{SUFFIX}"])
    (RESULTS / "hardware.txt").write_text(hardware)
    info = dict(line.split("=", 1) for line in hardware.splitlines())
    if os.name == "nt" and info["ac_line_status"] != "1" and not allow_battery:
        raise SystemExit("Connect charger before timing. --allow-battery labels provisional measurements.")
    masks = [int(value) for value in info["p_core_masks"].split(",") if value]
    cpus = [mask.bit_length() - 1 for mask in masks]
    metadata = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "python": sys.version, "os": platform.platform(),
                "power_scheme": execute(["powercfg", "/getactivescheme"]).strip() if os.name == "nt" else "unrecorded",
                "p_core_logical_indices": cpus, "timings_provisional": info.get("ac_line_status") != "1",
                "background_apps_closed": False,
                "background_note": "Codex and existing desktop applications remained open. Timing spread captures some interference.",
                "quantization": "FP32 synthetic operator inputs, model quantization and perplexity do not apply to Option 5",
                "gdn_chunk_tokens": 128, "warmup": "One graph call before each configuration",
                "repeats": 3, "minimum_repeat_seconds": 0.35,
                "llama_commit": "8bb3cf0d216a5ae48360179f2931daf284be4d01",
                "mamba_commit": "e9594ce1c732d97440f0332fdc43170a2294dbfa"}
    (RESULTS / "environment.json").write_text(json.dumps(metadata, indent=2))
    execute([BIN / f"gdn{SUFFIX}", "check"])
    configs = [(mode, length, threads, 1)
               for threads in range(1, len(cpus) + 1)
               for mode, lengths in (("prefill", (512, 4096, 32768)), ("decode", (1024, 8192, 32768)))
               for length in lengths]
    configs += [("decode", 1024, len(cpus), 0), ("prefill", 4096, len(cpus), 0),
                ("profile", 1024, 1, 1), ("profile_prefill", 4096, 1, 1)]
    with (RESULTS / "gdn_raw.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(GDN_COLUMNS)
        for mode, length, threads, pinned in configs:
            output = execute([BIN / f"gdn{SUFFIX}", "bench", threads, mode, length, pinned])
            rows = list(csv.reader(output.splitlines()))
            if len(rows) != 3 or any(len(row) != len(GDN_COLUMNS) for row in rows):
                raise RuntimeError(f"Unexpected measurement output {output}")
            for row in rows:
                if float(row[8]) > 4.5e9 or float(row[9]) < 1e9:
                    raise RuntimeError("Run violates memory budget or free-memory threshold")
            writer.writerows(rows)
            file.flush()
            print(f"GDN {mode} length={length} threads={threads} pinned={pinned}", flush=True)
    with (RESULTS / "stream_raw.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["threads", "repeat", "kernel", "best_MB_s", "average_seconds", "min_seconds", "max_seconds", "peak_rss_bytes"])
        for threads in range(1, len(cpus) + 1):
            for repeat in range(3):
                env = {**os.environ, "OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "FALSE"}
                with affinity(cpus):
                    process = subprocess.Popen([str(BIN / f"stream{SUFFIX}")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    peak = 0
                    monitor = psutil.Process(process.pid)
                    while process.poll() is None:
                        try:
                            mem = monitor.memory_info()
                            peak = max(peak, getattr(mem, "peak_wset", mem.rss))
                        except psutil.NoSuchProcess:
                            break
                        time.sleep(0.01)
                    output, err = process.communicate()
                    if process.returncode or "Solution Validates" not in output:
                        raise RuntimeError(output + err)
                (RESULTS / f"stream_t{threads}_r{repeat}.txt").write_text(output)
                for name, best, avg, low, high in re.findall(r"^(Copy|Scale|Add|Triad):\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", output, re.M):
                    writer.writerow([threads, repeat, name, best, avg, low, high, peak])
                file.flush()
            print(f"STREAM threads={threads}", flush=True)
    with (RESULTS / "mamba_raw.csv").open("w", newline="") as file:
        writer = None
        for heads in (4, 32, 512):
            for variant, rank in ((2, 1), (3, 1), (3, 4)):
                for simd in (0, 1):
                    with affinity([cpus[0]]):
                        output = execute([sys.executable, ROOT / "scripts/benchmark_mamba.py", heads, variant, rank, simd])
                    for row in json.loads(output):
                        if writer is None:
                            writer = csv.DictWriter(file, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                    file.flush()
                    print(f"Mamba heads={heads} variant={variant} rank={rank} simd={simd}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-battery", action="store_true")
    sweep(parser.parse_args().allow_battery)
