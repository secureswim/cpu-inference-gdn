"""Run serial, repeated CPU sweeps and retain every measurement as CSV."""
import argparse
import contextlib
import csv
import ctypes
import json
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
import uuid

import psutil

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build/cmake"
SUFFIX = ".exe" if os.name == "nt" else ""
RESULTS = ROOT / "results"
GDN_COLUMNS = ("mode,length,threads,pinned,repeat,tokens,seconds,tokens_s,peak_rss_bytes,"
               "free_ram_bytes,copy_s,decay_s,delta_s,update_s,readout_s").split(",")
FOOTPRINT_KIB = (128, 512, 1024, 2048, 4096, 8192, 16384, 24576, 49152, 98304, 196608)
BEST_PERFORMANCE = "ded574b5-45a0-4f42-8737-46345c09c238"
DESKTOP_APPS = {"chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "code", "devenv",
                "idea64", "pycharm64", "clion64", "cursor", "windsurf"}


def desktop_apps():
    processes = {proc.pid: Path(name).stem.lower() for proc in psutil.process_iter(["name"])
                 if (name := proc.info["name"]) and Path(name).stem.lower() in DESKTOP_APPS}
    user32 = ctypes.windll.user32
    from ctypes import wintypes
    window_callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    visible = set()

    @window_callback
    def inspect_window(window, _):
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
        if pid.value in processes and user32.IsWindowVisible(window) and user32.GetWindowTextLengthW(window):
            visible.add(processes[pid.value])
        return True

    if not user32.EnumWindows(inspect_window, 0):
        raise ctypes.WinError()
    return sorted(visible), sorted(set(processes.values()) - visible)


def require_clean_conditions(check_background=False):
    if os.name != "nt":
        raise RuntimeError("The clean rerun checks are specific to the assignment laptop")
    hardware = dict(line.split("=", 1) for line in execute([BIN / f"machine{SUFFIX}"]).splitlines())
    overlay = power_overlay()
    if hardware.get("ac_line_status") != "1" or overlay != {"status": 0, "guid": BEST_PERFORMANCE}:
        raise RuntimeError("Connect AC power and select Windows Best Performance")
    apps, background = desktop_apps()
    if apps:
        raise RuntimeError("Close open browser and IDE windows: " + ", ".join(apps))
    if check_background and background:
        raise RuntimeError("End leftover browser and IDE processes: " + ", ".join(background))
    return background


def write_measurement(name, runs):
    header = None
    with (RESULTS / name).open("w", newline="") as file:
        writer = csv.writer(file)
        for command in runs:
            output = execute(command)
            columns, *rows = csv.reader(output.splitlines())
            if header is None:
                header = columns
                writer.writerow(header)
            if columns != header or len(rows) != 3 or any(len(row) != len(header) for row in rows):
                raise RuntimeError(f"Unexpected {name} output: {output}")
            writer.writerows(rows)
            file.flush()
            print(name, *command[1:], flush=True)


def measure_working_sets(cpus):
    endpoints = (1, len(cpus))
    gdn = BIN / f"gdn{SUFFIX}"
    ceilings = BIN / f"ceilings{SUFFIX}"
    write_measurement("history_states.csv", ([gdn, "history", n] for n in endpoints))
    runs = [[gdn, "rotate", n, count, diagnostic]
            for n in endpoints for count in (1, 4, 12, 36) for diagnostic in (0, 1)]
    random.Random(24).shuffle(runs)
    write_measurement("working_sets.csv", runs)
    runs = [[ceilings, "copy", n, kib * 1024] for n in endpoints for kib in FOOTPRINT_KIB]
    runs += [[ceilings, "fma", n] for n in endpoints]
    write_measurement("ceilings.csv", runs)


def execute(args, env=None):
    result = subprocess.run([str(x) for x in args], cwd=ROOT, env=env, text=True,
                            capture_output=True, check=True)
    if result.stderr:
        with (RESULTS / "stderr.log").open("a", encoding="utf-8") as file:
            file.write(result.stderr)
    return result.stdout


def power_overlay():
    if os.name != "nt":
        return None
    guid = (ctypes.c_ubyte * 16)()
    try:
        status = ctypes.WinDLL("powrprof").PowerGetEffectiveOverlayScheme(ctypes.byref(guid))
        return {"status": status, "guid": str(uuid.UUID(bytes_le=bytes(guid)))}
    except (AttributeError, OSError):
        return {"status": "unavailable"}


@contextlib.contextmanager
def affinity(cpus):
    process = psutil.Process()
    previous = process.cpu_affinity()
    process.cpu_affinity(cpus)
    try:
        yield
    finally:
        process.cpu_affinity(previous)


def measure_stream(cpus):
    hardware = execute([BIN / f"machine{SUFFIX}"])
    (RESULTS / "stream_hardware.txt").write_text(hardware)
    cache_bytes = sum(int(line.split(",")[1]) for line in hardware.splitlines() if line.startswith("cache=3,"))
    bytes_per_array = 20000000 * 8
    if bytes_per_array < 4 * cache_bytes:
        raise RuntimeError("Increase STREAM_ARRAY_SIZE so each array exceeds four times the total LLC")
    metadata = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "elements_per_array": 20000000,
                "bytes_per_array": bytes_per_array, "total_array_bytes": bytes_per_array * 3,
                "last_level_cache_bytes": cache_bytes, "p_core_logical_indices": cpus,
                "effective_power_overlay": power_overlay()}
    (RESULTS / "stream_environment.json").write_text(json.dumps(metadata, indent=2))
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
                    if process.returncode or "Solution Validates" not in output or "Array size = 20000000" not in output:
                        raise RuntimeError(output + err)
                (RESULTS / f"stream_t{threads}_r{repeat}.txt").write_text(output)
                for name, best, avg, low, high in re.findall(r"^(Copy|Scale|Add|Triad):\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", output, re.M):
                    writer.writerow([threads, repeat, name, best, avg, low, high, peak])
                file.flush()
            print(f"STREAM threads={threads}", flush=True)


def measure_gdn(configs, clean, minimum_seconds=0.35, filename="gdn_raw.csv", check_background=False):
    with (RESULTS / filename).open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(GDN_COLUMNS)
        order = [(config, None) for config in configs]
        if clean:
            rng = random.Random(2023522)
            order = []
            for repeat in range(3):
                round_configs = configs.copy()
                rng.shuffle(round_configs)
                order.extend((config, repeat) for config in round_configs)
        for (mode, length, threads, pinned), repeat in order:
            if clean:
                require_clean_conditions(check_background)
            command = [BIN / f"gdn{SUFFIX}", "bench", threads, mode, length, pinned]
            output = execute(command + ([1, minimum_seconds] if clean else []))
            rows = list(csv.reader(output.splitlines()))
            if len(rows) != (1 if clean else 3) or any(len(row) != len(GDN_COLUMNS) for row in rows):
                raise RuntimeError(f"Unexpected measurement output {output}")
            for row in rows:
                if float(row[6]) < minimum_seconds or float(row[8]) > 4.5e9 or float(row[9]) < 1e9:
                    raise RuntimeError("Run violates timing or memory requirements")
                if repeat is not None:
                    row[4] = repeat
            writer.writerows(rows)
            file.flush()
            print(f"GDN {mode} length={length} threads={threads} pinned={pinned}", flush=True)


def long_repeats(check_background=False):
    RESULTS.mkdir(parents=True, exist_ok=True)
    background = require_clean_conditions(check_background)
    hardware = dict(line.split("=", 1) for line in execute([BIN / f"machine{SUFFIX}"]).splitlines())
    cores = len([mask for mask in hardware["p_core_masks"].split(",") if mask])
    configs = [("decode", length, n, 1) for n in (1, cores) for length in (1024, 8192, 32768)]
    configs += [("prefill", length, cores, 1) for length in (512, 4096, 32768)]
    configs.append(("prefill", 4096, 1, 1))
    start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    measure_gdn(configs, clean=True, minimum_seconds=2.0, filename="gdn_long_repeats.csv",
                check_background=check_background)
    require_clean_conditions(check_background)
    metadata = {
        "start": start, "end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu": hardware["cpu"], "p_core_count": cores, "repeats": 3, "minimum_seconds": 2.0,
        "effective_power_overlay": power_overlay(), "ac_power_checks_passed": True,
        "browser_and_ide_window_checks_passed": True, "background_browser_ide_processes_at_start": background,
        "background_browser_ide_processes_checked_each_repeat": check_background,
        "run_order_seed": 2023522,
    }
    (RESULTS / "long_repeats_environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


def sweep(allow_battery=False, stream_only=False, clean=False):
    RESULTS.mkdir(parents=True, exist_ok=True)
    background = require_clean_conditions() if clean else []
    hardware = execute([BIN / f"machine{SUFFIX}"])
    (RESULTS / ("stream_hardware.txt" if stream_only else "hardware.txt")).write_text(hardware)
    info = dict(line.split("=", 1) for line in hardware.splitlines())
    if os.name == "nt" and info["ac_line_status"] != "1" and not allow_battery:
        raise SystemExit("Connect charger before timing. --allow-battery labels provisional measurements.")
    masks = [int(value) for value in info["p_core_masks"].split(",") if value]
    cpus = [mask.bit_length() - 1 for mask in masks]
    if stream_only:
        measure_stream(cpus)
        return
    metadata = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "python": sys.version, "os": platform.platform(),
                "power_scheme": execute(["powercfg", "/getactivescheme"]).strip() if os.name == "nt" else "unrecorded",
                "effective_power_overlay": power_overlay(),
                "p_core_logical_indices": cpus, "timings_provisional": info.get("ac_line_status") != "1",
                "background_apps_closed": clean,
                "background_browser_ide_processes_at_start": background,
                "background_note": ("No visible browser or IDE windows. The Codex chat remained open."
                                    if clean else "Background app closure was not verified."),
                "quantization": "FP32 synthetic operator inputs, model quantization and perplexity do not apply to Option 5",
                "gdn_chunk_tokens": 128, "warmup": "One graph call before each configuration",
                "repeats": 3, "minimum_repeat_seconds": 0.35,
                "llama_commit": "8bb3cf0d216a5ae48360179f2931daf284be4d01"}
    (RESULTS / "environment.json").write_text(json.dumps(metadata, indent=2))
    correctness = execute([BIN / f"gdn{SUFFIX}", "check"])
    (RESULTS / "gdn_correctness.json").write_text(correctness, encoding="utf-8")
    configs = [(mode, length, threads, 1)
               for threads in range(1, len(cpus) + 1)
               for mode, lengths in (("prefill", (512, 4096, 32768)), ("decode", (1024, 8192, 32768)))
               for length in lengths]
    configs += [("decode", 1024, len(cpus), 0), ("prefill", 4096, len(cpus), 0),
                ("profile", 1024, 1, 1), ("profile_prefill", 4096, 1, 1)]
    measure_gdn(configs, clean)
    if clean:
        require_clean_conditions()
    measure_stream(cpus)
    if clean:
        require_clean_conditions()
        measure_working_sets(cpus)
        require_clean_conditions()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-battery", action="store_true")
    parser.add_argument("--stream-only", action="store_true")
    parser.add_argument("--clean-rerun", action="store_true",
                        help="Verify laptop conditions, interleave repeats, and collect working-set and ceiling data")
    parser.add_argument("--long-repeats", action="store_true",
                        help="Collect a separate AC-powered two-second timing study")
    parser.add_argument("--check-background-apps", action="store_true",
                        help="For the two-second study, also require browser and IDE background processes to be stopped")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.clean_rerun and (args.allow_battery or args.stream_only):
        parser.error("--clean-rerun requires the full AC-powered sweep")
    if args.long_repeats and (args.clean_rerun or args.allow_battery or args.stream_only):
        parser.error("--long-repeats is a separate AC-powered timing study")
    if args.check_background_apps and not args.long_repeats:
        parser.error("--check-background-apps applies to --long-repeats")
    RESULTS = args.output.resolve()
    if args.long_repeats:
        long_repeats(args.check_background_apps)
    else:
        sweep(args.allow_battery, args.stream_only, args.clean_rerun)
