"""Generate all figures and machine-readable summaries from the raw CSVs."""
import csv
import json
import os
from pathlib import Path
import statistics as stats

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "build/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PLOTS = ROOT / "output/plots"
PLOTS.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 160})
COLORS = ["#176B87", "#E27636", "#5C469C", "#2D936C", "#A63A50"]
MIB = 2**20
GDN_FLOPS = 32 * 128 * (7 * 128 + 1)  # FMA counts as two, excludes expf


def read(name):
    with (RESULTS / name).open() as file:
        return list(csv.DictReader(file))


def group(rows, keys, metric):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(float(row[metric]))
    return {key: dict(mean=stats.mean(values), sd=stats.stdev(values),
                      low=min(values), high=max(values), n=len(values))
            for key, values in grouped.items()}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}.png", bbox_inches="tight")
    fig.savefig(PLOTS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def working_set_analysis(max_threads, logical_bytes, stream_summary):
    rotation = read("working_sets.csv")
    ceilings = read("ceilings.csv")
    endpoints = (1, max_threads)
    counts = (1, 4, 12, 36)
    expected = {(str(n), str(count), str(diag))
                for n in endpoints for count in counts for diag in (0, 1)}
    groups = group(rotation, ("threads", "states", "diagnostic"), "seconds")
    if set(groups) != expected or any(row["n"] != 3 for row in groups.values()):
        raise ValueError("Incomplete rotating-state data")
    for row in rotation:
        steps, states = int(row["steps"]), int(row["states"])
        elapsed, graph, carry = (float(row[key]) for key in ("seconds", "graph_seconds", "carry_seconds"))
        if steps <= 0 or steps % states or elapsed < 0.5 or graph + carry > elapsed:
            raise ValueError("Invalid rotating-state timing")
        if not 0 < int(row["peak_rss_bytes"]) < 4.5e9:
            raise ValueError("Invalid rotating-state peak RSS")
        for field in ("seconds", "graph_seconds", "carry_seconds"):
            row[field] = float(row[field]) * 1e6 / steps
    latency = group(rotation, ("threads", "states", "diagnostic"), "seconds")
    graph = group(rotation, ("threads", "states", "diagnostic"), "graph_seconds")
    carry = group(rotation, ("threads", "states", "diagnostic"), "carry_seconds")
    rates = group(ceilings, ("mode", "threads", "total_buffer_bytes"), "giga_per_second")
    if any(row["n"] != 3 or row["low"] <= 0 for row in rates.values()):
        raise ValueError("Incomplete bandwidth or compute repeats")
    bandwidth = [
        {"threads": int(n), "bytes": int(size), **value}
        for (mode, n, size), value in sorted(rates.items(), key=lambda item: int(item[0][2]))
        if mode == "copy"
    ]
    history = read("history_states.csv")
    if {(int(r["threads"]), int(r["history"])) for r in history} != {
        (n, length) for n in endpoints for length in (1024, 8192, 32768)
    }:
        raise ValueError("Incomplete history snapshots")

    rows = []
    for n in endpoints:
        for count in counts:
            fast, timed = (str(n), str(count), "0"), (str(n), str(count), "1")
            rows.append({"threads": n, "states": count, "latency_us": latency[fast],
                         "diagnostic_us": latency[timed], "graph_us": graph[timed],
                         "carry_us": carry[timed]})

    fig, (ax, parts) = plt.subplots(1, 2, figsize=(9.5, 3.25))
    for index, n in enumerate(endpoints):
        subset = [r for r in rows if r["threads"] == n]
        ax.errorbar(counts, [r["latency_us"]["mean"] for r in subset],
                    yerr=[r["latency_us"]["sd"] for r in subset], marker="o", capsize=3,
                    label="1 P core" if n == 1 else f"{n} P cores", color=COLORS[index])
    ax.set(xscale="log", xticks=counts, xlabel="Independent recurrent states",
           ylabel="Mean µs per layer step", title="Uninstrumented decode")
    ax.set_xticklabels(counts)
    ax.legend(frameon=False)
    subset = [r for r in rows if r["states"] in (1, 36)]
    positions = np.arange(len(subset))
    graph_us = [r["graph_us"]["mean"] for r in subset]
    parts.bar(positions, graph_us, color=COLORS[0], label="Graph")
    parts.bar(positions, [r["carry_us"]["mean"] for r in subset],
              bottom=graph_us, color=COLORS[1], label="Serial driver carry")
    labels = [f'{r["threads"]}P\n{r["states"]} {"state" if r["states"] == 1 else "states"}' for r in subset]
    parts.set(xticks=positions, xticklabels=labels,
              ylabel="Mean µs per layer step", title="Separate diagnostic timings",
              ylim=(0, max(r["graph_us"]["mean"] + r["carry_us"]["mean"] for r in subset) * 1.25))
    parts.legend(frameon=False, fontsize=8)
    save(fig, "working_sets")

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    for index, n in enumerate(endpoints):
        subset = [r for r in bandwidth if r["threads"] == n]
        ax.errorbar([r["bytes"]/MIB for r in subset], [r["mean"] for r in subset],
                    yerr=[r["sd"] for r in subset], marker="o", capsize=2,
                    label="1 P core" if n == 1 else f"{n} P cores", color=COLORS[index])
    ax.axvline(24, color="#777777", linestyle="--", label="Shared L3 capacity")
    ax.set(xscale="log", xlabel="Combined source + destination footprint (MiB)",
           ylabel="Effective copy bandwidth (GB/s)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "cache_bandwidth")

    limits = []
    intensity = GDN_FLOPS / logical_bytes
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.6), sharey=True)
    x = np.geomspace(0.04, 100, 200)
    for ax, n in zip(axes, endpoints):
        compute = rates[("fma", str(n), "0")]["mean"]
        # Select fixed footprints before comparing the measured bandwidths.
        copy_at = lambda mib: min(
            (r for r in bandwidth if r["threads"] == n),
            key=lambda r: abs(r["bytes"] - mib * MIB))["mean"]
        limits.append({"threads": n, "l2_1mib": copy_at(1), "l3_16mib": copy_at(16),
                       "dram_stream": stream_summary[(str(n),)]["mean"] / 1000,
                       "dram_copy_192mib": copy_at(192), "compute_gflops": compute})
        limits_n = limits[-1]
        for index, (label, key) in enumerate((("1 MiB copy (L2-sized)", "l2_1mib"),
                                               ("16 MiB copy (L3-sized)", "l3_16mib"),
                                               ("DRAM STREAM Triad", "dram_stream"))):
            ax.plot(x, np.minimum(x * limits_n[key], compute), label=label, color=COLORS[index])
        ax.axhline(compute, color="#666666", linestyle=":", label="FP32 FMA ceiling")
        for index, count in enumerate((1, 36)):
            point = next(r for r in rows if r["threads"] == n and r["states"] == count)
            performance = GDN_FLOPS / point["latency_us"]["mean"] / 1000
            ax.scatter(intensity, performance, marker=("o", "x")[index], s=38,
                       color=("#222222", "#A63A50")[index], zorder=3, label=f"{count} state(s)")
        ax.set(xscale="log", yscale="log", xlabel="Logical FLOP/byte",
               title="1 P core" if n == 1 else f"{n} P cores")
    axes[0].set_ylabel("GFLOP/s")
    axes[1].legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "cache_roofline")
    return {"measurements": rows, "ceilings": limits, "copy_bandwidth": bandwidth,
            "flops_per_step": GDN_FLOPS, "logical_flops_per_byte": intensity,
            "history": history}


def analyze():
    gdn = read("gdn_raw.csv")
    stream = read("stream_raw.csv")
    summary = group(gdn, ("mode", "length", "threads", "pinned"), "tokens_s")
    threads = sorted({int(row["threads"]) for row in gdn})
    max_thread = max(threads)
    measured = lambda mode, length, t=max_thread, pin=1: summary[(mode, str(length), str(t), str(pin))]
    stream_summary = group([r for r in stream if r["kernel"] == "Triad"], ("threads",), "best_MB_s")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.1))
    for ax, mode, lengths, label in zip(axes, ("decode", "prefill"),
                                      ((1024, 8192, 32768), (512, 4096, 32768)),
                                      ("Starting context (tokens)", "Prompt length (tokens)")):
        for j, t in enumerate((1, max_thread)):
            data = [measured(mode, length, t) for length in lengths]
            ax.errorbar(lengths, [x["mean"] for x in data], yerr=[x["sd"] for x in data],
                        marker="o", capsize=4, label=f"{t} P core{'s' if t > 1 else ''}", color=COLORS[j])
        ax.set(xlabel=label, ylabel="Operator tokens/s", title=mode.capitalize(), xscale="log", xticks=lengths)
        ax.set_xticklabels([f"{x:,}" for x in lengths])
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
    save(fig, "length_sweeps")

    fig, ax = plt.subplots(figsize=(6.2, 3))
    for mode, length, color in (("decode", 1024, COLORS[0]), ("prefill", 4096, COLORS[1])):
        base = measured(mode, length, 1)["mean"]
        ax.errorbar(threads, [measured(mode, length, t)["mean"] / base for t in threads],
                    yerr=[measured(mode, length, t)["sd"] / base for t in threads],
                    marker="o", capsize=3, label=f"{mode}, {length:,} tokens", color=color)
    ax.plot(threads, threads, "--", color="#777777", linewidth=1, label="Ideal linear scaling")
    ax.set(xlabel="Physical performance cores", ylabel="Speedup over one core", xticks=threads)
    ax.legend(frameon=False, fontsize=9)
    save(fig, "thread_scaling")

    phase_keys = ["copy_s", "decay_s", "delta_s", "update_s", "readout_s"]
    phase_names = ["Initial state copy", "Decay", "Delta dot products", "Rank-one update", "Output dot products"]
    phase_data = {}
    fig, ax = plt.subplots(figsize=(6.2, 3))
    x = np.arange(5)
    for j, mode in enumerate(("profile", "profile_prefill")):
        rows = [r for r in gdn if r["mode"] == mode]
        seconds = [sum(float(r[k]) for r in rows) for k in phase_keys]
        percentages = 100 * np.array(seconds) / sum(seconds)
        phase_data[mode] = dict(zip(phase_names, percentages.tolist()))
        ax.bar(x + (j - 0.5) * 0.36, percentages, width=0.36,
               label="Decode" if j == 0 else "Prefill", color=COLORS[j])
    ax.set(xticks=x, ylabel="Share of scoped operator time (%)")
    ax.set_xticklabels(["Initial\ncopy", "Decay", "Delta\ndots", "State\nupdate", "Output\ndots"])
    ax.legend(frameon=False)
    save(fig, "operator_breakdown")

    # The minimum state read/write plus the driver's necessary out-to-in copy.
    logical_bytes = 4 * (4 * 32 * 128 * 128 + 2 * 16 * 128 + 2 * 32 * 128 + 2 * 32)
    predictions = [stream_summary[(str(t),)]["mean"] * 1e6 / logical_bytes for t in threads]
    fig, ax = plt.subplots(figsize=(6.2, 3))
    ax.plot(threads, predictions, "s--", label="STREAM / logical bytes", color=COLORS[1])
    ax.errorbar(threads, [measured("decode", 1024, t)["mean"] for t in threads],
                yerr=[measured("decode", 1024, t)["sd"] for t in threads], marker="o", capsize=3,
                label="Measured GDN + state carry", color=COLORS[0])
    ax.set(xlabel="Physical performance cores", ylabel="Operator tokens/s", xticks=threads)
    ax.legend(frameon=False)
    save(fig, "roofline")

    fig, ax = plt.subplots(figsize=(6.2, 3))
    contexts = np.array([1024, 4096, 8192, 16384, 32768])
    ax.plot(contexts, np.repeat(32 * 128 * 128 * 4 / 2**20, len(contexts)), "o-", label="GDN state, FP32", color=COLORS[0])
    ax.plot(contexts, 2 * contexts * 2 * 256 * 2 / MIB, "s-", label="Qwen3-Next GQA KV, F16", color=COLORS[1])
    ax.set(xlabel="Context tokens", ylabel="Persistent memory per layer (MiB)", xscale="log", yscale="log", xticks=contexts)
    ax.set_xticklabels(["1k", "4k", "8k", "16k", "32k"])
    ax.legend(frameon=False)
    save(fig, "state_vs_kv")

    fig, ax = plt.subplots(figsize=(6.2, 3))
    for j, pin in enumerate((1, 0)):
        data = [measured(mode, length, max_thread, pin) for mode, length in (("decode", 1024), ("prefill", 4096))]
        ax.bar(np.arange(2)+(j-.5)*.36, [d["mean"] for d in data], width=.36,
               yerr=[d["sd"] for d in data], capsize=3,
               label="P cores pinned" if pin else "OS placement", color=COLORS[j])
    ax.set(xticks=[0, 1], xticklabels=["Decode 1k", "Prefill 4k"], ylabel="Operator tokens/s")
    ax.legend(frameon=False)
    save(fig, "affinity_experiment")

    exported = {"gdn": [{**dict(zip(("mode", "length", "threads", "pinned"), key)), **value}
                        for key, value in summary.items()],
                "stream_triad": [{"threads": key[0], **value} for key, value in stream_summary.items()],
                "phase_percent": phase_data, "gdn_logical_bytes": logical_bytes,
                "max_threads": max_thread, "max_gdn_rss": max(float(r["peak_rss_bytes"]) for r in gdn),
                "min_free_ram": min(float(r["free_ram_bytes"]) for r in gdn)}
    supplemental = [RESULTS / name for name in ("history_states.csv", "working_sets.csv", "ceilings.csv")]
    if any(file.exists() for file in supplemental):
        exported["working_set_study"] = working_set_analysis(max_thread, logical_bytes, stream_summary)
    (RESULTS / "summary.json").write_text(json.dumps(exported, indent=2))
    print(f"Summarized {len(gdn)} GDN rows and {len(stream)} STREAM rows.")


if __name__ == "__main__":
    analyze()
