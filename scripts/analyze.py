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


def analyze():
    gdn = read("gdn_raw.csv")
    mamba = read("mamba_raw.csv")
    stream = read("stream_raw.csv")
    summary = group(gdn, ("mode", "length", "threads", "pinned"), "tokens_s")
    threads = sorted({int(row["threads"]) for row in gdn})
    max_thread = max(threads)
    measured = lambda mode, length, t=max_thread, pin=1: summary[(mode, str(length), str(t), str(pin))]
    stream_summary = group([r for r in stream if r["kernel"] == "Triad"], ("threads",), "best_MB_s")
    bonus = group(mamba, ("heads", "variant", "rank", "simd"), "seconds_step")

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
    ax.plot(contexts, 2 * contexts * 16 * 128 * 4 / 2**20, "s-", label="Attention KV, 16 KV heads, FP32", color=COLORS[1])
    ax.set(xlabel="Context tokens", ylabel="Persistent memory per layer (MiB)", xscale="log", yscale="log", xticks=contexts)
    ax.set_xticklabels(["1k", "4k", "8k", "16k", "32k"])
    ax.legend(frameon=False)
    save(fig, "state_vs_kv")

    fig, ax = plt.subplots(figsize=(6.2, 3))
    cases = ((2, 1, "Mamba-2"), (3, 1, "Mamba-3 SISO"), (3, 4, "Mamba-3 MIMO-4"))
    for j, simd in enumerate((0, 1)):
        data = [bonus[("32", str(v), str(r), str(simd))] for v, r, _ in cases]
        ax.bar(np.arange(3) + (j - .5) * .36, [d["mean"] * 1e6 for d in data],
               yerr=[d["sd"] * 1e6 for d in data], capsize=3, width=.36,
               label="Plain C++" if simd == 0 else "AVX2 fused readout", color=COLORS[j])
    ax.set(xticks=range(3), xticklabels=[c[2] for c in cases], ylabel="Microseconds per step", title="Same 2 MiB main state, one P core")
    ax.legend(frameon=False)
    save(fig, "bonus_latency")

    fig, ax = plt.subplots(figsize=(6.2, 3))
    for j, (v, r, label) in enumerate(cases):
        points = [bonus[(str(h), str(v), str(r), "1")] for h in (4, 32, 512)]
        ax.errorbar([0.25, 2, 32], [d["mean"] * 1e6 for d in points],
                    yerr=[d["sd"] * 1e6 for d in points], marker="o", capsize=3, label=label, color=COLORS[j])
    ax.set(xlabel="Main state size (MiB)", ylabel="Microseconds per step", xscale="log", yscale="log", xticks=[0.25, 2, 32])
    ax.set_xticklabels(["0.25", "2", "32"])
    ax.legend(frameon=False)
    save(fig, "bonus_state_sizes")

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
                "mamba": [{**dict(zip(("heads", "variant", "rank", "simd"), key)), **value}
                          for key, value in bonus.items()],
                "stream_triad": [{"threads": key[0], **value} for key, value in stream_summary.items()],
                "phase_percent": phase_data, "gdn_logical_bytes": logical_bytes,
                "max_threads": max_thread, "max_gdn_rss": max(float(r["peak_rss_bytes"]) for r in gdn),
                "min_free_ram": min(float(r["free_ram_bytes"]) for r in gdn),
                "max_mamba_rss": max(float(r["peak_rss_bytes"]) for r in mamba)}
    exported["mimo_siso_ratio_2MiB"] = bonus[("32", "3", "4", "1")]["mean"] / bonus[("32", "3", "1", "1")]["mean"]
    exported["mimo_siso_ratio_32MiB"] = bonus[("512", "3", "4", "1")]["mean"] / bonus[("512", "3", "1", "1")]["mean"]
    (RESULTS / "summary.json").write_text(json.dumps(exported, indent=2))
    print(json.dumps({key: exported[key] for key in ("mimo_siso_ratio_2MiB", "mimo_siso_ratio_32MiB", "max_gdn_rss")}))


if __name__ == "__main__":
    analyze()
