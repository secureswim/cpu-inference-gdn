"""Audit sweep coverage, repeats, rates, memory, reference checks, and outputs."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from system_config import use_project_python
if __name__ == "__main__":
    use_project_python()

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / "results" / name).open() as file:
        return list(csv.DictReader(file))


gdn = rows("gdn_raw.csv")
environment = json.loads((ROOT / "results/environment.json").read_text())
thread_count = len(environment["p_core_logical_indices"])
expected = {(mode, str(length), str(thread), "1")
            for thread in range(1, thread_count + 1)
            for mode, lengths in (("decode", (1024, 8192, 32768)), ("prefill", (512, 4096, 32768)))
            for length in lengths}
expected |= {("decode", "1024", str(thread_count), "0"), ("prefill", "4096", str(thread_count), "0"),
             ("profile", "1024", "1", "1"), ("profile_prefill", "4096", "1", "1")}
keys = lambda row: tuple(row[key] for key in ("mode", "length", "threads", "pinned"))
counts = Counter(keys(row) for row in gdn)
assert set(counts) == expected and set(counts.values()) == {3}
for key in expected:
    assert {r["repeat"] for r in gdn if keys(r) == key} == {"0", "1", "2"}
for row in gdn:
    assert math.isclose(float(row["tokens_s"]), float(row["tokens"]) / float(row["seconds"]), rel_tol=2e-9)
    assert 0 < float(row["peak_rss_bytes"]) < 4.5e9
    assert float(row["free_ram_bytes"]) > 1e9
    assert float(row["seconds"]) >= 0.35

stream = rows("stream_raw.csv")
stream_environment = json.loads((ROOT / "results/stream_environment.json").read_text())
assert stream_environment["bytes_per_array"] >= 4 * stream_environment["last_level_cache_bytes"] > 0
assert stream_environment["elements_per_array"] == 20000000
stream_counts = Counter((r["threads"], r["kernel"]) for r in stream)
assert set(stream_counts) == {(str(t), k) for t in range(1, thread_count + 1) for k in ("Copy", "Scale", "Add", "Triad")}
assert set(stream_counts.values()) == {3}
for row in stream:
    assert float(row["best_MB_s"]) > 0
    assert float(row["min_seconds"]) <= float(row["average_seconds"]) <= float(row["max_seconds"])
    assert 0 < int(row["peak_rss_bytes"]) < 4.5e9
    raw = (ROOT / f"results/stream_t{row['threads']}_r{row['repeat']}.txt").read_text()
    assert "Solution Validates" in raw
for thread in range(1, thread_count + 1):
    for kernel in ("Copy", "Scale", "Add", "Triad"):
        assert {r["repeat"] for r in stream if r["threads"] == str(thread) and r["kernel"] == kernel} == {"0", "1", "2"}

correctness = json.loads((ROOT / "results/gdn_correctness.json").read_text(encoding="utf-8-sig"))
assert correctness["steps"] == 1000
assert all(0 <= correctness[key] <= 2e-5 for key in ("max_state_abs", "max_output_abs", "chunk_abs"))
provenance = json.loads((ROOT / "results/source-provenance.json").read_text())
assert hashlib.sha256((ROOT / "vendor/stream.c").read_bytes()).hexdigest() == provenance["stream_sha256"]
reference_dir = ROOT / "results/param-rudra/186891"
reference, = json.loads((reference_dir / "reference.json").read_text())
assert reference["n_prompt"] == 512 and reference["n_gen"] == 128 and reference["n_gpu_layers"] == 0
assert reference["model_filename"] == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
assert len(reference["samples_ts"]) == len(reference["samples_ns"]) == 3
assert reference["avg_ts"] > 0 and reference["stddev_ts"] >= 0
assert math.isclose(reference["avg_ts"], statistics.mean(reference["samples_ts"]), rel_tol=1e-6)
assert math.isclose(reference["stddev_ts"], statistics.stdev(reference["samples_ts"]), rel_tol=1e-3)
assert 0 < json.loads((reference_dir / "peak-rss.json").read_text())["peak_rss_bytes"] < 4.5e9
assert (reference_dir / "llama-commit.txt").read_text().strip() == "8bb3cf0d216a5ae48360179f2931daf284be4d01"

assert len(PdfReader(ROOT / "output/pdf/Project_1_Report.pdf").pages) == 4
with ZipFile(ROOT / "output/Project_1_Presentation.pptx") as presentation:
    assert presentation.testzip() is None
    assert any(name.startswith("ppt/slides/slide") and name.endswith(".xml")
               for name in presentation.namelist())
# Recording/narration duration must be checked separately; a timing plan is not a recording.
summary = json.loads((ROOT / "results/summary.json").read_text())
study = summary["working_set_study"]
assert study, "The submitted own experiment is required"
if study:
    endpoints = (1, thread_count)
    rotation = rows("working_sets.csv")
    assert Counter((int(r["threads"]), int(r["states"]), int(r["diagnostic"])) for r in rotation) == {
        (n, count, diagnostic): 3 for n in endpoints for count in (1, 4, 12, 36) for diagnostic in (0, 1)
    }
    for row in rotation:
        assert int(row["steps"]) > 0 and int(row["steps"]) % int(row["states"]) == 0
        graph, carry, elapsed = (float(row[key]) for key in ("graph_seconds", "carry_seconds", "seconds"))
        assert 0 <= graph + carry <= elapsed and elapsed >= 0.5
        assert (graph > 0 and carry > 0) if int(row["diagnostic"]) else graph == carry == 0
        assert 0 < int(row["peak_rss_bytes"]) < 4.5e9
    for n in endpoints:
        ceiling = [r for r in rows("ceilings.csv") if int(r["threads"]) == n]
        assert Counter(row["mode"] for row in ceiling) == {"copy": 33, "fma": 3}
        for row in ceiling:
            assert float(row["seconds"]) >= 0.35
            assert math.isclose(float(row["giga_per_second"]),
                                int(row["work"]) / float(row["seconds"]) / 1e9, rel_tol=2e-8)
    assert len(rows("history_states.csv")) == 6
    assert len(list((ROOT / "output/plots").glob("*.png"))) == 9
else:
    assert len(list((ROOT / "output/plots").glob("*.png"))) == 6
assert (ROOT / "results/gdn_long_repeats.csv").exists(), "Missing submitted length-sweep study"
if (ROOT / "results/gdn_long_repeats.csv").exists():
    follow_up = rows("gdn_long_repeats.csv")
    expected_long = {(mode, str(length), str(n), "1")
                     for mode, n, lengths in (("decode", 1, (1024, 8192, 32768)),
                                               ("decode", thread_count, (1024, 8192, 32768)),
                                               ("prefill", thread_count, (512, 4096, 32768)),
                                               ("prefill", 1, (4096,)))
                     for length in lengths}
    assert Counter(keys(row) for row in follow_up) == dict.fromkeys(expected_long, 3)
    for key in expected_long:
        assert {r["repeat"] for r in follow_up if keys(r) == key} == {"0", "1", "2"}
    for row in follow_up:
        assert float(row["seconds"]) >= 2.0
        assert math.isclose(float(row["tokens_s"]), float(row["tokens"]) / float(row["seconds"]), rel_tol=2e-9)
        assert 0 < float(row["peak_rss_bytes"]) < 4.5e9 and float(row["free_ram_bytes"]) > 1e9
    follow_up_env = json.loads((ROOT / "results/long_repeats_environment.json").read_text())
    assert follow_up_env["ac_power_checks_passed"] and follow_up_env["browser_and_ide_window_checks_passed"]
    if follow_up_env.get("background_browser_ide_processes_checked_each_repeat"):
        assert not follow_up_env["background_browser_ide_processes_at_start"]
required_plots = {"length_sweeps", "thread_scaling", "operator_breakdown", "roofline",
                  "state_vs_kv", "affinity_experiment", "working_sets", "cache_bandwidth", "cache_roofline"}
for name in required_plots:
    for suffix in (".png", ".pdf"):
        assert (ROOT / "output/plots" / (name + suffix)).is_file(), f"Missing plot: {name}{suffix}"

def check_summary(raw, summarized, fields, metric):
    grouped = {}
    for row in raw:
        value = float(row[metric])
        assert math.isfinite(value) and value > 0
        grouped.setdefault(tuple(row[field] for field in fields), []).append(value)
    assert {tuple(row[field] for field in fields) for row in summarized} == set(grouped)
    for row in summarized:
        values = grouped[tuple(row[field] for field in fields)]
        assert row["n"] == len(values)
        for key, value in (("mean", statistics.mean(values)), ("sd", statistics.stdev(values)),
                           ("low", min(values)), ("high", max(values))):
            assert math.isclose(row[key], value, rel_tol=1e-12, abs_tol=1e-12), f"Stale summary {key}"

check_summary(gdn, summary["gdn"], ("mode", "length", "threads", "pinned"), "tokens_s")
check_summary([r for r in stream if r["kernel"] == "Triad"], summary["stream_triad"], ("threads",), "best_MB_s")
check_summary(follow_up, summary["long_repeat_study"]["measurements"], ("mode", "length", "threads", "pinned"), "tokens_s")
print(f"Validated {len(gdn)} GDN and {len(stream)} STREAM rows, working-set and long-repeat studies, "
      "FP64 correctness, STREAM source checksum, PARAM Rudra reference, summary statistics, required figures, four report pages and the presentation file.")
