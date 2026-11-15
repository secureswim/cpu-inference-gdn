"""Validate a local/CI reproduction independently of the committed Windows study."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from system_config import use_project_python
use_project_python()
from pypdf import PdfReader

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
folder = parser.parse_args().run.resolve()
results, out = folder / "results", folder / "output"
env = json.loads((results / "environment.json").read_text())
workers = env["worker_count"]


def read(name):
    with (results / name).open(newline="") as file:
        return list(csv.DictReader(file))


def expect(condition, message):
    if not condition:
        raise SystemExit(message)


gdn = read("gdn_raw.csv")
expected = {(mode, str(length), str(n), "1") for n in range(1, workers + 1)
            for mode, lengths in (("decode", (1024, 8192, 32768)), ("prefill", (512, 4096, 32768)))
            for length in lengths}
expected |= {("decode", "1024", str(workers), "0"), ("prefill", "4096", str(workers), "0"),
             ("profile", "1024", "1", "1"), ("profile_prefill", "4096", "1", "1")}
keys = ("mode", "length", "threads", "pinned")
counts = Counter(tuple(row[k] for k in keys) for row in gdn)
expect(set(counts) == expected and set(counts.values()) == {3}, "Incorrect GDN coverage/repeat count")
for row in gdn:
    expect(math.isclose(float(row["tokens_s"]), float(row["tokens"]) / float(row["seconds"]), rel_tol=2e-9), "Inconsistent GDN throughput")
    expect(float(row["seconds"]) >= 0.35 and 0 < int(row["peak_rss_bytes"]) < 4.5e9 and int(row["free_ram_bytes"]) >= 1e9, "Invalid timing or memory measurement")
for key in counts:
    expect({row["repeat"] for row in gdn if tuple(row[k] for k in keys) == key} == {"0", "1", "2"}, "Incorrect repeat IDs")
stream = read("stream_raw.csv")
counts = Counter((row["threads"], row["kernel"]) for row in stream)
expect(set(counts) == {(str(n), kernel) for n in range(1, workers + 1) for kernel in ("Copy", "Scale", "Add", "Triad")} and set(counts.values()) == {3}, "Incorrect STREAM coverage")
stream_env = json.loads((results / "stream_environment.json").read_text())
if stream_env["last_level_cache_bytes"] is not None:
    expect(stream_env["bytes_per_array"] > 4 * stream_env["last_level_cache_bytes"], "STREAM arrays do not exceed 4x L3")
endpoints = sorted({1, workers})
rotation = read("working_sets.csv")
counts = Counter((row["threads"], row["states"], row["diagnostic"]) for row in rotation)
expect(set(counts) == {(str(n), str(states), str(diag)) for n in endpoints for states in (1, 4, 12, 36) for diag in (0, 1)} and set(counts.values()) == {3}, "Incorrect working-set coverage")
correctness = json.loads((results / "gdn_correctness.json").read_text(encoding="utf-8-sig"))
expect(correctness["steps"] == 1000 and all(correctness[key] <= 2e-5 for key in ("max_state_abs", "max_output_abs", "chunk_abs")), "Correctness check failed")
report = PdfReader(out / "pdf/Project_1_Report.pdf")
expect(len(report.pages) == 4, "Report must have four pages")
text = "\n".join(page.extract_text() for page in report.pages)
expect(env["system_configuration"]["identity"]["architecture"] in text, "Report omits this host's architecture")
expect(len(list((out / "plots").glob("*.png"))) == 9, "Expected all nine plots")
print(f"Validated {len(gdn)} GDN rows, {len(stream)} STREAM rows, {len(rotation)} working-set rows, nine plots, and four report pages.")
