"""Audit sweep coverage, repeats, rates, memory, reference checks, and outputs."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
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

mamba = rows("mamba_raw.csv")
expected_mamba = {(str(heads), str(variant), str(rank), str(simd))
                  for heads in (4, 32, 512)
                  for variant, rank in ((2, 1), (3, 1), (3, 4)) for simd in (0, 1)}
counts = Counter(tuple(row[k] for k in ("heads", "variant", "rank", "simd")) for row in mamba)
assert set(counts) == expected_mamba and set(counts.values()) == {3}
for row in mamba:
    assert math.isclose(float(row["tokens_s"]) * float(row["seconds_step"]), 1.0, rel_tol=1e-12)
    assert int(row["state_bytes"]) == int(row["heads"]) * 128 * 128 * 4
    assert int(row["persistent_bytes"]) > int(row["state_bytes"])
    assert int(row["peak_rss_bytes"]) < 4.5e9

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

checks = json.loads((ROOT / "results/mamba_correctness.json").read_text())
assert len(checks) == 18 and all(row["steps"] == 1000 for row in checks)
assert all(max(row["max_abs"].values()) < 2e-5 for row in checks)
assert len(PdfReader(ROOT / "output/pdf/Project_1_Report.pdf").pages) == 4
assert sum(row["seconds"] for row in json.loads((ROOT / "results/talk-timings.json").read_text())) == 600
assert len(list((ROOT / "output/plots").glob("*.png"))) == 8
print(f"Validated {len(gdn)} GDN, {len(mamba)} Mamba and {len(stream)} STREAM rows, 18 reference cases, four report pages and 600 seconds of talk timings.")
