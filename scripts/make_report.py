"""Generate the four-page Option 5 report from submitted or newly collected data."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from system_config import ROOT, use_project_python
if __name__ == "__main__":
    use_project_python()

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, PageBreak
from pypdf import PdfReader


def make_report(results, out):
    def load(name):
        return json.loads((results / name).read_text(encoding="utf-8-sig"))

    data, env, check = load("summary.json"), load("environment.json"), load("gdn_correctness.json")
    submitted = results == ROOT / "results"
    config = env.get("system_configuration")
    threads = data["max_threads"]
    study = data["working_set_study"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=21,
                              leading=25, spaceAfter=11, textColor=colors.HexColor("#142A43")))
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=9.3, leading=12.3, spaceAfter=8))
    styles.add(ParagraphStyle(name="Caption", fontName="Helvetica", fontSize=8, leading=10,
                              alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallNote", fontName="Helvetica", fontSize=8, leading=10, spaceAfter=7))
    story = []

    def p(text, style="ReportBody"):
        return Paragraph(text, styles[style])

    def title(text):
        story.append(p(text, "ReportTitle"))

    def text(value, style="ReportBody"):
        story.append(p(value, style))

    def plot(name, caption, height=175):
        story.append(Image(str(out / "plots" / f"{name}.png"), width=510, height=height, kind="proportional"))
        story.append(p(caption, "Caption"))

    def table(rows, widths):
        cells = [[p(escape(str(cell)), "SmallNote") for cell in row] for row in rows]
        t = Table(cells, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F5")),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.append(t)

    def rate(mode, length, n, longer=False):
        rows = data["long_repeat_study"]["measurements"] if longer else data["gdn"]
        return next(row for row in rows if (row["mode"], row["length"], row["threads"], row["pinned"]) ==
                    (mode, str(length), str(n), "1"))

    longer = bool(data.get("long_repeat_study"))
    title("Gated DeltaNet decode on a laptop CPU")
    if submitted:
        text("Project 1, Option 5, Group 4<br/>Siddharth Kumar (2023522) and Vaishnavi Srivastava (2023574)")
        records = [["System", "Recorded configuration"],
                   ["CPU / RAM", "Intel Core Ultra 9 185H; 31.44 GiB usable RAM"],
                   ["Core types", "6 P cores, 8 E cores, 2 low-power E cores; 22 logical processors"],
                   ["Caches / SIMD", "P-core L1D 48 KiB, L1I 64 KiB, L2 2 MiB; shared L3 24 MiB; AVX2/FMA"],
                   ["OS / placement", "Windows; one logical processor per P core, 1-6 workers"],
                   ["Build / power", "GCC 13.2, Release -O3; explicit AVX2/FMA, native off; AC / Best Performance"]]
    else:
        hardware = dict(line.split("=", 1) for line in config["hardware"].splitlines() if "=" in line)
        records = [["System", "Detected configuration"],
                   ["OS / architecture", f"{config['system']['platform']} / {config['identity']['architecture']}"],
                   ["CPU / RAM", f"{hardware['cpu']} / {config['ram_bytes'] / 2**30:.2f} GiB"],
                   ["Workers / placement", f"{threads} / {config['affinity']}"],
                   ["Compiler", config["compiler"]["cxx"]],
                   ["Power", f"AC: {env['ac_power']}; provisional: {env['timings_provisional']}"]]
        text(f"Run recorded {escape(env['timestamp'])}. Measurements describe the configured host.")
    table(records, [100, 410])
    text("<b>Setup.</b> CPU-only llama.cpp/ggml at " + escape(env["llama_commit"]) +
         ". GPU backends are disabled. Synthetic FP32 tensors use 16 Q/K heads, 32 value heads and head size 128. "
         "Option 5 isolates the fused recurrence; model quantization and perplexity do not apply.")
    text(f"<b>Correctness.</b> An independent FP64 reference checked {check['steps']:,} changing-input steps. "
         f"Maximum absolute errors: state {check['max_state_abs']:.2e}, output {check['max_output_abs']:.2e}; "
         f"17-token graph versus recurrent calls {check['chunk_abs']:.2e}. Tolerance: 2e-5.")
    text("<b>Protocol.</b> Batch one; warmup and three repeats of at least 0.35 s. Prefill starts from zero state "
         "and uses repeated seeded 128-token chunks. Decode prepares the starting history, then carries state after each step. "
         "Timings include the serial driver state copy; error bars show sample SD. " +
         ("A separate two-second study supplies the length plot and headline values." if longer else ""))
    history_max = max(float(row["max_abs_from_1024"]) for row in study["history"])
    text(f"<b>History control.</b> Histories 1k, 8k and 32k end at the same repeated-block phase; "
         f"maximum state difference from 1k is {history_max:.2e}. Per-step work is identical, "
         "so differences across histories indicate timing variation rather than growing attention work.")
    note = env.get("background_note", "Background app closure was not verified.")
    if longer:
        long_env = load("long_repeats_environment.json")
        background = long_env.get("background_browser_ide_processes_at_start", [])
        if background:
            note += " Two-second study: background processes at startup included " + ", ".join(background) + "."
    text("<b>Measurement limits.</b> " + escape(note) + " Process peak RSS and available RAM are recorded; "
         "page-in/swap activity and thermal throttling were not measured directly. macOS worker placement is OS managed.", "SmallNote")
    story.append(PageBreak())

    title("Throughput and thread scaling")
    one, many = (rate("decode", 1024, n, longer) for n in (1, threads))
    prefill_one, prefill_many = (rate("prefill", 4096, n, longer) for n in (1, threads))
    text(f"At 1,024 tokens of starting history, decode measured <b>{one['mean']:,.0f} +/- {one['sd']:,.0f}</b> "
         f"operator tokens/s with one worker and <b>{many['mean']:,.0f} +/- {many['sd']:,.0f}</b> with {threads}. "
         f"Reciprocals of mean throughput are {1e6 / one['mean']:.0f} and {1e6 / many['mean']:.0f} us per layer step "
         f"({many['mean'] / one['mean']:.2f}x speedup). Prefill at 4k reached "
         f"{prefill_many['mean']:,.0f} +/- {prefill_many['sd']:,.0f} tokens/s, a "
         f"{prefill_many['mean'] / prefill_one['mean']:.2f}x speedup.")
    plot("length_sweeps", "Decode starting-history and prefill prompt-length sweeps. Three repeats per setting.", 210)
    plot("thread_scaling", "Full scaling sweep: standard 0.35-second minimum at every worker count.", 200)
    text("Decode does not show a consistent penalty from increasing prepared history. Prefill amortizes graph execution "
         "and state-copy costs over 128 tokens per call; decode pays them for every token. The limited repeat count "
         "and observed spread restrict the precision of scaling conclusions.")
    story.append(PageBreak())

    title("Operator costs and own experiment")
    plot("operator_breakdown", "One-worker scoped phase shares exclude graph scheduling and serial driver carry.", 155)
    for mode, label in (("profile", "Decode"), ("profile_prefill", "Prefill")):
        top = sorted(data["phase_percent"][mode].items(), key=lambda item: item[1], reverse=True)[:3]
        text(f"<b>{label} top three:</b> " + "; ".join(f"{name.lower()} {share:.1f}%" for name, share in top) + ".")
    plot("working_sets", "Own experiment: fixed work per step, rotating 1/4/12/36 independent states. Three repeats.", 165)
    for n in sorted({1, threads}):
        small, large = (next(row for row in study["measurements"] if row["threads"] == n and row["states"] == count)
                        for count in (1, 36))
        text(f"With {n} worker(s), one versus 36 states took {small['latency_us']['mean']:.0f} +/- "
             f"{small['latency_us']['sd']:.0f} and {large['latency_us']['mean']:.0f} +/- "
             f"{large['latency_us']['sd']:.0f} us/step. Diagnostic graph time was "
             f"{small['graph_us']['mean']:.0f} us and serial carry {small['carry_us']['mean']:.0f} us for one state.")
    text("Thirty-six states occupy 72 MiB. The footprint increase produced no consistent latency penalty resolvable "
         "against the observed spread. This does not prove cache residency or absence of memory stalls. "
         "Compare values within this experiment: its initialization and timing loop differ from the main sweep.")
    sampled = results / "profiles/sampled.json"
    if sampled.exists():
        text(escape(load("profiles/sampled.json")["report_note"]), "SmallNote")
    else:
        text("Scoped timers measure the internal phases of the single fused GDN operator. External Linux sampling can be "
             "collected separately with scripts/profile_linux.sh.", "SmallNote")
    story.append(PageBreak())

    title("Bandwidth, memory and conclusions")
    plot("roofline", "Direct assignment comparison: STREAM / logical bytes versus the standard decode sweep.", 150)
    bandwidth = next(row["mean"] / 1000 for row in data["stream_triad"] if int(row["threads"]) == threads)
    predicted = bandwidth * 1e9 / data["gdn_logical_bytes"]
    measured = rate("decode", 1024, threads)["mean"]
    text(f"STREAM Triad at {threads} workers measured <b>{bandwidth:.2f} GB/s</b>. Logical traffic is "
         f"{data['gdn_logical_bytes']:,} bytes ({data['gdn_logical_bytes'] / 2**20:.2f} MiB) per step, "
         f"including state read/write, the driver copy and small tensors. The prediction is <b>{predicted:,.0f} steps/s</b> "
         f"versus <b>{measured:,.0f}</b> measured ({measured / predicted:.2f}x). "
         f"The recurrence performs {study['flops_per_step'] / 1e6:.2f} MFLOP/step excluding exponentials, "
         f"or {study['logical_flops_per_byte']:.3f} logical FLOP/byte.")
    text("<b>Explaining the gap.</b> STREAM measures large-array bandwidth; logical GDN bytes are not measured DRAM traffic. "
         "Cache reuse, repeated state passes, copies and worker scheduling make the simple estimate imperfect. "
         "Profiling directly identifies copy and runtime costs. Copy/FMA ceilings alone cannot distinguish compute limits, "
         "cache latency or bandwidth bottlenecks. Hardware cache counters were not collected.")
    plot("state_vs_kv", "Analytical per-layer storage; F16 KV is not an F16 model-inference run.", 135)
    text(f"The FP32 recurrence holds 32 x 128 x 128 values: <b>2 MiB/layer</b>. Two attention KV heads of width 256 "
         f"need 2 KiB/token in F16: <b>64 MiB/layer at 32k</b>. Weights, convolution state and buffers are excluded from "
         f"this storage comparison. The main GDN sweep peaked at {data['max_gdn_rss'] / 2**20:.1f} MiB process RSS, "
         f"with at least {data['min_free_ram'] / 2**30:.2f} GiB available RAM.")
    if submitted:
        reference_dir = results / "param-rudra/186891"
        reference, = json.loads((reference_dir / "reference.json").read_text())
        rss = json.loads((reference_dir / "peak-rss.json").read_text())["peak_rss_bytes"]
        text(f"<b>Shared PARAM Rudra reference.</b> Job 186891, Xeon Gold 6240R, four threads: Llama-3.2-1B Instruct "
             f"Q4_K_M, 512 prompt + 128 decode, three repeats; {reference['avg_ts']:.2f} +/- {reference['stddev_ts']:.2f} "
             f"combined tokens/s, {rss / 2**30:.2f} GiB peak RSS.", "SmallNote")
    text("<b>What we learned.</b> GDN keeps state size fixed; decode scales weakly in this driver, while chunked prefill "
         "amortizes call overhead. The measurements support optimizing copies and runtime overhead, without proving "
         "a unique microarchitectural bottleneck.", "SmallNote")
    text('<b>Sources.</b> Pinned <link href="https://github.com/ggml-org/llama.cpp/tree/' + escape(env["llama_commit"]) +
         '">llama.cpp/ggml</link>; <link href="https://www.cs.virginia.edu/stream/">STREAM 5.10</link>; '
         '<link href="https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json">Qwen3-Next configuration</link>. '
         'Raw CSVs, hardware, build records and summary.json accompany the report.', "SmallNote")
    path = out / "pdf/Project_1_Report.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 23, "CPU inference project | Gated DeltaNet | " + ("Group 4" if submitted else "Local measurements"))
        canvas.drawRightString(553, 23, str(doc.page))

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=42, rightMargin=42,
                      topMargin=34, bottomMargin=43, title="Gated DeltaNet CPU inference").build(
                          story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(path).pages)
    if pages != 4:
        raise RuntimeError(f"Report layout requires review: expected four pages, produced {pages}")
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "build/reproduction/results")
    parser.add_argument("--output", type=Path, default=ROOT / "build/reproduction/output")
    args = parser.parse_args()
    make_report(args.results.resolve(), args.output.resolve())
