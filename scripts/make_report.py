"""Generate a report from this run, without assumptions about the original laptop."""
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
    config = env["system_configuration"]
    threads = data["max_threads"]
    study = data["working_set_study"]
    hardware = dict(line.split("=", 1) for line in config["hardware"].splitlines() if "=" in line)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=21,
                              leading=25, spaceAfter=12, textColor=colors.HexColor("#142A43")))
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=9))
    styles.add(ParagraphStyle(name="Caption", fontName="Helvetica", fontSize=8, leading=10,
                              alignment=TA_CENTER, spaceAfter=10))
    story = []

    def p(text, style="ReportBody"):
        return Paragraph(text, styles[style])

    def title(text):
        story.append(p(text, "ReportTitle"))

    def text(value):
        story.append(p(value))

    def plot(name, caption, height=175):
        story.append(Image(str(out / "plots" / f"{name}.png"), width=510, height=height, kind="proportional"))
        story.append(p(caption, "Caption"))

    def table(rows, widths):
        cells = [[p(escape(str(cell))) for cell in row] for row in rows]
        t = Table(cells, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F5")),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story.append(t)

    def rate(mode, length, n):
        return next(row for row in data["gdn"] if (row["mode"], row["length"], row["threads"], row["pinned"]) ==
                    (mode, str(length), str(n), "1"))

    title("Gated DeltaNet CPU inference")
    text(f"Run recorded {escape(env['timestamp'])}. This report describes measurements from the configured host. "
         "The committed Windows reference measurements are separate evidence.")
    table([["System", "Detected configuration"],
           ["OS / architecture", f"{config['system']['platform']} / {config['identity']['architecture']}"],
           ["CPU / RAM", f"{hardware['cpu']} / {config['ram_bytes'] / 2**30:.2f} GiB"],
           ["Workers / placement", f"{threads} / {config['affinity']}"],
           ["Compiler", config["compiler"]["cxx"]],
           ["Power", f"AC status: {env['ac_power']}; provisional: {env['timings_provisional']}"],
           ["Upstream", env["llama_commit"]]], [110, 400])
    text(f"<b>Correctness.</b> An independent FP64 reference checked {check['steps']:,} changing-input steps. "
         f"Maximum absolute errors: state {check['max_state_abs']:.2e}, output {check['max_output_abs']:.2e}, "
         f"multi-token versus recurrent calls {check['chunk_abs']:.2e}. The executable fails if an error exceeds 2e-5.")
    text("<b>Protocol.</b> Synthetic FP32 Qwen3-Next layer shapes, batch size one; three timed repeats with at least "
         "0.35 seconds per repeat and one warmup. Prefill uses 128-token chunks and starts from zero state; "
         "decode advances a recurrent state through the starting history. All configurations execute serially. "
         "Error bars show sample standard deviation. These are operator benchmarks, rather than full model generation.")
    text("<b>Measurement limits.</b> " + escape(env["background_note"]) + " macOS placement is managed by the OS; "
         "a configured-placement flag does not imply pinning there. STREAM peak RSS is sampled; GDN peak RSS uses OS process accounting. "
         "Memory limits are checked after each GDN measurement, rather than enforced as a continuous process cap.")
    if data.get("long_repeat_study"):
        text("An additional two-second study supplies the length-sweep plot; all worker counts in the scaling plot use the standard 0.35-second minimum.")
    story.append(PageBreak())

    title("Throughput and thread scaling")
    one, many = rate("decode", 1024, 1), rate("decode", 1024, threads)
    text(f"At 1,024 tokens of starting history, mean decode latency was {1e6 / one['mean']:.1f} us with one worker "
         f"and {1e6 / many['mean']:.1f} us with {threads} worker(s), including the driver state carry. "
         f"The throughput ratio is {many['mean'] / one['mean']:.2f}x. Prefill at 4,096 tokens has a "
         f"{rate('prefill', 4096, threads)['mean'] / rate('prefill', 4096, 1)['mean']:.2f}x ratio.")
    plot("length_sweeps", "Decode starting-history and prefill prompt-length sweeps.", 200)
    plot("thread_scaling", "Scaling relative to one worker. Worker placement follows the detected host policy.", 195)
    text("Starting histories end at the same phase of a repeated seeded input block. Their decode work per step is "
         "the same; differences across history lengths should be interpreted alongside run-to-run variation.")
    story.append(PageBreak())

    title("Working sets and operator costs")
    plot("working_sets", "One graph cycles through 1, 4, 12, or 36 independent states; instrumented timings are a separate diagnostic.", 180)
    plot("operator_breakdown", "Scoped phase timings exclude graph scheduling and the serial driver carry.", 160)
    for n in sorted({1, threads}):
        small = next(row for row in study["measurements"] if row["threads"] == n and row["states"] == 1)
        large = next(row for row in study["measurements"] if row["threads"] == n and row["states"] == 36)
        text(f"With {n} worker(s), one versus 36 states measured {small['latency_us']['mean']:.1f} +/- "
             f"{small['latency_us']['sd']:.1f} and {large['latency_us']['mean']:.1f} +/- "
             f"{large['latency_us']['sd']:.1f} us per step. Compare these values within the rotation experiment: "
             "its initialization and loop differ from the main sweep.")
    text(f"The recurrent state is 2 MiB per layer, so 36 states occupy 72 MiB. "
         f"The main sweep reached {data['max_gdn_rss'] / 2**20:.1f} MiB peak process RSS and recorded "
         f"at least {data['min_free_ram'] / 2**30:.2f} GiB available RAM.")
    story.append(PageBreak())

    title("Bandwidth and storage interpretation")
    plot("cache_roofline", "Empirical copy, STREAM Triad and FP32 multiply-add references; FLOP counts include two operations per FMA.", 185)
    plot("state_vs_kv", "Analytical per-layer storage: FP32 recurrent state versus Qwen3-Next F16 attention KV.", 170)
    text(f"The recurrence performs {study['flops_per_step'] / 1e6:.2f} MFLOP per step excluding exponentials. "
         f"Its logical traffic lower bound is {data['gdn_logical_bytes'] / 2**20:.2f} MiB per step, or "
         f"{study['logical_flops_per_byte']:.3f} FLOP/byte. Logical bytes do not measure actual cache or DRAM traffic. "
         "The 1 MiB and 16 MiB copy footprints are fixed reference sizes, rather than claims about this CPU's cache residency. "
         "Compute references use AVX2/FMA, ARM64 NEON FMA, or a portable scalar fallback as available at compilation.")
    text("<b>Sources and evidence.</b> Pinned llama.cpp/ggml revision " + escape(env["llama_commit"]) +
         "; STREAM 5.10 (www.cs.virginia.edu/stream/); Qwen3-Next-80B-A3B-Instruct config.json on Hugging Face. "
         "Raw CSVs, hardware details, correctness output, environment metadata and summary.json accompany this run.")
    path = out / "pdf/Project_1_Report.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(42, 23, "Gated DeltaNet | Local system measurements")
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
