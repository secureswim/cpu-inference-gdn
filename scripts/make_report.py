"""Build the four-page report directly from verified measurement summaries."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PLOTS = OUT / "plots"
INK = colors.HexColor("#142A43")
TEAL = colors.HexColor("#176B87")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=9.3, leading=12.3, spaceAfter=7))
styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=22, leading=25, textColor=INK, spaceAfter=10))
styles.add(ParagraphStyle(name="ReportHeading", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=TEAL, spaceBefore=7, spaceAfter=6))
styles.add(ParagraphStyle(name="SmallNote", fontName="Helvetica", fontSize=7.4, leading=9.3, spaceAfter=4))
styles.add(ParagraphStyle(name="Caption", fontName="Helvetica", fontSize=8, leading=10, alignment=TA_CENTER, spaceAfter=7))


def paragraph(text, style="ReportBody"):
    return Paragraph(text, styles[style])


def figure(name, width=510, height=164):
    return Image(str(PLOTS / f"{name}.png"), width=width, height=height, kind="proportional")


def table(rows, widths):
    wrapped = [[paragraph(escape(str(cell)), "SmallNote") for cell in row] for row in rows]
    result = Table(wrapped, colWidths=widths, hAlign="LEFT")
    result.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F5")),
                               ("LINEBELOW", (0, 0), (-1, 0), .5, TEAL),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                               ("TOPPADDING", (0, 0), (-1, -1), 4)]))
    return result


def make_report():
    data = json.loads((ROOT / "results/summary.json").read_text())
    env = json.loads((ROOT / "results/environment.json").read_text())
    gcheck = json.loads((ROOT / "results/gdn_correctness.json").read_text(encoding="utf-8-sig"))
    rudra, = json.loads((ROOT / "results/param-rudra/186881/reference.json").read_text())
    rudra_rerun = ROOT / "results/param-rudra/186891"
    rerun, = json.loads((rudra_rerun / "reference.json").read_text())
    rudra_rss = json.loads((rudra_rerun / "peak-rss.json").read_text())["peak_rss_bytes"]
    threads = data["max_threads"]
    study = data["working_set_study"]
    profile_path = ROOT / "results/profiles/sampled.json"
    sampled = json.loads(profile_path.read_text()) if profile_path.exists() else None

    def gd(mode, length, t=threads, pin=1):
        return next(r for r in data["gdn"] if (r["mode"], r["length"], r["threads"], r["pinned"]) == (mode, str(length), str(t), str(pin)))

    def rotating(n, count):
        return next(r for r in study["measurements"] if r["threads"] == n and r["states"] == count)

    def measurement(row):
        return f"{row['mean']:,.0f} +/- {row['sd']:,.0f}"

    latency1 = 1e6 / gd("decode", 1024, 1)["mean"]
    latency6 = 1e6 / gd("decode", 1024)["mean"]
    six = study["ceilings"][-1]
    bw = six["dram_stream"]
    predicted = bw * 1e9 / data["gdn_logical_bytes"]
    pinned = gd("decode", 1024)["mean"]
    unpinned = gd("decode", 1024, pin=0)["mean"]
    max_history = max(float(row["max_abs_from_1024"]) for row in study["history"])
    max_changed = max(int(row["changed_elements"]) for row in study["history"])
    history_note = ("The resulting snapshots matched bit for bit at both worker counts." if max_changed == 0
                    else f"The largest elementwise difference was {max_history:.2e}, with up to {max_changed:,} elements differing.")
    story = []
    add = story.append
    add(paragraph("Gated DeltaNet decode on a laptop CPU", "ReportTitle"))
    add(paragraph("Project 1, Option 5, Group 4<br/>Siddharth Kumar (2023522) and Vaishnavi Srivastava (2023574)", "ReportHeading"))
    add(paragraph(f"<b>One decode step.</b> At 1,024 tokens of starting history, one Gated DeltaNet layer took <b>{latency1:.0f} µs per token on one P core</b> and <b>{latency6:.0f} µs on {threads} P cores</b>, calculated as the reciprocal of mean throughput. The scope is a batch-one FP32 operator with synthetic Qwen3-Next shapes, including graph execution and the driver state carry."))
    add(paragraph("Hardware and setup", "ReportHeading"))
    add(table([["Hardware / build", "Recorded configuration"],
               ["CPU and memory", "Intel Core Ultra 9 185H, 6 P-cores, 8 E-cores, 2 low-power E-cores. 31.44 GiB usable RAM."],
               ["Caches and SIMD", "P-core L1D 48 KiB, L1I 64 KiB, L2 2 MiB per core. Shared L3 24 MiB. AVX2/FMA available."],
               ["Placement", "Windows affinity mask uses one logical processor per P-core, with 1 to 6 workers."],
               ["Build", "GCC 13.2, Release -O3, CPU-only ggml. Upstream revision 8bb3cf0d."]], [105, 405]))
    add(paragraph("Correctness and measurement protocol", "ReportHeading"))
    add(paragraph(f"The upstream fused operator uses 16 Q/K heads, 32 value heads and a head width of 128. An independent FP64 reference checked 1,000 changing-input steps. Maximum absolute errors were {gcheck['max_state_abs']:.2e} for state and {gcheck['max_output_abs']:.2e} for output. A 17-token graph matched 17 single-token calls exactly. Synthetic operator inputs use FP32."))
    app_note = ("Browser and IDE windows were closed, with no Chrome or Edge processes at startup. The chat remained open."
                if env.get("background_apps_closed") and not env.get("background_browser_ide_processes_at_start")
                else env.get("background_note", "Desktop app closure was not verified."))
    add(paragraph(f"The interleaved sweep ran on AC power in Best Performance mode. {escape(app_note)} Each configuration had a warmup and three repeats of at least 0.35 s. Prefill starts from zero and uses 128-token chunks. Decode advances the state through the starting history before timing. Error bars show sample SD."))
    add(figure("length_sweeps", height=155))
    add(paragraph(f"<b>History control.</b> All lengths end at the same phase of a repeated 128-token block. {history_note} These runs test fixed work per decode step. The earlier six-worker slowdown at 32k did not recur in the clean rerun."))
    add(paragraph("The synthetic update uses normalized keys, beta = 0.3 and a scalar decay multiplier exp(-0.05) = 0.951. The decay shrinks differences inherited from earlier states, while the key-directed correction further damps the component along the key. That explains why the repeated block produces nearly the same state at all three history lengths."))
    add(PageBreak())

    add(paragraph("Thread scaling and operator costs", "ReportTitle"))
    sweep = [["Mode", "Length", "1 P core, tokens/s", f"{threads} P cores, tokens/s"]]
    for mode, lengths in (("decode", (1024, 8192, 32768)), ("prefill", (512, 4096, 32768))):
        for length in lengths:
            sweep.append([mode, f"{length:,}", measurement(gd(mode, length, 1)), measurement(gd(mode, length))])
    add(table(sweep, [62, 66, 191, 191]))
    add(paragraph(f"Six-worker speedup was {pinned/gd('decode',1024,1)['mean']:.2f}x for decode at 1k and {gd('prefill',4096)['mean']/gd('prefill',4096,1)['mean']:.2f}x for prefill at 4k. Prefill shares graph dispatch and initial state copying across a 128-token chunk. Single-token decode pays those costs on each graph call."))
    add(figure("thread_scaling", height=150))
    add(paragraph("Profiling the two state copies", "ReportHeading"))
    descriptions = []
    for mode, label in (("profile", "decode"), ("profile_prefill", "prefill")):
        ordered = sorted(data["phase_percent"][mode].items(), key=lambda x: x[1], reverse=True)[:3]
        descriptions.append(label.capitalize() + " spends its largest measured shares on " + ", ".join(f"{name.lower()} ({value:.1f}%)" for name, value in ordered) + ".")
    add(paragraph(" ".join(descriptions)))
    add(figure("operator_breakdown", height=144))
    c1 = rotating(1, 1)["carry_us"]["mean"]
    c6 = rotating(threads, 1)["carry_us"]["mean"]
    serial_share = c1 / rotating(1, 1)["diagnostic_us"]["mean"]
    add(paragraph(f"The ggml initial copy runs inside worker-owned head ranges. The separate driver copy executes serially after graph completion. With one rotating state, it averaged {c1:.0f} µs on one worker and {c6:.0f} µs on six. It was {serial_share:.0%} of the one-worker diagnostic, implying an idealized Amdahl ceiling of {1/serial_share:.2f}x if this were the only serial cost. The scoped operator percentages exclude the driver copy and graph scheduling."))
    if sampled:
        add(paragraph(escape(sampled["report_note"]), "SmallNote"))
    else:
        add(paragraph("Scoped timing supplies the phase breakdown. Windows blocked WPR sampling with policy error 0xc5585011.", "SmallNote"))
    add(PageBreak())

    add(paragraph("Working-set size and bandwidth ceilings", "ReportTitle"))
    add(paragraph("Each test uses one reusable graph and cycles through 1, 4, 12 or 36 independent recurrent states. It keeps the per-step arithmetic, input tensors and output workspace fixed. The 36-state case models 72 MiB of recurrent state, above the 24 MiB shared L3. It represents the state footprint of Qwen3-Next's 36 GDN layers, while the graph still isolates the recurrence."))
    add(figure("working_sets", height=170))
    a, b = rotating(1, 1)["latency_us"]["mean"], rotating(1, 36)["latency_us"]["mean"]
    c, d = rotating(threads, 1)["latency_us"]["mean"], rotating(threads, 36)["latency_us"]["mean"]
    add(paragraph(f"One to 36 states changed latency from <b>{a:.0f} to {b:.0f} µs</b> on one P core and from <b>{c:.0f} to {d:.0f} µs</b> on six. Latency showed no consistent increase with footprint. The 36-state round took {36*b/1000:.1f} and {36*d/1000:.1f} ms, respectively, including all recurrent updates and their driver copies. Full-model inference also includes projections, attention and MoE computation."))
    add(paragraph("Measured cache-size, DRAM and compute references", "ReportHeading"))
    add(figure("cache_roofline", height=178))
    add(paragraph(f"The scalar-gated recurrence performs about {study['flops_per_step']/1e6:.2f} MFLOP per step, counting FMA as two operations and excluding exponentials. The logical lower bound is {data['gdn_logical_bytes']/2**20:.2f} MiB per step, or {study['logical_flops_per_byte']:.3f} FLOP/byte. It includes the input/output state copy, the driver copy and small tensors. Additional passes raise traffic, and cache residency changes how many bytes reach DRAM. Both operating points use this same logical denominator."))
    add(paragraph(f"At six cores, the 1 MiB and 16 MiB copy footprints sustained {six['l2_1mib']:.1f} and {six['l3_16mib']:.1f} GB/s, STREAM Triad measured {bw:.2f} GB/s and the FMA loop measured {six['compute_gflops']:.0f} GFLOP/s. Cache-sized copy results are empirical reference rates. The plot does not identify which cache served each request."))
    add(PageBreak())

    add(paragraph("Memory, shared reference and conclusions", "ReportTitle"))
    add(paragraph("Qwen3-Next state and F16 attention KV", "ReportHeading"))
    add(figure("state_vs_kv", height=165))
    add(paragraph(f"The recurrent state is 32 x 128 x 128 FP32 values, or <b>2 MiB per layer</b>. Each Qwen3-Next full-attention layer has two KV heads of width 256. F16 keys and values cost 2 KiB per token, or <b>64 MiB per layer at 32,768 tokens</b>. The storage model excludes weights, convolution state and compute buffers. The laptop GDN sweep's peak process working set was {data['max_gdn_rss']/2**20:.1f} MiB, with at least {data['min_free_ram']/2**30:.2f} GiB of free RAM."))
    add(paragraph("Assignment bandwidth comparison", "ReportHeading"))
    add(Table([[figure("roofline", 260, 131), paragraph(f"At six cores, STREAM divided by logical bytes per step estimates {predicted:,.0f} tokens/s. The 1k-history decode measurement is {pinned:,.0f} tokens/s. This comparison assumes the counted traffic reaches DRAM. The cache-sized copy tests and rotating states expose the role of footprint. Timing also includes multiple state passes and graph coordination. Placement changed mean decode throughput by {(pinned/unpinned-1)*100:+.1f}% in the secondary affinity experiment.", "SmallNote")]], colWidths=[265, 245]))
    add(paragraph("Shared PARAM Rudra CPU reference", "ReportHeading"))
    add(paragraph(f"Llama 3.2 1B Instruct Q4_K_M ran on four Xeon Gold 6240R threads with a 512-token prompt and 128-token decode. Job 186881 averaged {rudra['avg_ts']:.2f} +/- {rudra['stddev_ts']:.2f} combined tokens/s. Rerun 186891 averaged <b>{rerun['avg_ts']:.2f} +/- {rerun['stddev_ts']:.2f} tokens/s</b> over three repeats and measured <b>{rudra_rss/2**30:.2f} GiB peak RSS</b> via Linux child process accounting.", "SmallNote"))
    add(paragraph("Reproduction and sources", "ReportHeading"))
    add(paragraph("The repository contains pinned source, raw samples, correctness checks, clean-run scripts and plots. Primary sources are ggml-org/llama.cpp at 8bb3cf0d, Qwen/Qwen3-Next-80B-A3B-Instruct config.json, McCalpin STREAM 5.10 and Intel's Core Ultra 9 185H specifications.", "SmallNote"))
    path = OUT / "pdf/Project_1_Report.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setStrokeColor(TEAL); canvas.setLineWidth(.5); canvas.line(42, 33, 553, 33)
        canvas.setFont("Helvetica", 7.5); canvas.setFillColor(INK)
        canvas.drawString(42, 22, "CPU inference project | Gated DeltaNet")
        canvas.drawRightString(553, 22, str(doc.page))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=34, bottomMargin=43, title="Gated DeltaNet decode on a laptop CPU",
                            author="Siddharth Kumar and Vaishnavi Srivastava")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(path).pages)
    if pages != 4:
        raise RuntimeError(f"Expected four pages, produced {pages}")
    print(path)


if __name__ == "__main__":
    make_report()
