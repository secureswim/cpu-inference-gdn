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
    checks = json.loads((ROOT / "results/mamba_correctness.json").read_text())
    gcheck = json.loads((ROOT / "results/gdn_correctness.json").read_text(encoding="utf-8-sig"))
    threads = data["max_threads"]
    def gd(mode, length, t=threads, pin=1):
        return next(r for r in data["gdn"] if (r["mode"], r["length"], r["threads"], r["pinned"]) == (mode, str(length), str(t), str(pin)))
    def mb(heads, variant, rank, simd=1):
        return next(r for r in data["mamba"] if (r["heads"], r["variant"], r["rank"], r["simd"]) == (str(heads), str(variant), str(rank), str(simd)))
    def measurement(row):
        return f"{row['mean']:,.0f} +/- {row['sd']:,.0f}"
    bw = next(r["mean"] for r in data["stream_triad"] if r["threads"] == str(threads)) / 1000
    predicted = bw * 1e9 / data["gdn_logical_bytes"]
    pinned = gd("decode", 1024)["mean"]
    unpinned = gd("decode", 1024, pin=0)["mean"]
    mimo = mb(32, 3, 4)["mean"] * 1e6
    siso = mb(32, 3, 1)["mean"] * 1e6
    m2 = mb(32, 2, 1)["mean"] * 1e6
    state_error = max(r["max_abs"]["state"] for r in checks)
    output_error = max(r["max_abs"]["output"] for r in checks)
    provisional = env["timings_provisional"]
    story = []
    add = story.append
    add(paragraph("Recurrent inference on a laptop CPU", "ReportTitle"))
    add(paragraph("Option 5 - Gated DeltaNet operator study with Mamba-3 SISO / MIMO bonus<br/>Siddharth Kumar", "ReportHeading"))
    if provisional:
        add(paragraph("<b>Provisional timing run.</b> The laptop ran on battery. Repeat the sweep on AC power in performance mode before final assessment."))
    add(paragraph("<b>Research question.</b> What does one Gated DeltaNet decode step cost, and how does its recurrent state compare with attention's KV cache? The bonus asks whether Mamba-3's rank-4 MIMO update adds little CPU latency at a fixed main-state size. We study batch-one recurrence kernels with synthetic inputs. Operator tokens/s describes one layer, so it cannot establish full-model throughput or language quality."))
    add(table([["Hardware / build", "Recorded configuration"],
               ["CPU and memory", "Intel Core Ultra 9 185H. 6 P cores / 12 P logical threads and 10 lower-efficiency cores. 31.44 GiB usable RAM."],
               ["Caches and SIMD", "P-core L1D 48 KiB, L1I 64 KiB, L2 2 MiB per core. Shared L3 24 MiB. AVX2/FMA available, AVX-512 unavailable."],
               ["Placement", "One logical thread per P core. Logical processors 0, 10, 12, 14, 16, 18. Scaling covers 1 through 6 workers."],
               ["Software", "Windows, MinGW GCC 13.2, Release -O3. CPU-only ggml, explicit AVX2/FMA, OpenMP. Exact commits and environment are in results/environment.json."]], [107, 403]))
    add(paragraph("Setup and correctness", "ReportHeading"))
    add(paragraph(f"We call the upstream fused GATED_DELTA_NET operator [1] with 16 query/key heads, 32 value heads, and head size 128. Inputs and state use FP32, as this option has no model weights to quantize. Perplexity is inapplicable to synthetic tensors. The independent FP64 calculation checked every state and output over 1,000 changing-input steps. Maximum absolute errors were {gcheck['max_state_abs']:.2e} for state and {gcheck['max_output_abs']:.2e} for output. A 17-token graph matched repeated single-token calls exactly."))
    add(paragraph("Each configuration has one warmup and three timed repeats of at least 0.35 s. Prefill processes 512, 4,096, or 32,768 tokens in chunks of 128. The same fixed seeded input block repeats across chunks. Decode begins from a state advanced through 1,024, 8,192, or 32,768 tokens. State carry between ggml calls belongs to the timed region. Error bars show sample standard deviation across three repeats."))
    add(figure("length_sweeps", height=155))
    add(PageBreak())

    add(paragraph("Measured throughput and operator costs", "ReportTitle"))
    rows = [["Mode", "Length", "1 P core, tokens/s", f"{threads} P cores, tokens/s"]]
    for mode, lengths in (("decode", (1024, 8192, 32768)), ("prefill", (512, 4096, 32768))):
        for length in lengths:
            rows.append([mode, f"{length:,}", measurement(gd(mode, length, 1)), measurement(gd(mode, length))])
    add(table(rows, [62, 66, 191, 191]))
    add(paragraph(f"The {threads}-worker decode speedup at 1k context is {gd('decode',1024)['mean']/gd('decode',1024,1)['mean']:.2f}x. The corresponding 4k prefill speedup is {gd('prefill',4096)['mean']/gd('prefill',4096,1)['mean']:.2f}x. Chunked prefill amortizes graph dispatch and initial state copying over 128 tokens. Single-token decode pays those costs on each call. One-core decode varies little across history lengths. The six-worker 32k decline warrants a controlled rerun because recurrent storage and arithmetic have no context axis. These measurements cannot isolate scheduling, power changes, or thermal effects."))
    add(figure("thread_scaling", height=171))
    add(paragraph("Scoped profiling inside the upstream operator", "ReportHeading"))
    descriptions = []
    for mode, label in (("profile", "decode"), ("profile_prefill", "prefill")):
        ordered = sorted(data["phase_percent"][mode].items(), key=lambda x: x[1], reverse=True)[:3]
        descriptions.append(label.capitalize() + " spends its largest measured shares on " + ", ".join(f"{name.lower()} ({value:.1f}%)" for name, value in ordered) + ".")
    add(paragraph(" ".join(descriptions)))
    add(figure("operator_breakdown", height=162))
    add(paragraph("The plot uses timers inserted into the pinned ggml implementation and a single worker. Shares normalize to the five scoped phases. Graph scheduling, the driver's output-to-input state copy, and profiling overhead sit outside those shares. Instrumented runs have their own rows and never supply the headline throughput. A separate perf script supports sampled profiling on Linux. A VTune/uProf sampled trace on this Windows machine remains outstanding.", "SmallNote"))
    add(PageBreak())

    add(paragraph("Bandwidth prediction and an affinity experiment", "ReportTitle"))
    add(paragraph(f"The official STREAM 5.10 source [2] uses three arrays of 20 million doubles, totaling 480 MB and exceeding the 24 MiB last-level cache. Each of three independent executions has 20 passes. We average their best Triad rates. At {threads} P cores the result is {bw:.2f} GB/s. These are local measurements, without certification by the STREAM maintainers."))
    add(paragraph(f"Our logical traffic lower bound is {data['gdn_logical_bytes']/2**20:.3f} MiB per decoded token. It counts a state read/write, the driver's state carry copy, Q/K/V, gates, and output. STREAM bandwidth divided by this traffic predicts {predicted:,.0f} tokens/s, versus {pinned:,.0f} measured at 1k context ({pinned/predicted:.2f}x the prediction). This DRAM estimate assumes the counted bytes reach memory on every step. The 2 MiB state fits in cache, so that assumption can fail. Conversely, upstream GDN revisits state for decay, delta, update, and readout, creating additional cache traffic. A single DRAM-bandwidth estimate cannot model these cache levels or thread synchronization."))
    add(Table([[figure("roofline", 250, 150), figure("affinity_experiment", 250, 150)]], colWidths=[255, 255]))
    add(paragraph(f"<b>Own experiment.</b> We hold workload and worker count constant and compare P-core affinity with unrestricted OS placement. At 1k decode, the rates are {pinned:,.0f} and {unpinned:,.0f} tokens/s, a {(pinned/unpinned-1)*100:+.1f}% difference. This is smaller than the decode variation across repeats, so the runs do not resolve a pinning advantage for decode. Prefill's local difference is {(gd('prefill',4096)['mean']/gd('prefill',4096,pin=0)['mean']-1)*100:+.1f}%. Run order and background activity remain potential confounders."))
    add(paragraph("State memory and measurement limits", "ReportHeading"))
    add(figure("state_vs_kv", height=168))
    add(paragraph(f"GDN stores 32 x 128 x 128 FP32 values, exactly 2 MiB per layer. An attention layer with 16 KV heads of size 128 stores 2 x T x 16 x 128 x 4 bytes, reaching 512 MiB at 32k tokens. The plot is an analytical storage comparison. It excludes model weights, attention compute buffers, and GDN input/output buffers. Peak GDN process working set was {data['max_gdn_rss']/2**20:.1f} MiB and minimum observed system free RAM was {data['min_free_ram']/2**30:.2f} GiB. Peak working set includes process overhead. Memory checks kept every run below the 4.5 GB budget, but working-set sampling does not prove the absence of all hard page faults."))
    add(paragraph("The charger was connected and the effective Windows power overlay was Best Performance. The base power scheme reports Balanced. The separate power-confirmation record verifies the overlay. Desktop applications remained open. Raw files retain timestamps, free memory and per-configuration peaks. Windows blocked supplemental WPR sampling with policy error 0xc5585011. The shared PARAM Rudra result is deferred by request.", "SmallNote"))
    add(PageBreak())

    add(paragraph("Bonus - Mamba-3 decode on CPU", "ReportTitle"))
    add(paragraph("The implementation follows exponential-trapezoidal discretization [3]. In rotated coordinates, alpha = exp(A x DT), beta = (1 - lambda) x DT x alpha, and gamma = lambda x DT. The state update combines alpha times the old state, gamma times the current projected input outer product, and beta times the previous projected input outer product. Accumulated rotary angles implement the complex-valued transition using real coordinates. SISO uses pairwise rotation. MIMO uses the official halved pairing and sums rank-4 input and output contributions [4]."))
    add(paragraph(f"The test executes the unmodified official PyTorch recurrence and rotary reference functions on CPU via AST extraction. It checks every state, output, previous key, previous input, and angle for 1,000 changing-input steps per case. Sixteen Mamba-3 cases cover ranks 1/4, scalar/AVX2, and state dimensions 16/30/128, including SIMD tails and four cases at the 32-head benchmark shape. Two Euler cases verify Mamba-2. Tolerance is atol 2e-5 and rtol 2e-4. Across all cases, maximum state error is {state_error:.2e} and output error is {output_error:.2e}. The ggml custom-node stretch also matches direct execution exactly across 1,000 steps for both ranks."))
    add(figure("bonus_latency", height=165))
    add(table([["AVX2, same 2 MiB main state", "Mean step time", "Main-state arithmetic intensity"],
               ["Mamba-2 Euler", f"{m2:.1f} us", "~5 / 8 = 0.625 FLOP/byte"],
               ["Mamba-3 SISO", f"{siso:.1f} us", "~9 / 8 = 1.125 FLOP/byte"],
               ["Mamba-3 MIMO rank 4", f"{mimo:.1f} us", "~27 / 8 = 3.375 FLOP/byte"]], [215, 105, 190]))
    add(paragraph(f"<b>Does MIMO's extra compute come at little CPU cost?</b> At 32 heads and a 2 MiB main state, AVX2 MIMO takes {mimo:.1f} us versus {siso:.1f} us for SISO, or {data['mimo_siso_ratio_2MiB']:.2f}x the latency. With 512 heads and a 32 MiB main state, the ratio is {data['mimo_siso_ratio_32MiB']:.2f}x. These measured ratios answer the question for this CPU and kernel implementation. SIMD fuses state update and readout to reuse each updated vector in registers. It still performs extra rank work. The main recurrent matrix size stays fixed, while carried previous keys increase with rank. Raw CSVs include total persistent bytes and arithmetic intensity including side arrays. FLOP estimates omit exp/tanh/SiLU and preprocessing costs."))
    add(paragraph("The benchmark covers rotary preprocessing, recurrence, skip, SiLU gating, and MIMO output mixing. Dense layer projections and BC normalization are outside this kernel study. Random inputs establish numerical consistency and kernel cost. They do not support model perplexity claims. Public Mamba-3 checkpoints exist [5], but this experiment does not load them.", "SmallNote"))
    add(paragraph("References and remaining submission requirements", "ReportHeading"))
    add(paragraph("[1] ggml-org/llama.cpp, commit 8bb3cf0d, ggml/src/ggml-cpu/ops.cpp. github.com/ggml-org/llama.cpp<br/>[2] McCalpin, STREAM 5.10. cs.virginia.edu/stream/<br/>[3] Lahoti et al., Mamba-3, ICLR 2026. arxiv.org/abs/2603.15569<br/>[4] state-spaces/mamba, commit e9594ce1, official recurrence and rotary references. github.com/state-spaces/mamba<br/>[5] Official checkpoint. huggingface.co/state-spaces/mamba3-mimo-1.5b", "SmallNote"))
    add(paragraph("Pending external requirements are the shared PARAM Rudra run, course slide-template matching, team-member details, the student's recorded ten-minute delivery, and a sampled Windows profiler trace. Local phase profiling and the talk script accompany this report.", "SmallNote"))
    path = OUT / "pdf/Project_1_Report.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setStrokeColor(TEAL); canvas.setLineWidth(.5); canvas.line(42, 33, 553, 33)
        canvas.setFont("Helvetica", 7.5); canvas.setFillColor(INK)
        canvas.drawString(42, 22, "CPU inference project | Gated DeltaNet and Mamba-3")
        canvas.drawRightString(553, 22, str(doc.page))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=42, rightMargin=42,
                            topMargin=34, bottomMargin=43, title="CPU inference project", author="Siddharth Kumar")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(path).pages)
    if pages != 4:
        raise RuntimeError(f"Expected four pages, produced {pages}")
    print(path)


if __name__ == "__main__":
    make_report()
