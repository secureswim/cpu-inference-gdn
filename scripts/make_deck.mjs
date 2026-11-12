// Run with the Node runtime that provides Artifact Tool.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const modules = process.env.RUNTIME_NODE_MODULES;
const skill = process.env.PRESENTATIONS_SKILL_DIR;
if (!modules || !skill) {
  throw new Error('Set RUNTIME_NODE_MODULES and PRESENTATIONS_SKILL_DIR.');
}
const { Presentation, PresentationFile } = await import(pathToFileURL(
  path.join(modules, '@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const { finalizePresentation, applyPresentationChartFont } = await import(pathToFileURL(
  path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const readJSON = async file => JSON.parse(await fs.readFile(path.join(root, 'results', file), 'utf8'));
const data = await readJSON('summary.json');
const env = await readJSON('environment.json');
const correctness = await readJSON('gdn_correctness.json');
const [rudra] = await readJSON('param-rudra/186881/reference.json');

const cores = data.max_threads;
const threads = Array.from({ length: cores }, (_, i) => i + 1);
const sample = (mode, length, threads = cores, pinned = 1) => data.gdn.find(
  row => row.mode === mode && Number(row.length) === length &&
    Number(row.threads) === threads && Number(row.pinned) === pinned);
const bandwidth = n => data.stream_triad.find(row => Number(row.threads) === n).mean / 1000;
const predicted = n => bandwidth(n) * 1e9 / data.gdn_logical_bytes;
const fmt = n => Math.round(n).toLocaleString('en-US');
const rate = row => fmt(row.mean) + ' ± ' + fmt(row.sd);
const delta = (a, b) => (100 * (a.mean / b.mean - 1)).toFixed(1);
const mib = bytes => bytes / 2 ** 20;
const opSource = 'https://github.com/ggml-org/llama.cpp/blob/' + env.llama_commit + '/ggml/src/ggml-cpu/ops.cpp';

const paper = '#F7F5EF';
const ink = '#192D2A';
const muted = '#4E615B';
const teal = '#006A5D';
const rust = '#AD4B26';
const neutral = '#6B6B74';
const mint = '#A5E7CE';
const font = 'Arial';
const math = 'Cambria Math';
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const script = [];

function text(slide, value, x, y, w, h, { size = 26, color = ink, bold = false, family = font } = {}) {
  const box = slide.shapes.add({
    geometry: 'textbox',
    position: { left: x, top: y, width: w, height: h },
    fill: 'none',
    line: { fill: 'none', width: 0 },
  });
  box.text = value;
  box.text.style = { typeface: family, fontSize: size, color, bold };
  return box;
}

function slide(title, seconds, notes, sources, { cover = false } = {}) {
  const s = presentation.slides.add();
  s.background.fill = cover ? ink : paper;
  if (!cover) {
    text(s, title, 62, 48, 1156, 68, { size: 45, bold: true });
    text(s, String(presentation.slides.items.length).padStart(2, '0'), 1176, 663, 48, 30,
      { size: 16, color: muted });
  }
  s.speakerNotes.text = 'Target duration ' + seconds + ' seconds.\n\n' + notes + '\n\nSources\n' + sources;
  script.push({ title, seconds, text: notes });
  return s;
}

function label(s, value, x, y, w = 340, color = muted) {
  text(s, value, x, y, w, 35, { size: 20, color, bold: true });
}

function stat(s, value, caption, x, y, w = 340, color = teal, size = 60) {
  text(s, value, x, y, w, size + 18, { size, color, bold: true });
  text(s, caption, x, y + size + 15, w, 78, { size: 23, color: muted });
}

function foot(s, value) {
  text(s, value, 62, 635, 1090, 45, { size: 18, color: muted });
}

function chart(s, type, categories, series, {
  x = 58, y = 151, w = 780, h = 455, xTitle = '', yTitle = '', number = '#,##0', max,
} = {}) {
  const colors = [teal, rust, neutral];
  const c = s.charts.add(type, {
    position: { left: x, top: y, width: w, height: h },
    categories,
    series: series.map((entry, i) => ({
      name: entry.name,
      values: entry.values.map(value => Math.round(value * 1000) / 1000),
      fill: colors[i],
      line: { fill: colors[i], width: i === 2 ? 1.5 : 3, style: 'solid' },
      ...(type === 'line' && i !== 2 ? {
        marker: { symbol: i === 0 ? 'circle' : 'square', size: 7, fill: colors[i] },
      } : {}),
    })),
    hasLegend: true,
    legend: { position: 'bottom', overlay: false, textStyle: { fontSize: 18, fill: ink } },
    lineOptions: { smooth: false },
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 80 },
    xAxis: { title: xTitle, textStyle: { fontSize: 18, fill: ink } },
    yAxis: {
      min: 0, ...(max === undefined ? {} : { max }), title: yTitle, numberFormatCode: number,
      textStyle: { fontSize: 17, fill: ink }, majorGridlines: { fill: '#D9DED7', width: 1 },
    },
    chartFill: 'none', plotAreaFill: 'none',
  });
  applyPresentationChartFont(c, { fontFamily: font });
  return c;
}

function lengthChart(s, mode, lengths, xTitle, max) {
  chart(s, 'line', lengths.map(fmt), [1, cores].map(n => ({
    name: n === 1 ? '1 P core' : n + ' P cores',
    values: lengths.map(length => sample(mode, length, n).mean),
  })), { xTitle, yTitle: 'Operator tokens/s', max });
}

{
  const s = slide('Gated DeltaNet on a laptop CPU', 25,
    'We measured one Gated DeltaNet layer on a laptop CPU. The experiment asks how much one decode step costs and how its recurrent state compares with the memory used by an attention layer. I will also show the prefill and thread sweeps, explain the bandwidth prediction, and close with the shared PARAM Rudra test.',
    'Project 1 assignment brief, Option 5. Repository README and results/.', { cover: true });
  text(s, 'PROJECT 1  /  OPTION 5  /  GROUP 4', 72, 73, 820, 36, { size: 22, color: mint, bold: true });
  text(s, 'Gated DeltaNet\non a laptop CPU', 66, 158, 1120, 240,
    { size: 89, color: paper, bold: true });
  text(s, 'Throughput, scaling, and state memory in llama.cpp', 72, 440, 1110, 65,
    { size: 32, color: mint });
  text(s, 'Siddharth Kumar (2023522)\nVaishnavi Srivastava (2023574)', 72, 570, 1050, 100,
    { size: 28, color: paper });
}

{
  const s = slide('The Gated DeltaNet recurrence', 65,
    'Read these equations for one value head and one token. M sub t minus one is the incoming state. First we multiply the old state by the exponential of the log-decay gate. The resulting matrix, M bar, predicts the value from the current key. The difference between the actual value and that prediction is the residual. Beta scales that residual. Its outer product with the key adds a rank-one correction to the state. Finally the updated state multiplies the query, and we divide by the square root of the head dimension. The sixteen query and key heads are shared across thirty-two value heads using the value head index modulo sixteen. The state shape is fixed as the sequence grows.',
    'FP64 cross-check in src/gdn.cpp.\n' + opSource);
  const equations = [
    ['Decay the state', 'M̄ₜ = exp(gₜ) Mₜ₋₁'],
    ['Correct the key’s prediction', 'Mₜ = M̄ₜ + βₜ (vₜ − M̄ₜkₜ) kₜᵀ'],
    ['Read out with the query', 'yₜ = (Mₜqₜ) / √d'],
  ];
  for (const [i, [description, expression]] of equations.entries()) {
    const y = 155 + 139 * i;
    label(s, description, 68, y, 700);
    text(s, expression, 67, y + 35, 746, 78, { size: 46, family: math });
  }
  label(s, 'Per value head', 855, 161);
  text(s, 'State matrix     128 × 128\nHead size         d = 128\nLog-decay        gₜ\nUpdate gate    βₜ', 855, 205, 350, 175,
    { size: 24 });
  label(s, 'Tested Qwen3-Next shape', 855, 411);
  text(s, '16 query/key heads\n32 value heads\nFP32 synthetic inputs, batch 1', 855, 451, 365, 130, { size: 23 });
  foot(s, 'Queries, keys, values, and outputs are column vectors. Superscript T denotes transpose.');
}

{
  const s = slide('Laptop setup and timing protocol', 45,
    'The laptop has six performance cores and eight efficiency-class cores reported by Windows. We pin one worker per physical performance core and sweep one through six workers. The fused operator runs in a CPU-only ggml build using AVX2 and FMA. Each configuration gets a warmup followed by three repeats. Prefill processes chunks of one hundred twenty-eight tokens, while decode restores its starting history before every repeat. We record elapsed time and peak working set. The laptop is plugged in with Best Performance selected, and desktop applications remained open. We treat that activity as a possible source of variation.',
    'results/hardware.txt, results/environment.json, results/power-confirmation.json, results/build-info.txt');
  text(s, 'Intel Core Ultra 9 185H', 63, 141, 1150, 64, { size: 41, bold: true, color: teal });
  label(s, 'Hardware', 66, 251, 470);
  text(s, '6 P cores · 8 efficiency-class cores¹\n32 GB-class RAM · 24 MiB shared L3\nP-core cache 48 KiB L1D · 64 KiB L1I · 2 MiB L2\nAVX2/FMA build · AVX-512 unavailable', 64, 298, 565, 193, { size: 23 });
  label(s, 'Timing', 663, 251, 470);
  text(s, '1–6 pinned P-core workers\nWarmup, then 3 repeats ≥ 0.35 s each\n128-token prefill chunks\nAC power · Windows Best Performance', 661, 298, 548, 193, { size: 23 });
  foot(s, '¹ Core classes reported by Windows. Desktop applications remained open during the runs.');
}

{
  const s = slide('Correctness against an FP64 reference', 55,
    'We checked the same fused ggml operator used in the timing runs. The scalar reference implements the recurrence independently in double precision. Across a thousand recurrent steps, the query, key and value inputs change while the state carries forward. We compare both the output and the state after each step because errors in the state can affect future tokens. The largest absolute differences are about one times ten to the minus seven in state and about one times ten to the minus eight in output. We also compare one graph processing seventeen tokens with seventeen single-token graphs. The comparison returns zero difference for the tested input. These checks take the place of a model-level sanity check for this operator-only option.',
    'src/gdn.cpp, results/gdn_correctness.json');
  const sci = n => n.toExponential(2).replace('e-7', ' × 10⁻⁷').replace('e-8', ' × 10⁻⁸');
  stat(s, '1,000', 'Recurrent steps with\nchanging Q, K, and V', 64, 171, 360);
  stat(s, sci(correctness.max_state_abs), 'Maximum absolute\nstate error', 450, 171, 380, teal, 49);
  stat(s, sci(correctness.max_output_abs), 'Maximum absolute\noutput error', 856, 171, 365, teal, 49);
  label(s, 'Chunk equivalence', 65, 445, 570);
  text(s, '17 tokens in one graph vs. 17 single-token calls', 64, 492, 1120, 56, { size: 33, bold: true });
  text(s, 'Maximum absolute difference was ' + correctness.chunk_abs, 65, 552, 1000, 40,
    { size: 24, color: muted });
}

{
  const one = sample('decode', 1024, 1);
  const six = sample('decode', 1024);
  const s = slide('Decode speed vs. starting context', 50,
    'Each point measures one-token calls after advancing the state through the context shown on the horizontal axis. The timer includes the driver’s state carry. On one core, the mean remains between about two thousand one hundred and two thousand two hundred tokens per second across the three contexts. At one thousand twenty-four tokens the reciprocal of the mean throughput is ' + (1000 / one.mean).toFixed(2) + ' milliseconds per token. At six cores that same configuration reaches ' + rate(six) + ' tokens per second. The six-core mean falls at thirty-two thousand seven hundred sixty-eight tokens even though the state allocation and work per step stay fixed. These runs cannot isolate the cause of the fall.',
    'results/gdn_raw.csv, results/summary.json, output/plots/length_sweeps.pdf');
  lengthChart(s, 'decode', [1024, 8192, 32768], 'Starting context (tokens)', 2500);
  stat(s, (1000 / one.mean).toFixed(2) + ' ms', 'Per token at 1,024 context\non one P core', 860, 180);
  label(s, '6 P cores at 32,768 tokens', 865, 390);
  text(s, rate(sample('decode', 32768)), 861, 433, 350, 60, { size: 35, bold: true, color: rust });
  text(s, 'tokens/s', 865, 484, 300, 34, { size: 21, color: muted });
  foot(s, 'Three repeats per point. Chart shows means. The callout gives mean ± sample SD.');
}

{
  const fourK = sample('prefill', 4096);
  const s = slide('Prefill speed vs. prompt length', 45,
    'Prefill processes complete prompts of five hundred twelve, four thousand ninety-six and thirty-two thousand seven hundred sixty-eight tokens. The driver uses fixed-size chunks of one hundred twenty-eight tokens and starts each full prompt from zero state. The same seeded input block repeats across chunks, keeping the buffers bounded. At four thousand ninety-six tokens, six P cores reach ' + rate(fourK) + ' tokens per second across three repeats. At the longest prompt the mean is ' + fmt(sample('prefill', 32768).mean) + '. That is a lower rate, with considerable repeat variation. The next slide holds prompt length constant to isolate the effect of worker count.',
    'results/gdn_raw.csv, results/summary.json, output/plots/length_sweeps.pdf');
  lengthChart(s, 'prefill', [512, 4096, 32768], 'Prompt length (tokens)', 18000);
  stat(s, fmt(fourK.mean), 'tokens/s at 4,096 tokens\non six P cores', 860, 180);
  label(s, 'Sample SD at this point', 865, 391);
  text(s, '±' + fmt(fourK.sd) + ' tokens/s', 861, 435, 350, 60, { size: 31, bold: true, color: rust });
  foot(s, 'Three repeats per point. Chart shows means. Every prefill prompt starts from zero state.');
}

{
  const speedup = (mode, length, n) => sample(mode, length, n).mean / sample(mode, length, 1).mean;
  const s = slide('Thread scaling from one to six P cores', 45,
    'Here each series is normalized to its own one-core rate. The decode series starts from a one thousand twenty-four token history, and the prefill series uses a four thousand ninety-six token prompt. At six cores, prefill is ' + speedup('prefill', 4096, cores).toFixed(2) + ' times its one-core rate. Decode is ' + speedup('decode', 1024, cores).toFixed(2) + ' times its one-core rate. The backend divides work across heads. Each single-token call also pays dispatch, synchronization and state carry costs. Prefill spreads fixed call costs over one hundred twenty-eight tokens. These costs help explain the different scaling patterns.',
    'results/gdn_raw.csv, results/summary.json, output/plots/thread_scaling.pdf');
  chart(s, 'line', threads.map(String), [
    { name: 'Decode · 1,024 history', values: threads.map(n => speedup('decode', 1024, n)) },
    { name: 'Prefill · 4,096 prompt', values: threads.map(n => speedup('prefill', 4096, n)) },
    { name: 'Ideal linear', values: threads },
  ], { xTitle: 'Physical P cores', yTitle: 'Speedup over 1 P core', number: '0.0"×"', max: 6 });
  stat(s, speedup('prefill', 4096, cores).toFixed(2) + '×', 'Prefill at six P cores', 860, 181, 340, rust);
  stat(s, speedup('decode', 1024, cores).toFixed(2) + '×', 'Decode at six P cores', 860, 376);
}

{
  const phase = data.phase_percent;
  const names = ['Initial state copy', 'Decay', 'Delta dot products', 'Rank-one update', 'Output dot products'];
  const s = slide('Time inside the fused operator', 60,
    'These are scoped timer measurements from inside the fused operator with one worker. In decode, the three largest shares are initial state copy, decay, and output dot products. The rank-one update is almost tied with output dot products. In prefill the top three are output dot products, decay, and delta dot products. The same initial state copy serves the whole chunk during prefill, but decode performs it on every graph call. That changes which part of the operator dominates. The percentages normalize to the five timed phases and exclude graph scheduling and the separate state carry in the driver. We used uninstrumented runs for throughput. Windows blocked supplemental CPU sampling, so this breakdown uses scoped timers.',
    'results/gdn_raw.csv, results/profiles/README.md, output/plots/operator_breakdown.pdf');
  chart(s, 'bar', ['State copy', 'Decay', 'Delta dots', 'State update', 'Output dots'], [
    { name: 'Decode', values: names.map(name => phase.profile[name]) },
    { name: 'Prefill', values: names.map(name => phase.profile_prefill[name]) },
  ], { yTitle: 'Share of scoped time (%)', max: 50 });
  stat(s, phase.profile['Initial state copy'].toFixed(1) + '%', 'Decode time share\nin the initial state copy', 860, 178);
  stat(s, phase.profile_prefill['Initial state copy'].toFixed(1) + '%', 'Prefill time share\nin the initial state copy', 860, 380, 340, rust);
  foot(s, 'One worker. Five timed phases inside GDN. Windows blocked sampled profiling.');
}

{
  const s = slide('Predicted vs. measured decode speed', 65,
    'The roofline check divides STREAM Triad bandwidth by logical bytes moved per decoded token. At six cores, STREAM gives ' + bandwidth(cores).toFixed(2) + ' gigabytes per second. The logical traffic count includes reading and writing the operator state, the driver’s state carry copy, and the small inputs and output. That is ' + mib(data.gdn_logical_bytes).toFixed(2) + ' mebibytes per token. The resulting prediction is ' + fmt(predicted(cores)) + ' tokens per second. At the same core count the measured rate is ' + fmt(sample('decode', 1024).mean) + '. The prediction charges all counted traffic to DRAM. The two mebibyte state can fit in the shared cache, so some of that traffic may be served there. The kernel also revisits state in multiple phases, and dispatch and synchronization take time. Without hardware-counter data we cannot assign a share of the gap to each cause.',
    'results/stream_raw.csv, results/gdn_raw.csv, output/plots/roofline.pdf.\nhttps://www.cs.virginia.edu/stream/');
  chart(s, 'line', threads.map(String), [
    { name: 'Measured decode · 1,024 history', values: threads.map(n => sample('decode', 1024, n).mean) },
    { name: 'STREAM prediction', values: threads.map(predicted) },
  ], { xTitle: 'Physical P cores', yTitle: 'Operator tokens/s' });
  label(s, 'Six-core prediction', 865, 170);
  text(s, bandwidth(cores).toFixed(2) + ' GB/s', 860, 208, 340, 70, { size: 51, bold: true, color: teal });
  text(s, '÷ ' + mib(data.gdn_logical_bytes).toFixed(2) + ' MiB/token', 865, 279, 345, 46, { size: 26 });
  text(s, '= ' + fmt(predicted(cores)) + ' tokens/s', 865, 332, 348, 54, { size: 30, bold: true, color: rust });
  label(s, 'Six-core measurement', 865, 430);
  text(s, fmt(sample('decode', 1024).mean) + ' tokens/s', 861, 468, 355, 62,
    { size: 35, bold: true, color: teal });
  foot(s, 'The bandwidth estimate omits graph dispatch, synchronization, and additional state passes.');
}

{
  const contexts = [1024, 4096, 8192, 16384, 32768];
  const s = slide('State memory vs. attention KV cache', 50,
    'The Gated DeltaNet state holds thirty-two matrices, each one hundred twenty-eight by one hundred twenty-eight FP32 elements. This gives two mebibytes per layer at every context length. For attention we assume sixteen KV heads with the same head width and FP32 keys and values. Each extra token therefore adds sixteen kibibytes per layer, giving sixteen mebibytes at one thousand twenty-four tokens and five hundred twelve mebibytes at thirty-two thousand seven hundred sixty-eight. This chart compares persistent state only. Separately, the largest measured laptop process working set was ' + mib(data.max_gdn_rss).toFixed(1) + ' mebibytes. The raw results record peak working set for every configuration, including temporary buffers and runtime overhead.',
    'src/gdn.cpp, results/summary.json, output/plots/state_vs_kv.pdf');
  chart(s, 'line', contexts.map(n => fmt(n / 1024) + 'k'), [
    { name: 'GDN recurrent state', values: contexts.map(() => 2) },
    { name: 'Attention KV cache', values: contexts.map(n => mib(2 * n * 16 * 128 * 4)) },
  ], { xTitle: 'Context length (tokens)', yTitle: 'Persistent storage (MiB/layer)', max: 550 });
  stat(s, '2 MiB', 'GDN recurrent state\nacross all contexts', 860, 160);
  stat(s, '512 MiB', 'FP32 attention KV cache\nat 32,768 tokens', 860, 365, 350, rust);
  foot(s, 'Analytical storage comparison. Measured laptop process peak working set was ' +
    mib(data.max_gdn_rss).toFixed(1) + ' MiB.');
}

{
  const d = sample('decode', 1024);
  const p = sample('prefill', 4096);
  const dOS = sample('decode', 1024, cores, 0);
  const pOS = sample('prefill', 4096, cores, 0);
  const signed = (a, b) => {
    const change = Number(delta(a, b));
    return (change >= 0 ? '+' : '−') + Math.abs(change).toFixed(1) + '%';
  };
  const s = slide('Thread placement experiment', 50,
    'For the additional experiment we held the worker count and workload fixed and changed the processor placement policy. In the first configuration, workers are restricted to the physical P cores. In the second, the operating system can choose their placement. At one thousand twenty-four context, the pinned decode mean is ' + delta(d, dOS) + ' percent higher, a difference smaller than either sample standard deviation. Prefill shows a larger observed gain of ' + delta(p, pOS) + ' percent. We ran three repeats per setting, and the configurations ran at different times. Run order and background activity could affect the size of the prefill gain.',
    'results/gdn_raw.csv, results/summary.json, output/plots/affinity_experiment.pdf');
  const rows = [
    ['Decode · 1,024 history', rate(d), rate(dOS), signed(d, dOS)],
    ['Prefill · 4,096 prompt', rate(p), rate(pOS), signed(p, pOS)],
  ];
  const x = [64, 410, 687, 981];
  const w = [331, 257, 270, 230];
  ['Workload', 'P cores pinned', 'OS placement', 'Pinning change'].forEach((value, i) => label(s, value, x[i], 200, w[i]));
  for (const [index, row] of rows.entries()) {
    row.forEach((value, i) => text(s, value, x[i], 254 + 93 * index, w[i], 65,
      { size: i === 0 ? 26 : 31, bold: true, color: i === 3 ? teal : ink }));
  }
  text(s, 'The decode change falls within repeat variation. Prefill shows a larger gain.', 65, 503, 1130, 60,
    { size: 29, color: teal, bold: true });
  foot(s, 'Six workers per configuration. Rates are tokens/s, mean ± sample SD, three repeats.');
}

{
  const s = slide('Shared PARAM Rudra CPU reference', 45,
    'The course reference used Llama 3.2 1B Instruct Q4 K M on a Xeon Gold CPU with four threads. For a five hundred twelve token prompt followed by one hundred twenty-eight generated tokens, the combined rate was ' + rudra.avg_ts.toFixed(2) + ' tokens per second, with a sample standard deviation of ' + rudra.stddev_ts.toFixed(2) + ' across three repeats. Peak RSS is unavailable because the cluster returned zero and the compute node lacked GNU time. The reference is full-model inference. Our Option 5 operator measurements show a fixed two mebibyte recurrent state and stronger thread scaling in prefill than in single-token decode. Raw results, build notes, and scripts are in the repository.',
    'results/param-rudra/186881/reference.json, hardware.txt and slurm-accounting.txt in the same directory.');
  stat(s, rudra.avg_ts.toFixed(2) + ' ± ' + rudra.stddev_ts.toFixed(2),
    'Combined tokens/s, mean ± sample SD', 64, 152, 790, teal, 81);
  label(s, 'Shared workload', 69, 350, 490);
  text(s, 'Llama 3.2 1B Instruct · Q4_K_M\n512 prompt + 128 generated tokens\n3 repeats · 4 CPU threads', 65, 391, 545, 152, { size: 27 });
  label(s, 'Compute node', 654, 350, 440);
  text(s, 'Intel Xeon Gold 6240R\nrdcn20 · Slurm job 186881\nPeak RSS unavailable', 650, 391, 540, 152, { size: 27 });
  foot(s, 'Reproduction files at github.com/secureswim/cpu-inference-gdn-mamba3');
}

const buildDir = path.join(root, 'build/slides');
await fs.mkdir(buildDir, { recursive: true });
const candidate = path.join(buildDir, 'candidate.pptx');
const finalPath = path.join(root, process.env.DECK_OUTPUT || 'output/Project_1_Presentation.pptx');
await fs.mkdir(path.dirname(finalPath), { recursive: true });
await (await PresentationFile.exportPptx(presentation)).save(candidate);
await finalizePresentation({
  workspaceDir: root,
  candidatePath: candidate,
  finalPath,
  pythonExecutable: process.env.RUNTIME_PYTHON,
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-heading-fit'],
  explicitTotalSlideCount: 12,
  requiredNativeChartOwnerSlides: [5, 6, 7, 8, 9, 10],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: 'design', families: [font, math] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, path.basename(finalPath) + '.validation.json'),
});
for (const [index, s] of presentation.slides.items.entries()) {
  const png = await presentation.export({ slide: s, format: 'png', scale: 1 });
  await fs.writeFile(path.join(buildDir, 'slide-' + (index + 1) + '.png'),
    new Uint8Array(await png.arrayBuffer()));
}

const clock = n => Math.floor(n / 60) + ':' + String(n % 60).padStart(2, '0');
let cursor = 0;
let content = '# Ten-minute talk script\n\n';
for (const [index, s] of script.entries()) {
  content += '## Slide ' + (index + 1) + ' - ' + s.title + '\n\n' +
    clock(cursor) + ' to ' + clock(cursor + s.seconds) + '\n\n' + s.text + '\n\n';
  cursor += s.seconds;
}
if (cursor !== 600) throw new Error('Expected a ten-minute talk plan.');
await fs.writeFile(path.join(root, 'output/Talk_Script.md'), content.trimEnd() + '\n');
const timings = script.map(({ title, seconds }, index) => ({ slide: index + 1, title, seconds }));
await fs.writeFile(path.join(root, 'results/talk-timings.json'), JSON.stringify(timings, null, 2));
console.log(finalPath);
