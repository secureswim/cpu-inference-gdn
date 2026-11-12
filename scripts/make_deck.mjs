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
const [rudra] = await readJSON('param-rudra/186891/reference.json');
const rudraMemory = await readJSON('param-rudra/186891/peak-rss.json');
const study = data.working_set_study;
const sampledPath = path.join(root, 'results/profiles/sampled.json');
const sampled = await fs.readFile(sampledPath, 'utf8').then(JSON.parse).catch(error => {
  if (error.code !== 'ENOENT') throw error;
  return null;
});

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
const step = n => 1e6 / sample('decode', 1024, n).mean;
const rotation = (n, states) => study.measurements.find(row => row.threads === n && row.states === states);
const historyDelta = Math.max(...study.history.map(row => Number(row.max_abs_from_1024)));
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
    'At one thousand twenty-four tokens of starting history, one decode step took about ' + fmt(step(1)) + ' microseconds on one performance core and ' + fmt(step(cores)) + ' on six. That measures one recurrent layer, including the driver state copy. We then changed the number of recurrent states to test how the memory footprint affects latency.',
    'Project 1 assignment brief, Option 5. Repository README and results/.', { cover: true });
  text(s, 'PROJECT 1  /  OPTION 5  /  GROUP 4', 72, 73, 820, 36, { size: 22, color: mint, bold: true });
  text(s, 'Gated DeltaNet\non a laptop CPU', 66, 158, 1120, 240,
    { size: 89, color: paper, bold: true });
  text(s, fmt(step(1)) + ' µs on 1 P core  /  ' + fmt(step(cores)) + ' µs on 6 P cores per decode step', 72, 440, 1110, 65,
    { size: 32, color: mint });
  text(s, 'Siddharth Kumar (2023522)\nVaishnavi Srivastava (2023574)', 72, 570, 1050, 100,
    { size: 28, color: paper });
}

{
  const s = slide('The Gated DeltaNet recurrence', 55,
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
  const s = slide('Laptop setup and timing protocol', 40,
    'The laptop has six performance cores, eight efficiency cores and two low-power efficiency cores. The affinity mask uses one hardware thread per performance core. We sweep one through six workers with a CPU-only AVX2 and FMA build. Each configuration has a warmup and three timed repeats. A seeded order interleaves configurations. Prefill processes chunks of one hundred twenty-eight tokens, and decode restores its starting history before timing. The rerun used AC power and Best Performance with the browser and IDE windows closed.',
    'results/hardware.txt, results/environment.json, results/power-confirmation.json, results/build-info.txt');
  text(s, 'Intel Core Ultra 9 185H', 63, 141, 1150, 64, { size: 41, bold: true, color: teal });
  label(s, 'Hardware', 66, 251, 470);
  text(s, '6 P-cores · 8 E-cores · 2 low-power E-cores\n32 GB-class RAM · 24 MiB shared L3\nP-core cache 48 KiB L1D · 64 KiB L1I · 2 MiB L2\nAVX2/FMA build · AVX-512 unavailable', 64, 298, 565, 193, { size: 23 });
  label(s, 'Timing', 663, 251, 470);
  text(s, '1–6 pinned P-core workers\nWarmup, then 3 repeats ≥ 0.35 s each\n128-token prefill chunks\nAC power · Windows Best Performance', 661, 298, 548, 193, { size: 23 });
  foot(s, 'Browser and IDE windows closed. Chrome and Edge processes absent at startup.');
}

{
  const s = slide('Correctness against an FP64 reference', 45,
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
  const s = slide('Decode speed vs. starting context', 45,
    'Each point times single-token calls after preparing the context shown. Every history uses the same seeded block of one hundred twenty-eight tokens and ends at the same block boundary. The maximum difference between the prepared states was ' + (historyDelta === 0 ? 'zero' : historyDelta.toExponential(2)) + '. The tensor sizes and work per step stay fixed. The timing spread and the interleaved run order matter when interpreting differences. At one thousand twenty-four history, the one-core measurement is ' + fmt(step(1)) + ' microseconds per step. Six cores reached ' + rate(six) + ' tokens per second.',
    'results/gdn_raw.csv, results/summary.json, output/plots/length_sweeps.pdf');
  lengthChart(s, 'decode', [1024, 8192, 32768], 'Starting context (tokens)');
  stat(s, (1000 / one.mean).toFixed(2) + ' ms', 'Per token at 1,024 context\non one P core', 860, 180);
  label(s, '6 P cores at 32,768 tokens', 865, 390);
  text(s, rate(sample('decode', 32768)), 861, 433, 350, 60, { size: 35, bold: true, color: rust });
  text(s, 'tokens/s', 865, 484, 300, 34, { size: 21, color: muted });
  foot(s, 'Largest elementwise difference between prepared states was ' +
    (historyDelta === 0 ? 'zero.' : historyDelta.toExponential(2) + '.'));
}

{
  const fourK = sample('prefill', 4096);
  const s = slide('Prefill speed vs. prompt length', 40,
    'Prefill processes complete prompts of five hundred twelve, four thousand ninety-six and thirty-two thousand seven hundred sixty-eight tokens. The driver starts each full prompt from zero state and uses fixed chunks of one hundred twenty-eight. The same seeded block repeats across chunks. At four thousand ninety-six tokens, six P cores reach ' + rate(fourK) + ' tokens per second across three repeats. At the longest prompt the six-core mean is ' + fmt(sample('prefill', 32768).mean) + '.',
    'results/gdn_raw.csv, results/summary.json, output/plots/length_sweeps.pdf');
  lengthChart(s, 'prefill', [512, 4096, 32768], 'Prompt length (tokens)');
  stat(s, fmt(fourK.mean), 'tokens/s at 4,096 tokens\non six P cores', 860, 180);
  label(s, 'Sample SD at this point', 865, 391);
  text(s, '±' + fmt(fourK.sd) + ' tokens/s', 861, 435, 350, 60, { size: 31, bold: true, color: rust });
  foot(s, 'Three repeats per point. Chart shows means. Every prefill prompt starts from zero state.');
}

{
  const speedup = (mode, length, n) => sample(mode, length, n).mean / sample(mode, length, 1).mean;
  const s = slide('Thread scaling from one to six P cores', 40,
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
  const carry = rotation(1, 1).carry_us.mean;
  const serial = carry / rotation(1, 1).diagnostic_us.mean;
  const s = slide('Two state copies with different costs', 55,
    'The scoped timers measure five phases inside the fused operator on one worker. The ggml initial copy runs inside head ranges assigned to different workers, so its one-worker percentage cannot be used as a serial fraction. After ggml returns, the driver performs a separate serial copy to carry the next state. With one state that copy averages ' + fmt(carry) + ' microseconds, or ' + fmt(serial * 100) + ' percent of our diagnostic step. The operator percentages exclude this driver copy. ' +
      (sampled ? sampled.report_note : 'Windows blocked sampled profiling. The phase chart uses scoped timing.'),
    'results/gdn_raw.csv, results/working_sets.csv, results/profiles/, output/plots/operator_breakdown.pdf');
  chart(s, 'bar', ['State copy', 'Decay', 'Delta dots', 'State update', 'Output dots'], [
    { name: 'Decode', values: names.map(name => phase.profile[name]) },
    { name: 'Prefill', values: names.map(name => phase.profile_prefill[name]) },
  ], { yTitle: 'Share of scoped time (%)' });
  stat(s, phase.profile['Initial state copy'].toFixed(1) + '%', 'Initial ggml copy share\nwithin scoped decode', 860, 175);
  stat(s, fmt(carry) + ' µs', 'Separate serial driver copy\non one P core, one state', 860, 378, 350, rust);
  foot(s, sampled ? sampled.slide_note : 'Scoped timings use one worker. Windows blocked sampled profiling.');
}

{
  const s = slide('Predicted vs. measured decode speed', 45,
    'The assignment comparison divides STREAM Triad bandwidth by our logical byte count per step. At six cores, STREAM gives ' + bandwidth(cores).toFixed(2) + ' gigabytes per second. The logical count includes two state copies and the small input and output tensors. That gives ' + fmt(predicted(cores)) + ' tokens per second compared with ' + fmt(sample('decode', 1024).mean) + ' measured. The estimate treats the counted bytes as DRAM traffic. A single state can fit in the shared cache, and graph scheduling, arithmetic, and extra state passes also take time.',
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
  foot(s, 'The prediction assumes that all counted bytes travel to DRAM on each step.');
}

{
  const contexts = [1024, 4096, 8192, 16384, 32768];
  const s = slide('State memory vs. attention KV cache', 45,
    'The Gated DeltaNet state holds thirty-two matrices of one hundred twenty-eight by one hundred twenty-eight FP32 values. This is two mebibytes per layer. Qwen3-Next full attention uses two key-value heads of width two hundred fifty-six. With F16 keys and values, each token adds two kibibytes per attention layer. At thirty-two thousand seven hundred sixty-eight tokens that gives sixty-four mebibytes. These figures describe persistent recurrence and attention KV storage. Process working set includes buffers and runtime overhead.',
    'Qwen3-Next configuration at https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json\nsrc/gdn.cpp, output/plots/state_vs_kv.pdf');
  chart(s, 'line', contexts.map(n => fmt(n / 1024) + 'k'), [
    { name: 'GDN recurrent state', values: contexts.map(() => 2) },
    { name: 'Qwen3-Next attention KV (F16)', values: contexts.map(n => mib(2 * n * 2 * 256 * 2)) },
  ], { xTitle: 'Context length (tokens)', yTitle: 'Persistent storage (MiB/layer)', max: 70 });
  stat(s, '2 MiB', 'GDN recurrent state\nacross all contexts', 860, 160);
  stat(s, '64 MiB', 'F16 attention KV cache\nat 32,768 tokens', 860, 365, 350, rust);
  foot(s, 'Analytical storage comparison. Measured main sweep peak working set was ' +
    mib(data.max_gdn_rss).toFixed(1) + ' MiB.');
}

{
  const counts = [1, 4, 12, 36];
  const a = rotation(1, 1).latency_us.mean;
  const b = rotation(1, 36).latency_us.mean;
  const c = rotation(cores, 1).latency_us.mean;
  const d = rotation(cores, 36).latency_us.mean;
  const s = slide('Decode across independent states', 65,
    'The added experiment cycles through one, four, twelve and thirty-six independent states using one reusable graph. Each step performs the same arithmetic on the same input shapes. Thirty-six states represent seventy-two mebibytes of recurrent memory, compared with twenty-four mebibytes of shared L3. On one core latency changes from ' + fmt(a) + ' to ' + fmt(b) + ' microseconds per layer step. On six it changes from ' + fmt(c) + ' to ' + fmt(d) + '. One round over thirty-six states takes ' + (36 * b / 1000).toFixed(1) + ' milliseconds on one core and ' + (36 * d / 1000).toFixed(1) + ' on six. The footprint corresponds to the model’s thirty-six recurrent layers. The benchmark isolates the recurrent operation and its state carry. A complete model also has projections, attention, and expert layers.',
    'results/working_sets.csv, results/summary.json, output/plots/working_sets.pdf');
  chart(s, 'line', counts.map(String), [1, cores].map(n => ({
    name: n === 1 ? '1 P core' : n + ' P cores',
    values: counts.map(count => rotation(n, count).latency_us.mean),
  })), { xTitle: 'Independent states, 2 MiB each', yTitle: 'Microseconds per layer step' });
  stat(s, '72 MiB', '36 recurrent states\n24 MiB shared L3', 860, 168);
  label(s, '36-state round time', 865, 388);
  text(s, (36 * b / 1000).toFixed(1) + ' ms · 1 P core\n' +
    (36 * d / 1000).toFixed(1) + ' ms · 6 P cores', 865, 433, 338, 104, { size: 28, bold: true, color: rust });
  foot(s, 'Three repeats per point. Round times cover the recurrence and state carry only.');
}

{
  const limit = study.ceilings.at(-1);
  const s = slide('Cache and compute ceilings', 60,
    'The source-level count gives about ' + (study.flops_per_step / 1e6).toFixed(2) + ' million floating-point operations per step, excluding the exponentials. Dividing by the logical byte count gives ' + study.logical_flops_per_byte.toFixed(3) + ' FLOP per byte. We measured effective bandwidth at several working-set sizes and used the one and sixteen mebibyte copy points as cache-sized references. STREAM supplies the large-array DRAM reference. An AVX2 FMA loop gives the measured compute ceiling. We plot one-state and thirty-six-state decode at the same logical intensity. Cache capacity alone cannot tell us how many requests reached each level, and the byte model omits some passes over the state. The roofline puts the measurements on a common scale and shows the gap to those reference rates.',
    'results/ceilings.csv, results/stream_raw.csv, results/working_sets.csv, output/plots/cache_roofline.pdf');
  s.images.add({
    blob: await fs.readFile(path.join(root, 'output/plots/cache_roofline.png')),
    contentType: 'image/png', fit: 'contain',
    position: { left: 63, top: 158, width: 805, height: 398 },
  });
  label(s, 'Six-core reference rates', 888, 170);
  text(s, limit.l2_1mib.toFixed(0) + ' GB/s · 1 MiB copy\n' +
    limit.l3_16mib.toFixed(0) + ' GB/s · 16 MiB copy\n' +
    limit.dram_stream.toFixed(0) + ' GB/s · STREAM Triad\n' +
    limit.compute_gflops.toFixed(0) + ' GFLOP/s · FMA', 883, 220, 342, 162, { size: 22 });
  stat(s, study.logical_flops_per_byte.toFixed(3), 'Logical FLOP per byte', 886, 400, 325, teal, 48);
  foot(s, 'Cache-sized copy rates are empirical references. Arithmetic intensity uses logical traffic.');
}

{
  const s = slide('Shared PARAM Rudra CPU reference', 40,
    'The course reference used Llama 3.2 1B Instruct Q4 K M on four Xeon Gold CPU threads. The rerun delivered ' + rudra.avg_ts.toFixed(2) + ' combined tokens per second for a five hundred twelve token prompt and one hundred twenty-eight decode tokens. The benchmark process reached ' + (rudraMemory.peak_rss_bytes / 2 ** 30).toFixed(2) + ' gibibytes of peak resident memory. The full-model reference has a different scope from our layer study. The laptop results show stable state size and stronger scaling in prefill than in single-token decode.',
    'results/param-rudra/186891/reference.json, peak-rss.json, hardware.txt and slurm-accounting.txt.');
  stat(s, rudra.avg_ts.toFixed(2) + ' ± ' + rudra.stddev_ts.toFixed(2),
    'Combined tokens/s, mean ± sample SD', 64, 152, 790, teal, 81);
  label(s, 'Shared workload', 69, 350, 490);
  text(s, 'Llama 3.2 1B Instruct · Q4_K_M\n512 prompt + 128 generated tokens\n3 repeats · 4 CPU threads', 65, 391, 545, 152, { size: 27 });
  label(s, 'Compute node', 654, 350, 440);
  text(s, 'Intel Xeon Gold 6240R\nrdcn20 · Slurm job 186891\nPeak RSS ' +
    (rudraMemory.peak_rss_bytes / 2 ** 30).toFixed(2) + ' GiB', 650, 391, 540, 152, { size: 27 });
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
  explicitTotalSlideCount: 13,
  requiredNativeChartOwnerSlides: [5, 6, 7, 8, 9, 10, 11],
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
