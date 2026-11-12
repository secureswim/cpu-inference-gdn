// Run with the Codex bundled Node runtime. The experimental pipeline uses Python.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const modules=process.env.RUNTIME_NODE_MODULES;
const skill=process.env.PRESENTATIONS_SKILL_DIR;
if(!modules || !skill) throw new Error('Set RUNTIME_NODE_MODULES and PRESENTATIONS_SKILL_DIR to the bundled runtime and presentation skill.');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(modules,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const data=JSON.parse(await fs.readFile(path.join(root,'results/summary.json'),'utf8'));
const env=JSON.parse(await fs.readFile(path.join(root,'results/environment.json'),'utf8'));
const [rudra]=JSON.parse(await fs.readFile(path.join(root,'results/param-rudra/186881/reference.json'),'utf8'));
const rudraRate=`${rudra.avg_ts.toFixed(2)} ± ${rudra.stddev_ts.toFixed(2)} tokens/s`;
const t=data.max_threads;
const g=(mode,length,threads=t,pinned=1)=>data.gdn.find(r=>r.mode===mode && r.length===String(length) && r.threads===String(threads) && r.pinned===String(pinned));
const fmt=n=>Math.round(n).toLocaleString('en-US');
const bw=threads=>data.stream_triad.find(r=>r.threads===String(threads)).mean*1e6;
const navy='#142A43',teal='#176B87',orange='#D66A2B',muted='#4E6072',paper='#F8FAFB';
const presentation=Presentation.create({slideSize:{width:1280,height:720}});
const script=[];
const timings=[30,40,45,50,40,50,50,40,60,45,110,40];
function text(s,value,x,y,w,h,size=28,color=navy,bold=false){
  const box=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  box.text=value;box.text.style={fontFamily:'Arial',fontSize:size,color,bold};return box;
}
function slide(title,notes,sources='Local raw measurements in results/.'){
  const s=presentation.slides.add();s.background.fill=paper;
  text(s,title,64,40,1150,92,44,navy,true);
  text(s,String(presentation.slides.items.length),1185,673,42,24,18,muted);
  const duration=timings[presentation.slides.items.length-1];
  s.speakerNotes.text=`Target duration ${duration} seconds.\n\n${notes}\n\nSources\n${sources}`;
  script.push({title,seconds:duration,text:notes,sources});
  return s;
}
function chart(s,type,categories,series,{x=70,y=164,w=1130,h=420,yTitle='Operator tokens/s',xTitle='',legend=true}={}){
  const c=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories,
    series:series.map((v,i)=>({...v,values:v.values.map(n=>Math.round(n*1000)/1000),fill:[teal,orange,'#5C469C'][i%3],line:{fill:[teal,orange,'#5C469C'][i%3],width:3,style:'solid'}})),
    hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{fontSize:20,fill:muted}},
    lineOptions:{smooth:false},
    barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
    xAxis:{title:xTitle,textStyle:{fontSize:20,fill:muted}},
    yAxis:{min:0,title:yTitle,numberFormatCode:'#,##0.0',textStyle:{fontSize:20,fill:muted},majorGridlines:{fill:'#DAE2E9',width:1}},
    chartFill:'none',plotAreaFill:'none'});
  applyPresentationChartFont(c,{fontFamily:'Arial'});return c;
}

{
 const s=slide('Recurrent inference on a laptop CPU',
 `This project studies the cost of recurrent sequence-model operators on a laptop processor. The main experiment uses the actual Gated DeltaNet operation in llama.cpp. I will explain the correctness checks, measured throughput and memory, the bandwidth prediction, and the shared Rudra result. These are single-layer kernel measurements. Full-model generation has additional work.`);
 s.background.fill=navy;
 s.shapes.items[0].text.style={fontFamily:'Arial',fontSize:44,color:'#FFFFFF',bold:true};
 text(s,'Gated DeltaNet operator study',70,250,1120,62,42,'#FFFFFF');
 text(s,'Option 5 operator study',70,325,1120,55,34,'#C6DFE8');
 text(s,'Siddharth Kumar',70,535,1080,45,28,'#FFFFFF');
 text(s,env.timings_provisional?'Provisional battery measurements':'CPU-only measurements on AC power',70,596,1100,35,23,'#C6DFE8');
}
{
 const s=slide('Gated DeltaNet and fixed-size state',
 `Gated DeltaNet stores a matrix for each value head. At each token it decays that matrix, predicts the current value from the key, and uses a gated correction to update the matrix. It then reads the output using the query. We use the assignment's dimensions, sixteen query and key heads, thirty-two value heads, and head size one hundred and twenty-eight. The resulting main state is exactly two mebibytes. The state summarizes earlier tokens, so its storage has no context-length dimension. The attention comparison instead stores every earlier key and value.`,
 'https://github.com/ggml-org/llama.cpp/blob/8bb3cf0d216a5ae48360179f2931daf284be4d01/ggml/src/ggml-cpu/ops.cpp');
 text(s,'M = decay x M + beta x (v - decay x M k) k^T',70,176,1130,75,34,teal,true);
 text(s,'y = M q / sqrt(128)',70,267,1130,60,34,teal,true);
 text(s,'16 Q/K heads, 32 value heads, head size 128\n2 MiB recurrent state, 512 MiB attention KV at 32k\nFP32 storage per layer, 16 attention KV heads\nBatch one, synthetic inputs, actual ggml CPU operator',70,385,1120,200,30);
}
{
 const s=slide('Hardware and measurement protocol',
 `The CPU is an Intel Core Ultra nine, eighteen-five H. Windows reports six performance cores, and the benchmark allows one logical thread per performance core. It excludes the efficiency cores from the main sweep. The machine supports AVX2 and FMA, and the build explicitly selects those instructions. Every configuration gets three repeats, each lasting at least zero point three five seconds after warmup. Long prompts can run longer. We retain raw timings and peak working set. The charger is connected. The recorded base power scheme and background applications remain relevant limits, even with the selected performance mode.`,
 'results/hardware.txt, results/environment.json, results/python-lock.txt');
 text(s,'Intel Core Ultra 9 185H',70,175,1120,65,39,teal,true);
 text(s,'6 physical P cores, AVX2/FMA, 24 MiB L3\n1 to 6 workers, one logical thread per P core\nThree repeats, warmup, peak RSS for each configuration\nPrefill uses 128-token chunks and bounded buffers',70,285,1130,265,31);
 text(s,'Raw CSVs retain every repeat. Desktop applications remained open.',70,606,1130,50,22,muted);
}
{
 const s=slide('Decode and prefill length sweeps',
 `The left chart shows decode after different amounts of history. Before timing, the driver actually advances the recurrent state through the labelled context length. Every timed repeat restores that starting state. The right chart processes complete synthetic prompts in chunks of one hundred and twenty-eight tokens. The input block repeats across chunks, which makes the experiment reproducible with bounded buffers. Both charts show one and six performance cores. At one thousand twenty-four context tokens, six-core decode reaches ${fmt(g('decode',1024).mean)} operator tokens per second. These points describe a recurrent operator whose state allocation stays fixed. The saved plots show sample-standard-deviation error bars, and the CSVs retain each of the three repeats.`);
 chart(s,'line',['1,024','8,192','32,768'],[{name:'1 P core',values:[1024,8192,32768].map(n=>g('decode',n,1).mean)},{name:`${t} P cores`,values:[1024,8192,32768].map(n=>g('decode',n).mean)}],{x:60,w:560,yTitle:'Decode tokens/s',xTitle:'Starting context'});
 chart(s,'line',['512','4,096','32,768'],[{name:'1 P core',values:[512,4096,32768].map(n=>g('prefill',n,1).mean)},{name:`${t} P cores`,values:[512,4096,32768].map(n=>g('prefill',n).mean)}],{x:655,w:560,yTitle:'Prefill tokens/s',xTitle:'Prompt tokens'});
 text(s,'Mean of three repeats. Spread appears in the report plots and raw CSVs.',70,618,1130,40,23,muted);
}
{
 const threads=Array.from({length:t},(_,i)=>i+1);
 const s=slide('Scaling across performance cores',
 `This chart holds the workload constant while changing the number of workers from one to all six physical performance cores. Decode begins at a one-thousand-token context and prefill processes four thousand ninety-six tokens. I normalize each series to its own one-core rate. Decode reaches ${ (g('decode',1024).mean/g('decode',1024,1).mean).toFixed(2)} times the one-core rate, while prefill reaches ${(g('prefill',4096).mean/g('prefill',4096,1).mean).toFixed(2)} times. The operator partitions work across value heads. Parallel workers also incur synchronization and share resources. Departures from ideal scaling need to be interpreted with those costs and the measurement spread.`,
 'results/gdn_raw.csv, scripts/analyze.py');
 chart(s,'line',threads.map(String),[{name:'Decode 1k',values:threads.map(n=>g('decode',1024,n).mean/g('decode',1024,1).mean)},{name:'Prefill 4k',values:threads.map(n=>g('prefill',4096,n).mean/g('prefill',4096,1).mean)},{name:'Ideal',values:threads}],{yTitle:'Speedup over one core',xTitle:'Physical P cores'});
}
{
 const names=['Initial copy','Decay','Delta dots','State update','Output dots'];
 const full=['Initial state copy','Decay','Delta dot products','Rank-one update','Output dot products'];
 const s=slide('Time inside the upstream operator',
 `These profiles use scoped timers inserted into the pinned upstream CPU operator. They measure the initial state copy, decay, delta dot products, rank-one update, and final output dot products. The profile uses one worker so its phase percentages are easy to interpret. Prefill copies the initial state once for a chunk and then processes many tokens. Decode copies it for every one-token graph call. This changes the relative importance of copying. The percentages normalize to the five measured phases. They exclude graph scheduling, the driver's state carry copy, and other work outside those scopes. Instrumented runs supply these percentages, while separate uninstrumented runs supply the headline throughput. A sampled VTune or perf trace would provide an additional check.`);
 chart(s,'bar',names,[{name:'Decode',values:full.map(k=>data.phase_percent.profile[k])},{name:'Prefill',values:full.map(k=>data.phase_percent.profile_prefill[k])}],{yTitle:'Share of scoped time (%)'});
 text(s,'Single worker. Scoped timers perturb execution. Sampled profiler trace remains pending.',70,618,1140,45,22,muted);
}
{
 const threads=Array.from({length:t},(_,i)=>i+1);
 const pred=bw(t)/data.gdn_logical_bytes;
 const s=slide('A DRAM-bandwidth prediction',
 `STREAM uses three arrays totalling four hundred eighty megabytes, which exceeds the twenty-four mebibyte last-level cache. At six workers its mean best Triad bandwidth is ${(bw(t)/1e9).toFixed(2)} gigabytes per second. Dividing bandwidth by our logical traffic bound predicts ${fmt(pred)} tokens per second, compared with ${fmt(g('decode',1024).mean)} measured. The key assumption is that every counted byte reaches DRAM on every step. Our main recurrent state is only two mebibytes and can remain in cache, so that assumption may fail. The actual upstream implementation also makes separate passes over state. Those passes create extra cache traffic. This comparison motivates a cache-aware explanation of the measured gap, while hardware counters would be needed to quantify cache misses.`,
 'results/stream_raw.csv, results/gdn_raw.csv; https://www.cs.virginia.edu/stream/');
 chart(s,'line',threads.map(String),[{name:'STREAM / logical traffic',values:threads.map(n=>bw(n)/data.gdn_logical_bytes)},{name:'Measured decode',values:threads.map(n=>g('decode',1024,n).mean)}],{xTitle:'Physical P cores'});
 text(s,`Logical lower bound ${ (data.gdn_logical_bytes/2**20).toFixed(3)} MiB/token, including driver state carry`,70,618,1130,40,22,muted);
}
{
 const ratio=g('decode',1024).mean/g('decode',1024,t,0).mean;
 const s=slide('Own experiment - thread placement',
 `For the additional experiment I change only the allowed processor placement. The worker count and operator workload remain the same. One configuration pins execution to the physical performance cores. The other lets the operating system place those workers across the available logical processors. At one-thousand-token decode, the change is ${(100*(ratio-1)).toFixed(1)} percent in this run. Prefill provides a second workload. Each point still has three repeats. Small differences should be interpreted against their spread. The configurations ran at different times, so background activity and power behavior remain potential confounders. This experiment supports a local observation about placement rather than a universal claim about affinity.`);
 chart(s,'bar',['Decode 1k','Prefill 4k'],[{name:'P cores pinned',values:[g('decode',1024).mean,g('prefill',4096).mean]},{name:'OS placement',values:[g('decode',1024,t,0).mean,g('prefill',4096,t,0).mean]}]);
}
{
 const s=slide('Checking the Gated DeltaNet implementation',
 `The driver calls the actual upstream fused operator and tests it against an independent FP64 calculation. At each of one thousand steps, the inputs change and the check compares both the output and the carried recurrent state. Checking state matters because a small error may persist and influence subsequent steps. The multi-token check processes seventeen tokens as one graph and compares that result with seventeen single-token calls. The results matched exactly. These checks establish agreement for the tested shape and synthetic inputs. They make it possible to interpret throughput as the cost of the intended operator. The repository records the numerical errors and includes a command to repeat the check before benchmarking.`);
 text(s,'Independent FP64 recurrence over 1,000 changing-input steps\nCompared both output and carried state\n17-token graph matched 17 single-token calls exactly',70,180,1140,310,32);
}
{
 const s=slide('Recurrent state and attention cache',
 `The recurrent state contains thirty-two matrices, each with one hundred twenty-eight by one hundred twenty-eight FP32 entries. That gives two mebibytes per layer. The comparison uses an attention layer with sixteen key-value heads. Each cached token contributes both a key and a value, so the storage grows with the number of cached tokens. At thirty-two thousand tokens this is five hundred twelve mebibytes per layer. This is a comparison of persistent operator state. Model weights, temporary buffers, and process overhead belong to a separate memory accounting. That distinction explains why the resident set recorded during the experiment exceeds the recurrent state size.`);
 text(s,'Gated DeltaNet state stays at 2 MiB per layer\nFP32 attention KV cache grows with context length\n16 KV heads at 32k tokens use 512 MiB per layer\nProcess RSS includes buffers and runtime overhead',70,180,1140,310,32);
}
{
 const s=slide('What explains the measured CPU cost',
 `Single-token decode issues one graph call for each token. At that granularity the cost of dispatch and carrying state between calls can matter alongside the arithmetic. Prefill handles chunks of one hundred twenty-eight tokens and spreads those fixed costs across a larger amount of work. That helps explain the stronger scaling we observe for prefill. The roofline prediction assumes that counted bytes travel to DRAM on every step, and STREAM measures sustained sequential memory traffic. The real operator can reuse a portion of its state in cache. It also revisits that state across the five measured phases. In addition, it pays for synchronization and graph overhead. The gap therefore reflects several costs and cannot be assigned to a single cause from this experiment. For the affinity experiment we compared the same worker count and workload with pinning enabled or with operating system placement. The observed difference in decode was small relative to the spread across repeats. The raw runs preserve that variation. Together these experiments show why a fixed-size recurrent state is useful for capacity while the measured CPU time still depends on how the operator is called and scheduled.`);
 text(s,'Single-token calls pay graph dispatch and state-copy costs\nChunked prefill amortizes those costs\nSTREAM predicts a rate under a DRAM traffic assumption\nDecode affinity difference sits within repeat variation',70,180,1140,310,32);
}
{
 const s=slide('Shared PARAM Rudra reference',
 `The repository includes the exact source commits, the CPU-only build, the official reference functions, correctness evidence, raw repeated measurements, and the code that produces the figures and report. The peak working set remains within the assignment's budget. The main conclusions concern recurrent operator costs and thread scaling on this CPU. On PARAM Rudra, four CPU threads ran Llama 3.2 1B Instruct Q4_K_M with a 512-token prompt and 128-token decode. The combined rate was ${rudraRate} across three repeats. Slurm returned zero for peak RSS, so that measurement is unavailable. The course template and student recording also need to be completed. A separate sampled profiler trace would complement the scoped operator breakdown. Those remaining requirements are recorded explicitly so the submission can be finished without confusing missing evidence with a completed experiment.`);
 text(s,`Llama 3.2 1B Instruct Q4_K_M on four CPU threads\n512-token prompt + 128-token decode\n${rudraRate} across three repeats\nXeon Gold 6240R, node rdcn20, Slurm job 186881\nPeak RSS unavailable`,70,180,1140,310,32);
 text(s,'github.com/secureswim/cpu-inference-gdn-mamba3',70,580,1140,55,28,teal,true);
}

await fs.mkdir(path.join(root,'build/slides'),{recursive:true});
await fs.mkdir(path.join(root,'output'),{recursive:true});
const candidate=path.join(root,'build/slides/candidate.pptx');
await (await PresentationFile.exportPptx(presentation)).save(candidate);
const finalPath=path.join(root,process.env.DECK_OUTPUT || 'output/Project_1_Presentation.pptx');
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],
 explicitTotalSlideCount:12,requiredNativeChartOwnerSlides:[4,5,6,7,8],
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},
 verifyArtifactToolImport:true,receiptPath:path.join(root,`build/slides/${path.basename(finalPath)}.validation.json`)});
for(let i=0;i<presentation.slides.items.length;i++){
 const png=await presentation.export({slide:presentation.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(root,`build/slides/slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
}
let cursor=0;
let content='# Ten-minute talk script\n\nThe slide timings total 600 seconds. Rehearse with the plots and pauses. Record your own delivery in PowerPoint.\n\n';
for(const [index,s] of script.entries()){
 const clock=n=>`${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`;
 content+=`## Slide ${index+1} - ${s.title}\n\n${clock(cursor)} to ${clock(cursor+s.seconds)}\n\n${s.text}\n\n`;
 cursor+=s.seconds;
}
await fs.writeFile(path.join(root,'output/Talk_Script.md'),content.trimEnd()+'\n');
await fs.writeFile(path.join(root,'results/talk-timings.json'),JSON.stringify(script.map((s,i)=>({slide:i+1,title:s.title,seconds:s.seconds})),null,2));
console.log(finalPath);
