"""Export aggregate ensemble diagnostics as a self-contained HTML report."""
import json
from pathlib import Path


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fire Island ensemble report</title><style>
:root{--ink:#20231f;--muted:#687069;--line:#d7d5cd;--paper:#f2f0e8;--panel:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Georgia,"Times New Roman",serif}header{padding:18px 24px 12px;background:var(--panel);border-bottom:1px solid var(--line)}h1{margin:0;font-size:24px}header p{margin:4px 0 0;color:var(--muted);font-size:13px}
.shell{max-width:1340px;margin:auto;padding:16px}.summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.panel{background:var(--panel);border:1px solid var(--line);padding:12px}.panel h2{font-size:14px;text-transform:uppercase;letter-spacing:.07em;margin:0 0 10px}.stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 14px;font:12px/1.45 Consolas,monospace}.stats b{font-size:14px}.tables{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:10px}.stats-table{width:100%;border-collapse:collapse;font:11px Consolas,monospace}.stats-table th,.stats-table td{padding:4px 6px;border-bottom:1px solid #eee;text-align:left}.stats-table th{color:var(--muted);font-weight:normal}
.chart-tools{display:flex;gap:12px;align-items:center;margin-top:16px;color:var(--muted);font:11px Consolas,monospace}.chart-tools label{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.charts{margin-top:8px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.chart svg{display:block;width:100%;height:230px}.chart-note{margin-top:6px;color:var(--muted);font:10px/1.35 Consolas,monospace}.legend{display:flex;gap:8px;flex-wrap:wrap;font:10px Consolas,monospace;color:var(--muted)}.key{display:inline-flex;align-items:center;gap:3px}.sw{width:12px;height:3px;display:inline-block}.meta{margin-top:12px}.meta summary{cursor:pointer;color:var(--muted);font-size:12px}.meta pre{font:10px Consolas,monospace;white-space:pre-wrap;max-height:200px;overflow:auto}
@media(max-width:1100px){.summary,.tables{grid-template-columns:1fr}.charts{display:flex;overflow-x:auto}.chart{min-width:360px}}
</style></head><body>
<header><h1>Fire Island ensemble report</h1><p>Aggregated diagnostics across many seeds for debugging stability, deaths, and drift.</p></header>
<div class="shell"><section class="summary" id="summary"></section><div class="chart-tools"><label><input type="checkbox" id="hideBaseline"> hide baseline on death-source charts</label><span>baseline means ordinary cohort mortality from the survival schedule</span></div><section class="charts" id="charts"></section><details class="meta"><summary>Raw report payload</summary><pre id="meta"></pre></details></div>
<script>
const DATA=__DATA__;
const YEARS=DATA.years;
const scenarios=DATA.scenarios;
const SOURCE_COLORS={eruption:'#b33a3a',baseline:'#6b5b95',shortage:'#c76d28',storm:'#4f91a7',war:'#20231f'};
const GENOTYPE_COLORS={BB:'#7a4b2a',Bb:'#d9a05b',bb:'#2e75b6'};
const STAGE_COLORS={child:'#C76D28',adult:'#237A57',elder:'#6B5B95'};
document.getElementById('meta').textContent=JSON.stringify(DATA.provenance,null,2);
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function fmtPct(v){return `${Math.round(v*100)}%`}
function fmt3(v){return Number(v).toFixed(3)}
function fmt0(v){return Math.round(v).toString()}
function fmt1(v){return Number(v).toFixed(1)}
function label(v){return v.replaceAll('_',' ')}
function log1p10(v){return Math.log10(v+1)}

function summaryPanel(s){
  const deathsBySource=Object.entries(s.summary.final_deaths_by_source).map(([name,value])=>`<tr><td>${label(name)}</td><td>${fmt0(value)}</td></tr>`).join('');
  const deathsByGenotype=Object.entries(s.summary.final_deaths_by_genotype).map(([name,value])=>`<tr><td>${name}</td><td>${fmt0(value)}</td></tr>`).join('');
  const traitRows=s.summary.trait_success.slice(0,8).map(item=>`<tr><td>${label(item.trait)}</td><td>${fmtPct(item.median_share)}</td><td>${fmt0(item.median_population)}</td></tr>`).join('');
  const dominantRows=s.dominant_clan_runs.map(item=>`<tr><td>${item.seed}</td><td>${esc(item.clan)}</td><td>${fmt0(item.population)}</td><td>${esc(item.traits.map(label).join(', ')||'neutral')}</td></tr>`).join('');
  const dominantSummary=s.summary.dominant_clans.map(item=>`<tr><td>${esc(item.clan)}</td><td>${item.runs}/${s.count}</td><td>${fmt0(item.median_population)}</td></tr>`).join('');
  return `<section class="panel"><h2>${esc(s.label)}</h2><div class="stats"><span>runs</span><b>${s.count}</b><span>final pop median</span><b>${fmt0(s.summary.final_population_median)}</b><span>final pop 10-90%</span><b>${fmt0(s.summary.final_population_p10)}-${fmt0(s.summary.final_population_p90)}</b><span>minimum pop median</span><b>${fmt0(s.summary.minimum_population_median)}</b><span>minimum food median</span><b>${fmt3(s.summary.minimum_food_security_median)}</b><span>final q median</span><b>${fmt3(s.summary.final_q_median)}</b><span>FST median</span><b>${fmt3(s.summary.fst_median)}</b></div><div class="tables"><div><h3>Median deaths by source</h3><table class="stats-table"><thead><tr><th>source</th><th>deaths</th></tr></thead><tbody>${deathsBySource}</tbody></table></div><div><h3>Median deaths by genotype</h3><table class="stats-table"><thead><tr><th>genotype</th><th>deaths</th></tr></thead><tbody>${deathsByGenotype}</tbody></table></div><div><h3>Traits That Prospered Most</h3><table class="stats-table"><thead><tr><th>trait</th><th>median share</th><th>median pop</th></tr></thead><tbody>${traitRows}</tbody></table></div><div><h3>Dominant Clan Frequency</h3><table class="stats-table"><thead><tr><th>clan</th><th>runs</th><th>median final pop</th></tr></thead><tbody>${dominantSummary}</tbody></table></div></div><div style="margin-top:10px"><h3>Dominant clan at end of each run</h3><table class="stats-table"><thead><tr><th>seed</th><th>clan</th><th>final pop</th><th>traits</th></tr></thead><tbody>${dominantRows}</tbody></table></div></section>`;
}
document.getElementById('summary').innerHTML=scenarios.map(summaryPanel).join('');

const CW=350,CH=190,L=35,R=8,T=12,B=22;
function chartX(i,length){return L+i/Math.max(length-1,1)*(CW-L-R)}
function seriesMaxFromBands(seriesMap, transform=(value)=>value){return Math.max(...Object.values(seriesMap).flatMap(band=>band.p90.map(transform)),1)}
function baselineToggle(){const hide=document.getElementById('hideBaseline').checked;document.querySelectorAll('[data-series-key="baseline"]').forEach(el=>{el.style.display=hide?'none':''})}

function makeBandChart(title,key,format,options={}){
  const note=options.note?`<div class="chart-note">${esc(options.note)}</div>`:'';
  const wrap=document.createElement('section');wrap.className='panel chart';wrap.innerHTML=`<h2>${title}</h2><svg viewBox="0 0 ${CW} ${CH}"></svg><div class="legend"></div>${note}`;
  const svg=wrap.querySelector('svg');
  const transform=options.transform||((value)=>value);
  const yMax=Math.max(...scenarios.flatMap(s=>s.series[key].p90.map(transform)),1);
  for(let n=0;n<=4;n++){const y=T+n*(CH-T-B)/4;svg.innerHTML+=`<line x1="${L}" y1="${y}" x2="${CW-R}" y2="${y}" stroke="#e5e3dc"/><text x="2" y="${y+3}" font-size="9">${format(yMax*(1-n/4))}</text>`}
  scenarios.forEach(s=>{const band=s.series[key];const upper=band.p90.map((v,i)=>`${chartX(i,YEARS.length)},${T+(1-transform(v)/yMax)*(CH-T-B)}`).join(' ');const lower=[...band.p10].reverse().map((v,i)=>`${chartX(YEARS.length-1-i,YEARS.length)},${T+(1-transform(v)/yMax)*(CH-T-B)}`).join(' ');svg.innerHTML+=`<polygon points="${upper} ${lower}" fill="${s.color}" opacity="0.15"/><polyline points="${band.p50.map((v,i)=>`${chartX(i,YEARS.length)},${T+(1-transform(v)/yMax)*(CH-T-B)}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2"/>`});
  wrap.querySelector('.legend').innerHTML=scenarios.map(s=>`<span class="key"><i class="sw" style="background:${s.color}"></i>${esc(s.label)} median with 10-90% band</span>`).join('');
  return wrap;
}

function makeMultiLineChart(title,seriesMap,colors,format,options={}){
  const entries=Object.entries(seriesMap);
  const note=options.note?`<div class="chart-note">${esc(options.note)}</div>`:'';
  const wrap=document.createElement('section');wrap.className='panel chart';wrap.innerHTML=`<h2>${title}</h2><svg viewBox="0 0 ${CW} ${CH}"></svg><div class="legend"></div>${note}`;
  const svg=wrap.querySelector('svg');
  const transform=options.transform||((value)=>value);
  const yMax=seriesMaxFromBands(seriesMap, transform);
  for(let n=0;n<=4;n++){const y=T+n*(CH-T-B)/4;svg.innerHTML+=`<line x1="${L}" y1="${y}" x2="${CW-R}" y2="${y}" stroke="#e5e3dc"/><text x="2" y="${y+3}" font-size="9">${format(yMax*(1-n/4))}</text>`}
  entries.forEach(([name,band])=>{svg.innerHTML+=`<polyline points="${band.p50.map((v,i)=>`${chartX(i,YEARS.length)},${T+(1-transform(v)/yMax)*(CH-T-B)}`).join(' ')}" fill="none" stroke="${colors[name]}" stroke-width="2" data-series-key="${esc(name)}"/>`});
  wrap.querySelector('.legend').innerHTML=entries.map(([name])=>`<span class="key" data-series-key="${esc(name)}"><i class="sw" style="background:${colors[name]}"></i>${esc(label(name))}</span>`).join('');
  return wrap;
}

const chartRoot=document.getElementById('charts');
chartRoot.append(
  makeBandChart('Island population', 'population', fmt0),
  makeBandChart('Blue-allele frequency', 'q', fmtPct),
  makeBandChart('Minimum valley food security', 'min_food_security', fmtPct),
  makeMultiLineChart(`${scenarios[0].label} deaths by source (log scale)`, scenarios[0].deaths_by_source, SOURCE_COLORS, v=>fmt0(10**v-1), {note:'Log scale keeps small violent-loss series visible beside baseline.', transform:log1p10}),
  makeMultiLineChart(`${scenarios[0].label} deaths per 1000 people by source`, scenarios[0].deaths_by_source_rate, SOURCE_COLORS, fmt1, {note:'Uses the previous-turn island population as denominator.'}),
  makeMultiLineChart(`${scenarios[0].label} baseline deaths per turn by life stage`, scenarios[0].baseline_deaths_by_stage, STAGE_COLORS, fmt0),
  makeMultiLineChart(`${scenarios[0].label} deaths by genotype`, scenarios[0].deaths_by_genotype, GENOTYPE_COLORS, fmt0),
  makeMultiLineChart(`${scenarios[1].label} deaths by source (log scale)`, scenarios[1].deaths_by_source, SOURCE_COLORS, v=>fmt0(10**v-1), {note:'Control scenario for comparison against the volcanic run.', transform:log1p10}),
  makeMultiLineChart(`${scenarios[1].label} deaths per 1000 people by source`, scenarios[1].deaths_by_source_rate, SOURCE_COLORS, fmt1)
);
document.getElementById('hideBaseline').addEventListener('change', baselineToggle);
baselineToggle();
</script></body></html>"""


def export_ensemble_html(data: dict, path: str = "island_experiments.html") -> str:
    html = _TEMPLATE.replace("__DATA__", json.dumps(data))
    output = Path(path)
    output.write_text(html, encoding="utf-8")
    return str(output.resolve())