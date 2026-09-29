"""Export completed simulations as self-contained HTML reports."""
import json
from pathlib import Path
from typing import List

VALLEY_COLORS = ["#237A57", "#C76D28", "#6B5B95", "#B33A3A"]

_REPLAY_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fire Island replay</title><style>
:root{--ink:#20231f;--muted:#687069;--line:#d7d5cd;--paper:#f2f0e8;--panel:#fff;--sea:#e7f1f3;--path:#987b55;--canoe:#4f91a7}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:Georgia,"Times New Roman",serif}button,select,input{font:inherit}
header{padding:18px 24px 12px;background:var(--panel);border-bottom:1px solid var(--line)}h1{margin:0;font-size:24px;letter-spacing:0}header p{margin:4px 0 0;color:var(--muted);font-size:13px}
.controls{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:10px;padding:10px 24px;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line)}.icon{width:36px;height:34px;border:1px solid var(--line);background:white;cursor:pointer}.controls input{flex:1}.time{min-width:145px;text-align:right;font:13px Consolas,monospace}
.shell{max-width:1340px;margin:auto;padding:16px}.top{display:grid;grid-template-columns:680px 1fr;gap:16px}.panel{background:var(--panel);border:1px solid var(--line);padding:12px}.panel h2{font-size:14px;text-transform:uppercase;letter-spacing:.07em;margin:0 0 10px}.panel h3{font-size:13px;margin:0 0 8px}
.map{position:relative;width:640px;height:480px;margin:auto;background:radial-gradient(circle at center,#efe6d4 0 16%,transparent 17%),var(--sea);overflow:hidden}.routes{position:absolute;inset:0;width:100%;height:100%}.path{stroke:var(--path);stroke-width:3}.canoe{stroke:var(--canoe);stroke-width:2;stroke-dasharray:7 6}.event-route{stroke:#c02f2f;stroke-width:5;stroke-linecap:round;opacity:.85}.event-route.canoe-event{stroke:#167b9c;stroke-dasharray:8 5}
.volcano{position:absolute;left:264px;top:188px;width:112px;height:104px;text-align:center;padding-top:20px;font-size:43px}.volcano small{display:block;font:12px Georgia;color:#743c26}.valley{position:absolute;width:210px;min-height:154px;padding:10px;border:3px solid var(--color);background:rgba(255,255,255,.94);box-shadow:0 2px 7px #46544a24}.v0{left:215px;top:14px}.v1{right:14px;top:174px}.v2{left:215px;bottom:14px}.v3{left:14px;top:174px}.valley h3{font-size:15px;margin:0 0 6px;display:flex;justify-content:space-between}.badge{font-size:20px;min-width:26px}.line{display:flex;justify-content:space-between;font:12px/1.45 Consolas,monospace}.bars{display:grid;grid-template-columns:40px 1fr;gap:4px 6px;margin-top:6px;font:10px Consolas,monospace}.bar{height:8px;background:#e6e5df;overflow:hidden}.bar i{display:block;height:100%}.clans{margin-top:8px;font:10px/1.4 Consolas,monospace;color:var(--muted);white-space:normal}
.legend{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;color:var(--muted);font-size:11px}.events{height:440px;overflow:auto;font:12px/1.4 Consolas,monospace}.event{padding:7px 6px;border-bottom:1px solid #eee;cursor:pointer}.event:hover,.event.current{background:#f5e9bd}.empty{color:var(--muted)}
.traits{margin-top:16px}.trait-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}.trait-table{width:100%;border-collapse:collapse;font:11px Consolas,monospace}.trait-table th,.trait-table td{padding:4px 6px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}.trait-table th{color:var(--muted);font-weight:normal}.trait-name{font-family:Georgia,"Times New Roman",serif;text-transform:capitalize}.scroll-list{max-height:300px;overflow:auto;border-top:1px solid #eee;border-bottom:1px solid #eee}.stat-list{font:11px/1.45 Consolas,monospace}.stat-row{display:grid;grid-template-columns:1fr auto;gap:8px;padding:6px 4px;border-bottom:1px solid #eee}.stat-row:last-child{border-bottom:0}.stat-row small{display:block;color:var(--muted);font-size:10px}.stack{display:grid;gap:12px}
.pie-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:8px}.pie-card{border:1px solid #eee;padding:10px}.pie-card h3{display:flex;justify-content:space-between;gap:8px}.pie-wrap{display:grid;grid-template-columns:110px 1fr;gap:12px;align-items:center}.pie{width:100px;height:100px;border-radius:50%;position:relative;border:1px solid #ddd}.pie::after{content:'';position:absolute;inset:22px;border-radius:50%;background:var(--panel);border:1px solid #eee}.pie-legend{display:grid;gap:4px}.pie-row{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center}
.chart-tools{display:flex;gap:12px;align-items:center;margin-top:16px;color:var(--muted);font:11px Consolas,monospace}.chart-tools label{display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.charts{margin-top:8px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.chart{min-width:0}.chart svg{display:block;width:100%;height:230px}.chart.bar-chart svg{height:auto;min-height:230px}.chart-note{margin-top:6px;color:var(--muted);font:10px/1.35 Consolas,monospace}.chart-legend{display:flex;gap:8px;flex-wrap:wrap;font:10px Consolas,monospace;color:var(--muted)}.key{display:inline-flex;align-items:center;gap:3px}.sw{width:12px;height:3px;display:inline-block}.marker-glyph{font:12px Consolas,monospace}.meta{margin-top:12px}.meta summary{cursor:pointer;color:var(--muted);font-size:12px}.meta pre{font:10px Consolas,monospace;white-space:pre-wrap;max-height:180px;overflow:auto}
@media(max-width:1100px){.shell{padding:8px}.top{display:block}.top>.panel:first-child{overflow-x:auto}.map{transform-origin:top left}.events{height:240px}.trait-grid{grid-template-columns:1fr}.pie-grid{grid-template-columns:1fr 1fr}.charts{display:flex;overflow-x:auto}.chart{min-width:360px}.controls{padding:8px}.time{min-width:100px}.controls select{max-width:74px}}
@media(max-width:760px){.pie-grid{grid-template-columns:1fr}.pie-wrap{grid-template-columns:1fr}.pie{margin:auto}}
</style></head><body>
<header><h1>Fire Island</h1><p>Replay with clan lifecycle and mortality debugging overlays.</p></header>
<div class="controls"><button class="icon" id="play" title="Play">&#9654;</button><input id="slider" type="range" min="0" value="0"><select id="speed" title="Playback speed"><option value="650">0.5x</option><option value="320" selected>1x</option><option value="120">3x</option></select><span class="time" id="time"></span></div>
<div class="shell"><div class="top"><section class="panel"><h2>Island</h2><div class="map" id="map"><svg class="routes" viewBox="0 0 640 480"><line class="path" x1="320" y1="80" x2="550" y2="240"/><line class="path" x1="550" y1="240" x2="320" y2="410"/><line class="path" x1="320" y1="410" x2="90" y2="240"/><line class="path" x1="90" y1="240" x2="320" y2="80"/><line class="canoe" x1="320" y1="80" x2="320" y2="410"/><line class="canoe" x1="90" y1="240" x2="550" y2="240"/><g id="eventRoutes"></g></svg><div class="volcano">🌋<small>central volcano</small></div><div id="valleys"></div></div><div class="legend"><span>solid: mountain path</span><span>dashed: canoe route</span><span>ancestry markers: BB / Bb / bb</span><span>events show for selected turn</span></div></section><aside class="panel"><h2>Event log</h2><div class="events" id="events"></div></aside></div><section class="panel traits"><h2>Politics and habitability</h2><div class="trait-grid" id="traits"></div></section><section class="panel"><h2>Clan distribution by valley</h2><div class="pie-grid" id="clanPies"></div></section><div class="chart-tools"><label><input type="checkbox" id="hideBaseline"> hide baseline on death-source charts</label><span>baseline means ordinary cohort mortality from the survival schedule</span></div><section class="charts" id="charts"></section><details class="meta"><summary>Run provenance</summary><pre id="meta"></pre></details></div>
<script>
const DATA=__DATA__,COLORS=__COLORS__,history=DATA.history,events=DATA.events,names=DATA.valleyOrder;
const POS=[[320,80],[550,240],[320,410],[90,240]];
const ICONS={eruption:'🌋',war:'⚔️',migration:'🚶',storm:'⛈️',refugees:'🧳',clan_split:'✂️',clan_merge:'🧬',clan_absorbed:'⇋',clan_extinct:'☠️',clan_arrival:'🚢',clan_departure:'⇢',clan_expedition:'⛵',succession_crisis:'👑'};
const DEATH_SOURCE_COLORS={eruption:'#b33a3a',baseline:'#6b5b95',shortage:'#c76d28',storm:'#4f91a7',war:'#20231f'};
const GENOTYPE_COLORS={BB:'#7a4b2a',Bb:'#d9a05b',bb:'#2e75b6'};
const STAGE_COLORS={child:'#C76D28',adult:'#237A57',elder:'#6B5B95'};
const CLAN_COLORS=['#237A57','#C76D28','#6B5B95','#B33A3A','#587291','#9A6D38','#3E8A8A','#884E72','#6F7C35','#A35D5D'];
const slider=document.getElementById('slider'),time=document.getElementById('time'),valleyRoot=document.getElementById('valleys');
slider.max=history.length-1;document.getElementById('meta').textContent=JSON.stringify(DATA.provenance,null,2);
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function pct(v){return Math.round(v*100)}
function label(v){return v.replaceAll('_',' ')}
function signed(v){return `${v>=0?'+':''}${Math.round(v*100)}%`}
function clanColor(name){let hash=0;for(const ch of name){hash=(hash*31+ch.charCodeAt(0))>>>0}return CLAN_COLORS[hash%CLAN_COLORS.length]}
function log1p10(v){return Math.log10(v+1)}
function clanList(summary){return (summary||[]).slice(0,3).map(item=>`${item.clan} ${item.population}`).join(', ')}
function eventText(ev){
  if(ev.type==='migration')return `y${ev.year} migration ${ev.from} → ${ev.to} ${ev.count}`;
  if(ev.type==='storm')return `y${ev.year} storm ${ev.from} → ${ev.to} lost ${ev.lost}`;
  if(ev.type==='refugees')return `y${ev.year} refugees ${ev.from} → ${ev.to} ${ev.count}`;
  if(ev.type==='eruption')return `y${ev.year} eruption ${ev.valley} deaths ${ev.deaths}`;
  if(ev.type==='war')return `y${ev.year} war ${ev.valley} ${clanList(ev.winner_clans)||ev.winner} beat ${clanList(ev.loser_clans)||'rivals'} deaths ${ev.deaths} refugees ${ev.refugees}`;
  if(ev.type==='clan_merge')return `y${ev.year} clan merge ${ev.first} + ${ev.second} → ${ev.new_clan}`;
  if(ev.type==='clan_split')return `y${ev.year} clan split ${ev.from_clan} → ${ev.new_clan} ${ev.count}`;
  if(ev.type==='clan_absorbed')return `y${ev.year} clan absorbed ${ev.from_clan} → ${ev.to_clan} ${ev.count}`;
  if(ev.type==='clan_extinct')return `y${ev.year} clan extinct ${ev.tribe} total deaths ${ev.total_deaths}`;
  if(ev.type==='clan_arrival')return `y${ev.year} arrival ${ev.tribe} at ${ev.valley} ${ev.count} depart by y${ev.departure_year}`;
  if(ev.type==='clan_departure')return `y${ev.year} departure ${ev.tribe} ${ev.count}`;
  if(ev.type==='clan_expedition')return `y${ev.year} expedition ${ev.tribe} ${ev.from} → ${ev.to} ${ev.count}`;
  if(ev.type==='succession_crisis')return `y${ev.year} succession crisis ${ev.clan} fractured into ${ev.factions} factions`;
  return `y${ev.year} ${label(ev.type)}`;
}
function incremental(values){return values.map((value,index)=>value-(index?values[index-1]:0))}
function per1000(events, population){return events.map((value,index)=>index===0||population[index-1]<=0?0:value*1000/population[index-1])}
function clampUnit(value){return Math.min(Math.max(value,0),1)}
function currentIslandPopulationSeries(){return history.map(snapshot=>snapshot.totals.population||snapshot.valleys.reduce((sum,valley)=>sum+valley.population,0))}
function seriesMax(collections){return Math.max(...collections.flatMap(item=>item.values),1)}
function eventBadge(turn,name){return [...new Set(events.filter(e=>e.turn===turn&&(e.valley===name||e.from===name||e.to===name)).map(e=>ICONS[e.type]||''))].join('')}

const eventRows=[];
if(!events.length)document.getElementById('events').innerHTML='<div class="empty">No events</div>';
events.forEach(ev=>{const row=document.createElement('div');row.className='event';row.textContent=`${ICONS[ev.type]||'•'} ${eventText(ev)}`;row.onclick=()=>{slider.value=ev.turn;render(ev.turn)};document.getElementById('events').appendChild(row);eventRows.push([ev.turn,row])});

function renderTraits(index){
  const snapshot=history[index];
  const genes=DATA.provenance.config.traits.genes;
  const clanTraits=DATA.provenance.config.traits.clans||{};
  const geneRows=Object.entries(genes).map(([name,values])=>`<tr><td class="trait-name">${label(name)}</td>${values.map(value=>`<td>${signed(value)}</td>`).join('')}</tr>`).join('');
  const populationByClan=snapshot.totals.population_by_clan||{};
  const totalPopulation=snapshot.totals.population||0;
  const dominantClanRows=Object.entries(populationByClan).sort((a,b)=>b[1]-a[1]).slice(0,12).map(([name,population])=>{
    const traits=Object.entries(clanTraits[name]||{}).filter(([,value])=>value>0.01).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([key,value])=>`${label(key)} ${signed(value)}`).join(', ')||'neutral';
    const share=totalPopulation?Math.round(population*100/totalPopulation):0;
    return `<div class="stat-row"><div><strong>${esc(name)}</strong><small>${share}% of island population</small><small>${esc(traits)}</small></div><b>${population}</b></div>`;
  }).join('')||'<div class="empty">No clans</div>';
  const traitTotals=new Map();
  Object.entries(populationByClan).forEach(([name,population])=>{
    Object.entries(clanTraits[name]||{}).forEach(([key,value])=>{
      if(value<=0)return;
      traitTotals.set(key,(traitTotals.get(key)||0)+population*value);
    });
  });
  const dominantTraitRows=[...traitTotals.entries()].sort((a,b)=>b[1]-a[1]).slice(0,12).map(([key,weighted])=>{
    const share=totalPopulation?weighted/totalPopulation:0;
    return `<div class="stat-row"><div><strong>${esc(label(key))}</strong><small>population-weighted cultural pull</small></div><b>${signed(share)}</b></div>`;
  }).join('')||'<div class="empty">No dominant traits</div>';
  const habitabilityRows=snapshot.valleys.slice().sort((a,b)=>b.habitability-a.habitability).map(valley=>{
    const traits=(valley.dominant_traits||[]).slice(0,2).map(item=>`${label(item.trait)} ${signed(item.score)}`).join(', ')||'neutral';
    return `<div class="stat-row"><div><strong>${esc(valley.name)}</strong><small>dominant clan: ${esc(valley.dominant_clan||'none')}</small><small>${esc(traits)}</small></div><b>${pct(valley.habitability)}%</b></div>`;
  }).join('');
  document.getElementById('traits').innerHTML=`<div class="stack"><div><h3>Ancestry markers</h3><table class="trait-table"><thead><tr><th>effect</th><th>BB</th><th>Bb</th><th>bb</th></tr></thead><tbody>${geneRows}</tbody></table></div><div><h3>Dominant island traits</h3><div class="scroll-list stat-list">${dominantTraitRows}</div></div></div><div class="stack"><div><h3>Largest clans</h3><div class="scroll-list stat-list">${dominantClanRows}</div></div></div><div class="stack"><div><h3>Valley habitability</h3><div class="scroll-list stat-list">${habitabilityRows}</div></div></div>`;
}

function pieGradient(items,total){
  if(total<=0||!items.length)return '#e6e5df';
  let angle=0;
  return `conic-gradient(${items.map(item=>{
    const start=angle;
    angle += item.population/total*360;
    return `${item.color} ${start}deg ${angle}deg`;
  }).join(',')})`;
}

function renderClanPies(index){
  const snapshot=history[index];
  document.getElementById('clanPies').innerHTML=snapshot.valleys.map(valley=>{
    const items=(valley.clan_distribution||[]).map(item=>({
      clan:item.clan,
      population:item.population,
      color:item.clan==='Other'?'#c9c4b8':clanColor(item.clan),
    }));
    const total=items.reduce((sum,item)=>sum+item.population,0);
    const legend=items.map(item=>`<div class="pie-row"><span class="key"><i class="sw" style="background:${item.color}"></i>${esc(item.clan)}</span><b>${total?Math.round(item.population*100/total):0}%</b></div>`).join('')||'<div class="empty">No clans</div>';
    return `<div class="pie-card"><h3><span>${esc(valley.name)}</span><span>${valley.population}</span></h3><div class="pie-wrap"><div class="pie" style="background:${pieGradient(items,total)}"></div><div class="stat-list pie-legend">${legend}</div></div></div>`;
  }).join('');
}

function bars(v){return `<div class="bars"><span>genes</span><div class="bar"><i style="width:${v.population?Math.max(0,v.n_BB/v.population*100):0}%;background:#7a4b2a;float:left"></i><i style="width:${v.population?Math.max(0,v.n_Bb/v.population*100):0}%;background:#d9a05b;float:left"></i><i style="width:${v.population?Math.max(0,v.n_bb/v.population*100):0}%;background:#2e75b6;float:left"></i></div><span>food</span><div class="bar"><i style="width:${Math.min(v.food_security,1)*100}%;background:#3d8f55"></i></div><span>land</span><div class="bar"><i style="width:${v.land_health*100}%;background:#81734c"></i></div></div>`}

function renderRoutes(turn){const root=document.getElementById('eventRoutes');root.innerHTML='';const seen=new Set();events.filter(e=>e.turn===turn&&e.from&&e.to).forEach(e=>{const a=names.indexOf(e.from),b=names.indexOf(e.to),key=[Math.min(a,b),Math.max(a,b),e.route].join(':');if(a<0||b<0||seen.has(key))return;seen.add(key);const line=document.createElementNS('http://www.w3.org/2000/svg','line');line.setAttribute('x1',POS[a][0]);line.setAttribute('y1',POS[a][1]);line.setAttribute('x2',POS[b][0]);line.setAttribute('y2',POS[b][1]);line.setAttribute('class','event-route '+(e.route==='canoe'?'canoe-event':''));root.appendChild(line)})}

function traitSummary(valley){
  return (valley.dominant_traits||[]).slice(0,2).map(item=>`${label(item.trait)} ${signed(item.score)}`).join(' · ')||'neutral';
}

function render(index){
  const snapshot=history[index];
  time.textContent=`turn ${snapshot.turn} · year ${snapshot.year}`;
  valleyRoot.innerHTML=snapshot.valleys.map((v,j)=>`<article class="valley v${j}" style="--color:${COLORS[j]}"><h3><span>${esc(v.name)}</span><span class="badge">${eventBadge(snapshot.turn,v.name)}</span></h3><div class="line"><span>population</span><b>${v.population}</b></div><div class="line"><span>blue allele q</span><b>${v.q===null?'n/a':v.q.toFixed(3)}</b></div><div class="line"><span>food / land</span><b>${pct(v.food_security)}% / ${pct(v.land_health)}%</b></div><div class="line"><span>habitability</span><b>${pct(v.habitability)}%</b></div>${bars(v)}<div class="clans"><strong>dominant clan</strong> ${esc(v.dominant_clan||'none')}<br><strong>dominant traits</strong> ${esc(traitSummary(v))}</div></article>`).join('');
  renderTraits(index);
  renderClanPies(index);
  renderRoutes(snapshot.turn);
  eventRows.forEach(([turn,row])=>row.classList.toggle('current',turn===snapshot.turn));
  document.querySelectorAll('.marker').forEach(m=>{m.setAttribute('x1',chartX(index,m.dataset.length));m.setAttribute('x2',chartX(index,m.dataset.length))});
}

const CW=350,CH=190,L=35,R=8,T=12,B=22;
function chartX(i,length){const count=Math.max((length?+length:history.length)-1,1);return L+i/count*(CW-L-R)}

function makeLineChart(title,series,yMax,format,options={}){
  const note=options.note?`<div class="chart-note">${esc(options.note)}</div>`:'';
  const wrap=document.createElement('section');wrap.className='panel chart';wrap.innerHTML=`<h2>${title}</h2><svg viewBox="0 0 ${CW} ${CH}"></svg><div class="chart-legend"></div>${note}`;
  const svg=wrap.querySelector('svg');
  const scaleValue=options.scaleValue||((value,max)=>max?Math.min(value,max)/max:0);
  for(let n=0;n<=4;n++){const y=T+n*(CH-T-B)/4;svg.innerHTML+=`<line x1="${L}" y1="${y}" x2="${CW-R}" y2="${y}" stroke="#e5e3dc"/><text x="2" y="${y+3}" font-size="9">${format(yMax*(1-n/4))}</text>`}
  series.forEach(item=>{
    const key=item.key||'';
    const points=item.values.map((v,i)=>`${chartX(i,item.values.length)},${T+(1-clampUnit(scaleValue(v,yMax)))*(CH-T-B)}`).join(' ');
    svg.innerHTML+=`<polyline points="${points}" fill="none" stroke="${item.color}" stroke-width="2" data-series-key="${esc(key)}"/>`;
    if(options.seriesMarkers){
      options.seriesMarkers(item).forEach(marker=>{
        const value=item.values[marker.index]??0;
        const x=chartX(marker.index,item.values.length);
        const y=T+(1-clampUnit(scaleValue(value,yMax)))*(CH-T-B);
        svg.innerHTML+=`<text x="${x}" y="${y-4}" text-anchor="middle" fill="${marker.color||item.color}" class="marker-glyph" data-series-key="${esc(key)}" title="${esc(marker.title||'')}">${esc(marker.glyph)}</text>`;
      });
    }
  });
  const marker=document.createElementNS('http://www.w3.org/2000/svg','line');marker.setAttribute('class','marker');marker.dataset.length=String(history.length);marker.setAttribute('y1',T);marker.setAttribute('y2',CH-B);marker.setAttribute('stroke','#222');svg.appendChild(marker);
  wrap.querySelector('.chart-legend').innerHTML=series.map(item=>`<span class="key" data-series-key="${esc(item.key||'')}"><i class="sw" style="background:${item.color}"></i>${esc(item.name)}</span>`).join('');
  return wrap;
}

function makeBarChart(title,items){
  const rowHeight=24,padTop=10,padBottom=18,labelWidth=120,barLeft=labelWidth+10,chartWidth=350,usableWidth=chartWidth-barLeft-60,maxValue=Math.max(...items.map(item=>item.value),1),height=padTop+padBottom+items.length*rowHeight;
  const wrap=document.createElement('section');wrap.className='panel chart bar-chart';wrap.innerHTML=`<h2>${title}</h2><svg viewBox="0 0 ${chartWidth} ${height}"></svg><div class="chart-legend"></div>`;
  const svg=wrap.querySelector('svg');
  items.forEach((item,index)=>{const y=padTop+index*rowHeight;const width=item.value/maxValue*usableWidth;svg.innerHTML+=`<text x="0" y="${y+11}" font-size="10">${esc(item.name.slice(0,18))}</text><rect x="${barLeft}" y="${y}" width="${width}" height="12" fill="${item.color}"/><text x="${barLeft+width+4}" y="${y+11}" font-size="10">${item.value}</text>`});
  wrap.querySelector('.chart-legend').innerHTML=items.map(item=>`<span class="key"><i class="sw" style="background:${item.color}"></i>${esc(item.name)}${item.note?` (${esc(item.note)})`:''}</span>`).join('');
  return wrap;
}

function topClanNames(limit){
  return (DATA.trackedClans||[]).slice(0,limit);
}

function clanPopulationSeries(clan){return history.map(snapshot=>snapshot.totals.population_by_clan[clan]||0)}
function clanCumulativeDeathSeries(clan){return history.map(snapshot=>snapshot.totals.deaths_by_clan[clan]||0)}

function baselineToggle(){
  const hide=document.getElementById('hideBaseline').checked;
  document.querySelectorAll('[data-series-key="baseline"]').forEach(el=>{el.style.display=hide?'none':''});
}

const qSeries=names.map((name,j)=>({name,key:name,color:COLORS[j],values:history.map(s=>s.valleys[j].q??0)}));
const popSeries=names.map((name,j)=>({name,key:name,color:COLORS[j],values:history.map(s=>s.valleys[j].population)}));
const foodSeries=names.map((name,j)=>({name,key:name,color:COLORS[j],values:history.map(s=>s.valleys[j].food_security)}));
const meanLand={name:'mean land',key:'mean_land',color:'#222',values:history.map(s=>s.valleys.reduce((a,v)=>a+v.land_health,0)/s.valleys.length)};
foodSeries.push(meanLand);
const islandPopulationSeries=currentIslandPopulationSeries();
const finalTotals=history[history.length-1].totals;
const deathSourceSeries=Object.entries(DEATH_SOURCE_COLORS).map(([name,color])=>({name:label(name),key:name,color,values:history.map(s=>s.totals.deaths_by_source[name]||0)}));
const deathSourceLogMax=Math.max(...deathSourceSeries.flatMap(item=>item.values.map(log1p10)),1);
const incidentDeathSourceSeries=Object.entries(DEATH_SOURCE_COLORS).map(([name,color])=>({name:label(name),key:name,color,values:incremental(history.map(s=>s.totals.deaths_by_source[name]||0))})).filter(item=>item.values.some(value=>value>0));
const deathSourceRateSeries=incidentDeathSourceSeries.map(item=>({name:item.name,key:item.key,color:item.color,values:per1000(item.values,islandPopulationSeries)}));
const baselineStageIncidentSeries=Object.entries(STAGE_COLORS).map(([stage,color])=>({name:label(stage),key:stage,color,values:incremental(history.map(s=>s.totals.baseline_deaths_by_stage[stage]||0))}));
const genotypePopulationSeries=Object.entries(GENOTYPE_COLORS).map(([name,color])=>({name,key:name,color,values:history.map(snapshot=>snapshot.valleys.reduce((sum,valley)=>sum+(valley[`n_${name}`]||0),0))}));
const clanBars=Object.entries(finalTotals.deaths_by_clan).filter(([,value])=>value>0).sort((a,b)=>b[1]-a[1]).map(([name,value])=>{const life=finalTotals.clan_lifecycles[name]||{};const note=life.extinct_year===null||life.extinct_year===undefined?`active, born y${life.birth_year??0}`:`y${life.birth_year??0}–${life.extinct_year}`;return {name,value,color:clanColor(name),note}});
const trackedClans=topClanNames(8);
const clanPopulationChartSeries=trackedClans.map(name=>({name,key:name,color:clanColor(name),values:clanPopulationSeries(name),life:finalTotals.clan_lifecycles[name]||{}}));
const clanBbPopulationSeries=trackedClans.map(name=>({name,key:name,color:clanColor(name),values:history.map(snapshot=>snapshot.totals.bb_population_by_clan[name]||0),life:finalTotals.clan_lifecycles[name]||{}}));
const clanDeathSeries=trackedClans.map(name=>({name,key:name,color:clanColor(name),values:incremental(clanCumulativeDeathSeries(name)),life:finalTotals.clan_lifecycles[name]||{}})).filter(item=>item.values.some(value=>value>0));
function clanMarkers(item){const markers=[];if(item.life.birth_turn>0)markers.push({index:item.life.birth_turn,glyph:'V',title:`${item.name} formed in year ${item.life.birth_year}`});if(item.life.extinct_turn!==null&&item.life.extinct_turn!==undefined)markers.push({index:item.life.extinct_turn,glyph:'X',title:`${item.name} extinct in year ${item.life.extinct_year}`});return markers}
const maxPop=Math.max(...popSeries.flatMap(item=>item.values),1);
const maxDeathSourceRate=Math.max(...deathSourceRateSeries.flatMap(item=>item.values),1);
const maxBaselineStageIncident=Math.max(...baselineStageIncidentSeries.flatMap(item=>item.values),1);
const maxGenotypePopulation=Math.max(...genotypePopulationSeries.flatMap(item=>item.values),1);
const maxClanPopulation=Math.max(...clanPopulationChartSeries.flatMap(item=>item.values),1);
const maxClanBbPopulation=Math.max(...clanBbPopulationSeries.flatMap(item=>item.values),1);
const maxClanDeaths=Math.max(...clanDeathSeries.flatMap(item=>item.values),1);
document.getElementById('charts').append(
  makeLineChart('Blue-allele frequency',qSeries,1,v=>Math.round(v*100)+'%'),
  makeLineChart('Population',popSeries,maxPop,v=>Math.round(v)),
  makeLineChart('Food security / mean land',foodSeries,1.2,v=>Math.round(v*100)+'%'),
  makeLineChart('Deaths by source (log scale)',deathSourceSeries,deathSourceLogMax,v=>Math.round(10**v-1),{note:'Log scale keeps eruption, storm, shortage, and war visible beside baseline.',scaleValue:(value,max)=>max?log1p10(value)/max:0}),
  makeLineChart('Deaths per turn by source',incidentDeathSourceSeries,seriesMax(incidentDeathSourceSeries),v=>Math.round(v),{note:'Per-turn counts are the fastest way to spot specific shocks.'}),
  makeLineChart('Deaths per 1000 people by source',deathSourceRateSeries,maxDeathSourceRate,v=>v.toFixed(1),{note:'Uses previous-turn island population as the denominator.'}),
  makeLineChart('Baseline deaths per turn by life stage',baselineStageIncidentSeries,maxBaselineStageIncident,v=>Math.round(v),{note:'This splits ordinary background mortality into child, adult, and elder components.'}),
  makeLineChart('Population by ancestry marker',genotypePopulationSeries,maxGenotypePopulation,v=>Math.round(v),{note:'Use this instead of cumulative ancestry deaths to see whether BB, Bb, or bb is actually persisting on the island.'}),
  makeLineChart('Clan populations over time',clanPopulationChartSeries,maxClanPopulation,v=>Math.round(v),{note:'V marks clan formation; X marks extinction.',seriesMarkers:clanMarkers}),
  makeLineChart('Blue-eyed population by clan',clanBbPopulationSeries,maxClanBbPopulation,v=>Math.round(v),{note:'Absolute bb counts are usually more informative than shares when clan sizes diverge. V marks clan formation; X marks extinction.',seriesMarkers:clanMarkers}),
  makeLineChart('Deaths per turn by clan',clanDeathSeries.length?clanDeathSeries:[{name:'none',key:'none',color:'#ccc',values:history.map(()=>0),life:{}}],maxClanDeaths,v=>Math.round(v),{note:'Tracked clans are the biggest lineages by peak population. V marks formation; X marks extinction.',seriesMarkers:clanMarkers}),
  makeBarChart('Total deaths by clan',clanBars.length?clanBars:[{name:'none',value:0,color:'#ccc',note:''}])
);
document.getElementById('hideBaseline').addEventListener('change',baselineToggle);
baselineToggle();

let timer=null;const play=document.getElementById('play'),speed=document.getElementById('speed');
function stop(){clearInterval(timer);timer=null;play.innerHTML='&#9654;'}
play.onclick=()=>{if(timer){stop();return}play.innerHTML='&#10074;&#10074;';timer=setInterval(()=>{slider.value=(+slider.value+1)%history.length;render(+slider.value)},+speed.value)};
speed.onchange=()=>{if(timer){stop();play.click()}};
slider.oninput=()=>render(+slider.value);
render(0);
</script></body></html>"""


def export_history_html(history: List[dict], event_log: List[dict], valley_order: List[str],
                        provenance: dict, path: str = "island_replay.html") -> str:
  peaks = {}
  for snapshot in history:
    for clan, population in snapshot["totals"].get("population_by_clan", {}).items():
      peaks[clan] = max(peaks.get(clan, 0), int(population))
  tracked_clans = [
    clan for clan, _ in sorted(peaks.items(), key=lambda item: (-item[1], item[0]))[:16]
  ]

  compact_history = []
  for snapshot in history:
    compact_history.append({
      "turn": snapshot["turn"],
      "year": snapshot["year"],
      "valleys": [{
        "name": valley["name"],
        "population": valley["population"],
        "n_BB": valley["n_BB"],
        "n_Bb": valley["n_Bb"],
        "n_bb": valley["n_bb"],
        "q": valley["q"],
        "food_security": valley["food_security"],
        "land_health": valley["land_health"],
        "habitability": valley["habitability"],
        "clan_distribution": valley["clan_distribution"],
        "dominant_clan": valley["dominant_clan"],
        "dominant_traits": valley["dominant_traits"],
      } for valley in snapshot["valleys"]],
      "totals": {
        "population": snapshot["totals"]["population"],
        "deaths_by_source": snapshot["totals"]["deaths_by_source"],
        "baseline_deaths_by_stage": snapshot["totals"]["baseline_deaths_by_stage"],
        "deaths_by_genotype": snapshot["totals"]["deaths_by_genotype"],
        "deaths_by_clan": {
          clan: snapshot["totals"]["deaths_by_clan"].get(clan, 0)
          for clan in tracked_clans
        },
        "population_by_clan": {
          clan: snapshot["totals"]["population_by_clan"].get(clan, 0)
          for clan in tracked_clans
          if snapshot["totals"]["population_by_clan"].get(clan, 0) > 0
        },
        "bb_population_by_clan": {
          clan: snapshot["totals"]["bb_population_by_clan"].get(clan, 0)
          for clan in tracked_clans
          if snapshot["totals"]["bb_population_by_clan"].get(clan, 0) > 0
        },
        "clan_lifecycles": snapshot["totals"]["clan_lifecycles"],
      },
    })
  data = {
    "history": compact_history,
    "events": event_log,
    "trackedClans": tracked_clans,
    "valleyOrder": valley_order,
    "provenance": provenance,
  }
  html = _REPLAY_TEMPLATE.replace("__DATA__", json.dumps(data)).replace(
    "__COLORS__", json.dumps(VALLEY_COLORS)
  )
  output = Path(path)
  output.write_text(html, encoding="utf-8")
  return str(output.resolve())