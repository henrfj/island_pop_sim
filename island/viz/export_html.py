"""Exports a completed simulation run as a single self-contained HTML file: an animated
map of the island (valley size = population, valley fill = blue-allele frequency on a
sequential blue ramp, ring color = which valley for cross-reference with the chart),
connecting lines for mountain paths (solid) and canoe routes (dashed), event badges for
volcanoes/wars, a play/scrub timeline, and a line chart of blue-allele frequency per
valley over time. No server or network access needed -- just double-click the file, or
point the browser pane at it.
"""
import json
from pathlib import Path
from typing import List

# Fixed categorical colors, one per valley, in founding order (never reassigned/reused --
# see the dataviz convention this project follows: identity is encoded by color, not rank).
VALLEY_COLORS = ["#2E7D5B", "#C9762C", "#6A4C93", "#B23A3A", "#3E7CB1", "#8A8A00"]

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Island replay</title>
<style>
  :root {
    --ink: #222; --muted: #6b6b6b; --grid: #e2e2e2; --surface: #ffffff;
    --panel: #f7f6f4; --border: #d8d5cf;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--ink); background: var(--surface); margin: 0; padding: 20px;
  }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
  .layout { display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 14px;
  }
  .controls { display: flex; align-items: center; gap: 10px; margin: 10px 0 4px; flex-wrap: wrap; }
  button {
    font: inherit; padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
    background: white; cursor: pointer;
  }
  button:hover { background: #f0f0f0; }
  input[type=range] { flex: 1; min-width: 220px; }
  .gen-label { font-variant-numeric: tabular-nums; min-width: 90px; text-align: right; }
  .legend-row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
  .valley-list { list-style: none; padding: 0; margin: 8px 0 0; font-size: 13px; }
  .valley-list li { display: flex; justify-content: space-between; align-items: baseline;
    gap: 8px; padding: 4px 0; flex-wrap: nowrap; white-space: nowrap; }
  .valley-list .name { display: flex; align-items: center; }
  .valley-list .stat { color: var(--muted); font-variant-numeric: tabular-nums; }
  .event-log { max-height: 220px; overflow-y: auto; font-size: 12px; margin-top: 6px; }
  .event-log div { padding: 3px 4px; border-radius: 4px; cursor: pointer; }
  .event-log div:hover { background: #eee; }
  .event-log .current { background: #fff2c9; }
  .grad-bar { height: 10px; width: 140px; border-radius: 5px;
    background: linear-gradient(to right, #EAF2FA, #1B3A5C); }
  svg text { font-size: 11px; fill: var(--muted); }
  .map-label { font-size: 11px; fill: var(--ink); text-anchor: middle; }
</style>
</head>
<body>
<h1>Island replay</h1>
<div class="sub">Circle size = population &middot; fill = blue-allele frequency (q) &middot; ring color identifies the valley (matches the chart below)</div>

<div class="controls">
  <button id="playBtn">&#9654; Play</button>
  <input id="slider" type="range" min="0" max="0" value="0">
  <span class="gen-label" id="genLabel">gen 0</span>
  <select id="speedSel">
    <option value="500">0.5x</option>
    <option value="250" selected>1x</option>
    <option value="120">2x</option>
    <option value="40">5x</option>
  </select>
</div>

<div class="layout">
  <div class="panel">
    <svg id="map" width="640" height="440"></svg>
    <div class="legend-row">
      <span>Blue-allele frequency:</span>
      <div class="grad-bar"></div>
      <span>0% &rarr; 100%</span>
      <span>&#127755; volcano</span>
      <span>&#9760;&#65039; devastated (uninhabitable)</span>
      <span>&#9876;&#65039; war</span>
      <span>&#129523; migration</span>
      <span>&#128758; emigration</span>
      <span>&#127754; storm (event log)</span>
    </div>
  </div>

  <div class="panel" style="min-width:300px;">
    <strong>Valleys</strong>
    <ul class="valley-list" id="valleyList"></ul>
    <strong style="display:block; margin-top:12px;">Event log</strong>
    <div class="event-log" id="eventLog"></div>
  </div>
</div>

<div class="panel" style="margin-top:16px;">
  <strong>Blue-allele frequency over time</strong>
  <svg id="chart" width="900" height="260"></svg>
</div>

<script>
const DATA = __DATA_JSON__;
const COLORS = __COLORS_JSON__;
const history = DATA.history;
const events = DATA.events;
const names = DATA.valleyOrder;
const n = names.length;
const maxGen = history.length - 1;

const svgNS = "http://www.w3.org/2000/svg";
function el(tag, attrs) {
  const e = document.createElementNS(svgNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

// ---------- Map layout: valleys evenly spaced on a ring around a central volcano ----------
const cx = 300, cy = 210, R = 150;
const positions = [];
for (let i = 0; i < n; i++) {
  const angle = -Math.PI / 2 + i * (2 * Math.PI / n);
  positions.push({ x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle) });
}

const mapSvg = document.getElementById("map");
// connecting lines: solid ring path between neighbors, dashed water route otherwise
for (let i = 0; i < n; i++) {
  for (let j = i + 1; j < n; j++) {
    const isNeighbor = (j === i + 1) || (i === 0 && j === n - 1);
    const line = el("line", {
      x1: positions[i].x, y1: positions[i].y, x2: positions[j].x, y2: positions[j].y,
      stroke: isNeighbor ? "#a68b6c" : "#8fb3d1",
      "stroke-width": isNeighbor ? 2 : 1.2,
      "stroke-dasharray": isNeighbor ? "" : "5,4",
    });
    mapSvg.appendChild(line);
  }
}
// volcano marker
const volcano = el("path", {
  d: `M ${cx-14} ${cy+10} L ${cx} ${cy-16} L ${cx+14} ${cy+10} Z`,
  fill: "#8a4a2a", stroke: "#5c3018", "stroke-width": 1.5,
});
mapSvg.appendChild(volcano);
mapSvg.appendChild(el("text", { x: cx, y: cy + 28, class: "map-label" })).textContent = "Volcano";

const valleyCircles = [], valleyLabels = [], valleyBadges = [];
for (let i = 0; i < n; i++) {
  const c = el("circle", { cx: positions[i].x, cy: positions[i].y, r: 20, fill: "#EAF2FA",
                            stroke: COLORS[i % COLORS.length], "stroke-width": 3 });
  mapSvg.appendChild(c);
  valleyCircles.push(c);
  const label = el("text", { x: positions[i].x, y: positions[i].y + 40, class: "map-label" });
  mapSvg.appendChild(label);
  valleyLabels.push(label);
  const badge = el("text", { x: positions[i].x + 22, y: positions[i].y - 22, "font-size": 18 });
  mapSvg.appendChild(badge);
  valleyBadges.push(badge);
}

function lerp(a, b, t) { return a + (b - a) * t; }
function hexToRgb(h) {
  h = h.replace("#", "");
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}
function qColor(q) {
  const c0 = hexToRgb("EAF2FA"), c1 = hexToRgb("1B3A5C");
  const r = Math.round(lerp(c0[0], c1[0], q)), g = Math.round(lerp(c0[1], c1[1], q)), b = Math.round(lerp(c0[2], c1[2], q));
  return `rgb(${r},${g},${b})`;
}
function radiusFor(pop) {
  return Math.max(10, Math.min(50, 6 + Math.sqrt(pop) * 1.8));
}

// ---------- Chart: blue-allele frequency per valley over time ----------
const chartSvg = document.getElementById("chart");
const cW = 900, cH = 260, padL = 40, padR = 20, padT = 14, padB = 26;
const plotW = cW - padL - padR, plotH = cH - padT - padB;
function xPix(gen) { return padL + (gen / maxGen) * plotW; }
function yPix(pct) { return padT + plotH - (pct / 100) * plotH; }

for (let pct = 0; pct <= 100; pct += 25) {
  chartSvg.appendChild(el("line", { x1: padL, y1: yPix(pct), x2: cW - padR, y2: yPix(pct), stroke: "#e2e2e2" }));
  const t = el("text", { x: padL - 6, y: yPix(pct) + 3, "text-anchor": "end" });
  t.textContent = pct + "%";
  chartSvg.appendChild(t);
}
chartSvg.appendChild(el("line", { x1: padL, y1: padT, x2: padL, y2: cH - padB, stroke: "#999" }));
chartSvg.appendChild(el("line", { x1: padL, y1: cH - padB, x2: cW - padR, y2: cH - padB, stroke: "#999" }));
for (let g = 0; g <= maxGen; g += Math.max(1, Math.round(maxGen / 10))) {
  const t = el("text", { x: xPix(g), y: cH - padB + 14, "text-anchor": "middle" });
  t.textContent = g;
  chartSvg.appendChild(t);
}

for (let i = 0; i < n; i++) {
  const pts = history.map(snap => `${xPix(snap.gen)},${yPix(snap.valleys[i].q * 100)}`).join(" ");
  chartSvg.appendChild(el("polyline", { points: pts, fill: "none", stroke: COLORS[i % COLORS.length], "stroke-width": 2 }));
}
const marker = el("line", { x1: padL, y1: padT, x2: padL, y2: cH - padB, stroke: "#222", "stroke-width": 1, "stroke-dasharray": "3,3" });
chartSvg.appendChild(marker);

// legend for the chart (categorical colors = which valley)
const legendY = padT + 4;
for (let i = 0; i < n; i++) {
  const lx = padL + 10 + i * 130;
  chartSvg.appendChild(el("circle", { cx: lx, cy: legendY, r: 4, fill: COLORS[i % COLORS.length] }));
  const t = el("text", { x: lx + 8, y: legendY + 4 });
  t.textContent = names[i];
  chartSvg.appendChild(t);
}

// ---------- Side panel ----------
const valleyListEl = document.getElementById("valleyList");
const rows = [];
for (let i = 0; i < n; i++) {
  const li = document.createElement("li");
  li.innerHTML = `<span class="name"><span class="swatch" style="background:${COLORS[i % COLORS.length]}"></span>${names[i]}</span><span class="stat"></span>`;
  valleyListEl.appendChild(li);
  rows.push(li.querySelector(".stat"));
}

const eventLogEl = document.getElementById("eventLog");
const eventEntries = [];
const EVENT_ICON = { volcano: "\u{1F30B}", war: "⚔️", migration: "\u{1F9F3}", emigration: "\u{1F6F6}", storm: "\u{1F30A}" };
function eventText(ev) {
  switch (ev.type) {
    case "volcano": return `${ev.valley} — eruption`;
    case "war": return `${ev.valley} — war (${ev.refugees} refugees expelled)`;
    case "migration": return `${ev.count} ${ev.kind === "refugee" ? "refugees fled" : "moved"} ${ev.from} → ${ev.to} (by ${ev.route === "canoe" ? "canoe" : "foot"})`;
    case "emigration": return `${ev.valley} — ${ev.left} left for fortune elsewhere`;
    case "storm": return `storm sank ${ev.lost} ${ev.kind === "refugee" ? "refugees" : "migrants"} (${ev.from} → ${ev.to})`;
    default: return JSON.stringify(ev);
  }
}
events.forEach((ev) => {
  const div = document.createElement("div");
  div.textContent = `${EVENT_ICON[ev.type] || "?"} gen ${ev.gen} — ${eventText(ev)}`;
  div.onclick = () => { slider.value = ev.gen; render(ev.gen); };
  eventLogEl.appendChild(div);
  eventEntries.push({ gen: ev.gen, el: div });
});

// ---------- Render / playback ----------
const slider = document.getElementById("slider");
const genLabel = document.getElementById("genLabel");
const playBtn = document.getElementById("playBtn");
const speedSel = document.getElementById("speedSel");
slider.max = maxGen;

function render(gen) {
  const snap = history[gen];
  genLabel.textContent = "gen " + gen;
  marker.setAttribute("x1", xPix(gen));
  marker.setAttribute("x2", xPix(gen));
  for (let i = 0; i < n; i++) {
    const v = snap.valleys[i];
    valleyCircles[i].setAttribute("r", radiusFor(v.population));
    valleyCircles[i].setAttribute("fill", qColor(v.q));
    valleyLabels[i].textContent = `${v.name}: ${v.population} (q=${(v.q*100).toFixed(1)}%)`;
    let badge = "";
    if (v.eruption) badge += "\u{1F30B}";
    else if (v.devastated) badge += "\u{2620}\u{FE0F}";
    if (v.war) badge += "⚔️";
    if (v.emigration) badge += "\u{1F6F6}";
    valleyBadges[i].textContent = badge;
    rows[i].textContent = `${v.population}/${Math.round(v.capacity)}  ·  q=${(v.q*100).toFixed(1)}%`;
  }
  eventEntries.forEach(e => e.el.classList.toggle("current", e.gen === gen));
}

let playing = false, timer = null;
function tick() {
  let g = parseInt(slider.value, 10) + 1;
  if (g > maxGen) g = 0;
  slider.value = g;
  render(g);
}
playBtn.onclick = () => {
  playing = !playing;
  playBtn.innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  if (playing) {
    timer = setInterval(tick, parseInt(speedSel.value, 10));
  } else {
    clearInterval(timer);
  }
};
speedSel.onchange = () => { if (playing) { clearInterval(timer); timer = setInterval(tick, parseInt(speedSel.value, 10)); } };
slider.oninput = () => render(parseInt(slider.value, 10));

render(0);
</script>
</body>
</html>
"""


def export_history_html(history: List[dict], event_log: List[dict], valley_order: List[str],
                         path: str = "island_replay.html") -> str:
    data = {"history": history, "valleyOrder": valley_order}
    html = _TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    html = html.replace("__COLORS_JSON__", json.dumps(VALLEY_COLORS))
    # events are embedded separately via the same DATA object's sibling key for clarity in the JS
    html = html.replace("const events = DATA.events;", f"const events = {json.dumps(event_log)};")
    Path(path).write_text(html, encoding="utf-8")
    return str(Path(path).resolve())
