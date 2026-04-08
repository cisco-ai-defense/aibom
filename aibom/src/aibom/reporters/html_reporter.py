# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import defaultdict
from typing import IO, Any

import jinja2

from ..models import AIComponent, ComponentRelationship, ScanResult
from ..models.enums import AIComponentType, RelationshipType
from .base import BaseReporter

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AIBOM Report</title>
  <style>
    :root {
      --bg: #0f1419;
      --surface: #1a2332;
      --border: #2d3a4d;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3d8bfd;
      --risk-low: #3ecf8e;
      --risk-mid: #f5a623;
      --risk-high: #f85149;
      --heat-green: #22c55e;
      --heat-yellow: #eab308;
      --heat-orange: #f97316;
      --heat-red: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system,
        "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      font-size: 15px;
    }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
    h1 {
      font-size: 1.75rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      margin: 0 0 0.25rem;
    }
    .sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }
    h2 {
      font-size: 1.1rem;
      font-weight: 600;
      margin: 2.5rem 0 1rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
      color: var(--accent);
    }
    .dashboard-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 1rem;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem 1.1rem;
    }
    .card .label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    .card .value {
      font-size: 1.5rem;
      font-weight: 600;
      margin-top: 0.35rem;
    }
    .severity-critical, .severity-high { color: var(--risk-high); }
    .severity-medium { color: var(--risk-mid); }
    .severity-low, .severity-info { color: var(--risk-low); }
    .layout-two {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
    }
    @media (max-width: 900px) {
      .layout-two { grid-template-columns: 1fr; }
    }
    #aibom-graph-wrap {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      position: relative;
      min-height: 480px;
    }
    #aibom-graph {
      display: block;
      width: 100%;
      height: 480px;
      cursor: grab;
    }
    #aibom-graph:active { cursor: grabbing; }
    #aibom-graph-tooltip {
      position: absolute;
      display: none;
      pointer-events: none;
      z-index: 20;
      max-width: 320px;
      padding: 0.65rem 0.85rem;
      background: rgba(15, 20, 25, 0.95);
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 0.8rem;
      box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    }
    #aibom-graph-tooltip .tt-name { font-weight: 600; color: var(--accent); }
    #aibom-graph-tooltip .tt-muted { color: var(--muted); font-size: 0.75rem; }
    #aibom-risk-heatmap {
      height: 28px;
      border-radius: 8px;
      border: 1px solid var(--border);
      position: relative;
      background: linear-gradient(90deg,
        var(--heat-green) 0%,
        var(--heat-yellow) 33%,
        var(--heat-orange) 66%,
        var(--heat-red) 100%);
      margin-top: 0.5rem;
    }
    #aibom-risk-heatmap .marker {
      position: absolute;
      top: -4px;
      width: 4px;
      height: calc(100% + 8px);
      margin-left: -2px;
      background: #fff;
      border-radius: 2px;
      box-shadow: 0 0 0 2px var(--bg);
    }
    .type-bar-row {
      display: grid;
      grid-template-columns: 120px 1fr 48px;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.45rem;
      font-size: 0.82rem;
    }
    .type-bar-track {
      height: 10px;
      background: rgba(255,255,255,0.06);
      border-radius: 5px;
      overflow: hidden;
    }
    .type-bar-fill {
      height: 100%;
      border-radius: 5px;
      background: linear-gradient(90deg, var(--accent), #7ab8ff);
      min-width: 2px;
    }
    .pie-wrap {
      display: flex;
      align-items: center;
      gap: 1.5rem;
      flex-wrap: wrap;
    }
    .pie {
      width: 160px;
      height: 160px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .pie-legend { font-size: 0.85rem; color: var(--muted); }
    .pie-legend span { display: inline-block; width: 12px; height: 12px;
      border-radius: 2px; margin-right: 0.35rem; vertical-align: middle; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
      font-size: 0.88rem;
    }
    th, td {
      text-align: left;
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid var(--border);
    }
    th {
      background: rgba(61, 139, 253, 0.12);
      color: var(--muted);
      font-weight: 600;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255,255,255,0.02); }
    .type-badge {
      display: inline-block;
      font-size: 0.72rem;
      padding: 0.2rem 0.5rem;
      border-radius: 6px;
      background: rgba(61, 139, 253, 0.15);
      color: var(--accent);
    }
    .empty { color: var(--muted); font-style: italic; padding: 1rem 0; }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85em;
      color: #a8c7fa;
    }
    .cov-yes { color: var(--risk-low); font-weight: 600; }
    .cov-no { color: var(--muted); }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AI Bill of Materials</h1>
    <p class="sub">Scan report — {{ raw_component_count }} components across
      {{ total_sources }} source(s)</p>

    <h2>Dashboard</h2>
    <div id="aibom-dashboard-summary" class="dashboard-grid">
      <div class="card">
        <div class="label">Total components</div>
        <div class="value">{{ raw_component_count }}</div>
      </div>
      <div class="card">
        <div class="label">Risk score</div>
        <div class="value severity-{{ risk_severity }}">{{ risk_score }}</div>
      </div>
      <div class="card">
        <div class="label">Severity</div>
        <div class="value severity-{{ risk_severity }}">{{ risk_severity }}</div>
      </div>
      <div class="card">
        <div class="label">Agentic candidates</div>
        <div class="value">{{ agentic_candidates }}</div>
      </div>
      <div class="card">
        <div class="label">Test-only</div>
        <div class="value">{{ test_only_components }}</div>
      </div>
      <div class="card">
        <div class="label">Relationships</div>
        <div class="value">{{ total_relationships }}</div>
      </div>
    </div>

    <div id="aibom-risk-heatmap-wrap" class="card" style="margin-top:1.25rem;">
      <div class="label">Risk score (0–100)</div>
      <div id="aibom-risk-heatmap" aria-label="Risk heatmap">
        <div class="marker" id="aibom-risk-marker" style="left: {{ risk_marker_pct }}%;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:var(--muted);margin-top:0.35rem;">
        <span>0</span><span>100</span>
      </div>
    </div>

    <div class="layout-two" style="margin-top:1.25rem;">
      <div class="card">
        <div class="label" style="margin-bottom:0.75rem;">Component types</div>
        {% if type_breakdown %}
          {% for row in type_breakdown %}
          <div class="type-bar-row">
            <span>{{ row.type }}</span>
            <div class="type-bar-track">
              <div class="type-bar-fill" style="width: {{ row.pct }}%;"></div>
            </div>
            <span style="text-align:right;color:var(--muted);">{{ row.count }}</span>
          </div>
          {% endfor %}
        {% else %}
          <p class="empty" style="margin:0;">No components.</p>
        {% endif %}
      </div>
      <div class="card">
        <div class="label" style="margin-bottom:0.75rem;">Test-only vs production</div>
        <div class="pie-wrap">
          <div class="pie" id="aibom-test-prod-pie"
            style="background: conic-gradient(
              var(--accent) 0deg {{ test_prod_deg_prod }}deg,
              var(--muted) {{ test_prod_deg_prod }}deg 360deg
            );"></div>
          <div class="pie-legend">
            <div><span style="background:var(--accent);"></span>Production ({{ test_prod_production }})</div>
            <div style="margin-top:0.35rem;"><span style="background:var(--muted);"></span>Test-only ({{ test_prod_test }})</div>
          </div>
        </div>
      </div>
    </div>

    <h2>Component graph</h2>
    <div id="aibom-graph-wrap">
      <svg id="aibom-graph" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 480" preserveAspectRatio="xMidYMid meet">
        <rect width="900" height="480" fill="transparent"/>
        <g id="aibom-graph-edges"></g>
        <g id="aibom-graph-nodes"></g>
      </svg>
      <div id="aibom-graph-tooltip"></div>
    </div>

    <h2 id="aibom-model-inventory-heading">Model inventory</h2>
    <table id="aibom-model-inventory">
      <thead>
        <tr>
          <th>Name</th>
          <th>Model</th>
          <th>Framework</th>
          <th>File</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
        {% if model_rows %}
          {% for m in model_rows %}
          <tr>
            <td>{{ m.name }}</td>
            <td>{{ m.model_name }}</td>
            <td>{{ m.framework }}</td>
            <td><code>{{ m.file_path }}</code></td>
            <td>{{ m.confidence }}</td>
          </tr>
          {% endfor %}
        {% else %}
          <tr><td colspan="5" class="empty">No model components.</td></tr>
        {% endif %}
      </tbody>
    </table>

    <h2 id="aibom-coverage-matrix-heading">Guardrail / observability coverage</h2>
    <table id="aibom-coverage-matrix">
      <thead>
        <tr>
          <th>Type</th>
          <th>Count</th>
          <th>Guardrail link</th>
          <th>Observability link</th>
        </tr>
      </thead>
      <tbody>
        {% for row in coverage_rows %}
        <tr>
          <td><span class="type-badge">{{ row.type }}</span></td>
          <td>{{ row.count }}</td>
          <td class="{{ row.guard_class }}">{{ row.guard_text }}</td>
          <td class="{{ row.obs_class }}">{{ row.obs_text }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    <h2>Components</h2>
    {% if components_by_type %}
      {% for ctype, rows in components_by_type.items() %}
      <h3 style="font-size:1rem;margin:1.5rem 0 0.75rem;color:var(--muted);">
        <span class="type-badge">{{ ctype }}</span></h3>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>File</th>
            <th>Line</th>
            <th>Framework</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody>
          {% for c in rows %}
          <tr>
            <td>{{ c.name }}</td>
            <td><code>{{ c.file_path }}</code></td>
            <td>{{ c.line_number }}</td>
            <td>{{ c.framework }}</td>
            <td>{{ c.detection_source }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% endfor %}
    {% else %}
      <p class="empty">No components detected.</p>
    {% endif %}

    <h2>Relationships</h2>
    {% if relationships %}
    <table>
      <thead>
        <tr>
          <th>From</th>
          <th>To</th>
          <th>Type</th>
          <th>Label</th>
        </tr>
      </thead>
      <tbody>
        {% for r in relationships %}
        <tr>
          <td>{{ r.source_name or r.source_instance_id }}</td>
          <td>{{ r.target_name or r.target_instance_id }}</td>
          <td>{{ r.relationship_type }}</td>
          <td>{{ r.label }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
      <p class="empty">No relationships.</p>
    {% endif %}

    <h2>Risk flags</h2>
    {% if risk_flags %}
    <table>
      <thead>
        <tr>
          <th>Flag</th>
          <th>Severity</th>
          <th>Weight</th>
          <th>Description</th>
          <th>Location</th>
        </tr>
      </thead>
      <tbody>
        {% for f in risk_flags %}
        <tr>
          <td>{{ f.flag }}</td>
          <td class="severity-{{ f.severity }}">{{ f.severity }}</td>
          <td>{{ f.weight }}</td>
          <td>{{ f.description }}</td>
          <td><code>{{ f.file_path }}</code> :{{ f.line_number }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
      <p class="empty">No risk flags.</p>
    {% endif %}
  </div>
  <script type="application/json" id="aibom-graph-data">{{ graph_json|safe }}</script>
  <script>
(function() {
  var TYPE_COLORS = {{ type_colors_json|safe }};
  var el = document.getElementById('aibom-graph-data');
  if (!el) return;
  var raw = el.textContent || '';
  var data;
  try { data = JSON.parse(raw); } catch (e) { return; }
  var nodes = data.nodes || [];
  var edges = data.edges || [];
  var svg = document.getElementById('aibom-graph');
  var gEdges = document.getElementById('aibom-graph-edges');
  var gNodes = document.getElementById('aibom-graph-nodes');
  var tooltip = document.getElementById('aibom-graph-tooltip');
  var wrap = document.getElementById('aibom-graph-wrap');
  if (!svg || !gEdges || !gNodes || !nodes.length) {
    if (gEdges) { while (gEdges.firstChild) gEdges.removeChild(gEdges.firstChild); }
    if (gNodes) {
      while (gNodes.firstChild) gNodes.removeChild(gNodes.firstChild);
      var t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('x', '450');
      t.setAttribute('y', '240');
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('fill', '#8b9cb3');
      t.textContent = nodes.length ? '' : 'No components to graph';
      gNodes.appendChild(t);
    }
    return;
  }
  var W = 900, H = 480;
  var cx = W / 2, cy = H / 2;
  nodes.forEach(function(n, i) {
    var ang = (i / nodes.length) * Math.PI * 2;
    var r = Math.min(W, H) * 0.25;
    n.x = cx + Math.cos(ang) * r * 0.5;
    n.y = cy + Math.sin(ang) * r * 0.5;
    n.vx = 0;
    n.vy = 0;
  });
  var idToNode = {};
  nodes.forEach(function(n) { idToNode[n.id] = n; });
  function radiusFor(n) {
    var c = typeof n.confidence === 'number' ? n.confidence : 1;
    c = Math.max(0.15, Math.min(1, c));
    return 6 + c * 14;
  }
  function colorFor(n) {
    return TYPE_COLORS[n.type] || '#6b7280';
  }
  var edgeEls = [];
  edges.forEach(function() {
    var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('stroke', 'rgba(139,156,179,0.35)');
    line.setAttribute('stroke-width', '1');
    gEdges.appendChild(line);
    edgeEls.push(line);
  });
  var nodeEls = [];
  nodes.forEach(function(n) {
    var c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('fill', colorFor(n));
    c.setAttribute('stroke', 'rgba(15,20,25,0.9)');
    c.setAttribute('stroke-width', '1.5');
    c.style.cursor = 'pointer';
    c.addEventListener('mouseenter', function(ev) { showTip(n, ev); });
    c.addEventListener('mousemove', function(ev) { moveTip(ev); });
    c.addEventListener('mouseleave', function() { hideTip(); });
    gNodes.appendChild(c);
    nodeEls.push(c);
  });
  function showTip(n, ev) {
    if (!tooltip || !wrap) return;
    while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
    function addLine(text, cls) {
      var d = document.createElement('div');
      if (cls) d.className = cls;
      d.textContent = text;
      tooltip.appendChild(d);
    }
    addLine(n.name, 'tt-name');
    addLine(n.type, 'tt-muted');
    addLine('confidence: ' + String(n.confidence));
    if (n.file_path) addLine(n.file_path + ':' + String(n.line_number), 'tt-muted');
    if (n.model_name) addLine('model: ' + n.model_name);
    if (n.framework) addLine('framework: ' + n.framework);
    tooltip.style.display = 'block';
    moveTip(ev);
  }
  function moveTip(ev) {
    if (!tooltip || !wrap) return;
    var rect = wrap.getBoundingClientRect();
    var x = ev.clientX - rect.left + 12;
    var y = ev.clientY - rect.top + 12;
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
  }
  function hideTip() {
    if (tooltip) tooltip.style.display = 'none';
  }
  var kRep = 3200;
  var kSpring = 0.018;
  var idealLen = 90;
  var damping = 0.88;
  var centerK = 0.0008;
  function step() {
    var i, j, dx, dy, d2, d, f, nx, ny;
    for (i = 0; i < nodes.length; i++) {
      nodes[i].fx = 0;
      nodes[i].fy = 0;
    }
    for (i = 0; i < nodes.length; i++) {
      for (j = i + 1; j < nodes.length; j++) {
        dx = nodes[j].x - nodes[i].x;
        dy = nodes[j].y - nodes[i].y;
        d2 = dx * dx + dy * dy + 0.01;
        d = Math.sqrt(d2);
        f = kRep / d2;
        nx = (dx / d) * f;
        ny = (dy / d) * f;
        nodes[i].fx -= nx;
        nodes[i].fy -= ny;
        nodes[j].fx += nx;
        nodes[j].fy += ny;
      }
    }
    edges.forEach(function(e) {
      var a = idToNode[e.source];
      var b = idToNode[e.target];
      if (!a || !b) return;
      dx = b.x - a.x;
      dy = b.y - a.y;
      d = Math.sqrt(dx * dx + dy * dy) + 0.01;
      f = kSpring * (d - idealLen);
      nx = (dx / d) * f;
      ny = (dy / d) * f;
      a.fx += nx;
      a.fy += ny;
      b.fx -= nx;
      b.fy -= ny;
    });
    for (i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      n.fx -= (n.x - cx) * centerK * nodes.length;
      n.fy -= (n.y - cy) * centerK * nodes.length;
      n.vx = (n.vx + n.fx) * damping;
      n.vy = (n.vy + n.fy) * damping;
      n.x += n.vx;
      n.y += n.vy;
      n.x = Math.max(24, Math.min(W - 24, n.x));
      n.y = Math.max(24, Math.min(H - 24, n.y));
    }
  }
  function frame() {
    for (var s = 0; s < 2; s++) step();
    edges.forEach(function(e, idx) {
      var a = idToNode[e.source];
      var b = idToNode[e.target];
      var line = edgeEls[idx];
      if (!a || !b || !line) return;
      line.setAttribute('x1', a.x);
      line.setAttribute('y1', a.y);
      line.setAttribute('x2', b.x);
      line.setAttribute('y2', b.y);
    });
    nodes.forEach(function(n, idx) {
      var c = nodeEls[idx];
      var r = radiusFor(n);
      c.setAttribute('cx', n.x);
      c.setAttribute('cy', n.y);
      c.setAttribute('r', r);
    });
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
  </script>
</body>
</html>
"""


def _group_components(
    components: list[AIComponent],
) -> dict[str, list[dict[str, str | int]]]:
    by: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for c in components:
        by[c.component_type.value].append(
            {
                "name": c.name,
                "file_path": c.file_path,
                "line_number": c.line_number,
                "framework": c.framework,
                "detection_source": c.detection_source.value,
            }
        )
    return dict(sorted(by.items()))


_TYPE_PALETTE: dict[str, str] = {
    "model": "#60a5fa",
    "agent": "#a78bfa",
    "tool": "#34d399",
    "mcp_server": "#f472b6",
    "mcp_client": "#fb7185",
    "embedding": "#38bdf8",
    "vector_store": "#2dd4bf",
    "dataset": "#4ade80",
    "prompt": "#fcd34d",
    "guardrail": "#f87171",
    "memory": "#818cf8",
    "retriever": "#5eead4",
    "observability": "#94a3b8",
    "secret": "#ef4444",
    "dependency": "#78716c",
    "other": "#6b7280",
}


def _graph_payload(
    components: list[AIComponent], relationships: list[ComponentRelationship]
) -> dict[str, Any]:
    nodes = [
        {
            "id": c.instance_id,
            "name": c.name,
            "type": c.component_type.value,
            "confidence": c.confidence,
            "file_path": c.file_path,
            "line_number": c.line_number,
            "model_name": c.model_name or "",
            "framework": c.framework or "",
        }
        for c in components
    ]
    ids = {c.instance_id for c in components}
    edges: list[dict[str, str]] = []
    for r in relationships:
        if r.source_instance_id in ids and r.target_instance_id in ids:
            edges.append({"source": r.source_instance_id, "target": r.target_instance_id})
    return {"nodes": nodes, "edges": edges}


def _type_breakdown(components: list[AIComponent]) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    for c in components:
        counts[c.component_type.value] += 1
    mx = max(counts.values(), default=0)
    rows = []
    for t in sorted(counts.keys()):
        n = counts[t]
        rows.append({"type": t, "count": n, "pct": round(100.0 * n / mx, 2) if mx else 0})
    return rows


def _model_inventory_rows(components: list[AIComponent]) -> list[dict[str, Any]]:
    rows = []
    for c in components:
        if not c.component_type.is_model_related:
            continue
        rows.append(
            {
                "name": c.name,
                "model_name": c.model_name or "—",
                "framework": c.framework or "—",
                "file_path": c.file_path or "—",
                "confidence": c.confidence,
            }
        )
    return sorted(rows, key=lambda x: (x["file_path"], x["name"]))


def _other_instance_id(rel: ComponentRelationship, cid: str) -> str:
    if rel.source_instance_id == cid:
        return rel.target_instance_id
    if rel.target_instance_id == cid:
        return rel.source_instance_id
    return ""


def _has_guardrail_link(
    c: AIComponent,
    relationships: list[ComponentRelationship],
    by_id: dict[str, AIComponent],
) -> bool:
    for r in relationships:
        if r.source_instance_id != c.instance_id and r.target_instance_id != c.instance_id:
            continue
        oid = _other_instance_id(r, c.instance_id)
        other = by_id.get(oid)
        if r.relationship_type == RelationshipType.USES_GUARDRAIL:
            return True
        if other and other.component_type == AIComponentType.GUARDRAIL:
            return True
    return False


def _has_observability_link(
    c: AIComponent,
    relationships: list[ComponentRelationship],
    by_id: dict[str, AIComponent],
) -> bool:
    for r in relationships:
        if r.source_instance_id != c.instance_id and r.target_instance_id != c.instance_id:
            continue
        oid = _other_instance_id(r, c.instance_id)
        other = by_id.get(oid)
        if r.relationship_type in (
            RelationshipType.LOGS_TO,
            RelationshipType.OBSERVES,
        ):
            return True
        if other and other.component_type == AIComponentType.OBSERVABILITY:
            return True
    return False


def _coverage_rows(
    components: list[AIComponent], relationships: list[ComponentRelationship]
) -> list[dict[str, Any]]:
    by_id = {c.instance_id: c for c in components}
    rows: list[dict[str, Any]] = []
    for ctype in (
        AIComponentType.AGENT,
        AIComponentType.MCP_SERVER,
        AIComponentType.TOOL,
    ):
        comps = [c for c in components if c.component_type == ctype]
        total = len(comps)
        g_n = sum(1 for c in comps if _has_guardrail_link(c, relationships, by_id))
        o_n = sum(1 for c in comps if _has_observability_link(c, relationships, by_id))
        rows.append(
            {
                "type": ctype.value,
                "count": total,
                "guard_text": f"{g_n} / {total}" if total else "—",
                "guard_class": "cov-yes" if total and g_n == total else ("cov-no" if total else "cov-no"),
                "obs_text": f"{o_n} / {total}" if total else "—",
                "obs_class": "cov-yes" if total and o_n == total else ("cov-no" if total else "cov-no"),
            }
        )
    return rows


def _safe_json_script(obj: Any) -> str:
    s = json.dumps(obj, separators=(",", ":"))
    return s.replace("</script>", "<\\/script>")


class HtmlReporter(BaseReporter):
    name = "html"
    file_extension = ".html"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        summary = result.summary
        components = result.all_components
        rels = result.all_relationships
        raw_count = len(components)
        risk_score = int(summary["risk_score"])
        graph = _graph_payload(components, rels)
        type_rows = _type_breakdown(components)
        test_only = summary["test_only_components"]
        production = max(0, raw_count - test_only)
        total_tp = raw_count if raw_count else 1
        deg_prod = round(360.0 * production / total_tp, 2) if raw_count else 0

        ctx = {
            "raw_component_count": raw_count,
            "total_sources": summary["total_sources"],
            "total_relationships": summary["total_relationships"],
            "risk_score": risk_score,
            "risk_severity": summary["risk_severity"],
            "risk_marker_pct": min(100, max(0, risk_score)),
            "agentic_candidates": summary["agentic_candidates"],
            "test_only_components": test_only,
            "test_prod_test": test_only,
            "test_prod_production": production,
            "test_prod_deg_prod": deg_prod,
            "type_breakdown": type_rows,
            "components_by_type": _group_components(components),
            "model_rows": _model_inventory_rows(components),
            "coverage_rows": _coverage_rows(components, rels),
            "relationships": [
                {
                    "source_instance_id": r.source_instance_id,
                    "target_instance_id": r.target_instance_id,
                    "source_name": r.source_name,
                    "target_name": r.target_name,
                    "relationship_type": r.relationship_type.value,
                    "label": r.label,
                }
                for r in rels
            ],
            "risk_flags": [
                {
                    "flag": f.flag,
                    "severity": f.severity.value,
                    "weight": f.weight,
                    "description": f.description,
                    "file_path": f.file_path,
                    "line_number": f.line_number,
                }
                for f in result.risk.flags
            ],
            "graph_json": _safe_json_script(graph),
            "type_colors_json": _safe_json_script(_TYPE_PALETTE),
        }
        env = jinja2.Environment(
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )
        tpl = env.from_string(_HTML_TEMPLATE)
        output.write(tpl.render(**ctx))
