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

from collections import defaultdict
from typing import IO

import jinja2

from ..models import AIComponent, ScanResult
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
    .wrap { max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
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
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
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
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AI Bill of Materials</h1>
    <p class="sub">Scan report — {{ total_components }} components across
      {{ total_sources }} source(s)</p>

    <h2>Summary</h2>
    <div class="summary-grid">
      <div class="card">
        <div class="label">Total components</div>
        <div class="value">{{ total_components }}</div>
      </div>
      <div class="card">
        <div class="label">Risk score</div>
        <div class="value">{{ risk_score }}</div>
      </div>
      <div class="card">
        <div class="label">Severity</div>
        <div class="value severity-{{ risk_severity }}">{{
          risk_severity }}</div>
      </div>
      <div class="card">
        <div class="label">Relationships</div>
        <div class="value">{{ total_relationships }}</div>
      </div>
    </div>
    {% if type_counts %}
    <div class="summary-grid" style="margin-top:1rem;">
      {% for t, n in type_counts.items() %}
      <div class="card">
        <div class="label">{{ t }}</div>
        <div class="value">{{ n }}</div>
      </div>
      {% endfor %}
    </div>
    {% endif %}

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


class HtmlReporter(BaseReporter):
    name = "html"
    file_extension = ".html"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        summary = result.summary
        ctx = {
            "total_components": summary["total_components"],
            "total_sources": summary["total_sources"],
            "total_relationships": summary["total_relationships"],
            "risk_score": summary["risk_score"],
            "risk_severity": summary["risk_severity"],
            "type_counts": summary["component_types"],
            "components_by_type": _group_components(result.all_components),
            "relationships": [
                {
                    "source_instance_id": r.source_instance_id,
                    "target_instance_id": r.target_instance_id,
                    "source_name": r.source_name,
                    "target_name": r.target_name,
                    "relationship_type": r.relationship_type.value,
                    "label": r.label,
                }
                for r in result.all_relationships
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
        }
        env = jinja2.Environment(
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )
        tpl = env.from_string(_HTML_TEMPLATE)
        output.write(tpl.render(**ctx))
