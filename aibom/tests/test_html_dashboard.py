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

from io import StringIO

from aibom.models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    RelationshipType,
    RiskFlag,
    RiskScore,
    ScanResult,
    Severity,
    SourceResult,
)
from aibom.reporters.html_reporter import HtmlReporter


def _render(result: ScanResult) -> str:
    buf = StringIO()
    HtmlReporter().render(result, buf)
    return buf.getvalue()


def test_html_dashboard_empty_scan_valid_html():
    html = _render(ScanResult(sources=[], risk=RiskScore()))
    assert "<!DOCTYPE html>" in html
    assert "<html" in html.lower()
    assert "</html>" in html.lower()
    assert 'id="aibom-dashboard-summary"' in html
    assert 'id="aibom-graph-wrap"' in html


def test_html_dashboard_with_graph_and_tables():
    agent = AIComponent(
        name="my-agent",
        component_type=AIComponentType.AGENT,
        file_path="/app/agent.py",
        line_number=1,
        confidence=0.9,
    )
    tool = AIComponent(
        name="my-tool",
        component_type=AIComponentType.TOOL,
        file_path="/app/tools.py",
        line_number=2,
    )
    guard = AIComponent(
        name="g1",
        component_type=AIComponentType.GUARDRAIL,
        file_path="/app/g.py",
        line_number=3,
    )
    obs = AIComponent(
        name="o1",
        component_type=AIComponentType.OBSERVABILITY,
        file_path="/app/o.py",
        line_number=4,
    )
    model = AIComponent(
        name="m1",
        component_type=AIComponentType.MODEL,
        file_path="/app/m.py",
        line_number=5,
        model_name="gpt-test",
        framework="openai",
        confidence=0.95,
    )
    rels = [
        ComponentRelationship(
            source_instance_id=agent.instance_id,
            target_instance_id=tool.instance_id,
            relationship_type=RelationshipType.USES_TOOL,
            source_name=agent.name,
            target_name=tool.name,
            source_type=AIComponentType.AGENT,
            target_type=AIComponentType.TOOL,
        ),
        ComponentRelationship(
            source_instance_id=agent.instance_id,
            target_instance_id=guard.instance_id,
            relationship_type=RelationshipType.USES_GUARDRAIL,
            source_name=agent.name,
            target_name=guard.name,
            source_type=AIComponentType.AGENT,
            target_type=AIComponentType.GUARDRAIL,
        ),
        ComponentRelationship(
            source_instance_id=agent.instance_id,
            target_instance_id=obs.instance_id,
            relationship_type=RelationshipType.LOGS_TO,
            source_name=agent.name,
            target_name=obs.name,
            source_type=AIComponentType.AGENT,
            target_type=AIComponentType.OBSERVABILITY,
        ),
    ]
    result = ScanResult(
        sources=[
            SourceResult(
                path="/app",
                components=[agent, tool, guard, obs, model],
                relationships=rels,
            )
        ],
        risk=RiskScore(),
    )
    html = _render(result)
    assert 'id="aibom-dashboard-summary"' in html
    assert 'id="aibom-graph-wrap"' in html
    assert 'id="aibom-graph"' in html
    assert 'id="aibom-model-inventory"' in html
    assert "gpt-test" in html
    assert 'id="aibom-coverage-matrix"' in html
    assert "<table" in html


def test_html_dashboard_risk_flags_show_heatmap():
    risk = RiskScore()
    risk.add_flag(
        RiskFlag(
            flag="x",
            severity=Severity.MEDIUM,
            weight=40,
            description="d",
        )
    )
    html = _render(ScanResult(sources=[], risk=risk))
    assert 'id="aibom-risk-heatmap"' in html
    assert 'id="aibom-risk-marker"' in html


def test_html_dashboard_major_sections_present():
    c = AIComponent(name="c", component_type=AIComponentType.MODEL, file_path="f.py")
    html = _render(
        ScanResult(
            sources=[SourceResult(path=".", components=[c], relationships=[])],
            risk=RiskScore(),
        )
    )
    assert "Dashboard" in html
    assert "Component graph" in html
    assert "Model inventory" in html
    assert "Guardrail" in html and "observability" in html.lower()
