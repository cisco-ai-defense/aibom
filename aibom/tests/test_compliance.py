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

import pytest

from aibom.compliance import (
    ComplianceFramework,
    evaluate_compliance,
    parse_compliance_cli_value,
)
from aibom.models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    RelationshipType,
    ScanResult,
    SourceResult,
)


def _src(components: list[AIComponent], rels: list[ComponentRelationship] | None = None) -> SourceResult:
    return SourceResult(path=".", components=components, relationships=rels or [])


def test_eu1_model_without_model_name_fails():
    m = AIComponent(name="m", component_type=AIComponentType.MODEL, model_name=None)
    r = evaluate_compliance(ScanResult(sources=[_src([m])]), ComplianceFramework.EU_AI_ACT)
    row = next(x for x in r.results if x.requirement_id == "eu-1")
    assert row.status == "fail"
    assert "m" in row.affected_components


def test_eu2_training_run_without_dataset_fails():
    tr = AIComponent(name="run1", component_type=AIComponentType.TRAINING_RUN)
    r = evaluate_compliance(ScanResult(sources=[_src([tr])]), ComplianceFramework.EU_AI_ACT)
    row = next(x for x in r.results if x.requirement_id == "eu-2")
    assert row.status == "fail"


def test_eu4_agent_without_guardrail_fails():
    ag = AIComponent(name="a1", component_type=AIComponentType.AGENT, description="x", framework="y")
    r = evaluate_compliance(ScanResult(sources=[_src([ag])]), ComplianceFramework.EU_AI_ACT)
    row = next(x for x in r.results if x.requirement_id == "eu-4")
    assert row.status == "fail"
    assert "a1" in row.affected_components


def test_owasp4_high_confidence_secret_fails():
    sec = AIComponent(
        name="leak",
        component_type=AIComponentType.SECRET,
        heuristic_confidence=0.95,
        file_path="x.py",
    )
    r = evaluate_compliance(ScanResult(sources=[_src([sec])]), ComplianceFramework.OWASP_AGENTIC)
    row = next(x for x in r.results if x.requirement_id == "owasp-4")
    assert row.status == "fail"
    assert "leak" in row.affected_components


def test_owasp5_agent_with_observes_passes():
    obs = AIComponent(name="otel", component_type=AIComponentType.OBSERVABILITY, file_path="t.py")
    ag = AIComponent(name="a1", component_type=AIComponentType.AGENT, description="d", framework="f", file_path="a.py")
    rel = ComponentRelationship(
        source_instance_id=ag.instance_id,
        target_instance_id=obs.instance_id,
        relationship_type=RelationshipType.OBSERVES,
        source_name=ag.name,
        target_name=obs.name,
        source_type=AIComponentType.AGENT,
        target_type=AIComponentType.OBSERVABILITY,
    )
    r = evaluate_compliance(
        ScanResult(sources=[_src([ag, obs], [rel])]),
        ComplianceFramework.OWASP_AGENTIC,
    )
    row = next(x for x in r.results if x.requirement_id == "owasp-5")
    assert row.status == "pass"


def test_nist1_empty_scan_fails():
    r = evaluate_compliance(ScanResult(), ComplianceFramework.NIST_AI_RMF)
    row = next(x for x in r.results if x.requirement_id == "nist-1")
    assert row.status == "fail"


def test_nist2_model_without_version_fails():
    m = AIComponent(name="m", component_type=AIComponentType.MODEL, model_name="gpt-4")
    r = evaluate_compliance(ScanResult(sources=[_src([m])]), ComplianceFramework.NIST_AI_RMF)
    row = next(x for x in r.results if x.requirement_id == "nist-2")
    assert row.status == "fail"


def test_nist2_registry_style_model_passes():
    m = AIComponent(name="m", component_type=AIComponentType.MODEL, model_name="acme-org/my-model")
    r = evaluate_compliance(ScanResult(sources=[_src([m])]), ComplianceFramework.NIST_AI_RMF)
    row = next(x for x in r.results if x.requirement_id == "nist-2")
    assert row.status == "pass"


def test_all_frameworks_evaluate():
    scan = ScanResult(sources=[_src([])])
    reports = [
        evaluate_compliance(scan, fw)
        for fw in ComplianceFramework
    ]
    assert len(reports) == 3
    assert all(len(rep.results) == 5 for rep in reports)


def test_coverage_percentage_excludes_not_applicable():
    scan = ScanResult()
    r = evaluate_compliance(scan, ComplianceFramework.EU_AI_ACT)
    assert r.summary["not_applicable"] >= 1
    assert r.summary["passed"] + r.summary["failed"] + r.summary["not_applicable"] == 5
    p, f = r.summary["passed"], r.summary["failed"]
    expected = (p / (p + f) * 100.0) if (p + f) else 100.0
    assert abs(r.summary["coverage_pct"] - expected) < 1e-6
    assert abs(r.coverage_pct - expected) < 1e-6


def test_not_applicable_requirement():
    scan = ScanResult()
    r = evaluate_compliance(scan, ComplianceFramework.OWASP_AGENTIC)
    ow2 = next(x for x in r.results if x.requirement_id == "owasp-2")
    assert ow2.status == "not_applicable"


def test_parse_compliance_cli_value():
    assert parse_compliance_cli_value("eu-ai-act") == ComplianceFramework.EU_AI_ACT
    assert parse_compliance_cli_value("EU_AI_ACT") == ComplianceFramework.EU_AI_ACT
    assert parse_compliance_cli_value("all") == "all"


def test_parse_compliance_cli_value_invalid():
    with pytest.raises(ValueError):
        parse_compliance_cli_value("nope")
