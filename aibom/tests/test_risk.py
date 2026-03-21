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

from aibom.models import (
    AIComponent,
    AIComponentType,
    RiskFlag,
    RiskScore,
    ScanResult,
    SourceResult,
    Severity,
)
from aibom.risk import RiskScorer


def test_empty_scan_result_score_zero_info():
    result = ScanResult()
    risk = RiskScorer().score(result)
    assert risk.score == 0
    assert risk.severity == Severity.INFO
    assert risk.flags == []


def test_secret_triggers_hardcoded_api_key():
    comp = AIComponent(name="k", component_type=AIComponentType.SECRET, file_path="x.py")
    result = ScanResult(sources=[SourceResult(path="x.py", components=[comp])])
    risk = RiskScorer().score(result)
    assert any(f.flag == "hardcoded_api_key" for f in risk.flags)


def test_mcp_server_without_framework_triggers_mcp_unknown_server():
    comp = AIComponent(
        name="srv",
        component_type=AIComponentType.MCP_SERVER,
        framework="",
        file_path="mcp.json",
    )
    result = ScanResult(sources=[SourceResult(path="mcp.json", components=[comp])])
    risk = RiskScorer().score(result)
    assert any(f.flag == "mcp_unknown_server" for f in risk.flags)


def test_unpinned_model_flag():
    unpinned = AIComponent(
        name="m",
        component_type=AIComponentType.MODEL,
        model_name="gpt-4",
    )
    pinned = AIComponent(
        name="m2",
        component_type=AIComponentType.MODEL,
        model_name="gpt-4-0613",
    )
    r1 = RiskScorer().score(
        ScanResult(sources=[SourceResult(path="a", components=[unpinned])])
    )
    r2 = RiskScorer().score(
        ScanResult(sources=[SourceResult(path="b", components=[pinned])])
    )
    assert any(f.flag == "unpinned_model" for f in r1.flags)
    assert not any(f.flag == "unpinned_model" for f in r2.flags)


def test_metadata_risk_flags_critical_cve():
    comp = AIComponent(
        name="dep",
        component_type=AIComponentType.DEPENDENCY,
        metadata={"risk_flags": ["critical_cve"]},
    )
    result = ScanResult(sources=[SourceResult(path="p", components=[comp])])
    risk = RiskScorer().score(result)
    assert any(f.flag == "critical_cve" for f in risk.flags)


@pytest.mark.parametrize(
    ("weight", "expected_severity"),
    [
        (0, Severity.INFO),
        (10, Severity.LOW),
        (30, Severity.MEDIUM),
        (55, Severity.HIGH),
        (80, Severity.CRITICAL),
    ],
)
def test_severity_bands(weight: int, expected_severity: Severity):
    risk = RiskScore()
    if weight > 0:
        risk.add_flag(
            RiskFlag(
                flag="band",
                severity=Severity.INFO,
                weight=weight,
                description="",
            )
        )
    assert risk.score == weight
    assert risk.severity == expected_severity


@pytest.mark.parametrize(
    ("risk_severity", "threshold", "expected"),
    [
        (Severity.INFO, Severity.HIGH, False),
        (Severity.LOW, Severity.HIGH, False),
        (Severity.MEDIUM, Severity.HIGH, False),
        (Severity.HIGH, Severity.HIGH, True),
        (Severity.CRITICAL, Severity.HIGH, True),
        (Severity.LOW, Severity.LOW, True),
    ],
)
def test_should_fail(risk_severity: Severity, threshold: Severity, expected: bool):
    risk = RiskScore(severity=risk_severity, score=0)
    assert RiskScorer().should_fail(risk, threshold) is expected


def test_weight_override_hardcoded_api_key():
    comp = AIComponent(name="k", component_type=AIComponentType.SECRET)
    result = ScanResult(sources=[SourceResult(path="x", components=[comp])])
    risk = RiskScorer(weight_overrides={"hardcoded_api_key": 10}).score(result)
    assert risk.score == 10
    assert any(f.flag == "hardcoded_api_key" and f.weight == 10 for f in risk.flags)


def test_score_caps_at_100():
    comps = [
        AIComponent(
            name=f"c{i}",
            component_type=AIComponentType.DEPENDENCY,
            metadata={"risk_flags": ["critical_cve"]},
        )
        for i in range(5)
    ]
    result = ScanResult(sources=[SourceResult(path="p", components=comps)])
    risk = RiskScorer().score(result)
    assert risk.score == 100
