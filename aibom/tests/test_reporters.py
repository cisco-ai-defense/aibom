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
from io import StringIO
from xml.etree import ElementTree as ET

import pytest

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
from aibom.reporters import (
    CsvReporter,
    HtmlReporter,
    JsonReporter,
    JunitReporter,
    MarkdownReporter,
    PlaintextReporter,
    SpdxReporter,
    get_reporter,
    reporter_registry,
)
from aibom.reporters.cyclonedx_reporter import CycloneDxReporter
from aibom.reporters.sarif_reporter import SarifReporter

EXPECTED_REPORTER_NAMES = frozenset(
    {
        "json",
        "plaintext",
        "cyclonedx",
        "sarif",
        "spdx",
        "html",
        "markdown",
        "csv",
        "junit",
    }
)


def _source_with_three_components(
    path: str, name_prefix: str
) -> SourceResult:
    model = AIComponent(
        name=f"{name_prefix}-model",
        component_type=AIComponentType.MODEL,
        file_path=path,
        line_number=10,
    )
    agent = AIComponent(
        name=f"{name_prefix}-agent",
        component_type=AIComponentType.AGENT,
        file_path=path,
        line_number=20,
    )
    tool = AIComponent(
        name=f"{name_prefix}-tool",
        component_type=AIComponentType.TOOL,
        file_path=path,
        line_number=30,
    )
    rel = ComponentRelationship(
        source_instance_id=agent.instance_id,
        target_instance_id=tool.instance_id,
        relationship_type=RelationshipType.USES_TOOL,
        source_name=agent.name,
        target_name=tool.name,
        source_type=AIComponentType.AGENT,
        target_type=AIComponentType.TOOL,
    )
    return SourceResult(
        path=path,
        components=[model, agent, tool],
        relationships=[rel],
    )


@pytest.fixture
def sample_scan_result() -> ScanResult:
    risk = RiskScore()
    risk.add_flag(
        RiskFlag(
            flag="hardcoded_api_key",
            severity=Severity.HIGH,
            weight=35,
            description="API key embedded in source",
        )
    )
    return ScanResult(
        metadata={
            "analyzer_version": "2.0.0-test",
            "run_id": "run-test-001",
        },
        sources=[
            _source_with_three_components("/proj/src/alpha", "alpha"),
            _source_with_three_components("/proj/src/beta", "beta"),
        ],
        risk=risk,
    )


def test_reporter_registry_lists_all_nine_reporters():
    names = {cls.name for cls in reporter_registry if cls.name}
    assert names == EXPECTED_REPORTER_NAMES


@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("json", JsonReporter),
        ("plaintext", PlaintextReporter),
        ("cyclonedx", CycloneDxReporter),
        ("sarif", SarifReporter),
        ("spdx", SpdxReporter),
        ("html", HtmlReporter),
        ("markdown", MarkdownReporter),
        ("csv", CsvReporter),
        ("junit", JunitReporter),
    ],
)
def test_get_reporter_returns_expected_instance(name, expected_cls):
    rep = get_reporter(name)
    assert rep is not None
    assert isinstance(rep, expected_cls)


def test_get_reporter_unknown_returns_none():
    assert get_reporter("not_a_real_format") is None


def test_friendly_source_name():
    from aibom.reporters.json_reporter import _friendly_source_name

    assert _friendly_source_name("/Users/me/work/github.com/acme-org/my-service") == "acme-org/my-service"
    assert _friendly_source_name("/Users/me/work/github.com/org/repo") == "org/repo"
    assert _friendly_source_name("/tmp/sample-ai-app") == "sample-ai-app"
    assert _friendly_source_name("/proj/src/alpha") == "alpha"


def test_json_reporter_render(sample_scan_result: ScanResult):
    buf = StringIO()
    get_reporter("json").render(sample_scan_result, buf)
    data = json.loads(buf.getvalue())
    assert "aibom_analysis" in data
    analysis = data["aibom_analysis"]
    assert analysis["metadata"]["analyzer_version"] == "2.0.0-test"
    assert analysis["metadata"]["run_id"] == "run-test-001"
    source_keys = set(analysis["sources"].keys())
    assert source_keys == {"alpha", "beta"}
    for src_path, src_data in analysis["sources"].items():
        comps = src_data["components"]
        assert set(comps.keys()) == {"model", "agent", "tool"}
        assert len(comps["model"]) == 1
        assert len(comps["agent"]) == 1
        assert len(comps["tool"]) == 1
        assert "summary" in src_data


def test_plaintext_reporter_render(sample_scan_result: ScanResult):
    buf = StringIO()
    get_reporter("plaintext").render(sample_scan_result, buf)
    text = buf.getvalue()
    assert len(text.strip()) > 0
    assert "AIBOM Analysis Report" in text
    assert "Metadata" in text
    assert "Sources" in text
    assert "Relationships summary" in text
    assert "Risk assessment" in text


def test_cyclonedx_reporter_render(sample_scan_result: ScanResult):
    buf = StringIO()
    get_reporter("cyclonedx").render(sample_scan_result, buf)
    doc = json.loads(buf.getvalue())
    assert doc["bomFormat"] == "CycloneDX"
    assert doc["specVersion"] == "1.6"
    assert isinstance(doc.get("components"), list)
    assert len(doc["components"]) == 6


def test_sarif_reporter_render(sample_scan_result: ScanResult):
    buf = StringIO()
    get_reporter("sarif").render(sample_scan_result, buf)
    doc = json.loads(buf.getvalue())
    assert doc["version"] == "2.1.0"
    runs = doc["runs"]
    assert isinstance(runs, list) and len(runs) >= 1
    driver = runs[0]["tool"]["driver"]
    assert driver["name"] == "cisco-aibom"
    assert "rules" in driver


def test_spdx_reporter_render(sample_scan_result: ScanResult):
    buf = StringIO()
    get_reporter("spdx").render(sample_scan_result, buf)
    doc = json.loads(buf.getvalue())
    assert "@context" in doc
    assert doc["@context"]


def test_html_reporter_render(sample_scan_result: ScanResult):
    buf = StringIO()
    get_reporter("html").render(sample_scan_result, buf)
    html = buf.getvalue()
    assert "<html" in html.lower()
    assert "alpha-model" in html
    assert "beta-tool" in html


def test_markdown_reporter_render(sample_scan_result: ScanResult):
    buf = StringIO()
    get_reporter("markdown").render(sample_scan_result, buf)
    md = buf.getvalue()
    assert md.startswith("# AI BOM Report")
    assert "| --- |" in md
    assert "| Metric | Value |" in md


def test_csv_reporter_render(sample_scan_result: ScanResult):
    buf = StringIO()
    get_reporter("csv").render(sample_scan_result, buf)
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 7
    header = lines[0]
    assert "name" in header and "component_type" in header


def test_junit_reporter_render(sample_scan_result: ScanResult):
    buf = StringIO()
    get_reporter("junit").render(sample_scan_result, buf)
    root = ET.fromstring(buf.getvalue())
    assert root.tag == "testsuites"
