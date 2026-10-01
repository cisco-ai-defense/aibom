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
    CodeSnippet,
    ComponentRelationship,
    DecisionAnnotation,
    EvidenceLocation,
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


def _source_with_three_components(path: str, name_prefix: str) -> SourceResult:
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


def test_friendly_source_name(tmp_path, monkeypatch):
    from unittest.mock import patch

    from aibom.reporters.json_reporter import _friendly_source_name

    def _fake_run(cmd, **_kw):
        class R:
            returncode = 0
            stdout = "git@github.com:acme-org/my-service.git\n"

        return R()

    with patch("aibom.reporters.json_reporter.subprocess.run", side_effect=_fake_run):
        assert _friendly_source_name(str(tmp_path)) == "acme-org/my-service"

    def _fake_https(cmd, **_kw):
        class R:
            returncode = 0
            stdout = "https://github.com/org/repo.git\n"

        return R()

    with patch("aibom.reporters.json_reporter.subprocess.run", side_effect=_fake_https):
        assert _friendly_source_name(str(tmp_path)) == "org/repo"

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
    assert analysis["metadata"]["report_schema_version"] == "2"
    source_keys = set(analysis["sources"].keys())
    assert source_keys == {"alpha", "beta"}
    for src_path, src_data in analysis["sources"].items():
        comps = src_data["components"]
        assert set(comps.keys()) == {"model", "agent", "tool"}
        assert len(comps["model"]) == 1
        assert len(comps["agent"]) == 1
        assert len(comps["tool"]) == 1
        assert "summary" in src_data


def test_json_reporter_stamps_kb_identity_and_groups_coverage_gaps(monkeypatch):
    monkeypatch.setenv("CISCO_AI_DEFENSE_API_KEY", "test-key")
    monkeypatch.setattr(
        "aibom.kb.manager.KBManager.output_metadata",
        lambda _self: {
            "kb_version": "2.4.0",
            "build_type": "delta",
            "schema_version": 2,
            "cli_version": "2.0.0",
        },
    )
    components = [
        AIComponent(
            name="ExampleClient",
            component_type=AIComponentType.MODEL,
            metadata={
                "uncatalogued_ai_symbol": True,
                "uncatalogued_symbol": "example_sdk.ExampleClient",
                "ecosystem": "pypi",
                "package_name": "example-sdk",
            },
        ),
        AIComponent(
            name="ExampleModel",
            component_type=AIComponentType.MODEL,
            metadata={
                "uncatalogued_ai_symbol": True,
                "uncatalogued_symbol": "example_sdk.ExampleModel",
                "ecosystem": "pypi",
                "package_name": "example-sdk",
            },
        ),
    ]
    result = ScanResult(
        metadata={"run_id": "scan-cache-001"},
        sources=[SourceResult(path="/tmp/example", components=components)],
    )
    output = StringIO()
    JsonReporter().render(result, output)
    analysis = json.loads(output.getvalue())["aibom_analysis"]

    assert (
        analysis["metadata"]
        | {
            "kb_version": "2.4.0",
            "build_type": "delta",
            "schema_version": 2,
            "cli_version": "2.0.0",
        }
        == analysis["metadata"]
    )
    assert analysis["coverage_gaps"] == {
        "informational": True,
        "uncatalogued_ai_symbol_count": 2,
        "scan_cache_id": "scan-cache-001",
        "packages": [
            {
                "ecosystem": "pypi",
                "package_name": "example-sdk",
                "symbols": [
                    "example_sdk.ExampleClient",
                    "example_sdk.ExampleModel",
                ],
            }
        ],
        "request_hint": (
            "Run `cisco-aibom kb request --from-scan <report.json>` to request "
            "coverage for these symbols."
        ),
    }


def test_json_reporter_preserves_decision_annotations():
    component = AIComponent(
        name="router_agent",
        component_type=AIComponentType.AGENT,
        file_path="/proj/src/app.py",
        line_number=12,
        decision_annotation=DecisionAnnotation(
            decision="confirmed",
            justification="The code instantiates and invokes the agent in the request path.",
            evidence_kinds=["code_context"],
            evidence_locations=[
                EvidenceLocation(
                    file_path="/proj/src/app.py",
                    start_line=12,
                    end_line=18,
                    role="primary",
                )
            ],
            code_snippet=CodeSnippet(
                file_path="/proj/src/app.py",
                start_line=12,
                end_line=14,
                text="agent = RouterAgent()\nagent.run(task)\n",
                truncated=False,
            ),
        ),
    )
    result = ScanResult(
        metadata={"analyzer_version": "2.0.0-test", "run_id": "run-test-annotations"},
        sources=[
            SourceResult(path="/proj/src", components=[component], relationships=[])
        ],
        risk=RiskScore(),
    )

    buf = StringIO()
    JsonReporter().render(result, buf)
    data = json.loads(buf.getvalue())

    rendered = data["aibom_analysis"]["sources"]["src"]["components"]["agent"][0]
    assert rendered["decision_annotation"]["decision"] == "confirmed"
    assert rendered["decision_annotation"]["evidence_locations"][0]["role"] == "primary"
    assert rendered["decision_annotation"]["code_snippet"]["truncated"] is False


def test_json_reporter_disambiguates_colliding_source_names(
    sample_scan_result: ScanResult,
):
    from unittest.mock import patch

    buf = StringIO()
    with patch(
        "aibom.reporters.json_reporter._friendly_source_name", return_value="dup"
    ):
        get_reporter("json").render(sample_scan_result, buf)

    data = json.loads(buf.getvalue())
    sources = data["aibom_analysis"]["sources"]
    assert set(sources.keys()) == {"dup", "dup#2"}
    assert sources["dup"]["source_name"] == "dup"
    assert sources["dup"]["source_path"] == "/proj/src/alpha"
    assert sources["dup#2"]["source_name"] == "dup"
    assert sources["dup#2"]["source_path"] == "/proj/src/beta"


def test_json_reporter_emits_source_attribution_block(
    sample_scan_result: ScanResult,
):
    buf = StringIO()
    JsonReporter().render(sample_scan_result, buf)
    data = json.loads(buf.getvalue())

    sources = data["aibom_analysis"]["sources"]
    for src in sources.values():
        meta = src["metadata"]
        # The attribution triple is always present.
        assert "source_kind" in meta
        assert "source_ref_canonical" in meta
        assert "source_ref_version" in meta
        # Fixture paths are not real checkouts -> local-path, empty refs.
        assert meta["source_kind"] == "local-path"
        assert meta["source_ref_canonical"] == ""
        assert meta["source_ref_version"] == ""


def test_json_reporter_source_attribution_prefers_supplied_detail(
    sample_scan_result: ScanResult,
):
    # Pipeline / cross-repo discovery may already know the attribution; those
    # values must win over local-path derivation.
    sample_scan_result.metadata["_report_source_details"] = {
        "/proj/src/alpha": {
            "source_kind": "git",
            "source_ref_canonical": "github.com/org/repo",
            "source_ref_version": "a" * 40,
        }
    }
    buf = StringIO()
    JsonReporter().render(sample_scan_result, buf)
    data = json.loads(buf.getvalue())

    alpha = data["aibom_analysis"]["sources"]["alpha"]
    assert alpha["summary"]["source_kind"] == "git"
    assert alpha["metadata"]["source_kind"] == "git"
    assert alpha["metadata"]["source_ref_canonical"] == "github.com/org/repo"
    assert alpha["metadata"]["source_ref_version"] == "a" * 40


def test_json_reporter_omits_component_summary_by_default(
    sample_scan_result: ScanResult,
):
    buf = StringIO()
    JsonReporter().render(sample_scan_result, buf)
    data = json.loads(buf.getvalue())

    assert "component_summary" not in data["aibom_analysis"]


def test_json_reporter_emits_component_summary_when_enabled(
    sample_scan_result: ScanResult,
):
    buf = StringIO()
    JsonReporter(include_component_summary=True).render(sample_scan_result, buf)
    data = json.loads(buf.getvalue())

    analysis = data["aibom_analysis"]
    assert "component_summary" in analysis
    summary = analysis["component_summary"]

    assert set(summary.keys()) == set(analysis["sources"].keys()) == {"alpha", "beta"}

    alpha_entries = summary["alpha"]
    assert alpha_entries == [
        {
            "component_type": "agent",
            "name": "alpha-agent",
            "file_path": "/proj/src/alpha",
            "line_number": 20,
        },
        {
            "component_type": "model",
            "name": "alpha-model",
            "file_path": "/proj/src/alpha",
            "line_number": 10,
        },
        {
            "component_type": "tool",
            "name": "alpha-tool",
            "file_path": "/proj/src/alpha",
            "line_number": 30,
        },
    ]

    assert [e["name"] for e in summary["beta"]] == [
        "beta-agent",
        "beta-model",
        "beta-tool",
    ]

    keys = list(analysis.keys())
    assert keys.index("component_summary") == keys.index("summary") + 1
    assert keys.index("component_summary") < keys.index("risk")


def test_json_reporter_component_summary_excludes_test_only():
    real_agent = AIComponent(
        name="router_agent",
        component_type=AIComponentType.AGENT,
        file_path="/proj/src/app.py",
        line_number=12,
    )
    test_fixture_agent = AIComponent(
        name="fixture_agent",
        component_type=AIComponentType.AGENT,
        file_path="/proj/tests/test_app.py",
        line_number=7,
        metadata={"test_only": True},
    )
    result = ScanResult(
        metadata={"analyzer_version": "2.0.0-test", "run_id": "run-test-only"},
        sources=[
            SourceResult(
                path="/proj/src",
                components=[real_agent, test_fixture_agent],
                relationships=[],
            )
        ],
        risk=RiskScore(),
    )

    buf = StringIO()
    JsonReporter(include_component_summary=True).render(result, buf)
    data = json.loads(buf.getvalue())

    src_entry = data["aibom_analysis"]["sources"]["src"]
    agent_names_in_sources = {c["name"] for c in src_entry["components"]["agent"]}
    assert agent_names_in_sources == {"router_agent", "fixture_agent"}

    summary_entries = data["aibom_analysis"]["component_summary"]["src"]
    assert [e["name"] for e in summary_entries] == ["router_agent"]


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
    properties = {item["name"]: item["value"] for item in doc["properties"]}
    assert set(properties) >= {
        "cisco-aibom:kb_version",
        "cisco-aibom:build_type",
        "cisco-aibom:schema_version",
        "cisco-aibom:cli_version",
        "cisco-aibom:uncatalogued_ai_symbol_count",
    }


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
    properties = doc["@graph"][0]["extension_cisco_aibom"]["extensionProperties"]
    assert set(properties) >= {
        "cisco-aibom:kb_version",
        "cisco-aibom:build_type",
        "cisco-aibom:schema_version",
        "cisco-aibom:cli_version",
        "cisco-aibom:uncatalogued_ai_symbol_count",
    }


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
