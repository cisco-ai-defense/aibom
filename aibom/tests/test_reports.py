# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from aibom.__main__ import _generate_json_report
from aibom.cli import _build_submission_payload


def test_json_report_includes_workflow_ids(tmp_path):
    component = {
        "name": "example_component",
        "file_path": "/tmp/app/module.py",
        "line_number": 42,
        "workflows": [
            {
                "function": "app.workflow.entrypoint",
                "file_path": "/tmp/app/workflows.py",
                "line": 10,
                "distance": 0,
            }
        ],
    }
    output = SimpleNamespace(components={"model": [component]}, relationships=[])
    report_path = tmp_path / "report.json"

    _generate_json_report({"sample_source": output}, report_path)

    parsed = json.loads(report_path.read_text())
    source = parsed["aibom_analysis"]["sources"]["sample_source"]

    assert source["total_workflows"] == 1
    workflows = source["workflows"]
    assert workflows and "id" in workflows[0]

    component_workflows = source["components"]["model"][0]["workflows"]
    assert component_workflows[0]["id"] == workflows[0]["id"]

    summary = parsed["aibom_analysis"]["summary"]
    assert summary["total_workflows"] == 1


def test_submission_payload_wraps_report(tmp_path):
    component = {
        "name": "example_component",
        "file_path": "/tmp/app/module.py",
        "line_number": 42,
    }
    output = SimpleNamespace(components={"model": [component]}, relationships=[])
    report_path = tmp_path / "report.json"
    metadata = {
        "run_id": "run-123",
        "analyzer_version": "1.2.3",
        "completed_at": "2025-01-01T12:00:00Z",
    }
    source_summaries = {
        "repo": {
            "source_kind": "local-path",
            "status": "completed",
        }
    }

    report = _generate_json_report({"repo": output}, report_path, metadata=metadata, source_summaries=source_summaries)
    payload = _build_submission_payload(
        report,
        {"repo": {"source_kind": "local-path", "source_path": "/app", "source_name": "repo"}},
    )

    assert payload["run_id"] == "run-123"
    assert payload["analyzer_version"] == "1.2.3"
    assert payload["submitted_at"] == "2025-01-01T12:00:00Z"
    assert payload["source_kind"] == "SOURCE_KIND_LOCAL_PATH"
    assert payload["sources"] == [{"name": "repo", "path": "/app"}]
    assert payload["report"] == report
