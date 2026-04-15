from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from aibom.cli import app
from aibom.models import AIComponent, AIComponentType, RiskScore, ScanResult, SourceResult
from aibom.reporters.json_reporter import JsonReporter

runner = CliRunner()


def _write_report(path: Path, *, include_version: bool = True) -> dict:
    result = ScanResult(
        metadata={
            "run_id": "run-123",
            "analyzer_version": "1.2.3",
            "completed_at": "2026-04-11T12:00:00Z",
        },
        sources=[
            SourceResult(
                path="/repo/service-a",
                components=[
                    AIComponent(
                        name="router_agent",
                        component_type=AIComponentType.AGENT,
                        file_path="/repo/service-a/app.py",
                        line_number=12,
                    )
                ],
                relationships=[],
            )
        ],
        risk=RiskScore(),
        errors=[],
    )
    buf = StringIO()
    JsonReporter().render(result, buf)
    data = json.loads(buf.getvalue())
    if not include_version:
        data["aibom_analysis"]["metadata"].pop("report_schema_version", None)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def test_report_root_compatibility_renders_report(tmp_path: Path):
    report_file = tmp_path / "report.json"
    _write_report(report_file)

    result = runner.invoke(app, ["report", str(report_file)])

    assert result.exit_code == 0
    assert "Report Summary" in result.output


def test_report_show_renders_report(tmp_path: Path):
    report_file = tmp_path / "report.json"
    _write_report(report_file)

    result = runner.invoke(app, ["report", "show", str(report_file)])

    assert result.exit_code == 0
    assert "Report Summary" in result.output


@patch("aibom.cli.post_report_with_retries")
def test_report_upload_posts_submission_payload(mock_post, tmp_path: Path):
    report_file = tmp_path / "report.json"
    _write_report(report_file)

    result = runner.invoke(
        app,
        [
            "report",
            "upload",
            str(report_file),
            "--format",
            "json",
            "--post-url",
            "https://mgmt.example.test/upload",
            "--ai-defense-api-key",
            "tenant-key",
        ],
    )

    assert result.exit_code == 0
    mock_post.assert_called_once()
    payload = mock_post.call_args.args[1]
    assert payload["run_id"] == "run-123"
    assert payload["source_kind"] == "SOURCE_KIND_LOCAL_PATH"
    assert payload["sources"] == [{"name": "service-a", "path": "/repo/service-a"}]


@patch("aibom.cli.post_report_with_retries")
def test_report_upload_accepts_unversioned_json_with_warning(mock_post, tmp_path: Path):
    report_file = tmp_path / "report.json"
    _write_report(report_file, include_version=False)

    result = runner.invoke(
        app,
        [
            "report",
            "upload",
            str(report_file),
            "--format",
            "json",
            "--post-url",
            "https://mgmt.example.test/upload",
        ],
    )

    assert result.exit_code == 0
    assert "deprecated schema" in result.output.lower()
    mock_post.assert_called_once()


def test_report_upload_rejects_non_aibom_json(tmp_path: Path):
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps({"not": "aibom"}), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "report",
            "upload",
            str(report_file),
            "--format",
            "json",
            "--post-url",
            "https://mgmt.example.test/upload",
        ],
    )

    assert result.exit_code == 1
    assert "aibom_analysis" in result.output
