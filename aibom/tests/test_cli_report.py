from __future__ import annotations

import json
import uuid
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


def _write_multi_source_report(
    path: Path, sources_spec: list[dict], *, run_id: str = "run-123"
) -> None:
    """Write a report whose per-source metadata carries an attribution triple.

    Injecting ``source_outcomes`` into the scan metadata lets the JSON reporter
    embed the triple deterministically without touching git or a registry.
    """
    source_outcomes: dict = {}
    sources = []
    for spec in sources_spec:
        sources.append(
            SourceResult(
                path=spec["path"],
                components=[
                    AIComponent(
                        name="router_agent",
                        component_type=AIComponentType.AGENT,
                        file_path=spec["path"] + "/app.py",
                        line_number=12,
                    )
                ],
                relationships=[],
            )
        )
        source_outcomes[spec["path"]] = {
            "source_name": spec["name"],
            "source_path": spec["path"],
            "source_kind": spec["kind"],
            "source_ref_canonical": spec["canonical"],
            "source_ref_version": spec["version"],
            "status": "completed",
        }
    result = ScanResult(
        metadata={
            "run_id": run_id,
            "analyzer_version": "1.2.3",
            "completed_at": "2026-04-11T12:00:00Z",
            "source_outcomes": source_outcomes,
        },
        sources=sources,
        risk=RiskScore(),
        errors=[],
    )
    buf = StringIO()
    JsonReporter().render(result, buf)
    path.write_text(buf.getvalue(), encoding="utf-8")


def _invoke_upload(report_file: Path):
    return runner.invoke(
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


@patch("aibom.cli.post_report_with_retries")
def test_single_git_source_uploads_once_with_full_triple(mock_post, tmp_path: Path):
    report_file = tmp_path / "report.json"
    _write_multi_source_report(
        report_file,
        [
            {
                "path": "/repo/service-a",
                "name": "org/service-a",
                "kind": "git",
                "canonical": "github.com/org/service-a",
                "version": "abc123",
            }
        ],
    )

    result = _invoke_upload(report_file)

    assert result.exit_code == 0, result.output
    assert mock_post.call_count == 1
    payload = mock_post.call_args.args[1]
    assert payload["run_id"] == "run-123"
    assert payload["source_attribution"] == {
        "source_kind": "git",
        "source_ref_canonical": "github.com/org/service-a",
        "source_ref_version": "abc123",
    }


@patch("aibom.cli.post_report_with_retries")
def test_two_sources_fan_out_to_two_uploads(mock_post, tmp_path: Path):
    report_file = tmp_path / "report.json"
    _write_multi_source_report(
        report_file,
        [
            {
                "path": "/repo/service-a",
                "name": "org/service-a",
                "kind": "git",
                "canonical": "github.com/org/service-a",
                "version": "aaa",
            },
            {
                "path": "registry/app:1.0",
                "name": "registry/app",
                "kind": "container_image",
                "canonical": "registry.example.com/app",
                "version": "sha256:deadbeef",
            },
        ],
    )

    result = _invoke_upload(report_file)

    assert result.exit_code == 0, result.output
    assert mock_post.call_count == 2
    payloads = [call.args[1] for call in mock_post.call_args_list]

    run_ids = {p["run_id"] for p in payloads}
    assert len(run_ids) == 2, f"run_ids must be distinct, got {run_ids}"
    # A fanned-out run_id must stay a valid UUID (the backend keys ingests on it).
    for rid in run_ids:
        uuid.UUID(str(rid))

    canonicals = {p["source_attribution"]["source_ref_canonical"] for p in payloads}
    assert canonicals == {"github.com/org/service-a", "registry.example.com/app"}

    kinds = {p["source_attribution"]["source_kind"] for p in payloads}
    assert kinds == {"git", "container_image"}

    # Each upload's report is scoped to exactly one source.
    for p in payloads:
        assert len(p["report"]["aibom_analysis"]["sources"]) == 1


@patch("aibom.cli.post_report_with_retries")
def test_unresolved_source_uploads_without_attribution(mock_post, tmp_path: Path):
    report_file = tmp_path / "report.json"
    _write_multi_source_report(
        report_file,
        [
            {
                "path": "/tmp/extracted-tarball",
                "name": "extracted",
                "kind": "local-path",
                "canonical": "",
                "version": "",
            }
        ],
    )

    result = _invoke_upload(report_file)

    assert result.exit_code == 0, result.output
    assert mock_post.call_count == 1
    payload = mock_post.call_args.args[1]
    assert "source_attribution" not in payload
