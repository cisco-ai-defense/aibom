# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

from typer.testing import CliRunner

from aibom.cli import app
from aibom.scan_pipeline import PipelineResult

runner = CliRunner()


def test_analyze_requires_output_file_for_json():
    result = runner.invoke(app, ["analyze", "src", "--output-format", "json"])
    assert result.exit_code != 0


def test_analyze_rejects_invalid_output_format():
    result = runner.invoke(app, ["analyze", "src", "--output-format", "bad"])
    assert result.exit_code != 0
    assert "Invalid output format" in result.output


def test_analyze_rejects_legacy_ui_output_format():
    result = runner.invoke(app, ["analyze", "src", "--output-format", "ui"])
    assert result.exit_code != 0
    assert "Invalid output format" in result.output


def test_analyze_defaults_cache_root_for_scan_and_agentic(tmp_path):
    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")
    report = tmp_path / "report.json"
    shared_root = tmp_path / "shared-cache"
    seen: dict[str, object] = {}

    def fake_pipeline_run(self):
        seen["agentic_cache_dir"] = self.agentic_cache_dir
        return PipelineResult(
            components=[],
            relationships=[],
            agentic_risk_flags=[],
            agentic_candidate_count=0,
            external_deps=[],
            timings=[],
            total_elapsed_s=0.0,
        )

    with patch("aibom.cli.ensure_llm_runtime_available"):
        with patch("aibom.cli.resolve_cache_root", return_value=shared_root):
            with patch("aibom.scan_cache.load_cached", return_value=None) as mock_load:
                with patch("aibom.scan_cache.save_cached") as mock_save:
                    with patch("aibom.scan_pipeline.ScanPipeline.run", fake_pipeline_run):
                        result = runner.invoke(
                            app,
                            [
                                "analyze",
                                str(source_dir),
                                "--output-format",
                                "json",
                                "--output-file",
                                str(report),
                                "--llm-model",
                                "test-model",
                            ],
                        )

    assert result.exit_code == 0
    assert mock_load.call_args.args[0] == shared_root / "scan"
    assert mock_save.call_args.args[0] == shared_root / "scan"
    assert seen["agentic_cache_dir"] == shared_root / "agentic"


def test_analyze_component_summary_flag_adds_key_to_json_report(tmp_path):
    from aibom.models import AIComponent, AIComponentType

    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")
    report = tmp_path / "report.json"

    detected = [
        AIComponent(
            name="router_agent",
            component_type=AIComponentType.AGENT,
            file_path=str(source_dir / "app.py"),
            line_number=12,
        ),
        AIComponent(
            name="calc_tool",
            component_type=AIComponentType.TOOL,
            file_path=str(source_dir / "app.py"),
            line_number=25,
        ),
    ]

    def fake_pipeline_run(self):
        return PipelineResult(
            components=list(detected),
            relationships=[],
            agentic_risk_flags=[],
            agentic_candidate_count=0,
            external_deps=[],
            timings=[],
            total_elapsed_s=0.0,
        )

    with patch("aibom.cli.ensure_llm_runtime_available"):
        with patch("aibom.scan_cache.load_cached", return_value=None):
            with patch("aibom.scan_cache.save_cached"):
                with patch("aibom.scan_pipeline.ScanPipeline.run", fake_pipeline_run):
                    result = runner.invoke(
                        app,
                        [
                            "analyze",
                            str(source_dir),
                            "--output-format",
                            "json",
                            "--output-file",
                            str(report),
                            "--component-summary",
                            "--llm-model",
                            "test-model",
                        ],
                    )

    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text())
    analysis = data["aibom_analysis"]
    assert "component_summary" in analysis
    source_summaries = analysis["component_summary"]
    assert len(source_summaries) == 1
    entries = next(iter(source_summaries.values()))
    assert entries == [
        {
            "component_type": "agent",
            "name": "router_agent",
            "file_path": str(source_dir / "app.py"),
            "line_number": 12,
        },
        {
            "component_type": "tool",
            "name": "calc_tool",
            "file_path": str(source_dir / "app.py"),
            "line_number": 25,
        },
    ]


def test_analyze_without_component_summary_flag_omits_key(tmp_path):
    from aibom.models import AIComponent, AIComponentType

    source_dir = tmp_path / "repo"
    source_dir.mkdir()
    (source_dir / "app.py").write_text("print('hello')\n", encoding="utf-8")
    report = tmp_path / "report.json"

    def fake_pipeline_run(self):
        return PipelineResult(
            components=[
                AIComponent(
                    name="router_agent",
                    component_type=AIComponentType.AGENT,
                    file_path=str(source_dir / "app.py"),
                    line_number=12,
                )
            ],
            relationships=[],
            agentic_risk_flags=[],
            agentic_candidate_count=0,
            external_deps=[],
            timings=[],
            total_elapsed_s=0.0,
        )

    with patch("aibom.cli.ensure_llm_runtime_available"):
        with patch("aibom.scan_cache.load_cached", return_value=None):
            with patch("aibom.scan_cache.save_cached"):
                with patch("aibom.scan_pipeline.ScanPipeline.run", fake_pipeline_run):
                    result = runner.invoke(
                        app,
                        [
                            "analyze",
                            str(source_dir),
                            "--output-format",
                            "json",
                            "--output-file",
                            str(report),
                            "--llm-model",
                            "test-model",
                        ],
                    )

    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text())
    assert "component_summary" not in data["aibom_analysis"]


@patch("aibom.cli.ensure_llm_runtime_available", return_value=None)
@patch("aibom.multi_repo.is_git_url", return_value=True)
@patch("aibom.multi_repo.ClonedRepo")
def test_analyze_records_clone_failures_in_json_output(mock_cloned_repo, _mock_is_git_url, _mock_preflight, tmp_path):
    report = tmp_path / "report.json"
    mock_cloned_repo.return_value.__enter__.side_effect = RuntimeError("network down")

    result = runner.invoke(
        app,
        [
            "analyze",
            "https://github.com/acme/bad-repo.git",
            "--output-format",
            "json",
            "--output-file",
            str(report),
            "--llm-model",
            "test-model",
            "--llm-api-base",
            "http://localhost:11434",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(report.read_text())
    analysis = data["aibom_analysis"]
    assert analysis["metadata"]["error_count"] == 1
    assert analysis["metadata"]["sources_analyzed"] == 1
    assert analysis["metadata"]["sources_with_errors"] == 1
    assert analysis["metadata"]["status"] == "failed"
    assert analysis["errors"] == ["Clone failed: network down"]
