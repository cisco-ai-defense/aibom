# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import patch

from typer.testing import CliRunner

from aibom.cli import app

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


@patch("aibom.multi_repo.is_git_url", return_value=True)
@patch("aibom.multi_repo.ClonedRepo")
def test_analyze_records_clone_failures_in_json_output(mock_cloned_repo, _mock_is_git_url, tmp_path):
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
