# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

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
