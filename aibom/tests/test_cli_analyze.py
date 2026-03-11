# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CLI success path tests for the analyze command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from aibom.cli import app

runner = CliRunner()


@pytest.fixture
def fixture_dir():
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def _mock_db(test_catalog_db_path):
    """Patch ensure_local_database to return the test catalog for all tests."""
    with patch("aibom.cli.ensure_local_database", return_value=test_catalog_db_path):
        yield


class TestAnalyzeCommand:
    def test_analyze_json_output(self, fixture_dir, tmp_path):
        output = tmp_path / "out.json"
        result = runner.invoke(app, [
            "analyze", str(fixture_dir), "-o", "json", "-O", str(output),
            "--languages", "python",
        ])
        assert result.exit_code == 0, result.output
        assert output.exists()
        data = json.loads(output.read_text())
        assert "aibom_analysis" in data

    def test_analyze_plaintext_output(self, fixture_dir, tmp_path):
        output = tmp_path / "out.txt"
        result = runner.invoke(app, [
            "analyze", str(fixture_dir), "-o", "plaintext", "-O", str(output),
            "--languages", "python",
        ])
        assert result.exit_code == 0

    def test_analyze_show_summary(self, fixture_dir, tmp_path):
        output = tmp_path / "out.json"
        result = runner.invoke(app, [
            "analyze", str(fixture_dir), "-o", "json", "-O", str(output),
            "--languages", "python", "--show-summary",
        ])
        assert result.exit_code == 0

    def test_analyze_languages_flag_python(self, fixture_dir, tmp_path):
        output = tmp_path / "out.json"
        result = runner.invoke(app, [
            "analyze", str(fixture_dir), "-o", "json", "-O", str(output),
            "--languages", "python",
        ])
        assert result.exit_code == 0

    def test_analyze_nonexistent_source(self, tmp_path):
        output = tmp_path / "out.json"
        result = runner.invoke(app, [
            "analyze", "/nonexistent/path/abc123", "-o", "json", "-O", str(output),
        ])
        assert result.exit_code == 0

    def test_analyze_invalid_language(self, fixture_dir, tmp_path):
        output = tmp_path / "out.json"
        result = runner.invoke(app, [
            "analyze", str(fixture_dir), "-o", "json", "-O", str(output),
            "--languages", "ruby",
        ])
        assert result.exit_code == 1

    def test_analyze_mixed_valid_invalid_languages(self, fixture_dir, tmp_path):
        output = tmp_path / "out.json"
        result = runner.invoke(app, [
            "analyze", str(fixture_dir), "-o", "json", "-O", str(output),
            "--languages", "python,ruby",
        ])
        assert result.exit_code == 0
        assert output.exists()
        assert "Unsupported language" in result.output or output.stat().st_size > 0
