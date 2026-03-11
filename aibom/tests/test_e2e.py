# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests that exercise the CLI analyze command on fixture dirs."""

from __future__ import annotations

import json
import shutil
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


class TestAnalyzeE2E:
    def test_analyze_python_directory(self, fixture_dir, tmp_path):
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "analyze",
            str(fixture_dir),
            "-o", "json",
            "-O", str(output_file),
            "--languages", "python",
        ])
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        report = json.loads(output_file.read_text())
        assert "aibom_analysis" in report

    def test_analyze_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "analyze",
            str(empty_dir),
            "-o", "json",
            "-O", str(output_file),
        ])
        assert result.exit_code == 0

    def test_analyze_with_completeness(self, fixture_dir, tmp_path):
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "analyze",
            str(fixture_dir),
            "-o", "json",
            "-O", str(output_file),
            "--languages", "python",
            "--completeness",
        ])
        assert result.exit_code == 0, result.output
        if output_file.exists():
            report = json.loads(output_file.read_text())
            sources = report.get("aibom_analysis", {}).get("sources", {})
            for _source_name, source_data in sources.items():
                if "completeness" in source_data:
                    assert "score" in source_data["completeness"]
                    assert "warnings" in source_data["completeness"]

    def test_analyze_plaintext_output(self, fixture_dir, tmp_path):
        output_file = tmp_path / "report.txt"
        result = runner.invoke(app, [
            "analyze",
            str(fixture_dir),
            "-o", "plaintext",
            "-O", str(output_file),
            "--languages", "python",
        ])
        assert result.exit_code == 0

    def test_analyze_languages_python_only(self, fixture_dir, tmp_path):
        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "analyze",
            str(fixture_dir),
            "-o", "json",
            "-O", str(output_file),
            "--languages", "python",
        ])
        assert result.exit_code == 0

    @pytest.mark.skipif(not shutil.which("node"), reason="Node.js not available")
    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "src" / "aibom" / "js_parser" / "node_modules").is_dir(),
        reason="JS parser node_modules not installed (run npm install in js_parser/)",
    )
    def test_analyze_js_fixture_with_relationships(self, fixture_dir, tmp_path):
        ts_file = fixture_dir / "vercel_ai_app.ts"
        if not ts_file.exists():
            pytest.skip("vercel_ai_app.ts fixture missing")

        output_file = tmp_path / "report.json"
        result = runner.invoke(app, [
            "analyze",
            str(ts_file),
            "-o", "json",
            "-O", str(output_file),
            "--languages", "javascript",
            "--completeness",
        ])
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        report = json.loads(output_file.read_text())
        sources = report.get("aibom_analysis", {}).get("sources", {})
        assert len(sources) > 0

        for _source, source_data in sources.items():
            comps = source_data.get("components", {})
            all_categories = set(comps.keys())
            assert "agent" in all_categories, f"Expected agent, got: {all_categories}"
            assert "model" in all_categories or "tool" in all_categories

            rels = source_data.get("relationships", [])
            rel_labels = {r["label"] for r in rels}
            assert "USES_LLM" in rel_labels, f"Expected USES_LLM, got: {rel_labels}"
