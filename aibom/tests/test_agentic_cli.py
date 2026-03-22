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

"""CLI integration tests for agentic enrichment via --llm-model."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from aibom.cli import app

runner = CliRunner()


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(
        'from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n'
    )
    return tmp_path


class TestAgenticEnrichmentViaCLI:
    def test_without_llm_model_runs_deterministic_only(self, sample_dir, tmp_path):
        out = tmp_path / "report.txt"
        result = runner.invoke(
            app,
            [
                "analyze", str(sample_dir),
                "--output-format", "plaintext",
                "--output-file", str(out),
            ],
        )
        assert result.exit_code == 0
        assert "Enriching" not in result.output

    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_llm_model_triggers_agentic_enrichment(self, mock_create, sample_dir, tmp_path):
        out = tmp_path / "report.txt"
        agent_response = json.dumps({
            "enriched_components": [],
            "new_components": [],
            "new_relationships": [
                {
                    "source_name": "client",
                    "target_name": "gpt-4o",
                    "relationship_type": "USES_MODEL",
                }
            ],
            "risk_findings": [],
        })
        mock_msg = MagicMock()
        mock_msg.content = agent_response
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        result = runner.invoke(
            app,
            [
                "analyze", str(sample_dir),
                "--output-format", "plaintext",
                "--output-file", str(out),
                "--llm-model", "test-model",
                "--llm-api-base", "http://localhost:11434",
            ],
        )
        assert result.exit_code == 0
        mock_create.assert_called_once()

    def test_llm_model_with_legacy_mode_skips_agentic(self, sample_dir, tmp_path):
        import duckdb

        db_file = tmp_path / "test_catalog.duckdb"
        con = duckdb.connect(str(db_file))
        con.execute(
            "CREATE TABLE component_catalog ("
            "id TEXT, label TEXT, concept TEXT, framework TEXT, "
            "sig_name TEXT, type TEXT, catalog_label TEXT)"
        )
        con.close()

        out = tmp_path / "report.txt"
        with patch(
            "aibom.cli.ensure_local_database", return_value=db_file
        ):
            result = runner.invoke(
                app,
                [
                    "analyze", str(sample_dir),
                    "--output-format", "plaintext",
                    "--output-file", str(out),
                    "--legacy-mode",
                    "--llm-model", "test-model",
                    "--llm-api-base", "http://localhost:11434",
                ],
            )
        assert result.exit_code == 0

    def test_llm_model_help_mentions_agentic(self):
        import re

        result = runner.invoke(app, ["analyze", "--help"])
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--llm-model" in clean
        assert "agentic" in clean
