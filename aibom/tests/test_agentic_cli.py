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

import importlib
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
    def test_without_llm_model_exits_with_error(self, sample_dir, tmp_path):
        """--llm-model is mandatory; omitting it should fail with a clear message."""
        out = tmp_path / "report.txt"
        result = runner.invoke(
            app,
            [
                "analyze",
                str(sample_dir),
                "--output-format",
                "plaintext",
                "--output-file",
                str(out),
            ],
            env={"AIBOM_LLM_MODEL": ""},
        )
        assert result.exit_code == 1
        assert (
            "llm-model" in result.output.lower() or "AIBOM_LLM_MODEL" in result.output
        )

    @patch("aibom.scan_pipeline.ensure_llm_runtime_available", return_value=None)
    @patch("aibom.cli.ensure_llm_runtime_available", return_value=None)
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_llm_model_triggers_agentic_enrichment(
        self,
        mock_create,
        _mock_build,
        _mock_close,
        _mock_cli_preflight,
        _mock_pipeline_preflight,
        sample_dir,
        tmp_path,
    ):
        out = tmp_path / "report.txt"
        agent_response = json.dumps(
            {
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
            }
        )
        mock_msg = MagicMock()
        mock_msg.content = agent_response
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        result = runner.invoke(
            app,
            [
                "analyze",
                str(sample_dir),
                "--output-format",
                "plaintext",
                "--output-file",
                str(out),
                "--llm-model",
                "test-model",
                "--llm-api-base",
                "http://localhost:11434",
                "--agentic-scope",
                "all",
            ],
        )
        assert result.exit_code == 0
        assert mock_create.call_count >= 1

    def test_llm_model_help_mentions_agentic(self):
        import re

        result = runner.invoke(app, ["analyze", "--help"])
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--llm-model" in clean
        assert "agentic" in clean

    def test_llm_max_tokens_help_listed(self):
        import re

        result = runner.invoke(app, ["analyze", "--help"])
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--llm-max-tokens" in clean

    @patch("aibom.scan_pipeline.ensure_llm_runtime_available", return_value=None)
    @patch("aibom.cli.ensure_llm_runtime_available", return_value=None)
    @patch("aibom.agentic.agent._close_model_clients")
    @patch("aibom.agentic.agent._build_model", return_value=MagicMock())
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_llm_max_tokens_flag_threaded_into_config(
        self,
        mock_create,
        mock_build,
        _mock_close,
        _mock_cli_preflight,
        _mock_pipeline_preflight,
        sample_dir,
        tmp_path,
    ):
        out = tmp_path / "report.txt"
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({"enriched_components": [], "new_components": []})
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        result = runner.invoke(
            app,
            [
                "analyze",
                str(sample_dir),
                "--output-format",
                "plaintext",
                "--output-file",
                str(out),
                "--llm-model",
                "test-model",
                "--llm-api-base",
                "http://localhost:11434",
                "--llm-max-tokens",
                "99999",
                "--agentic-scope",
                "all",
            ],
        )
        assert result.exit_code == 0
        # The --llm-max-tokens value must reach _build_model via llm_config.
        found = False
        for call in mock_build.call_args_list:
            for arg in list(call.args) + list(call.kwargs.values()):
                if isinstance(arg, dict) and arg.get("max_tokens") == 99999:
                    found = True
        assert found, "llm_config['max_tokens']=99999 not passed to _build_model"

    def test_llm_max_tokens_rejects_non_positive(self, sample_dir, tmp_path):
        import re

        result = runner.invoke(
            app,
            [
                "analyze",
                str(sample_dir),
                "--llm-model",
                "test-model",
                "--llm-max-tokens",
                "0",
            ],
        )
        clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert result.exit_code != 0
        assert "max-tokens" in clean.lower() or "range" in clean.lower()

    @patch("aibom.repo_triage.RepoTriager")
    def test_triage_config_includes_max_tokens(self, mock_triager_cls):
        from aibom.cli import _gather_analysis_sources

        inst = MagicMock()
        inst.triage_repos.return_value = []
        mock_triager_cls.return_value = inst

        _gather_analysis_sources(
            sources=["repoA", "repoB"],
            images_file=None,
            repos_file=None,
            discover_repos=False,
            github_org=None,
            gitlab_group=None,
            bitbucket_project=None,
            platform_token=None,
            repo_name_filter=None,
            repo_topic_filter=None,
            max_repos=None,
            parallel_repos=1,
            llm_model="test-model",
            llm_provider=None,
            llm_api_base=None,
            llm_api_key=None,
            llm_api_version=None,
            llm_max_tokens=4242,
        )
        _, kwargs = mock_triager_cls.call_args
        assert kwargs["llm_config"].get("max_tokens") == 4242

    def test_missing_agentic_extras_fail_fast_with_install_hint(
        self, monkeypatch, sample_dir, tmp_path
    ):
        out = tmp_path / "report.txt"

        monkeypatch.setattr("aibom.llm_factory.init_chat_model", None)

        result = runner.invoke(
            app,
            [
                "analyze",
                str(sample_dir),
                "--output-format",
                "plaintext",
                "--output-file",
                str(out),
                "--llm-model",
                "test-model",
                "--llm-api-base",
                "http://localhost:11434",
            ],
        )

        assert result.exit_code == 1
        assert "cisco-aibom[agentic]" in result.output
        assert "install" in result.output.lower()

    def test_missing_openai_provider_extra_fails_fast_with_install_hint(
        self, monkeypatch, sample_dir, tmp_path
    ):
        out = tmp_path / "report.txt"

        monkeypatch.setattr("aibom.llm_factory.init_chat_model", lambda *a, **k: None)

        real_import_module = importlib.import_module

        def fake_import_module(name: str, package: str | None = None):
            if name == "langchain_openai":
                raise ImportError("missing langchain_openai")
            return real_import_module(name, package)

        monkeypatch.setattr("importlib.import_module", fake_import_module)

        result = runner.invoke(
            app,
            [
                "analyze",
                str(sample_dir),
                "--output-format",
                "plaintext",
                "--output-file",
                str(out),
                "--llm-model",
                "gpt-5.4",
                "--llm-provider",
                "openai",
                "--llm-api-key",
                "not-a-real-key",
            ],
        )

        assert result.exit_code == 1
        assert "llm-openai" in result.output
        assert "cisco-aibom[agentic,llm-openai]" in result.output
