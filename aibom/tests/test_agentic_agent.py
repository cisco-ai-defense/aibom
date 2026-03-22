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

"""Tests for agent factory and enrichment pipeline.

These tests do NOT require deepagents/langchain to be installed --
they test the helper functions and mock the external dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aibom.agentic.agent import _build_context_message, _extract_final_message
from aibom.models import AIComponent, AIComponentType, ComponentRelationship


class TestBuildContextMessage:
    def test_includes_components_and_paths(self):
        comps = [
            AIComponent(
                name="gpt-4o",
                component_type=AIComponentType.MODEL,
                file_path="app.py",
                line_number=10,
                model_name="gpt-4o",
            )
        ]
        msg = _build_context_message(comps, [], ["/tmp/repo"])
        assert "gpt-4o" in msg
        assert "/tmp/repo" in msg
        assert "deterministic scan results" in msg

    def test_json_is_parseable(self):
        comps = [
            AIComponent(
                name="test",
                component_type=AIComponentType.TOOL,
                file_path="t.py",
                line_number=1,
            )
        ]
        msg = _build_context_message(comps, [], ["/code"])
        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])
        assert data["total_components"] == 1
        assert data["scan_paths"] == ["/code"]


class TestExtractFinalMessage:
    def test_extracts_from_message_objects(self):
        mock_msg = MagicMock()
        mock_msg.content = "final answer"
        result = {"messages": [MagicMock(), mock_msg]}
        assert _extract_final_message(result) == "final answer"

    def test_extracts_from_dict_messages(self):
        result = {"messages": [{"content": "hello"}, {"content": "final"}]}
        assert _extract_final_message(result) == "final"

    def test_returns_none_for_empty(self):
        assert _extract_final_message({"messages": []}) is None
        assert _extract_final_message({}) is None


class TestLazyImport:
    def test_aibom_import_does_not_import_deepagents(self):
        """Importing aibom.agentic should NOT trigger deepagents import."""
        import sys
        import importlib

        mods_before = set(sys.modules.keys())
        importlib.import_module("aibom.agentic")
        mods_after = set(sys.modules.keys())
        new_mods = mods_after - mods_before
        deepagent_mods = [m for m in new_mods if "deepagent" in m.lower()]
        assert deepagent_mods == [], (
            f"Importing aibom.agentic pulled in deepagents: {deepagent_mods}"
        )

    def test_aibom_import_does_not_import_langchain(self):
        """Importing aibom.agentic should NOT trigger langchain import."""
        import sys
        import importlib

        mods_before = set(sys.modules.keys())
        importlib.import_module("aibom.agentic")
        mods_after = set(sys.modules.keys())
        new_mods = mods_after - mods_before
        langchain_mods = [m for m in new_mods if "langchain" in m.lower()]
        assert langchain_mods == [], (
            f"Importing aibom.agentic pulled in langchain: {langchain_mods}"
        )


class TestRunAgenticEnrichment:
    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_merges_agent_output_into_components(self, mock_create):
        from aibom.agentic.agent import run_agentic_enrichment

        agent_response = json.dumps({
            "enriched_components": [],
            "new_components": [
                {
                    "name": "agent-found-model",
                    "component_type": "model",
                    "file_path": "new.py",
                    "line_number": 1,
                    "framework": "openai",
                    "model_name": "gpt-5",
                }
            ],
            "new_relationships": [],
            "risk_findings": [],
        })

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = agent_response
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        det_comps = [
            AIComponent(
                name="existing",
                component_type=AIComponentType.DEPENDENCY,
                file_path="app.py",
                line_number=1,
            )
        ]
        comps, rels, flags = run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=det_comps,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
        )
        assert len(comps) == 2
        names = {c.name for c in comps}
        assert "existing" in names
        assert "agent-found-model" in names

    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_handles_agent_failure_gracefully(self, mock_create):
        from aibom.agentic.agent import AgenticEnrichmentError, run_agentic_enrichment

        mock_create.return_value = MagicMock(
            invoke=MagicMock(side_effect=RuntimeError("LLM unavailable"))
        )
        with pytest.raises(AgenticEnrichmentError, match="LLM unavailable"):
            run_agentic_enrichment(
                model_string="bad-model",
                deterministic_components=[],
                deterministic_relationships=[],
                scan_paths=["/tmp"],
            )
