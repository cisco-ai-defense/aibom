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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aibom.agentic.agent import _build_context_message, _extract_structured_response
from aibom.models import AIComponent, AIComponentType, ComponentRelationship


@pytest.fixture(autouse=True)
def _isolate_agentic_cache():
    """Prevent on-disk agentic cache from leaking between tests."""
    with patch("aibom.agentic.agent._default_agentic_cache_dir", return_value=None):
        yield


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
        assert "deterministic scan results" in msg.lower()

    def test_full_context_separates_batch_from_others(self):
        batch_comp = AIComponent(
            name="dataset",
            component_type=AIComponentType.DATASET,
            file_path="data.py",
            line_number=5,
        )
        other_comp = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="app.py",
            line_number=10,
            model_name="gpt-4o",
        )
        msg = _build_context_message(
            [batch_comp], [], ["/tmp"],
            all_components=[batch_comp, other_comp],
        )
        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])
        assert len(data["enrich_these"]) == 1
        assert data["enrich_these"][0]["name"] == "dataset"
        assert data["enrich_these"][0]["ENRICH"] is True
        assert len(data["other_detected_components"]) == 1
        assert data["other_detected_components"][0]["name"] == "gpt-4o"
        assert "ENRICH" not in data["other_detected_components"][0]

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
        assert len(data["enrich_these"]) == 1
        assert data["enrich_these"][0]["ENRICH"] is True
        assert data["scan_paths"] == ["/code"]
        assert data["other_detected_components"] == []

    def test_includes_code_context_for_real_file(self, tmp_path):
        src = tmp_path / "example.py"
        src.write_text("import openai\nclient = openai.OpenAI()\nresult = client.chat.completions.create(model='gpt-4o')\n")
        comps = [
            AIComponent(
                name="gpt-4o",
                component_type=AIComponentType.MODEL,
                file_path=str(src),
                line_number=3,
                model_name="gpt-4o",
            )
        ]
        msg = _build_context_message(comps, [], [str(tmp_path)])
        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])
        comp = data["enrich_these"][0]
        assert "code_context" in comp
        assert "import openai" in comp["code_context"]
        assert "model='gpt-4o'" in comp["code_context"]

    def test_no_code_context_for_missing_file(self):
        comps = [
            AIComponent(
                name="x",
                component_type=AIComponentType.MODEL,
                file_path="/nonexistent/path.py",
                line_number=1,
            )
        ]
        msg = _build_context_message(comps, [], ["/tmp"])
        json_start = msg.index("```json\n") + 8
        json_end = msg.index("\n```", json_start)
        data = json.loads(msg[json_start:json_end])
        assert "code_context" not in data["enrich_these"][0]


class TestExtractStructuredResponse:
    def test_prefers_structured_response(self):
        result = {
            "structured_response": {"enriched_components": [], "new_components": []},
            "messages": [MagicMock()],
        }
        data = _extract_structured_response(result)
        assert data == {"enriched_components": [], "new_components": []}

    def test_falls_back_to_json_message(self):
        mock_msg = MagicMock()
        mock_msg.content = '{"enriched_components": []}'
        result = {"messages": [mock_msg]}
        data = _extract_structured_response(result)
        assert data == {"enriched_components": []}

    def test_returns_none_for_empty(self):
        assert _extract_structured_response({"messages": []}) is None
        assert _extract_structured_response({}) is None


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
        from aibom.agentic.agent import run_agentic_enrichment

        mock_create.return_value = MagicMock(
            invoke=MagicMock(side_effect=RuntimeError("LLM unavailable"))
        )
        comp = AIComponent(
            name="test-model",
            component_type=AIComponentType.MODEL,
            file_path="x.py",
            line_number=1,
            model_name="gpt-4o",
        )
        comps, rels, flags = run_agentic_enrichment(
            model_string="bad-model",
            deterministic_components=[comp],
            deterministic_relationships=[],
            scan_paths=["/tmp"],
        )
        assert len(comps) == 1
        assert comps[0].name == "test-model"

    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_batching_splits_large_input(self, mock_create):
        from aibom.agentic.agent import run_agentic_enrichment

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({
            "enriched_components": [],
            "new_components": [],
            "new_relationships": [],
            "risk_findings": [],
        })
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_msg]})
        mock_create.return_value = mock_agent

        comps = [
            AIComponent(
                name=f"model-{i}",
                component_type=AIComponentType.MODEL,
                file_path=f"f{i}.py",
                line_number=i,
                model_name=f"gpt-{i}",
            )
            for i in range(12)
        ]
        result_comps, _, _ = run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=comps,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            batch_size=5,
            max_concurrent=3,
        )
        assert mock_agent.ainvoke.call_count == 3  # 5 + 5 + 2, parallel

    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_single_batch_uses_sequential(self, mock_create):
        """A single batch should use invoke (sequential), not ainvoke."""
        from aibom.agentic.agent import run_agentic_enrichment

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({
            "enriched_components": [],
            "new_components": [],
            "new_relationships": [],
            "risk_findings": [],
        })
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        comps = [
            AIComponent(
                name="gpt-4o",
                component_type=AIComponentType.MODEL,
                file_path="a.py",
                line_number=1,
                model_name="gpt-4o",
            )
        ]
        run_agentic_enrichment(
            model_string="test-model",
            deterministic_components=comps,
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            batch_size=5,
        )
        assert mock_agent.invoke.call_count == 1

    @patch("aibom.agentic.agent.create_aibom_agent")
    def test_tiered_model_uses_fast_for_simple(self, mock_create):
        from aibom.agentic.agent import run_agentic_enrichment

        mock_agent = MagicMock()
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({
            "enriched_components": [],
            "new_components": [],
            "new_relationships": [],
            "risk_findings": [],
        })
        mock_agent.invoke.return_value = {"messages": [mock_msg]}
        mock_create.return_value = mock_agent

        simple = AIComponent(
            name="gpt-4o",
            component_type=AIComponentType.MODEL,
            file_path="a.py",
            line_number=1,
            model_name="gpt-4o",
        )
        complex_ = AIComponent(
            name="some-agent",
            component_type=AIComponentType.AGENT,
            file_path="b.py",
            line_number=5,
        )
        run_agentic_enrichment(
            model_string="expensive-model",
            deterministic_components=[simple, complex_],
            deterministic_relationships=[],
            scan_paths=["/tmp"],
            fast_model="cheap-model",
        )
        calls = mock_create.call_args_list
        assert calls[0][0][0] == "cheap-model"
        assert calls[1][0][0] == "expensive-model"


class TestToolStats:
    def test_tool_stats_isolated_per_reset(self):
        from aibom.agentic.tools import _reset_tool_stats, get_tool_stats, _get_stats_dict

        _reset_tool_stats()
        assert get_tool_stats() == {}
        stats = _get_stats_dict()
        stats["test_tool"] = {"calls": 1, "total_s": 0.5, "errors": 0}
        assert get_tool_stats()["test_tool"]["calls"] == 1

        _reset_tool_stats()
        assert get_tool_stats() == {}

    def test_track_tool_decorator(self):
        from aibom.agentic.tools import _reset_tool_stats, get_tool_stats, _track_tool

        _reset_tool_stats()

        @_track_tool("my_tool")
        def dummy(x):
            return x * 2

        assert dummy(5) == 10
        stats = get_tool_stats()
        assert "my_tool" in stats
        assert stats["my_tool"]["calls"] == 1
        assert stats["my_tool"]["errors"] == 0


class TestLocalityAwareBatching:
    """Locality-aware batching groups co-located components."""

    def test_groups_by_directory(self):
        from aibom.agentic.agent import _locality_aware_batches

        comps = [
            AIComponent(name="a", component_type=AIComponentType.MODEL, file_path="/repo/dir1/a.py", line_number=1),
            AIComponent(name="b", component_type=AIComponentType.MODEL, file_path="/repo/dir1/b.py", line_number=2),
            AIComponent(name="c", component_type=AIComponentType.MODEL, file_path="/repo/dir2/c.py", line_number=1),
            AIComponent(name="d", component_type=AIComponentType.MODEL, file_path="/repo/dir2/d.py", line_number=2),
        ]
        batches = _locality_aware_batches(comps, batch_size=3)
        assert len(batches) == 2
        dirs_b0 = {str(Path(c.file_path).parent) for c in batches[0]}
        assert len(dirs_b0) <= 2

    def test_single_dir_stays_together(self):
        from aibom.agentic.agent import _locality_aware_batches

        comps = [
            AIComponent(name=f"m{i}", component_type=AIComponentType.MODEL, file_path=f"/repo/pkg/{i}.py", line_number=i)
            for i in range(4)
        ]
        batches = _locality_aware_batches(comps, batch_size=5)
        assert len(batches) == 1
        assert len(batches[0]) == 4

    def test_respects_batch_size(self):
        from aibom.agentic.agent import _locality_aware_batches

        comps = [
            AIComponent(name=f"m{i}", component_type=AIComponentType.MODEL, file_path=f"/repo/pkg/{i}.py", line_number=i)
            for i in range(7)
        ]
        batches = _locality_aware_batches(comps, batch_size=3)
        assert all(len(b) <= 3 for b in batches)
        assert sum(len(b) for b in batches) == 7


class TestAgenticResultCache:
    """Content-hash result cache for agentic enrichment."""

    def test_cache_miss_then_hit(self, tmp_path):
        from aibom.agentic.agent import _AgenticResultCache, _component_cache_key

        cache = _AgenticResultCache(tmp_path / "cache")
        comp = AIComponent(
            name="gpt-4o", component_type=AIComponentType.MODEL,
            file_path="a.py", line_number=1, model_name="gpt-4o",
        )
        key = _component_cache_key(comp)
        assert cache.get(key) is None

        cache.put(key, {"enriched_components": [], "new_components": []})
        assert cache.get(key) is not None

    def test_partition_splits_cached_and_uncached(self, tmp_path):
        from aibom.agentic.agent import _AgenticResultCache, _component_cache_key

        cache = _AgenticResultCache(tmp_path / "cache")
        c1 = AIComponent(name="a", component_type=AIComponentType.MODEL, file_path="a.py", line_number=1, model_name="gpt-4o")
        c2 = AIComponent(name="b", component_type=AIComponentType.MODEL, file_path="b.py", line_number=1, model_name="gpt-5")
        cache.put(_component_cache_key(c1), {"enriched_components": []})

        cached, uncached = cache.partition([c1, c2])
        assert len(cached) == 1
        assert cached[0].name == "a"
        assert len(uncached) == 1
        assert uncached[0].name == "b"

    def test_disk_persistence(self, tmp_path):
        from aibom.agentic.agent import _AgenticResultCache, _component_cache_key

        cache_dir = tmp_path / "cache"
        cache1 = _AgenticResultCache(cache_dir)
        comp = AIComponent(name="x", component_type=AIComponentType.MODEL, file_path="x.py", line_number=1)
        key = _component_cache_key(comp)
        cache1.put(key, {"enriched_components": [], "test": True})

        cache2 = _AgenticResultCache(cache_dir)
        assert cache2.get(key) is not None
        assert cache2.get(key)["test"] is True


class TestSubAgentGrouping:
    """Sub-agent dispatch groups components by scan root."""

    def test_groups_by_scan_root(self):
        from aibom.agentic.agent import _group_by_top_dir

        comps = [
            AIComponent(name="a", component_type=AIComponentType.MODEL, file_path="/repo1/src/a.py", line_number=1),
            AIComponent(name="b", component_type=AIComponentType.MODEL, file_path="/repo1/src/b.py", line_number=2),
            AIComponent(name="c", component_type=AIComponentType.MODEL, file_path="/repo2/lib/c.py", line_number=1),
        ]
        groups = _group_by_top_dir(comps, scan_paths=["/repo1", "/repo2"])
        assert len(groups) == 2

    def test_single_root_single_group(self):
        from aibom.agentic.agent import _group_by_top_dir

        comps = [
            AIComponent(name=f"m{i}", component_type=AIComponentType.MODEL, file_path=f"/repo/d{i}/f.py", line_number=i)
            for i in range(5)
        ]
        groups = _group_by_top_dir(comps, scan_paths=["/repo"])
        assert len(groups) == 1
