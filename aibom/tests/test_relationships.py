# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for relationship derivation in the categorizer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aibom.categorizer import (
    _derive_relationships,
    _build_instance_id,
    _resolve_component_reference,
    TOOL_ARGUMENT_HINTS,
    LLM_ARGUMENT_HINTS,
    MEMORY_ARGUMENT_HINTS,
    PROMPT_ARGUMENT_HINTS,
    EMBEDDING_ARGUMENT_HINTS,
)
from aibom.structures import ComponentRelationship


def _make_component(name, instance_id, category, file_path="/test/file.py", line=1):
    return {
        "name": name,
        "instance_id": instance_id,
        "category": category,
        "file_path": file_path,
        "line_number": line,
    }


def _make_metadata(instance_id, arguments=None, file_path="/test/file.py"):
    return {
        "arguments": arguments or {},
        "file_path": file_path,
    }


class TestBuildInstanceId:
    def test_includes_file_path(self):
        comp = {"name": "ChatOpenAI", "line_number": 5, "file_path": "/test/file.py"}
        iid = _build_instance_id(comp)
        assert "/test/file.py" in iid
        assert "ChatOpenAI" in iid
        assert "5" in iid

    def test_different_files_get_different_ids(self):
        comp1 = {"name": "ChatOpenAI", "line_number": 5, "file_path": "/test/a.py"}
        comp2 = {"name": "ChatOpenAI", "line_number": 5, "file_path": "/test/b.py"}
        assert _build_instance_id(comp1) != _build_instance_id(comp2)

    def test_defaults_for_missing_fields(self):
        comp = {}
        iid = _build_instance_id(comp)
        assert "component" in iid
        assert "0" in iid


class TestDeriveRelationships:
    def test_agent_uses_llm(self):
        agent = _make_component("Agent", "agent_1", "agent")
        model = _make_component("LLM", "model_1", "model")
        components = {"agent": [agent], "model": [model]}
        metadata = {"agent_1": _make_metadata("agent_1", {"llm": "VARIABLE:model_1"})}
        lookup_var = {("/test/file.py", "model_1"): model}
        lookup_name = {"LLM": [model]}

        rels = _derive_relationships(
            components, metadata, lookup_var, lookup_name,
        )
        assert len(rels) >= 1
        assert any(r.label == "USES_LLM" for r in rels)

    def test_agent_uses_tool(self):
        agent = _make_component("Agent", "agent_1", "agent")
        tool = _make_component("Search", "tool_1", "tool")
        components = {"agent": [agent], "tool": [tool]}
        metadata = {"agent_1": _make_metadata("agent_1", {"tools": "VARIABLE:tool_1"})}
        lookup_var = {("/test/file.py", "tool_1"): tool}
        lookup_name = {"Search": [tool]}

        rels = _derive_relationships(
            components, metadata, lookup_var, lookup_name,
        )
        assert any(r.label == "USES_TOOL" for r in rels)

    def test_agent_uses_memory(self):
        agent = _make_component("Agent", "agent_1", "agent")
        mem = _make_component("MemorySaver", "mem_1", "memory")
        components = {"agent": [agent], "memory": [mem]}
        metadata = {"agent_1": _make_metadata("agent_1", {"checkpointer": "VARIABLE:mem_1"})}
        lookup_var = {("/test/file.py", "mem_1"): mem}
        lookup_name = {"MemorySaver": [mem]}

        rels = _derive_relationships(
            components, metadata, lookup_var, lookup_name,
        )
        assert any(r.label == "USES_MEMORY" for r in rels)

    def test_agent_uses_prompt(self):
        agent = _make_component("Agent", "agent_1", "agent")
        prompt = _make_component("MyPrompt", "prompt_1", "prompt")
        components = {"agent": [agent], "prompt": [prompt]}
        metadata = {"agent_1": _make_metadata("agent_1", {"prompt": "VARIABLE:prompt_1"})}
        lookup_var = {("/test/file.py", "prompt_1"): prompt}
        lookup_name = {"MyPrompt": [prompt]}

        rels = _derive_relationships(
            components, metadata, lookup_var, lookup_name,
        )
        assert any(r.label == "USES_PROMPT" for r in rels)

    def test_no_duplicate_relationships(self):
        agent = _make_component("Agent", "agent_1", "agent")
        model = _make_component("LLM", "model_1", "model")
        components = {"agent": [agent], "model": [model]}
        metadata = {
            "agent_1": _make_metadata("agent_1", {
                "llm": "VARIABLE:model_1",
                "model": "VARIABLE:model_1",
            }),
        }
        lookup_var = {("/test/file.py", "model_1"): model}
        lookup_name = {"LLM": [model]}

        rels = _derive_relationships(
            components, metadata, lookup_var, lookup_name,
        )
        uses_llm = [r for r in rels if r.label == "USES_LLM"]
        assert len(uses_llm) <= 1

    # ── File-level co-occurrence fallback tests ──────────────────────

    def test_file_cooccurrence_agent_with_model_same_file(self):
        agent = _make_component("StateGraph", "agent_1", "agent", file_path="/app/graph.ts")
        model = _make_component("ChatOpenAI", "model_1", "model", file_path="/app/graph.ts")
        components = {"agent": [agent], "model": [model]}
        metadata = {"agent_1": _make_metadata("agent_1", {}, file_path="/app/graph.ts")}
        lookup_var = {}
        lookup_name = {}

        rels = _derive_relationships(components, metadata, lookup_var, lookup_name)
        assert any(r.label == "USES_LLM" and r.source_instance_id == "agent_1" for r in rels)

    def test_file_cooccurrence_agent_with_tool_same_file(self):
        agent = _make_component("StateGraph", "agent_1", "agent", file_path="/app/graph.ts")
        tool = _make_component("ToolNode", "tool_1", "tool", file_path="/app/graph.ts")
        components = {"agent": [agent], "tool": [tool]}
        metadata = {"agent_1": _make_metadata("agent_1", {}, file_path="/app/graph.ts")}
        lookup_var = {}
        lookup_name = {}

        rels = _derive_relationships(components, metadata, lookup_var, lookup_name)
        assert any(r.label == "USES_TOOL" and r.source_instance_id == "agent_1" for r in rels)

    def test_file_cooccurrence_skipped_when_arg_hints_exist(self):
        agent = _make_component("Agent", "agent_1", "agent")
        model = _make_component("LLM", "model_1", "model")
        tool = _make_component("Search", "tool_1", "tool")
        components = {"agent": [agent], "model": [model], "tool": [tool]}
        metadata = {"agent_1": _make_metadata("agent_1", {"llm": "VARIABLE:model_1"})}
        lookup_var = {("/test/file.py", "model_1"): model}
        lookup_name = {"LLM": [model]}

        rels = _derive_relationships(components, metadata, lookup_var, lookup_name)
        uses_tool = [r for r in rels if r.label == "USES_TOOL"]
        assert len(uses_tool) == 0

    def test_file_cooccurrence_skipped_when_no_file_path(self):
        agent = _make_component("Agent", "agent_1", "agent", file_path=None)
        model = _make_component("LLM", "model_1", "model", file_path=None)
        components = {"agent": [agent], "model": [model]}
        metadata = {"agent_1": _make_metadata("agent_1", {})}
        lookup_var = {}
        lookup_name = {}

        rels = _derive_relationships(components, metadata, lookup_var, lookup_name)
        assert len(rels) == 0


class TestResolveComponentReference:
    def test_single_candidate(self):
        comp = _make_component("LLM", "model_1", "model")
        lookup_name = {"LLM": [comp]}
        result = _resolve_component_reference("LLM", "/test/file.py", {}, lookup_name)
        assert result is comp

    def test_multiple_candidates_same_file_preferred(self):
        comp_a = _make_component("openai", "m1", "model", file_path="/a.py")
        comp_b = _make_component("openai", "m2", "model", file_path="/b.py")
        lookup_name = {"openai": [comp_a, comp_b]}
        result = _resolve_component_reference("openai", "/b.py", {}, lookup_name)
        assert result is comp_b

    def test_multiple_candidates_no_file_returns_first(self):
        comp_a = _make_component("openai", "m1", "model", file_path="/a.py")
        comp_b = _make_component("openai", "m2", "model", file_path="/b.py")
        lookup_name = {"openai": [comp_a, comp_b]}
        result = _resolve_component_reference("openai", None, {}, lookup_name)
        assert result is comp_a

    def test_no_candidates(self):
        result = _resolve_component_reference("unknown", "/test/file.py", {}, {})
        assert result is None

    def test_var_lookup_takes_priority(self):
        comp_var = _make_component("llm", "m_var", "model")
        comp_name = _make_component("llm", "m_name", "model")
        lookup_var = {("/test/file.py", "llm"): comp_var}
        lookup_name = {"llm": [comp_name]}
        result = _resolve_component_reference("llm", "/test/file.py", lookup_var, lookup_name)
        assert result is comp_var
