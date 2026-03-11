# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests verifying that catalog entries for each framework are matched correctly."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aibom.categorizer import categorize_symbols
from aibom.structures import (
    AssignmentObservation,
    CallObservation,
    CodeAnalysisResult,
    DecoratorObservation,
)


def _make_connector(entries):
    """Create a mock CatalogDB returning *entries* for any suffix query."""
    mock = MagicMock()
    mock.find_components_by_suffixes.return_value = entries
    mock.is_excluded.return_value = False
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


def _analyze_single(qualified_name, entries, target_name="obj", imports=None):
    """Helper to run categorize_symbols with a single assignment observation."""
    result = CodeAnalysisResult(
        file_path="/test/file.py",
        assignments=[
            AssignmentObservation(
                target_qualified_name=target_name,
                call=CallObservation(
                    qualified_name=qualified_name,
                    arguments={},
                    line_number=1,
                ),
                line_number=1,
            ),
        ],
        imports=imports or [f"from {qualified_name.rsplit('.', 1)[0]} import {qualified_name.rsplit('.', 1)[-1]}"],
    )
    connector = _make_connector(entries)
    output = categorize_symbols([result], connector)
    return output


class TestCrewAIDetection:
    def test_crewai_agent(self):
        entries = [{"id": "crewai.Agent", "concept": "agent", "label": "Agent"}]
        output = _analyze_single("crewai.Agent", entries)
        assert "agent" in output.components
        assert len(output.components["agent"]) > 0

    def test_crewai_crew(self):
        entries = [{"id": "crewai.Crew", "concept": "agent", "label": "Crew"}]
        output = _analyze_single("crewai.Crew", entries)
        assert "agent" in output.components


class TestAutoGenDetection:
    def test_autogen_conversable_agent(self):
        entries = [{"id": "autogen.ConversableAgent", "concept": "agent", "label": "ConversableAgent"}]
        output = _analyze_single("autogen.ConversableAgent", entries)
        assert "agent" in output.components


class TestDSPyDetection:
    def test_dspy_module(self):
        entries = [{"id": "dspy.Module", "concept": "agent", "label": "Module"}]
        output = _analyze_single("dspy.Module", entries)
        assert "agent" in output.components

    def test_dspy_retrieve(self):
        entries = [{"id": "dspy.Retrieve", "concept": "retriever", "label": "Retrieve"}]
        output = _analyze_single("dspy.Retrieve", entries)
        assert "retriever" in output.components


class TestHaystackDetection:
    def test_haystack_pipeline(self):
        entries = [{"id": "haystack.Pipeline", "concept": "agent", "label": "Pipeline"}]
        output = _analyze_single("haystack.Pipeline", entries)
        assert "agent" in output.components

    def test_haystack_openai_generator(self):
        entries = [{"id": "haystack.components.generators.openai.OpenAIGenerator", "concept": "model", "label": "OpenAIGenerator"}]
        output = _analyze_single("haystack.components.generators.openai.OpenAIGenerator", entries)
        assert "model" in output.components


class TestLlamaIndexDetection:
    def test_llamaindex_vector_store(self):
        entries = [{"id": "llama_index.core.VectorStoreIndex", "concept": "datastore", "label": "VectorStoreIndex"}]
        output = _analyze_single("llama_index.core.VectorStoreIndex", entries)
        assert "datastore" in output.components

    def test_llamaindex_openai_agent(self):
        entries = [{"id": "llama_index.agent.openai.OpenAIAgent", "concept": "agent", "label": "OpenAIAgent"}]
        output = _analyze_single("llama_index.agent.openai.OpenAIAgent", entries)
        assert "agent" in output.components


class TestSemanticKernelDetection:
    def test_kernel(self):
        entries = [{"id": "semantic_kernel.Kernel", "concept": "agent", "label": "Kernel"}]
        output = _analyze_single("semantic_kernel.Kernel", entries)
        assert "agent" in output.components


class TestSmolagentsDetection:
    def test_code_agent(self):
        entries = [{"id": "smolagents.CodeAgent", "concept": "agent", "label": "CodeAgent"}]
        output = _analyze_single("smolagents.CodeAgent", entries)
        assert "agent" in output.components

    def test_tool_calling_agent(self):
        entries = [{"id": "smolagents.ToolCallingAgent", "concept": "agent", "label": "ToolCallingAgent"}]
        output = _analyze_single("smolagents.ToolCallingAgent", entries)
        assert "agent" in output.components


class TestGoogleGenAIDetection:
    def test_generative_model(self):
        entries = [{"id": "google.generativeai.GenerativeModel", "concept": "model", "label": "GenerativeModel"}]
        output = _analyze_single("google.generativeai.GenerativeModel", entries)
        assert "model" in output.components

    def test_chat_session(self):
        entries = [{"id": "google.generativeai.ChatSession", "concept": "agent", "label": "ChatSession"}]
        output = _analyze_single("google.generativeai.ChatSession", entries)
        assert "agent" in output.components
