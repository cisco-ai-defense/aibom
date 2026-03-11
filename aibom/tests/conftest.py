# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the AIBOM test suite."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest

from aibom.structures import (
    AssignmentObservation,
    CallObservation,
    ClassDefObservation,
    CodeAnalysisResult,
    CategorizationOutput,
    ComponentRelationship,
    DecoratorObservation,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_TEST_CATALOG_ENTRIES = [
    ("langchain_openai.ChatOpenAI", "ChatOpenAI", "model", "langchain", "ChatOpenAI", "class", "LangChain"),
    ("langgraph.graph.StateGraph", "StateGraph", "agent", "langgraph", "StateGraph", "class", "LangGraph"),
    ("langchain_core.tools.tool", "tool", "tool", "langchain", "tool", "decorator", "LangChain"),
    ("langchain_core.prompts.ChatPromptTemplate", "ChatPromptTemplate", "prompt", "langchain", "ChatPromptTemplate", "class", "LangChain"),
    ("langgraph.checkpoint.memory.MemorySaver", "MemorySaver", "memory", "langgraph", "MemorySaver", "class", "LangGraph"),
    ("langchain_openai.OpenAIEmbeddings", "OpenAIEmbeddings", "embedding", "langchain", "OpenAIEmbeddings", "class", "LangChain"),
    ("crewai.Agent", "Agent", "agent", "crewai", "Agent", "class", "CrewAI"),
    ("crewai.Task", "Task", "tool", "crewai", "Task", "class", "CrewAI"),
    ("crewai.Crew", "Crew", "agent", "crewai", "Crew", "class", "CrewAI"),
    # Vercel AI SDK (JS/TS)
    ("ai.generateText", "generateText", "agent", "vercel-ai", "generateText", "function", "Vercel AI SDK"),
    ("ai.streamText", "streamText", "agent", "vercel-ai", "streamText", "function", "Vercel AI SDK"),
    ("ai.generateObject", "generateObject", "agent", "vercel-ai", "generateObject", "function", "Vercel AI SDK"),
    ("@ai-sdk/openai.openai", "openai", "model", "vercel-ai", "openai", "function", "Vercel AI SDK"),
    # LangChain.js
    ("@langchain/openai.ChatOpenAI", "ChatOpenAI", "model", "langchain-js", "ChatOpenAI", "class", "LangChain.js"),
    ("@langchain/langgraph.StateGraph", "StateGraph", "agent", "langgraph-js", "StateGraph", "class", "LangGraph.js"),
]


@pytest.fixture(scope="session")
def test_catalog_db_path() -> Path:
    """Build a minimal DuckDB catalog for CLI integration tests.

    Session-scoped so the file is created once and shared across all tests.
    """
    tmp_dir = tempfile.mkdtemp(prefix="aibom_test_catalog_")
    db_path = Path(tmp_dir) / "test_catalog.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE component_catalog (
            id TEXT PRIMARY KEY,
            label TEXT,
            concept TEXT,
            framework TEXT,
            sig_name TEXT,
            type TEXT,
            catalog_label TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO component_catalog VALUES (?, ?, ?, ?, ?, ?, ?)",
        _TEST_CATALOG_ENTRIES,
    )
    conn.close()
    return db_path


@pytest.fixture
def fixture_dir() -> Path:
    """Path to the tests/fixtures/ directory."""
    return FIXTURES_DIR


@pytest.fixture
def sample_python_analysis() -> CodeAnalysisResult:
    """A CodeAnalysisResult from a realistic Python AI file."""
    return CodeAnalysisResult(
        file_path="/test/agent.py",
        assignments=[
            AssignmentObservation(
                target_qualified_name="llm",
                call=CallObservation(
                    qualified_name="langchain_openai.ChatOpenAI",
                    arguments={"model": "gpt-4o-mini"},
                    line_number=5,
                    raw_code='ChatOpenAI(model="gpt-4o-mini")',
                ),
                line_number=5,
            ),
            AssignmentObservation(
                target_qualified_name="graph",
                call=CallObservation(
                    qualified_name="langgraph.graph.StateGraph",
                    arguments={},
                    line_number=10,
                    raw_code="StateGraph(dict)",
                ),
                line_number=10,
            ),
        ],
        decorators=[
            DecoratorObservation(
                decorator_qualified_name="langchain_core.tools.tool",
                decorated_function_name="search",
                line_number=15,
            ),
        ],
        imports=[
            "from langchain_openai import ChatOpenAI",
            "from langgraph.graph import StateGraph",
            "from langchain_core.tools import tool",
        ],
    )


@pytest.fixture
def sample_js_analysis() -> CodeAnalysisResult:
    """A CodeAnalysisResult from a realistic JS/TS AI file."""
    return CodeAnalysisResult(
        file_path="/test/agent.ts",
        assignments=[
            AssignmentObservation(
                target_qualified_name="llm",
                call=CallObservation(
                    qualified_name="@langchain/openai.ChatOpenAI",
                    arguments={"model": "gpt-4o-mini"},
                    line_number=5,
                    raw_code='new ChatOpenAI({ model: "gpt-4o-mini" })',
                ),
                line_number=5,
            ),
            AssignmentObservation(
                target_qualified_name="graph",
                call=CallObservation(
                    qualified_name="@langchain/langgraph.StateGraph",
                    arguments={},
                    line_number=10,
                    raw_code="new StateGraph({})",
                ),
                line_number=10,
            ),
        ],
        imports=[
            'from @langchain/openai import ChatOpenAI',
            'from @langchain/langgraph import StateGraph',
        ],
    )


@pytest.fixture
def catalog_db(tmp_path):
    """A mock CatalogDB."""
    mock = MagicMock()
    mock.find_components_by_suffixes.return_value = [
        {"id": "langchain_openai.ChatOpenAI", "concept": "model", "label": "ChatOpenAI"},
        {"id": "langgraph.graph.StateGraph", "concept": "agent", "label": "StateGraph"},
        {"id": "langchain_core.tools.tool", "concept": "tool", "label": "tool"},
        {"id": "langchain_core.prompts.ChatPromptTemplate", "concept": "prompt", "label": "ChatPromptTemplate"},
        {"id": "langgraph.checkpoint.memory.MemorySaver", "concept": "memory", "label": "MemorySaver"},
        {"id": "langchain_openai.OpenAIEmbeddings", "concept": "embedding", "label": "OpenAIEmbeddings"},
    ]
    mock.is_excluded.return_value = False
    return mock


@pytest.fixture
def sample_categorization_output() -> CategorizationOutput:
    """A CategorizationOutput with agents, models, tools, and relationships."""
    return CategorizationOutput(
        components={
            "agent": [
                {
                    "name": "StateGraph",
                    "instance_id": "agent.py:StateGraph_10",
                    "file_path": "/test/agent.py",
                    "line_number": 10,
                    "category": "agent",
                }
            ],
            "model": [
                {
                    "name": "ChatOpenAI",
                    "instance_id": "agent.py:ChatOpenAI_5",
                    "file_path": "/test/agent.py",
                    "line_number": 5,
                    "category": "model",
                    "model_name": "gpt-4o-mini",
                }
            ],
            "tool": [
                {
                    "name": "search",
                    "instance_id": "agent.py:search_15",
                    "file_path": "/test/agent.py",
                    "line_number": 15,
                    "category": "tool",
                }
            ],
        },
        relationships=[
            ComponentRelationship(
                source_instance_id="agent.py:StateGraph_10",
                target_instance_id="agent.py:ChatOpenAI_5",
                label="USES_LLM",
                source_name="StateGraph",
                target_name="ChatOpenAI",
                source_category="agent",
                target_category="model",
            ),
            ComponentRelationship(
                source_instance_id="agent.py:StateGraph_10",
                target_instance_id="agent.py:search_15",
                label="USES_TOOL",
                source_name="StateGraph",
                target_name="search",
                source_category="agent",
                target_category="tool",
            ),
        ],
    )
