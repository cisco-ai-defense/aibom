# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in supplements to the prebuilt DuckDB component catalog.

These entries fill coverage gaps for frameworks that post-date the current
catalog snapshot.  They are injected into :class:`~aibom.catalog_db.CatalogDB`
at scan start so KB enrichment can recognise common Strands, AWS, and other
modern agent SDK symbols without requiring end-users to maintain a custom
``.aibom.yaml``.

Schema mirrors the dict produced by
:meth:`aibom.custom_catalog.CustomComponentEntry.to_catalog_dict`:
``{id, label, concept, framework, sig_name, type, catalog_label}``.

All entries go through the same concept allowlist
(:data:`aibom.scanners.kb_enrichment_scanner.ALLOWED_CONCEPTS`), so only
``agent``/``model``/``tool``/``memory``/``prompt`` entries surface as
components.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _entry(
    entry_id: str,
    *,
    concept: str,
    label: str,
    framework: str,
) -> Dict[str, Any]:
    """Build a single catalog-dict entry in the shape :class:`CatalogDB` expects."""
    return {
        "id": entry_id,
        "label": label,
        "concept": concept,
        "framework": framework,
        "sig_name": None,
        "type": None,
        "catalog_label": None,
    }


# ---------------------------------------------------------------------------
# Strands Agents SDK (https://github.com/strands-agents/sdk-python)
# ---------------------------------------------------------------------------

# Public Agent entry point.  Used as ``from strands import Agent`` or
# ``strands.Agent``.  Internal re-exports (``strands.agent.agent.Agent``) are
# also common so we register the common aliases.
_STRANDS_AGENT_ENTRIES: List[Dict[str, Any]] = [
    _entry("strands.Agent", concept="agent", label="class", framework="strands"),
    _entry("strands.agent.Agent", concept="agent", label="class", framework="strands"),
    _entry(
        "strands.agent.agent.Agent",
        concept="agent",
        label="class",
        framework="strands",
    ),
]

# Model provider classes exported from ``strands.models``.
# List sourced from the Strands SDK README (provider packages).
_STRANDS_MODEL_CLASSES: tuple[str, ...] = (
    "AnthropicModel",
    "BedrockModel",
    "GeminiModel",
    "LiteLLMModel",
    "LlamaAPIModel",
    "LlamaCppModel",
    "MistralAIModel",
    "OllamaModel",
    "OpenAIModel",
    "SageMakerAIModel",
    "WriterModel",
)

_STRANDS_MODEL_ENTRIES: List[Dict[str, Any]] = [
    _entry(
        f"strands.models.{cls_name}",
        concept="model",
        label="class",
        framework="strands",
    )
    for cls_name in _STRANDS_MODEL_CLASSES
]

# MCP client class exposed via ``strands.tools.mcp``.
_STRANDS_MCP_ENTRIES: List[Dict[str, Any]] = [
    _entry(
        "strands.tools.mcp.MCPClient",
        concept="tool",
        label="class",
        framework="strands",
    ),
    _entry(
        "strands.tools.mcp.mcp_client.MCPClient",
        concept="tool",
        label="class",
        framework="strands",
    ),
]

# Built-in tool functions exported from ``strands_tools``.
# List sourced from https://github.com/strands-agents/tools README.
_STRANDS_TOOL_FUNCTIONS: tuple[str, ...] = (
    "agent_core_memory",
    "agent_graph",
    "batch",
    "browser",
    "calculator",
    "chat_video",
    "cron",
    "current_time",
    "diagram",
    "editor",
    "elasticsearch_memory",
    "environment",
    "exa_search",
    "file_read",
    "file_write",
    "generate_image",
    "graph",
    "handoff_to_user",
    "http_request",
    "journal",
    "mcp_client",
    "mem0_memory",
    "memory",
    "mongodb_memory",
    "python_repl",
    "retrieve",
    "rss",
    "search_video",
    "shell",
    "slack",
    "sleep",
    "speak",
    "stop",
    "swarm",
    "tavily_search",
    "think",
    "use_agent",
    "use_aws",
    "use_computer",
    "use_llm",
    "workflow",
)

_STRANDS_TOOL_ENTRIES: List[Dict[str, Any]] = [
    _entry(
        f"strands_tools.{name}",
        concept="tool",
        label="function",
        framework="strands",
    )
    for name in _STRANDS_TOOL_FUNCTIONS
]


def _all_entries() -> List[Dict[str, Any]]:
    """Return the full built-in catalog supplement.

    Kept as a function so tests can assert the shape/size without importing
    every individual list.
    """
    return [
        *_STRANDS_AGENT_ENTRIES,
        *_STRANDS_MODEL_ENTRIES,
        *_STRANDS_MCP_ENTRIES,
        *_STRANDS_TOOL_ENTRIES,
    ]


BUILTIN_CATALOG_ENTRIES: List[Dict[str, Any]] = _all_entries()
"""The full list of built-in catalog supplements injected at scan start."""
