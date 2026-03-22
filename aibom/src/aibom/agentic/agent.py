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

"""Factory for creating an AIBOM agentic scanner powered by Deep Agents.

All ``deepagents`` and ``langchain`` imports are **lazy** — they only
execute when :func:`create_aibom_agent` is called, which happens only
when the user passes ``--agent-model`` to the CLI.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..models import (
    AIComponent,
    ComponentRelationship,
    DetectionSource,
    RiskFlag,
    ScanResult,
    SourceResult,
)
from .middleware import AIBOMScannerMiddleware
from .prompts import AIBOM_AGENT_SYSTEM_PROMPT

_LOGGER = logging.getLogger(__name__)


class AgenticEnrichmentError(Exception):
    """Raised when the agentic enrichment pipeline fails."""


def create_aibom_agent(
    model_string: str,
    *,
    llm_config: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> Any:
    """Create a Deep Agents-powered AIBOM scanning agent.

    Parameters
    ----------
    model_string:
        Model identifier accepted by ``langchain.chat_models.init_chat_model``
        (e.g. ``"claude-sonnet-4-20250514"``, ``"openai:gpt-5"``,
        ``"ollama/llama3"``).
    llm_config:
        Optional dict with ``api_key``, ``api_base``, ``api_version``
        passed through from the CLI ``--llm-*`` flags.
    system_prompt:
        Override the default AIBOM agent system prompt.

    Returns
    -------
    A compiled LangGraph ``CompiledStateGraph`` ready for ``.invoke()``.
    """
    from deepagents import create_deep_agent
    from langchain.chat_models import init_chat_model

    from .tools import build_tools

    init_kwargs: dict[str, Any] = {}
    if llm_config:
        if llm_config.get("api_key"):
            init_kwargs["api_key"] = llm_config["api_key"]
        if llm_config.get("api_base"):
            init_kwargs["base_url"] = llm_config["api_base"]

    model = init_chat_model(model_string, **init_kwargs)
    tools = build_tools()

    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt or AIBOM_AGENT_SYSTEM_PROMPT,
        name="aibom-scanner",
    )
    return agent


def run_agentic_enrichment(
    model_string: str,
    deterministic_components: list[AIComponent],
    deterministic_relationships: list[ComponentRelationship],
    scan_paths: list[str],
    llm_config: dict[str, Any] | None = None,
) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
    """Run the full agentic enrichment pipeline.

    1. Creates the AIBOM agent with the specified model.
    2. Feeds the deterministic scan results as context.
    3. Invokes the agent to enrich, resolve, and discover.
    4. Parses the agent's output and merges into AIBOM model objects.

    Returns
    -------
    Tuple of (enriched_components, new_relationships, risk_flags).
    The enriched_components list includes both the original components
    (with any updates applied) and any new components the agent discovered.
    """
    agent = create_aibom_agent(model_string, llm_config=llm_config)
    middleware = AIBOMScannerMiddleware()

    summary = _build_context_message(
        deterministic_components, deterministic_relationships, scan_paths
    )

    _LOGGER.info(
        "Running agentic enrichment with %s (%d components, %d relationships)",
        model_string,
        len(deterministic_components),
        len(deterministic_relationships),
    )

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": summary}]}
        )
    except Exception as exc:
        raise AgenticEnrichmentError(
            f"Agent invocation failed: {exc}"
        ) from exc

    final_message = _extract_final_message(result)
    if not final_message:
        _LOGGER.warning("Agent returned no usable output")
        return list(deterministic_components), [], []

    new_components, new_relationships, risk_flags = middleware.extract_findings(
        final_message
    )
    enriched = middleware.apply_enrichments(deterministic_components, final_message)

    all_components = enriched + new_components

    _LOGGER.info(
        "Agentic enrichment complete: %d enriched, %d new components, "
        "%d new relationships, %d risk flags",
        len(enriched),
        len(new_components),
        len(new_relationships),
        len(risk_flags),
    )

    return all_components, new_relationships, risk_flags


def _build_context_message(
    components: list[AIComponent],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
) -> str:
    """Build the user message that seeds the agent with deterministic results."""
    comp_summaries = []
    for c in components:
        entry: dict[str, Any] = {
            "instance_id": c.instance_id,
            "name": c.name,
            "type": c.component_type.value,
            "file": c.file_path,
            "line": c.line_number,
            "framework": c.framework,
        }
        if c.model_name:
            entry["model_name"] = c.model_name
        if c.metadata:
            entry["metadata"] = c.metadata
        comp_summaries.append(entry)

    rel_summaries = [
        {
            "source": r.source_name,
            "target": r.target_name,
            "type": r.relationship_type.value,
        }
        for r in relationships
    ]

    context = {
        "scan_paths": scan_paths,
        "total_components": len(components),
        "total_relationships": len(relationships),
        "components": comp_summaries,
        "relationships": rel_summaries,
    }

    return (
        "Here are the deterministic scan results from the AIBOM scanners. "
        "Please enrich these findings following the workflow in your instructions.\n\n"
        f"```json\n{json.dumps(context, indent=2)}\n```"
    )


def _extract_final_message(result: Any) -> Optional[str]:
    """Extract the text content from the agent's final response."""
    messages = result.get("messages", [])
    if not messages:
        return None
    last = messages[-1]
    if hasattr(last, "content"):
        return str(last.content)
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(last)
