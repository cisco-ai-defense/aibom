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

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_core import PydanticUndefined

from ..agent_signatures import AgentSignatureCatalog
from ..cache_paths import cache_read_dirs, ensure_cache_dir
from ..models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    DetectionSource,
    RelationshipType,
    RiskFlag,
    ScanResult,
    SourceResult,
)
from .evidence_injection import DossierIndex, build_dossier_index
from .middleware import AIBOMScannerMiddleware
from .prompts import AGENTIC_COERCION_PROMPT, AIBOM_AGENT_SYSTEM_PROMPT

_IID_DESC = (
    "The EXACT instance_id string as provided in the input — "
    "copy it verbatim including underscores, full absolute paths, "
    "and trailing line numbers. Do NOT shorten, reformat, or omit any part."
)


@dataclass
class TokenUsage:
    """Aggregated LLM token counts across batches."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Subset of prompt_tokens served from the provider's prompt cache (cache
    # read). Lets us measure prompt-caching savings — Azure/OpenAI cache stable
    # prompts automatically, but the win is invisible without this.
    cached_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens


_token_accumulator = TokenUsage()


def _reset_token_usage() -> None:
    global _token_accumulator
    _token_accumulator = TokenUsage()


def get_token_usage() -> TokenUsage:
    return _token_accumulator


def _as_int(value: Any) -> int:
    """Coerce a token count to int, treating missing/invalid as 0."""
    return value if isinstance(value, int) else 0


def _nested_int(d: Any, *path: str) -> int:
    """Read a nested int (e.g. cache-read tokens), 0 if any level is missing."""
    for key in path:
        if not isinstance(d, dict):
            return 0
        d = d.get(key)
    return _as_int(d)


def _resolve_message_usage(msg: Any) -> tuple[int, int, int]:
    """Resolve ``(prompt, completion, total)`` tokens for one AI message.

    LangChain's standardized ``usage_metadata`` is preferred, but several
    providers do not populate it and instead surface usage under
    ``response_metadata``:

    * OpenAI / Azure (incl. the ``gpt-5.3-codex`` deployment):
      ``response_metadata['token_usage']`` with
      ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``.
    * AWS Bedrock Converse: ``response_metadata['usage']`` with
      ``input_tokens`` / ``output_tokens``.
    * AWS Bedrock Invoke: ``response_metadata['amazon-bedrock-invocationMetrics']``
      with ``inputTokenCount`` / ``outputTokenCount``.

    The first populated carrier wins, so ``usage_metadata`` (which LangChain
    often derives from ``response_metadata``) is never double-counted. ``total``
    is synthesized from prompt+completion when the provider omits it. Returns
    ``(0, 0, 0)`` — logged at debug — when no carrier resolves.
    """
    um = getattr(msg, "usage_metadata", None)
    if isinstance(um, dict):
        prompt = _as_int(um.get("input_tokens"))
        completion = _as_int(um.get("output_tokens"))
        total = _as_int(um.get("total_tokens"))
        cached = _nested_int(um, "input_token_details", "cache_read")
        if prompt or completion or total:
            return prompt, completion, total or (prompt + completion), cached

    rm = getattr(msg, "response_metadata", None)
    if isinstance(rm, dict):
        token_usage = rm.get("token_usage")
        if isinstance(token_usage, dict):
            prompt = _as_int(token_usage.get("prompt_tokens"))
            completion = _as_int(token_usage.get("completion_tokens"))
            total = _as_int(token_usage.get("total_tokens"))
            cached = _nested_int(token_usage, "prompt_tokens_details", "cached_tokens")
            if prompt or completion or total:
                return prompt, completion, total or (prompt + completion), cached

        usage = rm.get("usage")
        if isinstance(usage, dict):
            prompt = _as_int(usage.get("input_tokens"))
            completion = _as_int(usage.get("output_tokens"))
            total = _as_int(usage.get("total_tokens"))
            cached = _as_int(usage.get("cache_read_input_tokens"))
            if prompt or completion or total:
                return prompt, completion, total or (prompt + completion), cached

        metrics = rm.get("amazon-bedrock-invocationMetrics")
        if isinstance(metrics, dict):
            prompt = _as_int(metrics.get("inputTokenCount"))
            completion = _as_int(metrics.get("outputTokenCount"))
            if prompt or completion:
                return prompt, completion, prompt + completion, 0

    return 0, 0, 0, 0


def _accumulate_token_usage(result: Any) -> None:
    """Sum token usage across all AI messages in a LangGraph result.

    Reads ``usage_metadata`` when present and falls back to
    ``response_metadata`` carriers for providers that omit it (see
    :func:`_resolve_message_usage`).
    """
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for msg in messages:
        prompt, completion, total, cached = _resolve_message_usage(msg)
        if not (prompt or completion or total):
            um = getattr(msg, "usage_metadata", None)
            rm = getattr(msg, "response_metadata", None)
            if um or rm:
                _LOGGER.debug(
                    "No resolvable token usage for %s message "
                    "(usage_metadata=%s, response_metadata keys=%s)",
                    type(msg).__name__,
                    bool(um),
                    sorted(rm) if isinstance(rm, dict) else None,
                )
            continue
        _token_accumulator.prompt_tokens += prompt
        _token_accumulator.completion_tokens += completion
        _token_accumulator.total_tokens += total
        _token_accumulator.cached_tokens += cached


class _NullTolerantModel(BaseModel):
    """Base for the agent's structured-output schemas.

    LLMs routinely emit an explicit ``null`` for a field that carries a
    default (e.g. ``{"evidence_snippet": null}``) instead of omitting it.
    Pydantic rejects ``null`` for a non-Optional defaulted field, and because
    an entire batch reply is validated against a single ``AgentResponse``
    schema, one stray ``null`` from the model would invalidate the whole
    response — discarding every component in the batch and forcing a
    cooldown retry.

    This validator drops any explicit ``null`` for a field that has a
    non-``None`` default, so the field falls back to that default. Fields
    that are genuinely ``Optional`` with a ``None`` default keep the ``null``,
    and any provided non-null value is left untouched.
    """

    @model_validator(mode="before")
    @classmethod
    def _drop_null_for_defaulted_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned = data
        for name, field_info in cls.model_fields.items():
            if data.get(name, PydanticUndefined) is not None:
                continue
            has_default = field_info.default is not PydanticUndefined
            has_factory = field_info.default_factory is not None
            if not (has_default or has_factory):
                continue
            # A ``None`` default means the field is legitimately nullable;
            # preserve the explicit ``null`` rather than dropping it.
            if has_default and field_info.default is None:
                continue
            if cleaned is data:
                cleaned = dict(data)
            cleaned.pop(name)
        return cleaned


class _EvidenceLocation(_NullTolerantModel):
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    role: str = ""


class _DecisionAnnotation(_NullTolerantModel):
    decision: str = ""
    justification: str = ""
    evidence_kinds: list[str] = Field(default_factory=list)
    evidence_locations: list[_EvidenceLocation] = Field(default_factory=list)


class AgentEvidence(_NullTolerantModel):
    """Structured evidence that a component is truly an agent.

    Every classification that results in ``component_type == "agent"`` MUST
    populate this. The verification gate in
    :class:`aibom.agentic.middleware.AIBOMScannerMiddleware` checks that the
    claimed evidence actually exists in the scanned source code (file path
    resolves, the line range contains ``evidence_snippet`` after whitespace
    normalization, and ``pattern`` is one of the accepted values).

    Patterns
    --------
    framework_agent
        Direct use of a known agent-framework entrypoint — e.g. LangChain
        ``AgentExecutor``, LangGraph ``create_react_agent``, AutoGen
        ``AssistantAgent``, CrewAI ``Agent``.
    react_loop
        An explicit reasoning loop: ``while``/``for`` body that calls an LLM
        and dispatches to tools based on the LLM's structured output.
    framework_inheritance
        Subclasses a known agent base class (e.g. ``BaseSingleActionAgent``,
        LlamaIndex ``BaseAgent``) and implements agent methods.
    a2a_server
        Registered as an A2A agent — serves an Agent Card at
        ``/.well-known/agent.json`` or instantiates ``A2AServer``.
    remote_proxy
        Thin client invoking a remote agent, where the remote side has been
        independently verified (A2A Agent Card found, OpenAI Assistants API
        used, or cross-repo resolution matched a verified agent).
    other
        Custom pattern — ``justification`` must explain why.
    """

    pattern: Literal[
        "framework_agent",
        "react_loop",
        "framework_inheritance",
        "a2a_server",
        "remote_proxy",
        "other",
    ] = "other"
    definition_file: str = ""
    definition_start_line: int = 0
    definition_end_line: int = 0
    evidence_snippet: str = ""
    justification: str = ""


class _EnrichedComponent(_NullTolerantModel):
    instance_id: str = Field(default="", description=_IID_DESC)
    updates: dict[str, Any] = Field(default_factory=dict)
    decision_annotation: _DecisionAnnotation | None = None
    agent_evidence: AgentEvidence | None = None


class _NewComponent(_NullTolerantModel):
    name: str = ""
    component_type: str = "other"
    file_path: str = ""
    line_number: int = 0
    framework: str = ""
    model_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    decision_annotation: _DecisionAnnotation | None = None
    agent_evidence: AgentEvidence | None = None


class _RemoveComponent(_NullTolerantModel):
    instance_id: str = Field(default="", description=_IID_DESC)
    reason: str = ""


class _ReclassifyComponent(_NullTolerantModel):
    instance_id: str = Field(default="", description=_IID_DESC)
    new_type: str = ""
    reason: str = ""
    agent_evidence: AgentEvidence | None = None


class _Relationship(_NullTolerantModel):
    source_name: str = ""
    target_name: str = ""
    relationship_type: str = ""
    source_type: str = ""
    target_type: str = ""
    decision_annotation: _DecisionAnnotation | None = None


class _RiskFinding(_NullTolerantModel):
    flag: str = ""
    description: str = ""
    file_path: str = ""
    line_number: int = 0
    severity: str = "info"
    decision_annotation: _DecisionAnnotation | None = None


class AgentResponse(_NullTolerantModel):
    """Structured output schema for the AIBOM agent."""

    enriched_components: list[_EnrichedComponent] = Field(default_factory=list)
    new_components: list[_NewComponent] = Field(default_factory=list)
    remove_components: list[_RemoveComponent] = Field(default_factory=list)
    reclassify_components: list[_ReclassifyComponent] = Field(default_factory=list)
    new_relationships: list[_Relationship] = Field(default_factory=list)
    risk_findings: list[_RiskFinding] = Field(default_factory=list)


_LOGGER = logging.getLogger(__name__)


class AgenticEnrichmentError(Exception):
    """Raised when the agentic enrichment pipeline fails."""


def _build_rate_limiter(
    requests_per_second: float = 1.0,
    max_bucket_size: int = 10,
) -> Any:
    """Return a client-side rate limiter for LLM calls.

    ``rate_limiter`` is a first-class field on LangChain's ``BaseChatModel``,
    so it works for *every* provider (Bedrock, OpenAI, Anthropic, Azure,
    Ollama, etc.) without provider-specific branching.  It proactively
    throttles outgoing requests so the provider never sees a burst — preventing
    rate-limit errors rather than recovering from them after the fact.

    ``requests_per_second`` / ``max_bucket_size`` default to a conservative
    1 req/s (burst 10) — safe for the tightest provider quotas. Operators who
    have confirmed a higher quota can raise the rate via ``--agentic-rate-limit``;
    the default is intentionally left unchanged.
    """
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=max_bucket_size,
    )


def _build_model(
    model_string: str,
    llm_config: dict[str, Any] | None = None,
) -> Any:
    """Build a LangChain ``BaseChatModel`` for the given model string."""
    from ..llm_factory import build_chat_model

    cfg = llm_config or {}
    rate_limiter = _build_rate_limiter(
        requests_per_second=cfg.get("rate_limit_rps") or 1.0,
        max_bucket_size=cfg.get("rate_limit_bucket") or 10,
    )
    return build_chat_model(
        model_string,
        provider=cfg.get("provider"),
        api_key=cfg.get("api_key"),
        api_base=cfg.get("api_base"),
        api_version=cfg.get("api_version"),
        # No default cap: omit max_tokens so the model may generate up to its
        # own context limit (proven behavior; standard OpenAI/vLLM semantics).
        # A fixed default is wrong-in-some-direction — too low truncates verbose
        # reasoners, too high reserves the whole context window and starves the
        # input (HTTP 400). Users opt in via --llm-max-tokens / llm_config.
        max_tokens=cfg.get("max_tokens"),
        rate_limiter=rate_limiter,
        reasoning=cfg.get("reasoning", "auto"),
        init_kwargs_extra=cfg.get("init_kwargs"),
    )


def _close_model_clients(*models: Any) -> None:
    """Best-effort shutdown of async HTTP transports inside LangChain models.

    LangChain eagerly creates both sync and async OpenAI clients.  We only
    use the sync path (``.invoke()``), but the unused ``async_client``'s
    ``__del__`` tries to schedule cleanup on a closed event loop at
    interpreter exit, producing noisy ``RuntimeError: Event loop is closed``
    tracebacks.  Closing explicitly while the loop is still reachable
    prevents that.
    """
    for model in models:
        try:
            ac = getattr(model, "async_client", None)
            if ac is not None and hasattr(ac, "close"):
                ac.close()
        except Exception:
            pass
        try:
            sc = getattr(model, "client", None)
            if sc is not None and hasattr(sc, "close"):
                sc.close()
        except Exception:
            pass


class _AgentBundle:
    """Wraps a compiled deep-agent graph and carries the underlying chat model.

    The batch runners invoke the graph via ``.invoke()`` / ``.ainvoke()``; those
    (and any other graph attribute) are transparently proxied. The extra
    ``aibom_chat_model`` and ``needs_coercion`` attributes let the two-phase
    structured-output path run a tool-less coercion call without
    threading the model through every batch/tier signature.
    """

    def __init__(self, graph: Any, chat_model: Any, *, needs_coercion: bool):
        self._graph = graph
        self.aibom_chat_model = chat_model
        self.needs_coercion = needs_coercion

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not set on the bundle itself → proxy to the
        # underlying compiled graph (invoke, ainvoke, get_state, …).
        return getattr(self._graph, name)


# Ephemeral cache breakpoint (5m default TTL). aibom's batches run seconds apart,
# and every cache hit refreshes the 5m window for free, so 5m comfortably covers a
# whole multi-batch run without paying the higher 1h write premium.
_BEDROCK_CACHE_CONTROL = {"type": "ephemeral", "ttl": "5m"}

# Model-id substrings whose Bedrock family supports Anthropic-style prompt caching.
# Mirrors langchain_aws's ``_is_supported_model`` so a non-Claude/Nova Bedrock
# model (e.g. Llama) is never poked with a ``cache_control`` block it may reject.
_BEDROCK_CACHEABLE_MODEL_MARKERS = ("anthropic", "amazon.nova")


def _build_bedrock_cache_middleware() -> Any:
    """Build the middleware that caches the stable system+tools prefix on Bedrock.

    ``langchain_aws``'s built-in ``BedrockPromptCachingMiddleware`` places the
    breakpoint on the LAST message. aibom re-sends an identical ~12k-token system
    prompt and tool schema with a DIFFERENT last message on every batch (there can
    be dozens), so a last-message breakpoint never matches across batches — no
    cache read (verified live: ``cached_tokens=0``). Tagging the stable SYSTEM
    block instead caches the whole tools+system prefix (Anthropic caches
    everything before the breakpoint, and ``ChatBedrock`` prepends the tool schema
    ahead of the system text), which every batch shares — verified live to yield
    cross-batch ``cache_read`` on Bedrock Claude via the InvokeModel path.

    Defined lazily (the ``AgentMiddleware`` base ships with the agentic extra) so
    importing this module never requires deepagents/langchain.
    """
    from langchain.agents.middleware import AgentMiddleware, ModelRequest
    from langchain_core.messages import SystemMessage

    class _BedrockSystemPromptCacheMiddleware(AgentMiddleware):
        """Tag the system prompt's last block with an ephemeral cache breakpoint."""

        def _cache_tagged_system(self, request: ModelRequest) -> Any | None:
            sm = request.system_message
            if sm is None:
                return None
            model_id = (
                getattr(request.model, "model_id", "")
                or getattr(request.model, "model", "")
                or ""
            ).lower()
            if not any(m in model_id for m in _BEDROCK_CACHEABLE_MODEL_MARKERS):
                return None

            content = sm.content
            if isinstance(content, str):
                # Plain-string system prompt → a single cache-tagged text block.
                if not content:
                    return None
                new_content: list[Any] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": _BEDROCK_CACHE_CONTROL,
                    }
                ]
            elif isinstance(content, list) and content:
                # Structured content (the deep agent delivers the system prompt as
                # a content-block list): copy every block verbatim and add the
                # breakpoint to the LAST text block only — never flatten or drop
                # blocks. Anthropic caches everything up to that breakpoint.
                new_content = [dict(b) if isinstance(b, dict) else b for b in content]
                last_text = next(
                    (
                        i
                        for i in range(len(new_content) - 1, -1, -1)
                        if isinstance(new_content[i], dict)
                        and new_content[i].get("type") == "text"
                    ),
                    None,
                )
                if last_text is None:
                    return None
                new_content[last_text] = {
                    **new_content[last_text],
                    "cache_control": _BEDROCK_CACHE_CONTROL,
                }
            else:
                return None

            return SystemMessage(content=new_content)

        def wrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
            new_sm = self._cache_tagged_system(request)
            if new_sm is not None:
                return handler(request.override(system_message=new_sm))
            return handler(request)

        async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
            new_sm = self._cache_tagged_system(request)
            if new_sm is not None:
                return await handler(request.override(system_message=new_sm))
            return await handler(request)

    return _BedrockSystemPromptCacheMiddleware()


def _is_bedrock_model(
    model: Any, model_string: str, llm_config: dict[str, Any] | None
) -> bool:
    """True when the model is ``ChatBedrock`` — the Bedrock InvokeModel path.

    Scoped deliberately to InvokeModel: the breakpoint we inject uses the
    Anthropic ``cache_control`` shape, which is how ``ChatBedrock`` (InvokeModel)
    expresses caching. ``ChatBedrockConverse`` (Converse API) needs a ``cachePoint``
    content block instead, so tagging it with ``cache_control`` would silently fail
    to cache — and aibom builds only ``ChatBedrock`` today, so Converse is excluded
    rather than mis-tagged.

    The built *model* object is the authoritative signal: the tier runners call
    :func:`create_aibom_agent` with a pre-built model and a BARE model id
    (``us.anthropic.…``, no ``bedrock/`` prefix) and WITHOUT ``llm_config``. Fall
    back to provider resolution — matching the ``bedrock`` provider exactly, so
    ``bedrock_converse`` is not swept in — for callers that pass a ``bedrock/``
    prefix or an explicit provider but no model.
    """
    cls = type(model).__name__
    if cls == "ChatBedrock":
        return True
    if cls == "ChatBedrockConverse":
        return False
    from ..llm_factory import resolve_provider

    provider = resolve_provider(model_string, (llm_config or {}).get("provider")) or ""
    return provider.lower() == "bedrock"


def _prompt_caching_middleware(
    model: Any, model_string: str, llm_config: dict[str, Any] | None
) -> list[Any]:
    """Explicit prompt-caching middleware, gated strictly to the Bedrock path.

    * **bedrock** → one middleware that puts an ephemeral ``cache_control``
      breakpoint on the stable system+tools prefix aibom re-sends every batch
      (see :func:`_build_bedrock_cache_middleware`). Anthropic-on-Bedrock does not
      cache automatically, and the cache-read tokens it earns already surface in
      ``usage_metadata`` and are tallied by :func:`_resolve_message_usage`.
    * **everything else** → ``[]``. Native Anthropic (``ChatAnthropic``) is
      already cached by deepagents' built-in ``AnthropicPromptCachingMiddleware``
      — adding our own would duplicate breakpoints against the 4-breakpoint
      budget. OpenAI/Azure cache server-side automatically and reject/ignore an
      explicit breakpoint, so they stay untouched.

    Returns ``[]`` (a graceful no-op) if the agentic extra is not installed.
    """
    if not _is_bedrock_model(model, model_string, llm_config):
        return []
    try:
        return [_build_bedrock_cache_middleware()]
    except ImportError:
        _LOGGER.debug(
            "langchain agents middleware unavailable; skipping Bedrock prompt caching",
            exc_info=True,
        )
        return []


def create_aibom_agent(
    model_string: str,
    *,
    llm_config: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    tools: list[Any] | None = None,
    model: Any | None = None,
    response_format: Any | None = None,
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
    tools:
        Override the default tool set.  When ``None`` the standard
        per-repo enrichment tools are used.
    model:
        Pre-built ``BaseChatModel``.  When provided, *model_string* and
        *llm_config* are ignored for model construction (the agent graph
        is still new).

    Returns
    -------
    An :class:`_AgentBundle` wrapping the compiled LangGraph agent. It proxies
    ``.invoke()`` / ``.ainvoke()`` to the graph and also carries the underlying
    chat model so the two-phase structured-output path can run a
    tool-less coercion call.

    Two-phase structured output
    ----------------------------------------
    By default (``response_format is None``) the agent runs the tool loop
    **unforced** — no structured-output tool is bound, so LangChain uses
    ``tool_choice="auto"`` and the loop terminates naturally on every provider
    (instead of ``ToolStrategy``'s per-turn ``tool_choice="any"``, which
    tool-eager Claude-on-Bedrock never escapes). The structured ``AgentResponse``
    is then produced by a separate, tool-less coercion call (see
    :func:`_coerce_structured`). Callers that want the legacy in-loop structured
    output (e.g. the provider-native fallback agent) pass an explicit
    *response_format*.
    """
    from deepagents import create_deep_agent

    from .tools import build_tools

    if model is None:
        model = _build_model(model_string, llm_config)

    graph = create_deep_agent(
        model=model,
        tools=tools if tools is not None else build_tools(),
        system_prompt=system_prompt or AIBOM_AGENT_SYSTEM_PROMPT,
        response_format=response_format,
        # Explicit Bedrock prompt caching on the stable system+tools prefix; a
        # no-op ([]) for every non-Bedrock provider (see
        # :func:`_prompt_caching_middleware`). Gated on the built ``model`` object
        # because the tier runners pass a bare model id and no llm_config.
        middleware=_prompt_caching_middleware(model, model_string, llm_config),
        name="aibom-scanner",
    )
    return _AgentBundle(graph, model, needs_coercion=response_format is None)


def _supports_inloop_structured_output(
    resolved_provider: str | None, model_id: str
) -> bool:
    """True when the model's LangChain integration reliably supports NATIVE
    in-loop structured output (``create_agent`` ProviderStrategy).

    Capability gate for the two-phase decouple. When True, aibom
    keeps the single-pass agent (``response_format=AgentResponse`` → native
    json_schema terminal — the proven, full-fidelity path for GPT). When False,
    it uses the provider-general two-phase decouple, because LangChain's
    ``AutoStrategy`` would otherwise route the model to ``ToolStrategy`` and bind
    ``tool_choice="any"`` on every turn — the trap that makes tool-eager
    Claude-on-Bedrock (and self-hosted/vLLM open models) never terminate.

    Mirrors ``create_agent``'s ProviderStrategy capability for the OpenAI family
    (gpt-*, o-series, grok, gpt-oss) on the ``openai``/``azure_openai`` providers
    (or LangChain-inferred). Bedrock Claude (InvokeModel), native Anthropic on
    pinned ids, Google, Ollama, and non-GPT open models served via an
    OpenAI-compatible endpoint all fall to the two-phase path.
    """
    prov = (resolved_provider or "").lower()
    if prov not in ("openai", "azure_openai", ""):
        return False
    leaf = model_id.rsplit("/", 1)[-1].strip().lower()
    if leaf.startswith(("gpt-", "grok", "gpt-oss")):
        return True
    # o-series reasoning models: o1 / o3-mini / o4-… (an ``o`` then a digit).
    return len(leaf) >= 2 and leaf[0] == "o" and leaf[1].isdigit()


def _agent_response_format(model_id: str, llm_config: dict[str, Any] | None) -> Any:
    """Pick the ``create_aibom_agent`` ``response_format`` for *model_id*.

    ``AgentResponse`` (single-pass native structured output) for
    ProviderStrategy-capable models; ``None`` (two-phase decouple) otherwise.
    """
    from ..llm_factory import resolve_provider

    provider = resolve_provider(model_id, (llm_config or {}).get("provider"))
    if _supports_inloop_structured_output(provider, model_id):
        return AgentResponse
    return None


def _build_fallback_agent(model_string: str, model_obj: Any) -> Any | None:
    """Build an alternate-strategy agent for the structured-output fallback.

    The primary agent uses the default (tool-calling) structured-output
    strategy; this builds one that uses the provider-native (json_schema)
    strategy, which recovers models that emit nothing usable via tool calls.
    Returns ``None`` if the strategy class is unavailable (agentic extras
    missing) or the agent cannot be built — the caller then skips the fallback.
    """
    try:
        from langchain.agents.structured_output import ProviderStrategy
    except ImportError:
        return None
    try:
        return create_aibom_agent(
            model_string,
            model=model_obj,
            response_format=ProviderStrategy(AgentResponse),
        )
    except Exception:
        _LOGGER.debug("Could not build structured-output fallback agent", exc_info=True)
        return None


_DEFAULT_BATCH_SIZE = 15

# LangGraph default since v1.0.6 and also the Deep Agents default.
# Set explicitly so the intent is clear: this is a safety net against
# infinite loops, NOT a workload budget.
_RECURSION_LIMIT = 1000

_DEFAULT_AGENTIC_TIMEOUT_S = 120
_DEFAULT_MAX_CONSECUTIVE_FAILURES = 3

# Aggregate wall-clock budget (seconds) for ALL retry activity across a single
# enrichment run. Bounds the retry pass so a persistently-failing model/gateway
# degrades the affected components and the scan finishes, instead of retrying in
# ever-smaller batches for hours. 0 disables the retry pass.
_DEFAULT_MAX_RETRY_SECONDS = 1200

# Hints whose batches produced no usable structured output under the primary
# strategy and are worth one re-run with the alternate structured-output
# strategy (see _strategy_fallback_pass). Distinct from _RETRYABLE_HINTS, which
# re-runs with the SAME strategy.
_STRATEGY_FALLBACK_HINTS = frozenset({"no_usable_output", "model_refused"})

_SIMPLE_CANDIDATE_TYPES = frozenset({"model", "dependency", "embedding"})

_SUB_AGENT_THRESHOLD = 50

_RETRY_COOLDOWN_S = 30
_RETRYABLE_HINTS = frozenset(
    {
        "batch_timeout",
        "batch_recursion_limit",
        "circuit_breaker_tripped",
        # Transient provider-side failures — worth one retry pass.
        "provider_outage",
        "rate_limited",
        "structured_output_parse_error",
    }
)

# Degraded hints that indicate the provider/deployment couldn't keep up with the
# offered load (timeouts, 429s, tripped circuit breaker, exhausted retry budget).
# When these dominate a partial degradation, lowering --agentic-concurrency /
# --agentic-rate-limit is the actionable remedy.
_DEGRADED_LOAD_HINTS = frozenset(
    {
        "batch_timeout",
        "rate_limited",
        "provider_outage",
        "circuit_breaker_tripped",
        "retry_budget_exhausted",
        "retry_failed",
    }
)


def _classify_failure_hint(exc: Exception) -> str:
    """Classify a batch-invocation exception into a precise agentic hint.

    Distinguishes provider outages and rate limits (HTTP status carried by
    ``openai.APIStatusError`` and friends) and structured-output parse failures
    (``langchain`` ``StructuredOutputValidationError`` carries an ``ai_message``)
    from the generic recursion/unknown bucket. Uses duck-typed attributes so no
    optional provider/agent dependency is imported.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 429:
            return "rate_limited"
        if status >= 500:
            return "provider_outage"
    if getattr(exc, "ai_message", None) is not None:
        return "structured_output_parse_error"
    if type(exc).__name__ in (
        "StructuredOutputValidationError",
        "OutputParserException",
    ):
        return "structured_output_parse_error"
    return "batch_recursion_limit"


def _refusal_present(result: Any) -> bool:
    """True when the final message is a model refusal with no usable content.

    Providers surface refusals in ``additional_kwargs['refusal']``; aibom uses
    this to record a distinct ``model_refused`` status instead of conflating a
    refusal with "examined and found nothing".
    """
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        return False
    extra = getattr(messages[-1], "additional_kwargs", None)
    return isinstance(extra, dict) and bool(extra.get("refusal"))


def _collect_failed(
    enriched: list[AIComponent],
) -> tuple[list[AIComponent], list[AIComponent]]:
    """Partition enriched results into ok and retryable components."""
    ok: list[AIComponent] = []
    retry: list[AIComponent] = []
    for c in enriched:
        if c.agentic_hint in _RETRYABLE_HINTS:
            retry.append(
                c.model_copy(update={"needs_agentic": True, "agentic_hint": ""})
            )
        else:
            ok.append(c)
    return ok, retry


def _classify_candidates(
    components: list[AIComponent],
) -> tuple[list[AIComponent], list[AIComponent]]:
    """Split candidates into simple (registry-confirmable) vs complex.

    Simple candidates:
    - known model/dependency/embedding with a model_name that just needs
      registry lookup confirmation.
    - dependency components tagged with ``known_ai_package=True`` (from
      the KNOWN_AI_PACKAGES hint list).
    Complex candidates: everything else — ambiguous types, missing model
    names, unknown packages, multi-file reasoning required.
    """
    simple: list[AIComponent] = []
    complex_: list[AIComponent] = []
    for c in components:
        known_ai_dep = (
            c.component_type.value == "dependency"
            and c.metadata.get("known_ai_package") is True
        )
        is_simple = known_ai_dep or (
            c.component_type.value in _SIMPLE_CANDIDATE_TYPES
            and c.model_name
            and not c.metadata.get("env")
            and not c.metadata.get("env_context")
            and not c.metadata.get("partial_kb_id")
            and not c.metadata.get("suggestive_signal")
        )
        if is_simple:
            simple.append(c)
        else:
            complex_.append(c)
    return simple, complex_


def _locality_aware_batches(
    components: list[AIComponent],
    batch_size: int,
) -> list[list[AIComponent]]:
    """Group components by parent directory, then split into batches.

    Co-located components share imports and context, so batching them
    together lets the agent reason about a directory as a unit and
    reduces redundant file reads.
    """
    from collections import defaultdict

    by_dir: dict[str, list[AIComponent]] = defaultdict(list)
    for c in components:
        parent = str(Path(c.file_path).parent) if c.file_path else "__unknown__"
        by_dir[parent].append(c)

    batches: list[list[AIComponent]] = []
    current: list[AIComponent] = []
    for _dir_key in sorted(by_dir):
        group = by_dir[_dir_key]
        for comp in group:
            current.append(comp)
            if len(current) >= batch_size:
                batches.append(current)
                current = []
    if current:
        batches.append(current)
    return batches


import hashlib as _hashlib


def _component_cache_key(c: AIComponent) -> str:
    """Derive a content-based cache key for a component."""
    parts = [
        c.file_path or "",
        str(c.line_number),
        c.name,
        c.component_type.value,
        c.model_name or "",
    ]
    if c.file_path and c.line_number:
        snippet = _read_code_window(c.file_path, c.line_number)
        if snippet:
            parts.append(snippet)
    raw = "|".join(parts)
    return _hashlib.sha256(raw.encode()).hexdigest()[:24]


_TIER_CACHE_VERSION = 1
_BATCH_CACHE_VERSION = 1
_CROSS_REPO_CACHE_VERSION = 1


@dataclass
class _BatchArtifact:
    inputs: list[AIComponent]
    new: list[AIComponent]
    rels: list[ComponentRelationship]
    flags: list[RiskFlag]


def _tier_cache_key(components: list[AIComponent]) -> str:
    """Derive a cache key for the exact inputs to one tier run."""
    raw = "|".join(_component_cache_key(c) for c in components)
    return f"tier_{_hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _batch_cache_key(components: list[AIComponent]) -> str:
    """Derive a cache key for one executed batch."""
    raw = "|".join(_component_cache_key(c) for c in components)
    return f"batch_{_hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _build_tier_cache_payload(
    enriched: list[AIComponent],
    new: list[AIComponent],
    rels: list[ComponentRelationship],
    flags: list[RiskFlag],
) -> dict[str, Any]:
    """Serialize the exact tier outputs for deterministic cache replay."""
    return {
        "_tier_cache_version": _TIER_CACHE_VERSION,
        "tier_enriched": [c.model_dump(mode="json") for c in enriched],
        "tier_new": [c.model_dump(mode="json") for c in new],
        "tier_rels": [r.model_dump(mode="json") for r in rels],
        "tier_flags": [f.model_dump(mode="json") for f in flags],
    }


def _load_tier_cache_payload(
    data: dict[str, Any] | None,
) -> (
    tuple[
        list[AIComponent],
        list[AIComponent],
        list[ComponentRelationship],
        list[RiskFlag],
    ]
    | None
):
    """Deserialize a cached tier payload, returning None for non-tier entries."""
    if not data or data.get("_tier_cache_version") != _TIER_CACHE_VERSION:
        return None
    return (
        [AIComponent.model_validate(item) for item in data.get("tier_enriched", [])],
        [AIComponent.model_validate(item) for item in data.get("tier_new", [])],
        [
            ComponentRelationship.model_validate(item)
            for item in data.get("tier_rels", [])
        ],
        [RiskFlag.model_validate(item) for item in data.get("tier_flags", [])],
    )


def _build_batch_cache_payload(
    new: list[AIComponent],
    rels: list[ComponentRelationship],
    flags: list[RiskFlag],
) -> dict[str, Any]:
    """Serialize batch-scope findings for mixed cache-hit replay."""
    return {
        "_batch_cache_version": _BATCH_CACHE_VERSION,
        "batch_new": [c.model_dump(mode="json") for c in new],
        "batch_rels": [r.model_dump(mode="json") for r in rels],
        "batch_flags": [f.model_dump(mode="json") for f in flags],
    }


def _load_batch_cache_payload(
    data: dict[str, Any] | None,
) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag]] | None:
    """Deserialize cached batch findings, returning None for non-batch entries."""
    if not data or data.get("_batch_cache_version") != _BATCH_CACHE_VERSION:
        return None
    return (
        [AIComponent.model_validate(item) for item in data.get("batch_new", [])],
        [
            ComponentRelationship.model_validate(item)
            for item in data.get("batch_rels", [])
        ],
        [RiskFlag.model_validate(item) for item in data.get("batch_flags", [])],
    )


def _normalized_cross_repo_results(
    per_repo_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for source in sorted(per_repo_results):
        data = per_repo_results[source]
        components: list[dict[str, Any]] = []
        for component in data.get("components", []):
            if hasattr(component, "model_dump"):
                item = component.model_dump(mode="json")
            else:
                item = dict(component)
            components.append(item)
        components.sort(
            key=lambda item: (
                str(item.get("instance_id", "")),
                str(item.get("name", "")),
                str(item.get("file_path", "")),
                int(item.get("line_number", 0) or 0),
            )
        )
        normalized[source] = {
            "components": components,
            "_unresolved_env_vars": sorted(
                str(v) for v in data.get("_unresolved_env_vars", [])
            ),
        }
    return normalized


def _cross_repo_cache_key(
    model_string: str,
    per_repo_results: dict[str, dict[str, Any]],
) -> str:
    raw = json.dumps(
        {
            "model_string": model_string,
            "per_repo_results": _normalized_cross_repo_results(per_repo_results),
        },
        sort_keys=True,
        default=str,
    )
    return f"xrepo_{_hashlib.sha256(raw.encode()).hexdigest()[:24]}"


def _build_cross_repo_cache_payload(
    rels: list[ComponentRelationship],
    flags: list[RiskFlag],
) -> dict[str, Any]:
    return {
        "_cross_repo_cache_version": _CROSS_REPO_CACHE_VERSION,
        "cross_repo_rels": [r.model_dump(mode="json") for r in rels],
        "cross_repo_flags": [f.model_dump(mode="json") for f in flags],
    }


def _load_cross_repo_cache_payload(
    data: dict[str, Any] | None,
) -> tuple[list[ComponentRelationship], list[RiskFlag]] | None:
    if not data or data.get("_cross_repo_cache_version") != _CROSS_REPO_CACHE_VERSION:
        return None
    return (
        [
            ComponentRelationship.model_validate(item)
            for item in data.get("cross_repo_rels", [])
        ],
        [RiskFlag.model_validate(item) for item in data.get("cross_repo_flags", [])],
    )


def _load_cached_component_snapshot(data: dict[str, Any] | None) -> AIComponent | None:
    """Deserialize a full component snapshot from the per-component cache."""
    if not data:
        return None
    raw_component = data.get("cached_component")
    if not isinstance(raw_component, dict):
        return None
    try:
        return AIComponent.model_validate(raw_component)
    except ValidationError:
        return None


def _record_memo_verdicts(
    memo: _DecisionMemo | None,
    original_components: list[AIComponent],
    enriched_components: list[AIComponent],
) -> None:
    """Populate the decision memo from resolved tier outputs."""
    if memo is None:
        return
    enriched_by_id = {c.instance_id: c for c in enriched_components}
    for c_before in original_components:
        memo.record(c_before, enriched_by_id.get(c_before.instance_id))
    _LOGGER.info("Decision memo now holds %d verdicts", len(memo))


class _AgenticResultCache:
    """In-process + optional on-disk cache for agentic batch results.

    Keyed by content hash of each component so unchanged code across
    re-runs skips the LLM entirely.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        fallback_dirs: list[Path] | None = None,
    ) -> None:
        self._mem: dict[str, dict[str, Any]] = {}
        self._disk_dir = cache_dir
        self._fallback_dirs = fallback_dirs or []
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk()
        for fallback_dir in self._fallback_dirs:
            self._load_disk(fallback_dir)

    def _load_disk(self, disk_dir: Path | None = None) -> None:
        target_dir = disk_dir or self._disk_dir
        if not target_dir:
            return
        for p in target_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self._mem.setdefault(p.stem, data)
            except (json.JSONDecodeError, OSError):
                continue

    def get(self, key: str) -> dict[str, Any] | None:
        return self._mem.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._mem[key] = value
        if self._disk_dir:
            # Write atomically: serialize to a temp file in the same directory,
            # then os.replace() into place. A crash mid-write leaves only the
            # temp file (ignored by the ``*.json`` resume glob), never a
            # half-written ``key.json`` that a resume would read as a corrupt
            # cache hit.
            dest = self._disk_dir / f"{key}.json"
            tmp = self._disk_dir / f".{key}.json.{os.getpid()}.tmp"
            try:
                tmp.write_text(json.dumps(value, default=str), encoding="utf-8")
                os.replace(tmp, dest)
            except OSError:
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def partition(
        self,
        components: list[AIComponent],
    ) -> tuple[list[AIComponent], list[AIComponent]]:
        """Split components into cached (hit) and uncached (miss)."""
        cached: list[AIComponent] = []
        uncached: list[AIComponent] = []
        for c in components:
            key = _component_cache_key(c)
            if self.get(key) is not None:
                cached.append(c)
            else:
                uncached.append(c)
        return cached, uncached

    def apply_cached(
        self,
        components: list[AIComponent],
        middleware: AIBOMScannerMiddleware,
    ) -> tuple[
        list[AIComponent],
        list[AIComponent],
        list[ComponentRelationship],
        list[RiskFlag],
    ]:
        """Apply cached agentic results to components."""
        enriched: list[AIComponent] = []
        all_new: list[AIComponent] = []
        all_rels: list[ComponentRelationship] = []
        all_flags: list[RiskFlag] = []
        seen_batch_keys: set[str] = set()
        for c in components:
            key = _component_cache_key(c)
            data = self.get(key)
            if data:
                cached_component = _load_cached_component_snapshot(data)
                if cached_component is not None:
                    enriched.append(middleware.hydrate_component(cached_component))
                else:
                    enriched_batch = middleware.apply_enrichments_from_dict([c], data)
                    enriched.extend(enriched_batch)

                batch_key = data.get("batch_artifact_key")
                batch_payload = None
                if (
                    isinstance(batch_key, str)
                    and batch_key
                    and batch_key not in seen_batch_keys
                ):
                    seen_batch_keys.add(batch_key)
                    batch_payload = _load_batch_cache_payload(self.get(batch_key))

                if batch_payload is not None:
                    new, rels, flags = batch_payload
                else:
                    new, rels, flags = middleware.extract_findings_from_dict(data)

                all_new.extend(new)
                all_rels.extend(rels)
                all_flags.extend(flags)
            else:
                enriched.append(c)
        return enriched, all_new, all_rels, all_flags


_MEMO_SAFE_TYPES: frozenset[str] = frozenset(
    {
        "dependency",
        "model",
        "model_artifact",
        "embedding",
    }
)


class _DecisionMemo:
    """Intra-run cache of agent verdicts for context-free component types.

    Only components whose type is in ``_MEMO_SAFE_TYPES`` are eligible.
    Keyed by ``(canonical_name, component_type)`` so the same package/model
    encountered in a later tier or sub-agent group reuses the earlier verdict
    without an additional LLM call.
    """

    def __init__(self) -> None:
        self._verdicts: dict[tuple[str, str], dict[str, Any]] = {}

    def _key(self, c: AIComponent) -> tuple[str, str] | None:
        if c.component_type.value not in _MEMO_SAFE_TYPES:
            return None
        canonical = (c.model_name or c.name).lower().strip()
        return (canonical, c.component_type.value)

    def record(self, c_before: AIComponent, c_after: AIComponent | None) -> None:
        """Record the agent's verdict for *c_before*.

        *c_after* is the enriched component after the agent processed it,
        or ``None`` if the agent removed it.
        """
        k = self._key(c_before)
        if k is None:
            return
        if c_after is None:
            self._verdicts[k] = {"action": "remove"}
        elif c_after.component_type != c_before.component_type:
            self._verdicts[k] = {
                "action": "reclassify",
                "new_type": c_after.component_type.value,
                "heuristic_confidence": c_after.heuristic_confidence,
            }
        else:
            self._verdicts[k] = {
                "action": "keep",
                "heuristic_confidence": c_after.heuristic_confidence,
            }

    def lookup(self, c: AIComponent) -> dict[str, Any] | None:
        k = self._key(c)
        return self._verdicts.get(k) if k else None

    def partition(
        self,
        components: list[AIComponent],
    ) -> tuple[list[AIComponent], list[AIComponent]]:
        """Split into (memo_hits, memo_misses)."""
        hits: list[AIComponent] = []
        misses: list[AIComponent] = []
        for c in components:
            if self.lookup(c) is not None:
                hits.append(c)
            else:
                misses.append(c)
        return hits, misses

    def apply(self, components: list[AIComponent]) -> list[AIComponent]:
        """Apply cached verdicts.  Returns only kept/reclassified components."""
        result: list[AIComponent] = []
        for c in components:
            verdict = self.lookup(c)
            if verdict is None:
                result.append(c)
                continue
            action = verdict["action"]
            if action == "remove":
                continue
            elif action == "reclassify":
                try:
                    new_type = AIComponentType(verdict["new_type"])
                except ValueError:
                    result.append(c)
                    continue
                result.append(
                    c.model_copy(
                        update={
                            "component_type": new_type,
                            "heuristic_confidence": verdict.get(
                                "heuristic_confidence", c.heuristic_confidence
                            ),
                            "needs_agentic": False,
                        }
                    )
                )
            else:
                result.append(
                    c.model_copy(
                        update={
                            "heuristic_confidence": verdict.get(
                                "heuristic_confidence", c.heuristic_confidence
                            ),
                            "needs_agentic": False,
                        }
                    )
                )
        return result

    def __len__(self) -> int:
        return len(self._verdicts)


def _degraded_batch_components(
    batch: list[AIComponent],
    *,
    hint: str,
) -> list[AIComponent]:
    return [
        c.model_copy(update={"needs_agentic": False, "agentic_hint": hint})
        for c in batch
    ]


def _circuit_breaker_skipped_batch(batch: list[AIComponent]) -> list[AIComponent]:
    return _degraded_batch_components(batch, hint="circuit_breaker_tripped")


def _apply_batch_findings(
    middleware: AIBOMScannerMiddleware,
    batch: list[AIComponent],
    data: dict[str, Any],
    batch_num: int,
) -> tuple[
    list[AIComponent],
    list[AIComponent],
    list[ComponentRelationship],
    list[RiskFlag],
    bool,
]:
    """Apply a parsed structured response to *batch*, failing OPEN per batch.

    ``extract_findings_from_dict`` / ``apply_enrichments_from_dict`` operate on
    model-supplied data that can be partial or malformed after structured-output
    recovery. Any unexpected error here must degrade only this batch — never
    propagate up and abort the whole agentic stage. The batch is
    marked with a retryable hint so a later pass can still recover it.

    Returns ``(enriched, new_components, new_relationships, risk_flags,
    degraded)`` matching the batch-runner contract.
    """
    try:
        new_components, new_rels, risk_flags = middleware.extract_findings_from_dict(
            data
        )
        enriched = middleware.apply_enrichments_from_dict(batch, data)
        return enriched, new_components, new_rels, risk_flags, False
    except Exception:
        _LOGGER.warning(
            "Batch %d: could not apply structured output (malformed/partial "
            "recovered data); degrading this batch only",
            batch_num,
            exc_info=True,
        )
        degraded = _degraded_batch_components(
            batch, hint="structured_output_parse_error"
        )
        return degraded, [], [], [], True


def _all_batches_failed(
    enriched: list[AIComponent],
    new: list[AIComponent],
    rels: list[ComponentRelationship],
    flags: list[RiskFlag],
) -> bool:
    """True when the agentic layer produced no usable output at all.

    Distinguishes a *total* failure — every returned component degraded and
    nothing discovered (e.g. a provider rejecting every batch) —
    from the benign "examined everything and found nothing to add". Lets the
    run surface a degraded status instead of logging "enrichment complete".
    """
    if new or rels or flags:
        return False
    considered = [c for c in enriched if c.instance_id]
    if not considered:
        return False
    return all(c.agentic_hint for c in considered)


def _count_degraded(components: list[AIComponent]) -> int:
    """Number of components left degraded (a non-empty ``agentic_hint`` in the
    agentic-output set marks a failed enrichment — same convention as
    ``_all_batches_failed``). Successful enrichment clears the hint to ``""``.
    """
    return sum(1 for c in components if c.agentic_hint)


def _dominant_degraded_hint(components: list[AIComponent]) -> str | None:
    """Most common degraded ``agentic_hint`` among *components*, or ``None`` when
    none are degraded. Used to pick the remediation message."""
    from collections import Counter

    hints = Counter(c.agentic_hint for c in components if c.agentic_hint)
    return hints.most_common(1)[0][0] if hints else None


def _coerce_structured(model: Any, messages: list[Any]) -> dict[str, Any] | None:
    """Phase 2 of the two-phase decouple.

    Turn the unforced agent's finished transcript into a schema-valid
    ``AgentResponse`` via a single, tool-less ``with_structured_output`` call.
    This uses each provider's *native* structured-output mechanism (OpenAI/Azure
    json_schema, Anthropic output_config, Google response_json_schema, Bedrock a
    single forced tool call, Ollama format=json_schema) and, having no tools,
    cannot loop — so it terminates cleanly on every provider.

    ``include_raw=True`` keeps the raw ``AIMessage`` so Phase-2 token usage is
    still accounted and a parse error can be recovered from the message carriers.
    Returns the response dict, or ``None`` if coercion is unavailable/failed.
    """
    if model is None:
        return None
    from langchain_core.messages import HumanMessage

    try:
        structured = model.with_structured_output(AgentResponse, include_raw=True)
    except Exception:
        # Some wrappers don't accept include_raw; retry without it.
        try:
            structured = model.with_structured_output(AgentResponse)
        except Exception:
            _LOGGER.debug(
                "with_structured_output unavailable for Phase-2 coercion",
                exc_info=True,
            )
            return None

    prompt = list(messages) + [HumanMessage(content=AGENTIC_COERCION_PROMPT)]
    try:
        out = structured.invoke(prompt)
    except Exception as exc:
        # StructuredOutputValidationError and friends carry the offending message.
        ai_message = getattr(exc, "ai_message", None)
        if ai_message is not None:
            _accumulate_token_usage({"messages": [ai_message]})
            return _extract_structured_response({"messages": [ai_message]})
        _LOGGER.warning("Phase-2 structured coercion failed: %s", exc)
        return None

    # include_raw shape: {"raw": AIMessage, "parsed": <model|None>, "parsing_error"}
    if isinstance(out, dict) and ("raw" in out or "parsed" in out):
        raw = out.get("raw")
        if raw is not None:
            _accumulate_token_usage({"messages": [raw]})
        parsed = out.get("parsed")
        if isinstance(parsed, BaseModel):
            return parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed
        if raw is not None:
            return _extract_structured_response({"messages": [raw]})
        return None

    # No include_raw: a direct model / dict.
    if isinstance(out, BaseModel):
        return out.model_dump()
    if isinstance(out, dict):
        return out
    return None


def _resolve_batch_data(agent: Any, result: Any) -> dict[str, Any] | None:
    """Get the structured ``AgentResponse`` dict from a finished batch.

    Prefers the cheap multi-carrier extractor (``structured_response`` / tool
    calls / parsed / JSON-in-text) — which already yields data for legacy or
    provider-native (ProviderStrategy fallback) agents, and for unforced agents
    whose final message is clean JSON. Only when that finds nothing AND the agent
    is an unforced Phase-1 agent does it spend a Phase-2 ``_coerce_structured``
    call. Keeps ``recursion_limit`` untouched and adds no extra call on the happy
    path.
    """
    data = _extract_structured_response(result)
    if data:
        return data
    # ``is True`` (not just truthy) so an incidental MagicMock agent in unrelated
    # tests never trips Phase 2 — only a real unforced bundle sets a bool True.
    if getattr(agent, "needs_coercion", False) is True:
        model = getattr(agent, "aibom_chat_model", None)
        messages = result.get("messages", []) if isinstance(result, dict) else []
        return _coerce_structured(model, messages)
    return None


class _InvokeTimeout(Exception):
    """Raised when a synchronous ``agent.invoke`` exceeds its wall-clock deadline.

    The generic timeout sentinel for every synchronous agent-invocation site.
    Subclasses :class:`Exception` (not ``BaseException``) so a site-level
    ``except Exception`` still catches it and fails open.
    """


class _BatchTimeout(_InvokeTimeout):
    """Raised when a synchronous *batch* invocation exceeds its deadline.

    A subclass of :class:`_InvokeTimeout` so the batch path's existing
    ``except _BatchTimeout`` handling is unchanged while the deadline mechanism
    is shared with every other invoke site.
    """


def _invoke_agent_bounded(
    agent: Any,
    content: str,
    timeout_s: int,
    *,
    recursion_limit: int | None = None,
) -> Any:
    """Run ``agent.invoke`` in a daemon thread with a hard wall-clock deadline.

    ``agent.invoke`` is a blocking deep-agent loop that cannot be cancelled.
    Running it via ``asyncio.run(asyncio.wait_for(asyncio.to_thread(...)))``
    returned control on timeout but then blocked **forever** on event-loop
    shutdown, which joins the orphaned (non-cancellable) executor thread — so a
    genuinely hung LLM call wedged the whole scan.

    Instead, run the call in a *daemon* thread and ``join`` it for at most
    ``timeout_s``. On timeout we raise :class:`_InvokeTimeout` and abandon the
    thread; being a daemon, it never blocks process/loop shutdown and dies with
    the interpreter. This is the single bounded-invoke primitive used by every
    synchronous agent-invocation site (batch enrichment, cross-repo
    coordination, container-layout resolution, and repo triage).

    ``recursion_limit`` is forwarded as ``config={"recursion_limit": N}`` only
    when set; left ``None`` the call passes no ``config`` so the agent applies
    its own default (some sites intentionally do not cap recursion).
    """
    box: dict[str, Any] = {}
    config: dict[str, Any] | None = (
        {"recursion_limit": recursion_limit} if recursion_limit is not None else None
    )

    def _worker() -> None:
        try:
            message = {"messages": [{"role": "user", "content": content}]}
            if config is not None:
                box["result"] = agent.invoke(message, config=config)
            else:
                box["result"] = agent.invoke(message)
        except BaseException as exc:  # noqa: BLE001 - propagate to caller thread
            box["error"] = exc

    worker = threading.Thread(
        target=_worker,
        name="aibom-agentic-invoke",
        daemon=True,
    )
    worker.start()
    worker.join(timeout_s)

    if worker.is_alive():
        # Timed out: abandon the still-running daemon worker. If its invoke
        # later completes it writes box["result"], but the caller never reads
        # box again after raising, so that late write is harmless (and dict
        # writes are GIL-atomic). The daemon dies with the interpreter.
        raise _InvokeTimeout()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _invoke_with_deadline(agent: Any, summary: str, timeout_s: int) -> Any:
    """Bounded batch invocation — thin wrapper over :func:`_invoke_agent_bounded`.

    Preserves the batch path's ``_BatchTimeout`` contract (so ``_run_batch``'s
    ``except _BatchTimeout`` is unchanged) while sharing the daemon-thread
    deadline mechanism with every other invoke site.
    """
    try:
        return _invoke_agent_bounded(
            agent, summary, timeout_s, recursion_limit=_RECURSION_LIMIT
        )
    except _InvokeTimeout as exc:
        raise _BatchTimeout() from exc


def _run_async_bounded(coro: Any) -> Any:
    """Run *coro* to completion on a private event loop without wedging the
    scan, then return its result (re-raising any exception).

    Each batch in *coro* is wrapped in ``asyncio.timeout``, so a stalled
    ``agent.ainvoke`` is cancelled at the coroutine level and the batch fails
    open. The problem this helper solves is teardown: ``asyncio.run`` finishes
    by calling ``loop.shutdown_default_executor()``, which *joins* the loop's
    default ``ThreadPoolExecutor`` with ``wait=True``. If the provider stack
    offloaded any blocking work via ``run_in_executor`` and that worker is
    slow/stuck, the scan blocks there long after every batch has timed out —
    the same class of hang fixed on the sync path.

    Instead we run the loop with ``run_until_complete`` inside a daemon thread
    and tear it down manually, deliberately **not** calling
    ``shutdown_default_executor``; the executor is abandoned with
    ``shutdown(wait=False)``. So once the bounded *coro* returns, the scan
    returns immediately regardless of executor state. Running inside a daemon
    thread also makes any executor workers daemon (threads inherit daemon
    status from their creator), so they don't keep the interpreter alive.

    Caveat (Python limitation, not specific to this code): if a provider hands
    a *truly blocking* call to ``run_in_executor`` that never returns, that
    worker cannot be hard-cancelled by any means, and ``concurrent.futures``
    joins pool workers in its own ``atexit`` handler — so process *exit* can
    lag on that single orphan. The scan itself still completes on time. Real
    langchain async providers use native-async I/O, which cancels cleanly.
    """
    import concurrent.futures

    box: dict[str, Any] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        # Created inside this daemon thread, so its workers are daemon too.
        executor = concurrent.futures.ThreadPoolExecutor(
            thread_name_prefix="aibom-agentic-io",
        )
        loop.set_default_executor(executor)
        asyncio.set_event_loop(loop)
        try:
            box["result"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 - propagate to caller
            box["error"] = exc
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:  # noqa: BLE001
                pass
            # Abandon (do NOT join) any in-flight executor work, then close.
            # Daemon workers die with the process; nothing blocks teardown.
            executor.shutdown(wait=False)
            asyncio.set_event_loop(None)
            loop.close()

    thread = threading.Thread(target=_runner, name="aibom-agentic-loop", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _run_batch(
    agent: Any,
    middleware: AIBOMScannerMiddleware,
    batch: list[AIComponent],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    batch_num: int,
    total_batches: int,
    all_components: list[AIComponent] | None = None,
    *,
    timeout_s: int = _DEFAULT_AGENTIC_TIMEOUT_S,
    dossier_index: DossierIndex | None = None,
) -> tuple[
    list[AIComponent],
    list[AIComponent],
    list[ComponentRelationship],
    list[RiskFlag],
    bool,
]:
    """Invoke the agent on a single batch and return parsed results."""
    from .tools import _reset_tool_stats, get_tool_stats

    _reset_tool_stats()
    _LOGGER.info(
        "Agentic batch %d/%d — %d components [%s]",
        batch_num,
        total_batches,
        len(batch),
        ", ".join(c.name for c in batch),
    )
    summary = _build_context_message(
        batch,
        relationships,
        scan_paths,
        all_components=all_components,
        dossier_index=dossier_index,
    )
    t0 = time.monotonic()

    result = None

    try:
        result = _invoke_with_deadline(agent, summary, timeout_s)
    except _BatchTimeout:
        elapsed = time.monotonic() - t0
        stats = get_tool_stats()
        _LOGGER.warning(
            "Batch %d timed out after %.1fs | tool_stats=%s",
            batch_num,
            elapsed,
            json.dumps(stats),
        )
        enriched = _degraded_batch_components(batch, hint="batch_timeout")
        return enriched, [], [], [], True
    except Exception as exc:
        elapsed = time.monotonic() - t0
        stats = get_tool_stats()
        _LOGGER.warning(
            "Batch %d failed after %.1fs: %s | tool_stats=%s",
            batch_num,
            elapsed,
            exc,
            json.dumps(stats),
        )
        data = None
        if result is not None:
            _accumulate_token_usage(result)
            data = _extract_structured_response(result)
        if not data:
            # langchain's StructuredOutputValidationError carries the offending
            # AIMessage; recover the answer from its carriers (tool-call args,
            # parsed object, or text) instead of dropping the whole batch.
            ai_message = getattr(exc, "ai_message", None)
            if ai_message is not None:
                recovered = {"messages": [ai_message]}
                _accumulate_token_usage(recovered)
                data = _extract_structured_response(recovered)
        if data:
            _LOGGER.info(
                "Batch %d: recovering partial results from failed run", batch_num
            )
            return _apply_batch_findings(middleware, batch, data, batch_num)
        hint = _classify_failure_hint(exc)
        enriched = _degraded_batch_components(batch, hint=hint)
        return enriched, [], [], [], True

    _accumulate_token_usage(result)
    elapsed = time.monotonic() - t0
    stats = get_tool_stats()
    total_tool_calls = sum(s["calls"] for s in stats.values())
    total_tool_time = sum(s["total_s"] for s in stats.values())

    _LOGGER.info(
        "Batch %d completed in %.1fs — %d tool calls (%.1fs tool time) | breakdown=%s",
        batch_num,
        elapsed,
        total_tool_calls,
        total_tool_time,
        json.dumps(stats),
    )

    data = _resolve_batch_data(agent, result)
    if not data:
        if _refusal_present(result):
            _LOGGER.warning("Batch %d: model refused", batch_num)
            return (
                _degraded_batch_components(batch, hint="model_refused"),
                [],
                [],
                [],
                True,
            )
        _LOGGER.warning("Batch %d returned no usable output", batch_num)
        return (
            _degraded_batch_components(batch, hint="no_usable_output"),
            [],
            [],
            [],
            True,
        )

    return _apply_batch_findings(middleware, batch, data, batch_num)


async def _run_batch_async(
    agent: Any,
    middleware: AIBOMScannerMiddleware,
    batch: list[AIComponent],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    batch_num: int,
    total_batches: int,
    all_components: list[AIComponent] | None = None,
    *,
    timeout_s: int = _DEFAULT_AGENTIC_TIMEOUT_S,
    dossier_index: DossierIndex | None = None,
) -> tuple[
    list[AIComponent],
    list[AIComponent],
    list[ComponentRelationship],
    list[RiskFlag],
    bool,
]:
    """Async version of _run_batch using agent.ainvoke()."""
    from .tools import _reset_tool_stats, get_tool_stats

    _reset_tool_stats()
    _LOGGER.info(
        "Agentic batch %d/%d — %d components [%s]",
        batch_num,
        total_batches,
        len(batch),
        ", ".join(c.name for c in batch),
    )
    summary = _build_context_message(
        batch,
        relationships,
        scan_paths,
        all_components=all_components,
        dossier_index=dossier_index,
    )
    t0 = time.monotonic()

    result = None
    try:
        # asyncio.timeout() (3.11+) is the recommended primitive over
        # wait_for: it cancels the awaited ainvoke at the coroutine level and
        # raises TimeoutError outside the block. Any blocking work the provider
        # stack offloaded to the loop executor is abandoned at teardown by
        # _run_async_bounded rather than joined.
        async with asyncio.timeout(timeout_s):
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": summary}]},
                config={"recursion_limit": _RECURSION_LIMIT},
            )
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        stats = get_tool_stats()
        _LOGGER.warning(
            "Batch %d timed out after %.1fs | tool_stats=%s",
            batch_num,
            elapsed,
            json.dumps(stats),
        )
        enriched = _degraded_batch_components(batch, hint="batch_timeout")
        return enriched, [], [], [], True
    except Exception as exc:
        elapsed = time.monotonic() - t0
        stats = get_tool_stats()
        _LOGGER.warning(
            "Batch %d failed after %.1fs: %s | tool_stats=%s",
            batch_num,
            elapsed,
            exc,
            json.dumps(stats),
        )
        data = None
        if result is not None:
            _accumulate_token_usage(result)
            data = _extract_structured_response(result)
        if not data:
            # langchain's StructuredOutputValidationError carries the offending
            # AIMessage; recover the answer from its carriers (tool-call args,
            # parsed object, or text) instead of dropping the whole batch.
            ai_message = getattr(exc, "ai_message", None)
            if ai_message is not None:
                recovered = {"messages": [ai_message]}
                _accumulate_token_usage(recovered)
                data = _extract_structured_response(recovered)
        if data:
            _LOGGER.info(
                "Batch %d: recovering partial results from failed run", batch_num
            )
            return _apply_batch_findings(middleware, batch, data, batch_num)
        hint = _classify_failure_hint(exc)
        enriched = _degraded_batch_components(batch, hint=hint)
        return enriched, [], [], [], True

    _accumulate_token_usage(result)
    elapsed = time.monotonic() - t0
    stats = get_tool_stats()
    total_tool_calls = sum(s["calls"] for s in stats.values())
    total_tool_time = sum(s["total_s"] for s in stats.values())

    _LOGGER.info(
        "Batch %d completed in %.1fs — %d tool calls (%.1fs tool time) | breakdown=%s",
        batch_num,
        elapsed,
        total_tool_calls,
        total_tool_time,
        json.dumps(stats),
    )

    data = _resolve_batch_data(agent, result)
    if not data:
        if _refusal_present(result):
            _LOGGER.warning("Batch %d: model refused", batch_num)
            return (
                _degraded_batch_components(batch, hint="model_refused"),
                [],
                [],
                [],
                True,
            )
        _LOGGER.warning("Batch %d returned no usable output", batch_num)
        return (
            _degraded_batch_components(batch, hint="no_usable_output"),
            [],
            [],
            [],
            True,
        )

    return _apply_batch_findings(middleware, batch, data, batch_num)


async def _run_batches_parallel(
    agent: Any,
    middleware: AIBOMScannerMiddleware,
    batches: list[list[AIComponent]],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    all_components: list[AIComponent] | None = None,
    max_concurrent: int = 1,
    *,
    timeout_s: int = _DEFAULT_AGENTIC_TIMEOUT_S,
    max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES,
    dossier_index: DossierIndex | None = None,
) -> tuple[
    list[AIComponent],
    list[AIComponent],
    list[ComponentRelationship],
    list[RiskFlag],
    list[_BatchArtifact],
]:
    """Run locality-aware batches with a concurrency-limited semaphore.

    Up to *max_concurrent* batches are in-flight simultaneously.  A shared
    circuit breaker cancels remaining work after *max_consecutive_failures*
    sequential failures (checked per-window, not globally across in-flight
    batches, to avoid race conditions).
    """
    sem = asyncio.Semaphore(max_concurrent)
    tripped = asyncio.Event()
    total = len(batches)

    _BatchResult = tuple[
        int,  # idx (1-based)
        list[AIComponent],  # input batch
        list[AIComponent],
        list[AIComponent],
        list[ComponentRelationship],
        list[RiskFlag],
        bool,  # failed
    ]

    async def _guarded(idx: int, batch: list[AIComponent]) -> _BatchResult:
        if tripped.is_set():
            return idx, batch, _circuit_breaker_skipped_batch(batch), [], [], [], True
        async with sem:
            if tripped.is_set():
                return (
                    idx,
                    batch,
                    _circuit_breaker_skipped_batch(batch),
                    [],
                    [],
                    [],
                    True,
                )
            try:
                enriched, new, rels, flags, failed = await _run_batch_async(
                    agent,
                    middleware,
                    batch,
                    relationships,
                    scan_paths,
                    idx,
                    total,
                    all_components=all_components,
                    timeout_s=timeout_s,
                    dossier_index=dossier_index,
                )
                return idx, batch, enriched, new, rels, flags, failed
            except Exception as exc:
                _LOGGER.warning("Parallel batch %d raised: %s", idx, exc)
                return idx, batch, list(batch), [], [], [], True

    tasks = [
        asyncio.create_task(_guarded(idx, batch))
        for idx, batch in enumerate(batches, 1)
    ]

    results: list[_BatchResult] = []
    consecutive_failures = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        results.append(r)
        _, _, _, _, _, _, failed = r
        if failed:
            consecutive_failures += 1
        else:
            consecutive_failures = 0
        if consecutive_failures >= max_consecutive_failures and not tripped.is_set():
            _LOGGER.warning(
                "Agentic circuit breaker tripped after %d consecutive failures — "
                "cancelling remaining batches",
                max_consecutive_failures,
            )
            tripped.set()

    results.sort(key=lambda r: r[0])

    all_enriched: list[AIComponent] = []
    all_new: list[AIComponent] = []
    all_rels: list[ComponentRelationship] = []
    all_flags: list[RiskFlag] = []
    artifacts: list[_BatchArtifact] = []
    for _, batch, enriched, new, rels, flags, _ in results:
        all_enriched.extend(enriched)
        all_new.extend(new)
        all_rels.extend(rels)
        all_flags.extend(flags)
        artifacts.append(_BatchArtifact(batch, new, rels, flags))

    return all_enriched, all_new, all_rels, all_flags, artifacts


def _default_agentic_cache_dir() -> Path | None:
    """Return the default on-disk cache directory, or None if unavailable."""
    try:
        return ensure_cache_dir("agentic")
    except OSError:
        return None


def _strategy_fallback_pass(
    fallback_agent: Any,
    middleware: AIBOMScannerMiddleware,
    enriched: list[AIComponent],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    all_components: list[AIComponent] | None,
    *,
    batch_size: int,
    timeout_s: int,
    max_consecutive_failures: int,
    dossier_index: DossierIndex | None = None,
) -> tuple[
    list[AIComponent],
    list[AIComponent],
    list[ComponentRelationship],
    list[RiskFlag],
    list[_BatchArtifact],
]:
    """Re-run components that produced no usable output under the primary
    structured-output strategy, using an alternate-strategy *fallback_agent*.

    Models differ in which structured-output method works (some emit a usable
    tool call, others only the provider-native json_schema object); this gives
    the failed components a second chance via the other method. Returns the
    (possibly updated) ``enriched`` list plus any new components / relationships
    / risk flags and batch artifacts produced by the fallback run.
    """
    targets = [c for c in enriched if c.agentic_hint in _STRATEGY_FALLBACK_HINTS]
    if not targets:
        return enriched, [], [], [], []

    _LOGGER.info(
        "Structured-output fallback: retrying %d component(s) via alternate strategy",
        len(targets),
    )
    fb_inputs = [
        c.model_copy(update={"needs_agentic": True, "agentic_hint": ""})
        for c in targets
    ]
    fb_batches = _locality_aware_batches(fb_inputs, batch_size)
    recovered: dict[str, AIComponent] = {}
    removed_ids: set[str] = set()
    fb_new: list[AIComponent] = []
    fb_rels: list[ComponentRelationship] = []
    fb_flags: list[RiskFlag] = []
    fb_artifacts: list[_BatchArtifact] = []
    consecutive = 0
    for idx, batch in enumerate(fb_batches, 1):
        if consecutive >= max_consecutive_failures:
            _LOGGER.warning(
                "Structured-output fallback circuit breaker: skipping batches %d-%d",
                idx,
                len(fb_batches),
            )
            break
        e, n, r, f, batch_failed = _run_batch(
            fallback_agent,
            middleware,
            batch,
            relationships,
            scan_paths,
            idx,
            len(fb_batches),
            all_components=all_components,
            timeout_s=timeout_s,
            dossier_index=dossier_index,
        )
        for c in e:
            recovered[c.instance_id] = c
        if not batch_failed:
            # A successful fallback can remove components (middleware omits them
            # from `e`); track those so the merge drops them instead of keeping
            # the stale degraded original.
            returned_ids = {c.instance_id for c in e}
            removed_ids.update(
                c.instance_id for c in batch if c.instance_id not in returned_ids
            )
        fb_new.extend(n)
        fb_rels.extend(r)
        fb_flags.extend(f)
        fb_artifacts.append(_BatchArtifact(batch, n, r, f))
        consecutive = consecutive + 1 if batch_failed else 0

    if recovered or removed_ids:
        n_ok = sum(1 for c in recovered.values() if not c.agentic_hint)
        _LOGGER.info(
            "Structured-output fallback: %d/%d component(s) recovered, %d removed",
            n_ok,
            len(targets),
            len(removed_ids),
        )
        merged: list[AIComponent] = []
        for c in enriched:
            if c.instance_id in recovered:
                merged.append(recovered[c.instance_id])
            elif c.instance_id in removed_ids:
                continue  # removed by a successful fallback run
            else:
                merged.append(c)
        enriched = merged
    return enriched, fb_new, fb_rels, fb_flags, fb_artifacts


def _run_tier(
    agent: Any,
    middleware: AIBOMScannerMiddleware,
    components: list[AIComponent],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    batch_size: int,
    max_concurrent: int,
    all_components: list[AIComponent] | None,
    cache: _AgenticResultCache | None,
    *,
    memo: _DecisionMemo | None = None,
    timeout_s: int = _DEFAULT_AGENTIC_TIMEOUT_S,
    max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES,
    retry_deadline: float | None = None,
    dossier_index: DossierIndex | None = None,
    fallback_agent_factory: Any | None = None,
) -> tuple[
    list[AIComponent], list[AIComponent], list[ComponentRelationship], list[RiskFlag]
]:
    """Execute a single tier (simple or complex) with caching and parallel batches."""
    tier_enriched: list[AIComponent] = []
    tier_new: list[AIComponent] = []
    tier_rels: list[ComponentRelationship] = []
    tier_flags: list[RiskFlag] = []
    tier_cache_key = _tier_cache_key(components) if cache else None

    if cache and tier_cache_key is not None:
        cached_tier = _load_tier_cache_payload(cache.get(tier_cache_key))
        if cached_tier is not None:
            _record_memo_verdicts(memo, components, cached_tier[0])
            _LOGGER.info(
                "Tier cache hit for %d components — skipping LLM",
                len(components),
            )
            return cached_tier

    to_send = components
    if cache:
        cached_comps, to_send = cache.partition(components)
        if cached_comps:
            _LOGGER.info(
                "Cache hit for %d/%d components — skipping LLM",
                len(cached_comps),
                len(components),
            )
            e, n, r, f = cache.apply_cached(cached_comps, middleware)
            tier_enriched.extend(e)
            tier_new.extend(n)
            tier_rels.extend(r)
            tier_flags.extend(f)

    if memo is not None and to_send:
        memo_hits, to_send = memo.partition(to_send)
        if memo_hits:
            _LOGGER.info(
                "Memo hit for %d/%d components — reusing earlier verdicts",
                len(memo_hits),
                len(memo_hits) + len(to_send),
            )
            memo_results = memo.apply(memo_hits)
            tier_enriched.extend(memo_results)

    if not to_send:
        _record_memo_verdicts(memo, components, tier_enriched)
        if cache and tier_cache_key is not None:
            cache.put(
                tier_cache_key,
                _build_tier_cache_payload(
                    tier_enriched, tier_new, tier_rels, tier_flags
                ),
            )
        return tier_enriched, tier_new, tier_rels, tier_flags

    batches = _locality_aware_batches(to_send, batch_size)
    _LOGGER.info(
        "%d components → %d locality-aware batches (concurrency=%d)",
        len(to_send),
        len(batches),
        max_concurrent,
    )

    if max_concurrent > 1 and len(batches) > 1:
        enriched, new, rels, flags, batch_artifacts = _run_async_bounded(
            _run_batches_parallel(
                agent,
                middleware,
                batches,
                relationships,
                scan_paths,
                all_components=all_components,
                max_concurrent=max_concurrent,
                timeout_s=timeout_s,
                max_consecutive_failures=max_consecutive_failures,
                dossier_index=dossier_index,
            )
        )
    else:
        enriched: list[AIComponent] = []
        new: list[AIComponent] = []
        rels: list[ComponentRelationship] = []
        flags: list[RiskFlag] = []
        batch_artifacts: list[_BatchArtifact] = []
        consecutive_failures = 0
        for idx, batch in enumerate(batches, 1):
            if consecutive_failures >= max_consecutive_failures:
                _LOGGER.warning(
                    "Agentic circuit breaker: skipping batches %d–%d after "
                    "%d consecutive failures",
                    idx,
                    len(batches),
                    max_consecutive_failures,
                )
                for b in batches[idx - 1 :]:
                    enriched.extend(_circuit_breaker_skipped_batch(b))
                break
            e, n, r, f, batch_failed = _run_batch(
                agent,
                middleware,
                batch,
                relationships,
                scan_paths,
                idx,
                len(batches),
                all_components=all_components,
                timeout_s=timeout_s,
                dossier_index=dossier_index,
            )
            enriched.extend(e)
            new.extend(n)
            rels.extend(r)
            flags.extend(f)
            batch_artifacts.append(_BatchArtifact(batch, n, r, f))
            if batch_failed:
                consecutive_failures += 1
            else:
                consecutive_failures = 0

    ok, retry_candidates = _collect_failed(enriched)
    if retry_candidates:
        _LOGGER.info(
            "Retry pass: %d degraded components, cooling down %ds",
            len(retry_candidates),
            _RETRY_COOLDOWN_S,
        )
        time.sleep(_RETRY_COOLDOWN_S)

        retry_batch_size = batch_size
        has_recursion_failures = any(
            c_orig.agentic_hint == "batch_recursion_limit"
            for c_orig in enriched
            if c_orig.agentic_hint in _RETRYABLE_HINTS
        )
        if has_recursion_failures:
            retry_batch_size = max(1, batch_size // 2)

        retry_batches = _locality_aware_batches(retry_candidates, retry_batch_size)
        _LOGGER.info(
            "Retrying %d batches sequentially (batch_size=%d)",
            len(retry_batches),
            retry_batch_size,
        )

        retry_enriched: list[AIComponent] = []
        retry_new: list[AIComponent] = []
        retry_rels: list[ComponentRelationship] = []
        retry_flags: list[RiskFlag] = []
        retry_consecutive = 0
        for idx, batch in enumerate(retry_batches, 1):
            if retry_deadline is not None and time.monotonic() >= retry_deadline:
                remaining = sum(len(b) for b in retry_batches[idx - 1 :])
                _LOGGER.warning(
                    "Aborting retries after budget exhausted; "
                    "%d component(s) left degraded",
                    remaining,
                )
                for b in retry_batches[idx - 1 :]:
                    retry_enriched.extend(
                        _degraded_batch_components(b, hint="retry_budget_exhausted")
                    )
                break
            if retry_consecutive >= max_consecutive_failures:
                _LOGGER.warning(
                    "Retry circuit breaker: skipping retry batches %d–%d",
                    idx,
                    len(retry_batches),
                )
                for b in retry_batches[idx - 1 :]:
                    retry_enriched.extend(
                        _degraded_batch_components(b, hint="retry_failed")
                    )
                break
            e, n, r, f, batch_failed = _run_batch(
                agent,
                middleware,
                batch,
                relationships,
                scan_paths,
                idx,
                len(retry_batches),
                all_components=all_components,
                timeout_s=timeout_s,
                dossier_index=dossier_index,
            )
            retry_enriched.extend(e)
            retry_new.extend(n)
            retry_rels.extend(r)
            retry_flags.extend(f)
            batch_artifacts.append(_BatchArtifact(batch, n, r, f))
            if batch_failed:
                retry_consecutive += 1
            else:
                retry_consecutive = 0

        # A component is genuinely recovered only when it comes back with no
        # failure hint. Degraded terminal states (retry_failed,
        # retry_budget_exhausted) or a still-set retryable hint are NOT
        # recovered, so they are not counted here.
        recovered = sum(1 for c in retry_enriched if not c.agentic_hint)
        still_degraded = len(retry_candidates) - recovered
        _LOGGER.info(
            "Retry pass complete: %d/%d recovered, %d still degraded",
            recovered,
            len(retry_candidates),
            still_degraded,
        )

        enriched = ok + retry_enriched
        new.extend(retry_new)
        rels.extend(retry_rels)
        flags.extend(retry_flags)

    # Structured-output strategy fallback: components that yielded no usable
    # output under the primary strategy get one re-run via the alternate
    # (provider-native) strategy. Only fires when such failures exist, so the
    # happy path makes no extra LLM call. Skipped once the retry budget is
    # exhausted so it cannot extend a runaway.
    budget_left = retry_deadline is None or time.monotonic() < retry_deadline
    if (
        fallback_agent_factory is not None
        and budget_left
        and any(c.agentic_hint in _STRATEGY_FALLBACK_HINTS for c in enriched)
    ):
        fb_agent = fallback_agent_factory()
        if fb_agent is not None:
            enriched, fb_new, fb_rels, fb_flags, fb_artifacts = _strategy_fallback_pass(
                fb_agent,
                middleware,
                enriched,
                relationships,
                scan_paths,
                all_components,
                batch_size=batch_size,
                timeout_s=timeout_s,
                max_consecutive_failures=max_consecutive_failures,
                dossier_index=dossier_index,
            )
            new.extend(fb_new)
            rels.extend(fb_rels)
            flags.extend(fb_flags)
            batch_artifacts.extend(fb_artifacts)

    if cache:
        artifact_key_by_instance_id: dict[str, str] = {}
        for artifact in batch_artifacts:
            if not (artifact.new or artifact.rels or artifact.flags):
                continue
            batch_key = _batch_cache_key(artifact.inputs)
            cache.put(
                batch_key,
                _build_batch_cache_payload(artifact.new, artifact.rels, artifact.flags),
            )
            for c_in in artifact.inputs:
                artifact_key_by_instance_id[c_in.instance_id] = batch_key

        enriched_by_id = {c.instance_id: c for c in enriched}
        all_batch_components = [c for batch in batches for c in batch]
        for c_before in all_batch_components:
            key = _component_cache_key(c_before)
            c_after = enriched_by_id.get(c_before.instance_id)
            if c_after is None:
                entry: dict[str, Any] = {
                    "enriched_components": [],
                    "new_components": [],
                    "remove_components": [
                        {
                            "instance_id": c_before.instance_id,
                            "reason": "cached_removal",
                        }
                    ],
                    "reclassify_components": [],
                    "new_relationships": [],
                    "risk_findings": [],
                }
                batch_key = artifact_key_by_instance_id.get(c_before.instance_id)
                if batch_key:
                    entry["batch_artifact_key"] = batch_key
                cache.put(key, entry)
            else:
                entry: dict[str, Any] = {
                    "cached_component": c_after.model_dump(mode="json"),
                    "enriched_components": [],
                    "new_components": [],
                    "remove_components": [],
                    "reclassify_components": [],
                    "new_relationships": [],
                    "risk_findings": [],
                }
                batch_key = artifact_key_by_instance_id.get(c_before.instance_id)
                if batch_key:
                    entry["batch_artifact_key"] = batch_key
                cache.put(key, entry)

    tier_enriched.extend(enriched)
    tier_new.extend(new)
    tier_rels.extend(rels)
    tier_flags.extend(flags)
    _record_memo_verdicts(memo, components, tier_enriched)
    if cache and tier_cache_key is not None:
        cache.put(
            tier_cache_key,
            _build_tier_cache_payload(tier_enriched, tier_new, tier_rels, tier_flags),
        )
    return tier_enriched, tier_new, tier_rels, tier_flags


def _group_by_top_dir(
    components: list[AIComponent],
    scan_paths: list[str],
) -> dict[str, list[AIComponent]]:
    """Group components by their nearest scan root for sub-agent dispatch."""
    from collections import defaultdict

    resolved_roots = [str(Path(p).resolve()) for p in scan_paths]
    groups: dict[str, list[AIComponent]] = defaultdict(list)

    for c in components:
        fp = str(Path(c.file_path).resolve()) if c.file_path else ""
        matched_root = "__default__"
        for root in resolved_roots:
            if fp.startswith(root):
                matched_root = root
                break
        groups[matched_root].append(c)

    return dict(groups)


def run_agentic_enrichment(
    model_string: str,
    deterministic_components: list[AIComponent],
    deterministic_relationships: list[ComponentRelationship],
    scan_paths: list[str],
    llm_config: dict[str, Any] | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    max_concurrent: int = 1,
    fast_model: str | None = None,
    timeout_s: int = _DEFAULT_AGENTIC_TIMEOUT_S,
    max_consecutive_failures: int = _DEFAULT_MAX_CONSECUTIVE_FAILURES,
    max_retry_seconds: int = _DEFAULT_MAX_RETRY_SECONDS,
    cache_dir: Path | None = None,
    include_code_snippets: bool = False,
    agent_signature_catalog: AgentSignatureCatalog | None = None,
) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag], TokenUsage]:
    """Run the full agentic enrichment pipeline.

    Features:
      - Locality-aware batching (groups co-located components)
      - Configurable parallel batches via *max_concurrent*
      - Content-hash result caching across re-runs
      - Sub-agent dispatch for large repos (>50 candidates per scan root)
      - Agent-evidence dossier injection: every ENRICH target that is an
        AGENT / AGENT_PROXY / MCP_SERVER / MCP_CLIENT candidate is shown the
        verbatim class body plus a CST-derived evidence dossier

    Parameters
    ----------
    model_string:
        Primary LLM model (used for complex candidates).
    deterministic_components:
        Components from the deterministic pipeline.
    deterministic_relationships:
        Relationships from the deterministic pipeline.
    scan_paths:
        Paths being scanned.
    llm_config:
        LLM connection config (api_key, api_base, etc.).
    batch_size:
        Max components per agent invocation (default 5).
    max_concurrent:
        Max parallel agentic LLM batches (default 1 = sequential).
    fast_model:
        Optional cheaper/faster model for simple confirmations.
        Falls back to ``model_string`` if not provided.
    cache_dir:
        Optional on-disk cache directory override. When not provided, the
        default agentic cache location is used.
    agent_signature_catalog:
        Optional merged agent-signature catalog (built-ins + user
        overrides). When omitted, :func:`aibom.agent_signatures.resolve_catalog`
        is used to pick up built-in defaults.

    Returns
    -------
    Tuple of (enriched_components, new_relationships, risk_flags, token_usage).
    """
    from .tools import set_allowed_search_roots

    set_allowed_search_roots([str(Path(p).resolve()) for p in scan_paths])
    _reset_token_usage()

    # Shared retry deadline for the whole run: bounds total retry wall-clock
    # across every tier / sub-agent / strategy-fallback pass.
    retry_deadline = time.monotonic() + max_retry_seconds

    simple, complex_ = _classify_candidates(deterministic_components)

    _LOGGER.info(
        "Running agentic enrichment with %s (%d simple, %d complex, %d relationships, concurrency=%d)",
        model_string,
        len(simple),
        len(complex_),
        len(deterministic_relationships),
        max_concurrent,
    )

    dossier_index: DossierIndex = build_dossier_index(
        deterministic_components,
        catalog=agent_signature_catalog,
    )
    if dossier_index:
        _LOGGER.info(
            "Built agent-evidence dossier index with %d class entries",
            len(dossier_index),
        )

    resolved_cache_dir = (
        cache_dir if cache_dir is not None else _default_agentic_cache_dir()
    )
    fallback_dirs: list[Path] = []
    if cache_dir is None and resolved_cache_dir is not None:
        fallback_dirs = [
            p for p in cache_read_dirs("agentic") if p != resolved_cache_dir
        ]
    cache = _AgenticResultCache(resolved_cache_dir, fallback_dirs=fallback_dirs)
    middleware = AIBOMScannerMiddleware(
        include_code_snippets=include_code_snippets,
        allowed_roots=scan_paths,
    )
    memo = _DecisionMemo()

    all_enriched: list[AIComponent] = []
    all_new: list[AIComponent] = []
    all_rels: list[ComponentRelationship] = []
    all_flags: list[RiskFlag] = []

    tier_model_name = fast_model or model_string
    simple_batch_size = batch_size

    tier_model_obj = _build_model(tier_model_name, llm_config)
    if model_string == tier_model_name:
        complex_model_obj = tier_model_obj
        models_to_close = [tier_model_obj]
    else:
        complex_model_obj = _build_model(model_string, llm_config)
        models_to_close = [tier_model_obj, complex_model_obj]

    def _simple_fb_factory(ms=tier_model_name, mo=tier_model_obj):
        return _build_fallback_agent(ms, mo)

    def _complex_fb_factory(ms=model_string, mo=complex_model_obj):
        return _build_fallback_agent(ms, mo)

    try:
        if simple:
            _LOGGER.info(
                "Tier 1 (simple confirmations): %d candidates via %s (batch=%d)",
                len(simple),
                tier_model_name,
                simple_batch_size,
            )
            agent = create_aibom_agent(
                tier_model_name,
                model=tier_model_obj,
                response_format=_agent_response_format(tier_model_name, llm_config),
            )
            e, n, r, f = _run_tier(
                agent,
                middleware,
                simple,
                deterministic_relationships,
                scan_paths,
                simple_batch_size,
                max_concurrent,
                deterministic_components,
                cache,
                memo=memo,
                timeout_s=timeout_s,
                max_consecutive_failures=max_consecutive_failures,
                retry_deadline=retry_deadline,
                dossier_index=dossier_index,
                fallback_agent_factory=_simple_fb_factory,
            )
            all_enriched.extend(e)
            all_new.extend(n)
            all_rels.extend(r)
            all_flags.extend(f)

        if complex_:
            dir_groups = _group_by_top_dir(complex_, scan_paths)
            use_sub_agents = (
                len(dir_groups) > 1 and len(complex_) > _SUB_AGENT_THRESHOLD
            )

            if use_sub_agents:
                _LOGGER.info(
                    "Sub-agent dispatch: %d directory groups for %d complex candidates",
                    len(dir_groups),
                    len(complex_),
                )
                for dir_key, group in sorted(dir_groups.items()):
                    dir_label = (
                        Path(dir_key).name if dir_key != "__default__" else "default"
                    )
                    _LOGGER.info(
                        "Sub-agent [%s]: %d candidates via %s",
                        dir_label,
                        len(group),
                        model_string,
                    )
                    agent = create_aibom_agent(
                        model_string,
                        model=complex_model_obj,
                        response_format=_agent_response_format(
                            model_string, llm_config
                        ),
                    )
                    e, n, r, f = _run_tier(
                        agent,
                        middleware,
                        group,
                        deterministic_relationships,
                        scan_paths,
                        batch_size,
                        max_concurrent,
                        deterministic_components,
                        cache,
                        memo=memo,
                        timeout_s=timeout_s,
                        max_consecutive_failures=max_consecutive_failures,
                        retry_deadline=retry_deadline,
                        dossier_index=dossier_index,
                        fallback_agent_factory=_complex_fb_factory,
                    )
                    all_enriched.extend(e)
                    all_new.extend(n)
                    all_rels.extend(r)
                    all_flags.extend(f)
            else:
                _LOGGER.info(
                    "Tier 2 (complex reasoning): %d candidates via %s",
                    len(complex_),
                    model_string,
                )
                agent = create_aibom_agent(
                    model_string,
                    model=complex_model_obj,
                    response_format=_agent_response_format(model_string, llm_config),
                )
                e, n, r, f = _run_tier(
                    agent,
                    middleware,
                    complex_,
                    deterministic_relationships,
                    scan_paths,
                    batch_size,
                    max_concurrent,
                    deterministic_components,
                    cache,
                    memo=memo,
                    timeout_s=timeout_s,
                    max_consecutive_failures=max_consecutive_failures,
                    retry_deadline=retry_deadline,
                    dossier_index=dossier_index,
                    fallback_agent_factory=_complex_fb_factory,
                )
                all_enriched.extend(e)
                all_new.extend(n)
                all_rels.extend(r)
                all_flags.extend(f)
    finally:
        _close_model_clients(*models_to_close)

    all_components = all_enriched + all_new

    usage = get_token_usage()
    if _all_batches_failed(all_enriched, all_new, all_rels, all_flags):
        from collections import Counter

        hints = Counter(c.agentic_hint for c in all_enriched if c.agentic_hint)
        dominant = hints.most_common(1)[0][0] if hints else "unknown"
        _LOGGER.warning(
            "Agentic enrichment DEGRADED: all %d components failed enrichment "
            "(dominant hint: %s); the LLM added nothing, so the BOM is "
            "deterministic-only. Check the provider/model configuration "
            "(credentials, model access, request params). "
            "tokens=%d (prompt=%d, completion=%d)",
            len(all_enriched),
            dominant,
            usage.total_tokens,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
    else:
        _LOGGER.info(
            "Agentic enrichment complete: %d enriched, %d new components, "
            "%d new relationships, %d risk flags, "
            "tokens=%d (prompt=%d, completion=%d)",
            len(all_enriched),
            len(all_new),
            len(all_rels),
            len(all_flags),
            usage.total_tokens,
            usage.prompt_tokens,
            usage.completion_tokens,
        )
        degraded = _count_degraded(all_enriched)
        if degraded:
            dominant = _dominant_degraded_hint(all_enriched)
            remedy = (
                " Consider lowering --agentic-concurrency / --agentic-rate-limit "
                "(or raising --agentic-timeout) if your provider is overloaded."
                if dominant in _DEGRADED_LOAD_HINTS
                else ""
            )
            _LOGGER.warning(
                "%d component(s) left degraded after enrichment "
                "(dominant hint: %s); the BOM may be incomplete for those "
                "components.%s",
                degraded,
                dominant,
                remedy,
            )

    return all_components, all_rels, all_flags, usage


_CODE_CONTEXT_RADIUS = 15


def _read_code_window(file_path: str, line: int) -> str | None:
    """Read a window of source code around *line* (±15 lines)."""
    from pathlib import Path

    p = Path(file_path)
    if not p.is_file():
        return None
    try:
        lines = p.read_text(errors="replace").splitlines()
    except OSError:
        return None
    start = max(0, line - _CODE_CONTEXT_RADIUS - 1)
    end = min(len(lines), line + _CODE_CONTEXT_RADIUS)
    numbered = [f"{i + 1:>5}| {lines[i]}" for i in range(start, end)]
    return "\n".join(numbered)


_MAX_CLASS_BODY_CHARS = 8000


def _truncate_class_body(body: str) -> tuple[str, bool]:
    """Cap *body* at :data:`_MAX_CLASS_BODY_CHARS`. Returns (text, truncated?)."""
    if len(body) <= _MAX_CLASS_BODY_CHARS:
        return body, False
    head = body[:_MAX_CLASS_BODY_CHARS]
    return head, True


def _component_to_summary(
    c: AIComponent,
    *,
    include_code: bool = False,
    enrich_target: bool = False,
    dossier_index: "DossierIndex | None" = None,
) -> dict[str, Any]:
    """Serialize a single component for the agent prompt.

    When ``dossier_index`` is provided and the component is an
    ENRICH target that covers a class captured by the evidence
    builder, the returned summary also contains:

    * ``class_body_source`` — the verbatim class body (truncated to
      :data:`_MAX_CLASS_BODY_CHARS` characters).
    * ``agent_evidence_dossier`` — the structured matches produced by
      :mod:`aibom.scanners.agent_evidence_builder`.
    """
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
    if c.agentic_hint:
        entry["agentic_hint"] = c.agentic_hint
    if enrich_target:
        entry["ENRICH"] = True

    if include_code and c.file_path and c.line_number:
        snippet = _read_code_window(c.file_path, c.line_number)
        if snippet:
            entry["code_context"] = snippet

    if enrich_target and dossier_index:
        from ..scanners.agent_evidence_builder import render_dossier_for_prompt
        from .evidence_injection import lookup_dossier

        dossier = lookup_dossier(c, dossier_index)
        if dossier is not None:
            if dossier.class_body_source:
                body, truncated = _truncate_class_body(dossier.class_body_source)
                entry["class_body_source"] = body
                if truncated:
                    entry["class_body_truncated"] = True
            entry["agent_evidence_dossier"] = render_dossier_for_prompt(dossier)

    return entry


def _build_context_message(
    batch: list[AIComponent],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    all_components: list[AIComponent] | None = None,
    dossier_index: "DossierIndex | None" = None,
) -> str:
    """Build the user message that seeds the agent with deterministic results.

    *batch* — components the agent must enrich (with code context and,
    when available, the CST-derived agent evidence dossier).
    *all_components* — full scan results for situational awareness (without
    code context, to keep the prompt compact).
    *dossier_index* — optional map from ``(file_path, class_start_line)`` to
    :class:`AgentEvidenceDossier`; produced by
    :func:`aibom.agentic.evidence_injection.build_dossier_index`.
    """
    batch_ids = {c.instance_id for c in batch}

    enrich_summaries = [
        _component_to_summary(
            c,
            include_code=True,
            enrich_target=True,
            dossier_index=dossier_index,
        )
        for c in batch
    ]

    context_summaries: list[dict[str, Any]] = []
    if all_components:
        context_summaries = [
            _component_to_summary(c)
            for c in all_components
            if c.instance_id not in batch_ids
        ]

    rel_summaries = [
        {
            "source": r.source_name,
            "target": r.target_name,
            "type": r.relationship_type.value,
        }
        for r in relationships
    ]

    context: dict[str, Any] = {
        "scan_paths": scan_paths,
        "enrich_these": enrich_summaries,
        "other_detected_components": context_summaries,
        "relationships": rel_summaries,
    }

    return (
        "Below are the deterministic scan results. Components in "
        "`enrich_these` (marked ENRICH=true) need your analysis. Each "
        "includes a `code_context` window; agent / MCP / agent_proxy "
        "candidates also include `class_body_source` (verbatim class body) "
        "and `agent_evidence_dossier` (structured CST-derived evidence). "
        "`other_detected_components` shows everything else already found; "
        "use it to discover relationships and missing components but do "
        "NOT re-enrich those.\n\n"
        f"```json\n{json.dumps(context, indent=2)}\n```"
    )


def _structured_from_tool_calls(message: Any) -> dict[str, Any] | None:
    """Return the structured object from a message's tool-call args, if any.

    The ``function_calling`` structured-output method carries the answer as a
    dict in ``tool_calls[].args``. This is uniform across LangChain providers
    (OpenAI, Anthropic, Bedrock, Google), so reading it is provider-agnostic.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    response_fields = set(AgentResponse.model_fields)
    for call in reversed(tool_calls):
        if isinstance(call, dict):
            args = call.get("args")
        else:
            args = getattr(call, "args", None)
        # Only treat a tool call as the structured response when its args carry
        # AgentResponse fields. A normal tool invocation (search_codebase,
        # lookup_model, ...) must not be misread as the agent's response.
        if isinstance(args, dict) and (response_fields & args.keys()):
            return args
    return None


def _structured_from_parsed(message: Any) -> dict[str, Any] | None:
    """Return the parsed object from ``additional_kwargs['parsed']``, if any.

    The ``json_schema`` structured-output method (ChatOpenAI's default)
    deposits the parsed object here while leaving ``content`` empty.
    """
    extra = getattr(message, "additional_kwargs", None)
    if not isinstance(extra, dict):
        return None
    parsed = extra.get("parsed")
    if isinstance(parsed, BaseModel):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return parsed
    return None


def _message_text(message: Any) -> str:
    """Return the textual content of a message.

    Handles both string content and list-form content blocks
    (thinking/tool_use/text) returned by Anthropic, Bedrock, and Gemini by
    concatenating only the ``text`` blocks — never ``str()``-coercing the whole
    list, which would yield an unparseable Python repr.
    """
    if hasattr(message, "content"):
        content = message.content
    elif isinstance(message, dict):
        content = message.get("content", "")
    else:
        return str(message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _dedouble_candidates(text: str) -> list[str]:
    """Return candidate repairs for content corrupted by gateway character-
    doubling ("hheelllloo" -> "hello").

    The caller re-validates each candidate with ``json.loads`` and discards any
    that does not parse to an object, so an inexact heuristic can never make
    things worse. Token/word-level collapse is intentionally NOT attempted: it
    would silently corrupt a legitimate repeated substring inside a JSON string
    (e.g. a real value "abcabc" -> "abc") while still parsing. Returns an empty
    list when no repair applies.
    """
    if len(text) >= 2 and len(text) % 2 == 0 and text[0::2] == text[1::2]:
        return [text[0::2]]
    return []


def _extract_structured_response(result: Any) -> dict[str, Any] | None:
    """Extract the structured response from the agent's final state.

    When ``response_format`` is provided, Deep Agents populates
    ``structured_response`` in the graph state.  When it does not — because the
    model emitted the answer via a tool call, a parsed object, or list-form
    content blocks instead of a JSON string — recover it from the last
    message's carriers in order of reliability before falling back to JSON
    parsing of the message text.
    """
    sr = result.get("structured_response")
    if sr is not None:
        if isinstance(sr, BaseModel):
            return sr.model_dump()
        if isinstance(sr, dict):
            return sr

    messages = result.get("messages", [])
    if not messages:
        return None
    last = messages[-1]

    # Prefer the already-parsed carriers (no JSON parsing needed): tool-call
    # args (function_calling), then a parsed object (json_schema).
    from_tools = _structured_from_tool_calls(last)
    if from_tools is not None:
        return from_tools
    from_parsed = _structured_from_parsed(last)
    if from_parsed is not None:
        return from_parsed

    # Fall back to JSON embedded in the message text, concatenating text blocks
    # for list-form content rather than str()-coercing the whole list.
    content = _message_text(last).strip()
    if not content:
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    # Only a JSON object is a usable response; arrays/scalars would pass a
    # truthy check and then break middleware's ``data.get(...)``.
    if isinstance(parsed, dict):
        return parsed
    if parsed is None:
        # Best-effort: some OpenAI-compatible gateways echo content doubled
        # (character- or token/word-level). Re-validate each candidate via
        # json.loads and accept only one that parses to an object, so we never
        # accept a worse result.
        for candidate in _dedouble_candidates(content):
            try:
                redone = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(redone, dict):
                _LOGGER.info(
                    "Recovered structured output after de-doubling "
                    "gateway-corrupted content"
                )
                return redone
    _LOGGER.warning(
        "Failed to parse agent JSON output — first 300 chars: %s",
        content[:300],
    )
    return None


_CROSS_REPO_COORDINATOR_PROMPT = """\
You are an AI-BOM cross-repository coordinator. You receive an orientation
summary from multiple repository scans and must identify cross-repo
relationships that individual per-repo scans could not resolve.

## Tools

- **get_repo_components(repo_name)** — Returns full component data (names,
  types, model names, file paths, metadata, unresolved env vars) for one
  repository. Call this to drill into repos that look interesting from the
  orientation summary.
- **resolve_env_var(var_name)** — Resolve an environment variable across all
  scanned repos. Checks .env, docker-compose, Terraform, Helm, K8s configs.
- **resolve_iac_ref(ref_expression, iac_type)** — Resolve an IaC reference
  (Terraform var, Helm value, CloudFormation Ref, ARM parameter).

## Input structure

The orientation summary contains:
- **repos** — Per-repo overview: total component count, breakdown by type,
  and unresolved env var names.
- **pre_resolved_env_vars** — Env vars already resolved deterministically
  (component → env var → concrete value + where it is defined). These are
  confirmed facts — do NOT re-resolve them.
- **shared_env_vars** — Env var names defined in multiple repos.
- **shared_packages** — Packages used by multiple repos.
- **unresolved_env_var_refs** — Env var names referenced in code but not
  defined in any scanned config file.

## Cross-repo patterns to trace

For EVERY relationship you emit, you MUST specify ``source_repo`` and
``target_repo`` — the repo where the source and target live respectively.
These must be different repos; same-repo relationships are already captured
in per-repo scans.

Trace these general patterns across repositories:

- **Agent -> Model (via deployment config)**: Application code in repo A
  uses an agent/orchestrator that consumes a model. Repo B's Helm values
  or deployment templates define that model ID (via ENGINE, MODEL_NAME,
  or similar keys). Emit ``USES_MODEL`` with source_repo=A, target_repo=B.
- **Code -> LLM Endpoint (via env var)**: Repo A code reads an env var
  whose value is an LLM endpoint URL. Repo B's Helm chart or deployment
  config defines that env var. Emit ``USES_LLM_ENDPOINT`` with
  source_repo=A, target_repo=B.
- **Code -> Vector Store (via env var)**: Repo A code reads an env var
  pointing to a vector store endpoint (Weaviate, Pinecone, Qdrant, etc.).
  Repo B deploys that vector store. Emit ``USES_VECTOR_STORE``.
- **MCP client -> MCP server**: Repo A instantiates an MCP client
  connecting to an endpoint URL. Repo B deploys the MCP server at that
  endpoint. Emit ``USES_MCP_SERVER``.
- **Endpoint -> Model (transitive)**: Repo B's deployment config maps an
  endpoint URL to a model engine/deployment name. Repo A's code consumes
  that endpoint. The transitive chain is Agent(A) -> Endpoint(B) ->
  Model(B). Emit both links.
- **Shared model IDs alone are NOT cross-repo relationships**. The same
  model ID appearing in both repos is already captured deterministically
  as ``SHARED_MODEL``. Only emit a relationship when you can trace a
  specific component-to-component dependency.

## Workflow

1. Read the orientation summary. Note which repos share env vars, packages,
   or have unresolved references.
2. Call `get_repo_components` for each repo to get full component details.
3. Cross-reference: for each agent/orchestrator in repo A, check if it
   reads env vars that are defined in repo B (Helm values, deployment
   templates). Use `resolve_env_var` or `resolve_iac_ref` for references
   not already pre-resolved.
4. Identify relationships using the patterns above. Every relationship
   MUST cross repo boundaries (source_repo != target_repo).
5. Flag risks: mismatched model versions, env vars with different values
   across repos, unresolved references, or missing configurations.
6. Output your JSON — then STOP.

## Output format

Return a SINGLE JSON object:
```json
{
  "resolved_references": [
    {
      "source_repo": "...",
      "target_repo": "...",
      "reference_type": "env_var|model|service|dependency",
      "source_component": "...",
      "resolved_value": "...",
      "explanation": "..."
    }
  ],
  "new_relationships": [
    {
      "source_name": "...",
      "source_repo": "repo-path-or-name (REQUIRED)",
      "target_name": "...",
      "target_repo": "repo-path-or-name (REQUIRED)",
      "relationship_type": "USES_MODEL|USES_LLM_ENDPOINT|USES_VECTOR_STORE|USES_EMBEDDING|USES_MCP_SERVER|HOSTS_MODEL|OBSERVED_BY|CUSTOM"
    }
  ],
  "risk_findings": [
    {
      "flag": "cross_repo_env_var_mismatch|...",
      "description": "...",
      "severity": "high|medium|low|info"
    }
  ]
}
```

## Rules

1. Do NOT hallucinate. Every finding must be backed by tool results or the
   orientation data.
2. Pre-resolved env vars are confirmed facts — use them, do not re-resolve.
3. Every ``new_relationships`` entry MUST have ``source_repo`` and
   ``target_repo`` set to a repo name or path from the orientation summary.
   They MUST be different repos. Omitting them or setting them to the same
   repo renders the link unusable.
4. Do NOT emit ``SHARED_MODEL`` relationships — those are already detected
   deterministically. Only emit relationships that trace a specific
   component-to-component cross-repo dependency.
5. Your FINAL message must be valid JSON and nothing else. No preamble,
   no markdown fences, no explanation. First character `{`, last character `}`.
"""


def run_cross_repo_coordination(
    model_string: str,
    per_repo_results: dict[str, dict[str, Any]],
    llm_config: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> tuple[list[ComponentRelationship], list[RiskFlag]]:
    """Coordinate findings across multiple repos using an LLM.

    After all per-repo scans are complete, this function invokes the agent
    to resolve cross-repo references (env vars, shared models, service links)
    that individual scans could not resolve.

    The coordinator receives a slim orientation prompt with pre-resolved env
    vars and cross-repo summary data.  Full component data is available
    on-demand via the ``get_repo_components`` tool.
    """
    if len(per_repo_results) < 2:
        return [], []

    cache = _AgenticResultCache(
        cache_dir if cache_dir is not None else _default_agentic_cache_dir()
    )
    cache_key = _cross_repo_cache_key(model_string, per_repo_results)
    cached = _load_cross_repo_cache_payload(cache.get(cache_key))
    if cached is not None:
        _LOGGER.info(
            "Cross-repo coordination cache hit across %d repos",
            len(per_repo_results),
        )
        return cached

    scan_paths = list(per_repo_results.keys())

    # --- Deterministic pre-computation -----------------------------------
    from ..cross_ref import build_env_index
    from .cross_repo import (
        build_cross_repo_tools,
        cross_repo_summary_tool,
    )

    summary_json = cross_repo_summary_tool(scan_paths)
    summary = json.loads(summary_json)

    env_index = build_env_index(scan_paths)

    pre_resolved: list[dict[str, str]] = []
    for source, data in per_repo_results.items():
        for c in data.get("components", []):
            mn = c.model_name if hasattr(c, "model_name") else c.get("model_name", "")
            if not mn or not mn.startswith("env:"):
                continue
            var_name = mn[4:]
            entries = env_index.env.get(var_name, [])
            if entries:
                pre_resolved.append(
                    {
                        "component": (
                            c.name if hasattr(c, "name") else c.get("name", "")
                        ),
                        "repo": source,
                        "env_var": var_name,
                        "resolved_value": entries[0].value,
                        "defined_in": entries[0].source_path,
                    }
                )

    repo_overview: list[dict[str, Any]] = []
    for source, data in per_repo_results.items():
        components = data.get("components", [])
        type_counts: dict[str, int] = {}
        for c in components:
            ct = (
                c.component_type.value
                if hasattr(c, "component_type")
                else c.get("type", "unknown")
            )
            type_counts[ct] = type_counts.get(ct, 0) + 1
        unresolved = data.get("_unresolved_env_vars", [])
        repo_overview.append(
            {
                "repo": source,
                "total_components": len(components),
                "by_type": type_counts,
                "unresolved_env_vars": unresolved,
            }
        )

    orientation = {
        "repos": repo_overview,
        "pre_resolved_env_vars": pre_resolved,
        "shared_env_vars": summary.get("shared_env_vars", []),
        "shared_packages": summary.get("shared_packages", []),
        "unresolved_env_var_refs": summary.get("unresolved_env_var_refs", []),
    }

    prompt = (
        "Below is an orientation summary across all scanned repositories.\n"
        "Use `get_repo_components` to drill into any repo for full details.\n\n"
        f"```json\n{json.dumps(orientation, indent=2, default=str)}\n```"
    )

    # Log only the repo count — the model id is already logged per tier during
    # enrichment, and ``model_string`` is read from the credential-bearing
    # ``llm_config`` dict, so logging it trips clear-text-logging taint analysis.
    _LOGGER.info(
        "Running cross-repo coordination across %d repos",
        len(per_repo_results),
    )

    xrepo_tools = build_cross_repo_tools(per_repo_results, scan_paths)

    model_obj = _build_model(model_string, llm_config)
    try:
        agent = create_aibom_agent(
            model_string,
            system_prompt=_CROSS_REPO_COORDINATOR_PROMPT,
            tools=xrepo_tools,
            model=model_obj,
        )
        # Daemon-thread deadline: a hung coordinator invoke is abandoned, never
        # wedging the scan at teardown. No recursion_limit here —
        # the coordinator intentionally uses the agent's default.
        result = _invoke_agent_bounded(agent, prompt, _DEFAULT_AGENTIC_TIMEOUT_S)
    except _InvokeTimeout:
        _LOGGER.warning(
            "Cross-repo coordination timed out after %ds",
            _DEFAULT_AGENTIC_TIMEOUT_S,
        )
        return [], []
    except Exception as exc:
        _LOGGER.warning("Cross-repo coordination failed: %s", exc)
        return [], []
    finally:
        _close_model_clients(model_obj)

    data = _extract_structured_response(result)
    if not data:
        return [], []

    rels: list[ComponentRelationship] = []
    for item in data.get("new_relationships", []):
        try:
            rel_type = RelationshipType(item.get("relationship_type", "CUSTOM"))
        except ValueError:
            rel_type = RelationshipType.CUSTOM
        rels.append(
            ComponentRelationship(
                source_instance_id="",
                target_instance_id="",
                source_name=item.get("source_name", ""),
                target_name=item.get("target_name", ""),
                relationship_type=rel_type,
                source_repo=item.get("source_repo", ""),
                target_repo=item.get("target_repo", ""),
            )
        )

    from ..models import Severity as Sev

    flags: list[RiskFlag] = []
    for item in data.get("risk_findings", []):
        try:
            sev = Sev(item.get("severity", "info"))
        except ValueError:
            sev = Sev.INFO
        flags.append(
            RiskFlag(
                flag=item.get("flag", "cross_repo_issue"),
                severity=sev,
                weight=5,
                description=item.get("description", ""),
            )
        )

    _LOGGER.info(
        "Cross-repo coordination: %d relationships, %d risk flags",
        len(rels),
        len(flags),
    )
    cache.put(cache_key, _build_cross_repo_cache_payload(rels, flags))
    return rels, flags


# ---------------------------------------------------------------------------
# Container layout resolution (agentic)
# ---------------------------------------------------------------------------


class _SelectedDirectory(_NullTolerantModel):
    path: str = ""
    reason: str = ""


class ContainerLayoutResponse(_NullTolerantModel):
    """Structured output for container app directory identification."""

    selected_directories: list[_SelectedDirectory] = Field(default_factory=list)
    excluded_directories: list[_SelectedDirectory] = Field(default_factory=list)
    reasoning: str = ""


_CONTAINER_LAYOUT_PROMPT = """\
You are a container image analyst for Cisco AI Defense.

You will receive:
- The image config (WORKDIR, ENTRYPOINT, CMD, ENV vars)
- A list of candidate directories that MAY contain application source code
- A file listing from the image filesystem

Your task: determine which candidate directories contain **application source code**
(Python, JavaScript, TypeScript, Java, Go, Rust, etc.) as opposed to:
- Infrastructure / deployment config (Helm values, Terraform, CI scripts)
- Vendored dependencies or generated code
- Documentation or test-only artifacts
- System files or package manager caches

Return ONLY directories that contain code the developer wrote for this application.
Be conservative — it is better to include a borderline directory than to miss real app code.

Guidelines:
- WORKDIR is the strongest signal — it almost always contains app code
- ENTRYPOINT/CMD target directories are strong signals
- Directories with requirements.txt, pyproject.toml, package.json, go.mod alongside
  source files are strong signals
- Directories named "tests", "docs", "scripts", "deploy", "helm", "terraform",
  "k8s", ".github" are usually NOT app code (but "tests" alongside app code is fine
  to include if the main app dir is the parent)
"""


def resolve_container_layout(
    model_string: str,
    image_config: dict[str, Any],
    candidate_dirs: list[str],
    file_listing: list[str],
    *,
    llm_config: dict[str, Any] | None = None,
    timeout_s: int = 60,
) -> list[str]:
    """Use an LLM to pick application directories from candidates.

    Parameters
    ----------
    model_string:
        Model identifier for ``init_chat_model``.
    image_config:
        Dict with ``workdir``, ``entrypoint``, ``cmd``, ``env``.
    candidate_dirs:
        Directories identified by deterministic heuristics.
    file_listing:
        Subset of the image file listing (capped for token budget).
    llm_config:
        Optional API keys / endpoints.
    timeout_s:
        Max seconds to wait for the LLM.

    Returns
    -------
    Filtered list of directories the LLM considers application code.
    Falls back to *candidate_dirs* on any failure.
    """
    from deepagents import create_deep_agent

    try:
        model = _build_model(model_string, llm_config)
    except Exception:
        _LOGGER.warning(
            "Failed to init LLM for container layout, using all candidates",
            exc_info=True,
        )
        return candidate_dirs

    try:
        agent = create_deep_agent(
            model=model,
            tools=[],
            system_prompt=_CONTAINER_LAYOUT_PROMPT,
            response_format=ContainerLayoutResponse,
            name="aibom-container-layout",
        )

        file_sample = file_listing[:2000]
        user_message = json.dumps(
            {
                "image_config": image_config,
                "candidate_directories": candidate_dirs,
                "file_listing_sample": file_sample,
            },
            indent=2,
        )

        # Daemon-thread deadline: a hung layout invoke is abandoned, never
        # wedging the scan at teardown. _InvokeTimeout is an
        # Exception subclass, so it is caught below and falls back cleanly.
        result = _invoke_agent_bounded(
            agent, user_message, timeout_s, recursion_limit=25
        )
    except Exception:
        _LOGGER.warning(
            "Container layout agent failed, using all candidates", exc_info=True
        )
        return candidate_dirs
    finally:
        _close_model_clients(model)

    parsed = _extract_structured_response(result)
    if not parsed:
        _LOGGER.warning("Container layout agent returned unparseable response")
        return candidate_dirs

    selected = [d.get("path", "") for d in parsed.get("selected_directories", [])]
    selected = [d for d in selected if d]

    if not selected:
        _LOGGER.info(
            "Container layout agent selected no directories, using all candidates"
        )
        return candidate_dirs

    _LOGGER.info(
        "Container layout agent selected %d of %d dirs: %s",
        len(selected),
        len(candidate_dirs),
        selected,
    )
    return selected
