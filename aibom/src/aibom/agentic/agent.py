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
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ..models import (
    AIComponent,
    ComponentRelationship,
    DetectionSource,
    RelationshipType,
    RiskFlag,
    ScanResult,
    SourceResult,
)
from .middleware import AIBOMScannerMiddleware
from .prompts import AIBOM_AGENT_SYSTEM_PROMPT


class _EnrichedComponent(BaseModel):
    instance_id: str = ""
    updates: dict[str, Any] = Field(default_factory=dict)


class _NewComponent(BaseModel):
    name: str = ""
    component_type: str = "other"
    file_path: str = ""
    line_number: int = 0
    framework: str = ""
    model_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class _RemoveComponent(BaseModel):
    instance_id: str = ""
    reason: str = ""


class _ReclassifyComponent(BaseModel):
    instance_id: str = ""
    new_type: str = ""
    reason: str = ""


class _Relationship(BaseModel):
    source_name: str = ""
    target_name: str = ""
    relationship_type: str = ""


class _RiskFinding(BaseModel):
    flag: str = ""
    description: str = ""
    file_path: str = ""
    line_number: int = 0
    severity: str = "info"


class AgentResponse(BaseModel):
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


def _configure_rate_limiter(init_kwargs: dict[str, Any]) -> None:
    """Attach a client-side rate limiter to the LLM unless one is already set.

    ``rate_limiter`` is a first-class field on LangChain's ``BaseChatModel``,
    so it works for *every* provider (Bedrock, OpenAI, Anthropic, Azure,
    Ollama, etc.) without provider-specific branching.  It proactively
    throttles outgoing requests so the provider never sees a burst — preventing
    rate-limit errors rather than recovering from them after the fact.
    """
    if "rate_limiter" not in init_kwargs:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        init_kwargs["rate_limiter"] = InMemoryRateLimiter(
            requests_per_second=1.0,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )


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

    model_id = model_string
    provider_prefix = ""
    if "/" in model_string:
        provider_prefix, _, model_id = model_string.partition("/")
        init_kwargs.setdefault("model_provider", provider_prefix)

    if llm_config:
        if llm_config.get("api_key"):
            init_kwargs["api_key"] = llm_config["api_key"]
        if llm_config.get("api_base"):
            if provider_prefix == "azure_openai":
                init_kwargs["azure_endpoint"] = llm_config["api_base"]
            else:
                init_kwargs["base_url"] = llm_config["api_base"]
        if llm_config.get("api_version") and provider_prefix == "azure_openai":
            init_kwargs["api_version"] = llm_config["api_version"]

    _configure_rate_limiter(init_kwargs)

    try:
        model = init_chat_model(model_id, **init_kwargs)
    except ImportError as exc:
        raise ImportError(str(exc)) from exc
    tools = build_tools()

    # NOTE: LangGraph's recommended proactive approach (RemainingSteps in
    # the graph state) cannot be used here because Deep Agents merges all
    # middleware state schemas into Input/Output schemas where managed
    # channels are forbidden.  We rely on the reactive approach instead:
    # _RECURSION_LIMIT=1000 (safety net) + GraphRecursionError catch in
    # _run_batch / _run_batch_async.
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt or AIBOM_AGENT_SYSTEM_PROMPT,
        response_format=AgentResponse,
        name="aibom-scanner",
    )
    return agent


_DEFAULT_BATCH_SIZE = 5

# LangGraph default since v1.0.6 and also the Deep Agents default.
# Set explicitly so the intent is clear: this is a safety net against
# infinite loops, NOT a workload budget.
_RECURSION_LIMIT = 1000

_SIMPLE_CANDIDATE_TYPES = frozenset({"model", "dependency", "embedding"})

_SUB_AGENT_THRESHOLD = 50


def _classify_candidates(
    components: list[AIComponent],
) -> tuple[list[AIComponent], list[AIComponent]]:
    """Split candidates into simple (registry-confirmable) vs complex.

    Simple candidates: known model/dependency/embedding with a model_name
    that just needs registry lookup confirmation.
    Complex candidates: everything else — ambiguous types, missing model
    names, multi-file reasoning required.
    """
    simple: list[AIComponent] = []
    complex_: list[AIComponent] = []
    for c in components:
        is_simple = (
            c.component_type.value in _SIMPLE_CANDIDATE_TYPES
            and c.model_name
            and not c.metadata.get("env_var_ref")
            and not c.metadata.get("env")
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


class _AgenticResultCache:
    """In-process + optional on-disk cache for agentic batch results.

    Keyed by content hash of each component so unchanged code across
    re-runs skips the LLM entirely.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._mem: dict[str, dict[str, Any]] = {}
        self._disk_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk()

    def _load_disk(self) -> None:
        if not self._disk_dir:
            return
        for p in self._disk_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                self._mem[p.stem] = data
            except (json.JSONDecodeError, OSError):
                continue

    def get(self, key: str) -> dict[str, Any] | None:
        return self._mem.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._mem[key] = value
        if self._disk_dir:
            try:
                (self._disk_dir / f"{key}.json").write_text(
                    json.dumps(value, default=str), encoding="utf-8",
                )
            except OSError:
                pass

    def partition(
        self, components: list[AIComponent],
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
    ) -> tuple[list[AIComponent], list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
        """Apply cached agentic results to components."""
        enriched: list[AIComponent] = []
        all_new: list[AIComponent] = []
        all_rels: list[ComponentRelationship] = []
        all_flags: list[RiskFlag] = []
        for c in components:
            key = _component_cache_key(c)
            data = self.get(key)
            if data:
                new, rels, flags = middleware.extract_findings_from_dict(data)
                enriched_batch = middleware.apply_enrichments_from_dict([c], data)
                enriched.extend(enriched_batch)
                all_new.extend(new)
                all_rels.extend(rels)
                all_flags.extend(flags)
            else:
                enriched.append(c)
        return enriched, all_new, all_rels, all_flags


def _run_batch(
    agent: Any,
    middleware: AIBOMScannerMiddleware,
    batch: list[AIComponent],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    batch_num: int,
    total_batches: int,
    all_components: list[AIComponent] | None = None,
) -> tuple[list[AIComponent], list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
    """Invoke the agent on a single batch and return parsed results."""
    from .tools import _reset_tool_stats, get_tool_stats

    _reset_tool_stats()
    _LOGGER.info(
        "Agentic batch %d/%d — %d components [%s]",
        batch_num, total_batches, len(batch),
        ", ".join(c.name for c in batch),
    )
    summary = _build_context_message(
        batch, relationships, scan_paths, all_components=all_components,
    )
    t0 = time.monotonic()

    result = None
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": summary}]},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        stats = get_tool_stats()
        _LOGGER.warning(
            "Batch %d failed after %.1fs: %s | tool_stats=%s",
            batch_num, elapsed, exc, json.dumps(stats),
        )
        if result is not None:
            data = _extract_structured_response(result)
            if data:
                _LOGGER.info("Batch %d: recovering partial results from failed run", batch_num)
                new_c, new_r, rf = middleware.extract_findings_from_dict(data)
                enriched = middleware.apply_enrichments_from_dict(batch, data)
                return enriched, new_c, new_r, rf
        enriched = [
            c.model_copy(update={"needs_agentic": False, "agentic_hint": "batch_recursion_limit"})
            for c in batch
        ]
        return enriched, [], [], []

    elapsed = time.monotonic() - t0
    stats = get_tool_stats()
    total_tool_calls = sum(s["calls"] for s in stats.values())
    total_tool_time = sum(s["total_s"] for s in stats.values())

    _LOGGER.info(
        "Batch %d completed in %.1fs — %d tool calls (%.1fs tool time) | breakdown=%s",
        batch_num, elapsed, total_tool_calls, total_tool_time,
        json.dumps(stats),
    )

    data = _extract_structured_response(result)
    if not data:
        _LOGGER.warning("Batch %d returned no usable output", batch_num)
        return batch, [], [], []

    new_components, new_rels, risk_flags = middleware.extract_findings_from_dict(data)
    enriched = middleware.apply_enrichments_from_dict(batch, data)
    return enriched, new_components, new_rels, risk_flags


async def _run_batch_async(
    agent: Any,
    middleware: AIBOMScannerMiddleware,
    batch: list[AIComponent],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    batch_num: int,
    total_batches: int,
    all_components: list[AIComponent] | None = None,
) -> tuple[list[AIComponent], list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
    """Async version of _run_batch using agent.ainvoke()."""
    from .tools import _reset_tool_stats, get_tool_stats

    _reset_tool_stats()
    _LOGGER.info(
        "Agentic batch %d/%d — %d components [%s]",
        batch_num, total_batches, len(batch),
        ", ".join(c.name for c in batch),
    )
    summary = _build_context_message(
        batch, relationships, scan_paths, all_components=all_components,
    )
    t0 = time.monotonic()

    result = None
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": summary}]},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
    except Exception as exc:
        elapsed = time.monotonic() - t0
        stats = get_tool_stats()
        _LOGGER.warning(
            "Batch %d failed after %.1fs: %s | tool_stats=%s",
            batch_num, elapsed, exc, json.dumps(stats),
        )
        if result is not None:
            data = _extract_structured_response(result)
            if data:
                _LOGGER.info("Batch %d: recovering partial results from failed run", batch_num)
                new_c, new_r, rf = middleware.extract_findings_from_dict(data)
                enriched = middleware.apply_enrichments_from_dict(batch, data)
                return enriched, new_c, new_r, rf
        enriched = [
            c.model_copy(update={"needs_agentic": False, "agentic_hint": "batch_recursion_limit"})
            for c in batch
        ]
        return enriched, [], [], []

    elapsed = time.monotonic() - t0
    stats = get_tool_stats()
    total_tool_calls = sum(s["calls"] for s in stats.values())
    total_tool_time = sum(s["total_s"] for s in stats.values())

    _LOGGER.info(
        "Batch %d completed in %.1fs — %d tool calls (%.1fs tool time) | breakdown=%s",
        batch_num, elapsed, total_tool_calls, total_tool_time,
        json.dumps(stats),
    )

    data = _extract_structured_response(result)
    if not data:
        _LOGGER.warning("Batch %d returned no usable output", batch_num)
        return batch, [], [], []

    new_components, new_rels, risk_flags = middleware.extract_findings_from_dict(data)
    enriched = middleware.apply_enrichments_from_dict(batch, data)
    return enriched, new_components, new_rels, risk_flags


async def _run_batches_parallel(
    agent: Any,
    middleware: AIBOMScannerMiddleware,
    batches: list[list[AIComponent]],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    all_components: list[AIComponent] | None = None,
    max_concurrent: int = 1,
) -> tuple[list[AIComponent], list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
    """Run batches concurrently, capped at *max_concurrent* simultaneous LLM conversations."""
    sem = asyncio.Semaphore(max_concurrent)

    async def _guarded(batch: list[AIComponent], idx: int) -> tuple[list[AIComponent], list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
        async with sem:
            return await _run_batch_async(
                agent, middleware, batch,
                relationships, scan_paths,
                idx, len(batches),
                all_components=all_components,
            )

    tasks = [_guarded(batch, idx) for idx, batch in enumerate(batches, 1)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_enriched: list[AIComponent] = []
    all_new: list[AIComponent] = []
    all_rels: list[ComponentRelationship] = []
    all_flags: list[RiskFlag] = []

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            _LOGGER.warning("Parallel batch %d raised: %s", i + 1, r)
            all_enriched.extend(batches[i])
            continue
        enriched, new, rels, flags = r
        all_enriched.extend(enriched)
        all_new.extend(new)
        all_rels.extend(rels)
        all_flags.extend(flags)

    return all_enriched, all_new, all_rels, all_flags


def _default_agentic_cache_dir() -> Path | None:
    """Return the default on-disk cache directory, or None if unavailable."""
    try:
        d = Path.home() / ".cache" / "cisco-aibom" / "agentic"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        return None


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
) -> tuple[list[AIComponent], list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
    """Execute a single tier (simple or complex) with caching and parallel batches."""
    tier_enriched: list[AIComponent] = []
    tier_new: list[AIComponent] = []
    tier_rels: list[ComponentRelationship] = []
    tier_flags: list[RiskFlag] = []

    to_send = components
    if cache:
        cached_comps, to_send = cache.partition(components)
        if cached_comps:
            _LOGGER.info(
                "Cache hit for %d/%d components — skipping LLM",
                len(cached_comps), len(components),
            )
            e, n, r, f = cache.apply_cached(cached_comps, middleware)
            tier_enriched.extend(e)
            tier_new.extend(n)
            tier_rels.extend(r)
            tier_flags.extend(f)

    if not to_send:
        return tier_enriched, tier_new, tier_rels, tier_flags

    batches = _locality_aware_batches(to_send, batch_size)
    _LOGGER.info(
        "%d components → %d locality-aware batches (concurrency=%d)",
        len(to_send), len(batches), max_concurrent,
    )

    if max_concurrent > 1 and len(batches) > 1:
        enriched, new, rels, flags = asyncio.run(
            _run_batches_parallel(
                agent, middleware, batches,
                relationships, scan_paths,
                all_components=all_components,
                max_concurrent=max_concurrent,
            )
        )
    else:
        enriched: list[AIComponent] = []
        new: list[AIComponent] = []
        rels: list[ComponentRelationship] = []
        flags: list[RiskFlag] = []
        for idx, batch in enumerate(batches, 1):
            e, n, r, f = _run_batch(
                agent, middleware, batch,
                relationships, scan_paths,
                idx, len(batches),
                all_components=all_components,
            )
            enriched.extend(e)
            new.extend(n)
            rels.extend(r)
            flags.extend(f)

    if cache:
        for batch in batches:
            for c in batch:
                key = _component_cache_key(c)
                cache.put(key, {
                    "enriched_components": [],
                    "new_components": [],
                    "remove_components": [],
                    "reclassify_components": [],
                    "new_relationships": [],
                    "risk_findings": [],
                })

    tier_enriched.extend(enriched)
    tier_new.extend(new)
    tier_rels.extend(rels)
    tier_flags.extend(flags)
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
) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
    """Run the full agentic enrichment pipeline.

    Features:
      - Locality-aware batching (groups co-located components)
      - Configurable parallel batches via *max_concurrent*
      - Content-hash result caching across re-runs
      - Sub-agent dispatch for large repos (>50 candidates per scan root)

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

    Returns
    -------
    Tuple of (enriched_components, new_relationships, risk_flags).
    """
    from .tools import set_allowed_search_roots

    set_allowed_search_roots([str(Path(p).resolve()) for p in scan_paths])

    simple, complex_ = _classify_candidates(deterministic_components)

    _LOGGER.info(
        "Running agentic enrichment with %s (%d simple, %d complex, %d relationships, concurrency=%d)",
        model_string,
        len(simple),
        len(complex_),
        len(deterministic_relationships),
        max_concurrent,
    )

    cache = _AgenticResultCache(_default_agentic_cache_dir())
    middleware = AIBOMScannerMiddleware()

    all_enriched: list[AIComponent] = []
    all_new: list[AIComponent] = []
    all_rels: list[ComponentRelationship] = []
    all_flags: list[RiskFlag] = []

    tier_model = fast_model or model_string
    if simple:
        _LOGGER.info(
            "Tier 1 (simple confirmations): %d candidates via %s",
            len(simple), tier_model,
        )
        agent = create_aibom_agent(tier_model, llm_config=llm_config)
        e, n, r, f = _run_tier(
            agent, middleware, simple,
            deterministic_relationships, scan_paths,
            batch_size, max_concurrent, deterministic_components, cache,
        )
        all_enriched.extend(e)
        all_new.extend(n)
        all_rels.extend(r)
        all_flags.extend(f)

    if complex_:
        dir_groups = _group_by_top_dir(complex_, scan_paths)
        use_sub_agents = (
            len(dir_groups) > 1
            and len(complex_) > _SUB_AGENT_THRESHOLD
        )

        if use_sub_agents:
            _LOGGER.info(
                "Sub-agent dispatch: %d directory groups for %d complex candidates",
                len(dir_groups), len(complex_),
            )
            for dir_key, group in sorted(dir_groups.items()):
                dir_label = Path(dir_key).name if dir_key != "__default__" else "default"
                _LOGGER.info(
                    "Sub-agent [%s]: %d candidates via %s",
                    dir_label, len(group), model_string,
                )
                agent = create_aibom_agent(model_string, llm_config=llm_config)
                e, n, r, f = _run_tier(
                    agent, middleware, group,
                    deterministic_relationships, scan_paths,
                    batch_size, max_concurrent, deterministic_components, cache,
                )
                all_enriched.extend(e)
                all_new.extend(n)
                all_rels.extend(r)
                all_flags.extend(f)
        else:
            _LOGGER.info(
                "Tier 2 (complex reasoning): %d candidates via %s",
                len(complex_), model_string,
            )
            agent = create_aibom_agent(model_string, llm_config=llm_config)
            e, n, r, f = _run_tier(
                agent, middleware, complex_,
                deterministic_relationships, scan_paths,
                batch_size, max_concurrent, deterministic_components, cache,
            )
            all_enriched.extend(e)
            all_new.extend(n)
            all_rels.extend(r)
            all_flags.extend(f)

    all_components = all_enriched + all_new

    _LOGGER.info(
        "Agentic enrichment complete: %d enriched, %d new components, "
        "%d new relationships, %d risk flags",
        len(all_enriched),
        len(all_new),
        len(all_rels),
        len(all_flags),
    )

    return all_components, all_rels, all_flags


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


def _component_to_summary(
    c: AIComponent,
    *,
    include_code: bool = False,
    enrich_target: bool = False,
) -> dict[str, Any]:
    """Serialize a single component for the agent prompt."""
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
    if enrich_target:
        entry["ENRICH"] = True

    if include_code and c.file_path and c.line_number:
        snippet = _read_code_window(c.file_path, c.line_number)
        if snippet:
            entry["code_context"] = snippet

    return entry


def _build_context_message(
    batch: list[AIComponent],
    relationships: list[ComponentRelationship],
    scan_paths: list[str],
    all_components: list[AIComponent] | None = None,
) -> str:
    """Build the user message that seeds the agent with deterministic results.

    *batch* — components the agent must enrich (with code context).
    *all_components* — full scan results for situational awareness (without
    code context, to keep the prompt compact).
    """
    batch_ids = {c.instance_id for c in batch}

    enrich_summaries = [
        _component_to_summary(c, include_code=True, enrich_target=True)
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
        "`enrich_these` (marked ENRICH=true) need your analysis — each "
        "includes a code_context window. `other_detected_components` shows "
        "everything else already found; use it to discover relationships and "
        "missing components but do NOT re-enrich those.\n\n"
        f"```json\n{json.dumps(context, indent=2)}\n```"
    )


def _extract_structured_response(result: Any) -> dict[str, Any] | None:
    """Extract the structured response from the agent's final state.

    When ``response_format`` is provided, Deep Agents populates
    ``structured_response`` in the graph state.  Falls back to parsing
    the last message as JSON if structured output is unavailable.
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
    content = ""
    if hasattr(last, "content"):
        content = last.content if isinstance(last.content, str) else str(last.content)
    elif isinstance(last, dict):
        content = str(last.get("content", ""))
    else:
        content = str(last)

    content = content.strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        _LOGGER.warning(
            "Failed to parse agent JSON output — first 300 chars: %s",
            content[:300],
        )
        return None


_CROSS_REPO_COORDINATOR_PROMPT = """\
You are an AI-BOM cross-repository coordinator. You have received scan results
from multiple repositories. Your job is to resolve cross-repo references that
individual per-repo scans could not resolve on their own.

Focus on:
1. **Env var resolution**: If repo A uses os.getenv("MODEL_NAME") and repo B
   sets MODEL_NAME=gpt-4o in docker-compose.yaml, link them.
2. **Shared model references**: If multiple repos reference the same model
   by different names or through env vars, unify them.
3. **Service-to-service links**: If repo A calls POST /api/predict and repo B
   serves that endpoint, create a relationship.
4. **Shared dependencies**: If repos share internal packages via git references
   or local paths, note the cross-repo dependency.

Return a JSON object:
```json
{
  "resolved_references": [
    {
      "source_repo": "...",
      "target_repo": "...",
      "reference_type": "env_var|model|service|dependency",
      "source_component_id": "...",
      "resolved_value": "...",
      "explanation": "..."
    }
  ],
  "new_relationships": [
    {
      "source_name": "...",
      "target_name": "...",
      "relationship_type": "USES_MODEL|DEPENDS_ON|CALLS_SERVICE|..."
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
"""


def run_cross_repo_coordination(
    model_string: str,
    per_repo_results: dict[str, dict[str, Any]],
    llm_config: dict[str, Any] | None = None,
) -> tuple[list[ComponentRelationship], list[RiskFlag]]:
    """Coordinate findings across multiple repos using an LLM.

    After all per-repo scans are complete, this function invokes the agent
    to resolve cross-repo references (env vars, shared models, service links)
    that individual scans could not resolve.

    Parameters
    ----------
    model_string:
        LLM model identifier.
    per_repo_results:
        Dict mapping source name → scan output dict (must contain
        ``components`` and optionally ``_unresolved_env_vars``).
    llm_config:
        LLM connection config.

    Returns
    -------
    Tuple of (new_cross_repo_relationships, risk_flags).
    """
    if len(per_repo_results) < 2:
        return [], []

    repo_summaries: list[dict[str, Any]] = []
    for source, data in per_repo_results.items():
        components = data.get("components", [])
        comp_list = []
        for c in components:
            if hasattr(c, "model_dump"):
                comp_list.append({
                    "name": c.name,
                    "type": c.component_type.value,
                    "model_name": c.model_name,
                    "file": c.file_path,
                    "metadata": c.metadata,
                })
            elif isinstance(c, dict):
                comp_list.append(c)

        unresolved = data.get("_unresolved_env_vars", [])
        repo_summaries.append({
            "repo": source,
            "component_count": len(comp_list),
            "components": comp_list[:50],
            "unresolved_env_vars": unresolved[:20],
        })

    context = json.dumps(repo_summaries, indent=2, default=str)
    prompt = (
        "Here are scan results from multiple repositories. "
        "Please identify cross-repo references and resolve them.\n\n"
        f"```json\n{context}\n```"
    )

    _LOGGER.info(
        "Running cross-repo coordination across %d repos with %s",
        len(per_repo_results), model_string,
    )

    try:
        agent = create_aibom_agent(
            model_string,
            llm_config=llm_config,
            system_prompt=_CROSS_REPO_COORDINATOR_PROMPT,
        )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
    except Exception as exc:
        _LOGGER.warning("Cross-repo coordination failed: %s", exc)
        return [], []

    data = _extract_structured_response(result)
    if not data:
        return [], []

    rels: list[ComponentRelationship] = []
    for item in data.get("new_relationships", []):
        try:
            rel_type = RelationshipType(item.get("relationship_type", "CUSTOM"))
        except ValueError:
            rel_type = RelationshipType.CUSTOM
        rels.append(ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name=item.get("source_name", ""),
            target_name=item.get("target_name", ""),
            relationship_type=rel_type,
        ))

    from ..models import Severity as Sev

    flags: list[RiskFlag] = []
    for item in data.get("risk_findings", []):
        try:
            sev = Sev(item.get("severity", "info"))
        except ValueError:
            sev = Sev.INFO
        flags.append(RiskFlag(
            flag=item.get("flag", "cross_repo_issue"),
            severity=sev,
            weight=5,
            description=item.get("description", ""),
        ))

    _LOGGER.info(
        "Cross-repo coordination: %d relationships, %d risk flags",
        len(rels), len(flags),
    )
    return rels, flags
