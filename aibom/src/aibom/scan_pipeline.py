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

"""Orchestrates the four-stage v2 scan pipeline.

Stages:
  1. **Scan** — run all registered scanners (Tier 1 + Tier 2 EnvVarResolver).
  2. **Cross-ref** — build env/package index, resolve env-var components.
  3. **Agentic** — classify all candidates via LLM agent (requires ``llm_config``).
  4. **Assemble** — apply ``--strict`` filtering, collect counts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .cross_ref import (
    CrossRefIndex,
    ExternalRepoDep,
    build_env_index,
    build_package_index,
    detect_external_repo_deps,
    resolve_components,
)
from .models import ScanContext
from .models.enums import AIComponentType
from .models.scan import AIComponent, ComponentRelationship
from .scanners import run_scanners
from .scanners.file_cache import cache_stats, clear_cache

_LOGGER = logging.getLogger(__name__)


@dataclass
class StageTiming:
    """Wall-clock elapsed time for a single pipeline stage."""

    name: str
    elapsed_s: float
    detail: str = ""


@dataclass
class PipelineResult:
    """Outcome of a single-source pipeline run."""

    components: list[AIComponent] = field(default_factory=list)
    relationships: list[ComponentRelationship] = field(default_factory=list)
    agentic_risk_flags: list[Any] = field(default_factory=list)
    agentic_candidate_count: int = 0
    env_index: CrossRefIndex | None = None
    pkg_index: CrossRefIndex | None = None
    external_deps: list[ExternalRepoDep] = field(default_factory=list)
    timings: list[StageTiming] = field(default_factory=list)
    total_elapsed_s: float = 0.0


def _service_dir(file_path: str, scan_paths: list[str] | None = None) -> str:
    """Extract a logical service/directory key from a file path.

    Walks up from the file until it finds a directory that contains a
    manifest (pyproject.toml, go.mod, package.json, Cargo.toml) or is
    two levels deep from a scan root.  Falls back to the immediate parent.
    """
    from pathlib import Path

    p = Path(file_path)
    for ancestor in p.parents:
        for marker in ("pyproject.toml", "go.mod", "package.json", "Cargo.toml"):
            if (ancestor / marker).exists():
                return str(ancestor)
    return str(p.parent)


def _consolidation_key(c: "AIComponent") -> tuple[str, str]:
    """Key for grouping duplicate components.

    Groups by (canonical_name, component_type) across the entire scan —
    repo-level dedup.  ``model_name`` is preferred over ``name`` when
    present so that different framework wrappers for the same model collapse.
    """
    canonical = (c.model_name or c.name).lower().strip()
    return (canonical, c.component_type.value)


_CONTEXT_FREE_TYPES: frozenset[str] = frozenset({
    "dependency", "model", "model_artifact", "embedding",
})


def _dedup_for_agentic(
    components: list["AIComponent"],
) -> tuple[list["AIComponent"], dict[str, list["AIComponent"]]]:
    """Collapse context-free duplicates before the agentic stage.

    Returns (deduped_list, fanout_map) where fanout_map maps each
    representative's instance_id to the list of all original instances
    sharing the same consolidation key.  Context-dependent components
    (env vars, endpoints, prompts, secrets) pass through unchanged.
    """
    from collections import OrderedDict

    fanout: dict[str, list["AIComponent"]] = {}
    groups: OrderedDict[tuple, list["AIComponent"]] = OrderedDict()
    passthrough: list["AIComponent"] = []

    for c in components:
        if c.component_type.value not in _CONTEXT_FREE_TYPES:
            passthrough.append(c)
            continue
        key = _consolidation_key(c)
        groups.setdefault(key, []).append(c)

    representatives: list["AIComponent"] = []
    for _key, group in groups.items():
        rep = max(group, key=lambda c: len(c.metadata) + len(c.description or ""))
        representatives.append(rep)
        fanout[rep.instance_id] = group

    return representatives + passthrough, fanout


def _fanout_agentic_results(
    enriched: list["AIComponent"],
    fanout: dict[str, list["AIComponent"]],
) -> list["AIComponent"]:
    """Propagate agentic enrichments from representatives to all instances."""
    result: list["AIComponent"] = []
    for c in enriched:
        siblings = fanout.get(c.instance_id)
        if not siblings or len(siblings) <= 1:
            result.append(c)
            continue
        for sib in siblings:
            clone = sib.model_copy(deep=True)
            clone.confidence = c.confidence
            clone.needs_agentic = c.needs_agentic
            if c.model_name:
                clone.model_name = c.model_name
            if c.component_type != sib.component_type:
                clone.component_type = c.component_type
            for k, v in c.metadata.items():
                if k not in sib.metadata:
                    clone.metadata[k] = v
            result.append(clone)
    return result


def _propagate_removals(
    sent: list["AIComponent"],
    received: list["AIComponent"],
    all_candidates: list["AIComponent"] | None = None,
    *,
    pre_fanout_removed_ids: set[str] | None = None,
) -> list["AIComponent"]:
    """If the agent removed ANY instance of (name, type), remove ALL.

    The agent processes each instance independently and sometimes makes
    inconsistent decisions (removes the usage at line 595 but keeps the
    import at line 85).  This function treats a removal of *any* instance
    as a removal of the logical component, keyed by consolidation key.

    *all_candidates*, when provided, is the full pre-dedup component list.
    Removal keys are built from this wider set so that siblings that were
    never sent to the agent (collapsed by dedup) are also caught.

    *pre_fanout_removed_ids*, when provided, is the set of instance_ids
    absent from the enriched output *before* fanout.  This avoids a subtle
    bug where fanout can re-introduce an instance_id from a different dedup
    group, masking the removal.
    """
    if pre_fanout_removed_ids is not None:
        removed_ids = pre_fanout_removed_ids
    else:
        sent_ids = {c.instance_id for c in sent}
        received_ids = {c.instance_id for c in received}
        removed_ids = sent_ids - received_ids
    if not removed_ids:
        return received

    removed_keys: set[tuple] = set()
    for c in sent:
        if c.instance_id in removed_ids:
            removed_keys.add(_consolidation_key(c))
    if not removed_keys:
        return received

    lookup_pool = all_candidates if all_candidates is not None else received
    removed_iids = {c.instance_id for c in lookup_pool if _consolidation_key(c) in removed_keys}
    result = [
        c for c in lookup_pool
        if _consolidation_key(c) not in removed_keys
        and c.instance_id not in removed_iids
    ]
    dropped = len(lookup_pool) - len(result)
    if dropped:
        _LOGGER.info(
            "Removal propagation: dropped %d additional component(s) "
            "matching %d removal key(s)",
            dropped, len(removed_keys),
        )
    return result


def _evidence_gate(
    before_agentic: list["AIComponent"],
    after_agentic: list["AIComponent"],
) -> list["AIComponent"]:
    """Remove post-agentic components that lack hard evidence.

    Two checks:

    1. **model + kb_enrichment**: If the name does not resolve in any model
       registry (LiteLLM, built-in regex, HuggingFace), auto-remove.  This
       does NOT touch ``model_detector`` detections (string literals from
       code) — only class-name-inferred components from ``kb_enrichment``.

    2. **memory + kb_enrichment**: If the agent kept the component unchanged
       (no enrichment, no reclassification), auto-remove.  A genuine memory
       component should have been enriched with specifics.
    """
    from .models.enums import DetectionSource
    from .scanners.model_detector import _registry_lookup

    before_map: dict[str, "AIComponent"] = {c.instance_id: c for c in before_agentic}

    result: list["AIComponent"] = []
    gate_removed = 0
    for c in after_agentic:
        orig = before_map.get(c.instance_id)
        if orig is None:
            result.append(c)
            continue

        if (
            c.component_type == AIComponentType.MODEL
            and orig.detection_source == DetectionSource.KB_ENRICHMENT
            and _registry_lookup(c.name) is None
        ):
            _LOGGER.info(
                "Evidence gate removed model '%s' (%s): "
                "not found in any model registry and detection_source=kb_enrichment",
                c.name, c.instance_id,
            )
            gate_removed += 1
            continue

        if (
            c.component_type == AIComponentType.MEMORY
            and orig.detection_source == DetectionSource.KB_ENRICHMENT
            and c.metadata == orig.metadata
            and c.model_name == orig.model_name
            and c.component_type == orig.component_type
        ):
            _LOGGER.info(
                "Evidence gate removed memory '%s' (%s): "
                "agent kept kb_enrichment component unchanged",
                c.name, c.instance_id,
            )
            gate_removed += 1
            continue

        result.append(c)

    if gate_removed:
        _LOGGER.info("Evidence gate: removed %d component(s)", gate_removed)
    return result


_TEST_PATH_SEGMENTS: frozenset[str] = frozenset({
    "tests", "test", "__tests__", "spec", "testing", "testdata",
    "test_data", "fixtures",
})


def _is_test_file(file_path: str) -> bool:
    """Return True when the file lives under a test directory."""
    from pathlib import Path
    parts = Path(file_path).parts
    return any(seg in _TEST_PATH_SEGMENTS for seg in parts)


def _consolidate_components(
    components: list["AIComponent"],
) -> list["AIComponent"]:
    """Merge per-file-reference components into per-logical-asset components.

    Groups by (canonical_name, component_type) across the entire repo.
    The highest-confidence occurrence is kept; all others become entries
    in ``metadata["evidence"]`` with file, line, and service fields.
    Components where **every** occurrence is in a test file are tagged
    ``metadata["test_only"] = True``.
    """
    from collections import OrderedDict

    groups: OrderedDict[tuple, list["AIComponent"]] = OrderedDict()
    for c in components:
        key = _consolidation_key(c)
        groups.setdefault(key, []).append(c)

    result: list["AIComponent"] = []
    for _key, group in groups.items():
        if len(group) == 1:
            c = group[0]
            if _is_test_file(c.file_path):
                merged_meta = dict(c.metadata)
                merged_meta["test_only"] = True
                c = c.model_copy(update={"metadata": merged_meta})
            result.append(c)
            continue

        best = max(group, key=lambda c: (c.confidence, -c.line_number))
        evidence = []
        all_test = True
        for c in group:
            is_test = _is_test_file(c.file_path)
            if not is_test:
                all_test = False
            if c is not best:
                evidence.append({
                    "file": c.file_path,
                    "line": c.line_number,
                    "service": _service_dir(c.file_path),
                    "test_only": is_test,
                })

        if not _is_test_file(best.file_path):
            all_test = False

        merged_meta = dict(best.metadata)
        merged_meta["evidence"] = evidence
        merged_meta["consolidated_count"] = len(group)
        if all_test:
            merged_meta["test_only"] = True

        needs = any(c.needs_agentic for c in group)

        merged = best.model_copy(update={
            "metadata": merged_meta,
            "needs_agentic": needs,
        })
        result.append(merged)

    return result


def _vector_store_technology(c: "AIComponent") -> str | None:
    if c.component_type != AIComponentType.VECTOR_STORE:
        return None
    name_l = (c.model_name or c.name or "").lower()
    if "weaviate" in name_l:
        return "weaviate"
    if "pinecone" in name_l:
        return "pinecone"
    if "chroma" in name_l:
        return "chromadb"
    if "milvus" in name_l:
        return "milvus"
    if "qdrant" in name_l:
        return "qdrant"
    if "faiss" in name_l:
        return "faiss"
    if "pgvector" in name_l:
        return "pgvector"
    if "redis" in name_l:
        return "redis"
    return None


def _consolidate_vector_stores(
    components: list["AIComponent"],
) -> list["AIComponent"]:
    from collections import OrderedDict

    tech_groups: OrderedDict[str, list["AIComponent"]] = OrderedDict()
    rest: list["AIComponent"] = []

    for c in components:
        tech = _vector_store_technology(c)
        if tech is None:
            rest.append(c)
        else:
            tech_groups.setdefault(tech, []).append(c)

    merged: list["AIComponent"] = list(rest)

    for tech, group in tech_groups.items():
        if len(group) == 1:
            c = group[0]
            meta = dict(c.metadata)
            meta["store_technology"] = tech
            merged.append(c.model_copy(update={"metadata": meta}))
            continue

        best = max(group, key=lambda c: (c.confidence, -c.line_number))
        seen: set[tuple[str, int]] = set()
        evidence: list[dict[str, Any]] = []

        def _add_evidence_entry(
            file_path: str,
            line_number: int,
            *,
            test_only: bool | None = None,
        ) -> None:
            if test_only is None:
                test_only = _is_test_file(file_path)
            key = (file_path, line_number)
            if key in seen:
                return
            seen.add(key)
            evidence.append({
                "file": file_path,
                "line": line_number,
                "service": _service_dir(file_path),
                "test_only": test_only,
            })

        def _merge_prior_evidence(meta: dict[str, Any]) -> None:
            for ev in meta.get("evidence") or []:
                if not isinstance(ev, dict):
                    continue
                fp = str(ev.get("file", ""))
                ln = int(ev.get("line", 0) or 0)
                t = ev.get("test_only")
                if isinstance(t, bool):
                    _add_evidence_entry(fp, ln, test_only=t)
                else:
                    _add_evidence_entry(fp, ln)

        _merge_prior_evidence(dict(best.metadata))

        for c in group:
            if c is best:
                continue
            _add_evidence_entry(c.file_path, c.line_number)
            _merge_prior_evidence(dict(c.metadata))

        best_key = (best.file_path, best.line_number)
        merged_evidence = [
            ev for ev in evidence
            if (str(ev.get("file", "")), int(ev.get("line", 0) or 0)) != best_key
        ]

        merged_meta = dict(best.metadata)
        merged_meta["evidence"] = merged_evidence
        merged_meta["consolidated_count"] = len(group)
        merged_meta["store_technology"] = tech

        all_test = all(_is_test_file(c.file_path) for c in group)
        if all_test:
            merged_meta["test_only"] = True

        needs = any(c.needs_agentic for c in group)

        merged.append(best.model_copy(update={
            "metadata": merged_meta,
            "needs_agentic": needs,
        }))

    return merged


def _dedup_tool_vs_vector_store(
    components: list["AIComponent"],
) -> list["AIComponent"]:
    """If the same entity appears as both TOOL and VECTOR_STORE, keep only VECTOR_STORE."""
    vs_names: set[str] = set()
    for c in components:
        if c.component_type == AIComponentType.VECTOR_STORE:
            vs_names.add((c.model_name or c.name).lower().strip())

    if not vs_names:
        return components

    result: list["AIComponent"] = []
    for c in components:
        if c.component_type == AIComponentType.TOOL:
            key = (c.model_name or c.name).lower().strip()
            if key in vs_names:
                continue
        result.append(c)
    return result


class ScanPipeline:
    """Four-stage v2 scan pipeline wired into a single ``run()`` call."""

    def __init__(
        self,
        scan_paths: list[str],
        *,
        output_format: str = "json",
        output_file: str | None = None,
        llm_config: dict[str, Any] | None = None,
        kb_path: str | None = None,
        fail_on: Any | None = None,
        min_severity: Any | None = None,
        strict: bool = False,
        exclude_patterns: list[str] | None = None,
        agentic_scope: str = "all",
        agentic_batch_size: int = 5,
        agentic_concurrency: int = 1,
        agentic_fast_model: str | None = None,
        agentic_timeout: int = 120,
    ) -> None:
        self.scan_paths = scan_paths
        self.output_format = output_format
        self.output_file = output_file
        self.llm_config = llm_config
        self.kb_path = kb_path
        self.fail_on = fail_on
        self.min_severity = min_severity
        self.strict = strict
        self.exclude_patterns = exclude_patterns or []
        self.agentic_scope = agentic_scope
        self.agentic_batch_size = agentic_batch_size
        self.agentic_concurrency = agentic_concurrency
        self.agentic_fast_model = agentic_fast_model
        self.agentic_timeout = agentic_timeout

    def run(self) -> PipelineResult:
        clear_cache()
        pipeline_start = time.monotonic()
        timings: list[StageTiming] = []

        ctx_kwargs: dict[str, Any] = {
            "paths": self.scan_paths,
            "output_format": self.output_format,
            "exclude_patterns": self.exclude_patterns,
        }
        if self.output_file is not None:
            ctx_kwargs["output_file"] = self.output_file
        if self.llm_config is not None:
            ctx_kwargs["llm_config"] = self.llm_config
        if self.kb_path is not None:
            ctx_kwargs["kb_path"] = self.kb_path
        if self.fail_on is not None:
            ctx_kwargs["fail_on"] = self.fail_on
        if self.min_severity is not None:
            ctx_kwargs["min_severity"] = self.min_severity
        ctx = ScanContext(**ctx_kwargs)

        t0 = time.monotonic()
        components, relationships = self._stage_scan(ctx)
        elapsed = time.monotonic() - t0
        fc = cache_stats()
        timings.append(StageTiming(
            "scan", elapsed,
            f"{len(components)} components, {len(relationships)} relationships, "
            f"file cache {fc['hits']} hits / {fc['misses']} misses",
        ))

        t0 = time.monotonic()
        components, env_idx, pkg_idx, ext_deps = self._stage_cross_ref(
            components
        )
        elapsed = time.monotonic() - t0
        timings.append(StageTiming(
            "cross_ref", elapsed,
            f"{sum(len(v) for v in env_idx.env.values())} env vars, "
            f"{len(pkg_idx.packages)} packages, {len(ext_deps)} external deps",
        ))

        t0 = time.monotonic()
        components, relationships, agentic_flags = self._stage_agentic(
            components, relationships
        )
        elapsed = time.monotonic() - t0
        skipped = not self.llm_config
        timings.append(StageTiming(
            "agentic", elapsed,
            "skipped (no --llm-model)" if skipped else f"{len(agentic_flags)} risk flags",
        ))

        t0 = time.monotonic()
        components, agentic_count = self._stage_assemble(components)
        elapsed = time.monotonic() - t0
        timings.append(StageTiming(
            "assemble", elapsed, f"{len(components)} final, {agentic_count} agentic",
        ))

        total_elapsed = time.monotonic() - pipeline_start

        return PipelineResult(
            components=components,
            relationships=relationships,
            agentic_risk_flags=agentic_flags,
            agentic_candidate_count=agentic_count,
            env_index=env_idx,
            pkg_index=pkg_idx,
            external_deps=ext_deps,
            timings=timings,
            total_elapsed_s=total_elapsed,
        )

    # ------------------------------------------------------------------
    # Stage 1: Two-pass scanning
    #   Pass 1 — manifest + config (structural, zero FPs) → ai_package_set
    #   Pass 2 — all other scanners scoped by the discovered package set
    # ------------------------------------------------------------------

    def _stage_scan(
        self, ctx: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        _LOGGER.info("Stage 1/4 — scanning %d path(s)", len(self.scan_paths))

        from .scanners.dependency_scanner import discover_ai_package_set

        ai_pkgs = discover_ai_package_set(ctx)
        if ai_pkgs:
            _LOGGER.info(
                "Pass 1: discovered %d AI package(s) from manifests: %s",
                len(ai_pkgs),
                ", ".join(sorted(ai_pkgs)[:10])
                + ("…" if len(ai_pkgs) > 10 else ""),
            )
        else:
            _LOGGER.debug("Pass 1: no AI packages found in manifests")

        ctx_pass2 = ctx.model_copy(update={"ai_package_set": ai_pkgs})

        idx = ctx_pass2.file_index()
        if idx:
            import asyncio
            from .scanners.file_cache import warm_cache_async

            all_paths = [e.path for entries in idx.values() for e in entries]
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    warmed = pool.submit(
                        asyncio.run, warm_cache_async(all_paths)
                    ).result()
            else:
                warmed = asyncio.run(warm_cache_async(all_paths))
            _LOGGER.info(
                "Pass 2 prep: pre-cached %d / %d files via async I/O",
                warmed, len(all_paths),
            )

        return run_scanners(ctx_pass2)

    # ------------------------------------------------------------------
    # Stage 2: Cross-reference resolution
    # ------------------------------------------------------------------

    def _stage_cross_ref(
        self, components: list[AIComponent]
    ) -> tuple[list[AIComponent], CrossRefIndex, CrossRefIndex, list[ExternalRepoDep]]:
        _LOGGER.info("Stage 2/4 — cross-reference resolution")
        env_idx = build_env_index(self.scan_paths)
        pkg_idx = build_package_index(self.scan_paths)

        _LOGGER.debug(
            "Cross-ref index: %d env vars, %d packages",
            sum(len(v) for v in env_idx.env.values()),
            len(pkg_idx.packages),
        )

        resolved = resolve_components(components, env_idx)

        ext_deps = detect_external_repo_deps(self.scan_paths)
        if ext_deps:
            _LOGGER.info(
                "Detected %d external repo dependency(ies)", len(ext_deps)
            )
            escaping = [d for d in ext_deps if d.escapes_root]
            if escaping:
                _LOGGER.warning(
                    "%d dependency(ies) reference paths outside scanned repos",
                    len(escaping),
                )

        return resolved, env_idx, pkg_idx, ext_deps

    # ------------------------------------------------------------------
    # Stage 3: Agentic classification (mandatory)
    # ------------------------------------------------------------------

    def _stage_agentic(
        self,
        components: list[AIComponent],
        relationships: list[ComponentRelationship],
    ) -> tuple[list[AIComponent], list[ComponentRelationship], list[Any]]:
        if not self.llm_config or not components:
            return components, relationships, []

        _LOGGER.info(
            "Stage 3/4 — agentic classification (all %d components)",
            len(components),
        )

        try:
            from .agentic.agent import run_agentic_enrichment
        except ImportError:
            _LOGGER.warning(
                "Agentic classification requires 'cisco-aibom[agentic]'. Skipping."
            )
            return components, relationships, []

        try:
            deduped, fanout = _dedup_for_agentic(components)
            if len(deduped) < len(components):
                _LOGGER.info(
                    "Pre-agentic dedup: %d → %d components (%d context-free duplicates removed)",
                    len(components), len(deduped), len(components) - len(deduped),
                )

            model_str = self.llm_config["model"]
            enriched, agentic_rels, agentic_flags = run_agentic_enrichment(
                model_string=model_str,
                deterministic_components=deduped,
                deterministic_relationships=relationships,
                scan_paths=self.scan_paths,
                llm_config=self.llm_config,
                batch_size=self.agentic_batch_size,
                max_concurrent=self.agentic_concurrency,
                fast_model=self.agentic_fast_model,
                timeout_s=self.agentic_timeout,
            )
            deduped_ids = {c.instance_id for c in deduped}
            enriched_deduped_ids = {c.instance_id for c in enriched if c.instance_id in deduped_ids}
            new_components = [c for c in enriched if c.instance_id not in deduped_ids]
            pre_fanout_removed = deduped_ids - enriched_deduped_ids
            enriched_for_fanout = [c for c in enriched if c.instance_id in deduped_ids]
            enriched = _fanout_agentic_results(enriched_for_fanout, fanout)
            enriched = _propagate_removals(
                deduped, enriched, all_candidates=components,
                pre_fanout_removed_ids=pre_fanout_removed,
            )
            enriched = _evidence_gate(components, enriched)
            if new_components:
                enriched = enriched + new_components
            relationships = relationships + agentic_rels
            return enriched, relationships, agentic_flags

        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Agentic classification failed: %s", exc, exc_info=True)

        return components, relationships, []

    # ------------------------------------------------------------------
    # Stage 4: Assemble — consolidate, filter, count
    # ------------------------------------------------------------------

    def _stage_assemble(
        self, components: list[AIComponent]
    ) -> tuple[list[AIComponent], int]:
        _LOGGER.info("Stage 4/4 — assembling results")

        before = len(components)
        components = _consolidate_components(components)
        after = len(components)
        if before != after:
            _LOGGER.info(
                "Consolidation: %d → %d components (-%d duplicates)",
                before, after, before - after,
            )

        before_vs = len(components)
        components = _consolidate_vector_stores(components)
        after_vs = len(components)
        if before_vs != after_vs:
            _LOGGER.info(
                "Vector store dedup: %d → %d components (-%d)",
                before_vs, after_vs, before_vs - after_vs,
            )

        before_td = len(components)
        components = _dedup_tool_vs_vector_store(components)
        after_td = len(components)
        if before_td != after_td:
            _LOGGER.info(
                "Tool/vector_store priority dedup: %d → %d (-%d)",
                before_td, after_td, before_td - after_td,
            )

        agentic_count = sum(1 for c in components if c.needs_agentic)

        if self.strict:
            components = [c for c in components if not c.needs_agentic]
            _LOGGER.info(
                "Strict mode: filtered %d agentic candidates", agentic_count
            )

        return components, agentic_count
