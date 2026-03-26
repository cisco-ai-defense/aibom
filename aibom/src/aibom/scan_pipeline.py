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
  3. **Agentic** — optionally enrich with LLM reasoning (if ``llm_config``).
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
        agentic_scope: str = "candidates",
        agentic_batch_size: int = 5,
        agentic_concurrency: int = 1,
        agentic_fast_model: str | None = None,
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
    # Stage 3: Agentic enrichment (optional)
    # ------------------------------------------------------------------

    def _stage_agentic(
        self,
        components: list[AIComponent],
        relationships: list[ComponentRelationship],
    ) -> tuple[list[AIComponent], list[ComponentRelationship], list[Any]]:
        if not self.llm_config or not components:
            return components, relationships, []

        candidates = [c for c in components if c.needs_agentic]
        confirmed = [c for c in components if not c.needs_agentic]

        if self.agentic_scope == "candidates":
            to_enrich = candidates
            _LOGGER.info(
                "Stage 3/4 — agentic enrichment (%d candidates of %d total)",
                len(to_enrich), len(components),
            )
        else:
            to_enrich = components
            _LOGGER.info(
                "Stage 3/4 — agentic enrichment (all %d components)", len(components),
            )

        if not to_enrich:
            _LOGGER.info("Stage 3/4 — no agentic candidates, skipping LLM calls")
            return components, relationships, []

        try:
            from .agentic.agent import run_agentic_enrichment
        except ImportError:
            _LOGGER.warning(
                "Agentic enrichment requires 'cisco-aibom[agentic]'. Skipping."
            )
            return components, relationships, []

        try:
            model_str = self.llm_config["model"]
            enriched, agentic_rels, agentic_flags = run_agentic_enrichment(
                model_string=model_str,
                deterministic_components=to_enrich,
                deterministic_relationships=relationships,
                scan_paths=self.scan_paths,
                llm_config=self.llm_config,
                batch_size=self.agentic_batch_size,
                max_concurrent=self.agentic_concurrency,
                fast_model=self.agentic_fast_model,
            )
            relationships = relationships + agentic_rels

            if self.agentic_scope == "candidates":
                components = confirmed + enriched
            else:
                components = enriched

            return components, relationships, agentic_flags

        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Agentic enrichment failed: %s", exc, exc_info=True)

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

        agentic_count = sum(1 for c in components if c.needs_agentic)

        if self.strict:
            components = [c for c in components if not c.needs_agentic]
            _LOGGER.info(
                "Strict mode: filtered %d agentic candidates", agentic_count
            )

        return components, agentic_count
