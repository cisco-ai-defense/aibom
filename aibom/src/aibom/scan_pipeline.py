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

_LOGGER = logging.getLogger(__name__)


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

    def run(self) -> PipelineResult:
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

        components, relationships = self._stage_scan(ctx)
        components, env_idx, pkg_idx, ext_deps = self._stage_cross_ref(
            components
        )
        components, relationships, agentic_flags = self._stage_agentic(
            components, relationships
        )
        components, agentic_count = self._stage_assemble(components)

        return PipelineResult(
            components=components,
            relationships=relationships,
            agentic_risk_flags=agentic_flags,
            agentic_candidate_count=agentic_count,
            env_index=env_idx,
            pkg_index=pkg_idx,
            external_deps=ext_deps,
        )

    # ------------------------------------------------------------------
    # Stage 1: Run all registered scanners
    # ------------------------------------------------------------------

    def _stage_scan(
        self, ctx: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        _LOGGER.info("Stage 1/4 — scanning %d path(s)", len(self.scan_paths))
        return run_scanners(ctx)

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

        _LOGGER.info("Stage 3/4 — agentic enrichment")
        try:
            from .agentic.agent import run_agentic_enrichment

            model_str = self.llm_config["model"]
            components, agentic_rels, agentic_flags = run_agentic_enrichment(
                model_string=model_str,
                deterministic_components=components,
                deterministic_relationships=relationships,
                scan_paths=self.scan_paths,
                llm_config=self.llm_config,
            )
            relationships = relationships + agentic_rels
            return components, relationships, agentic_flags

        except ImportError:
            _LOGGER.warning(
                "Agentic enrichment requires 'cisco-aibom[agentic]'. Skipping."
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Agentic enrichment failed: %s", exc)

        return components, relationships, []

    # ------------------------------------------------------------------
    # Stage 4: Assemble — apply strict filtering, count agentic candidates
    # ------------------------------------------------------------------

    def _stage_assemble(
        self, components: list[AIComponent]
    ) -> tuple[list[AIComponent], int]:
        _LOGGER.info("Stage 4/4 — assembling results")
        agentic_count = sum(1 for c in components if c.needs_agentic)

        if self.strict:
            components = [c for c in components if not c.needs_agentic]
            _LOGGER.info(
                "Strict mode: filtered %d agentic candidates", agentic_count
            )

        return components, agentic_count
