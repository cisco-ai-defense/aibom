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
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENV_PLACEHOLDER_RE = re.compile(r"\$\{[^}]+\}")

_ENV_NAME_PREFIXES: tuple[str, ...] = (
    "env:",
    "env_model_",
    "env_embedding_",
    "dockerfile_env_",
)


def _strip_env_prefix(name: str) -> str:
    """Strip a known env-var naming prefix from a component ``name``.

    Multiple scanners tag env-backed components with distinct naming
    conventions (``env:VAR``, ``env_model_VAR``, ``env_embedding_VAR``,
    ``dockerfile_env_VAR``). The canonicalizer uses this helper so a
    single gate matches every shape and collapses them all to the
    resolved literal model id when one is available.
    """
    for prefix in _ENV_NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


from .agent_signatures import AgentSignatureCatalog, resolve_catalog
from .cross_ref import (
    CrossRefIndex,
    ExternalRepoDep,
    build_env_index,
    build_package_index,
    detect_external_repo_deps,
    resolve_components,
)
from .custom_catalog import CustomCatalogConfig
from .llm_factory import ensure_llm_runtime_available
from .models import ScanContext
from .models.enums import AIComponentType, RelationshipType
from .models.scan import AIComponent, ComponentRelationship
from .scanners import run_scanners
from .scanners.dependency_scanner import discover_ai_package_set, is_known_ai_package
from .scanners.file_cache import cache_stats, clear_cache
from .scanners.model_detector import is_known_embedding_model_name

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
    agentic_degraded_count: int = 0
    env_index: CrossRefIndex | None = None
    pkg_index: CrossRefIndex | None = None
    external_deps: list[ExternalRepoDep] = field(default_factory=list)
    timings: list[StageTiming] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0


def _service_dir(file_path: str) -> str:
    """Extract a logical service/directory key from a file path.

    Walks up from the file until it finds a directory that contains a
    manifest (pyproject.toml, go.mod, package.json, Cargo.toml). Falls back
    to the immediate parent.
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


_CONTEXT_FREE_TYPES: frozenset[str] = frozenset(
    {
        "dependency",
        "model",
        "model_artifact",
        "embedding",
    }
)


def _partition_agentic_secrets(
    components: list["AIComponent"],
    review_secrets: bool = False,
) -> tuple[list["AIComponent"], list["AIComponent"]]:
    """Split components into ``(to_agentic, held_back)``.

    ``SECRET`` detections (e.g. detect-secrets high-entropy strings) are
    false-positive-prone and are not a core AI component. By default they are
    held back from the agentic LLM stage rather than spending frontier-LLM
    budget adjudicating hundreds of high-entropy strings, which can be a
    substantial cost on secret-heavy repos. Pass ``review_secrets=True`` to
    route them through the agent as before.
    """
    if review_secrets:
        return list(components), []
    to_agentic: list["AIComponent"] = []
    held: list["AIComponent"] = []
    for c in components:
        target = held if c.component_type == AIComponentType.SECRET else to_agentic
        target.append(c)
    return to_agentic, held


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
            clone.heuristic_confidence = c.heuristic_confidence
            clone.agentic_confidence = c.agentic_confidence
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


def _is_import_only_candidate(c: "AIComponent") -> bool:
    """True when the component was emitted *only* from a bare ``from X import Y`` line.

    ``kb_enrichment_scanner`` tags these weak, import-inferred detections
    with ``metadata["import_statement"]`` and never sets ``call_pattern``
    (which is the marker for a real callsite emission).

    The distinction matters for :func:`_propagate_removals`: an LLM
    removal of an import-inferred component is evidence that *the import
    alone* is not an AI component — it is **not** evidence that the
    usage-line sibling with the same lowercased canonical name (e.g.
    ``agent = Agent(...)``) should also disappear.
    """
    md = c.metadata or {}
    return bool(md.get("import_statement")) and not md.get("call_pattern")


def _is_protected_dependency(c: "AIComponent") -> bool:
    """True for a dependency row that is a recognized AI package.

    A dependency declared in a manifest and matched against the AI-package
    allow-list is a deterministic, verifiable fact — not a heuristic candidate
    that needs the agent's keep/prune judgment. The agentic stage may still
    omit it from its returned set (it processes each candidate independently
    and an unfamiliar SDK name can be pruned), which would otherwise cause
    :func:`_propagate_removals` to drop every instance of that package. Such
    rows must survive regardless of the agent's decision.
    """
    return c.component_type == AIComponentType.DEPENDENCY and bool(
        (c.metadata or {}).get("known_ai_package")
    )


def _reinstate_protected_dependencies(
    all_candidates: list["AIComponent"],
    enriched: list["AIComponent"],
) -> list["AIComponent"]:
    """Re-add recognized AI dependencies the agent dropped.

    Covers the case where the agent omits the sole dedup representative of a
    package: fanout then restores none of its instances, so a dependency that
    is a verified manifest fact would silently disappear. Any protected
    dependency present in the original candidate set but absent from the
    enriched output is re-appended (preserving any enrichment already applied
    to a surviving instance with the same key).
    """
    present_keys = {
        _consolidation_key(c) for c in enriched if _is_protected_dependency(c)
    }
    present_ids = {c.instance_id for c in enriched}
    readded = 0
    for c in all_candidates:
        if not _is_protected_dependency(c):
            continue
        if c.instance_id in present_ids:
            continue
        if _consolidation_key(c) in present_keys:
            continue
        enriched.append(c)
        present_keys.add(_consolidation_key(c))
        present_ids.add(c.instance_id)
        readded += 1
    if readded:
        _LOGGER.info(
            "Reinstated %d recognized AI dependency component(s) dropped "
            "by the agentic stage",
            readded,
        )
    return enriched


def _propagate_removals(
    sent: list["AIComponent"],
    received: list["AIComponent"],
    all_candidates: list["AIComponent"] | None = None,
    *,
    pre_fanout_removed_ids: set[str] | None = None,
) -> list["AIComponent"]:
    """If the agent removed ANY instance of (name, type), remove ALL — with
    two asymmetries that *compose*: (1) import-only removals are *weak* and
    only cascade to other import-only siblings; (2) removals of a *test-file*
    instance are *test-scoped* and only cascade to other test-file siblings,
    never to a production sibling sharing the same canonical key. A removal's
    reach is the intersection of these restrictions, so e.g. a test-file
    import-only removal cascades only to siblings that are *both* import-only
    *and* in test files. An unrestricted production call-site removal cascades
    to all siblings, as before.

    The agent processes each instance independently and sometimes makes
    inconsistent decisions (removes the usage at line 595 but keeps the
    import at line 85).  This function treats a *substantive* removal of
    any instance as a removal of the logical component, keyed by
    consolidation key.

    **Weak/strong distinction.** A removal is *weak* when the removed
    component is an import-only detection (see
    :func:`_is_import_only_candidate`).  Weak removals propagate only to
    other import-only siblings sharing the same consolidation key; they
    must never take out a usage-line sibling whose lowercased canonical
    name happens to collide.  Without this guard, the LLM rejecting a
    ``from strands import Agent`` line (correct — an import alone is
    not an agent) would wipe out every real ``agent = Agent(...)``
    because ``_consolidation_key`` lowercases both to ``("agent",
    "agent")``.

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

    # For each consolidation key, record how *restricted* its removal cascade
    # is. A removal's reach is the intersection of two independent axes:
    #   * import-only ("weak"): may only cascade to other import-only siblings
    #     (an import alone is not evidence the usage-line is invalid);
    #   * test-file ("test-scoped"): may only cascade to other test-file
    #     siblings (pruning a test artifact must not delete a production one).
    # A production call-site removal is unrestricted (cascades to all). When
    # the SAME key is removed from multiple instances, the cascade takes the
    # LEAST restrictive removal seen (e.g. a production call-site removal
    # overrides a test-file one for that key).
    restrict_import: dict[tuple, bool] = {}
    restrict_test: dict[tuple, bool] = {}
    for c in sent:
        if c.instance_id not in removed_ids:
            continue
        # A recognized AI dependency is a deterministic manifest fact; never
        # let the agent omitting it become a removal signal for that package.
        if _is_protected_dependency(c):
            continue
        key = _consolidation_key(c)
        weak = _is_import_only_candidate(c)
        test = _is_test_file(c.file_path)
        if key in restrict_import:
            restrict_import[key] = restrict_import[key] and weak
            restrict_test[key] = restrict_test[key] and test
        else:
            restrict_import[key] = weak
            restrict_test[key] = test
    if not restrict_import:
        return received

    lookup_pool = all_candidates if all_candidates is not None else received
    drop_ids: set[str] = set()
    for c in lookup_pool:
        if _is_protected_dependency(c):
            continue
        key = _consolidation_key(c)
        if key not in restrict_import:
            continue
        # Drop only if the sibling satisfies every restriction the removal
        # carries: an import-only-restricted removal drops only import-only
        # siblings; a test-scoped-restricted removal drops only test-file
        # siblings.
        if restrict_import[key] and not _is_import_only_candidate(c):
            continue
        if restrict_test[key] and not _is_test_file(c.file_path):
            continue
        drop_ids.add(c.instance_id)

    result = [c for c in lookup_pool if c.instance_id not in drop_ids]
    dropped = len(lookup_pool) - len(result)
    if dropped:
        weak_n = sum(1 for v in restrict_import.values() if v)
        test_n = sum(1 for v in restrict_test.values() if v)
        _LOGGER.info(
            "Removal propagation: dropped %d additional component(s) "
            "(%d key(s); %d import-only-restricted, %d test-scoped)",
            dropped,
            len(restrict_import),
            weak_n,
            test_n,
        )
    return result


def _evidence_gate(
    before_agentic: list["AIComponent"],
    after_agentic: list["AIComponent"],
    *,
    scan_paths: list[str] | None = None,
) -> list["AIComponent"]:
    """Remove post-agentic components that lack hard evidence.

    Three checks:

    1. **model + kb_enrichment**: If the name does not resolve in any model
       registry (LiteLLM, built-in regex, HuggingFace), auto-remove.  This
       does NOT touch ``model_detector`` detections (string literals from
       code) — only class-name-inferred components from ``kb_enrichment``.

    2. **memory + kb_enrichment**: If the agent kept the component unchanged
       (no enrichment, no reclassification), auto-remove.  A genuine memory
       component should have been enriched with specifics.

    3. **agent / agent_proxy symmetric check**: Any post-agentic AGENT or
       AGENT_PROXY must carry a verifiable ``agent_evidence`` payload in
       ``metadata`` when ANY of the following is true:

       * Structural origin (``discovery == "structural_react_loop"``).
       * Type was flipped to agent/agent_proxy by the LLM
         (``orig.component_type != c.component_type``).
       * Import-only candidate from ``kb_enrichment_scanner``
         (``metadata.import_statement`` is set and ``metadata.call_pattern``
         is not). These are weak class-name matches that need evidence
         the import is actually exercised as an LLM-driven loop.

       The payload is re-verified against the on-disk source using the
       same offline gate that Phase 6 uses for LLM verdicts. KB /
       framework scanner detections that come from real callsite
       emissions (``metadata.call_pattern`` set) are left untouched —
       their authority comes from the framework match itself.
    """
    from .agentic.middleware import _verify_agent_evidence
    from .models.enums import DetectionSource
    from .scanners.model_detector import registry_lookup

    before_map: dict[str, "AIComponent"] = {c.instance_id: c for c in before_agentic}
    allowed_roots = list(scan_paths or [])

    _AGENT_TYPES: frozenset[AIComponentType] = frozenset(
        {
            AIComponentType.AGENT,
            AIComponentType.AGENT_PROXY,
        }
    )

    result: list["AIComponent"] = []
    gate_removed = 0
    agent_evidence_removed = 0
    for c in after_agentic:
        orig = before_map.get(c.instance_id)
        if orig is None:
            result.append(c)
            continue

        if (
            c.component_type == AIComponentType.MODEL
            and orig.detection_source == DetectionSource.KB_ENRICHMENT
            and registry_lookup(c.name) is None
        ):
            _LOGGER.info(
                "Evidence gate removed model '%s' (%s): "
                "not found in any model registry and detection_source=kb_enrichment",
                c.name,
                c.instance_id,
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
                c.name,
                c.instance_id,
            )
            gate_removed += 1
            continue

        if c.component_type in _AGENT_TYPES:
            discovery = (c.metadata or {}).get("discovery")
            is_structural = discovery == "structural_react_loop"
            type_flipped = orig.component_type != c.component_type
            import_only = _is_import_only_candidate(orig)
            if is_structural or type_flipped or import_only:
                raw_evidence = (c.metadata or {}).get("agent_evidence")
                ok, reason = _verify_agent_evidence(
                    raw_evidence,
                    allowed_roots=allowed_roots,
                )
                if not ok:
                    _LOGGER.warning(
                        "Evidence gate: dropping %s '%s' (%s) — %s "
                        "(structural=%s, type_flipped=%s, import_only=%s)",
                        c.component_type.value,
                        c.name,
                        c.instance_id,
                        reason,
                        is_structural,
                        type_flipped,
                        import_only,
                    )
                    agent_evidence_removed += 1
                    continue

        result.append(c)

    total_removed = gate_removed + agent_evidence_removed
    if total_removed:
        _LOGGER.info(
            "Evidence gate: removed %d component(s) "
            "(%d model/memory, %d agent/agent_proxy)",
            total_removed,
            gate_removed,
            agent_evidence_removed,
        )
    return result


# Path segments that unambiguously denote test-only code on their own.
_TEST_PATH_SEGMENTS: frozenset[str] = frozenset(
    {
        "tests",
        "test",
        "__tests__",
        "testdata",
        "test_data",
    }
)

# Segments that are commonly used for test scaffolding but ALSO occur in
# production trees (an OpenAPI ``spec/``, a shipped ``testing`` utility
# package, a ``fixtures/`` data loader). These count as test-only only when
# accompanied by another test signal — an unambiguous test directory ancestor
# or a test-style filename — so a production asset living under one of these
# directories is not silently dropped from the BOM.
_AMBIGUOUS_TEST_PATH_SEGMENTS: frozenset[str] = frozenset(
    {
        "spec",
        "testing",
        "fixtures",
    }
)


def _has_test_filename(path: "Path") -> bool:
    """True when the filename itself follows a test naming convention."""
    stem = path.stem.lower()
    return stem.startswith("test_") or stem.endswith("_test")


def _is_test_file(file_path: str) -> bool:
    """Return True when the file is clearly test-only by path or filename.

    Unambiguous test directories (``tests``, ``__tests__``, ``testdata`` …)
    and test-style filenames (``test_*`` / ``*_test``) classify on their own.
    The ambiguous segments (``spec``/``testing``/``fixtures``) classify only
    when a second test signal is present, so production code under those
    directory names is not misclassified as test-only.
    """
    from pathlib import Path

    path = Path(file_path)
    parts = set(path.parts)

    if parts & _TEST_PATH_SEGMENTS:
        return True

    has_test_name = _has_test_filename(path)
    if has_test_name:
        return True

    if parts & _AMBIGUOUS_TEST_PATH_SEGMENTS:
        # Only test-only when corroborated by an unambiguous test directory
        # ancestor (the filename signal is already handled above).
        return bool(parts & _TEST_PATH_SEGMENTS)

    return False


def _has_instantiation_marker(c: "AIComponent") -> bool:
    """True when the component was captured at an instantiation site.

    Instantiation-site markers include ``call_pattern`` (e.g. ``strands.Agent``),
    ``assigned_to`` / ``assigned_target`` (LHS of ``x = Agent()``), and
    ``kb_id`` (observation matched a KB class entry). Invocation sites
    (``x(...)``) and bare decorator/context uses generally lack these
    markers, so preferring instantiation markers yields a richer
    representative for consolidation collapses.
    """
    meta = c.metadata or {}
    return any(
        meta.get(key)
        for key in ("call_pattern", "assigned_to", "assigned_target", "kb_id")
    )


def _consolidate_components(
    components: list["AIComponent"],
) -> list["AIComponent"]:
    """Merge per-file-reference components into per-logical-asset components.

    Groups by (canonical_name, component_type) across the entire repo.
    The highest-confidence occurrence is kept; all others become entries
    in ``metadata["evidence"]`` with file, line, and service fields.
    Components where **every** occurrence is in a test file are tagged
    ``metadata["test_only"] = True``.

    Representative selection prefers instantiation-site components
    (``call_pattern`` / ``assigned_to`` / ``kb_id`` set) over invocation
    sites so the consolidated row surfaces the richer signal.
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
            merged_meta = dict(c.metadata)
            merged_meta.setdefault("evidence_count", 1)
            merged_meta.setdefault("evidence_files", [c.file_path])
            if _is_test_file(c.file_path):
                merged_meta["test_only"] = True
            c = c.model_copy(update={"metadata": merged_meta})
            result.append(c)
            continue

        best = max(
            group,
            key=lambda c: (
                c.heuristic_confidence,
                1 if _has_instantiation_marker(c) else 0,
                0 if _is_test_file(c.file_path) else 1,
                -c.line_number,
            ),
        )
        evidence = []
        evidence_files: list[str] = []
        seen_files: set[str] = set()
        all_test = True
        for c in group:
            is_test = _is_test_file(c.file_path)
            if not is_test:
                all_test = False
            if c.file_path not in seen_files:
                seen_files.add(c.file_path)
                evidence_files.append(c.file_path)
            if c is not best:
                evidence.append(
                    {
                        "file": c.file_path,
                        "line": c.line_number,
                        "service": _service_dir(c.file_path),
                        "test_only": is_test,
                    }
                )

        if not _is_test_file(best.file_path):
            all_test = False

        merged_meta = dict(best.metadata)
        merged_meta["evidence"] = evidence
        merged_meta["evidence_count"] = len(group)
        merged_meta["evidence_files"] = evidence_files
        merged_meta["consolidated_count"] = len(group)
        if all_test:
            merged_meta["test_only"] = True

        needs = any(c.needs_agentic for c in group)

        merged = best.model_copy(
            update={
                "metadata": merged_meta,
                "needs_agentic": needs,
            }
        )
        result.append(merged)

    return result


def _canonicalize_env_var_names(
    components: list["AIComponent"],
) -> list["AIComponent"]:
    """Rename MODEL / EMBEDDING components whose ``name`` still carries the
    env var label to the resolved literal model id.

    Deterministic scanners (``env_var_resolver``, ``config_scanner``,
    ``deployment_detector``) tag env-backed components with several
    naming conventions:

    - ``env:VAR``                 — env_var_resolver, config_scanner endpoint branch
    - ``env_model_VAR``           — config_scanner ``.env`` model branch
    - ``env_embedding_VAR``       — config_scanner ``.env`` embedding branch
    - ``dockerfile_env_VAR``      — config_scanner Dockerfile ``ENV`` branch
    - bare ``VAR``                — agentic stage and ``env_var_resolver`` post-resolve

    The Dockerfile branch records the key under ``metadata["env"]``; all
    other branches use ``metadata["env_var"]``. This gate accepts both
    shapes and every known name prefix, so any scanner that correctly
    pairs an env-var marker with a resolved literal in ``model_name``
    gets promoted to the literal in a single pass.
    """
    out: list[AIComponent] = []
    for c in components:
        if c.component_type not in (
            AIComponentType.MODEL,
            AIComponentType.EMBEDDING,
        ):
            out.append(c)
            continue

        meta = c.metadata or {}
        env_var = meta.get("env_var") or meta.get("env")
        model_name = c.model_name
        if (
            isinstance(env_var, str)
            and env_var
            and _strip_env_prefix(c.name) == env_var
            and isinstance(model_name, str)
            and model_name
            and model_name != c.name
            and not model_name.lower().startswith(("http://", "https://"))
            and not _ENV_PLACEHOLDER_RE.search(model_name)
        ):
            canon_meta = dict(meta)
            canon_meta.setdefault("env_var", env_var)
            out.append(
                c.model_copy(
                    update={
                        "name": model_name,
                        "metadata": canon_meta,
                    }
                )
            )
        else:
            out.append(c)
    return out


def _dedup_mcp_clients(
    components: list["AIComponent"],
) -> list["AIComponent"]:
    """Collapse MCP_CLIENT rows that point at the same ``(file, line)``.

    Multiple scanners can emit an MCP client component for the same call
    site — typically one from the MCP detector (basename-style name) and
    one from KB enrichment (wrapper-variable name). When both land on the
    same ``(file_path, line_number)`` we keep the richer representative:
    prefer components whose name is not a bare basename (e.g.
    ``mcp_integration_mcp_client`` beats ``stdio_client``) and whose
    metadata carries ``call_pattern`` / ``assigned_to`` markers.
    """

    def _rank(c: "AIComponent") -> tuple[int, int, float]:
        meta = c.metadata or {}
        name = c.name or ""
        basename_like = 1 if (name.endswith("_client") and "_" in name) else 0
        marker = 1 if _has_instantiation_marker(c) else 0
        return (marker, basename_like, c.heuristic_confidence)

    key_map: dict[tuple[str, int], int] = {}
    kept: list[AIComponent] = []
    for c in components:
        if c.component_type != AIComponentType.MCP_CLIENT:
            kept.append(c)
            continue
        key = (c.file_path, c.line_number)
        idx = key_map.get(key)
        if idx is None:
            key_map[key] = len(kept)
            kept.append(c)
            continue
        if _rank(c) > _rank(kept[idx]):
            kept[idx] = c
    return kept


def _filter_default_bom_scope_components(
    components: list["AIComponent"],
) -> list["AIComponent"]:
    """Remove components that only occur under test/fixture paths."""
    return [c for c in components if c.metadata.get("test_only") is not True]


def _filter_ai_only_dependency_components(
    components: list["AIComponent"],
) -> list["AIComponent"]:
    """Keep only AI-relevant dependency rows in the default BOM."""
    filtered: list[AIComponent] = []
    for comp in components:
        if comp.component_type != AIComponentType.DEPENDENCY:
            filtered.append(comp)
            continue
        if comp.metadata.get("known_ai_package") is True:
            filtered.append(comp)
            continue
        ecosystem = str(comp.metadata.get("ecosystem", "") or "")
        if ecosystem and is_known_ai_package(ecosystem, comp.name):
            meta = dict(comp.metadata)
            meta["known_ai_package"] = True
            filtered.append(comp.model_copy(update={"metadata": meta}))
    return filtered


def _filter_relationships_for_components(
    relationships: list["ComponentRelationship"],
    components: list["AIComponent"],
) -> list["ComponentRelationship"]:
    """Drop relationships that reference excluded components."""
    component_ids = {c.instance_id for c in components}
    component_names = {
        name.strip().lower()
        for c in components
        for name in (c.name, c.model_name or "")
        if name
    }

    def _endpoint_present(instance_id: str, name: str) -> bool:
        if instance_id:
            return instance_id in component_ids
        if name:
            return name.strip().lower() in component_names
        return False

    return [
        rel
        for rel in relationships
        if _endpoint_present(rel.source_instance_id, rel.source_name)
        and _endpoint_present(rel.target_instance_id, rel.target_name)
    ]


def _filter_risk_flags_for_default_scope(flags: list[Any]) -> list[Any]:
    """Drop risk flags whose evidence comes only from test/fixture files."""
    filtered: list[Any] = []
    for flag in flags:
        file_path = ""
        if isinstance(flag, dict):
            file_path = str(flag.get("file_path", "") or "")
        else:
            file_path = str(getattr(flag, "file_path", "") or "")
        if file_path and _is_test_file(file_path):
            continue
        filtered.append(flag)
    return filtered


def _vector_store_technology(c: "AIComponent") -> str | None:
    if c.component_type != AIComponentType.VECTOR_STORE:
        return None
    meta_tech = c.metadata.get("store_technology")
    if isinstance(meta_tech, str) and meta_tech.strip():
        return meta_tech.strip().lower()
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

        best = max(
            group,
            key=lambda c: (
                c.heuristic_confidence,
                0 if _is_test_file(c.file_path) else 1,
                -c.line_number,
            ),
        )
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
            evidence.append(
                {
                    "file": file_path,
                    "line": line_number,
                    "service": _service_dir(file_path),
                    "test_only": test_only,
                }
            )

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
            ev
            for ev in evidence
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

        merged.append(
            best.model_copy(
                update={
                    "metadata": merged_meta,
                    "needs_agentic": needs,
                }
            )
        )

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


def _rel_endpoint_key(instance_id: str, name: str) -> str:
    """Return a hybrid dedup key for one endpoint of a relationship.

    Scanner-produced edges populate ``instance_id``; LLM-produced edges
    (:mod:`aibom.agentic.middleware`, :mod:`aibom.agentic.agent`) leave it
    blank and only set ``name``. Using ``instance_id`` when present lets
    us distinguish identically-named components in different files, while
    the ``"name:"`` prefix on the fallback keeps the two key-spaces
    disjoint so a scanner edge cannot collide with an LLM edge that
    happens to reference the same literal name.
    """
    if instance_id:
        return f"id:{instance_id}"
    if name:
        return f"name:{name}"
    return ""


def _backfill_relationship_instance_ids(
    relationships: list[ComponentRelationship],
    components: list[AIComponent],
) -> list[ComponentRelationship]:
    """Deterministically populate blank endpoint ``instance_id``s on edges.

    The agentic/LLM path emits relationships with empty
    ``source_instance_id``/``target_instance_id`` even though every component
    carries a stable, deterministic ``instance_id``. Downstream consumers that
    link edges to components by id therefore drop those edges entirely.

    This pass resolves a blank endpoint by matching its ``*_name`` against the
    final component set, using the same rule as the deterministic resolver:
    assign the component's ``instance_id`` only when the name resolves to
    exactly one component (or when a model-name alias resolves uniquely).
    Ambiguous or unresolvable endpoints are left blank — no guessed ids, so
    the existing safe name-fallback behaviour is preserved. Resolution is
    fully deterministic and never depends on LLM output.
    """
    by_name: dict[str, list[AIComponent]] = {}
    for comp in components:
        if comp.name:
            by_name.setdefault(comp.name, []).append(comp)
        if comp.model_name and comp.model_name != comp.name:
            by_name.setdefault(comp.model_name, []).append(comp)

    def _resolve(name: str) -> str | None:
        candidates = by_name.get(name)
        if candidates and len(candidates) == 1:
            return candidates[0].instance_id
        return None

    result: list[ComponentRelationship] = []
    filled = 0
    for rel in relationships:
        updates: dict[str, Any] = {}
        if not rel.source_instance_id and rel.source_name:
            resolved = _resolve(rel.source_name)
            if resolved:
                updates["source_instance_id"] = resolved
        if not rel.target_instance_id and rel.target_name:
            resolved = _resolve(rel.target_name)
            if resolved:
                updates["target_instance_id"] = resolved
        if updates:
            filled += 1
            result.append(rel.model_copy(update=updates))
        else:
            result.append(rel)
    if filled:
        _LOGGER.info("Backfilled endpoint instance_ids on %d relationship(s)", filled)
    return result


def _resolve_relationship_types(
    relationships: list[ComponentRelationship],
    components: list[AIComponent],
) -> list[ComponentRelationship]:
    """Populate ``source_type``/``target_type`` from the component registry."""
    name_to_type: dict[str, AIComponentType] = {}
    for comp in components:
        name_to_type[comp.name] = comp.component_type
        if comp.model_name and comp.component_type.is_model_related:
            name_to_type[comp.model_name] = comp.component_type
    result: list[ComponentRelationship] = []
    for rel in relationships:
        updates: dict[str, Any] = {}
        if rel.source_type == AIComponentType.OTHER:
            resolved = name_to_type.get(rel.source_name)
            if resolved:
                updates["source_type"] = resolved
        if rel.target_type == AIComponentType.OTHER:
            resolved = name_to_type.get(rel.target_name)
            if resolved:
                updates["target_type"] = resolved
        result.append(rel.model_copy(update=updates) if updates else rel)
    return result


def _propagate_model_from_relationships(
    components: list[AIComponent],
    relationships: list[ComponentRelationship],
) -> list[AIComponent]:
    """For components with ``model_name=None``, resolve from relationships.

    Keys the lookup on ``source_instance_id`` when available (falling back
    to a ``"name:"``-prefixed name for LLM-produced edges with blank
    instance ids) so two identically-named components in different files
    don't all inherit the same ``model_name``.
    """
    model_targets: dict[str, str] = {}
    for rel in relationships:
        if rel.relationship_type in (
            RelationshipType.USES_EMBEDDING,
            RelationshipType.USES_MODEL,
        ):
            key = _rel_endpoint_key(rel.source_instance_id, rel.source_name)
            if key:
                model_targets[key] = rel.target_name
    result: list[AIComponent] = []
    for comp in components:
        if comp.model_name is None:
            target = (
                model_targets.get(f"id:{comp.instance_id}")
                if comp.instance_id
                else None
            )
            if target is None:
                target = model_targets.get(f"name:{comp.name}") if comp.name else None
            if target is not None:
                result.append(comp.model_copy(update={"model_name": target}))
                continue
        result.append(comp)
    return result


def _dedup_relationships(
    relationships: list[ComponentRelationship],
) -> list[ComponentRelationship]:
    """Dedup by ``(source_endpoint, target_endpoint, relationship_type)``.

    The endpoint key prefers ``instance_id`` over ``name`` so two
    identically-named components in different files are treated as
    distinct endpoints. See :func:`_rel_endpoint_key`.
    """
    seen: dict[tuple[str, str, str], ComponentRelationship] = {}
    for rel in relationships:
        key = (
            _rel_endpoint_key(rel.source_instance_id, rel.source_name),
            _rel_endpoint_key(rel.target_instance_id, rel.target_name),
            rel.relationship_type.value,
        )
        if key not in seen:
            seen[key] = rel
    return list(seen.values())


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
        agentic_max_consecutive_failures: int = 3,
        agentic_max_retry_seconds: int = 1200,
        agentic_cache_dir: str | Path | None = None,
        agentic_review_secrets: bool = False,
        include_code_snippets: bool = False,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        custom_catalog: CustomCatalogConfig | None = None,
        atr_enrichment: bool = False,
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
        self.agentic_max_consecutive_failures = agentic_max_consecutive_failures
        self.agentic_max_retry_seconds = agentic_max_retry_seconds
        self.agentic_cache_dir = (
            Path(agentic_cache_dir) if agentic_cache_dir is not None else None
        )
        self.include_code_snippets = include_code_snippets
        self.agentic_review_secrets = agentic_review_secrets
        self.progress_callback = progress_callback
        self.custom_catalog = custom_catalog
        self.atr_enrichment = atr_enrichment

    def _emit_progress(self, event: str, **payload: Any) -> None:
        """Send a best-effort progress event to the CLI."""
        if not self.progress_callback:
            return
        progress_event = {"event": event, **payload}
        self.progress_callback(progress_event)

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

        self._emit_progress("stage_started", stage="scan", total_stages=4)
        t0 = time.monotonic()
        components, relationships = self._stage_scan(ctx)
        elapsed = time.monotonic() - t0
        fc = cache_stats()
        timings.append(
            StageTiming(
                "scan",
                elapsed,
                f"{len(components)} components, {len(relationships)} relationships, "
                f"file cache {fc['hits']} hits / {fc['misses']} misses",
            )
        )
        self._emit_progress(
            "stage_completed",
            stage="scan",
            elapsed_s=elapsed,
            detail=timings[-1].detail,
        )

        if self.atr_enrichment:
            from .security_enrichment import enrich_components

            components = enrich_components(components, enabled=True)

        self._emit_progress("stage_started", stage="cross_ref", total_stages=4)
        t0 = time.monotonic()
        components, env_idx, pkg_idx, ext_deps = self._stage_cross_ref(components)
        elapsed = time.monotonic() - t0
        timings.append(
            StageTiming(
                "cross_ref",
                elapsed,
                f"{sum(len(v) for v in env_idx.env.values())} env vars, "
                f"{len(pkg_idx.packages)} packages, {len(ext_deps)} external deps",
            )
        )
        self._emit_progress(
            "stage_completed",
            stage="cross_ref",
            elapsed_s=elapsed,
            detail=timings[-1].detail,
        )

        self._emit_progress("stage_started", stage="agentic", total_stages=4)
        t0 = time.monotonic()
        components, relationships, agentic_flags = self._stage_agentic(
            components, relationships
        )
        elapsed = time.monotonic() - t0
        skipped = not self.llm_config
        timings.append(
            StageTiming(
                "agentic",
                elapsed,
                (
                    "skipped (no --llm-model)"
                    if skipped
                    else f"{len(agentic_flags)} risk flags"
                ),
            )
        )
        self._emit_progress(
            "stage_completed",
            stage="agentic",
            elapsed_s=elapsed,
            detail=timings[-1].detail,
        )

        self._emit_progress("stage_started", stage="assemble", total_stages=4)
        t0 = time.monotonic()
        components, agentic_count = self._stage_assemble(components)
        elapsed = time.monotonic() - t0
        timings.append(
            StageTiming(
                "assemble",
                elapsed,
                f"{len(components)} final, {agentic_count} agentic",
            )
        )
        self._emit_progress(
            "stage_completed",
            stage="assemble",
            elapsed_s=elapsed,
            detail=timings[-1].detail,
        )
        relationships = _resolve_relationship_types(relationships, components)
        components = _propagate_model_from_relationships(components, relationships)
        relationships = _dedup_relationships(relationships)
        relationships = _filter_relationships_for_components(relationships, components)
        agentic_flags = _filter_risk_flags_for_default_scope(agentic_flags)

        total_elapsed = time.monotonic() - pipeline_start

        tu = getattr(self, "_agentic_token_usage", None)
        return PipelineResult(
            components=components,
            relationships=relationships,
            agentic_risk_flags=agentic_flags,
            agentic_candidate_count=agentic_count,
            agentic_degraded_count=getattr(self, "_agentic_degraded_count", 0),
            env_index=env_idx,
            pkg_index=pkg_idx,
            external_deps=ext_deps,
            timings=timings,
            total_elapsed_s=total_elapsed,
            prompt_tokens=tu.prompt_tokens if tu else 0,
            completion_tokens=tu.completion_tokens if tu else 0,
            total_tokens=tu.total_tokens if tu else 0,
            cached_tokens=tu.cached_tokens if tu else 0,
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

        ai_pkgs = discover_ai_package_set(ctx)
        if ai_pkgs:
            _LOGGER.info(
                "Pass 1: discovered %d AI package(s) from manifests: %s",
                len(ai_pkgs),
                ", ".join(sorted(ai_pkgs)[:10]) + ("…" if len(ai_pkgs) > 10 else ""),
            )
        else:
            _LOGGER.debug("Pass 1: no AI packages found in manifests")

        ctx_pass2 = ctx.model_copy(update={"ai_package_set": ai_pkgs})

        idx = ctx_pass2.file_index()
        if idx:
            import asyncio

            from .scanners.file_cache import warm_cache_async

            all_paths = [e.path for entries in idx.values() for e in entries]
            self._emit_progress(
                "file_cache_prep_started",
                files_total=len(all_paths),
            )
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
                warmed,
                len(all_paths),
            )
            self._emit_progress(
                "file_cache_prep_completed",
                files_total=len(all_paths),
                files_warmed=warmed,
            )

        return run_scanners(
            ctx_pass2,
            progress_callback=self.progress_callback,
        )

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
            _LOGGER.info("Detected %d external repo dependency(ies)", len(ext_deps))
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

        ensure_llm_runtime_available(
            self.llm_config["model"],
            provider=self.llm_config.get("provider"),
        )

        try:
            from .agentic.agent import _count_degraded, run_agentic_enrichment
        except ImportError as exc:
            raise ImportError(
                "Agentic classification requires the agentic runtime. "
                'Install with: uv tool install "cisco-aibom[agentic]"'
            ) from exc

        try:
            components, held_secrets = _partition_agentic_secrets(
                components, review_secrets=self.agentic_review_secrets
            )
            if held_secrets:
                # Held-back secrets are intentionally NOT merged back into the
                # final BOM: the success path below returns the agentic-enriched
                # set only. This preserves the prior post-agentic output, where
                # the agent already pruned SECRET detections as false positives.
                # Pass --agentic-review-secrets to route them through the agent
                # (and keep any it confirms) instead.
                _LOGGER.info(
                    "Held %d secret candidate(s) back from the agentic stage "
                    "(pass --agentic-review-secrets to include them).",
                    len(held_secrets),
                )
            deduped, fanout = _dedup_for_agentic(components)
            if len(deduped) < len(components):
                _LOGGER.info(
                    "Pre-agentic dedup: %d → %d components (%d context-free duplicates removed)",
                    len(components),
                    len(deduped),
                    len(components) - len(deduped),
                )

            model_str = self.llm_config["model"]
            user_sigs: AgentSignatureCatalog | None = (
                self.custom_catalog.agent_signatures
                if self.custom_catalog is not None
                else None
            )
            agent_catalog: AgentSignatureCatalog = resolve_catalog(user_sigs)
            if user_sigs is not None and not user_sigs.is_empty:
                _LOGGER.info(
                    "Agentic: merged user agent_signatures "
                    "(frameworks=%d, protocols=%d, anti_patterns=%d)",
                    len(user_sigs.frameworks),
                    len(user_sigs.protocols),
                    len(user_sigs.anti_patterns),
                )
            enriched, agentic_rels, agentic_flags, agentic_token_usage = (
                run_agentic_enrichment(
                    model_string=model_str,
                    deterministic_components=deduped,
                    deterministic_relationships=relationships,
                    scan_paths=self.scan_paths,
                    llm_config=self.llm_config,
                    batch_size=self.agentic_batch_size,
                    max_concurrent=self.agentic_concurrency,
                    fast_model=self.agentic_fast_model,
                    timeout_s=self.agentic_timeout,
                    max_consecutive_failures=self.agentic_max_consecutive_failures,
                    max_retry_seconds=self.agentic_max_retry_seconds,
                    cache_dir=self.agentic_cache_dir,
                    include_code_snippets=self.include_code_snippets,
                    agent_signature_catalog=agent_catalog,
                )
            )
            # Count degraded components from the raw agentic output, before
            # post-filters below may strip components (and their hints) from the
            # BOM.
            self._agentic_degraded_count = _count_degraded(enriched)
            deduped_ids = {c.instance_id for c in deduped}
            enriched_deduped_ids = {
                c.instance_id for c in enriched if c.instance_id in deduped_ids
            }
            new_components = [c for c in enriched if c.instance_id not in deduped_ids]
            pre_fanout_removed = deduped_ids - enriched_deduped_ids
            enriched_for_fanout = [c for c in enriched if c.instance_id in deduped_ids]
            enriched = _fanout_agentic_results(enriched_for_fanout, fanout)
            enriched = _propagate_removals(
                deduped,
                enriched,
                all_candidates=components,
                pre_fanout_removed_ids=pre_fanout_removed,
            )
            enriched = _reinstate_protected_dependencies(components, enriched)
            enriched = _evidence_gate(
                components,
                enriched,
                scan_paths=self.scan_paths,
            )
            if new_components:
                enriched = enriched + new_components

            from .agentic.middleware import (
                _drop_env_placeholder_identifiers,
                _reject_class_name_models,
                _remove_unresolved_embedders,
            )

            enriched = _reject_class_name_models(enriched)
            all_rels = relationships + agentic_rels
            enriched = _remove_unresolved_embedders(enriched, all_rels)
            enriched = _drop_env_placeholder_identifiers(enriched)
            all_rels = _backfill_relationship_instance_ids(all_rels, enriched)
            self._agentic_token_usage = agentic_token_usage
            return enriched, all_rels, agentic_flags

        except ImportError:
            raise
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

        before_canon = len(components)
        components = _canonicalize_env_var_names(components)
        if before_canon:
            _LOGGER.debug(
                "Env-var canonicalization applied to %d components",
                before_canon,
            )

        before_mcp = len(components)
        components = _dedup_mcp_clients(components)
        if before_mcp != len(components):
            _LOGGER.info(
                "MCP client dedup: %d → %d components (-%d)",
                before_mcp,
                len(components),
                before_mcp - len(components),
            )

        # Normalize MODEL → EMBEDDING for components whose name is a
        # registry-known embedding identifier (e.g. ``text-embedding-3-large``
        # is labelled ``mode=embedding`` in LiteLLM). This happens BEFORE
        # consolidation so duplicates produced by scanners that disagreed on
        # type (one said MODEL, another said EMBEDDING) collapse into a single
        # consolidation key ``(name, EMBEDDING)``. Authority: the model
        # registry. See :func:`is_known_embedding_model_name`.
        reclassified = 0
        for component in components:
            if (
                component.component_type == AIComponentType.MODEL
                and is_known_embedding_model_name(component.name)
            ):
                component.component_type = AIComponentType.EMBEDDING
                reclassified += 1
        if reclassified:
            _LOGGER.info(
                "Embedding reclassification: %d model(s) relabeled as embedding "
                "via registry (mode=embedding)",
                reclassified,
            )

        before = len(components)
        components = _consolidate_components(components)
        after = len(components)
        if before != after:
            _LOGGER.info(
                "Consolidation: %d → %d components (-%d duplicates)",
                before,
                after,
                before - after,
            )

        before_vs = len(components)
        components = _consolidate_vector_stores(components)
        after_vs = len(components)
        if before_vs != after_vs:
            _LOGGER.info(
                "Vector store dedup: %d → %d components (-%d)",
                before_vs,
                after_vs,
                before_vs - after_vs,
            )

        before_td = len(components)
        components = _dedup_tool_vs_vector_store(components)
        after_td = len(components)
        if before_td != after_td:
            _LOGGER.info(
                "Tool/vector_store priority dedup: %d → %d (-%d)",
                before_td,
                after_td,
                before_td - after_td,
            )

        before_scope = len(components)
        components = _filter_default_bom_scope_components(components)
        after_scope = len(components)
        if before_scope != after_scope:
            _LOGGER.info(
                "Default BOM scope: %d → %d components (-%d test/fixture-only)",
                before_scope,
                after_scope,
                before_scope - after_scope,
            )

        before_deps = len(components)
        components = _filter_ai_only_dependency_components(components)
        after_deps = len(components)
        if before_deps != after_deps:
            _LOGGER.info(
                "AI-only dependency policy: %d → %d components (-%d non-AI dependencies)",
                before_deps,
                after_deps,
                before_deps - after_deps,
            )

        agentic_count = sum(1 for c in components if c.needs_agentic)

        if self.strict:
            components = [c for c in components if not c.needs_agentic]
            _LOGGER.info("Strict mode: filtered %d agentic candidates", agentic_count)

        return components, agentic_count
