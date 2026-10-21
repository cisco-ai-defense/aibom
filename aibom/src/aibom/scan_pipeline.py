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

import ast
import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
from .agentic_telemetry import AgenticTelemetry
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
from .models.enums import AIComponentType, DetectionSource, RelationshipType
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
    agentic_input_count: int = 0
    agentic_reviewed_count: int = 0
    agentic_removed_count: int = 0
    agentic_status: str = "skipped"
    agentic_degraded_count: int = 0
    agentic_degradation_reasons: dict[str, int] = field(default_factory=dict)
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
            clone.agentic_hint = c.agentic_hint
            clone.decision_annotation = (
                c.decision_annotation.model_copy(
                    deep=True,
                    update={
                        "evidence_locations": [],
                        "code_snippet": None,
                    },
                )
                if c.decision_annotation is not None
                else None
            )
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
    # A tool anchored at an import line is that file's reference: its removal
    # reaches only the same file.
    restrict_files: dict[tuple, set[str]] = {}
    for c in sent:
        if c.instance_id in removed_ids and _is_import_anchored_tool(c):
            restrict_files.setdefault(_consolidation_key(c), set()).add(c.file_path)
    drop_ids: set[str] = set()
    for c in lookup_pool:
        if _is_protected_dependency(c):
            continue
        key = _consolidation_key(c)
        if key not in restrict_import:
            continue
        if key in restrict_files and c.file_path not in restrict_files[key]:
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

    # ``lookup_pool`` is deliberately wider than ``received`` so removals can
    # be resolved against siblings that pre-agentic deduplication collapsed.
    # The result must still be filtered from ``received``: rebuilding it from
    # the original candidates would discard successful verdicts, repaired
    # evidence, and degradation reasons attached by the agentic stage.
    result = [c for c in received if c.instance_id not in drop_ids]
    dropped = len(received) - len(result)
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
        if c.agentic_hint:
            # A failed-open candidate did not receive an agentic verdict.
            # Preserve it as unreviewed instead of applying evidence rules
            # intended for affirmative agent decisions.
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


_REVIEWED_COMPONENT_DECISIONS = frozenset({"confirmed", "added"})
_MISSING_AGENTIC_VERDICT = "missing_agentic_verdict"


def _mark_missing_agentic_verdicts(
    components: list["AIComponent"],
    *,
    input_ids: set[str],
) -> list["AIComponent"]:
    """Mark retained inputs without a usable component verdict as degraded."""

    result: list["AIComponent"] = []
    for component in components:
        if component.instance_id not in input_ids:
            result.append(component)
            continue
        annotation = component.decision_annotation
        decision = annotation.decision.strip().lower() if annotation else ""
        if decision in _REVIEWED_COMPONENT_DECISIONS:
            result.append(
                component.model_copy(
                    update={"needs_agentic": False, "agentic_hint": ""}
                )
            )
            continue
        result.append(
            component.model_copy(
                update={
                    "needs_agentic": True,
                    "agentic_hint": (
                        component.agentic_hint or _MISSING_AGENTIC_VERDICT
                    ),
                }
            )
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


_DEF_LINE_RE = re.compile(r"\s*(async\s+)?def\s")
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _scope_statements(node: ast.AST):
    """Yield the nodes of *node*'s own scope, not descending into nested scopes."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, _SCOPE_NODES + (ast.Lambda,)):
            yield from _scope_statements(child)


def _scope_bindings(node: ast.AST) -> dict[str, list[tuple[int, str]]]:
    """Name -> [(line, kind)] for every binding in *node*'s own scope.

    ``kind`` is ``"def"`` for function definitions, ``"class"`` for class
    definitions, ``"import"`` for imports, ``"call"`` for a name assigned the
    result of a call, and ``"other"`` for parameters and other name bindings.
    """
    out: dict[str, list[tuple[int, str]]] = {}
    call_targets: set[int] = set()
    for child in _scope_statements(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(child.name, []).append((child.lineno, "def"))
        elif isinstance(child, ast.ClassDef):
            out.setdefault(child.name, []).append((child.lineno, "class"))
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                name = (alias.asname or alias.name).split(".")[0]
                out.setdefault(name, []).append((child.lineno, "import"))
        elif isinstance(child, (ast.Assign, ast.AnnAssign)) and isinstance(
            child.value, ast.Call
        ):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            call_targets.update(id(t) for t in targets)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            kind = "call" if id(child) in call_targets else "other"
            out.setdefault(child.id, []).append((child.lineno, kind))
        elif isinstance(child, ast.ExceptHandler) and child.name:
            out.setdefault(child.name, []).append((child.lineno, "other"))
        elif isinstance(child, ast.arg):
            # parameters are bound before any statement in the body runs
            out.setdefault(child.arg, []).append((0, "other"))
    return out


class _FileScopes:
    """Lexical scopes of one Python file, for resolving which ``def`` a name refers to."""

    def __init__(self, tree: ast.Module) -> None:
        # (start, end, is_class, bindings); the module scope spans every line
        self.scopes: list[tuple[int, int, bool, dict[str, list[tuple[int, str]]]]] = [
            (0, 10**9, False, _scope_bindings(tree))
        ]
        for node in ast.walk(tree):
            if isinstance(node, _SCOPE_NODES):
                self.scopes.append(
                    (
                        node.lineno,
                        getattr(node, "end_lineno", None) or node.lineno,
                        isinstance(node, ast.ClassDef),
                        _scope_bindings(node),
                    )
                )

    def def_line_for(self, name: str, line: int) -> int | None:
        """Line of the ``def`` that *name* refers to at *line*, or None if not a local def."""
        bound = self.binding_for(name, line)
        return bound[0] if bound and bound[1] == "def" else None

    def binding_for(self, name: str, line: int) -> tuple[int, str] | None:
        """(line, kind) of the binding *name* refers to at *line*; None if unbound."""
        chain = sorted(
            (sc for sc in self.scopes if sc[0] <= line <= sc[1]),
            key=lambda sc: sc[0],
            reverse=True,
        )
        for depth, (_, _, is_class, bindings) in enumerate(chain):
            if depth and is_class:
                continue  # class bodies are not visible to nested functions
            found = bindings.get(name)
            if not found:
                continue
            if depth == 0:
                # own scope: the binding in effect at the registration line
                preceding = [b for b in found if b[0] <= line]
                if not preceding:
                    return None  # bound only later in this scope
                return max(preceding)
            # enclosing scope has finished running when this code executes
            return max(found)
        return None


# Registration calls that take the tool function itself (``mcp.add_tool(fn)``), as
# recorded by the scanner's argument-aware pass in ``extracted_via``.
_REGISTRATION_CALL_PREFIXES = ("add_tool(", "register_tool(", "register_function(")
_MAX_REEXPORT_DEPTH = 3


def _is_registration_call_tool(comp: AIComponent) -> bool:
    via = str((comp.metadata or {}).get("extracted_via") or "")
    return via.startswith(_REGISTRATION_CALL_PREFIXES)


def _is_tools_kwarg_tool(comp: AIComponent) -> bool:
    """True for a tool the scanner read from an agent's ``tools=[...]`` list."""
    return (comp.metadata or {}).get("extracted_via") == "agent_tools_kwarg"


def _is_import_anchored_tool(comp: AIComponent) -> bool:
    """True for a tool anchored at the import line of the file that uses it."""
    return (
        comp.component_type == AIComponentType.TOOL
        and (comp.metadata or {}).get("anchor") == "import"
    )


# Registration calls whose argument may be a loop variable over a tool inventory;
# ``FunctionTool.from_function(tool)`` wraps it before ``add_tool``.
_LOOP_REGISTRATION_PREFIXES = _REGISTRATION_CALL_PREFIXES + ("from_function(",)


def _expand_loop_registered_tools(
    components: list[AIComponent], scan_paths: list[str]
) -> list[AIComponent]:
    """Replace a tool registered in a loop with one tool per function it registers.

    ``for tool in inventory: mcp.add_tool(tool)`` yields a single scanner row
    named after the loop variable. When the inventory can be followed to the
    functions it lists, each becomes a tool anchored at its ``def``; when it
    cannot, the loop-variable row is dropped rather than reported as a tool.
    """
    from .registration_flow import LoopRegistrationResolver

    roots = sorted(
        (Path(p).resolve() for p in scan_paths if Path(p).is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    resolvers: dict[Path, LoopRegistrationResolver] = {}
    out: list[AIComponent] = []
    expanded = dropped = 0
    for comp in components:
        via = str((comp.metadata or {}).get("extracted_via") or "")
        path = Path(comp.file_path or "")
        if (
            comp.component_type != AIComponentType.TOOL
            or not comp.line_number
            or path.suffix != ".py"
            or not via.startswith(_LOOP_REGISTRATION_PREFIXES)
        ):
            out.append(comp)
            continue
        resolved = path.resolve()
        root = next((r for r in roots if resolved.is_relative_to(r)), None)
        if root is None:
            out.append(comp)
            continue
        resolver = resolvers.setdefault(root, LoopRegistrationResolver(root))
        refs = resolver.resolve(resolved, comp.line_number, comp.name)
        if refs is None:
            out.append(comp)
            continue
        if not refs:
            dropped += 1
            continue
        for ref in refs:
            file_path = str(ref.path)
            out.append(
                comp.model_copy(
                    update={
                        "name": ref.name,
                        "file_path": file_path,
                        "line_number": ref.line,
                        "instance_id": f"{ref.name}_{file_path}_{ref.line}",
                        "metadata": {
                            **(comp.metadata or {}),
                            "registered_in_loop": True,
                            "reported_file": comp.file_path,
                            "reported_line": comp.line_number,
                        },
                    }
                )
            )
        expanded += 1
    if expanded or dropped:
        _LOGGER.info(
            "Loop-registered tools: expanded %d registration(s) into %d tool(s); "
            "dropped %d unresolved",
            expanded,
            len(out) - (len(components) - expanded - dropped),
            dropped,
        )
    return out


def _import_target_file(importer: Path, module: str, level: int) -> Path | None:
    """File that ``from <module> import ...`` in *importer* refers to, if local."""
    parts = module.split(".") if module else []
    if level:
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        bases = [base]
    else:
        bases = list(importer.parents)
    for base in bases:
        target = base.joinpath(*parts)
        candidates = [target / "__init__.py"]
        if parts:
            candidates.insert(0, target.with_suffix(".py"))
        for cand in candidates:
            if cand.is_file():
                return cand
    return None


def _imported_def(
    path: Path, name: str, trees: dict[Path, ast.Module | None], depth: int = 0
) -> tuple[Path, int] | None:
    """(file, line) of the ``def`` a module-level ``from ... import name`` binds to.

    Follows re-exports through package ``__init__`` files up to a small depth.
    """
    if depth > _MAX_REEXPORT_DEPTH:
        return None
    if path not in trees:
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            trees[path] = ast.parse(source)
        except (OSError, SyntaxError, ValueError):
            trees[path] = None
    tree = trees[path]
    if tree is None:
        return None
    binding: ast.stmt | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                binding = node
        elif isinstance(node, ast.ImportFrom):
            if any((a.asname or a.name) == name for a in node.names):
                binding = node
    if isinstance(binding, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return (path, binding.lineno) if depth else None
    if not isinstance(binding, ast.ImportFrom):
        return None
    alias = next(a for a in binding.names if (a.asname or a.name) == name)
    target = _import_target_file(path, binding.module or "", binding.level)
    if target is None:
        return None
    return _imported_def(target, alias.name, trees, depth + 1)


def _align_tool_anchors(components: list[AIComponent]) -> list[AIComponent]:
    """Anchor Python tools at their defining ``def`` statement.

    Scanners report tools at the decorator line or at the registration call
    (``Agent(tools=[fn])``, ``add_tool(fn)``). A decorator-line anchor moves to the
    function it decorates. A registered tool moves to the same-file ``def`` that the
    name is bound to at the registration line, following Python scoping; imports,
    assignments and definitions that only appear later keep the registration line.
    A tool registered through ``add_tool(fn)`` / ``register_tool(fn)`` /
    ``register_function(fn)`` whose name is imported follows the import to the
    ``def`` in the defining module. A tool handed to an agent's ``tools=[...]``
    under an imported name is anchored at that import line; one bound to neither
    an import, a ``def``, a class nor a call result (a local list of tools, a
    parameter) is dropped.
    """
    trees: dict[Path, ast.Module | None] = {}
    cache: dict[str, tuple[_FileScopes | None, list[str]]] = {}

    def file_info(path: str) -> tuple[_FileScopes | None, list[str]]:
        if path not in cache:
            scopes: _FileScopes | None = None
            lines: list[str] = []
            try:
                source = Path(path).read_text(encoding="utf-8", errors="ignore")
                lines = source.splitlines()
                scopes = _FileScopes(ast.parse(source))
            except (OSError, SyntaxError, ValueError):
                pass
            cache[path] = (scopes, lines)
        return cache[path]

    moved = 0
    dropped: set[int] = set()
    for comp in components:
        if (
            comp.component_type != AIComponentType.TOOL
            or not comp.line_number
            or not str(comp.file_path or "").endswith(".py")
        ):
            continue
        scopes, lines = file_info(str(comp.file_path))
        target: int | None = None
        idx = comp.line_number - 1
        if 0 <= idx < len(lines) and lines[idx].lstrip().startswith("@"):
            j = idx
            while j < len(lines) and not _DEF_LINE_RE.match(lines[j]):
                j += 1
            if j < len(lines):
                target = j + 1
        elif scopes is not None:
            target = scopes.def_line_for(comp.name, comp.line_number)
            if target is None and _is_tools_kwarg_tool(comp):
                bound = scopes.binding_for(comp.name, comp.line_number)
                if bound and bound[1] == "import":
                    target = bound[0]
                    comp.metadata = {**(comp.metadata or {}), "anchor": "import"}
                elif not bound or bound[1] not in ("class", "call"):
                    dropped.add(id(comp))
                    continue
            if target is None and _is_registration_call_tool(comp):
                found = _imported_def(Path(comp.file_path), comp.name, trees)
                if found is not None:
                    comp.metadata = {
                        **(comp.metadata or {}),
                        "reported_file": comp.file_path,
                        "reported_line": comp.line_number,
                    }
                    comp.file_path = str(found[0])
                    comp.line_number = found[1]
                    moved += 1
                    continue
        if target and target != comp.line_number:
            comp.metadata = {**(comp.metadata or {}), "reported_line": comp.line_number}
            comp.line_number = target
            moved += 1
    if moved:
        _LOGGER.info(
            "Tool anchor alignment: moved %d tool(s) to their defining line", moved
        )
    if dropped:
        _LOGGER.info(
            "Tool anchor alignment: dropped %d tools= name(s) bound to no tool",
            len(dropped),
        )
        components = [c for c in components if id(c) not in dropped]
    return components


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
        if _is_import_anchored_tool(c):
            # Each file imports its own reference to the tool.
            key = key + (c.file_path,)
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

        # Among equally ranked agent sites spread over several files, the
        # file that creates the agent most often is its factory; anchor there
        # rather than at whichever file has the lowest line number.
        per_file: dict[str, int] = {}
        if group[0].component_type.value == "agent":
            for c in group:
                per_file[c.file_path] = per_file.get(c.file_path, 0) + 1
        best = max(
            group,
            key=lambda c: (
                c.heuristic_confidence,
                1 if _has_instantiation_marker(c) else 0,
                0 if _is_test_file(c.file_path) else 1,
                per_file.get(c.file_path, 0),
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
                        **(
                            {"reported_line": c.metadata["reported_line"]}
                            if (c.metadata or {}).get("reported_line")
                            else {}
                        ),
                        **(
                            {"reported_file": c.metadata["reported_file"]}
                            if (c.metadata or {}).get("reported_file")
                            else {}
                        ),
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


# Bedrock model ids: an optional cross-region inference-profile prefix, the
# provider, the base model id, and a version suffix
# (``us.anthropic.claude-3-5-haiku-20241022-v1:0``).
_BEDROCK_MODEL_ID_RE = re.compile(
    r"^(?P<region>(?:us|us-gov|eu|apac|ap|ca|jp|au|global)\.)?"
    r"(?:anthropic|amazon|meta|mistral|cohere|ai21|deepseek|writer|qwen|openai)\."
    r"(?P<base>[a-z0-9][a-z0-9.\-]*?)(?P<version>-v\d+(?::\d+)?)?$",
    re.IGNORECASE,
)


def _bedrock_base_model_id(model_id: str) -> str | None:
    """Base model id of a Bedrock model or inference-profile id, else ``None``."""
    m = _BEDROCK_MODEL_ID_RE.match(model_id.strip())
    if not m or not (m.group("region") or m.group("version")):
        return None
    base = m.group("base")
    return base if any(ch.isdigit() for ch in base) else None


def _canonicalize_bedrock_model_ids(
    components: list["AIComponent"],
) -> list["AIComponent"]:
    """Name MODEL / EMBEDDING rows that carry a Bedrock id by their base model id.

    ``us.anthropic.claude-3-5-haiku-20241022-v1:0`` and
    ``anthropic.claude-3-5-haiku-20241022-v1:0`` are the Bedrock routes to
    ``claude-3-5-haiku-20241022``. The Bedrock id is kept in
    ``metadata["bedrock_model_id"]``.
    """
    for c in components:
        if c.component_type not in (AIComponentType.MODEL, AIComponentType.EMBEDDING):
            continue
        base = _bedrock_base_model_id(c.name)
        if not base:
            continue
        c.metadata = {**(c.metadata or {}), "bedrock_model_id": c.name}
        if c.model_name in (None, c.name):
            c.model_name = base
        c.name = base
    return components


def _is_unresolved_model_class(c: "AIComponent") -> bool:
    """True for a MODEL row that names a client class, not a model.

    The scanner reports a provider class (``AnthropicModel(model_id=VAR)``)
    whose model id it could not read, either as a bare-client row or as a
    catalog class row; resolving the id is left to the LLM stage.
    """
    if c.component_type != AIComponentType.MODEL or c.model_name:
        return False
    meta = c.metadata or {}
    if meta.get("detection_method") == "bare_provider_client":
        return True
    return c.detection_source == DetectionSource.KB_ENRICHMENT and (
        c.kb_label == "class" and c.kb_concept == "model"
    )


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

    # Keyed on location as well as technology: separate stores of the same
    # technology (e.g. two FAISS indexes) are distinct assets; only rows that
    # describe the same store site are folded together.
    tech_groups: OrderedDict[tuple[str, str, int], list["AIComponent"]] = (
        OrderedDict()
    )
    rest: list["AIComponent"] = []

    for c in components:
        tech = _vector_store_technology(c)
        if tech is None:
            rest.append(c)
        else:
            group_key = (tech, c.file_path, c.line_number)
            tech_groups.setdefault(group_key, []).append(c)

    merged: list["AIComponent"] = list(rest)

    for (tech, _file, _line), group in tech_groups.items():
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
        agentic_telemetry: AgenticTelemetry | None = None,
        invoke_callback_factory: Callable[[], Any] | None = None,
        telemetry_source_id: str | None = None,
        telemetry_source_kind: str = "unknown",
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
        self.agentic_telemetry = agentic_telemetry
        self.invoke_callback_factory = invoke_callback_factory
        self.telemetry_source_id = telemetry_source_id
        self.telemetry_source_kind = telemetry_source_kind
        self._agentic_degraded_count = 0
        self._agentic_degradation_reasons: dict[str, int] = {}
        self._agentic_reviewed_count = 0
        self._agentic_removed_count = 0
        self._agentic_telemetry_degraded_count: int | None = None
        self._agentic_token_usage: Any | None = None
        self._agentic_input_count = 0
        self._agentic_input_snapshot: list[AIComponent] = []
        self._agentic_output_snapshot: list[AIComponent] = []
        self._agentic_stage_input_ids: set[str] = set()
        self._agentic_degraded_ids: set[str] = set()
        self._agentic_added_relationship_count = 0
        self._agentic_raw_risk_count = 0

    def _telemetry_enabled(self) -> bool:
        """Return whether source-boundary telemetry can actually emit.

        Tests may pass a duck-typed recorder without an ``enabled`` property;
        a real disabled/no-op telemetry object must not add copy/diff work to
        the normal scan path.
        """

        telemetry = self.agentic_telemetry
        if telemetry is None:
            return False
        try:
            return bool(getattr(telemetry, "enabled", True))
        except Exception:
            return False

    def _emit_progress(self, event: str, **payload: Any) -> None:
        """Send a best-effort progress event to the CLI."""
        if not self.progress_callback:
            return
        progress_event = {"event": event, **payload}
        self.progress_callback(progress_event)

    def run(self) -> PipelineResult:
        clear_cache()
        self._agentic_degraded_count = 0
        self._agentic_degradation_reasons = {}
        self._agentic_reviewed_count = 0
        self._agentic_removed_count = 0
        self._agentic_telemetry_degraded_count = None
        self._agentic_token_usage = None
        self._agentic_input_count = 0
        self._agentic_input_snapshot = []
        self._agentic_output_snapshot = []
        self._agentic_stage_input_ids = set()
        self._agentic_degraded_ids = set()
        self._agentic_added_relationship_count = 0
        self._agentic_raw_risk_count = 0
        pipeline_started_at = datetime.now(timezone.utc)
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
        telemetry_enabled = self._telemetry_enabled()
        if telemetry_enabled and not self.llm_config:
            # A skipped summary still reports the deterministic/final boundary,
            # but never attributes those rows to an LLM decision.
            self._agentic_input_snapshot = [
                component.model_copy(deep=True) for component in components
            ]
            self._agentic_stage_input_ids = {
                component.instance_id for component in components
            }
            self._agentic_input_count = len(self._agentic_input_snapshot)
        components, relationships, agentic_flags = self._stage_agentic(
            components, relationships
        )
        if telemetry_enabled and not self.llm_config:
            # Preserve the actual Stage-3 boundary. Stage 4 can legitimately
            # canonicalize, filter, or consolidate deterministic findings, but
            # those changes are not agentic decisions.
            self._agentic_output_snapshot = [
                component.model_copy(deep=True) for component in components
            ]
        elapsed = time.monotonic() - t0
        skipped = not self.llm_config
        telemetry_skipped = skipped or self._agentic_input_count == 0
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
        agentic_degraded_count = getattr(self, "_agentic_degraded_count", 0)
        if agentic_degraded_count:
            agentic_status = "degraded"
        elif not self.llm_config or self._agentic_input_count == 0:
            agentic_status = "skipped"
        else:
            agentic_status = "success"
        if telemetry_enabled and self.agentic_telemetry is not None:
            degraded_count = (
                self._agentic_telemetry_degraded_count
                if self._agentic_telemetry_degraded_count is not None
                else getattr(self, "_agentic_degraded_count", 0)
            )
            before_by_id = {
                component.instance_id: component
                for component in self._agentic_input_snapshot
            }
            after_by_id = {
                component.instance_id: component
                for component in self._agentic_output_snapshot
            }
            common_ids = before_by_id.keys() & after_by_id.keys()
            degraded_ids = common_ids & self._agentic_degraded_ids
            decided_common_ids = common_ids - degraded_ids
            removed_ids = (
                before_by_id.keys() - after_by_id.keys()
            ) - self._agentic_degraded_ids
            reclassified = sum(
                1
                for component_id in decided_common_ids
                if before_by_id[component_id].component_type
                != after_by_id[component_id].component_type
            )
            enrichment_fields = (
                "model_name",
                "embedding_model",
                "description",
                "text",
                "framework",
                "detection_source",
                "heuristic_confidence",
                "transport",
                "config_source",
                "storage_uri",
                "dataset_source",
                "skill_format",
                "hyperparameters",
                "training_info",
                "metrics",
                "kb_concept",
                "kb_label",
                "sdk_version",
                "metadata",
            )
            enriched_count = sum(
                1
                for component_id in decided_common_ids
                if before_by_id[component_id].component_type
                == after_by_id[component_id].component_type
                and any(
                    getattr(before_by_id[component_id], field_name)
                    != getattr(after_by_id[component_id], field_name)
                    for field_name in enrichment_fields
                )
            )
            agentic_summary_decisions = {
                "kept": max(0, len(decided_common_ids) - reclassified - enriched_count),
                "enriched": enriched_count,
                "removed": len(removed_ids),
                "reclassified": reclassified,
                "discovered": len(
                    (
                        after_by_id.keys()
                        - (self._agentic_stage_input_ids or set(before_by_id.keys()))
                    )
                    - self._agentic_degraded_ids
                ),
                "relationships": self._agentic_added_relationship_count,
                "risk_findings": self._agentic_raw_risk_count,
                "degraded": degraded_count,
            }
            if telemetry_skipped:
                agentic_summary_decisions = {
                    key: 0 for key in agentic_summary_decisions
                }
            self.agentic_telemetry.record_summary(
                source_id=self.telemetry_source_id
                or (self.scan_paths[0] if self.scan_paths else "unknown"),
                source_kind=self.telemetry_source_kind,
                status=(
                    "skipped"
                    if telemetry_skipped
                    else ("degraded" if degraded_count else "success")
                ),
                candidate_count=len(self._agentic_input_snapshot),
                candidate_count_available=True,
                agentic_component_count=len(self._agentic_output_snapshot),
                # The action diff uses the raw agentic population above, while
                # this field deliberately reports the actual post-assemble BOM
                # count promised by the source-summary contract.
                final_component_count=len(components),
                degraded_candidate_count=degraded_count,
                duration_s=total_elapsed,
                created_at=pipeline_started_at,
                prompt_tokens=tu.prompt_tokens if tu else 0,
                completion_tokens=tu.completion_tokens if tu else 0,
                cached_tokens=tu.cached_tokens if tu else 0,
                decisions=agentic_summary_decisions,
            )
        return PipelineResult(
            components=components,
            relationships=relationships,
            agentic_risk_flags=agentic_flags,
            agentic_candidate_count=agentic_count,
            agentic_input_count=self._agentic_input_count,
            agentic_reviewed_count=self._agentic_reviewed_count,
            agentic_removed_count=self._agentic_removed_count,
            agentic_status=agentic_status,
            agentic_degraded_count=agentic_degraded_count,
            agentic_degradation_reasons=dict(
                self._agentic_degradation_reasons
            ),
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
        # Before classification, so classifiers see the registered functions.
        resolved = _expand_loop_registered_tools(resolved, self.scan_paths)

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
            from .agentic.agent import run_agentic_enrichment
        except ImportError as exc:
            raise ImportError(
                "Agentic classification requires the agentic runtime. "
                'Install with: uv tool install "cisco-aibom[agentic]"'
            ) from exc

        try:
            components, held_secrets = _partition_agentic_secrets(
                components, review_secrets=self.agentic_review_secrets
            )
            if self._telemetry_enabled():
                # Keep the complete post-partition population so fanout of a
                # deduplicated candidate is not misreported as discovery.
                self._agentic_stage_input_ids = {
                    component.instance_id for component in components
                }
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
            self._agentic_input_count = len(deduped)
            if self._telemetry_enabled():
                self._agentic_input_snapshot = [
                    component.model_copy(deep=True) for component in deduped
                ]
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
                    telemetry=(
                        self.agentic_telemetry if self._telemetry_enabled() else None
                    ),
                    telemetry_source_id=self.telemetry_source_id
                    or (self.scan_paths[0] if self.scan_paths else "unknown"),
                    invoke_callback_factory=self.invoke_callback_factory,
                )
            )
            self._agentic_added_relationship_count = len(agentic_rels)
            self._agentic_raw_risk_count = len(agentic_flags)
            deduped_ids = {c.instance_id for c in deduped}
            enriched = _mark_missing_agentic_verdicts(
                enriched,
                input_ids=deduped_ids,
            )
            enriched_deduped_ids = {
                c.instance_id for c in enriched if c.instance_id in deduped_ids
            }
            removed_ids = deduped_ids - enriched_deduped_ids
            new_components = [c for c in enriched if c.instance_id not in deduped_ids]
            pre_fanout_removed = removed_ids
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

            # Reconcile terminal outcomes at the guarded Stage-3 boundary.
            # Removal propagation and evidence filters can legitimately drop
            # a raw degraded result, so reporting from the unguarded LLM
            # response would overstate how many unresolved findings remain in
            # the BOM. Fanout siblings are mapped back to their one deduped
            # input representative to preserve the accounting invariant.
            candidate_instance_ids = {
                component.instance_id for component in components
            }
            enriched = _mark_missing_agentic_verdicts(
                enriched,
                input_ids=candidate_instance_ids,
            )
            enriched_by_id = {
                component.instance_id: component for component in enriched
            }
            reviewed_ids: set[str] = set()
            degraded_ids: set[str] = set()
            degradation_reasons: Counter[str] = Counter()
            for representative in deduped:
                siblings = fanout.get(
                    representative.instance_id,
                    [representative],
                )
                retained = [
                    enriched_by_id[sibling.instance_id]
                    for sibling in siblings
                    if sibling.instance_id in enriched_by_id
                ]
                if not retained:
                    continue
                if any(
                    component.decision_annotation is not None
                    and component.decision_annotation.decision.strip().lower()
                    in _REVIEWED_COMPONENT_DECISIONS
                    for component in retained
                ):
                    reviewed_ids.add(representative.instance_id)
                    continue
                degraded_ids.add(representative.instance_id)
                reason = next(
                    (
                        component.agentic_hint
                        for component in retained
                        if component.agentic_hint
                    ),
                    _MISSING_AGENTIC_VERDICT,
                )
                degradation_reasons[reason] += 1

            self._agentic_degraded_ids = degraded_ids
            self._agentic_degraded_count = len(degraded_ids)
            self._agentic_degradation_reasons = dict(degradation_reasons)
            self._agentic_reviewed_count = len(reviewed_ids)
            self._agentic_removed_count = (
                self._agentic_input_count
                - self._agentic_reviewed_count
                - self._agentic_degraded_count
            )
            self._agentic_token_usage = agentic_token_usage
            if self._telemetry_enabled():
                # Snapshot the true Stage-3 return boundary. Everything above
                # can remove, reinstate, or fan out components after the raw
                # agent response, and source-summary decisions must reflect
                # that guarded result rather than the unvalidated response.
                self._agentic_output_snapshot = [
                    component.model_copy(deep=True) for component in enriched
                ]
            return enriched, all_rels, agentic_flags

        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            exact_candidates = self._agentic_input_count or len(components)
            self._agentic_input_count = exact_candidates
            failure_hint = "agentic_stage_error"
            components = [
                component.model_copy(
                    update={
                        "needs_agentic": True,
                        "agentic_hint": failure_hint,
                    }
                )
                for component in components
            ]
            self._agentic_degraded_count = exact_candidates
            self._agentic_degradation_reasons = {
                failure_hint: exact_candidates
            }
            self._agentic_reviewed_count = 0
            self._agentic_removed_count = 0
            self._agentic_degraded_ids = {
                component.instance_id for component in components
            }
            if self._telemetry_enabled():
                self._agentic_telemetry_degraded_count = exact_candidates
                if not self._agentic_input_snapshot:
                    self._agentic_input_snapshot = [
                        component.model_copy(deep=True) for component in components
                    ]
                    self._agentic_input_count = len(self._agentic_input_snapshot)
                if not self._agentic_stage_input_ids:
                    self._agentic_stage_input_ids = {
                        component.instance_id for component in components
                    }
                self._agentic_output_snapshot = [
                    component.model_copy(deep=True) for component in components
                ]
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

        components = _align_tool_anchors(components)
        components = _canonicalize_bedrock_model_ids(components)

        if not self.llm_config:
            # No LLM stage will resolve a client class to its model id, so a
            # row named after the class would ship as a model as-is.
            before_cls = len(components)
            components = [c for c in components if not _is_unresolved_model_class(c)]
            if before_cls != len(components):
                _LOGGER.info(
                    "No-LLM: dropped %d model row(s) naming an unresolved client class",
                    before_cls - len(components),
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
