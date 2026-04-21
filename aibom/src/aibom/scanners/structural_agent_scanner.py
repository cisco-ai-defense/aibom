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

"""Structural agent-loop candidate scanner.

Closes the gap where a Python class clearly implements a ReAct-style
orchestration loop (iterative control flow + multiple distinct callees
+ conditional dispatch) but is **not** detected by the existing
framework-centric scanners because it does not inherit from a known
agent framework and is not wired up via a config file.

What it does
------------

For every Python source file reachable from the scan context this
scanner:

1. Parses the file with the shared :mod:`aibom.cst_parser` pipeline.
2. Builds an :class:`aibom.scanners.agent_evidence_builder.AgentEvidenceDossier`
   for every class definition using the merged
   :class:`aibom.agent_signatures.AgentSignatureCatalog`.
3. Emits an :class:`AIComponent` of type
   :class:`AIComponentType.AGENT` for every class that has a
   ``react_loop`` match and **no** anti-pattern exclusion (e.g. Temporal
   workflow, Prefect flow). ``framework_matches`` are intentionally
   skipped here because they are already surfaced by the KB/config
   scanners.

The scanner does **not** make the final call. It surfaces
structurally-plausible candidates so the Phase 5 LLM sees them in its
prompt (via :mod:`aibom.agentic.evidence_injection`) and the Phase 6
verification gate checks each evidence citation. The classification
decision still belongs to the LLM.

Guardrails
----------

* Test/example directories are excluded to prevent obvious false
  positives (e.g. a toy fake loop in a unit test). The exclusion list
  matches filename patterns (``test_*.py``, ``*_test.py``,
  ``conftest.py``) and directory names (``tests``, ``test``,
  ``__tests__``, ``testing``, ``examples``, ``example``).
* Files larger than :data:`_MAX_PY_FILE_SIZE_BYTES` are skipped. This
  matches the size guard used by the A2A detector and the remote-agent
  resolver.
* A per-file budget on class count is enforced so pathological files
  with hundreds of classes cannot blow up the prompt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pathspec import PathSpec

from ..agent_signatures import (
    AgentSignatureCatalog,
    resolve_catalog,
)
from ..cst_parser import parse_source_code
from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from ..structures import CodeAnalysisResult
from .agent_evidence_builder import (
    AgentEvidenceDossier,
    build_dossier_for_class,
)
from .base import BaseScanner
from .file_cache import is_python_source, read_python_source

_LOGGER = logging.getLogger(__name__)

_MAX_PY_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MiB, matches a2a_detector
_MAX_CLASSES_PER_FILE = 200

_TEST_DIR_NAMES: frozenset[str] = frozenset({
    "tests",
    "test",
    "__tests__",
    "testing",
    "examples",
    "example",
})

_TEST_FILENAME_PREFIXES: tuple[str, ...] = ("test_",)
_TEST_FILENAME_SUFFIXES: tuple[str, ...] = ("_test.py",)
_TEST_FILENAMES: frozenset[str] = frozenset({"conftest.py"})


# High-precision substrings that identify an LLM client SDK or agent
# runtime SDK import. A file must contain at least one of these imports
# for its classes to be eligible for a structural-ReAct match. Without
# this gate, pure-control-flow heuristics routinely flag HTTP pollers,
# ETL activities, and retry wrappers as "agents" because their loops
# structurally resemble ReAct.
#
# The list is deliberately narrow: it covers the provider SDKs and
# agent frameworks our target population actually uses, and nothing
# more. Users who need to extend it should add a framework or remote
# agent SDK signature in ``.aibom.yaml`` — those ``import_substrings``
# are ORed with this list at emission time.
_BUILTIN_LLM_SDK_IMPORT_HINTS: tuple[str, ...] = (
    "openai",
    "anthropic",
    "langchain",
    "langgraph",
    "llama_index",
    "llamaindex",
    "litellm",
    "cohere",
    "mistralai",
    "ollama",
    "google.generativeai",
    "google.genai",
    "vertexai",
    "autogen",
    "autogen_core",
    "crewai",
    "bedrock_agent",
    "bedrock_agentcore",
    "semantic_kernel",
    "smolagents",
    "pydantic_ai",
    "pydantic_graph",
    "strands",
    "strands_tools",
)


def _load_exclude_spec(patterns: list[str]) -> Optional[PathSpec]:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _is_excluded(
    file_path: Path, root: Path, spec: Optional[PathSpec]
) -> bool:
    if not spec:
        return False
    try:
        rel = file_path.relative_to(root).as_posix()
    except ValueError:
        rel = file_path.as_posix()
    return spec.match_file(rel)


def _is_test_file(path: Path) -> bool:
    """Return True for files that structurally resemble unit/example code."""
    name = path.name
    if name in _TEST_FILENAMES:
        return True
    if any(name.startswith(p) for p in _TEST_FILENAME_PREFIXES):
        return True
    if any(name.endswith(s) for s in _TEST_FILENAME_SUFFIXES):
        return True
    for part in path.parts[:-1]:
        if part in _TEST_DIR_NAMES:
            return True
    return False


def _iter_python_files_from_context(context: ScanContext) -> list[Path]:
    """Return the Python files to scan, honoring ``.aibomignore``/excludes."""
    idx = context.file_index()
    if idx:
        entries = idx.get(".py", [])
        return [entry.path for entry in entries]
    spec = _load_exclude_spec(context.exclude_patterns)
    files: list[Path] = []
    for raw in context.paths:
        root = Path(raw).expanduser()
        if root.is_file():
            if is_python_source(root) and not _is_excluded(
                root, root.parent, spec
            ):
                files.append(root)
            continue
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            if p.is_file() and is_python_source(p) and not _is_excluded(
                p, root, spec
            ):
                files.append(p)
    return files


def _safe_parse(path: Path) -> CodeAnalysisResult | None:
    """Parse *path* with the shared CST pipeline and a size guard."""
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _MAX_PY_FILE_SIZE_BYTES:
        _LOGGER.debug(
            "structural_agent_scanner: skipping oversized file %s (%d bytes)",
            path, size,
        )
        return None
    try:
        source = read_python_source(str(path))
    except (OSError, ValueError):
        return None
    try:
        return parse_source_code(str(path), source)
    except Exception as exc:  # pragma: no cover — defensive
        _LOGGER.debug(
            "structural_agent_scanner: parse failure for %s: %s", path, exc
        )
        return None


def _collect_sdk_import_hints(
    catalog: AgentSignatureCatalog,
) -> tuple[str, ...]:
    """Return the full SDK import-substring set for the emission gate.

    Merges three sources, de-duplicated and lower-cased:

    * :data:`_BUILTIN_LLM_SDK_IMPORT_HINTS` — built-in LLM provider SDKs
    * ``catalog.remote_agent_sdks[*].import_substrings`` — extensible
    * ``catalog.frameworks[*].import_substrings`` — extensible

    Users can extend this gate without code changes by adding a
    framework or remote agent SDK in their ``.aibom.yaml``.
    """
    merged: set[str] = set()
    for hint in _BUILTIN_LLM_SDK_IMPORT_HINTS:
        merged.add(hint.lower())
    for sdk in catalog.remote_agent_sdks:
        for substr in sdk.import_substrings:
            if substr:
                merged.add(substr.lower())
    for framework in catalog.frameworks:
        for substr in framework.import_substrings:
            if substr:
                merged.add(substr.lower())
    return tuple(sorted(merged))


def _file_has_llm_sdk_import(
    result: CodeAnalysisResult, hints: tuple[str, ...]
) -> bool:
    """Return True iff any import line in *result* mentions an SDK *hint*.

    Imports are compared case-insensitively against the full import
    statement text captured by the CST parser (e.g. ``from openai import
    OpenAI``, ``import langchain.agents as la``). A single substring
    match is enough — we want permissive file-level eligibility, with
    the per-class react-loop check doing the narrow filtering.
    """
    if not hints:
        return True
    for _line, stmt in result.imports:
        lowered = stmt.lower()
        if any(hint in lowered for hint in hints):
            return True
    return False


def _should_emit(dossier: AgentEvidenceDossier) -> bool:
    """Decide whether *dossier* is a structural-agent candidate.

    Rules
    -----
    1. At least one ``react_loop`` match (iterative control flow with
       multiple distinct callees, a branch, and at least one LLM-like
       call in the loop body).
    2. No anti-pattern match (the class is not a workflow orchestrator,
       data pipeline runner, etc.).
    3. Anything already covered by a framework signature is skipped
       because the KB / config scanners will emit a candidate for it;
       our job is to cover the framework-less gap.

    The module-level SDK-import gate is enforced in
    :func:`iter_structural_agent_candidates` so that non-AI files short
    circuit before we pay to build a dossier.
    """
    if not dossier.react_loop_matches:
        return False
    if dossier.is_excluded_by_anti_pattern:
        return False
    if dossier.framework_matches:
        return False
    return True


_MAX_EVIDENCE_SNIPPET_CHARS = 4_000


def _read_loop_snippet(
    file_path: str, start_line: int, end_line: int
) -> str:
    """Return the source text of the cited loop, bounded for prompt safety.

    The snippet must be readable back from *file_path* by the Phase 6
    verification gate (:func:`aibom.agentic.middleware._verify_agent_evidence`),
    which re-reads the file and searches for the snippet inside the
    declared line range after whitespace normalization. We therefore
    write the snippet verbatim from the same line range we cite.

    Returns an empty string on any I/O or encoding failure; the caller
    uses that to decide not to attach ``agent_evidence`` for this
    dossier rather than attaching an unverifiable citation.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError):
        _LOGGER.debug(
            "structural_agent_scanner: could not read %s for evidence",
            file_path,
        )
        return ""
    total = len(lines)
    if start_line < 1 or end_line < start_line or end_line > total:
        _LOGGER.debug(
            "structural_agent_scanner: invalid snippet range %d-%d for %s "
            "(file has %d lines)",
            start_line, end_line, file_path, total,
        )
        return ""
    snippet = "\n".join(lines[start_line - 1:end_line])
    if len(snippet) > _MAX_EVIDENCE_SNIPPET_CHARS:
        # A snippet longer than the bound will still be verifiable (we
        # truncate from the end of the cited range so the retained
        # prefix still appears inside the range).
        snippet = snippet[:_MAX_EVIDENCE_SNIPPET_CHARS]
    return snippet


def _build_machine_agent_evidence(
    dossier: AgentEvidenceDossier, best_loop
) -> dict[str, object] | None:
    """Build a verifiable ``agent_evidence`` payload for a structural hit.

    Returns a dict that conforms to
    :class:`aibom.agentic.agent.AgentEvidence`. The payload is designed
    to pass the Phase 6 verification gate unchanged: ``definition_file``
    is the on-disk path, ``definition_start_line`` / ``definition_end_line``
    bound the class, and ``evidence_snippet`` is the exact source of
    the cited loop.

    Returns ``None`` if the snippet cannot be read — callers must then
    omit ``agent_evidence`` so the symmetric evidence gate can drop the
    component rather than promote an unverifiable citation.
    """
    snippet = _read_loop_snippet(
        dossier.file_path, best_loop.start_line, best_loop.end_line
    )
    if not snippet.strip():
        return None
    return {
        "pattern": "react_loop",
        "definition_file": dossier.file_path,
        "definition_start_line": dossier.class_start_line,
        "definition_end_line": dossier.class_end_line,
        "evidence_snippet": snippet,
        "justification": (
            "Structural scanner detected an iterative control-flow loop "
            f"in class '{dossier.class_name}' between lines "
            f"{best_loop.start_line}-{best_loop.end_line}, with "
            "multiple distinct callees, a conditional branch, and at "
            "least one LLM-like invocation hint. "
            f"Rationale: {best_loop.rationale}"
        ),
    }


def _dossier_to_component(dossier: AgentEvidenceDossier) -> AIComponent:
    """Create an :class:`AIComponent` for the LLM to judge.

    The metadata encodes the loop location so the Phase 5 prompt can
    show the exact lines to the LLM and the Phase 6 verification gate
    can re-verify the citation.

    We also attach a machine-generated ``agent_evidence`` payload to
    ``metadata["agent_evidence"]`` so downstream consumers — in
    particular the symmetric evidence gate in the scan pipeline — can
    verify this component on the same terms as an LLM-driven agent
    classification. The payload uses the exact loop line range and the
    exact source snippet, so it round-trips through
    :func:`aibom.agentic.middleware._verify_agent_evidence` without
    whitespace fiddling.
    """
    best_loop = max(
        dossier.react_loop_matches,
        key=lambda m: m.end_line - m.start_line,
    )
    metadata: dict[str, object] = {
        "discovery": "structural_react_loop",
        "structural_signature_id": best_loop.signature_id,
        "react_loop_start_line": best_loop.start_line,
        "react_loop_end_line": best_loop.end_line,
        "react_loop_rationale": best_loop.rationale,
        "class_start_line": dossier.class_start_line,
        "class_end_line": dossier.class_end_line,
    }
    if dossier.qualified_name:
        metadata["qualified_name"] = dossier.qualified_name
    if dossier.protocol_matches:
        metadata["protocol_match_count"] = len(dossier.protocol_matches)
    agent_evidence = _build_machine_agent_evidence(dossier, best_loop)
    if agent_evidence is not None:
        metadata["agent_evidence"] = agent_evidence
    return AIComponent(
        name=dossier.class_name,
        component_type=AIComponentType.AGENT,
        file_path=dossier.file_path,
        line_number=dossier.class_start_line,
        framework="unknown",
        detection_source=DetectionSource.CODE_ANALYSIS,
        heuristic_confidence=0.5,
        needs_agentic=True,
        agentic_hint=(
            "Structural ReAct-style loop detected "
            f"(lines {best_loop.start_line}-{best_loop.end_line}). "
            "LLM must verify iterative tool dispatch and LLM-driven control flow."
        ),
        metadata=metadata,
    )


def iter_structural_agent_candidates(
    context: ScanContext,
    *,
    catalog: AgentSignatureCatalog | None = None,
) -> list[AIComponent]:
    """Yield structural agent-loop candidate components.

    Exposed for tests and for callers that want the same list without
    paying the cost of running a full scanner instance.
    """
    resolved_catalog = catalog or resolve_catalog()
    sdk_hints = _collect_sdk_import_hints(resolved_catalog)
    components: list[AIComponent] = []
    seen: set[tuple[str, int, str]] = set()
    for path in _iter_python_files_from_context(context):
        if _is_test_file(path):
            continue
        result = _safe_parse(path)
        if result is None:
            continue
        if not _file_has_llm_sdk_import(result, sdk_hints):
            continue
        class_obs_list = result.class_bodies[:_MAX_CLASSES_PER_FILE]
        for class_obs in class_obs_list:
            dossier = build_dossier_for_class(
                resolved_catalog, result, class_obs
            )
            if not _should_emit(dossier):
                continue
            comp = _dossier_to_component(dossier)
            key = (comp.file_path, comp.line_number, comp.name)
            if key in seen:
                continue
            seen.add(key)
            components.append(comp)
    return components


class StructuralAgentScanner(BaseScanner):
    """Registered scanner wrapper for :func:`iter_structural_agent_candidates`."""

    name = "structural_agent_scanner"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components = iter_structural_agent_candidates(context)
        _LOGGER.debug(
            "StructuralAgentScanner emitted %d candidate(s)", len(components)
        )
        return components, []


__all__ = [
    "StructuralAgentScanner",
    "iter_structural_agent_candidates",
]
