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

"""Build evidence dossiers for the LLM prompt.

The dossier index maps a component's source location to the
:class:`AgentEvidenceDossier` produced by the CST-based evidence
builder. :func:`_component_to_summary` in
:mod:`aibom.agentic.agent` consults this index to attach the
verbatim class body and the structured evidence to the component
summary that gets embedded in the LLM prompt.

Design notes
~~~~~~~~~~~~
* Parsing is done lazily and deduplicated per-file so cost is bounded
  to *O(files with agent-candidate components)*.
* Files larger than :data:`_MAX_PY_FILE_SIZE_BYTES` are skipped. This
  guards against pathological files that would bloat the prompt or
  stall the CST parser.
* The helper is completely side-effect free except for the internal
  per-call cache it maintains inside :func:`build_dossier_index`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from ..agent_signatures import AgentSignatureCatalog, resolve_catalog
from ..cst_parser import parse_source_code
from ..models import AIComponent, AIComponentType
from ..scanners.agent_evidence_builder import (
    AgentEvidenceDossier,
    build_dossiers,
)
from ..scanners.file_cache import is_python_source, read_python_source
from ..structures import CodeAnalysisResult

_LOGGER = logging.getLogger(__name__)

_MAX_PY_FILE_SIZE_BYTES = 2 * 1024 * 1024  # 2 MiB — mirrors a2a_detector / remote_agent_resolver

_AGENT_EVIDENCE_TARGET_TYPES: frozenset[AIComponentType] = frozenset({
    AIComponentType.AGENT,
    AIComponentType.AGENT_PROXY,
    AIComponentType.MCP_SERVER,
    AIComponentType.MCP_CLIENT,
})


DossierIndex = dict[tuple[str, int], AgentEvidenceDossier]


def _candidate_files(components: Iterable[AIComponent]) -> list[str]:
    """Return the set of Python files that host at least one candidate.

    Only components whose ``component_type`` is in
    :data:`_AGENT_EVIDENCE_TARGET_TYPES` contribute a file — the
    dossier is only useful for those types. ``file_path`` must point
    to an existing Python source file.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for c in components:
        if c.component_type not in _AGENT_EVIDENCE_TARGET_TYPES:
            continue
        if not c.file_path:
            continue
        if c.file_path in seen:
            continue
        p = Path(c.file_path)
        if not p.is_file() or not is_python_source(p):
            continue
        seen.add(c.file_path)
        ordered.append(c.file_path)
    return ordered


def _safe_parse(file_path: str) -> CodeAnalysisResult | None:
    """Parse *file_path* with :func:`parse_source_code`, with a size guard.

    Returns ``None`` if the file is too large, unreadable, or has a
    Python syntax error we cannot recover from.
    """
    try:
        size = Path(file_path).stat().st_size
    except OSError:
        return None
    if size > _MAX_PY_FILE_SIZE_BYTES:
        _LOGGER.debug(
            "evidence_injection: skipping oversized file %s (%d bytes)",
            file_path, size,
        )
        return None
    try:
        source = read_python_source(file_path)
    except (OSError, ValueError):
        return None
    try:
        return parse_source_code(file_path, source)
    except Exception as exc:  # pragma: no cover — defensive
        _LOGGER.debug(
            "evidence_injection: parse_source_code failed for %s: %s",
            file_path, exc,
        )
        return None


def build_dossier_index(
    components: Iterable[AIComponent],
    *,
    catalog: AgentSignatureCatalog | None = None,
) -> DossierIndex:
    """Build an index of :class:`AgentEvidenceDossier` objects by source location.

    The returned mapping is keyed by ``(file_path, class_start_line)``
    because class definitions are uniquely identified by their starting
    line within a file. Callers look the dossier up via
    :func:`lookup_dossier`, which accepts the *component*'s file/line
    and returns the dossier whose class span contains that line.

    Parameters
    ----------
    components:
        All deterministic components under consideration. Only
        components whose type is in :data:`_AGENT_EVIDENCE_TARGET_TYPES`
        contribute a file to parse.
    catalog:
        Optional merged agent-signature catalog. When ``None``, the
        built-in defaults (no user overrides) are used.
    """
    resolved_catalog = catalog or resolve_catalog()
    files = _candidate_files(components)
    if not files:
        return {}

    index: DossierIndex = {}
    for file_path in files:
        result = _safe_parse(file_path)
        if result is None:
            continue
        dossiers = build_dossiers(resolved_catalog, [result])
        for dossier in dossiers:
            key = (dossier.file_path, dossier.class_start_line)
            index[key] = dossier
    _LOGGER.debug(
        "evidence_injection: built %d dossier(s) across %d file(s)",
        len(index), len(files),
    )
    return index


def lookup_dossier(
    component: AIComponent,
    index: DossierIndex | None,
) -> AgentEvidenceDossier | None:
    """Return the dossier whose class span contains *component*'s line.

    Returns ``None`` when *index* is ``None`` / empty, when the
    component has no ``file_path`` / ``line_number``, or when no dossier
    covers the component's source position.
    """
    if not index:
        return None
    if not component.file_path or not component.line_number:
        return None
    file_matches = [
        dossier for (fp, _), dossier in index.items()
        if fp == component.file_path
    ]
    if not file_matches:
        return None
    line = component.line_number
    for dossier in file_matches:
        if dossier.class_start_line <= line <= dossier.class_end_line:
            return dossier
    return None


__all__ = [
    "DossierIndex",
    "build_dossier_index",
    "lookup_dossier",
]
