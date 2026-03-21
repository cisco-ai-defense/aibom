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

"""KB Enrichment Scanner -- combines LibCST parsing with the DuckDB knowledge
base to detect and classify AI framework usage in Python source code.

Unlike the legacy ``categorize_symbols`` path which emits every KB match, this
scanner filters results to only high-signal AI asset concepts: agents, models,
tools, vector stores, embeddings, prompts, memory, and retrievers.  When no KB
is installed the scanner gracefully no-ops, letting the other v2 detectors
carry the workload.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..catalog_db import CatalogDB
from ..cst_parser import parse_source_code
from ..db_loader import DatabaseLoadError, ensure_local_database
from ..models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    DetectionSource,
    ScanContext,
)
from ..structures import CodeAnalysisResult
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)

ALLOWED_CONCEPTS: frozenset[str] = frozenset(
    {"agent", "model", "tool", "datastore", "embedding", "prompt", "memory", "retriever"}
)

_CONCEPT_TO_TYPE: dict[str, AIComponentType] = {
    "agent": AIComponentType.AGENT,
    "model": AIComponentType.MODEL,
    "tool": AIComponentType.TOOL,
    "datastore": AIComponentType.VECTOR_STORE,
    "embedding": AIComponentType.EMBEDDING,
    "prompt": AIComponentType.PROMPT,
    "memory": AIComponentType.MEMORY,
    "retriever": AIComponentType.RETRIEVER,
}


def _resolve_kb_path(context: ScanContext) -> Optional[Path]:
    """Locate the KB DuckDB file.  Returns ``None`` when unavailable."""
    if context.kb_path:
        p = Path(context.kb_path)
        if p.is_file():
            return p

    try:
        return ensure_local_database()
    except (DatabaseLoadError, Exception):  # noqa: BLE001
        pass

    catalogs_dir = Path.home() / ".aibom" / "catalogs"
    if catalogs_dir.is_dir():
        dbs = sorted(catalogs_dir.glob("*.duckdb"), reverse=True)
        if dbs:
            return dbs[0]

    return None


def _extract_frameworks_from_imports(imports: list[str]) -> set[str]:
    """Derive top-level package names from import statements.

    e.g. ``"from langchain_openai import ChatOpenAI"`` → ``{"langchain_openai"}``
    """
    frameworks: set[str] = set()
    for stmt in imports:
        if stmt.startswith("from "):
            module = stmt.split()[1] if len(stmt.split()) > 1 else ""
            top = module.split(".")[0]
            if top:
                frameworks.add(top)
        elif stmt.startswith("import "):
            module = stmt.split()[1] if len(stmt.split()) > 1 else ""
            top = module.split(".")[0]
            if top:
                frameworks.add(top)
    return frameworks


class KBEnrichmentScanner(BaseScanner):
    """Detect AI framework usage by matching LibCST observations against the KB.

    Only emits for concepts in :data:`ALLOWED_CONCEPTS`, suppressing the noise
    that made the legacy categorizer output hard to use.
    """

    name = "kb_enrichment"

    def supports(self, context: ScanContext) -> bool:
        return _resolve_kb_path(context) is not None

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        kb_path = _resolve_kb_path(context)
        if not kb_path:
            return [], []

        _LOGGER.info("KB enrichment: using %s", kb_path)
        components: list[AIComponent] = []

        with CatalogDB(kb_path) as db:
            custom_cfg = context.config.get("custom_catalog")
            if custom_cfg is not None:
                try:
                    from ..custom_catalog import CustomCatalogConfig

                    if isinstance(custom_cfg, CustomCatalogConfig) and not custom_cfg.is_empty:
                        db.add_custom_entries(
                            [c.to_catalog_dict() for c in custom_cfg.components]
                        )
                        if custom_cfg.excludes:
                            db.add_excludes(custom_cfg.excludes)
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("KB enrichment: custom catalog load failed", exc_info=True)

            for py_file in _find_python_files(context):
                try:
                    source = py_file.read_text(encoding="utf-8")
                    result = parse_source_code(str(py_file), source)
                    components.extend(_process_file(result, db))
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("KB enrichment: failed to parse %s", py_file, exc_info=True)

        return components, []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_python_files(context: ScanContext) -> list[Path]:
    files: list[Path] = []
    for p in context.paths:
        path = Path(p)
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def _process_file(
    result: CodeAnalysisResult, db: CatalogDB
) -> list[AIComponent]:
    """Match one file's parsed observations against the KB."""

    variable_map: dict[str, str] = {}
    for assignment in result.assignments:
        if assignment.target_qualified_name and assignment.call.qualified_name:
            variable_map[assignment.target_qualified_name] = assignment.call.qualified_name

    observations: list[dict[str, Any]] = []
    symbols: set[str] = set()

    for assignment in result.assignments:
        name = _resolve_chain(assignment.call.qualified_name, variable_map)
        observations.append(
            _obs(name, result.file_path, assignment.line_number, "assignment",
                 args=assignment.call.arguments,
                 assigned_target=assignment.target_qualified_name)
        )
        symbols.add(name)

    for dec in result.decorators:
        name = dec.decorator_qualified_name
        if dec.instance_variable and dec.instance_variable in variable_map:
            base = variable_map[dec.instance_variable]
            attr = name.split(".", 1)[-1] if "." in name else name
            name = f"{base}.{attr}"
        name = _resolve_chain(name, variable_map)
        observations.append(
            _obs(name, result.file_path, dec.line_number, "decorator",
                 decorated=dec.decorated_function_name)
        )
        symbols.add(name)

    for ctx in result.context_managers:
        if not ctx.context_expr_qualified_name:
            continue
        name = _resolve_chain(ctx.context_expr_qualified_name, variable_map)
        observations.append(
            _obs(name, result.file_path, ctx.line_number, "context_manager")
        )
        symbols.add(name)

    if not symbols:
        return []

    query_suffixes = set(symbols)
    for s in symbols:
        if "." in s:
            query_suffixes.add(s.rsplit(".", 1)[-1])
            cls = _extract_class_segment(s)
            if cls:
                query_suffixes.add(cls)

    matched = db.find_components_by_suffixes(list(query_suffixes))

    kb_by_id: dict[str, dict[str, Any]] = {}
    for entry in matched:
        concept = (entry.get("concept") or "").lower()
        if concept not in ALLOWED_CONCEPTS:
            continue
        eid = entry["id"]
        if eid not in kb_by_id:
            kb_by_id[eid] = entry

    imported_frameworks = _extract_frameworks_from_imports(getattr(result, "imports", []) or [])

    components: list[AIComponent] = []
    seen: set[tuple[str, int]] = set()

    for obs_data in observations:
        key = (obs_data["file"], obs_data["line"])
        if key in seen:
            continue

        kb_entry = _match_observation(obs_data["name"], kb_by_id, imported_frameworks)
        if not kb_entry:
            continue

        concept = kb_entry["concept"].lower()
        comp_type = _CONCEPT_TO_TYPE.get(concept)
        if not comp_type:
            continue

        seen.add(key)

        display_name = obs_data["name"]
        if obs_data["type"] == "decorator" and obs_data.get("decorated"):
            display_name = obs_data["decorated"]

        components.append(
            AIComponent(
                name=display_name,
                component_type=comp_type,
                file_path=obs_data["file"],
                line_number=obs_data["line"],
                framework=kb_entry.get("framework", ""),
                detection_source=DetectionSource.KB_ENRICHMENT,
                kb_concept=concept,
                kb_label=kb_entry.get("label", ""),
                metadata={
                    "kb_id": kb_entry["id"],
                    "observation_type": obs_data["type"],
                },
            )
        )

    return components


def _obs(
    name: str,
    file_path: str,
    line: int,
    obs_type: str,
    *,
    args: Optional[dict[str, Any]] = None,
    assigned_target: Optional[str] = None,
    decorated: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "file": file_path,
        "line": line,
        "type": obs_type,
        "args": args or {},
        "assigned_target": assigned_target,
        "decorated": decorated,
    }


def _resolve_chain(name: str, variable_map: dict[str, str]) -> str:
    """Resolve ``var.method`` → ``FullyQualified.method`` via the variable map."""
    if "." in name:
        head, tail = name.split(".", 1)
        if head in variable_map:
            return f"{variable_map[head]}.{tail}"
    return name


def _match_observation(
    obs_name: str,
    kb_by_id: dict[str, dict[str, Any]],
    imported_frameworks: set[str],
) -> Optional[dict[str, Any]]:
    """Return the best KB entry for *obs_name*, or ``None``.

    Tier 1: exact match on KB id.
    Tier 2: suffix match on full qualified name.
    Tier 3: suffix match on the SHORT class/function name (last dot-segment),
            disambiguated by imported frameworks.  This handles the common case
            where a wrapper package (``langchain_openai``) re-exports from an
            implementation package (``langchain_community``).
    """
    # Tier 1 -- exact
    if obs_name in kb_by_id:
        return kb_by_id[obs_name]

    obs_module = obs_name.split(".")[0] if "." in obs_name else ""

    # Tier 2 -- full qualified suffix, validated against framework family.
    # Without this check ``openai.OpenAI`` would match
    # ``langchain_community.llms.openai.OpenAI`` via the coincidental
    # ``.openai.OpenAI`` submodule path.
    candidates: list[dict[str, Any]] = []
    for kb_id, entry in kb_by_id.items():
        if kb_id.endswith("." + obs_name):
            if not obs_module or _frameworks_related(obs_module, entry.get("framework", "")):
                candidates.append(entry)

    if candidates:
        return _pick_best(candidates, imported_frameworks, obs_module)

    # Tier 3 -- short-name suffix using the nearest class-like (uppercase)
    # segment from the observation name.  This handles wrapper re-exports
    # (``langchain_openai.ChatOpenAI`` → KB has ``langchain_community.*.ChatOpenAI``)
    # and factory calls (``FAISS.from_texts`` → class segment ``FAISS``).
    class_name = _extract_class_segment(obs_name)
    if not class_name:
        return None

    for kb_id, entry in kb_by_id.items():
        if kb_id.endswith("." + class_name):
            candidates.append(entry)

    if not candidates:
        return None

    # Tier 3 is the loosest match; require the best candidate's framework to
    # share a package-family prefix with the observation's module so that
    # ``openai.OpenAI.chat.completions.create`` doesn't match a
    # ``langchain_community`` entry just because both have an ``OpenAI`` class.
    best = _pick_best(candidates, imported_frameworks, obs_module)
    if obs_module and not _frameworks_related(obs_module, best.get("framework", "")):
        return None
    return best


def _extract_class_segment(obs_name: str) -> Optional[str]:
    """Return the nearest class-like (uppercase-start) segment from a dotted name.

    Scans right-to-left so that ``pkg.FAISS.from_texts`` yields ``FAISS`` and
    ``openai.OpenAI.chat.completions.create`` yields ``OpenAI``.  Returns
    ``None`` when no uppercase segment is found or the result would be the
    entire *obs_name* (already covered by Tier 1).
    """
    if "." not in obs_name:
        return None
    parts = obs_name.split(".")
    for segment in reversed(parts):
        if segment and segment[0].isupper():
            if segment == obs_name:
                return None
            return segment
    return None


def _frameworks_related(obs_module: str, kb_framework: str) -> bool:
    """Return ``True`` when the observation module and KB framework belong to
    the same package family.

    ``langchain_openai`` and ``langchain_community`` share the ``langchain``
    prefix; ``crewai`` matches ``crewai``; ``openai`` does not match
    ``langchain_community``.
    """
    if not obs_module or not kb_framework:
        return True
    if obs_module == kb_framework:
        return True
    obs_top = obs_module.split("_")[0].split("-")[0]
    kb_top = kb_framework.split("_")[0].split("-")[0]
    return obs_top == kb_top


def _pick_best(
    candidates: list[dict[str, Any]],
    imported_frameworks: set[str],
    obs_module: str = "",
) -> dict[str, Any]:
    """From a set of KB candidates, prefer the one whose framework matches best.

    Priority: framework matches the observation's module prefix > framework is
    imported > first candidate.
    """
    if len(candidates) == 1:
        return candidates[0]

    # Prefer the candidate whose framework matches the observation module prefix.
    if obs_module:
        for c in candidates:
            fw = c.get("framework") or ""
            if fw == obs_module or fw.startswith(obs_module):
                return c

    for c in candidates:
        fw = c.get("framework") or ""
        fw_top = fw.split("_")[0].split("-")[0]
        if fw in imported_frameworks or fw_top in imported_frameworks:
            return c

    return candidates[0]
