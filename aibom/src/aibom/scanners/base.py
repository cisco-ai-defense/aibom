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

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext

_LOGGER = logging.getLogger(__name__)

scanner_registry: list[type["BaseScanner"]] = []


class BaseScanner(ABC):
    """Interface that all AIBOM scanners implement.

    Subclasses auto-register by setting a non-empty ``name`` class variable.
    """

    name: str = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name:
            scanner_registry.append(cls)
            _LOGGER.debug("Registered scanner: %s", cls.name)

    @abstractmethod
    def supports(self, context: ScanContext) -> bool:
        """Return True if this scanner can produce useful results for *context*."""
        ...

    @abstractmethod
    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        """Run the scanner and return detected components and relationships."""
        ...


def _load_aibomignore(paths: list[str]) -> Optional[PathSpec]:
    """Load .aibomignore from the first path that contains one."""
    for p in paths:
        ignore_file = Path(p) / ".aibomignore"
        if not ignore_file.is_file():
            parent = Path(p).parent
            ignore_file = parent / ".aibomignore"
        if ignore_file.is_file():
            lines = ignore_file.read_text().splitlines()
            return PathSpec.from_lines("gitwildmatch", lines)
    return None


def run_scanners(
    context: ScanContext,
    workers: int = 4,
    extra_scanners: Optional[list[type[BaseScanner]]] = None,
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    """Instantiate and run all registered scanners in parallel.

    Returns the merged component and relationship lists.
    """
    ignore_spec = _load_aibomignore(context.paths)
    if ignore_spec:
        merged = list(context.exclude_patterns)
        merged.extend(ignore_spec.patterns)
        context = context.model_copy(update={"exclude_patterns": merged})

    all_scanner_types = list(scanner_registry)
    if extra_scanners:
        all_scanner_types.extend(extra_scanners)

    applicable: list[BaseScanner] = []
    for cls in all_scanner_types:
        instance = cls()
        if instance.supports(context):
            applicable.append(instance)

    if not applicable:
        _LOGGER.warning("No scanners applicable for the given context")
        return [], []

    all_components: list[AIComponent] = []
    all_relationships: list[ComponentRelationship] = []

    if len(applicable) == 1:
        comps, rels = applicable[0].scan(context)
        return comps, rels

    with ThreadPoolExecutor(max_workers=min(workers, len(applicable))) as pool:
        futures = {pool.submit(s.scan, context): s for s in applicable}
        for future in as_completed(futures):
            scanner = futures[future]
            try:
                comps, rels = future.result()
                all_components.extend(comps)
                all_relationships.extend(rels)
            except Exception:
                _LOGGER.exception("Scanner %s failed", scanner.name)

    all_components = _merge_model_duplicates(all_components)
    return all_components, all_relationships


_MERGE_LINE_PROXIMITY = 5


def _merge_model_duplicates(
    components: list[AIComponent],
) -> list[AIComponent]:
    """Merge model_detector name entries with KB enrichment wrapper class
    entries when they refer to the same LLM constructor call.

    Heuristic: same file, both MODEL type, within ``_MERGE_LINE_PROXIMITY``
    lines, one has ``model_name`` set (model_detector) and the other has
    ``kb_id`` in metadata (KB enrichment).  The merged component keeps the
    model name as the display name and absorbs the wrapper class into metadata.
    """
    from ..models.enums import AIComponentType, DetectionSource

    name_entries: list[tuple[int, AIComponent]] = []
    kb_entries: list[tuple[int, AIComponent]] = []

    for idx, c in enumerate(components):
        if c.component_type != AIComponentType.MODEL:
            continue
        if c.detection_source == DetectionSource.KB_ENRICHMENT and c.metadata.get("kb_id"):
            kb_entries.append((idx, c))
        elif c.model_name and c.detection_source in (
            DetectionSource.CODE_ANALYSIS,
            DetectionSource.CONFIG_FILE,
        ):
            name_entries.append((idx, c))

    if not name_entries or not kb_entries:
        return components

    merged_indices: set[int] = set()

    for name_idx, name_comp in name_entries:
        for kb_idx, kb_comp in kb_entries:
            if kb_idx in merged_indices:
                continue
            if name_comp.file_path != kb_comp.file_path:
                continue
            if abs(name_comp.line_number - kb_comp.line_number) > _MERGE_LINE_PROXIMITY:
                continue

            merged_meta = dict(name_comp.metadata)
            merged_meta["wrapper_class"] = kb_comp.name
            merged_meta["kb_id"] = kb_comp.metadata.get("kb_id", "")
            if kb_comp.framework and name_comp.framework in ("unknown", ""):
                fw = kb_comp.framework
            else:
                fw = name_comp.framework

            components[name_idx] = name_comp.model_copy(
                update={"metadata": merged_meta, "framework": fw}
            )
            merged_indices.add(kb_idx)
            break

    if not merged_indices:
        return components

    return [c for i, c in enumerate(components) if i not in merged_indices]
