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

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext

_LOGGER = logging.getLogger(__name__)

scanner_timings: dict[str, float] = {}

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


async def _async_timed_scan(
    scanner: BaseScanner, ctx: ScanContext,
) -> tuple[str, float, list[AIComponent], list[ComponentRelationship], Exception | None]:
    """Run a scanner in a thread (to release the event loop) and time it."""
    t0 = time.monotonic()
    try:
        comps, rels = await asyncio.to_thread(scanner.scan, ctx)
    except Exception as exc:  # pragma: no cover - exercised via async wrapper
        elapsed = time.monotonic() - t0
        return scanner.name, elapsed, [], [], exc
    elapsed = time.monotonic() - t0
    return scanner.name, elapsed, comps, rels, None


async def _run_scanners_async(
    context: ScanContext,
    extra_scanners: Optional[list[type[BaseScanner]]] = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    """Run all applicable scanners concurrently using asyncio."""
    ignore_spec = _load_aibomignore(context.paths)
    if ignore_spec:
        merged = list(context.exclude_patterns)
        merged.extend(ignore_spec.patterns)
        context = context.model_copy(update={"exclude_patterns": merged})

    all_scanner_types = list(scanner_registry)

    try:
        from ..plugins import discover_scanner_plugins

        for plugin_cls in discover_scanner_plugins():
            if plugin_cls not in all_scanner_types:
                all_scanner_types.append(plugin_cls)
    except Exception:
        _LOGGER.debug("Plugin discovery failed", exc_info=True)

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
    scanner_timings.clear()

    if progress_callback:
        progress_callback({
            "event": "scanners_discovered",
            "total": len(applicable),
        })

    tasks = [asyncio.create_task(_async_timed_scan(scanner, context)) for scanner in applicable]
    completed = 0

    for task in asyncio.as_completed(tasks):
        name, elapsed, comps, rels, error = await task
        if error is not None:
            _LOGGER.exception("Scanner %s failed", name, exc_info=error)
            continue
        scanner_timings[name] = elapsed
        all_components.extend(comps)
        all_relationships.extend(rels)
        completed += 1
        if progress_callback:
            progress_callback({
                "event": "scanner_completed",
                "scanner": name,
                "completed": completed,
                "total": len(applicable),
                "elapsed_s": elapsed,
                "component_count": len(comps),
                "relationship_count": len(rels),
            })

    all_components = _merge_model_duplicates(all_components)
    return all_components, all_relationships


def run_scanners(
    context: ScanContext,
    workers: int = 4,
    extra_scanners: Optional[list[type[BaseScanner]]] = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[AIComponent], list[ComponentRelationship]]:
    """Instantiate and run all registered scanners concurrently via asyncio.

    Uses ``asyncio.to_thread`` per scanner to release the GIL for I/O-heavy
    scanners while still allowing true concurrency for native async work.
    The *workers* parameter is retained for API compatibility but no longer
    controls a thread pool size — asyncio manages scheduling.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                _run_scanners_async(
                    context,
                    extra_scanners,
                    progress_callback=progress_callback,
                ),
            )
            return future.result()

    return asyncio.run(
        _run_scanners_async(
            context,
            extra_scanners,
            progress_callback=progress_callback,
        )
    )


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
