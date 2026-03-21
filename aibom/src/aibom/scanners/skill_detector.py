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
from pathlib import Path
from typing import Any, Optional

from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)

_TRIGGER_STARTS = (
    "use when",
    "activate when",
    "trigger when",
    "invoke when",
    "should be used when",
)


def _load_exclude_spec(patterns: list[str]) -> Optional[PathSpec]:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _is_excluded(file_path: Path, root: Path, spec: Optional[PathSpec]) -> bool:
    if not spec:
        return False
    try:
        rel = file_path.relative_to(root).as_posix()
    except ValueError:
        rel = file_path.as_posix()
    return spec.match_file(rel)


def _list_files_in_skill_dir(
    skill_dir: Path, scan_root: Path, spec: Optional[PathSpec]
) -> list[str]:
    out: list[str] = []
    if not skill_dir.is_dir():
        return out
    for p in skill_dir.rglob("*"):
        if not p.is_file():
            continue
        if _is_excluded(p, scan_root, spec):
            continue
        try:
            out.append(p.relative_to(skill_dir).as_posix())
        except ValueError:
            out.append(p.name)
    return sorted(out)


def _parse_skill_markdown(text: str) -> tuple[Optional[str], Optional[str], list[str]]:
    lines = text.splitlines()
    title: Optional[str] = None
    title_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip()
            title_idx = i
            break
    triggers: list[str] = []
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if any(low.startswith(p) for p in _TRIGGER_STARTS):
            triggers.append(stripped)
    desc: Optional[str] = None
    if title_idx >= 0:
        i = title_idx + 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        parts: list[str] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                break
            if line.startswith("#"):
                break
            parts.append(line.strip())
            i += 1
        if parts:
            raw = " ".join(parts)
            desc = raw if len(raw) <= 200 else raw[:197] + "..."
    return title, desc, triggers


def _read_skill_doc(path: Path) -> tuple[Optional[str], Optional[str], list[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None, []
    return _parse_skill_markdown(text)


def _run_optional_skill_scanner(paths: list[str]) -> None:
    try:
        from skill_scanner import SkillScanner
    except ImportError:
        return
    try:
        scanner = SkillScanner()
        batch = getattr(scanner, "analyze_paths", None)
        if callable(batch):
            batch(paths)
            return
        one = getattr(scanner, "analyze_path", None)
        if callable(one):
            for p in paths:
                one(p)
            return
        fallback = getattr(scanner, "analyze", None)
        if callable(fallback):
            for p in paths:
                fallback(p)
    except Exception:
        _LOGGER.debug("skill_scanner.SkillScanner failed", exc_info=True)


def _component_for_skill_md(
    skill_md: Path,
    skill_format: str,
    scan_root: Path,
    spec: Optional[PathSpec],
    seen: set[str],
) -> Optional[AIComponent]:
    key = str(skill_md.resolve())
    if key in seen:
        return None
    seen.add(key)
    skill_dir = skill_md.parent
    title, desc, triggers = _read_skill_doc(skill_md)
    name = title or skill_dir.name
    files = _list_files_in_skill_dir(skill_dir, scan_root, spec)
    meta: dict[str, Any] = {
        "trigger_patterns": triggers,
        "file_count": len(files),
        "files": files,
    }
    return AIComponent(
        name=name,
        component_type=AIComponentType.SKILL,
        file_path=str(skill_md.resolve()),
        skill_format=skill_format,
        description=desc,
        detection_source=DetectionSource.CONFIG_FILE,
        metadata=meta,
    )


def _component_for_plugin_dir(
    plugin_dir: Path,
    skill_format: str,
    scan_root: Path,
    spec: Optional[PathSpec],
    seen: set[str],
) -> Optional[AIComponent]:
    key = str(plugin_dir.resolve())
    if key in seen:
        return None
    seen.add(key)
    skill_md = plugin_dir / "SKILL.md"
    title: Optional[str] = None
    desc: Optional[str] = None
    triggers: list[str] = []
    if skill_md.is_file() and not _is_excluded(skill_md, scan_root, spec):
        title, desc, triggers = _read_skill_doc(skill_md)
    name = title or plugin_dir.name
    files = _list_files_in_skill_dir(plugin_dir, scan_root, spec)
    fp = str(skill_md.resolve()) if skill_md.is_file() else str(plugin_dir.resolve())
    meta: dict[str, Any] = {
        "trigger_patterns": triggers,
        "file_count": len(files),
        "files": files,
    }
    return AIComponent(
        name=name,
        component_type=AIComponentType.SKILL,
        file_path=fp,
        skill_format=skill_format,
        description=desc,
        detection_source=DetectionSource.CONFIG_FILE,
        metadata=meta,
    )


def _component_for_single_file(
    path: Path,
    skill_format: str,
    scan_root: Path,
    spec: Optional[PathSpec],
    seen: set[str],
) -> Optional[AIComponent]:
    key = str(path.resolve())
    if key in seen:
        return None
    if _is_excluded(path, scan_root, spec):
        return None
    seen.add(key)
    title, desc, triggers = _read_skill_doc(path)
    name = title or path.stem
    meta: dict[str, Any] = {
        "trigger_patterns": triggers,
        "file_count": 1,
        "files": [path.name],
    }
    return AIComponent(
        name=name,
        component_type=AIComponentType.SKILL,
        file_path=str(path.resolve()),
        skill_format=skill_format,
        description=desc,
        detection_source=DetectionSource.CONFIG_FILE,
        metadata=meta,
    )


def _collect_from_root(
    root: Path,
    spec: Optional[PathSpec],
    seen: set[str],
    components: list[AIComponent],
    scan_roots: list[str],
) -> None:
    scan_root = root.resolve()
    if root.is_file():
        name = root.name.lower()
        if name == "agents.md" or name == ".agents.md":
            c = _component_for_single_file(
                root, "agents_md", scan_root, spec, seen
            )
            if c:
                components.append(c)
                scan_roots.append(str(root.resolve()))
        elif root.name == "copilot-instructions.md":
            c = _component_for_single_file(
                root, "copilot", scan_root, spec, seen
            )
            if c:
                components.append(c)
                scan_roots.append(str(root.resolve()))
        return

    if not root.is_dir():
        return

    for skill_md in root.glob("**/.cursor/skills/*/SKILL.md"):
        if _is_excluded(skill_md, scan_root, spec):
            continue
        c = _component_for_skill_md(
            skill_md, "cursor", scan_root, spec, seen
        )
        if c:
            components.append(c)
            scan_roots.append(str(skill_md.parent.resolve()))

    for skill_md in root.glob("**/.codex/skills/*/SKILL.md"):
        if _is_excluded(skill_md, scan_root, spec):
            continue
        c = _component_for_skill_md(
            skill_md, "codex", scan_root, spec, seen
        )
        if c:
            components.append(c)
            scan_roots.append(str(skill_md.parent.resolve()))

    for plugins in root.glob("**/.claude/plugins"):
        if not plugins.is_dir() or _is_excluded(plugins, scan_root, spec):
            continue
        for child in plugins.iterdir():
            if not child.is_dir() or _is_excluded(child, scan_root, spec):
                continue
            c = _component_for_plugin_dir(
                child, "claude", scan_root, spec, seen
            )
            if c:
                components.append(c)
                scan_roots.append(str(child.resolve()))

    for devin in root.glob("**/.devin"):
        if not devin.is_dir() or _is_excluded(devin, scan_root, spec):
            continue
        for child in devin.iterdir():
            if not child.is_dir() or _is_excluded(child, scan_root, spec):
                continue
            c = _component_for_plugin_dir(
                child, "devin", scan_root, spec, seen
            )
            if c:
                components.append(c)
                scan_roots.append(str(child.resolve()))

    for copilot in root.glob("**/.github/copilot-instructions.md"):
        c = _component_for_single_file(
            copilot, "copilot", scan_root, spec, seen
        )
        if c:
            components.append(c)
            scan_roots.append(str(copilot.resolve()))

    for agents in root.glob("**/AGENTS.md"):
        c = _component_for_single_file(
            agents, "agents_md", scan_root, spec, seen
        )
        if c:
            components.append(c)
            scan_roots.append(str(agents.resolve()))

    for agents in root.glob("**/.agents.md"):
        c = _component_for_single_file(
            agents, "agents_md", scan_root, spec, seen
        )
        if c:
            components.append(c)
            scan_roots.append(str(agents.resolve()))


class SkillDetector(BaseScanner):
    name = "skill_detector"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        spec = _load_exclude_spec(context.exclude_patterns)
        components: list[AIComponent] = []
        seen: set[str] = set()
        scan_roots: list[str] = []

        for raw in context.paths:
            root = Path(raw).expanduser()
            if not root.exists():
                continue
            _collect_from_root(root, spec, seen, components, scan_roots)

        if scan_roots:
            _run_optional_skill_scanner(sorted(set(scan_roots)))

        return components, []
