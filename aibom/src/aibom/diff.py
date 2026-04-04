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

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from .models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    RiskScore,
    ScanResult,
    SourceResult,
)

def _component_key(c: AIComponent) -> tuple[str, str]:
    ct = c.component_type
    v = ct.value if isinstance(ct, AIComponentType) else str(ct)
    return (c.name, v)


def _count_by_type(components: list[AIComponent]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for c in components:
        ct = c.component_type
        key = ct.value if isinstance(ct, AIComponentType) else str(ct)
        out[key] += 1
    return dict(sorted(out.items()))


def _collect_changes(before: AIComponent, after: AIComponent) -> list[str]:
    changes: list[str] = []
    if before.confidence != after.confidence:
        changes.append("confidence")
    if before.model_name != after.model_name:
        changes.append("model_name")
    if before.needs_agentic != after.needs_agentic:
        changes.append("needs_agentic")
    if before.framework != after.framework:
        changes.append("framework")
    if before.file_path != after.file_path:
        changes.append("file_path")
    return changes


class ComponentChange(BaseModel):
    before: AIComponent
    after: AIComponent
    changes: list[str]


class DiffResult(BaseModel):
    added: list[AIComponent] = Field(default_factory=list)
    removed: list[AIComponent] = Field(default_factory=list)
    changed: list[ComponentChange] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


def diff_scan_results(old: ScanResult, new: ScanResult) -> DiffResult:
    old_map: dict[tuple[str, str], AIComponent] = {}
    for c in old.all_components:
        old_map[_component_key(c)] = c
    new_map: dict[tuple[str, str], AIComponent] = {}
    for c in new.all_components:
        new_map[_component_key(c)] = c

    old_keys = set(old_map)
    new_keys = set(new_map)

    added: list[AIComponent] = [new_map[k] for k in sorted(new_keys - old_keys)]
    removed: list[AIComponent] = [old_map[k] for k in sorted(old_keys - new_keys)]

    changed: list[ComponentChange] = []
    unchanged_components: list[AIComponent] = []
    unchanged_count = 0
    for k in sorted(old_keys & new_keys):
        b, a = old_map[k], new_map[k]
        ch = _collect_changes(b, a)
        if ch:
            changed.append(ComponentChange(before=b, after=a, changes=ch))
        else:
            unchanged_count += 1
            unchanged_components.append(b)

    summary: dict[str, Any] = {
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "unchanged_count": unchanged_count,
        "by_type": {
            "added": _count_by_type(added),
            "removed": _count_by_type(removed),
            "changed": _count_by_type([c.before for c in changed]),
            "unchanged": _count_by_type(unchanged_components),
        },
    }

    return DiffResult(added=added, removed=removed, changed=changed, summary=summary)


def load_scan_result_json(path: Path | str) -> ScanResult:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if "aibom_analysis" in data:
        data = data["aibom_analysis"]
    sources_raw = data.get("sources", [])
    sources: list[SourceResult] = []
    for src in sources_raw:
        path_s = src.get("path", "")
        raw_comps = src.get("components", [])
        flat: list[AIComponent] = []
        if isinstance(raw_comps, dict):
            for items in raw_comps.values():
                for item in items:
                    flat.append(AIComponent.model_validate(item))
        else:
            for item in raw_comps:
                flat.append(AIComponent.model_validate(item))
        rels = [ComponentRelationship.model_validate(r) for r in src.get("relationships", [])]
        sources.append(SourceResult(path=path_s, components=flat, relationships=rels))
    risk_data = data.get("risk", {})
    risk = RiskScore.model_validate(risk_data) if risk_data else RiskScore()
    return ScanResult(
        metadata=data.get("metadata", {}),
        sources=sources,
        risk=risk,
        errors=list(data.get("errors", [])),
    )


def render_diff_table(diff: DiffResult, console: Console) -> None:
    s = diff.summary
    sum_table = Table(title="Diff summary", show_header=True, header_style="bold")
    sum_table.add_column("Metric")
    sum_table.add_column("Count", justify="right")
    sum_table.add_row("Added", str(s.get("added_count", 0)))
    sum_table.add_row("Removed", str(s.get("removed_count", 0)))
    sum_table.add_row("Changed", str(s.get("changed_count", 0)))
    sum_table.add_row("Unchanged", str(s.get("unchanged_count", 0)))
    console.print(sum_table)

    by_type = s.get("by_type", {})
    if by_type:
        bt = Table(title="By component type", show_header=True, header_style="bold")
        bt.add_column("Type")
        for col in ("added", "removed", "changed", "unchanged"):
            bt.add_column(col.title(), justify="right")
        type_keys = sorted(
            set()
            .union(*(by_type.get(k, {}) for k in ("added", "removed", "changed", "unchanged")))
        )
        for tk in type_keys:
            bt.add_row(
                tk,
                str(by_type.get("added", {}).get(tk, 0)),
                str(by_type.get("removed", {}).get(tk, 0)),
                str(by_type.get("changed", {}).get(tk, 0)),
                str(by_type.get("unchanged", {}).get(tk, 0)),
            )
        console.print(bt)

    if diff.added:
        t = Table(title="Added", show_lines=False)
        t.add_column("Name")
        t.add_column("Type")
        t.add_column("File")
        for c in diff.added:
            ct = c.component_type.value if isinstance(c.component_type, AIComponentType) else str(c.component_type)
            t.add_row(c.name, ct, c.file_path)
        console.print(t)

    if diff.removed:
        t = Table(title="Removed")
        t.add_column("Name")
        t.add_column("Type")
        t.add_column("File")
        for c in diff.removed:
            ct = c.component_type.value if isinstance(c.component_type, AIComponentType) else str(c.component_type)
            t.add_row(c.name, ct, c.file_path)
        console.print(t)

    if diff.changed:
        t = Table(title="Changed", show_lines=True)
        t.add_column("Name")
        t.add_column("Type")
        t.add_column("Fields")
        t.add_column("Before → After")
        for entry in diff.changed:
            b = entry.before
            a = entry.after
            ch = entry.changes
            ct = b.component_type.value if isinstance(b.component_type, AIComponentType) else str(b.component_type)
            detail_parts = []
            for field in ch:
                bv = getattr(b, field)
                av = getattr(a, field)
                detail_parts.append(f"{field}: {bv!r} → {av!r}")
            detail = "\n".join(detail_parts)
            t.add_row(b.name, ct, ", ".join(ch), detail)
        console.print(t)


def render_diff_markdown(diff: DiffResult) -> str:
    lines: list[str] = []
    s = diff.summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Added:** {s.get('added_count', 0)}")
    lines.append(f"- **Removed:** {s.get('removed_count', 0)}")
    lines.append(f"- **Changed:** {s.get('changed_count', 0)}")
    lines.append(f"- **Unchanged:** {s.get('unchanged_count', 0)}")
    lines.append("")

    by_type = s.get("by_type", {})
    if by_type:
        lines.append("### By type")
        lines.append("")
        lines.append("| Type | Added | Removed | Changed | Unchanged |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        type_keys = sorted(
            set()
            .union(*(by_type.get(k, {}) for k in ("added", "removed", "changed", "unchanged")))
        )
        for tk in type_keys:
            lines.append(
                f"| {tk} | {by_type.get('added', {}).get(tk, 0)} | "
                f"{by_type.get('removed', {}).get(tk, 0)} | "
                f"{by_type.get('changed', {}).get(tk, 0)} | "
                f"{by_type.get('unchanged', {}).get(tk, 0)} |"
            )
        lines.append("")

    if diff.added:
        lines.append("## Added")
        lines.append("")
        for c in diff.added:
            ct = c.component_type.value if isinstance(c.component_type, AIComponentType) else str(c.component_type)
            lines.append(f"- `{c.name}` ({ct}) — `{c.file_path}`")
        lines.append("")

    if diff.removed:
        lines.append("## Removed")
        lines.append("")
        for c in diff.removed:
            ct = c.component_type.value if isinstance(c.component_type, AIComponentType) else str(c.component_type)
            lines.append(f"- `{c.name}` ({ct}) — `{c.file_path}`")
        lines.append("")

    if diff.changed:
        lines.append("## Changed")
        lines.append("")
        for entry in diff.changed:
            b = entry.before
            a = entry.after
            ct = b.component_type.value if isinstance(b.component_type, AIComponentType) else str(b.component_type)
            lines.append(f"### `{b.name}` ({ct})")
            for field in entry.changes:
                bv = getattr(b, field)
                av = getattr(a, field)
                lines.append(f"- **{field}:** `{bv!r}` → `{av!r}`")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_diff_json(diff: DiffResult) -> str:
    payload = diff.model_dump(mode="json")
    return json.dumps(payload, indent=2) + "\n"


def render_diff(diff: DiffResult, fmt: str, console: Console) -> None:
    fmt_l = fmt.lower().strip()
    if fmt_l == "json":
        console.print(render_diff_json(diff), end="")
    elif fmt_l == "markdown":
        console.print(render_diff_markdown(diff), end="")
    elif fmt_l == "table":
        render_diff_table(diff, console)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")
