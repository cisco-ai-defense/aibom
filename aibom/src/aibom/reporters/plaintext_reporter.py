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

from collections import defaultdict
from typing import IO

from ..models import AIComponent, ComponentRelationship, ScanResult, SourceResult
from .base import BaseReporter


def _line(width: int, char: str = "=") -> str:
    return char * width


def _header(title: str, width: int = 72, char: str = "=") -> list[str]:
    return [title, _line(width, char), ""]


def _subheader(title: str, width: int = 72) -> list[str]:
    return [title, _line(width, "-"), ""]


def _format_component(comp: AIComponent) -> list[str]:
    lines = [
        f"  • {comp.name}",
        f"      type: {comp.component_type.value}",
        f"      location: {comp.file_path}:{comp.line_number}",
    ]
    if comp.framework:
        lines.append(f"      framework: {comp.framework}")
    if comp.model_name:
        lines.append(f"      model_name: {comp.model_name}")
    if comp.description:
        lines.append(f"      description: {comp.description}")
    lines.append("")
    return lines


def _format_relationship(rel: ComponentRelationship) -> str:
    parts = [
        rel.source_name or rel.source_instance_id,
        "→",
        rel.target_name or rel.target_instance_id,
        f"({rel.relationship_type.value})",
    ]
    return " ".join(parts)


class PlaintextReporter(BaseReporter):
    name = "plaintext"
    file_extension = ".txt"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        width = 72
        out: list[str] = []
        out.extend(_header("AIBOM Analysis Report", width))

        if result.metadata:
            out.extend(_subheader("Metadata", width))
            for key in sorted(result.metadata):
                out.append(f"  {key}: {result.metadata[key]}")
            out.append("")

        out.extend(_subheader("Sources", width))
        if not result.sources:
            out.append("  (no sources)")
            out.append("")
        else:
            for src in result.sources:
                out.extend(self._format_source(src, width))

        out.extend(_subheader("Relationships summary", width))
        all_rels = result.all_relationships
        out.append(f"  Total relationships: {len(all_rels)}")
        if all_rels:
            by_type: dict[str, int] = defaultdict(int)
            for rel in all_rels:
                by_type[rel.relationship_type.value] += 1
            for rtype in sorted(by_type):
                out.append(f"    {rtype}: {by_type[rtype]}")
        out.append("")

        if result.errors:
            out.extend(_subheader("Errors", width))
            for err in result.errors:
                out.append(f"  - {err}")
            out.append("")

        out.extend(_header("Risk assessment", width))
        out.append(f"  Score: {result.risk.score}")
        out.append(f"  Severity: {result.risk.severity.value}")
        out.append("")
        if result.risk.flags:
            out.extend(_subheader("Risk flags", width))
            for flag in result.risk.flags:
                out.append(f"  [{flag.severity.value}] {flag.flag} (weight {flag.weight})")
                if flag.description:
                    out.append(f"      {flag.description}")
                if flag.file_path:
                    loc = f"{flag.file_path}:{flag.line_number}" if flag.line_number else flag.file_path
                    out.append(f"      at {loc}")
                out.append("")
        else:
            out.append("  (no flags)")
            out.append("")

        output.write("\n".join(out).rstrip() + "\n")

    def _format_source(self, src: SourceResult, width: int) -> list[str]:
        lines: list[str] = []
        lines.extend(_subheader(f"Source: {src.path}", width))

        by_type: dict[str, list[AIComponent]] = defaultdict(list)
        for comp in src.components:
            by_type[comp.component_type.value].append(comp)

        if not src.components:
            lines.append("  (no components)")
            lines.append("")
        else:
            for ctype in sorted(by_type):
                lines.append(f"  {ctype}")
                lines.append(_line(width - 2, "-"))
                for comp in by_type[ctype]:
                    lines.extend(_format_component(comp))

        if src.relationships:
            lines.append("  Relationships")
            lines.append(_line(width - 2, "-"))
            for rel in src.relationships:
                lines.append(f"    {_format_relationship(rel)}")
            lines.append("")
        else:
            lines.append("  (no relationships in this source)")
            lines.append("")

        return lines
