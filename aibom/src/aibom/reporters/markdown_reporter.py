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

from ..models import AIComponent, ComponentRelationship, RiskFlag, ScanResult
from .base import BaseReporter


def _md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _components_by_type(
    components: list[AIComponent],
) -> dict[str, list[AIComponent]]:
    by: dict[str, list[AIComponent]] = defaultdict(list)
    for c in components:
        by[c.component_type.value].append(c)
    return dict(sorted(by.items()))


class MarkdownReporter(BaseReporter):
    name = "markdown"
    file_extension = ".md"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        s = result.summary
        lines: list[str] = [
            "# AI BOM Report",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total components | {s['total_components']} |",
            f"| Risk score | {s['risk_score']} |",
            f"| Severity | {_md_cell(s['risk_severity'])} |",
            "",
        ]

        grouped = _components_by_type(result.all_components)
        for ctype, comps in grouped.items():
            lines.append(f"## {_md_cell(ctype)}")
            lines.extend(
                [
                    "",
                    "| Name | File | Line | Framework | Detection source |",
                    "| --- | --- | --- | --- | --- |",
                ]
            )
            for c in comps:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md_cell(c.name),
                            _md_cell(c.file_path),
                            str(c.line_number),
                            _md_cell(c.framework),
                            _md_cell(c.detection_source.value),
                        ]
                    )
                    + " |"
                )
            lines.append("")

        lines.append("## Relationships")
        rels: list[ComponentRelationship] = result.all_relationships
        if rels:
            lines.extend(
                [
                    "",
                    "| Source | Target | Type | Label |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for r in rels:
                src = r.source_name or r.source_instance_id
                tgt = r.target_name or r.target_instance_id
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md_cell(src),
                            _md_cell(tgt),
                            _md_cell(r.relationship_type.value),
                            _md_cell(r.label),
                        ]
                    )
                    + " |"
                )
        else:
            lines.extend(["", "_No relationships._", ""])
        lines.append("")

        lines.append("## Risk Assessment")
        flags: list[RiskFlag] = result.risk.flags
        if flags:
            lines.extend(
                [
                    "",
                    "| Flag | Severity | Weight | Description | File | Line |",
                    "| --- | --- | --- | --- | --- | --- |",
                ]
            )
            for f in flags:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md_cell(f.flag),
                            _md_cell(f.severity.value),
                            str(f.weight),
                            _md_cell(f.description),
                            _md_cell(f.file_path),
                            str(f.line_number),
                        ]
                    )
                    + " |"
                )
        else:
            lines.extend(["", "_No risk flags._", ""])

        output.write("\n".join(lines) + "\n")
