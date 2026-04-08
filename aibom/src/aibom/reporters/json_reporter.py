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
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import IO, Any

from ..models import AIComponent, ScanResult
from .base import BaseReporter


def _friendly_source_name(path: str) -> str:
    """Derive a short, meaningful source label from a local path or URL.

    For paths containing ``github.com``, extracts ``org/repo``.
    Otherwise falls back to the last path component.
    """
    parts = PurePosixPath(path).parts
    if "github.com" in parts:
        idx = parts.index("github.com")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}/{parts[idx + 2]}"
    return PurePosixPath(path).name or path


def _components_by_type(components: Iterable[AIComponent]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for comp in components:
        key = comp.component_type.value
        grouped.setdefault(key, []).append(comp.model_dump(mode="json"))
    return grouped


def _aibom_payload(result: ScanResult) -> dict[str, Any]:
    base = result.model_dump(mode="json")
    sources_out: dict[str, dict[str, Any]] = {}
    for src in result.sources:
        comps = _components_by_type(src.components)
        total_components = sum(len(v) for v in comps.values())
        source_key = _friendly_source_name(src.path)
        sources_out[source_key] = {
            "components": comps,
            "relationships": [r.model_dump(mode="json") for r in src.relationships],
            "summary": {
                "status": "completed",
                "source_kind": "local-path",
                "assets_discovered": total_components,
                "last_generated_at": base["metadata"].get("completed_at"),
            },
        }
    return {
        "aibom_analysis": {
            "metadata": base["metadata"],
            "sources": sources_out,
            "summary": result.summary,
            "risk": base["risk"],
            "errors": base["errors"],
        }
    }


class JsonReporter(BaseReporter):
    name = "json"
    file_extension = ".json"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        payload = _aibom_payload(result)
        json.dump(payload, output, indent=2)
        output.write("\n")

    def validate(self, result: ScanResult) -> list[str]:
        return []
