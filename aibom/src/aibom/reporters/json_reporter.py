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
from typing import IO, Any

from ..models import AIComponent, ScanResult
from .base import BaseReporter


def _components_by_type(components: Iterable[AIComponent]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for comp in components:
        key = comp.component_type.value
        grouped.setdefault(key, []).append(comp.model_dump(mode="json"))
    return grouped


def _aibom_payload(result: ScanResult) -> dict[str, Any]:
    base = result.model_dump(mode="json")
    sources_out: list[dict[str, Any]] = []
    for src in result.sources:
        sources_out.append(
            {
                "path": src.path,
                "components": _components_by_type(src.components),
                "relationships": [r.model_dump(mode="json") for r in src.relationships],
            }
        )
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
