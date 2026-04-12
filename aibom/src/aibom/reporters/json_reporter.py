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
import logging
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import IO, Any

from ..models import AIComponent, ScanResult
from .base import BaseReporter

_LOGGER = logging.getLogger(__name__)

_REMOTE_ORG_REPO_RE = re.compile(r"[/:]([^/]+)/([^/]+?)(?:\.git)?$")
REPORT_SCHEMA_VERSION = "1"


def _friendly_source_name(path: str) -> str:
    """Derive a short, meaningful source label from a local filesystem path.

    Tries ``git remote get-url origin`` to extract ``org/repo`` from the
    actual remote.  Falls back to the last path component.
    """
    resolved = Path(path).resolve()
    if resolved.is_dir():
        try:
            result = subprocess.run(
                ["git", "-C", str(resolved), "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                m = _REMOTE_ORG_REPO_RE.search(result.stdout.strip())
                if m:
                    return f"{m.group(1)}/{m.group(2)}"
        except Exception:
            _LOGGER.debug("git remote lookup failed for %s", path)
    return PurePosixPath(path).name or path


def _components_by_type(components: Iterable[AIComponent]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for comp in components:
        key = comp.component_type.value
        grouped.setdefault(key, []).append(comp.model_dump(mode="json"))
    return grouped


def _disambiguate_source_key(source_name: str, seen: dict[str, int]) -> str:
    count = seen.get(source_name, 0) + 1
    seen[source_name] = count
    if count == 1:
        return source_name
    return f"{source_name}#{count}"


def _aibom_payload(result: ScanResult) -> dict[str, Any]:
    base = result.model_dump(mode="json")
    raw_metadata = dict(base["metadata"])
    source_outcomes = raw_metadata.pop("source_outcomes", {})
    source_details = raw_metadata.pop("_report_source_details", {})
    raw_metadata.setdefault("report_schema_version", REPORT_SCHEMA_VERSION)
    sources_out: dict[str, dict[str, Any]] = {}
    seen_source_names: dict[str, int] = {}
    for src in result.sources:
        comps = _components_by_type(src.components)
        total_components = sum(len(v) for v in comps.values())
        detail = source_outcomes.get(src.path) or source_details.get(src.path) or {}
        source_name = detail.get("source_name") or _friendly_source_name(src.path)
        source_key = _disambiguate_source_key(source_name, seen_source_names)
        source_path = detail.get("source_path") or src.path
        source_kind = detail.get("source_kind") or "local-path"
        sources_out[source_key] = {
            "source_name": source_name,
            "source_path": source_path,
            "components": comps,
            "relationships": [r.model_dump(mode="json") for r in src.relationships],
            "summary": {
                "status": detail.get("status") or "completed",
                "source_kind": source_kind,
                "assets_discovered": detail.get("assets_discovered") or total_components,
                "last_generated_at": (
                    detail.get("last_generated_at")
                    or raw_metadata.get("completed_at")
                ),
            },
        }
    return {
        "aibom_analysis": {
            "metadata": raw_metadata,
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
