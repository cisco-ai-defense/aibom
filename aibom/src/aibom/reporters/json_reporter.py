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
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import IO, Any

from ..models import AIComponent, ScanResult
from ..source_attribution import capture_git_remote
from .base import BaseReporter

_LOGGER = logging.getLogger(__name__)

_REMOTE_ORG_REPO_RE = re.compile(r"[/:]([^/]+)/([^/]+?)(?:\.git)?$")
REPORT_SCHEMA_VERSION = "2"


def _friendly_source_name(path: str) -> str:
    """Derive a short, meaningful source label from a local filesystem path.

    Reads the raw repository-local ``remote.origin.url`` to extract
    ``org/repo`` from the actual remote. Falls back to the last path component.
    """
    resolved = Path(path).resolve()
    if resolved.is_dir():
        try:
            remote = capture_git_remote(str(resolved))
            if remote:
                m = _REMOTE_ORG_REPO_RE.search(remote)
                if m:
                    return f"{m.group(1)}/{m.group(2)}"
        except Exception:
            _LOGGER.debug("git remote lookup failed for %s", path)
    return PurePosixPath(path).name or path


def _source_attribution(path: str, detail: dict[str, Any]) -> dict[str, str]:
    """Resolve the source-attribution triple for one scanned source.

    The ``source_kind`` returned here is the *projection* kind (``git`` /
    ``container_image`` / ``local-path``) a downstream backend uses to derive a
    stable asset identity. This is a different taxonomy from the *analyze*
    source kind the pipeline records (``git-url`` / ``container`` /
    ``local-path``): a local git checkout is legitimately ``local-path`` for
    analyze and ``git`` for projection. The projection kind is therefore derived
    from the git/container reality of the source, not the analyze ``detail``
    kind — only a ``detail`` kind that is already a valid projection kind (e.g.
    a cross-repo source that resolved the image directly) is reused. Resolved
    ref/version values supplied in ``detail`` still win.
    """
    from ..source_attribution import (
        SOURCE_KIND_CONTAINER_IMAGE,
        SOURCE_KIND_GIT,
        canonicalize_source_ref,
        capture_source_ref_version,
        detect_source_kind,
    )

    detail_kind = detail.get("source_kind")
    if detail_kind in (SOURCE_KIND_GIT, SOURCE_KIND_CONTAINER_IMAGE):
        source_kind = detail_kind
    else:
        # ``detail_kind`` here is an analyze kind (or absent). Only the analyze
        # ``container`` kind hints at an image; everything else is detected from
        # the source so a local checkout is attributed as ``git`` rather than
        # being suppressed by its ``local-path`` analyze kind.
        is_image = (detail_kind == "container") or None
        source_kind = detect_source_kind(path, is_container_image=is_image)

    canonical = detail.get("source_ref_canonical")
    if not canonical:
        raw_ref = detail.get("source_ref")
        if not raw_ref and source_kind != SOURCE_KIND_CONTAINER_IMAGE:
            raw_ref = capture_git_remote(path)
        canonical = canonicalize_source_ref(raw_ref, source_kind) if raw_ref else ""

    version = detail.get("source_ref_version")
    if not version:
        version = (
            capture_source_ref_version(
                path,
                source_kind,
                image_digest=detail.get("image_digest"),
            )
            or ""
        )

    return {
        "source_kind": source_kind,
        "source_ref_canonical": canonical or "",
        "source_ref_version": version or "",
    }


def _components_by_type(
    components: Iterable[AIComponent],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for comp in components:
        key = comp.component_type.value
        d = comp.model_dump(mode="json")
        d["confidence"] = (
            comp.agentic_confidence
            if comp.agentic_confidence is not None
            else comp.heuristic_confidence
        )
        grouped.setdefault(key, []).append(d)
    return grouped


def _disambiguate_source_key(source_name: str, seen: dict[str, int]) -> str:
    count = seen.get(source_name, 0) + 1
    seen[source_name] = count
    if count == 1:
        return source_name
    return f"{source_name}#{count}"


def _component_summary(
    components_by_source: dict[str, list[AIComponent]],
) -> dict[str, list[dict[str, Any]]]:
    """Build a flat, human-readable view of detected components per source.

    Each entry is ``{component_type, name, file_path, line_number}``, sorted
    by ``(component_type, name)`` for stable, grouped output. Components
    marked ``metadata["test_only"]`` are excluded so the summary mirrors the
    ``total_components`` accounting in ``ScanResult.summary``.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for source_key, comps in components_by_source.items():
        entries: list[dict[str, Any]] = []
        for c in comps:
            if c.metadata.get("test_only"):
                continue
            entries.append(
                {
                    "component_type": c.component_type.value,
                    "name": c.name,
                    "file_path": c.file_path,
                    "line_number": c.line_number,
                }
            )
        entries.sort(key=lambda e: (e["component_type"], e["name"]))
        out[source_key] = entries
    return out


def _aibom_payload(
    result: ScanResult,
    *,
    include_component_summary: bool = False,
) -> dict[str, Any]:
    base = result.model_dump(mode="json")
    raw_metadata = dict(base["metadata"])
    source_outcomes = raw_metadata.pop("source_outcomes", {})
    source_details = raw_metadata.pop("_report_source_details", {})
    raw_metadata.setdefault("report_schema_version", REPORT_SCHEMA_VERSION)
    sources_out: dict[str, dict[str, Any]] = {}
    components_by_source: dict[str, list[AIComponent]] = {}
    seen_source_names: dict[str, int] = {}
    for src in result.sources:
        comps = _components_by_type(src.components)
        total_components = sum(len(v) for v in comps.values())
        detail = source_outcomes.get(src.path) or source_details.get(src.path) or {}
        source_name = detail.get("source_name") or _friendly_source_name(src.path)
        source_key = _disambiguate_source_key(source_name, seen_source_names)
        source_path = detail.get("source_path") or src.path
        attribution = _source_attribution(src.path, detail)
        # The summary keeps the analyze-side source kind (``git-url`` /
        # ``container`` / ``local-path``); only the attribution ``metadata``
        # block carries the projection kind. Fall back to the projection kind
        # when the analyze kind is absent (e.g. cross-repo discovery).
        summary_source_kind = detail.get("source_kind") or attribution["source_kind"]
        per_source_meta: dict[str, Any] = {
            "source_kind": attribution["source_kind"],
            "source_ref_canonical": attribution["source_ref_canonical"],
            "source_ref_version": attribution["source_ref_version"],
        }
        for mk in (
            "elapsed_s",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cached_tokens",
        ):
            val = detail.get(mk)
            if val is not None:
                per_source_meta[mk] = val
        sources_out[source_key] = {
            "source_name": source_name,
            "source_path": source_path,
            "components": comps,
            "relationships": [r.model_dump(mode="json") for r in src.relationships],
            "summary": {
                "status": detail.get("status") or "completed",
                "source_kind": summary_source_kind,
                "assets_discovered": detail.get("assets_discovered")
                or total_components,
                "last_generated_at": (
                    detail.get("last_generated_at") or raw_metadata.get("completed_at")
                ),
            },
            "metadata": per_source_meta,
        }
        components_by_source[source_key] = list(src.components)
    cross_repo_links_out = [
        link.model_dump(mode="json") for link in result.cross_repo_links
    ]
    analysis: dict[str, Any] = {
        "metadata": raw_metadata,
        "sources": sources_out,
        "summary": result.summary,
    }
    if include_component_summary:
        analysis["component_summary"] = _component_summary(components_by_source)
    analysis["risk"] = base["risk"]
    analysis["errors"] = base["errors"]
    if cross_repo_links_out:
        analysis["cross_repo_links"] = cross_repo_links_out
    return {"aibom_analysis": analysis}


class JsonReporter(BaseReporter):
    name = "json"
    file_extension = ".json"

    def __init__(self, include_component_summary: bool = False) -> None:
        """Create a JSON reporter.

        Args:
            include_component_summary: When ``True``, the rendered report
                includes a flat ``component_summary`` key (grouped by source)
                listing each non-test component as
                ``{component_type, name, file_path, line_number}``. Off by
                default so existing consumers see an unchanged payload.
        """
        self.include_component_summary = include_component_summary

    def render(self, result: ScanResult, output: IO[str]) -> None:
        payload = _aibom_payload(
            result,
            include_component_summary=self.include_component_summary,
        )
        json.dump(payload, output, indent=2)
        output.write("\n")

    def validate(self, result: ScanResult) -> list[str]:
        return []
