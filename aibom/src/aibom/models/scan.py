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

import os
from typing import Any, Optional

from pydantic import BaseModel, Field

from .enums import AIComponentType, DetectionSource, RelationshipType, Severity


class AIComponent(BaseModel):
    """A detected AI asset in source code, configuration, or infrastructure."""

    name: str
    component_type: AIComponentType
    file_path: str = ""
    line_number: int = 0
    framework: str = ""
    detection_source: DetectionSource = DetectionSource.CODE_ANALYSIS
    confidence: float = 1.0
    needs_agentic: bool = False
    agentic_hint: str = ""

    model_name: Optional[str] = None
    embedding_model: Optional[str] = None
    description: Optional[str] = None
    text: Optional[str] = None
    transport: Optional[str] = None
    config_source: Optional[str] = None
    storage_uri: Optional[str] = None
    dataset_source: Optional[str] = None
    skill_format: Optional[str] = None

    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    training_info: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    kb_concept: Optional[str] = None
    kb_label: Optional[str] = None
    sdk_version: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)
    instance_id: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.instance_id:
            self.instance_id = f"{self.name}_{self.file_path}_{self.line_number}"


class ComponentRelationship(BaseModel):
    """A directed edge between two AI components."""

    source_instance_id: str
    target_instance_id: str
    relationship_type: RelationshipType = RelationshipType.CUSTOM
    label: str = ""
    source_name: str = ""
    target_name: str = ""
    source_type: AIComponentType = AIComponentType.OTHER
    target_type: AIComponentType = AIComponentType.OTHER

    def model_post_init(self, __context: Any) -> None:
        if not self.label:
            self.label = self.relationship_type.value


class RiskFlag(BaseModel):
    """A single risk indicator detected during scanning."""

    flag: str
    severity: Severity
    weight: int
    description: str
    file_path: str = ""
    line_number: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskScore(BaseModel):
    """Aggregated risk assessment from all scanners."""

    score: int = 0
    severity: Severity = Severity.INFO
    flags: list[RiskFlag] = Field(default_factory=list)

    def add_flag(self, flag: RiskFlag) -> None:
        self.flags.append(flag)
        self.score = min(100, self.score + flag.weight)
        self._recompute_severity()

    def _recompute_severity(self) -> None:
        if self.score >= 76:
            self.severity = Severity.CRITICAL
        elif self.score >= 51:
            self.severity = Severity.HIGH
        elif self.score >= 26:
            self.severity = Severity.MEDIUM
        elif self.score > 0:
            self.severity = Severity.LOW
        else:
            self.severity = Severity.INFO


class SourceResult(BaseModel):
    """Scan results for a single source path (file or directory)."""

    path: str
    components: list[AIComponent] = Field(default_factory=list)
    relationships: list[ComponentRelationship] = Field(default_factory=list)


class ScanContext(BaseModel):
    """Input context for a scan run."""

    model_config = {"arbitrary_types_allowed": True}

    paths: list[str]
    exclude_patterns: list[str] = Field(default_factory=list)
    output_format: str = "json"
    output_file: Optional[str] = None
    config: dict[str, Any] = Field(default_factory=dict)
    kb_path: Optional[str] = None
    llm_config: Optional[dict[str, Any]] = None
    fail_on: Optional[Severity] = None
    min_severity: Severity = Severity.INFO
    ai_package_set: Optional[frozenset[str]] = None

    _file_index: Optional[dict[str, list["_IndexedFile"]]] = None

    def file_index(self) -> dict[str, list["_IndexedFile"]]:
        """Shared file listing built once and reused by all scanners.

        Returns a dict keyed by file extension (e.g. ``".py"``, ``".yaml"``),
        with values being lists of ``_IndexedFile(path, root)`` tuples.
        Files matching exclude patterns or skip segments are pre-filtered.
        """
        if self._file_index is not None:
            return self._file_index

        from pathlib import Path
        from pathspec import PathSpec

        spec: Optional[PathSpec] = None
        if self.exclude_patterns:
            spec = PathSpec.from_lines("gitwildmatch", self.exclude_patterns)

        from ..utils.path_filter import should_skip_dir

        idx: dict[str, list[_IndexedFile]] = {}

        for scan_root in self.paths:
            root = Path(scan_root)
            if not root.exists():
                continue
            if root.is_file():
                ext = root.suffix.lower()
                idx.setdefault(ext, []).append(_IndexedFile(root, root.parent))
                continue
            base = root.resolve()
            for dirpath_s, dirnames, filenames in os.walk(base):
                dirpath = Path(dirpath_s)
                dirnames[:] = [
                    d for d in dirnames
                    if not should_skip_dir(d)
                ]
                for fn in filenames:
                    fpath = dirpath / fn
                    if spec:
                        try:
                            rel = fpath.relative_to(base).as_posix()
                            if spec.match_file(rel):
                                continue
                        except ValueError:
                            pass
                    ext = fpath.suffix.lower()
                    idx.setdefault(ext, []).append(_IndexedFile(fpath, base))

        self._file_index = idx
        return idx


class _IndexedFile:
    """Lightweight container for a file path and its scan root."""
    __slots__ = ("path", "root")

    def __init__(self, path: "Any", root: "Any") -> None:
        self.path = path
        self.root = root


class ScanResult(BaseModel):
    """Unified output from a complete scan run. All reporters consume this."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    sources: list[SourceResult] = Field(default_factory=list)
    risk: RiskScore = Field(default_factory=RiskScore)
    errors: list[str] = Field(default_factory=list)

    @property
    def all_components(self) -> list[AIComponent]:
        return [c for s in self.sources for c in s.components]

    @property
    def all_relationships(self) -> list[ComponentRelationship]:
        return [r for s in self.sources for r in s.relationships]

    @property
    def confirmed_components(self) -> list[AIComponent]:
        return [c for c in self.all_components if not c.needs_agentic]

    @property
    def agentic_candidates(self) -> list[AIComponent]:
        return [c for c in self.all_components if c.needs_agentic]

    @property
    def summary(self) -> dict[str, Any]:
        components = self.all_components
        by_type: dict[str, int] = {}
        for c in components:
            if not c.needs_agentic:
                by_type[c.component_type.value] = by_type.get(c.component_type.value, 0) + 1
        agentic = [c for c in components if c.needs_agentic]
        return {
            "total_sources": len(self.sources),
            "total_components": len(components) - len(agentic),
            "agentic_candidates": len(agentic),
            "component_types": by_type,
            "total_relationships": len(self.all_relationships),
            "risk_score": self.risk.score,
            "risk_severity": self.risk.severity.value,
        }
