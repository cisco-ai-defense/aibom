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

"""AIBOM middleware for the Deep Agents harness.

``AIBOMScannerMiddleware`` post-processes the agent's final message,
extracts structured AIBOM findings from the JSON output, and converts
them into ``AIComponent`` / ``ComponentRelationship`` / ``RiskFlag``
objects that merge into the deterministic ``ScanResult``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..models import (
    AIComponent,
    AIComponentType,
    CodeSnippet,
    ComponentRelationship,
    DetectionSource,
    DecisionAnnotation,
    EvidenceLocation,
    RelationshipType,
    RiskFlag,
    Severity,
)

_LOGGER = logging.getLogger(__name__)


def _ckey(c: AIComponent) -> tuple[str, str]:
    """Consolidation key matching ``scan_pipeline._consolidation_key``."""
    canonical = (c.model_name or c.name).lower().strip()
    return (canonical, c.component_type.value)


class AIBOMScannerMiddleware:
    """Extracts structured AIBOM data from agent output.

    After the agent finishes, call :meth:`extract_findings` on the final
    message content to obtain components, relationships, and risk flags
    that can be merged into the deterministic scan results.
    """

    def __init__(self, *, include_code_snippets: bool = False) -> None:
        self.include_code_snippets = include_code_snippets

    def extract_findings(
        self, agent_output: str
    ) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
        """Parse the agent's JSON string output into AIBOM model objects."""
        data = self._parse_json(agent_output)
        if data is None:
            return [], [], []
        return self.extract_findings_from_dict(data)

    def extract_findings_from_dict(
        self, data: dict[str, Any]
    ) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
        """Extract findings from an already-parsed dict."""
        components = self._extract_new_components(data)
        enrichments = self._extract_enrichments(data)
        relationships = self._extract_relationships(data)
        risk_flags = self._extract_risk_flags(data)
        return components + enrichments, relationships, risk_flags

    def apply_enrichments(
        self,
        existing: list[AIComponent],
        agent_output: str,
    ) -> list[AIComponent]:
        """Merge enrichments from a JSON string."""
        data = self._parse_json(agent_output)
        if data is None:
            return list(existing)
        return self.apply_enrichments_from_dict(existing, data)

    def hydrate_component(self, component: AIComponent) -> AIComponent:
        """Optionally attach a code snippet to an existing component annotation."""
        if not self.include_code_snippets or component.decision_annotation is None:
            return component
        annotation = self._hydrate_code_snippet(
            component.decision_annotation,
            fallback_file_path=component.file_path,
            fallback_line_number=component.line_number,
        )
        return component.model_copy(update={"decision_annotation": annotation})

    def apply_enrichments_from_dict(
        self,
        existing: list[AIComponent],
        data: dict[str, Any],
    ) -> list[AIComponent]:
        """Merge enrichments, removals, and reclassifications into *existing*.

        Processing order:
        1. Remove components flagged by ``remove_components``.
        2. Reclassify components flagged by ``reclassify_components``.
        3. Apply field updates from ``enriched_components``.

        Returns a new list.  Components not referenced are passed through.
        """

        remove_ids: set[str] = set()
        remove_keys: set[tuple[str, str]] = set()
        for item in data.get("remove_components", []):
            iid = item.get("instance_id", "")
            if iid:
                remove_ids.add(iid)
                _LOGGER.info(
                    "Agent removed component %s: %s",
                    iid, item.get("reason", ""),
                )

        reclassify_map: dict[str, str] = {}
        for item in data.get("reclassify_components", []):
            iid = item.get("instance_id", "")
            new_type = item.get("new_type", "")
            if iid and new_type:
                reclassify_map[iid] = new_type
                _LOGGER.info(
                    "Agent reclassified %s → %s: %s",
                    iid, new_type, item.get("reason", ""),
                )

        updates_by_id: dict[str, dict[str, Any]] = {}
        annotations_by_id: dict[str, DecisionAnnotation] = {}
        for item in data.get("enriched_components", []):
            iid = item.get("instance_id", "")
            if iid:
                updates_by_id[iid] = item.get("updates", {})
                annotation = self._decision_annotation_from_item(
                    item,
                    fallback_file_path=item.get("file_path", ""),
                    fallback_line_number=item.get("line_number", 0),
                )
                if annotation is not None:
                    annotations_by_id[iid] = annotation

        existing_ids = {c.instance_id for c in existing}
        unmatched_remove_ids = remove_ids - existing_ids
        if unmatched_remove_ids:
            id_to_key = {c.instance_id: _ckey(c) for c in existing}
            for bad_id in unmatched_remove_ids:
                name_part = bad_id.rsplit("_", 1)[0].rsplit("_", 1)[0] if "_" in bad_id else bad_id
                for comp in existing:
                    ck = id_to_key[comp.instance_id]
                    if ck[0] == name_part.lower().strip():
                        remove_keys.add(ck)
                        _LOGGER.warning(
                            "Removal fallback: agent returned unmatched id '%s'; "
                            "matched consolidation key %s via component %s",
                            bad_id, ck, comp.instance_id,
                        )
                        break

        result: list[AIComponent] = []
        for comp in existing:
            if comp.instance_id in remove_ids:
                continue
            if remove_keys and _ckey(comp) in remove_keys:
                _LOGGER.info(
                    "Removing %s via consolidation-key fallback", comp.instance_id,
                )
                continue

            new_type_str = reclassify_map.get(comp.instance_id)
            if new_type_str:
                try:
                    new_type = AIComponentType(new_type_str)
                    comp = comp.model_copy(update={
                        "component_type": new_type,
                        "needs_agentic": False,
                    })
                except ValueError:
                    _LOGGER.warning(
                        "Invalid reclassify type '%s' for %s",
                        new_type_str, comp.instance_id,
                    )

            upd = updates_by_id.get(comp.instance_id)
            annotation = annotations_by_id.get(comp.instance_id)
            if upd is not None:
                merged_meta = dict(comp.metadata)
                merged_meta.update(upd.pop("metadata", {}))
                raw_type = upd.pop("component_type", None)
                if isinstance(raw_type, str):
                    try:
                        upd["component_type"] = AIComponentType(raw_type)
                    except ValueError:
                        _LOGGER.warning("Invalid component_type '%s' in enrichment for %s", raw_type, comp.instance_id)
                comp = comp.model_copy(update={
                    **upd,
                    "metadata": merged_meta,
                    "decision_annotation": annotation,
                    "needs_agentic": False,
                })
            elif comp.needs_agentic:
                update: dict[str, Any] = {"needs_agentic": False}
                if annotation is not None:
                    update["decision_annotation"] = annotation
                comp = comp.model_copy(update=update)

            result.append(comp)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Parse the agent's final message as JSON."""
        text = text.strip()
        if not text:
            _LOGGER.warning("Agent returned empty output")
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            _LOGGER.warning(
                "Failed to parse agent JSON output — first 300 chars: %s",
                text[:300],
            )
            return None

    def _extract_new_components(self, data: dict[str, Any]) -> list[AIComponent]:
        components: list[AIComponent] = []
        for item in data.get("new_components", []):
            try:
                comp_type = AIComponentType(item.get("component_type", "other"))
            except ValueError:
                comp_type = AIComponentType.OTHER
            components.append(
                AIComponent(
                    name=item.get("name", "unknown"),
                    component_type=comp_type,
                    file_path=item.get("file_path", ""),
                    line_number=item.get("line_number", 0),
                    framework=item.get("framework", ""),
                    model_name=item.get("model_name"),
                    detection_source=DetectionSource.AGENTIC,
                    decision_annotation=self._decision_annotation_from_item(
                        item,
                        fallback_file_path=item.get("file_path", ""),
                        fallback_line_number=item.get("line_number", 0),
                    ),
                    metadata=item.get("metadata", {}),
                )
            )
        return components

    @staticmethod
    def _extract_enrichments(data: dict[str, Any]) -> list[AIComponent]:
        """Enrichments don't create new components; they update existing ones.

        We return an empty list here -- actual merging is done via
        :meth:`apply_enrichments`.
        """
        return []

    def _extract_relationships(self, data: dict[str, Any]) -> list[ComponentRelationship]:
        relationships: list[ComponentRelationship] = []
        for item in data.get("new_relationships", []):
            try:
                rel_type = RelationshipType(item.get("relationship_type", "CUSTOM"))
            except ValueError:
                rel_type = RelationshipType.CUSTOM
            relationships.append(
                ComponentRelationship(
                    source_instance_id="",
                    target_instance_id="",
                    source_name=item.get("source_name", ""),
                    target_name=item.get("target_name", ""),
                    relationship_type=rel_type,
                    decision_annotation=self._decision_annotation_from_item(item),
                )
            )
        return relationships

    def _extract_risk_flags(self, data: dict[str, Any]) -> list[RiskFlag]:
        from ..risk import RISK_WEIGHTS

        flags: list[RiskFlag] = []
        for item in data.get("risk_findings", []):
            flag_name = item.get("flag", "")
            try:
                severity = Severity(item.get("severity", "info"))
            except ValueError:
                severity = Severity.INFO

            weight_info = RISK_WEIGHTS.get(flag_name, {})
            weight = weight_info.get("weight", 5)

            flags.append(
                RiskFlag(
                    flag=flag_name,
                    severity=severity,
                    weight=weight,
                    description=item.get("description", ""),
                    file_path=item.get("file_path", ""),
                    line_number=item.get("line_number", 0),
                    decision_annotation=self._decision_annotation_from_item(
                        item,
                        fallback_file_path=item.get("file_path", ""),
                        fallback_line_number=item.get("line_number", 0),
                    ),
                )
            )
        return flags

    def _decision_annotation_from_item(
        self,
        item: dict[str, Any],
        *,
        fallback_file_path: str = "",
        fallback_line_number: int = 0,
    ) -> DecisionAnnotation | None:
        raw = item.get("decision_annotation")
        if not raw:
            return None
        try:
            annotation = DecisionAnnotation.model_validate(raw)
        except ValueError:
            _LOGGER.warning("Invalid decision_annotation in agent output: %s", raw)
            return None
        if not self.include_code_snippets:
            return annotation
        return self._hydrate_code_snippet(
            annotation,
            fallback_file_path=fallback_file_path,
            fallback_line_number=fallback_line_number,
        )

    def _hydrate_code_snippet(
        self,
        annotation: DecisionAnnotation,
        *,
        fallback_file_path: str = "",
        fallback_line_number: int = 0,
    ) -> DecisionAnnotation:
        if annotation.code_snippet is not None:
            return annotation

        location = next(
            (
                loc for loc in annotation.evidence_locations
                if loc.file_path and loc.start_line > 0
            ),
            None,
        )
        if location is None and fallback_file_path and fallback_line_number > 0:
            location = EvidenceLocation(
                file_path=fallback_file_path,
                start_line=fallback_line_number,
                end_line=fallback_line_number,
                role="primary",
            )
        if location is None:
            return annotation

        snippet = self._read_code_snippet(
            location.file_path,
            location.start_line,
            location.end_line or location.start_line,
        )
        if snippet is None:
            return annotation
        return annotation.model_copy(update={"code_snippet": snippet})

    @staticmethod
    def _read_code_snippet(
        file_path: str,
        start_line: int,
        end_line: int,
        *,
        max_lines: int = 30,
    ) -> CodeSnippet | None:
        if not file_path or start_line <= 0:
            return None
        path = Path(file_path)
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError:
            return None
        if not lines or start_line > len(lines):
            return None

        bounded_end = max(start_line, end_line)
        excerpt_end = min(bounded_end, start_line + max_lines - 1, len(lines))
        excerpt = "".join(lines[start_line - 1:excerpt_end])
        return CodeSnippet(
            file_path=file_path,
            start_line=start_line,
            end_line=excerpt_end,
            text=excerpt,
            truncated=bounded_end > excerpt_end,
        )
