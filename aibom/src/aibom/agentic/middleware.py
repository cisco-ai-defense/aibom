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
from typing import Any

from ..models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    DetectionSource,
    RelationshipType,
    RiskFlag,
    Severity,
)

_LOGGER = logging.getLogger(__name__)


class AIBOMScannerMiddleware:
    """Extracts structured AIBOM data from agent output.

    After the agent finishes, call :meth:`extract_findings` on the final
    message content to obtain components, relationships, and risk flags
    that can be merged into the deterministic scan results.
    """

    def extract_findings(
        self, agent_output: str
    ) -> tuple[list[AIComponent], list[ComponentRelationship], list[RiskFlag]]:
        """Parse the agent's JSON output into AIBOM model objects."""
        data = self._parse_json(agent_output)
        if data is None:
            return [], [], []

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
        """Merge enrichments, removals, and reclassifications into *existing*.

        Processing order:
        1. Remove components flagged by ``remove_components``.
        2. Reclassify components flagged by ``reclassify_components``.
        3. Apply field updates from ``enriched_components``.

        Returns a new list.  Components not referenced are passed through.
        """
        data = self._parse_json(agent_output)
        if data is None:
            return list(existing)

        remove_ids: set[str] = set()
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
        for item in data.get("enriched_components", []):
            iid = item.get("instance_id", "")
            if iid:
                updates_by_id[iid] = item.get("updates", {})

        result: list[AIComponent] = []
        for comp in existing:
            if comp.instance_id in remove_ids:
                continue

            new_type_str = reclassify_map.get(comp.instance_id)
            if new_type_str:
                try:
                    new_type = AIComponentType(new_type_str)
                    comp = comp.model_copy(update={"component_type": new_type})
                except ValueError:
                    _LOGGER.warning(
                        "Invalid reclassify type '%s' for %s",
                        new_type_str, comp.instance_id,
                    )

            upd = updates_by_id.get(comp.instance_id)
            if upd is not None:
                merged_meta = dict(comp.metadata)
                merged_meta.update(upd.pop("metadata", {}))
                comp = comp.model_copy(update={**upd, "metadata": merged_meta})

            result.append(comp)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Best-effort extraction of JSON from agent text output."""
        import re as _re

        text = text.strip()

        for fence in _re.finditer(r"```(?:json)?\s*\n(.*?)```", text, _re.DOTALL):
            block = fence.group(1).strip()
            if block.startswith("{"):
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

        start = text.find("{")
        if start == -1:
            _LOGGER.warning("Agent output contains no JSON object")
            return None
        end = text.rfind("}")
        if end == -1 or end <= start:
            _LOGGER.warning("Agent output contains malformed JSON")
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            _LOGGER.warning("Failed to parse agent JSON output")
            return None

    @staticmethod
    def _extract_new_components(data: dict[str, Any]) -> list[AIComponent]:
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

    @staticmethod
    def _extract_relationships(data: dict[str, Any]) -> list[ComponentRelationship]:
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
                )
            )
        return relationships

    @staticmethod
    def _extract_risk_flags(data: dict[str, Any]) -> list[RiskFlag]:
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
                )
            )
        return flags
