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

import logging
from typing import Any

from .models import (
    AIComponent,
    AIComponentType,
    RiskFlag,
    RiskScore,
    ScanResult,
    Severity,
)

_LOGGER = logging.getLogger(__name__)

RISK_WEIGHTS: dict[str, dict[str, Any]] = {
    "hardcoded_api_key": {
        "weight": 35,
        "severity": Severity.HIGH,
        "description": "Hardcoded AI API key detected in source code",
    },
    "shadow_ai": {
        "weight": 20,
        "severity": Severity.MEDIUM,
        "description": "Unregistered AI service usage detected",
    },
    "deprecated_model": {
        "weight": 15,
        "severity": Severity.MEDIUM,
        "description": "Deprecated model version in use",
    },
    "unpinned_model": {
        "weight": 10,
        "severity": Severity.LOW,
        "description": "Model version not pinned to a specific release",
    },
    "mcp_unknown_server": {
        "weight": 25,
        "severity": Severity.HIGH,
        "description": "MCP server with no known provenance",
    },
    "internet_facing": {
        "weight": 20,
        "severity": Severity.MEDIUM,
        "description": "AI endpoint exposed to the internet",
    },
    "no_auth": {
        "weight": 25,
        "severity": Severity.HIGH,
        "description": "AI endpoint with no authentication",
    },
    "critical_cve": {
        "weight": 40,
        "severity": Severity.CRITICAL,
        "description": "Critical vulnerability (CVSS >= 9.0) in AI dependency",
    },
    "high_cve": {
        "weight": 25,
        "severity": Severity.HIGH,
        "description": "High vulnerability (CVSS >= 7.0) in AI dependency",
    },
    "medium_cve": {
        "weight": 10,
        "severity": Severity.MEDIUM,
        "description": "Medium vulnerability (CVSS >= 4.0) in AI dependency",
    },
    "known_exploit": {
        "weight": 50,
        "severity": Severity.CRITICAL,
        "description": "Known exploited vulnerability (CISA KEV) in AI dependency",
    },
    "no_fix_available": {
        "weight": 15,
        "severity": Severity.MEDIUM,
        "description": "Vulnerability with no patched version available",
    },
    "mcp_security_finding": {
        "weight": 20,
        "severity": Severity.HIGH,
        "description": "Security finding from MCP server analysis",
    },
    "skill_security_finding": {
        "weight": 20,
        "severity": Severity.HIGH,
        "description": "Security finding from agent skill analysis",
    },
}


class RiskScorer:
    """Evaluates a ScanResult and populates its RiskScore."""

    def __init__(
        self,
        weight_overrides: dict[str, int] | None = None,
    ) -> None:
        self._weights = dict(RISK_WEIGHTS)
        if weight_overrides:
            for flag_name, weight in weight_overrides.items():
                if flag_name in self._weights:
                    self._weights[flag_name] = {
                        **self._weights[flag_name],
                        "weight": weight,
                    }

    def score(self, result: ScanResult) -> RiskScore:
        """Analyze all components in *result* and return the risk score."""
        risk = RiskScore()
        for component in result.all_components:
            self._check_component(component, risk)
        return risk

    def _check_component(self, component: AIComponent, risk: RiskScore) -> None:
        if component.component_type == AIComponentType.SECRET:
            self._add_flag(risk, "hardcoded_api_key", component)

        if (
            component.component_type == AIComponentType.MCP_SERVER
            and not component.framework
        ):
            self._add_flag(risk, "mcp_unknown_server", component)

        if (
            component.component_type.is_model_related
            and component.model_name
            and not _is_pinned(component.model_name)
        ):
            self._add_flag(risk, "unpinned_model", component)

        for flag_name in component.metadata.get("risk_flags", []):
            if flag_name in self._weights:
                self._add_flag(risk, flag_name, component)

    def _add_flag(
        self,
        risk: RiskScore,
        flag_name: str,
        component: AIComponent,
    ) -> None:
        config = self._weights.get(flag_name)
        if not config:
            return
        risk.add_flag(
            RiskFlag(
                flag=flag_name,
                severity=config["severity"],
                weight=config["weight"],
                description=config["description"],
                file_path=component.file_path,
                line_number=component.line_number,
            )
        )

    def should_fail(self, risk: RiskScore, threshold: Severity) -> bool:
        """Return True if *risk* meets or exceeds the given severity threshold."""
        return risk.severity >= threshold


def _is_pinned(model_name: str) -> bool:
    """Heuristic: a model name is pinned if it contains a date or version suffix."""
    parts = model_name.split("-")
    for part in parts:
        if part.isdigit() and len(part) >= 4:
            return True
        if any(c.isdigit() for c in part) and "." in part:
            return True
    return False
