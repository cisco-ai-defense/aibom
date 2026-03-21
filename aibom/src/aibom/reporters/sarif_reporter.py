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
import re
from collections.abc import Iterable
from typing import IO, Any

from ..models import (
    AIComponent,
    AIComponentType,
    RiskFlag,
    ScanResult,
    Severity,
)
from ..utils.version import resolve_package_version
from .base import BaseReporter

_SARIF_SCHEMA_URI = (
    "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/"
    "sarif-2.1/schema/sarif-schema-2.1.0.json"
)
_INFORMATION_URI = "https://github.com/cisco-ai-defense/aibom"


def _tool_version(meta: dict[str, Any]) -> str:
    for key in ("analyzer_version", "version", "tool_version"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            return v
    return resolve_package_version("cisco-aibom")


def _component_message(comp: AIComponent) -> str:
    parts: list[str] = [
        f"AI component '{comp.name}' ({comp.component_type.value})",
    ]
    if comp.framework:
        parts.append(f"framework: {comp.framework}")
    if comp.description:
        parts.append(comp.description)
    return " ".join(parts)


def _component_level(comp: AIComponent) -> str:
    if comp.component_type is AIComponentType.SECRET:
        return "warning"
    return "note"


def _risk_level(severity: Severity) -> str:
    if severity == Severity.CRITICAL:
        return "error"
    if severity in (Severity.HIGH, Severity.MEDIUM):
        return "warning"
    return "note"


def _rule_id_for_component(comp: AIComponent) -> str:
    return f"aibom/{comp.component_type.value}"


def _rule_id_for_risk(flag: RiskFlag) -> str:
    raw = flag.flag.strip().lower()
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-")
    if not slug:
        slug = "finding"
    return f"aibom/risk/{slug}"


def _location(comp: AIComponent) -> list[dict[str, Any]]:
    if not comp.file_path:
        return []
    loc: dict[str, Any] = {
        "physicalLocation": {
            "artifactLocation": {"uri": comp.file_path},
        }
    }
    if comp.line_number > 0:
        loc["physicalLocation"]["region"] = {"startLine": comp.line_number}
    return [loc]


def _location_risk(flag: RiskFlag) -> list[dict[str, Any]]:
    if not flag.file_path:
        return []
    loc: dict[str, Any] = {
        "physicalLocation": {
            "artifactLocation": {"uri": flag.file_path},
        }
    }
    if flag.line_number > 0:
        loc["physicalLocation"]["region"] = {"startLine": flag.line_number}
    return [loc]


def _collect_rule_ids(result: ScanResult) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for comp in result.all_components:
        rid = _rule_id_for_component(comp)
        if rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    for flag in result.risk.flags:
        rid = _rule_id_for_risk(flag)
        if rid not in seen:
            seen.add(rid)
            ordered.append(rid)
    return ordered


def _rules(rule_ids: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": rid,
            "name": rid.replace("aibom/", "").replace("/", " "),
            "shortDescription": {"text": rid},
        }
        for rid in rule_ids
    ]


def _build_results(result: ScanResult) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for comp in result.all_components:
        results.append(
            {
                "ruleId": _rule_id_for_component(comp),
                "level": _component_level(comp),
                "message": {"text": _component_message(comp)},
                "locations": _location(comp),
            }
        )
    for flag in result.risk.flags:
        results.append(
            {
                "ruleId": _rule_id_for_risk(flag),
                "level": _risk_level(flag.severity),
                "message": {"text": f"{flag.flag}: {flag.description}"},
                "locations": _location_risk(flag),
            }
        )
    return results


def _build_sarif(result: ScanResult) -> dict[str, Any]:
    rule_ids = _collect_rule_ids(result)
    return {
        "$schema": _SARIF_SCHEMA_URI,
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "cisco-aibom",
                        "version": _tool_version(result.metadata),
                        "informationUri": _INFORMATION_URI,
                        "rules": _rules(rule_ids),
                    }
                },
                "results": _build_results(result),
            }
        ],
    }


def _sarif_validation_errors(doc: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(doc, dict):
        return ["SARIF root must be a JSON object"]
    if doc.get("version") != "2.1.0":
        errs.append('SARIF "version" must be "2.1.0"')
    runs = doc.get("runs")
    if not isinstance(runs, list) or not runs:
        errs.append('SARIF "runs" must be a non-empty array')
        return errs
    run0 = runs[0]
    if not isinstance(run0, dict):
        errs.append("SARIF runs[0] must be an object")
        return errs
    tool = run0.get("tool")
    if not isinstance(tool, dict):
        errs.append('SARIF run must include object "tool"')
    else:
        driver = tool.get("driver")
        if not isinstance(driver, dict):
            errs.append('SARIF tool must include object "driver"')
        else:
            if driver.get("name") != "cisco-aibom":
                errs.append('SARIF driver "name" must be "cisco-aibom"')
            rules = driver.get("rules")
            if not isinstance(rules, list):
                errs.append('SARIF driver "rules" must be an array')
    res = run0.get("results")
    if not isinstance(res, list):
        errs.append('SARIF run "results" must be an array')
    else:
        for i, item in enumerate(res):
            if not isinstance(item, dict):
                errs.append(f"SARIF results[{i}] must be an object")
                continue
            if "ruleId" not in item or not isinstance(item["ruleId"], str):
                errs.append(f"SARIF results[{i}] must have string ruleId")
            msg = item.get("message")
            if not isinstance(msg, dict):
                errs.append(f"SARIF results[{i}] must have object message")
            elif not isinstance(msg.get("text"), str):
                errs.append(
                    f"SARIF results[{i}] must have message.text string"
                )
            if "level" in item and item["level"] not in (
                "none",
                "note",
                "warning",
                "error",
            ):
                errs.append(f"SARIF results[{i}] has invalid level")
            locs = item.get("locations")
            if locs is not None and not isinstance(locs, list):
                errs.append(
                    f"SARIF results[{i}] locations must be array or omitted"
                )
    return errs


class SarifReporter(BaseReporter):
    name = "sarif"
    file_extension = ".sarif.json"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        doc = _build_sarif(result)
        json.dump(doc, output, indent=2)
        output.write("\n")

    def validate(self, result: ScanResult) -> list[str]:
        doc = _build_sarif(result)
        return _sarif_validation_errors(doc)
