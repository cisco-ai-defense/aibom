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

from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from aibom.models import AIComponentType, ScanResult, Severity
from aibom.risk import _is_pinned

_DEFAULT_RULE_SEVERITY: dict[str, Severity] = {
    "no_hardcoded_keys": Severity.CRITICAL,
    "require_pinned_models": Severity.HIGH,
    "max_risk_score": Severity.HIGH,
    "blocked_models": Severity.HIGH,
    "require_guardrail_for": Severity.MEDIUM,
    "require_observability_for": Severity.MEDIUM,
    "max_unresolved_agentic": Severity.MEDIUM,
}


class PolicyRuleKind(str, Enum):
    NO_HARDCODED_KEYS = "no_hardcoded_keys"
    REQUIRE_PINNED_MODELS = "require_pinned_models"
    MAX_RISK_SCORE = "max_risk_score"
    BLOCKED_MODELS = "blocked_models"
    REQUIRE_GUARDRAIL_FOR = "require_guardrail_for"
    REQUIRE_OBSERVABILITY_FOR = "require_observability_for"
    MAX_UNRESOLVED_AGENTIC = "max_unresolved_agentic"


class PolicyRule(BaseModel):
    rule: PolicyRuleKind
    params: dict[str, Any] = Field(default_factory=dict)
    severity: Severity


class Policy(BaseModel):
    fail_on: Optional[Severity] = None
    rules: list[PolicyRule] = Field(default_factory=list)


class PolicyViolation(BaseModel):
    rule: str
    message: str
    severity: Severity
    component_name: Optional[str] = None
    file_path: Optional[str] = None


class PolicyResult(BaseModel):
    violations: list[PolicyViolation] = Field(default_factory=list)
    passed: bool = True
    summary: dict[str, Any] = Field(default_factory=dict)


def _parse_component_type(value: str) -> AIComponentType:
    key = value.strip().lower()
    for m in AIComponentType:
        if m.value == key:
            return m
    raise ValueError(f"Unknown component type: {value!r}")


def _rules_from_mapping(data: dict[str, Any]) -> list[PolicyRule]:
    rules: list[PolicyRule] = []
    raw_rules = data.get("rules")
    if raw_rules is not None:
        if not isinstance(raw_rules, list):
            raise ValueError("'rules' must be a list")
        for item in raw_rules:
            if not isinstance(item, dict):
                raise ValueError("Each policy rule must be a mapping")
            rk = item.get("rule")
            if rk is None:
                raise ValueError("Policy rule missing 'rule' field")
            kind = PolicyRuleKind(str(rk))
            sev_raw = item.get("severity")
            severity = (
                Severity(str(sev_raw).lower())
                if sev_raw is not None
                else _DEFAULT_RULE_SEVERITY[kind.value]
            )
            params = item.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("Policy rule 'params' must be a mapping")
            rules.append(PolicyRule(rule=kind, params=params, severity=severity))
        return rules

    if data.get("no_hardcoded_keys"):
        rules.append(
            PolicyRule(
                rule=PolicyRuleKind.NO_HARDCODED_KEYS,
                params={},
                severity=_DEFAULT_RULE_SEVERITY["no_hardcoded_keys"],
            )
        )
    if data.get("require_pinned_models"):
        rules.append(
            PolicyRule(
                rule=PolicyRuleKind.REQUIRE_PINNED_MODELS,
                params={},
                severity=_DEFAULT_RULE_SEVERITY["require_pinned_models"],
            )
        )
    mrs = data.get("max_risk_score")
    if mrs is not None:
        rules.append(
            PolicyRule(
                rule=PolicyRuleKind.MAX_RISK_SCORE,
                params={"threshold": int(mrs)},
                severity=_DEFAULT_RULE_SEVERITY["max_risk_score"],
            )
        )
    bm = data.get("blocked_models")
    if bm is not None:
        if not isinstance(bm, list):
            raise ValueError("blocked_models must be a list")
        rules.append(
            PolicyRule(
                rule=PolicyRuleKind.BLOCKED_MODELS,
                params={"models": [str(x) for x in bm]},
                severity=_DEFAULT_RULE_SEVERITY["blocked_models"],
            )
        )
    rg = data.get("require_guardrail_for")
    if rg is not None:
        if not isinstance(rg, list):
            raise ValueError("require_guardrail_for must be a list")
        rules.append(
            PolicyRule(
                rule=PolicyRuleKind.REQUIRE_GUARDRAIL_FOR,
                params={"component_types": [str(x) for x in rg]},
                severity=_DEFAULT_RULE_SEVERITY["require_guardrail_for"],
            )
        )
    ro = data.get("require_observability_for")
    if ro is not None:
        if not isinstance(ro, list):
            raise ValueError("require_observability_for must be a list")
        rules.append(
            PolicyRule(
                rule=PolicyRuleKind.REQUIRE_OBSERVABILITY_FOR,
                params={"component_types": [str(x) for x in ro]},
                severity=_DEFAULT_RULE_SEVERITY["require_observability_for"],
            )
        )
    mua = data.get("max_unresolved_agentic")
    if mua is not None:
        rules.append(
            PolicyRule(
                rule=PolicyRuleKind.MAX_UNRESOLVED_AGENTIC,
                params={"max": int(mua)},
                severity=_DEFAULT_RULE_SEVERITY["max_unresolved_agentic"],
            )
        )

    return rules


def load_policy(path: str | Path) -> Policy:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        raise ValueError("Policy file is empty or not valid YAML")
    if not isinstance(data, dict):
        raise ValueError("Policy file must contain a YAML mapping at the root")

    fail_raw = data.get("fail_on")
    fail_on: Optional[Severity] = None
    if fail_raw is not None:
        fail_on = Severity(str(fail_raw).lower())

    rules = _rules_from_mapping(data)
    return Policy(fail_on=fail_on, rules=rules)


def _violations_fail_policy(violations: list[PolicyViolation], fail_on: Optional[Severity]) -> bool:
    if not violations:
        return False
    if fail_on is None:
        return True
    return any(v.severity >= fail_on for v in violations)


def _model_matches_blocklist(model_name: str, blocked: list[str]) -> bool:
    lower = model_name.lower()
    for b in blocked:
        bl = b.lower()
        if bl in lower or lower == bl:
            return True
    return False


def evaluate_policy(policy: Policy, scan_result: ScanResult) -> PolicyResult:
    violations: list[PolicyViolation] = []
    components = scan_result.all_components

    for pr in policy.rules:
        kind = pr.rule
        params = pr.params
        sev = pr.severity

        if kind == PolicyRuleKind.NO_HARDCODED_KEYS:
            for c in components:
                if c.component_type == AIComponentType.SECRET:
                    violations.append(
                        PolicyViolation(
                            rule=kind.value,
                            message="Hardcoded secret or credential material detected",
                            severity=sev,
                            component_name=c.name,
                            file_path=c.file_path or None,
                        )
                    )

        elif kind == PolicyRuleKind.REQUIRE_PINNED_MODELS:
            for c in components:
                if c.component_type != AIComponentType.MODEL:
                    continue
                mn = c.model_name
                if not mn or not str(mn).strip():
                    violations.append(
                        PolicyViolation(
                            rule=kind.value,
                            message=f"Model {c.name!r} has no pinned model_name",
                            severity=sev,
                            component_name=c.name,
                            file_path=c.file_path or None,
                        )
                    )
                elif not _is_pinned(str(mn)):
                    violations.append(
                        PolicyViolation(
                            rule=kind.value,
                            message=(
                                f"Model {c.name!r} uses unpinned model_name {mn!r}"
                            ),
                            severity=sev,
                            component_name=c.name,
                            file_path=c.file_path or None,
                        )
                    )

        elif kind == PolicyRuleKind.MAX_RISK_SCORE:
            if "threshold" not in params:
                raise ValueError("max_risk_score rule requires params.threshold")
            threshold = int(params["threshold"])
            if scan_result.risk.score > threshold:
                violations.append(
                    PolicyViolation(
                        rule=kind.value,
                        message=(
                            f"Risk score {scan_result.risk.score} exceeds "
                            f"maximum {threshold}"
                        ),
                        severity=sev,
                    )
                )

        elif kind == PolicyRuleKind.BLOCKED_MODELS:
            blocked = list(params.get("models") or [])
            for c in components:
                if c.component_type != AIComponentType.MODEL:
                    continue
                mn = c.model_name
                if mn and _model_matches_blocklist(str(mn), blocked):
                    violations.append(
                        PolicyViolation(
                            rule=kind.value,
                            message=f"Blocked model {mn!r} on component {c.name!r}",
                            severity=sev,
                            component_name=c.name,
                            file_path=c.file_path or None,
                        )
                    )

        elif kind == PolicyRuleKind.REQUIRE_GUARDRAIL_FOR:
            types_raw = params.get("component_types") or []
            wanted: list[AIComponentType] = []
            for t in types_raw:
                wanted.append(_parse_component_type(str(t)))
            if not wanted:
                continue
            has_guard = any(c.component_type == AIComponentType.GUARDRAIL for c in components)
            for wt in wanted:
                if any(c.component_type == wt for c in components) and not has_guard:
                    violations.append(
                        PolicyViolation(
                            rule=kind.value,
                            message=(
                                f"Components of type {wt.value!r} require a "
                                f"guardrail component, but none was detected"
                            ),
                            severity=sev,
                        )
                    )

        elif kind == PolicyRuleKind.REQUIRE_OBSERVABILITY_FOR:
            types_raw = params.get("component_types") or []
            wanted = [_parse_component_type(str(t)) for t in types_raw]
            if not wanted:
                continue
            has_obs = any(
                c.component_type == AIComponentType.OBSERVABILITY for c in components
            )
            for wt in wanted:
                if any(c.component_type == wt for c in components) and not has_obs:
                    violations.append(
                        PolicyViolation(
                            rule=kind.value,
                            message=(
                                f"Components of type {wt.value!r} require an "
                                f"observability component, but none was detected"
                            ),
                            severity=sev,
                        )
                    )

        elif kind == PolicyRuleKind.MAX_UNRESOLVED_AGENTIC:
            if "max" not in params:
                raise ValueError("max_unresolved_agentic rule requires params.max")
            max_allowed = int(params["max"])
            unresolved = [c for c in components if c.needs_agentic]
            if len(unresolved) > max_allowed:
                violations.append(
                    PolicyViolation(
                        rule=kind.value,
                        message=(
                            f"{len(unresolved)} component(s) still need agentic "
                            f"resolution (max allowed: {max_allowed})"
                        ),
                        severity=sev,
                    )
                )

    by_rule: dict[str, int] = {}
    for v in violations:
        by_rule[v.rule] = by_rule.get(v.rule, 0) + 1

    passed = not _violations_fail_policy(violations, policy.fail_on)
    summary: dict[str, Any] = {
        "total_violations": len(violations),
        "by_rule": by_rule,
        "evaluated_rules": len(policy.rules),
    }
    return PolicyResult(violations=violations, passed=passed, summary=summary)
