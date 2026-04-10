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

from pathlib import Path

import pytest
import yaml

from aibom.models import (
    AIComponent,
    AIComponentType,
    RiskFlag,
    RiskScore,
    ScanResult,
    Severity,
    SourceResult,
)
from aibom.policy import (
    Policy,
    PolicyRuleKind,
    _rules_from_mapping,
    evaluate_policy,
    load_policy,
)


def test_load_valid_policy_yaml(tmp_path: Path) -> None:
    p = tmp_path / "policy.yaml"
    p.write_text(
        """
fail_on: medium
no_hardcoded_keys: true
max_risk_score: 40
blocked_models:
  - gpt-4
require_guardrail_for:
  - model
""",
        encoding="utf-8",
    )
    pol = load_policy(p)
    assert pol.fail_on == Severity.MEDIUM
    kinds = {r.rule for r in pol.rules}
    assert PolicyRuleKind.NO_HARDCODED_KEYS in kinds
    assert PolicyRuleKind.MAX_RISK_SCORE in kinds
    assert any(
        r.rule == PolicyRuleKind.MAX_RISK_SCORE and r.params["threshold"] == 40
        for r in pol.rules
    )
    assert any(
        r.rule == PolicyRuleKind.BLOCKED_MODELS and r.params["models"] == ["gpt-4"]
        for r in pol.rules
    )


def test_no_hardcoded_keys_violation() -> None:
    pol = load_policy_from_dict({"no_hardcoded_keys": True})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(
                        name="k",
                        component_type=AIComponentType.SECRET,
                        file_path="a.py",
                    )
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert not pr.passed
    assert any(v.rule == "no_hardcoded_keys" for v in pr.violations)


def test_require_pinned_models_violation() -> None:
    pol = load_policy_from_dict({"require_pinned_models": True})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(
                        name="m",
                        component_type=AIComponentType.MODEL,
                        model_name="gpt-4",
                    )
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert not pr.passed
    assert any(v.rule == "require_pinned_models" for v in pr.violations)


def test_require_pinned_models_accepts_registry_style_identifier() -> None:
    pol = load_policy_from_dict({"require_pinned_models": True})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(
                        name="m",
                        component_type=AIComponentType.MODEL,
                        model_name="acme-org/my-model",
                    )
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert pr.passed
    assert pr.violations == []


def test_max_risk_score_violation() -> None:
    pol = load_policy_from_dict({"max_risk_score": 10})
    risk = sr_risk_score(50)
    sr = ScanResult(sources=[SourceResult(path=".", components=[])], risk=risk)
    pr = evaluate_policy(pol, sr)
    assert not pr.passed
    assert any(v.rule == "max_risk_score" for v in pr.violations)


def test_blocked_models_violation() -> None:
    pol = load_policy_from_dict({"blocked_models": ["gpt-4"]})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(
                        name="chat",
                        component_type=AIComponentType.MODEL,
                        model_name="openai/gpt-4",
                    )
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert not pr.passed
    assert any(v.rule == "blocked_models" for v in pr.violations)


def test_require_guardrail_for_pass() -> None:
    pol = load_policy_from_dict({"require_guardrail_for": ["model"]})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(name="m", component_type=AIComponentType.MODEL),
                    AIComponent(name="g", component_type=AIComponentType.GUARDRAIL),
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert pr.passed
    assert pr.violations == []


def test_require_guardrail_for_fail() -> None:
    pol = load_policy_from_dict({"require_guardrail_for": ["model"]})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(name="m", component_type=AIComponentType.MODEL),
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert not pr.passed
    assert any(v.rule == "require_guardrail_for" for v in pr.violations)


def test_require_observability_for_pass() -> None:
    pol = load_policy_from_dict({"require_observability_for": ["agent"]})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(name="a", component_type=AIComponentType.AGENT),
                    AIComponent(name="o", component_type=AIComponentType.OBSERVABILITY),
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert pr.passed


def test_require_observability_for_fail() -> None:
    pol = load_policy_from_dict({"require_observability_for": ["agent"]})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(name="a", component_type=AIComponentType.AGENT),
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert not pr.passed
    assert any(v.rule == "require_observability_for" for v in pr.violations)


def test_max_unresolved_agentic_violation() -> None:
    pol = load_policy_from_dict({"max_unresolved_agentic": 0})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(
                        name="x",
                        component_type=AIComponentType.MODEL,
                        needs_agentic=True,
                    ),
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert not pr.passed
    assert any(v.rule == "max_unresolved_agentic" for v in pr.violations)


def test_empty_policy_passes() -> None:
    pol = load_policy_from_dict({})
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(name="m", component_type=AIComponentType.MODEL),
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert pr.passed
    assert pr.violations == []


def test_multiple_simultaneous_violations() -> None:
    pol = load_policy_from_dict(
        {
            "no_hardcoded_keys": True,
            "max_risk_score": 5,
        }
    )
    risk = sr_risk_score(80)
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(
                        name="k",
                        component_type=AIComponentType.SECRET,
                        file_path="s.py",
                    ),
                ],
            )
        ],
        risk=risk,
    )
    pr = evaluate_policy(pol, sr)
    assert not pr.passed
    rules = {v.rule for v in pr.violations}
    assert "no_hardcoded_keys" in rules
    assert "max_risk_score" in rules


def test_fail_on_filters_low_severity() -> None:
    pol = load_policy_from_dict({"require_guardrail_for": ["model"]})
    pol.fail_on = Severity.HIGH
    sr = ScanResult(
        sources=[
            SourceResult(
                path=".",
                components=[
                    AIComponent(name="m", component_type=AIComponentType.MODEL),
                ],
            )
        ]
    )
    pr = evaluate_policy(pol, sr)
    assert pr.violations
    assert pr.passed


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("[unclosed", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_policy(bad)


def test_empty_yaml_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_policy(p)


def load_policy_from_dict(data: dict) -> Policy:
    return Policy(fail_on=None, rules=_rules_from_mapping(data))


def sr_risk_score(score: int) -> RiskScore:
    r = RiskScore()
    if score > 0:
        r.add_flag(
            RiskFlag(
                flag="test",
                severity=Severity.INFO,
                weight=score,
                description="test",
            )
        )
    return r
