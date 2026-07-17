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

"""Hosted-evaluation privacy contract tests.

These tests intentionally exercise only the public experiment entry point.  The
application must receive the exact, validated fixture locally, while the hosted
dataset and the returned application output use one internally consistent set
of pseudonymous identities by default.
"""

from __future__ import annotations

import builtins
import copy
import json
from types import SimpleNamespace
from typing import Any

import pytest

from aibom.galileo_evaluation import (
    ALLOW_PUBLIC_CLOUD_ENV_VAR,
    DECISION_SUITE_SCHEMA_VERSION,
    EVALUATION_PROJECT_ID_ENV_VAR,
    EXACT_IDENTITIES_ENV_VAR,
    FULL_CONTENT_ENV_VAR,
    ExactIdentityLoggingDenied,
    run_galileo_custom_function_experiment,
)

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"
_RAW = {
    "case": "RAW-CASE-CANARY",
    "candidate_id": "RAW-CANDIDATE-ID-CANARY",
    "discovery_id": "RAW-DISCOVERY-ID-CANARY",
    "instance": "RAW-INSTANCE-ID-CANARY",
    "discovery_instance": "RAW-DISCOVERY-INSTANCE-CANARY",
    "repository": "RAW-REPOSITORY-CANARY",
    "candidate_path": "src/RAW_PATH_CANARY.py",
    "discovery_path": "src/RAW_DISCOVERY_PATH_CANARY.py",
    "candidate_name": "RAW-NAME-CANARY",
    "discovery_name": "RAW-DISCOVERY-NAME-CANARY",
    "evidence_content": "RAW_EVIDENCE_CONTENT_CANARY = build_agent()",
    "evidence_metadata": "RAW-EVIDENCE-METADATA-CANARY",
}
_RAW_LABELS = {
    "experiment": "RAW-EXPERIMENT-CANARY",
    "group": "RAW-GROUP-CANARY",
    "tag_key": "RAW-TAG-KEY-CANARY",
    "tag_value": "RAW-TAG-VALUE-CANARY",
}
_CONFIDENTIAL_CANARIES = tuple(_RAW.values()) + tuple(_RAW_LABELS.values())


def _candidate(*, final_type: str = "agent") -> dict[str, Any]:
    return {
        "component_type": final_type,
        "name": _RAW["candidate_name"],
        "repository": _RAW["repository"],
        "source_path": _RAW["candidate_path"],
        "line_number": 17,
        "stable_case_id": _RAW["candidate_id"],
        "instance_id": _RAW["instance"],
        "metadata": {"language": "python"},
    }


def _discovery() -> dict[str, Any]:
    return {
        "component_type": "model",
        "name": _RAW["discovery_name"],
        "repository": _RAW["repository"],
        "source_path": _RAW["discovery_path"],
        "line_number": 31,
        "stable_case_id": _RAW["discovery_id"],
        "instance_id": _RAW["discovery_instance"],
        "metadata": {"language": "python"},
    }


def _suite_payload() -> dict[str, Any]:
    final_candidate = _candidate()
    discovery = _discovery()
    return {
        "schema_version": DECISION_SUITE_SCHEMA_VERSION,
        "metadata": {"dataset_version": "v1"},
        "cases": [
            {
                "case_id": _RAW["case"],
                "candidates": [_candidate(final_type="tool")],
                "expected_actions": {
                    _RAW["candidate_id"]: {
                        "action": "reclassify",
                        "target_type": "agent",
                    },
                    _RAW["discovery_id"]: "discover",
                },
                "expected_components": [final_candidate, discovery],
                "expected_discovered_components": [discovery],
                "expected_relationships": [
                    {
                        "relationship_type": "uses_model",
                        "source_case_id": _RAW["candidate_id"],
                        "target_case_id": _RAW["discovery_id"],
                    }
                ],
                "expected_risks": [
                    {
                        "case_id": _RAW["candidate_id"],
                        "risk_type": "dynamic_tool_loading",
                        "severity": "high",
                    }
                ],
                "approved_evidence": [
                    {
                        "source_path": _RAW["candidate_path"],
                        "start_line": 15,
                        "end_line": 18,
                        "content": _RAW["evidence_content"],
                        "evidence_kind": "source_code",
                        "metadata": {
                            "review_ticket": _RAW["evidence_metadata"],
                        },
                    }
                ],
                "metadata": {"language": "python"},
            }
        ],
    }


def _application_output() -> dict[str, Any]:
    return {
        "final_components": [_candidate(), _discovery()],
        "actions": {
            _RAW["candidate_id"]: {
                "action": "reclassify",
                "target_type": "agent",
            },
            _RAW["discovery_id"]: "discover",
        },
        "relationships": [
            {
                "relationship_type": "uses_model",
                "source_id": _RAW["candidate_id"],
                "target_id": _RAW["discovery_id"],
            }
        ],
        "risk_flags": [
            {
                "case_id": _RAW["candidate_id"],
                "risk_type": "dynamic_tool_loading",
                "severity": "high",
            }
        ],
        "execution_outcome": {"status": "success", "schema_valid": True},
    }


class _FakeLocalMetric:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.__dict__.update(kwargs)


def _install_hosted_runner(monkeypatch, runner):
    real_import = builtins.__import__
    fake_experiment_module = SimpleNamespace(run_experiment=runner)
    fake_config_module = SimpleNamespace(
        GalileoPythonConfig=SimpleNamespace(_instance=None)
    )
    fake_galileo_module = SimpleNamespace(
        LocalMetric=_FakeLocalMetric,
        StepType=SimpleNamespace(trace="trace"),
    )

    def fake_import(name, *args, **kwargs):
        if name == "galileo.experiments":
            return fake_experiment_module
        if name == "galileo.config":
            return fake_config_module
        if name == "galileo":
            return fake_galileo_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _configure_hosted_default(monkeypatch) -> None:
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setenv(EVALUATION_PROJECT_ID_ENV_VAR, _PROJECT_ID)
    monkeypatch.setenv(
        "AIBOM_GALILEO_HMAC_KEY",
        "hosted-pseudonymization-test-key-not-a-secret",
    )
    monkeypatch.delenv(EXACT_IDENTITIES_ENV_VAR, raising=False)
    monkeypatch.delenv(FULL_CONTENT_ENV_VAR, raising=False)


def _assert_no_confidential_canaries(value: Any) -> None:
    serialized = json.dumps(value, sort_keys=True, default=str)
    for canary in _CONFIDENTIAL_CANARIES:
        assert canary not in serialized


def _identity_by_id(payload: dict[str, Any], field: str) -> dict[str, dict[str, Any]]:
    identity_fields = (
        "component_type",
        "name",
        "repository",
        "source_path",
        "line_number",
        "stable_case_id",
    )
    return {
        item["stable_case_id"]: {key: item[key] for key in identity_fields}
        for item in payload[field]
    }


def test_default_hosted_experiment_pseudonymizes_rows_outputs_and_metadata(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    application_inputs: list[dict[str, Any]] = []

    def application(dataset_input: dict[str, Any]) -> dict[str, Any]:
        application_inputs.append(copy.deepcopy(dataset_input))
        # A hostile/buggy application must not be able to mutate the registry
        # used by a later invocation of the same hosted dataset row.
        dataset_input["candidates"][0]["name"] = "MUTATED-NAME"
        dataset_input["approved_evidence"][0]["metadata"][
            "review_ticket"
        ] = "MUTATED-TICKET"
        return _application_output()

    def fake_run_experiment(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        hosted_input = json.loads(kwargs["dataset"][0]["input"])
        captured["outputs"] = [
            kwargs["function"](copy.deepcopy(hosted_input)),
            kwargs["function"](copy.deepcopy(hosted_input)),
        ]
        calls_before_unknown = len(application_inputs)
        unknown_input = copy.deepcopy(hosted_input)
        unknown_input["case_id"] = "unknown-hosted-case"
        captured["unknown_output"] = kwargs["function"](unknown_input)
        captured["unknown_app_calls"] = len(application_inputs) - calls_before_unknown
        return "default-pseudonymous-result"

    _configure_hosted_default(monkeypatch)
    _install_hosted_runner(monkeypatch, fake_run_experiment)

    result = run_galileo_custom_function_experiment(
        _suite_payload(),
        experiment_name=_RAW_LABELS["experiment"],
        function=application,
        metrics=[],
        experiment_tags={_RAW_LABELS["tag_key"]: _RAW_LABELS["tag_value"]},
        experiment_group=_RAW_LABELS["group"],
    )

    assert result == "default-pseudonymous-result"
    assert len(application_inputs) == 2
    assert application_inputs[0] == application_inputs[1]
    original_input = application_inputs[0]
    assert original_input["case_id"] == _RAW["case"]
    assert original_input["candidates"][0]["repository"] == _RAW["repository"]
    assert original_input["candidates"][0]["source_path"] == _RAW["candidate_path"]
    assert original_input["candidates"][0]["name"] == _RAW["candidate_name"]
    assert original_input["candidates"][0]["instance_id"] == _RAW["instance"]
    assert original_input["approved_evidence"][0]["content"] == _RAW["evidence_content"]
    assert (
        original_input["approved_evidence"][0]["metadata"]["review_ticket"]
        == _RAW["evidence_metadata"]
    )

    hosted_row = captured["kwargs"]["dataset"][0]
    hosted_input = json.loads(hosted_row["input"])
    hosted_truth = json.loads(hosted_row["ground_truth"])
    hosted_output = json.loads(captured["outputs"][0])
    # Cover every value handed to the SDK, including experiment name, group,
    # tag keys/values, function metadata, rows, and output envelopes.
    _assert_no_confidential_canaries(captured)

    # Discoveries, edge endpoints, risks, and action keys must all refer to the
    # same pseudonymous IDs used in the hosted ground truth.
    expected_by_id = _identity_by_id(hosted_truth, "expected_components")
    output_by_id = _identity_by_id(hosted_output, "final_components")
    assert output_by_id == expected_by_id
    discovery = hosted_truth["expected_discovered_components"][0]
    discovery_id = discovery["stable_case_id"]
    baseline_id = hosted_input["candidates"][0]["stable_case_id"]
    assert discovery_id in output_by_id
    assert set(hosted_output["actions"]) == {baseline_id, discovery_id}
    expected_relationship = hosted_truth["expected_relationships"][0]
    assert hosted_output["relationships"] == [
        {
            "predicted_present": True,
            "relationship_type": expected_relationship["relationship_type"],
            "source_id": baseline_id,
            "target_id": discovery_id,
        }
    ]
    expected_risk = hosted_truth["expected_risks"][0]
    assert hosted_output["risk_flags"] == [
        {
            "case_id": baseline_id,
            "predicted_present": True,
            "risk_type": expected_risk["risk_type"],
            "severity": "high",
        }
    ]

    assert captured["unknown_app_calls"] == 0
    assert json.loads(captured["unknown_output"])["schema_valid"] is False


@pytest.mark.parametrize(
    ("approved_fixture", "identity_approval"),
    [(False, None), (True, None), (False, "true"), (True, "false")],
)
def test_exact_identity_mode_retains_the_existing_two_part_approval_gate(
    monkeypatch, approved_fixture, identity_approval
) -> None:
    imported: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("galileo"):
            imported.append(name)
            raise AssertionError("identity approval must precede SDK import")
        return real_import(name, *args, **kwargs)

    _configure_hosted_default(monkeypatch)
    if identity_approval is not None:
        monkeypatch.setenv(EXACT_IDENTITIES_ENV_VAR, identity_approval)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ExactIdentityLoggingDenied):
        run_galileo_custom_function_experiment(
            _suite_payload(),
            experiment_name="exact-identity-gated",
            function=lambda row: row,
            metrics=[],
            approved_fixture=approved_fixture,
            exact_identities=True,
        )
    assert imported == []


def test_approved_exact_identity_mode_retains_raw_rows(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_experiment(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "exact-result"

    _configure_hosted_default(monkeypatch)
    monkeypatch.setenv(EXACT_IDENTITIES_ENV_VAR, "true")
    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")
    _install_hosted_runner(monkeypatch, fake_run_experiment)

    result = run_galileo_custom_function_experiment(
        _suite_payload(),
        experiment_name="approved-exact-identities",
        function=lambda row: _application_output(),
        metrics=[],
        approved_fixture=True,
        exact_identities=True,
    )

    assert result == "exact-result"
    serialized_rows = json.dumps(captured["kwargs"]["dataset"], sort_keys=True)
    for canary in _RAW.values():
        assert canary in serialized_rows


def test_default_pseudonyms_preserve_deterministic_metric_scores(monkeypatch) -> None:
    runs: list[dict[str, Any]] = []

    def fake_run_experiment(*_args, **kwargs):
        row = kwargs["dataset"][0]
        output = kwargs["function"](json.loads(row["input"]))
        runs.append({"row": row, "output": output, "metrics": kwargs["metrics"]})
        return runs[-1]

    _configure_hosted_default(monkeypatch)
    _install_hosted_runner(monkeypatch, fake_run_experiment)
    run_galileo_custom_function_experiment(
        _suite_payload(),
        experiment_name="pseudonymous-metrics",
        function=lambda _row: _application_output(),
    )

    monkeypatch.setenv(EXACT_IDENTITIES_ENV_VAR, "true")
    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")
    run_galileo_custom_function_experiment(
        _suite_payload(),
        experiment_name="exact-metrics",
        function=lambda _row: _application_output(),
        approved_fixture=True,
        exact_identities=True,
    )

    assert len(runs) == 2
    score_sets: list[dict[str, float | None]] = []
    for run in runs:
        trace = SimpleNamespace(
            dataset_input=run["row"]["input"],
            dataset_output=run["row"]["ground_truth"],
            output=run["output"],
        )
        score_sets.append(
            {metric.name: metric.scorer_fn(trace) for metric in run["metrics"]}
        )

    assert score_sets[0] == score_sets[1]
    assert score_sets[0]["aibom.components.f1"] == 1.0
    assert score_sets[0]["aibom.discoveries.f1"] == 1.0
    assert score_sets[0]["aibom.relationships.f1"] == 1.0
    assert score_sets[0]["aibom.risks.f1"] == 1.0
    assert score_sets[0]["aibom.action_macro_f1"] == 1.0


def test_default_hosted_pseudonyms_are_fresh_for_each_experiment(monkeypatch) -> None:
    datasets: list[list[dict[str, Any]]] = []

    def fake_run_experiment(*_args, **kwargs):
        datasets.append(copy.deepcopy(kwargs["dataset"]))
        return "ok"

    _configure_hosted_default(monkeypatch)
    _install_hosted_runner(monkeypatch, fake_run_experiment)

    for _ in range(2):
        assert (
            run_galileo_custom_function_experiment(
                _suite_payload(),
                experiment_name="fresh-pseudonyms",
                function=lambda _row: _application_output(),
                metrics=[],
            )
            == "ok"
        )

    assert len(datasets) == 2
    _assert_no_confidential_canaries(datasets)
    assert datasets[0] != datasets[1]
    first_input = json.loads(datasets[0][0]["input"])
    second_input = json.loads(datasets[1][0]["input"])
    assert first_input["case_id"] != second_input["case_id"]
    assert (
        first_input["candidates"][0]["stable_case_id"]
        != second_input["candidates"][0]["stable_case_id"]
    )
