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

import builtins
import copy
import json
import socket
import ssl
import urllib.request
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from aibom.galileo_evaluation import (
    AIBOM_EVIDENCE_GROUNDING_METRIC_NAME,
    AIBOM_EVIDENCE_GROUNDING_PROMPT,
    ALLOW_PUBLIC_CLOUD_ENV_VAR,
    DECISION_SUITE_SCHEMA_VERSION,
    EVALUATION_LOG_STREAM_ID_ENV_VAR,
    EVALUATION_PROJECT_ID_ENV_VAR,
    EXACT_IDENTITIES_ENV_VAR,
    FULL_CONTENT_ENV_VAR,
    FULL_TRAJECTORY_ENV_VAR,
    GALILEO_DECISION_OUTPUT_SCHEMA_VERSION,
    ApprovedEvidenceExcerpt,
    DecisionSuite,
    ExactIdentityLoggingDenied,
    FullContentLoggingDenied,
    GalileoIntegrationUnavailable,
    HostedGalileoDestinationRequired,
    adapt_pipeline_result_for_galileo,
    build_aibom_evidence_grounding_metric,
    build_galileo_decision_metrics,
    build_galileo_experiment_rows,
    create_full_trajectory_callback_factory,
    create_galileo_async_callback,
    run_galileo_custom_function_experiment,
    sanitize_galileo_decision_output,
    validate_decision_suite,
)

_EVALUATION_PROJECT_ID = "11111111-1111-4111-8111-111111111111"
_EVALUATION_LOG_STREAM_ID = "22222222-2222-4222-8222-222222222222"


def _approve_exact_identity_evaluation(monkeypatch) -> None:
    monkeypatch.setenv(EXACT_IDENTITIES_ENV_VAR, "true")
    monkeypatch.setenv(EVALUATION_PROJECT_ID_ENV_VAR, _EVALUATION_PROJECT_ID)


def _approve_full_trajectory_evaluation(monkeypatch) -> None:
    monkeypatch.setenv(EXACT_IDENTITIES_ENV_VAR, "true")
    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")
    monkeypatch.setenv(FULL_TRAJECTORY_ENV_VAR, "true")
    monkeypatch.setenv(EVALUATION_PROJECT_ID_ENV_VAR, _EVALUATION_PROJECT_ID)
    monkeypatch.setenv(
        EVALUATION_LOG_STREAM_ID_ENV_VAR,
        _EVALUATION_LOG_STREAM_ID,
    )


def _unsafe_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _suite_payload() -> dict[str, Any]:
    return {
        "schema_version": DECISION_SUITE_SCHEMA_VERSION,
        "metadata": {"revision": 2, "suite_name": "example-regression"},
        "cases": [
            {
                "case_id": "case-b",
                "candidate": {
                    "type": "TOOL",
                    "name": "model client",
                    "repository": "org/repo",
                    "file_path": "src/client.py",
                    "line": 18,
                    "stable_case_id": "case-b",
                    "instance_id": "scan-17-client",
                    "metadata": {
                        "confidence": 0.82,
                        "framework": "langchain",
                    },
                },
                "expected_action": {
                    "action": "RECLASSIFY",
                    "target_type": "AGENT",
                    "reason_codes": ["structural evidence", "type correction"],
                    "metadata": {"reviewer_count": 2},
                },
                "expected_relationships": [
                    {
                        "label": "USES_MODEL",
                        "source": "case-b",
                        "target": "case-b",
                        "metadata": {"evidence_count": 3},
                    }
                ],
                "expected_risks": [
                    {
                        "risk_flag": "unresolved_model_reference",
                        "severity": "medium",
                    },
                    {
                        "risk_flag": "dynamic_tool_loading",
                        "severity": "HIGH",
                        "expected_present": True,
                    },
                ],
                "metadata": {"category": "reclassification", "priority": 1},
            },
            {
                "case_id": "case-a",
                "candidate": {
                    "component_type": "MODEL",
                    "name": "gpt-4o",
                    "repository": "org/repo",
                    "source_path": "src/models.py",
                    "line_number": 7,
                    "metadata": {"eval_case_id": "case-a"},
                },
                "expected_action": "KEEP",
                "expected_relationships": [],
                "expected_risks": [],
                "metadata": {"category": "keep"},
            },
        ],
    }


def _suite_payload_with_evidence() -> dict[str, Any]:
    payload = _suite_payload()
    payload["cases"][0]["approved_evidence"] = [
        {
            "path": "src/./client.py",
            "start_line": 16,
            "end_line": 20,
            "content": "client = build_model_client()\nclient.invoke(request)\n",
            "kind": "Source Code",
            "metadata": {
                "language": "python",
                "review_ticket": "EXAMPLE-001",
            },
        }
    ]
    return payload


def _batch_suite_payload() -> dict[str, Any]:
    return {
        "schema_version": DECISION_SUITE_SCHEMA_VERSION,
        "metadata": {"dataset_version": "example-v1"},
        "cases": [
            {
                "case_id": "batch-001",
                "candidates": [
                    {
                        "stable_case_id": "candidate-router",
                        "component_type": "tool",
                        "name": "router",
                        "repository": "org-repo",
                        "source_path": "src/router.py",
                        "line_number": 12,
                    },
                    {
                        "stable_case_id": "candidate-obsolete",
                        "component_type": "dependency",
                        "name": "obsolete-helper",
                        "repository": "org-repo",
                        "source_path": "src/legacy.py",
                        "line_number": 7,
                    },
                ],
                "expected_actions": {
                    "candidate-router": {
                        "action": "reclassify",
                        "target_type": "agent",
                    },
                    "candidate-obsolete": "remove",
                    "discovery-model": "discover",
                },
                "expected_components": [
                    {
                        "stable_case_id": "candidate-router",
                        "component_type": "agent",
                        "name": "router",
                        "repository": "org-repo",
                        "source_path": "src/router.py",
                        "line_number": 12,
                    },
                    {
                        "stable_case_id": "discovery-model",
                        "component_type": "model",
                        "name": "internal-model",
                        "repository": "org-repo",
                        "source_path": "src/models.py",
                        "line_number": 4,
                    },
                ],
                "expected_discovered_components": [
                    {
                        "stable_case_id": "discovery-model",
                        "component_type": "model",
                        "name": "internal-model",
                        "repository": "org-repo",
                        "source_path": "src/models.py",
                        "line_number": 4,
                    }
                ],
                "expected_relationships": [
                    {
                        "relationship_type": "uses_model",
                        "source_case_id": "candidate-router",
                        "target_case_id": "discovery-model",
                    }
                ],
                "expected_risks": [
                    {
                        "case_id": "candidate-router",
                        "risk_type": "dynamic_tool_loading",
                        "severity": "high",
                    }
                ],
                "metadata": {"language": "python", "slice": "reclassification"},
            }
        ],
    }


def test_validate_decision_suite_normalizes_entity_labels():
    suite = validate_decision_suite(_suite_payload())

    assert isinstance(suite, DecisionSuite)
    assert suite.schema_version == DECISION_SUITE_SCHEMA_VERSION
    assert len(suite.cases) == 2
    case = suite.cases[0]
    assert case.candidate.component_type == "tool"
    assert case.candidate.source_path == "src/client.py"
    assert case.expected_action.action == "reclassify"
    assert case.expected_action.target_type == "agent"
    assert case.expected_action.reason_codes == [
        "structural_evidence",
        "type_correction",
    ]
    assert case.expected_relationships[0].relationship_type == "uses_model"
    assert case.expected_relationships[0].source_case_id == "case-b"
    assert [risk.risk_type for risk in case.expected_risks] == [
        "unresolved_model_reference",
        "dynamic_tool_loading",
    ]
    assert case.expected_risks[1].severity == "high"
    assert case.expected_risks[1].case_id == "case-b"
    assert case.approved_evidence == []
    assert case.candidates == [case.candidate]
    assert case.expected_actions["case-b"] == case.expected_action
    assert case.expected_components[0].component_type == "agent"


def test_validate_suite_accepts_evidence_without_serialization_approval(monkeypatch):
    monkeypatch.delenv(FULL_CONTENT_ENV_VAR, raising=False)

    suite = validate_decision_suite(_suite_payload_with_evidence())

    evidence = suite.cases[0].approved_evidence[0]
    assert isinstance(evidence, ApprovedEvidenceExcerpt)
    assert evidence.source_path == "src/client.py"
    assert evidence.start_line == 16
    assert evidence.end_line == 20
    assert evidence.evidence_kind == "source_code"
    assert evidence.metadata == {
        "language": "python",
        "review_ticket": "EXAMPLE-001",
    }


def test_validate_decision_suite_accepts_canonical_json():
    payload = _suite_payload()

    suite = validate_decision_suite(json.dumps(payload))

    assert suite.model_dump(mode="json") == validate_decision_suite(payload).model_dump(
        mode="json"
    )


def test_batch_contract_validates_entity_level_gold_and_builds_full_rows():
    suite = validate_decision_suite(_batch_suite_payload())
    case = suite.cases[0]

    assert case.candidate is None
    assert case.expected_action is None
    assert [_candidate.stable_case_id for _candidate in case.candidates] == [
        "candidate-router",
        "candidate-obsolete",
    ]
    assert set(case.expected_actions) == {
        "candidate-router",
        "candidate-obsolete",
        "discovery-model",
    }
    assert [item.stable_case_id for item in case.expected_components] == [
        "candidate-router",
        "discovery-model",
    ]

    rows = build_galileo_experiment_rows(suite)
    assert len(rows) == 1
    row_input = json.loads(rows[0]["input"])
    ground_truth = json.loads(rows[0]["ground_truth"])

    assert "candidate" not in row_input
    assert "deterministic_relationships" not in row_input
    assert [item["stable_case_id"] for item in row_input["candidates"]] == [
        "candidate-router",
        "candidate-obsolete",
    ]
    assert "expected_action" not in ground_truth
    assert set(ground_truth["expected_actions"]) == {
        "candidate-router",
        "candidate-obsolete",
        "discovery-model",
    }
    assert ground_truth["expected_actions"]["candidate-router"] == {
        "action": "reclassify",
        "metadata": {},
        "reason_codes": [],
        "target_type": "agent",
    }
    assert [
        item["stable_case_id"]
        for item in ground_truth["expected_discovered_components"]
    ] == ["discovery-model"]
    assert ground_truth["expected_relationships"][0]["target_case_id"] == (
        "discovery-model"
    )
    assert ground_truth["expected_risks"][0]["case_id"] == "candidate-router"
    assert rows[0]["metadata"]["aibom.candidate_count"] == "2"
    assert rows[0]["metadata"]["aibom.component_type"] == "batch"


def test_single_candidate_batch_with_distinct_batch_id_scores_normally():
    payload = _batch_suite_payload()
    case = payload["cases"][0]
    case["candidates"] = [case["candidates"][0]]
    case["expected_actions"] = {
        "candidate-router": {
            "action": "reclassify",
            "target_type": "agent",
        }
    }
    case["expected_components"] = [case["expected_components"][0]]
    case["expected_discovered_components"] = []
    case["expected_relationships"] = []
    case["expected_risks"] = []

    row = build_galileo_experiment_rows(payload)[0]
    assert "candidate" not in json.loads(row["input"])
    assert "expected_action" not in json.loads(row["ground_truth"])

    output = sanitize_galileo_decision_output(
        {
            "final_components": [case["expected_components"][0]],
            "actions": case["expected_actions"],
        },
        dataset_input=row["input"],
        dataset_ground_truth=row["ground_truth"],
    )
    trace = SimpleNamespace(
        dataset_input=row["input"],
        dataset_output=row["ground_truth"],
        output=output,
    )
    scores = {
        metric.name: metric.scorer_fn(trace)
        for metric in build_galileo_decision_metrics()
    }

    assert scores["aibom.schema_validity"] == 1.0
    assert scores["aibom.components.f1"] == 1.0
    assert scores["aibom.action_accuracy"] == 1.0


def test_expected_execution_outcome_is_structured_and_serialized():
    payload = _batch_suite_payload()
    payload["cases"][0]["expected_outcome"] = {
        "status": "PROVIDER OUTAGE",
        "schema_valid": False,
        "abstained": True,
        "degraded_candidate_count": 2,
        "retry_count": 1,
        "fallback_count": 1,
        "cache_hit": False,
        "tool_error_count": 0,
        "guard_denial_count": 0,
    }

    suite = validate_decision_suite(payload)
    outcome = suite.cases[0].expected_execution_outcome
    ground_truth = json.loads(build_galileo_experiment_rows(suite)[0]["ground_truth"])

    assert outcome is not None
    assert outcome.status == "provider_outage"
    assert ground_truth["expected_execution_outcome"] == {
        "abstained": True,
        "cache_hit": False,
        "degraded_candidate_count": 2,
        "fallback_count": 1,
        "guard_denial_count": 0,
        "retry_count": 1,
        "schema_valid": False,
        "status": "provider_outage",
        "tool_error_count": 0,
    }


def test_native_pipeline_result_adapter_scores_degradation_and_retry_outcome():
    payload = _batch_suite_payload()
    case = payload["cases"][0]
    case["expected_execution_outcome"] = {
        "status": "degraded",
        "schema_valid": True,
        "degraded_candidate_count": 1,
        "retry_count": 1,
    }
    row = build_galileo_experiment_rows(payload)[0]
    native_result = SimpleNamespace(
        components=case["expected_components"],
        relationships=case["expected_relationships"],
        agentic_risk_flags=case["expected_risks"],
        agentic_degraded_count=1,
    )

    adapted = adapt_pipeline_result_for_galileo(
        native_result,
        execution_outcome={"schema_valid": True, "retry_count": 1},
    )
    serialized = sanitize_galileo_decision_output(
        adapted,
        dataset_input=row["input"],
        dataset_ground_truth=row["ground_truth"],
    )
    trace = SimpleNamespace(
        dataset_input=row["input"],
        dataset_output=row["ground_truth"],
        output=serialized,
    )
    scores = {
        metric.name: metric.scorer_fn(trace)
        for metric in build_galileo_decision_metrics()
    }

    assert json.loads(serialized)["execution_outcome"] == {
        "degraded_candidate_count": 1,
        "retry_count": 1,
        "schema_valid": True,
        "status": "degraded",
    }
    assert scores["aibom.components.f1"] == 1.0
    assert scores["aibom.execution.status_accuracy"] == 1.0
    assert scores["aibom.execution.schema_validity_accuracy"] == 1.0
    assert scores["aibom.execution.degraded_count_accuracy"] == 1.0
    assert scores["aibom.execution.retry_count_accuracy"] == 1.0


def test_pipeline_result_adapter_accepts_explicit_enrich_actions():
    payload = _batch_suite_payload()
    case = payload["cases"][0]
    candidate = case["candidates"][0]
    case["candidates"] = [candidate]
    case["expected_actions"] = {"candidate-router": "enrich"}
    case["expected_components"] = [copy.deepcopy(candidate)]
    case["expected_discovered_components"] = []
    case["expected_relationships"] = []
    case["expected_risks"] = []
    row = build_galileo_experiment_rows(payload)[0]
    native_component = {**candidate, "framework": "langchain"}
    native_result = SimpleNamespace(
        components=[native_component],
        relationships=[],
        agentic_risk_flags=[],
        agentic_degraded_count=0,
    )

    def _scores(adapted: dict[str, Any]) -> dict[str, float | None]:
        serialized = sanitize_galileo_decision_output(
            adapted,
            dataset_input=row["input"],
            dataset_ground_truth=row["ground_truth"],
        )
        trace = SimpleNamespace(
            dataset_input=row["input"],
            dataset_output=row["ground_truth"],
            output=serialized,
        )
        return {
            metric.name: metric.scorer_fn(trace)
            for metric in build_galileo_decision_metrics()
        }

    inferred = _scores(
        adapt_pipeline_result_for_galileo(native_result, dataset_input=row["input"])
    )
    explicit = _scores(
        adapt_pipeline_result_for_galileo(
            native_result,
            actions={"candidate-router": "enrich"},
            dataset_input=row["input"],
        )
    )

    assert inferred["aibom.components.f1"] == 1.0
    assert inferred["aibom.action_accuracy"] == 0.0
    assert explicit["aibom.action_accuracy"] == 1.0
    assert explicit["aibom.action_macro_f1"] == 1.0


def test_degraded_native_passthrough_is_an_abstention_not_a_keep():
    payload = _batch_suite_payload()
    case = payload["cases"][0]
    candidate = case["candidates"][0]
    case["candidates"] = [candidate]
    case["expected_actions"] = {"candidate-router": "keep"}
    case["expected_components"] = [copy.deepcopy(candidate)]
    case["expected_discovered_components"] = []
    case["expected_relationships"] = []
    case["expected_risks"] = []
    case["expected_execution_outcome"] = {
        "status": "degraded",
        "schema_valid": True,
        "abstained": True,
        "degraded_candidate_count": 1,
    }
    row = build_galileo_experiment_rows(payload)[0]
    native_result = SimpleNamespace(
        components=[{**candidate, "agentic_hint": "batch_timeout"}],
        relationships=[],
        agentic_risk_flags=[],
        agentic_degraded_count=1,
    )
    adapted = adapt_pipeline_result_for_galileo(
        native_result,
        execution_outcome={"schema_valid": True},
        dataset_input=row["input"],
    )
    serialized = sanitize_galileo_decision_output(
        adapted,
        dataset_input=row["input"],
        dataset_ground_truth=row["ground_truth"],
    )
    trace = SimpleNamespace(
        dataset_input=row["input"],
        dataset_output=row["ground_truth"],
        output=serialized,
    )
    scores = {
        metric.name: metric.scorer_fn(trace)
        for metric in build_galileo_decision_metrics()
    }

    assert scores["aibom.components.f1"] == 1.0
    assert scores["aibom.action_accuracy"] == 0.0
    assert scores["aibom.decision_coverage"] == 0.0
    assert scores["aibom.execution.abstention_accuracy"] == 1.0
    assert json.loads(serialized)["actions"] == {}


def test_execution_outcome_requires_at_least_one_strict_dimension():
    empty = _batch_suite_payload()
    empty["cases"][0]["expected_execution_outcome"] = {}
    with pytest.raises(ValidationError, match="at least one dimension"):
        validate_decision_suite(empty)

    coerced = _batch_suite_payload()
    coerced["cases"][0]["expected_execution_outcome"] = {"cache_hit": "false"}
    with pytest.raises(ValidationError):
        validate_decision_suite(coerced)


def test_baseline_relationship_alias_validates_and_serializes_canonically():
    payload = _batch_suite_payload()
    payload["cases"][0]["baseline_relationships"] = [
        {
            "relationship_type": "uses_model",
            "source_case_id": "candidate-router",
            "target_case_id": "candidate-obsolete",
        }
    ]

    suite = validate_decision_suite(payload)
    relationships = suite.cases[0].deterministic_relationships

    assert relationships is not None
    assert len(relationships) == 1
    assert relationships[0].source_case_id == "candidate-router"
    row_input = json.loads(build_galileo_experiment_rows(suite)[0]["input"])
    assert "baseline_relationships" not in row_input
    assert row_input["deterministic_relationships"] == [
        {
            "expected_present": True,
            "metadata": {},
            "relationship_type": "uses_model",
            "source_case_id": "candidate-router",
            "target_case_id": "candidate-obsolete",
        }
    ]


def test_explicit_empty_deterministic_relationships_remains_distinct_from_absent():
    payload = _batch_suite_payload()
    payload["cases"][0]["deterministic_relationships"] = []

    suite = validate_decision_suite(payload)
    row_input = json.loads(build_galileo_experiment_rows(suite)[0]["input"])

    assert suite.cases[0].deterministic_relationships == []
    assert row_input["deterministic_relationships"] == []


def test_deterministic_relationships_reject_duplicates_absences_and_unknown_endpoints():
    edge = {
        "relationship_type": "uses_model",
        "source_case_id": "candidate-router",
        "target_case_id": "candidate-obsolete",
    }

    duplicate = _batch_suite_payload()
    duplicate["cases"][0]["deterministic_relationships"] = [
        edge,
        copy.deepcopy(edge),
    ]
    with pytest.raises(ValidationError, match="duplicate present edges"):
        validate_decision_suite(duplicate)

    absent = _batch_suite_payload()
    absent_edge = copy.deepcopy(edge)
    absent_edge["expected_present"] = False
    absent["cases"][0]["deterministic_relationships"] = [absent_edge]
    with pytest.raises(ValidationError, match="present edges only"):
        validate_decision_suite(absent)

    unknown = _batch_suite_payload()
    unknown_edge = copy.deepcopy(edge)
    unknown_edge["target_case_id"] = "unknown-component"
    unknown["cases"][0]["deterministic_relationships"] = [unknown_edge]
    with pytest.raises(ValidationError, match="candidate stable_case_id"):
        validate_decision_suite(unknown)


def test_deterministic_relationships_are_bounded():
    payload = _batch_suite_payload()
    payload["cases"][0]["deterministic_relationships"] = [
        {
            "relationship_type": f"relationship-{index}",
            "source_case_id": "candidate-router",
            "target_case_id": "candidate-obsolete",
        }
        for index in range(257)
    ]

    with pytest.raises(ValidationError, match="at most 256 items"):
        validate_decision_suite(payload)


def test_batch_candidates_require_unique_stable_ids_and_complete_actions():
    missing_id = _batch_suite_payload()
    missing_id["cases"][0]["candidates"][0].pop("stable_case_id")
    with pytest.raises(ValidationError, match="requires a stable_case_id"):
        validate_decision_suite(missing_id)

    duplicate_id = _batch_suite_payload()
    duplicate_id["cases"][0]["candidates"][1]["stable_case_id"] = "candidate-router"
    with pytest.raises(ValidationError, match="duplicate stable_case_id"):
        validate_decision_suite(duplicate_id)

    missing_action = _batch_suite_payload()
    missing_action["cases"][0]["expected_actions"].pop("candidate-obsolete")
    with pytest.raises(ValidationError, match="exactly one action"):
        validate_decision_suite(missing_action)


@pytest.mark.parametrize("field_name", ["repository", "source_path"])
def test_golden_components_require_complete_location_identity(field_name):
    candidate = _batch_suite_payload()
    candidate["cases"][0]["candidates"][0][field_name] = ""
    with pytest.raises(ValidationError, match="repository and source_path"):
        validate_decision_suite(candidate)

    expected = _batch_suite_payload()
    expected["cases"][0]["expected_components"][0][field_name] = ""
    with pytest.raises(ValidationError, match="repository and source_path"):
        validate_decision_suite(expected)


def test_expected_discoveries_must_be_exact_final_entities_with_discover_actions():
    not_exact = _batch_suite_payload()
    not_exact["cases"][0]["expected_discovered_components"][0]["line_number"] = 99
    with pytest.raises(ValidationError, match="exact subset"):
        validate_decision_suite(not_exact)

    wrong_action = _batch_suite_payload()
    wrong_action["cases"][0]["expected_actions"]["discovery-model"] = "keep"
    with pytest.raises(ValidationError, match="discover"):
        validate_decision_suite(wrong_action)

    deterministic_discovery = _batch_suite_payload()
    deterministic_discovery["cases"][0]["candidates"].append(
        copy.deepcopy(
            deterministic_discovery["cases"][0]["expected_discovered_components"][0]
        )
    )
    with pytest.raises(ValidationError, match="absent from deterministic candidates"):
        validate_decision_suite(deterministic_discovery)


def test_candidate_actions_preserve_exact_identity_except_reclassified_type():
    changed_name = _batch_suite_payload()
    changed_name["cases"][0]["expected_components"][0]["name"] = "other-router"
    with pytest.raises(ValidationError, match="may change only component_type"):
        validate_decision_suite(changed_name)

    enrich_changed_location = _batch_suite_payload()
    case = enrich_changed_location["cases"][0]
    case["expected_actions"]["candidate-router"] = "enrich"
    case["expected_components"][0]["component_type"] = "tool"
    case["expected_components"][0]["line_number"] = 13
    with pytest.raises(ValidationError, match="preserve exact component identity"):
        validate_decision_suite(enrich_changed_location)


@pytest.mark.parametrize(
    "version",
    [None, "aibom.galileo.decision_suite.v0", "2", ""],
)
def test_schema_version_is_required_and_exact(version):
    payload = _suite_payload()
    if version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = version

    with pytest.raises(ValidationError):
        validate_decision_suite(payload)


def test_duplicate_case_and_entity_labels_are_rejected():
    duplicate_cases = _suite_payload()
    duplicate_cases["cases"][1]["case_id"] = "case-b"
    duplicate_cases["cases"][1]["candidate"]["metadata"] = {}
    with pytest.raises(ValidationError, match="case_id values must be unique"):
        validate_decision_suite(duplicate_cases)

    duplicate_relationships = _suite_payload()
    relation = duplicate_relationships["cases"][0]["expected_relationships"][0]
    duplicate_relationships["cases"][0]["expected_relationships"].append(
        copy.deepcopy(relation)
    )
    with pytest.raises(ValidationError, match="duplicate labels"):
        validate_decision_suite(duplicate_relationships)

    orphan_relationship = _batch_suite_payload()
    orphan_relationship["cases"][0]["expected_relationships"][0][
        "target_case_id"
    ] = "unknown-component"
    with pytest.raises(ValidationError, match="expected_relationship endpoints"):
        validate_decision_suite(orphan_relationship)

    duplicate_risks = _suite_payload()
    duplicate_risks["cases"][0]["expected_risks"].append(
        {
            "risk_type": "unresolved_model_reference",
            "severity": "medium",
        }
    )
    with pytest.raises(ValidationError, match="duplicate labels"):
        validate_decision_suite(duplicate_risks)


def test_present_relationship_cannot_reference_a_removed_candidate():
    payload = _batch_suite_payload()
    relationship = payload["cases"][0]["expected_relationships"][0]
    relationship["target_case_id"] = "candidate-obsolete"

    with pytest.raises(
        ValidationError,
        match="present expected_relationship endpoints",
    ):
        validate_decision_suite(payload)

    relationship["expected_present"] = False
    accepted = validate_decision_suite(payload).cases[0].expected_relationships[0]
    assert accepted.expected_present is False
    assert accepted.target_case_id == "candidate-obsolete"


def test_reclassification_requires_a_target_and_other_actions_forbid_it():
    missing_target = _suite_payload()
    missing_target["cases"][0]["expected_action"].pop("target_type")
    with pytest.raises(ValidationError, match="require target_type"):
        validate_decision_suite(missing_target)

    unexpected_target = _suite_payload()
    unexpected_target["cases"][1]["expected_action"] = {
        "action": "keep",
        "target_type": "model",
    }
    with pytest.raises(ValidationError, match="valid only for reclassify"):
        validate_decision_suite(unexpected_target)


@pytest.mark.parametrize(
    "source_path",
    [
        "/private/repo/main.py",
        "../outside.py",
        "src/../../outside.py",
        "C:\\repo\\x.py",
    ],
)
def test_candidate_paths_must_be_repository_relative(source_path):
    payload = _suite_payload()
    payload["cases"][0]["candidate"]["file_path"] = source_path

    with pytest.raises(ValidationError, match="source_path"):
        validate_decision_suite(payload)


@pytest.mark.parametrize(
    "source_path",
    [
        "",
        ".",
        "/private/repo/main.py",
        "../outside.py",
        "src/../../outside.py",
        "C:\\repo\\x.py",
        "src/control\nfile.py",
        f"src/{'x' * 2_049}",
    ],
)
def test_approved_evidence_paths_are_bounded_and_repository_relative(source_path):
    payload = _suite_payload_with_evidence()
    payload["cases"][0]["approved_evidence"][0]["path"] = source_path

    with pytest.raises(
        ValidationError,
        match=r"source_path|approved_evidence\.0\.path",
    ):
        validate_decision_suite(payload)


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"start_line": 0}, "start_line"),
        ({"start_line": True}, "start_line"),
        ({"end_line": 10_000_001}, "end_line"),
        ({"start_line": 20, "end_line": 19}, "end_line must be"),
        ({"start_line": 1, "end_line": 501}, "at most 500 lines"),
        ({"content": ""}, "content"),
        ({"content": "   \n\t"}, "content"),
        ({"content": "x" * 16_385}, "content"),
        ({"content": "😀" * 4_097}, "byte UTF-8 limit"),
        ({"kind": "x" * 65}, "kind"),
    ],
)
def test_approved_evidence_line_content_and_kind_bounds(updates, match):
    payload = _suite_payload_with_evidence()
    payload["cases"][0]["approved_evidence"][0].update(updates)

    with pytest.raises(ValidationError, match=match):
        validate_decision_suite(payload)


def test_approved_evidence_has_per_case_size_and_count_limits():
    oversized = _suite_payload_with_evidence()
    oversized["cases"][0]["approved_evidence"] = [
        {
            "path": f"src/evidence-{index}.py",
            "start_line": 1,
            "end_line": 1,
            "content": "x" * 16_384,
            "kind": "source",
        }
        for index in range(9)
    ]
    with pytest.raises(ValidationError, match="per-case limit"):
        validate_decision_suite(oversized)

    too_many = _suite_payload_with_evidence()
    too_many["cases"][0]["approved_evidence"] = [
        {
            "path": f"src/evidence-{index}.py",
            "start_line": 1,
            "end_line": 1,
            "content": "approved fixture",
            "kind": "source",
        }
        for index in range(33)
    ]
    with pytest.raises(ValidationError, match="at most 32 items"):
        validate_decision_suite(too_many)


@pytest.mark.parametrize(
    "metadata",
    [
        {"source_code": "print('private')"},
        {"safe": {"api-key": "private"}},
        {"nested": [{"raw_response": "private"}]},
        {"apiKey": "synthetic"},
        {"APIKey": "synthetic"},
        {"accessToken": "synthetic"},
        {"rawPrompt": "synthetic"},
        {"sourceContent": "synthetic"},
        {"clientSecret": "synthetic"},
        {"privateKey": "synthetic"},
    ],
)
def test_raw_content_and_secret_metadata_keys_are_rejected(metadata):
    payload = _suite_payload()
    payload["cases"][0]["metadata"] = metadata

    with pytest.raises(ValidationError, match="disallowed raw-content or secret"):
        validate_decision_suite(payload)


@pytest.mark.parametrize(
    "metadata",
    [
        {"notes": "raw source code and prompt text"},
        {"owner": "alice@example.internal"},
        {"location": "src/private/model.py"},
        {"credential_label": "sk-syntheticcredential"},
        {"slice": "x" * 97},
        {"nested": {"review": "multi word prose"}},
    ],
)
def test_label_only_metadata_rejects_arbitrary_or_secret_shaped_strings(metadata):
    payload = _suite_payload()
    payload["cases"][0]["metadata"] = metadata

    with pytest.raises(ValidationError, match="bounded slice labels"):
        validate_decision_suite(payload)


def test_label_only_metadata_accepts_only_bounded_slice_dimensions():
    payload = _suite_payload()
    payload["cases"][0]["metadata"] = {
        "category": "reclassification",
        "dataset_version": "example-v1",
        "language": "python",
        "nested": {"review_ticket": "EXAMPLE-001", "approved": True},
        "revision": 3,
    }

    suite = validate_decision_suite(payload)

    assert suite.cases[0].metadata == payload["cases"][0]["metadata"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"api_key": "fixture-value"},
        {"client-secret": "fixture-value"},
        {"nested": {"private_key": "fixture-value"}},
        {"nested": [{"service_token": "fixture-value"}]},
        {"content": "must use the bounded content field"},
    ],
)
def test_approved_evidence_rejects_secret_or_content_metadata_keys(metadata):
    payload = _suite_payload_with_evidence()
    payload["cases"][0]["approved_evidence"][0]["metadata"] = metadata

    with pytest.raises(ValidationError, match="disallowed raw-content or secret"):
        validate_decision_suite(payload)


def test_raw_candidate_fields_and_mismatched_case_ids_are_rejected():
    raw_candidate = _suite_payload()
    raw_candidate["cases"][0]["candidate"]["raw_content"] = "private source"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_decision_suite(raw_candidate)

    mismatched = _suite_payload()
    mismatched["cases"][0]["candidate"]["stable_case_id"] = "another-case"
    with pytest.raises(ValidationError, match="must match the enclosing case_id"):
        validate_decision_suite(mismatched)


@pytest.mark.parametrize(("field", "value"), [("name", None), ("type", None)])
def test_required_candidate_identity_fields_reject_null(field, value):
    payload = _suite_payload()
    payload["cases"][0]["candidate"][field] = value

    with pytest.raises(ValidationError):
        validate_decision_suite(payload)


def test_experiment_rows_are_canonical_and_galileo_compatible():
    payload = _suite_payload()
    original = copy.deepcopy(payload)

    rows = build_galileo_experiment_rows(payload)

    assert payload == original
    assert [json.loads(row["input"])["case_id"] for row in rows] == [
        "case-a",
        "case-b",
    ]
    assert all(set(row) == {"input", "ground_truth", "metadata"} for row in rows)
    assert all(isinstance(row["input"], str) for row in rows)
    assert all(isinstance(row["ground_truth"], str) for row in rows)
    assert all(
        isinstance(key, str) and isinstance(value, str)
        for row in rows
        for key, value in row["metadata"].items()
    )

    first_input = json.loads(rows[0]["input"])
    assert first_input["candidate"]["component_type"] == "model"
    assert first_input["candidates"] == [first_input["candidate"]]
    assert first_input["approved_evidence"] == []
    assert "raw_content" not in first_input["candidate"]
    first_truth = json.loads(rows[0]["ground_truth"])
    assert first_truth["expected_action"]["action"] == "keep"
    assert first_truth["expected_actions"]["case-a"] == first_truth["expected_action"]
    assert first_truth["expected_components"][0]["stable_case_id"] == "case-a"
    assert first_truth["expected_discovered_components"] == []
    assert first_truth["expected_relationships"] == []
    assert first_truth["expected_risks"] == []
    assert rows[0]["metadata"]["aibom.schema_version"] == (
        DECISION_SUITE_SCHEMA_VERSION
    )
    assert rows[0]["metadata"]["aibom.suite.revision"] == "2"


@pytest.mark.parametrize(
    ("approved_fixture", "environment_value"),
    [
        (False, None),
        (True, None),
        (False, "true"),
        (True, "1"),
        (True, "yes"),
        (1, "true"),
    ],
)
def test_evidence_rows_require_both_explicit_approval_gates(
    monkeypatch, approved_fixture, environment_value
):
    if environment_value is None:
        monkeypatch.delenv(FULL_CONTENT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(FULL_CONTENT_ENV_VAR, environment_value)

    with pytest.raises(FullContentLoggingDenied):
        build_galileo_experiment_rows(
            _suite_payload_with_evidence(),
            approved_fixture=approved_fixture,
        )


def test_approved_evidence_is_included_only_in_experiment_input(monkeypatch):
    payload = _suite_payload_with_evidence()
    original = copy.deepcopy(payload)
    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, " TRUE ")

    rows = build_galileo_experiment_rows(payload, approved_fixture=True)

    assert payload == original
    row = next(
        item for item in rows if json.loads(item["input"])["case_id"] == "case-b"
    )
    row_input = json.loads(row["input"])
    assert row_input["approved_evidence"] == [
        {
            "content": "client = build_model_client()\nclient.invoke(request)\n",
            "end_line": 20,
            "evidence_kind": "source_code",
            "metadata": {
                "language": "python",
                "review_ticket": "EXAMPLE-001",
            },
            "source_path": "src/client.py",
            "start_line": 16,
        }
    ]
    assert "approved_evidence" not in json.loads(row["ground_truth"])


def test_approved_evidence_content_is_opaque_after_explicit_approval(monkeypatch):
    payload = _suite_payload_with_evidence()
    approved_content = "api_key = 'synthetic-review-fixture-not-a-credential'\n"
    payload["cases"][0]["approved_evidence"][0]["content"] = approved_content
    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")

    rows = build_galileo_experiment_rows(payload, approved_fixture=True)

    case_b = next(
        json.loads(row["input"])
        for row in rows
        if json.loads(row["input"])["case_id"] == "case-b"
    )
    assert case_b["approved_evidence"][0]["content"] == approved_content


def test_experiment_row_output_is_independent_of_case_and_mapping_order():
    first_payload = _suite_payload()
    second_payload = copy.deepcopy(first_payload)
    second_payload["cases"].reverse()
    second_payload["metadata"] = {
        "suite_name": "example-regression",
        "revision": 2,
    }
    for case in second_payload["cases"]:
        case["metadata"] = dict(reversed(list(case["metadata"].items())))

    assert build_galileo_experiment_rows(first_payload) == (
        build_galileo_experiment_rows(second_payload)
    )


def test_row_builder_performs_no_sdk_import_or_network_call(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("galileo", "langchain")):
            raise AssertionError(f"unexpected optional import: {name}")
        return real_import(name, *args, **kwargs)

    def forbidden_network(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)
    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")

    rows = build_galileo_experiment_rows(
        _suite_payload_with_evidence(),
        approved_fixture=True,
    )

    assert len(rows) == 2
    assert any(json.loads(row["input"])["approved_evidence"] for row in rows)


def test_custom_function_output_is_projected_to_canonical_safe_contract():
    dataset_input = json.loads(
        build_galileo_experiment_rows(_batch_suite_payload())[0]["input"]
    )
    canary = "RAW-PROMPT-AND-TOOL-OUTPUT-CANARY"
    raw_output = {
        "components": [
            {
                "component_type": "agent",
                "name": "router",
                "file_path": "src/router.py",
                "line_number": 12,
                "metadata": {"stable_case_id": "candidate-router"},
                "description": canary,
                "prompt": canary,
            },
            {
                "component_type": "model",
                "name": "internal-model",
                "file_path": "src/models.py",
                "line_number": 4,
                "stable_case_id": "discovery-model",
                "raw_response": canary,
            },
        ],
        "relationships": [
            {
                "type": "uses_model",
                "source": "candidate-router",
                "target": "discovery-model",
                "tool_output": canary,
            }
        ],
        "risks": [
            {
                "stable_case_id": "candidate-router",
                "flag": "dynamic_tool_loading",
                "severity": "high",
                "description": canary,
            }
        ],
        "actions": {
            "candidate-router": {
                "action": "reclassify",
                "new_type": "agent",
                "reason": canary,
            },
            "candidate-obsolete": "remove",
            "discovery-model": "discover",
        },
        "raw_output": canary,
    }

    serialized = sanitize_galileo_decision_output(
        raw_output,
        dataset_input=dataset_input,
    )
    output = json.loads(serialized)

    assert canary not in serialized
    assert output == {
        "actions": {
            "candidate-obsolete": {
                "action": "remove",
                "metadata": {},
                "reason_codes": [],
                "target_type": None,
            },
            "candidate-router": {
                "action": "reclassify",
                "metadata": {},
                "reason_codes": [],
                "target_type": "agent",
            },
            "discovery-model": {
                "action": "discover",
                "metadata": {},
                "reason_codes": [],
                "target_type": None,
            },
        },
        "final_components": [
            {
                "component_type": "agent",
                "instance_id": None,
                "line_number": 12,
                "metadata": {},
                "name": "router",
                "repository": "org-repo",
                "source_path": "src/router.py",
                "stable_case_id": "candidate-router",
            },
            {
                "component_type": "model",
                "instance_id": None,
                "line_number": 4,
                "metadata": {},
                "name": "internal-model",
                "repository": "org-repo",
                "source_path": "src/models.py",
                "stable_case_id": "discovery-model",
            },
        ],
        "relationships": [
            {
                "predicted_present": True,
                "relationship_type": "uses_model",
                "source_id": "candidate-router",
                "target_id": "discovery-model",
            }
        ],
        "risk_flags": [
            {
                "case_id": "candidate-router",
                "predicted_present": True,
                "risk_type": "dynamic_tool_loading",
                "severity": "high",
            }
        ],
        "schema_valid": True,
        "schema_version": GALILEO_DECISION_OUTPUT_SCHEMA_VERSION,
    }


def test_native_pipeline_output_is_contextualized_without_leaking_runtime_paths():
    row = build_galileo_experiment_rows(_batch_suite_payload())[0]
    raw_output = {
        "components": [
            {
                "component_type": "agent",
                "name": "router",
                "file_path": "/private/work/repo/src/router.py",
                "line_number": 12,
                "instance_id": "router_/private/work/repo/src/router.py_12",
            },
            {
                "component_type": "model",
                "name": "internal-model",
                "file_path": "/private/work/repo/src/models.py",
                "line_number": 4,
                "instance_id": "model_/private/work/repo/src/models.py_4",
            },
        ],
        "relationships": [
            {
                "relationship_type": "uses_model",
                "source_instance_id": "router_/private/work/repo/src/router.py_12",
                "target_instance_id": "model_/private/work/repo/src/models.py_4",
            }
        ],
        "agentic_risk_flags": [
            {
                "flag": "dynamic_tool_loading",
                "severity": "high",
                "file_path": "/private/work/repo/src/router.py",
                "line_number": 12,
            }
        ],
    }

    serialized = sanitize_galileo_decision_output(
        raw_output,
        dataset_input=row["input"],
        dataset_ground_truth=row["ground_truth"],
    )
    output = json.loads(serialized)

    assert "/private/" not in serialized
    assert [item["stable_case_id"] for item in output["final_components"]] == [
        "candidate-router",
        "discovery-model",
    ]
    assert output["relationships"][0]["source_id"] == "candidate-router"
    assert output["relationships"][0]["target_id"] == "discovery-model"
    assert output["risk_flags"][0]["case_id"] == "candidate-router"

    trace = SimpleNamespace(
        dataset_input=row["input"],
        dataset_output=row["ground_truth"],
        output=serialized,
    )
    scores = {
        metric.name: metric.scorer_fn(trace)
        for metric in build_galileo_decision_metrics()
    }
    assert scores["aibom.components.f1"] == 1.0
    assert scores["aibom.relationships.f1"] == 1.0
    assert scores["aibom.risks.f1"] == 1.0
    assert scores["aibom.action_accuracy"] == 1.0


def test_explicit_stable_id_cannot_repair_a_wrong_component_location():
    row = build_galileo_experiment_rows(_batch_suite_payload())[0]
    raw_output = {
        "final_components": [
            {
                "component_type": "agent",
                "name": "router",
                "repository": "org-repo",
                "source_path": "totally/wrong.py",
                "line_number": 999,
                "stable_case_id": "candidate-router",
            },
            {
                "component_type": "model",
                "name": "internal-model",
                "repository": "org-repo",
                "source_path": "src/models.py",
                "line_number": 4,
                "stable_case_id": "discovery-model",
            },
        ],
        "relationships": [
            {
                "relationship_type": "uses_model",
                "source_id": "candidate-router",
                "target_id": "discovery-model",
            }
        ],
        "risk_flags": [
            {
                "case_id": "candidate-router",
                "risk_type": "dynamic_tool_loading",
                "severity": "high",
            }
        ],
    }

    serialized = sanitize_galileo_decision_output(
        raw_output,
        dataset_input=row["input"],
        dataset_ground_truth=row["ground_truth"],
    )
    output = json.loads(serialized)

    assert output["schema_valid"] is True
    assert output["final_components"][0]["source_path"] == "totally/wrong.py"
    assert output["final_components"][0]["line_number"] == 999

    trace = SimpleNamespace(
        dataset_input=row["input"],
        dataset_output=row["ground_truth"],
        output=serialized,
    )
    scores = {
        metric.name: metric.scorer_fn(trace)
        for metric in build_galileo_decision_metrics()
    }
    assert scores["aibom.components.precision"] == 0.5
    assert scores["aibom.components.recall"] == 0.5
    assert scores["aibom.components.f1"] == 0.5


def test_runtime_instance_id_cannot_repair_a_wrong_component_location():
    payload = _batch_suite_payload()
    payload["cases"][0]["candidates"][0]["instance_id"] = "runtime-router"
    row = build_galileo_experiment_rows(payload)[0]
    raw_output = {
        "final_components": [
            {
                "component_type": "agent",
                "name": "router",
                "repository": "org-repo",
                "source_path": "totally/wrong.py",
                "line_number": 999,
                "instance_id": "runtime-router",
            },
            {
                "component_type": "model",
                "name": "internal-model",
                "repository": "org-repo",
                "source_path": "src/models.py",
                "line_number": 4,
                "stable_case_id": "discovery-model",
            },
        ]
    }

    serialized = sanitize_galileo_decision_output(
        raw_output,
        dataset_input=row["input"],
        dataset_ground_truth=row["ground_truth"],
    )
    output = json.loads(serialized)

    assert output["schema_valid"] is True
    assert output["final_components"][0]["source_path"] == "totally/wrong.py"
    assert output["final_components"][0]["line_number"] == 999
    assert output["final_components"][0]["stable_case_id"] is None

    trace = SimpleNamespace(
        dataset_input=row["input"],
        dataset_output=row["ground_truth"],
        output=serialized,
    )
    scores = {
        metric.name: metric.scorer_fn(trace)
        for metric in build_galileo_decision_metrics()
    }
    assert scores["aibom.components.precision"] == 0.5
    assert scores["aibom.components.recall"] == 0.5
    assert scores["aibom.components.f1"] == 0.5


def test_unknown_explicit_action_id_makes_output_schema_invalid() -> None:
    row = build_galileo_experiment_rows(_batch_suite_payload())[0]

    serialized = sanitize_galileo_decision_output(
        {
            "final_components": [],
            "actions": {
                "candidate-router": "remove",
                "candidate-obsolete": "remove",
                "phantom-entity": "keep",
            },
        },
        dataset_input=row["input"],
    )
    output = json.loads(serialized)

    assert output["schema_valid"] is False
    assert output["actions"] is None


def test_schema_invalid_output_retains_only_sanitized_execution_outcome():
    serialized = sanitize_galileo_decision_output(
        {
            "schema_valid": False,
            "execution_outcome": {
                "status": "parse_error",
                "schema_valid": False,
                "abstained": True,
                "retry_count": 1,
            },
            "raw_response": "PRIVATE-SOURCE-CANARY",
        }
    )
    output = json.loads(serialized)

    assert "PRIVATE-SOURCE-CANARY" not in serialized
    assert output["schema_valid"] is False
    assert output["execution_outcome"] == {
        "abstained": True,
        "retry_count": 1,
        "schema_valid": False,
        "status": "parse_error",
    }
    assert output["final_components"] == []


@pytest.mark.parametrize(
    "raw_output",
    [
        "raw source and prompt",
        {"raw_output": "private"},
        {"components": "not-a-list"},
        {"components": [], "final_components": []},
        {"components": [], "schema_version": "unknown"},
    ],
)
def test_invalid_application_output_becomes_content_free_failure_envelope(raw_output):
    serialized = sanitize_galileo_decision_output(raw_output)

    assert serialized == json.dumps(
        {
            "actions": None,
            "final_components": [],
            "relationships": [],
            "risk_flags": [],
            "schema_valid": False,
            "schema_version": GALILEO_DECISION_OUTPUT_SCHEMA_VERSION,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    assert "private" not in serialized
    assert "prompt" not in serialized


def _perfect_batch_trace(
    payload: dict[str, Any] | None = None,
) -> SimpleNamespace:
    row = build_galileo_experiment_rows(payload or _batch_suite_payload())[0]
    output = sanitize_galileo_decision_output(
        {
            "final_components": [
                {
                    "component_type": "agent",
                    "name": "router",
                    "source_path": "src/router.py",
                    "line_number": 12,
                    "stable_case_id": "candidate-router",
                },
                {
                    "component_type": "model",
                    "name": "internal-model",
                    "source_path": "src/models.py",
                    "line_number": 4,
                    "stable_case_id": "discovery-model",
                },
            ],
            "relationships": [
                {
                    "relationship_type": "uses_model",
                    "source_case_id": "candidate-router",
                    "target_case_id": "discovery-model",
                }
            ],
            "risk_flags": [
                {
                    "case_id": "candidate-router",
                    "risk_type": "dynamic_tool_loading",
                    "severity": "high",
                }
            ],
            "actions": {
                "candidate-router": {
                    "action": "reclassify",
                    "target_type": "agent",
                },
                "candidate-obsolete": "remove",
                "discovery-model": "discover",
            },
        },
        dataset_input=row["input"],
    )
    return SimpleNamespace(
        dataset_input=row["input"],
        dataset_output=row["ground_truth"],
        output=output,
    )


def test_primary_local_metrics_score_exact_batch_entities_with_galileo_2_4():
    galileo = pytest.importorskip("galileo")
    metrics = build_galileo_decision_metrics()

    assert len(metrics) == 29
    assert len({metric.name for metric in metrics}) == 29
    assert all(metric.scorable_types == [galileo.StepType.trace] for metric in metrics)
    trace = _perfect_batch_trace()
    scores = {metric.name: metric.scorer_fn(trace) for metric in metrics}

    assert scores["aibom.components.precision"] == 1.0
    assert scores["aibom.components.recall"] == 1.0
    assert scores["aibom.components.f1"] == 1.0
    assert scores["aibom.discoveries.precision"] == 1.0
    assert scores["aibom.discoveries.recall"] == 1.0
    assert scores["aibom.discoveries.f1"] == 1.0
    assert scores["aibom.net_recall_lift"] == 1.0
    assert scores["aibom.over_prune_rate"] == 0.0
    assert scores["aibom.relationships.f1"] == 1.0
    assert scores["aibom.risks.f1"] == 1.0
    assert scores["aibom.relationship_recall_lift"] is None
    assert scores["aibom.action_accuracy"] == 1.0
    assert scores["aibom.action_macro_f1"] == 1.0
    assert scores["aibom.decision_coverage"] == 1.0
    assert scores["aibom.reclassification_accuracy"] == 1.0
    assert scores["aibom.schema_validity"] == 1.0


def test_local_metrics_score_structured_execution_outcomes():
    pytest.importorskip("galileo")
    payload = _batch_suite_payload()
    expected_outcome = {
        "status": "success",
        "schema_valid": True,
        "abstained": False,
        "degraded_candidate_count": 0,
        "retry_count": 1,
        "fallback_count": 0,
        "cache_hit": False,
        "tool_error_count": 0,
        "guard_denial_count": 0,
    }
    payload["cases"][0]["expected_execution_outcome"] = expected_outcome
    trace = _perfect_batch_trace(payload)
    canonical_output = json.loads(trace.output)
    canonical_output["execution_outcome"] = expected_outcome
    trace.output = json.dumps(canonical_output)

    scores = {
        metric.name: metric.scorer_fn(trace)
        for metric in build_galileo_decision_metrics()
    }

    for metric_name in (
        "aibom.execution.status_accuracy",
        "aibom.execution.schema_validity_accuracy",
        "aibom.execution.abstention_accuracy",
        "aibom.execution.degraded_count_accuracy",
        "aibom.execution.retry_count_accuracy",
        "aibom.execution.fallback_count_accuracy",
        "aibom.execution.cache_hit_accuracy",
        "aibom.execution.tool_error_count_accuracy",
        "aibom.execution.guard_denial_count_accuracy",
    ):
        assert scores[metric_name] == 1.0

    canonical_output["execution_outcome"]["retry_count"] = 2
    trace.output = json.dumps(canonical_output)
    retry_metric = next(
        metric
        for metric in build_galileo_decision_metrics()
        if metric.name == "aibom.execution.retry_count_accuracy"
    )
    assert retry_metric.scorer_fn(trace) == 0.0


def test_expected_schema_failure_is_scored_without_entity_content():
    pytest.importorskip("galileo")
    payload = _batch_suite_payload()
    payload["cases"][0]["expected_execution_outcome"] = {
        "status": "parse_error",
        "schema_valid": False,
        "abstained": True,
    }
    row = build_galileo_experiment_rows(payload)[0]
    trace = SimpleNamespace(
        dataset_input=row["input"],
        dataset_output=row["ground_truth"],
        output=sanitize_galileo_decision_output(
            {
                "schema_valid": False,
                "execution_outcome": {
                    "status": "parse_error",
                    "schema_valid": False,
                    "abstained": True,
                },
            }
        ),
    )

    scores = {
        metric.name: metric.scorer_fn(trace)
        for metric in build_galileo_decision_metrics()
    }
    assert scores["aibom.schema_validity"] == 0.0
    assert scores["aibom.execution.status_accuracy"] == 1.0
    assert scores["aibom.execution.schema_validity_accuracy"] == 1.0
    assert scores["aibom.execution.abstention_accuracy"] == 1.0
    assert scores["aibom.components.f1"] is None


def test_relationship_recall_lift_uses_explicit_deterministic_edge_labels():
    pytest.importorskip("galileo")
    payload = _batch_suite_payload()
    payload["cases"][0]["deterministic_relationships"] = []
    trace = _perfect_batch_trace(payload)

    relationship_lift = next(
        metric
        for metric in build_galileo_decision_metrics()
        if metric.name == "aibom.relationship_recall_lift"
    )

    assert relationship_lift.scorer_fn(trace) == 1.0


def test_local_metrics_return_schema_failure_without_scoring_invalid_output():
    pytest.importorskip("galileo")
    metrics = build_galileo_decision_metrics()
    trace = _perfect_batch_trace()
    trace.output = "raw private source"

    scores = {metric.name: metric.scorer_fn(trace) for metric in metrics}

    assert scores["aibom.schema_validity"] == 0.0
    assert all(
        score is None
        for name, score in scores.items()
        if name != "aibom.schema_validity"
    )


def test_local_metrics_exclude_true_empty_entity_sets_from_row_averages():
    pytest.importorskip("galileo")
    payload = {
        "schema_version": DECISION_SUITE_SCHEMA_VERSION,
        "cases": [
            {
                "case_id": "negative-case",
                "candidate": {
                    "component_type": "tool",
                    "name": "not-ai",
                    "repository": "negative-repo",
                    "source_path": "src/plain.py",
                    "line_number": 3,
                    "stable_case_id": "negative-case",
                },
                "expected_action": "remove",
                "expected_relationships": [],
                "expected_risks": [],
            }
        ],
    }
    row = build_galileo_experiment_rows(payload)[0]
    trace = SimpleNamespace(
        dataset_input=row["input"],
        dataset_output=row["ground_truth"],
        output=sanitize_galileo_decision_output(
            {
                "components": [],
                "relationships": [],
                "risks": [],
                "actions": {"negative-case": "remove"},
            },
            dataset_input=row["input"],
        ),
    )

    scores = {
        metric.name: metric.scorer_fn(trace)
        for metric in build_galileo_decision_metrics()
    }

    for prefix in (
        "aibom.components",
        "aibom.discoveries",
        "aibom.relationships",
        "aibom.risks",
    ):
        assert scores[f"{prefix}.precision"] is None
        assert scores[f"{prefix}.recall"] is None
        assert scores[f"{prefix}.f1"] is None
    assert scores["aibom.action_accuracy"] == 1.0
    assert scores["aibom.schema_validity"] == 1.0


@pytest.mark.parametrize(
    ("approved_fixture", "environment_value"),
    [
        (False, None),
        (True, None),
        (False, "true"),
        (True, "1"),
        (1, "true"),
    ],
)
def test_evidence_grounding_metric_checks_approval_before_sdk_import(
    monkeypatch, approved_fixture, environment_value
):
    if environment_value is None:
        monkeypatch.delenv(FULL_CONTENT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(FULL_CONTENT_ENV_VAR, environment_value)

    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "galileo":
            imported.append(name)
            raise AssertionError("approval must precede the optional SDK import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(FullContentLoggingDenied):
        build_aibom_evidence_grounding_metric(
            approved_fixture=approved_fixture,
            judge_model="private-judge",
        )
    assert imported == []


def test_evidence_grounding_metric_uses_galileo_2_4_constructor(monkeypatch):
    captured = {}

    class FakeLlmMetric:
        def __init__(self, name, **kwargs):
            captured["name"] = name
            captured.update(kwargs)

        def create(self):
            raise AssertionError("the construction helper must not persist a metric")

    fake_module = SimpleNamespace(
        LlmMetric=FakeLlmMetric,
        StepType=SimpleNamespace(trace="trace"),
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo":
            return fake_module
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    metric = build_aibom_evidence_grounding_metric(
        approved_fixture=True,
        judge_model="private-grounding-judge",
        judges=2,
    )

    assert isinstance(metric, FakeLlmMetric)
    assert captured == {
        "cot_enabled": False,
        "description": (
            "Diagnostic judge for evidence support of AIBOM component, action, "
            "relationship, and risk decisions."
        ),
        "ground_truth": True,
        "judges": 2,
        "model": "private-grounding-judge",
        "name": AIBOM_EVIDENCE_GROUNDING_METRIC_NAME,
        "node_level": "trace",
        "output_type": "percentage",
        "prompt": AIBOM_EVIDENCE_GROUNDING_PROMPT,
        "tags": ["aibom", "diagnostic", "evidence-grounding"],
    }
    assert "approved_evidence" in captured["prompt"]
    assert "{input}" in captured["prompt"]
    assert "{output}" in captured["prompt"]
    assert "{reference_output}" in captured["prompt"]
    assert "0.0 to 1.0" in captured["prompt"]


def test_evidence_grounding_metric_constructs_without_network_on_galileo_2_4(
    monkeypatch,
):
    galileo = pytest.importorskip("galileo")

    def forbidden_network(*args, **kwargs):
        raise AssertionError("metric construction must not make a network call")

    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")
    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)

    metric = build_aibom_evidence_grounding_metric(
        approved_fixture=True,
        judge_model="private-grounding-judge",
    )

    assert isinstance(metric, galileo.LlmMetric)
    assert metric.id is None
    assert metric.model == "private-grounding-judge"
    assert metric.node_level == galileo.StepType.trace
    assert metric.ground_truth is True
    assert metric.output_type.value == "percentage"


def test_evidence_grounding_metric_requires_explicit_judge_model(monkeypatch):
    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")

    with pytest.raises(ValueError, match="explicit non-empty model"):
        build_aibom_evidence_grounding_metric(
            approved_fixture=True,
            judge_model="  ",
        )


@pytest.mark.parametrize(
    ("approved_fixture", "environment_value"),
    [
        (False, None),
        (True, None),
        (False, "true"),
        (True, "1"),
        (True, "yes"),
        (1, "true"),
    ],
)
def test_callback_is_denied_unless_fixture_and_full_content_gates_pass(
    monkeypatch, approved_fixture, environment_value
):
    # Isolate the full-content/full-trajectory gates exercised here from the
    # independent exact-identity approval.
    monkeypatch.setenv(EXACT_IDENTITIES_ENV_VAR, "true")
    if environment_value is None:
        monkeypatch.delenv(FULL_CONTENT_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(FULL_CONTENT_ENV_VAR, environment_value)

    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("galileo", "langchain")):
            imported.append(name)
            raise AssertionError("approval must be checked before optional imports")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(FullContentLoggingDenied):
        create_galileo_async_callback(approved_fixture=approved_fixture)
    assert imported == []


def test_callback_requires_separate_full_trajectory_approval_before_import(
    monkeypatch,
):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("galileo", "langchain")):
            imported.append(name)
            raise AssertionError("trajectory approval must precede optional imports")
        return real_import(name, *args, **kwargs)

    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")
    monkeypatch.setenv(EXACT_IDENTITIES_ENV_VAR, "true")
    monkeypatch.delenv(FULL_TRAJECTORY_ENV_VAR, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(FullContentLoggingDenied, match=FULL_TRAJECTORY_ENV_VAR):
        create_galileo_async_callback(approved_fixture=True)
    assert imported == []


@pytest.mark.parametrize(
    "unsafe_flags",
    [
        {"start_new_trace": False},
        {"flush_on_chain_end": False},
        {"start_new_trace": False, "flush_on_chain_end": False},
    ],
)
def test_callback_forces_complete_trace_ingestion_before_import(
    monkeypatch, unsafe_flags
):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("galileo", "langchain")):
            imported.append(name)
            raise AssertionError("unsafe callback flags must be rejected before import")
        return real_import(name, *args, **kwargs)

    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(
        HostedGalileoDestinationRequired,
        match="start_new_trace=True.*flush_on_chain_end=True",
    ):
        create_galileo_async_callback(approved_fixture=True, **unsafe_flags)
    assert imported == []


def test_approved_callback_uses_the_galileo_2_4_signature(monkeypatch):
    captured = {}

    class FakeGalileoLogger:
        def __init__(self, **kwargs):
            captured["logger_kwargs"] = kwargs
            self.project_id = kwargs["project_id"]
            self.log_stream_id = kwargs["log_stream_id"]

    class FakeGalileoAsyncCallback:
        def __init__(
            self,
            galileo_logger=None,
            start_new_trace=True,
            flush_on_chain_end=True,
            ingestion_hook=None,
        ):
            captured.update(
                galileo_logger=galileo_logger,
                start_new_trace=start_new_trace,
                flush_on_chain_end=flush_on_chain_end,
                ingestion_hook=ingestion_hook,
            )

    fake_galileo_module = SimpleNamespace(GalileoLogger=FakeGalileoLogger)
    fake_callback_module = SimpleNamespace(
        GalileoAsyncCallback=FakeGalileoAsyncCallback
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo":
            return fake_galileo_module
        if name == "galileo.handlers.langchain":
            return fake_callback_module
        if name == "galileo.config":
            return SimpleNamespace(GalileoPythonConfig=SimpleNamespace(_instance=None))
        return real_import(name, *args, **kwargs)

    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    callback = create_galileo_async_callback(approved_fixture=True)

    assert isinstance(callback, FakeGalileoAsyncCallback)
    assert captured["logger_kwargs"] == {
        "project_id": _EVALUATION_PROJECT_ID,
        "log_stream_id": _EVALUATION_LOG_STREAM_ID,
        "mode": "batch",
    }
    logger = captured["galileo_logger"]
    assert isinstance(logger, FakeGalileoLogger)
    assert captured["start_new_trace"] is True
    assert captured["flush_on_chain_end"] is True
    assert captured["ingestion_hook"] is None


def test_approved_callback_accepts_explicit_hosted_cloud_egress(monkeypatch):
    captured = {}

    class FakeGalileoLogger:
        def __init__(self, *, project_id, log_stream_id, mode):
            self.project_id = project_id
            self.log_stream_id = log_stream_id
            captured["logger_mode"] = mode

    class FakeGalileoAsyncCallback:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_callback_module = SimpleNamespace(
        GalileoAsyncCallback=FakeGalileoAsyncCallback
    )
    fake_galileo_module = SimpleNamespace(GalileoLogger=FakeGalileoLogger)
    fake_config_module = SimpleNamespace(
        GalileoPythonConfig=SimpleNamespace(_instance=None)
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo":
            return fake_galileo_module
        if name == "galileo.handlers.langchain":
            return fake_callback_module
        if name == "galileo.config":
            return fake_config_module
        return real_import(name, *args, **kwargs)

    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    callback = create_galileo_async_callback(approved_fixture=True)

    assert isinstance(callback, FakeGalileoAsyncCallback)
    assert captured["logger_mode"] == "batch"
    assert captured["galileo_logger"].project_id == _EVALUATION_PROJECT_ID
    assert captured["galileo_logger"].log_stream_id == _EVALUATION_LOG_STREAM_ID
    assert captured["start_new_trace"] is True
    assert captured["flush_on_chain_end"] is True
    assert captured["ingestion_hook"] is None


def _install_fake_galileo_for_session(monkeypatch, *, sessions_created):
    """Wire a fake Galileo SDK that records session_id per built logger.

    ``sessions_created`` accumulates one entry per ``start_session`` call so a
    test can assert the session is created exactly once and reused thereafter.
    """

    class FakeGalileoLogger:
        def __init__(self, **kwargs):
            self.project_id = kwargs["project_id"]
            self.log_stream_id = kwargs["log_stream_id"]
            self.session_id = None

        def start_session(self, *, name, external_id=None, metadata=None):
            session_id = f"33333333-3333-4333-8333-00000000000{len(sessions_created)}"
            sessions_created.append(
                {"name": name, "external_id": external_id, "metadata": metadata}
            )
            self.session_id = session_id
            return session_id

        def set_session(self, session_id):
            self.session_id = session_id

    built_loggers = []

    class FakeGalileoAsyncCallback:
        def __init__(self, galileo_logger=None, **kwargs):
            self.galileo_logger = galileo_logger
            built_loggers.append(galileo_logger)

    fake_galileo_module = SimpleNamespace(GalileoLogger=FakeGalileoLogger)
    fake_callback_module = SimpleNamespace(
        GalileoAsyncCallback=FakeGalileoAsyncCallback
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo":
            return fake_galileo_module
        if name == "galileo.handlers.langchain":
            return fake_callback_module
        if name == "galileo.config":
            return SimpleNamespace(GalileoPythonConfig=SimpleNamespace(_instance=None))
        return real_import(name, *args, **kwargs)

    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    return built_loggers


def test_full_trajectory_factory_groups_callbacks_in_one_session(monkeypatch):
    sessions_created: list[dict[str, Any]] = []
    built_loggers = _install_fake_galileo_for_session(
        monkeypatch, sessions_created=sessions_created
    )

    factory = create_full_trajectory_callback_factory(
        session_name="aibom-agentic-scan-2026-07-17T03:34:24Z",
        session_external_id="7d335cf2-c011-4a7b-99b7-c65e833b58e5",
    )

    # Three separate live invocations, as a multi-batch scan would produce.
    factory()
    factory()
    factory()

    # The session is created exactly once and every logger shares its id.
    assert len(sessions_created) == 1
    assert sessions_created[0]["external_id"] == "7d335cf2-c011-4a7b-99b7-c65e833b58e5"
    assert sessions_created[0]["name"] == "aibom-agentic-scan-2026-07-17T03:34:24Z"
    assert len(built_loggers) == 3
    session_ids = {logger.session_id for logger in built_loggers}
    assert len(session_ids) == 1
    assert None not in session_ids


def test_full_trajectory_factory_is_concurrency_safe(monkeypatch):
    import threading

    sessions_created: list[dict[str, Any]] = []
    built_loggers = _install_fake_galileo_for_session(
        monkeypatch, sessions_created=sessions_created
    )

    factory = create_full_trajectory_callback_factory(
        session_name="aibom-agentic-scan-concurrent",
        session_external_id="7d335cf2-c011-4a7b-99b7-c65e833b58e5",
    )

    start = threading.Barrier(8)
    errors: list[Exception] = []

    def _worker():
        start.wait()
        try:
            factory()
        except Exception as exc:  # pragma: no cover - surfaced via assert
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    # Even with concurrent first calls, exactly one session is created and all
    # loggers converge on the same id.
    assert len(sessions_created) == 1
    assert len(built_loggers) == 8
    assert len({logger.session_id for logger in built_loggers}) == 1


def test_full_trajectory_factory_recovers_from_bounded_session_timeout(monkeypatch):
    # A create that exceeds the bounded setup budget must not pin later loggers
    # to an unusable session: the session stays unset and the next invocation
    # retries the create.
    import time

    sessions_created: list[dict[str, Any]] = []

    class SlowThenFastLogger:
        calls = {"n": 0}

        def __init__(self, **kwargs):
            self.project_id = kwargs["project_id"]
            self.log_stream_id = kwargs["log_stream_id"]
            self.session_id = None

        def start_session(self, *, name, external_id=None, metadata=None):
            SlowThenFastLogger.calls["n"] += 1
            if SlowThenFastLogger.calls["n"] == 1:
                # First create blows the (test-shrunk) budget → treated as timeout.
                time.sleep(0.3)
            session_id = "44444444-4444-4444-8444-444444444444"
            sessions_created.append({"external_id": external_id})
            self.session_id = session_id
            return session_id

        def set_session(self, session_id):
            self.session_id = session_id

    built = []

    class FakeGalileoAsyncCallback:
        def __init__(self, galileo_logger=None, **kwargs):
            built.append(galileo_logger)

    fake_galileo_module = SimpleNamespace(GalileoLogger=SlowThenFastLogger)
    fake_callback_module = SimpleNamespace(
        GalileoAsyncCallback=FakeGalileoAsyncCallback
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo":
            return fake_galileo_module
        if name == "galileo.handlers.langchain":
            return fake_callback_module
        if name == "galileo.config":
            return SimpleNamespace(GalileoPythonConfig=SimpleNamespace(_instance=None))
        return real_import(name, *args, **kwargs)

    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    # Shrink the bounded-setup budget so the 0.3s first create is a timeout.
    monkeypatch.setenv("AIBOM_GALILEO_SETUP_BUDGET_S", "0.1")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    factory = create_full_trajectory_callback_factory(
        session_name="aibom-agentic-scan-timeout",
        session_external_id="7d335cf2-c011-4a7b-99b7-c65e833b58e5",
    )

    factory()  # first create times out → session left unset
    factory()  # retries the create → succeeds and binds

    # Two create attempts were made (the first timed out, the second bound).
    assert SlowThenFastLogger.calls["n"] == 2
    # The second logger converged on a real session id.
    assert built[1].session_id == "44444444-4444-4444-8444-444444444444"


def test_full_trajectory_factory_enforces_gates(monkeypatch):
    # No approval gates set: constructing a callback must still fail closed even
    # though the factory itself performs no I/O until called.
    factory = create_full_trajectory_callback_factory(
        session_name="aibom-agentic-scan-denied",
        session_external_id="7d335cf2-c011-4a7b-99b7-c65e833b58e5",
    )
    with pytest.raises(FullContentLoggingDenied):
        factory()


@pytest.mark.parametrize("unsafe_argument", ["logger", "hook"])
def test_full_content_callback_rejects_unverifiable_destinations(
    monkeypatch, unsafe_argument
):
    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    kwargs = (
        {"galileo_logger": object()}
        if unsafe_argument == "logger"
        else {"ingestion_hook": lambda _request: None}
    )

    with pytest.raises(HostedGalileoDestinationRequired, match="cannot be verified"):
        create_galileo_async_callback(approved_fixture=True, **kwargs)


def test_approved_callback_reports_missing_optional_integration(monkeypatch):
    real_import = builtins.__import__

    def missing_import(name, *args, **kwargs):
        if name == "galileo.handlers.langchain":
            raise ModuleNotFoundError("No module named 'galileo'")
        return real_import(name, *args, **kwargs)

    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", missing_import)

    with pytest.raises(GalileoIntegrationUnavailable) as exc_info:
        create_galileo_async_callback(approved_fixture=True)

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize(
    ("approved_fixture", "identity_approval"),
    [
        (False, None),
        (True, None),
        (False, "true"),
        (True, "false"),
        (1, "true"),
    ],
)
def test_custom_experiment_requires_exact_identity_approval_before_import(
    monkeypatch, approved_fixture, identity_approval
):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("galileo"):
            imported.append(name)
            raise AssertionError("exact-identity approval must precede SDK import")
        return real_import(name, *args, **kwargs)

    if identity_approval is None:
        monkeypatch.delenv(EXACT_IDENTITIES_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(EXACT_IDENTITIES_ENV_VAR, identity_approval)
    monkeypatch.setenv(EVALUATION_PROJECT_ID_ENV_VAR, _EVALUATION_PROJECT_ID)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ExactIdentityLoggingDenied, match=EXACT_IDENTITIES_ENV_VAR):
        run_galileo_custom_function_experiment(
            _suite_payload(),
            experiment_name="identity-gated",
            function=lambda row: row,
            approved_fixture=approved_fixture,
            exact_identities=True,
        )
    assert imported == []


def test_custom_experiment_rejects_caller_supplied_project_name_before_import(
    monkeypatch,
):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("galileo"):
            imported.append(name)
            raise AssertionError("project-name refusal must precede SDK import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(
        HostedGalileoDestinationRequired,
        match=EVALUATION_PROJECT_ID_ENV_VAR,
    ):
        run_galileo_custom_function_experiment(
            _suite_payload(),
            experiment_name="pinned-project-only",
            function=lambda row: row,
            project="example-project",
        )
    assert imported == []


@pytest.mark.parametrize(
    ("path", "environment_name", "environment_value"),
    [
        ("experiment", EVALUATION_PROJECT_ID_ENV_VAR, None),
        ("experiment", EVALUATION_PROJECT_ID_ENV_VAR, "example-project"),
        ("callback", EVALUATION_PROJECT_ID_ENV_VAR, None),
        ("callback", EVALUATION_PROJECT_ID_ENV_VAR, "not-a-uuid"),
        ("callback", EVALUATION_LOG_STREAM_ID_ENV_VAR, None),
        ("callback", EVALUATION_LOG_STREAM_ID_ENV_VAR, "benchmark"),
    ],
)
def test_networked_evaluation_requires_uuid_pinned_resources_before_import(
    monkeypatch, path, environment_name, environment_value
):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("galileo"):
            imported.append(name)
            raise AssertionError("resource pin validation must precede SDK import")
        return real_import(name, *args, **kwargs)

    _approve_exact_identity_evaluation(monkeypatch)
    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    if environment_value is None:
        monkeypatch.delenv(environment_name, raising=False)
    else:
        monkeypatch.setenv(environment_name, environment_value)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(HostedGalileoDestinationRequired, match=environment_name):
        if path == "callback":
            create_galileo_async_callback(approved_fixture=True)
        else:
            run_galileo_custom_function_experiment(
                _suite_payload(),
                experiment_name="resource-pinned",
                function=lambda row: row,
                approved_fixture=True,
            )
    assert imported == []


@pytest.mark.parametrize("mismatched_field", ["project_id", "log_stream_id"])
def test_callback_rejects_logger_that_does_not_retain_pinned_ids(
    monkeypatch, mismatched_field
):
    callback_created = False

    class MismatchedGalileoLogger:
        def __init__(self, *, project_id, log_stream_id, mode):
            del mode
            self.project_id = project_id
            self.log_stream_id = log_stream_id
            setattr(self, mismatched_field, "33333333-3333-4333-8333-333333333333")

    class FakeGalileoAsyncCallback:
        def __init__(self, **_kwargs):
            nonlocal callback_created
            callback_created = True

    fake_galileo_module = SimpleNamespace(GalileoLogger=MismatchedGalileoLogger)
    fake_callback_module = SimpleNamespace(
        GalileoAsyncCallback=FakeGalileoAsyncCallback
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo":
            return fake_galileo_module
        if name == "galileo.handlers.langchain":
            return fake_callback_module
        if name == "galileo.config":
            return SimpleNamespace(GalileoPythonConfig=SimpleNamespace(_instance=None))
        return real_import(name, *args, **kwargs)

    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(HostedGalileoDestinationRequired, match="did not retain"):
        create_galileo_async_callback(approved_fixture=True)
    assert callback_created is False


@pytest.mark.parametrize(
    "unsafe_kwargs",
    [
        {"experiment_name": "decision suite"},
        {"experiment_name": "../decision-suite"},
        {"experiment_name": "sk-supersecretvalue"},
        {
            "experiment_name": "decision-suite",
            "experiment_group": "private/group",
        },
        {
            "experiment_name": "decision-suite",
            "experiment_tags": {"dataset_version": "customer repo"},
        },
        {
            "experiment_name": "decision-suite",
            "experiment_tags": {"dataset_version": "sk-supersecretvalue"},
        },
    ],
)
def test_custom_experiment_rejects_unsafe_resource_labels_before_import(
    monkeypatch, unsafe_kwargs
):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("galileo"):
            imported.append(name)
            raise AssertionError("unsafe labels must be rejected before SDK import")
        return real_import(name, *args, **kwargs)

    _approve_exact_identity_evaluation(monkeypatch)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ValueError, match="bounded non-secret label"):
        run_galileo_custom_function_experiment(
            _suite_payload(),
            function=lambda row: row,
            approved_fixture=True,
            **unsafe_kwargs,
        )
    assert imported == []


def test_custom_function_experiment_uses_galileo_2_4_runner(monkeypatch):
    captured = {}

    def fake_run_experiment(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "experiment-result"

    fake_module = SimpleNamespace(run_experiment=fake_run_experiment)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo.experiments":
            return fake_module
        if name == "galileo.config":
            return SimpleNamespace(GalileoPythonConfig=SimpleNamespace(_instance=None))
        return real_import(name, *args, **kwargs)

    canary = "RAW-APPLICATION-OUTPUT"

    def application(row):
        return {
            "components": row["candidates"],
            "raw_output": canary,
        }

    _approve_exact_identity_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = run_galileo_custom_function_experiment(
        _suite_payload(),
        experiment_name=" decision-suite-v1 ",
        function=application,
        metrics=["aibom.components.f1"],
        experiment_tags={" dataset_version ": " v1 "},
        experiment_group=" decision-suite ",
        approved_fixture=True,
        exact_identities=True,
    )

    assert result == "experiment-result"
    assert captured["args"] == ("decision-suite-v1",)
    kwargs = captured["kwargs"]
    assert kwargs["project"] is None
    assert kwargs["project_id"] == _EVALUATION_PROJECT_ID
    assert kwargs["function"] is not application
    assert kwargs["function"].__wrapped__ is application
    assert kwargs["metrics"] == ["aibom.components.f1"]
    assert kwargs["experiment_group"] == "decision-suite"
    assert kwargs["experiment_tags"] == {"dataset_version": "v1"}
    assert len(kwargs["dataset"]) == 2
    assert all(
        set(row) == {"input", "ground_truth", "metadata"} for row in kwargs["dataset"]
    )
    application_output = kwargs["function"](json.loads(kwargs["dataset"][0]["input"]))
    assert json.loads(application_output)["schema_valid"] is True
    assert canary not in application_output


def test_custom_function_experiment_accepts_explicit_hosted_cloud_egress(
    monkeypatch,
):
    captured = {}

    def fake_run_experiment(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "hosted-experiment-result"

    fake_experiment_module = SimpleNamespace(run_experiment=fake_run_experiment)
    fake_config_module = SimpleNamespace(
        GalileoPythonConfig=SimpleNamespace(
            _instance=SimpleNamespace(
                console_url="https://app.galileo.ai/",
                api_url="https://api.galileo.ai/",
            )
        )
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo.experiments":
            return fake_experiment_module
        if name == "galileo.config":
            return fake_config_module
        return real_import(name, *args, **kwargs)

    _approve_exact_identity_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://APP.GALILEO.AI.:443/")
    monkeypatch.setenv("GALILEO_API_URL", "https://API.GALILEO.AI.:443/")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = run_galileo_custom_function_experiment(
        _suite_payload(),
        experiment_name="hosted-decision-suite",
        function=lambda row: {"components": row["candidates"]},
        metrics=[],
        approved_fixture=True,
    )

    assert result == "hosted-experiment-result"
    assert len(captured["args"]) == 1
    assert captured["args"][0].startswith("aibom-decision-")
    assert "hosted-decision-suite" not in captured["args"][0]
    assert captured["kwargs"]["project"] is None
    assert captured["kwargs"]["project_id"] == _EVALUATION_PROJECT_ID
    assert len(captured["kwargs"]["dataset"]) == 2


def test_custom_function_experiment_redacts_application_exceptions(monkeypatch):
    captured_output = ""

    def fake_run_experiment(*_args, **kwargs):
        nonlocal captured_output
        row = kwargs["dataset"][0]
        captured_output = kwargs["function"](json.loads(row["input"]))
        return captured_output

    fake_module = SimpleNamespace(run_experiment=fake_run_experiment)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo.experiments":
            return fake_module
        if name == "galileo.config":
            return SimpleNamespace(GalileoPythonConfig=SimpleNamespace(_instance=None))
        return real_import(name, *args, **kwargs)

    canary = "PRIVATE-SOURCE-IN-EXCEPTION"

    def failing_application(_row):
        raise RuntimeError(canary)

    _approve_exact_identity_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = run_galileo_custom_function_experiment(
        _suite_payload(),
        experiment_name="redacted-application-error",
        function=failing_application,
        metrics=[],
        approved_fixture=True,
    )

    assert result == captured_output
    assert canary not in captured_output
    assert json.loads(captured_output)["schema_valid"] is False


def test_custom_function_experiment_installs_primary_metrics_by_default(monkeypatch):
    captured = {}

    def fake_run_experiment(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "experiment-result"

    class FakeLocalMetric:
        def __init__(self, name, **kwargs):
            self.name = name
            self.__dict__.update(kwargs)

    fake_experiment_module = SimpleNamespace(run_experiment=fake_run_experiment)
    fake_galileo_module = SimpleNamespace(
        LocalMetric=FakeLocalMetric,
        StepType=SimpleNamespace(trace="trace"),
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo.experiments":
            return fake_experiment_module
        if name == "galileo":
            return fake_galileo_module
        if name == "galileo.config":
            return SimpleNamespace(GalileoPythonConfig=SimpleNamespace(_instance=None))
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _approve_exact_identity_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")

    result = run_galileo_custom_function_experiment(
        _suite_payload(),
        experiment_name="default-metrics",
        function=lambda row: {"components": row["candidates"]},
        approved_fixture=True,
    )

    assert result == "experiment-result"
    metrics = captured["kwargs"]["metrics"]
    assert len(metrics) == 29
    assert {metric.name for metric in metrics} >= {
        "aibom.schema_validity",
        "aibom.execution.status_accuracy",
        "aibom.execution.guard_denial_count_accuracy",
    }
    assert all(metric.scorable_types == ["trace"] for metric in metrics)


@pytest.mark.parametrize(
    "console_url",
    [
        None,
        "http://galileo.customer.internal",
        "https://galileo.customer.internal",
        "https://app.galileo.ai",
    ],
)
def test_networked_evaluation_paths_require_approved_hosted_console_before_import(
    monkeypatch, console_url
):
    _approve_exact_identity_evaluation(monkeypatch)
    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.delenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, raising=False)
    if console_url is None:
        monkeypatch.delenv("GALILEO_CONSOLE_URL", raising=False)
    else:
        monkeypatch.setenv("GALILEO_CONSOLE_URL", console_url)
    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("galileo"):
            imported.append(name)
            raise AssertionError("destination validation must precede SDK import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(HostedGalileoDestinationRequired):
        create_galileo_async_callback(approved_fixture=True)
    with pytest.raises(HostedGalileoDestinationRequired):
        run_galileo_custom_function_experiment(
            _suite_payload(),
            experiment_name="hosted-destination-required",
            function=lambda row: row,
            metrics=[],
            approved_fixture=True,
        )
    assert imported == []


@pytest.mark.parametrize(
    "console_url",
    [
        "https://galileo.ai",
        "https://api.galileo.ai",
        "https://customer.galileo.ai",
        "https://galileo.customer.internal",
        "https://app.galileo.ai:8443",
        "https://app.galileo.ai/project/customer",
        "https://app.galileo.ai?tenant=customer",
        "https://app.galileo.ai#fragment",
        "https://user:password@app.galileo.ai",
    ],
)
def test_hosted_evaluation_rejects_every_noncanonical_origin_before_sdk_import(
    monkeypatch, console_url
):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("galileo"):
            imported.append(name)
            raise AssertionError("destination validation must precede SDK import")
        return real_import(name, *args, **kwargs)

    _approve_exact_identity_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", console_url)
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(HostedGalileoDestinationRequired):
        run_galileo_custom_function_experiment(
            _suite_payload(),
            experiment_name="invalid-hosted-destination",
            function=lambda row: row,
            metrics=[],
            approved_fixture=True,
        )
    assert imported == []


@pytest.mark.parametrize(
    ("environment_name", "environment_value"),
    [
        ("GALILEO_API_URL", "http://api.galileo.ai"),
        ("GALILEO_API_URL", "https://attacker.example"),
        ("GALILEO_API_URL", "https://api.galileo.ai:8443"),
        ("GALILEO_API_URL", "https://api.galileo.ai/v1"),
        ("GALILEO_SSL_CONTEXT", "false"),
        ("GALILEO_SSL_CONTEXT", "f"),
        ("GALILEO_SSL_CONTEXT", "invalid"),
    ],
)
def test_hosted_evaluation_rejects_unsafe_sdk_overrides_before_import(
    monkeypatch, environment_name, environment_value
):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("galileo"):
            imported.append(name)
            raise AssertionError("SDK safety validation must precede import")
        return real_import(name, *args, **kwargs)

    _approve_exact_identity_evaluation(monkeypatch)
    _approve_full_trajectory_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setenv(FULL_CONTENT_ENV_VAR, "true")
    monkeypatch.setenv(environment_name, environment_value)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(HostedGalileoDestinationRequired):
        create_galileo_async_callback(approved_fixture=True)
    with pytest.raises(HostedGalileoDestinationRequired):
        run_galileo_custom_function_experiment(
            _suite_payload(),
            experiment_name="unsafe-sdk-override",
            function=lambda row: row,
            metrics=[],
            approved_fixture=True,
        )
    assert imported == []


@pytest.mark.parametrize(
    ("api_url", "ssl_context"),
    [
        ("https://attacker.example/", True),
        ("https://api.galileo.ai/", False),
        ("https://api.galileo.ai/", _unsafe_ssl_context()),
    ],
)
def test_evaluation_rejects_preloaded_unsafe_sdk_settings(
    monkeypatch, api_url, ssl_context
):
    runner_called = False
    mismatched_instance = SimpleNamespace(
        console_url="https://app.galileo.ai/",
        api_url=api_url,
        ssl_context=ssl_context,
    )
    fake_config = SimpleNamespace(_instance=mismatched_instance)
    fake_config_module = SimpleNamespace(GalileoPythonConfig=fake_config)

    def fake_run_experiment(*_args, **_kwargs):
        nonlocal runner_called
        runner_called = True

    fake_experiment_module = SimpleNamespace(run_experiment=fake_run_experiment)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "galileo.experiments":
            return fake_experiment_module
        if name == "galileo.config":
            return fake_config_module
        return real_import(name, *args, **kwargs)

    _approve_exact_identity_evaluation(monkeypatch)
    monkeypatch.setenv("GALILEO_CONSOLE_URL", "https://app.galileo.ai")
    monkeypatch.setenv(ALLOW_PUBLIC_CLOUD_ENV_VAR, "true")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(HostedGalileoDestinationRequired, match="restart"):
        run_galileo_custom_function_experiment(
            _suite_payload(),
            experiment_name="unsafe-sdk-singleton",
            function=lambda row: row,
            metrics=[],
            approved_fixture=True,
        )
    assert not runner_called


def test_metric_factory_reports_missing_optional_sdk(monkeypatch):
    real_import = builtins.__import__

    def missing_import(name, *args, **kwargs):
        if name == "galileo":
            raise ModuleNotFoundError("No module named 'galileo'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_import)

    with pytest.raises(GalileoIntegrationUnavailable) as exc_info:
        build_galileo_decision_metrics()

    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_custom_function_experiment_checks_content_approval_before_sdk_import(
    monkeypatch,
):
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("galileo"):
            imported.append(name)
            raise AssertionError("approval must precede SDK import")
        return real_import(name, *args, **kwargs)

    _approve_exact_identity_evaluation(monkeypatch)
    monkeypatch.delenv(FULL_CONTENT_ENV_VAR, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(FullContentLoggingDenied):
        run_galileo_custom_function_experiment(
            _suite_payload_with_evidence(),
            experiment_name="denied",
            function=lambda row: row,
            approved_fixture=True,
            exact_identities=True,
        )
    assert imported == []
