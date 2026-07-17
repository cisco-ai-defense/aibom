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

import json

import pytest

from aibom.decision_evaluation import (
    canonical_component_identity,
    canonical_relationship_identity,
    canonical_risk_identity,
    component_action_key,
    evaluate_decisions,
)
from aibom.models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    RelationshipType,
)


def _component(
    case_id: str,
    name: str,
    component_type: str,
    *,
    path: str = "src/app.py",
    line: int = 10,
    instance_id: str | None = None,
) -> dict[str, object]:
    component: dict[str, object] = {
        "name": name,
        "component_type": component_type,
        "file_path": path,
        "line_number": line,
        "metadata": {"stable_case_id": case_id},
    }
    if instance_id is not None:
        component["instance_id"] = instance_id
    return component


def test_component_identity_is_normalized_and_stable(tmp_path):
    repo_root = tmp_path / "My-Repo"
    absolute_path = repo_root / "src" / "agent.py"
    serialized = {
        "metadata": {"case_id": "candidate-7"},
        "line_number": "42",
        "file_path": str(absolute_path),
        "name": "  OpenAI   Client ",
        "component_type": "LLM_ENDPOINT",
    }
    reordered = {
        "component_type": "llm_endpoint",
        "name": "openai client",
        "file_path": "src/agent.py",
        "line_number": 42,
        "case_id": "candidate-7",
    }

    first = canonical_component_identity(serialized, repo_root=repo_root)
    second = canonical_component_identity(reordered, repo_root=repo_root)

    assert first == second
    assert first.source_path == "src/agent.py"
    assert first.repository == "my-repo"
    assert first.case_id == "candidate-7"
    assert first.key == second.key
    assert json.loads(first.key) == {
        "case_id": "candidate-7",
        "line": 42,
        "name": "openai client",
        "path": "src/agent.py",
        "repository": "my-repo",
        "type": "llm_endpoint",
    }
    assert component_action_key(serialized, repo_root=repo_root) == ("case:candidate-7")


def test_component_metrics_are_exact_unique_entity_set_metrics(tmp_path):
    repo_root = tmp_path / "repo"
    expected = [
        _component("a", "model-a", "model", path=str(repo_root / "a.py")),
        _component("b", "tool-b", "tool", path=str(repo_root / "b.py")),
    ]
    predicted_a = _component("a", " MODEL-A ", "MODEL", path="a.py")
    predicted = [
        predicted_a,
        dict(predicted_a),  # Duplicate detections count as one entity.
        _component("c", "agent-c", "agent", path="c.py"),
    ]

    result = evaluate_decisions(
        predicted_components=predicted,
        expected_components=expected,
        repo_root=repo_root,
    )

    assert result.components.true_positives == 1
    assert result.components.false_positives == 1
    assert result.components.false_negatives == 1
    assert result.components.predicted_count == 2
    assert result.components.precision == pytest.approx(0.5)
    assert result.components.recall == pytest.approx(0.5)
    assert result.components.f1 == pytest.approx(0.5)
    assert result.baseline_components is None
    assert result.net_recall_lift is None
    assert result.action_accuracy is None


def test_wrong_same_type_entity_is_not_a_true_positive():
    expected = [_component("shared", "model-a", "model", path="src/model.py")]
    predicted = [_component("shared", "model-b", "model", path="src/model.py")]

    result = evaluate_decisions(
        predicted_components=predicted,
        expected_components=expected,
        repository="repo",
    )

    assert result.components.true_positives == 0
    assert result.components.false_positives == 1
    assert result.components.false_negatives == 1
    assert result.components.f1 == 0.0


def test_relationship_identity_resolves_run_specific_instance_ids():
    expected_components = [
        _component("agent", "router", "agent", instance_id="expected-router"),
        _component("model", "gpt", "model", instance_id="expected-model"),
    ]
    predicted_components = [
        _component("agent", "Router", "AGENT", instance_id="run-17-router"),
        _component("model", "GPT", "MODEL", instance_id="run-17-model"),
    ]
    expected_relationship = ComponentRelationship(
        source_instance_id="expected-router",
        target_instance_id="expected-model",
        relationship_type=RelationshipType.USES_MODEL,
    )
    predicted_relationship = {
        "source_instance_id": "run-17-router",
        "target_instance_id": "run-17-model",
        "relationship_type": "uses_model",
    }

    expected_identity = canonical_relationship_identity(
        expected_relationship,
        components=expected_components,
        repository="example/repo",
    )
    predicted_identity = canonical_relationship_identity(
        predicted_relationship,
        components=predicted_components,
        repository="example/repo",
    )
    result = evaluate_decisions(
        predicted_components=predicted_components,
        expected_components=expected_components,
        predicted_relationships=[predicted_relationship],
        expected_relationships=[expected_relationship],
        repository="example/repo",
    )

    assert predicted_identity == expected_identity
    assert result.relationships.true_positives == 1
    assert result.relationships.precision == 1.0
    assert result.relationships.recall == 1.0
    assert result.relationships.f1 == 1.0


def test_relationship_identity_resolves_suite_case_ids_to_runtime_components():
    expected_components = [
        _component("agent", "router", "agent", instance_id="fixture-router"),
        _component("model", "gpt", "model", instance_id="fixture-model"),
    ]
    predicted_components = [
        _component("agent", "Router", "AGENT", instance_id="run-42-router"),
        _component("model", "GPT", "MODEL", instance_id="run-42-model"),
    ]
    expected_relationship = {
        "source_case_id": "agent",
        "target_case_id": "model",
        "relationship_type": "uses_model",
    }
    predicted_relationship = {
        "source_instance_id": "run-42-router",
        "target_instance_id": "run-42-model",
        "relationship_type": "uses_model",
    }

    expected_identity = canonical_relationship_identity(
        expected_relationship,
        components=expected_components,
        repository="example/repo",
    )
    predicted_identity = canonical_relationship_identity(
        predicted_relationship,
        components=predicted_components,
        repository="example/repo",
    )
    result = evaluate_decisions(
        predicted_components=predicted_components,
        expected_components=expected_components,
        predicted_relationships=[predicted_relationship],
        expected_relationships=[expected_relationship],
        repository="example/repo",
    )

    assert predicted_identity == expected_identity
    assert result.relationships.true_positives == 1
    assert result.relationships.false_positives == 0
    assert result.relationships.false_negatives == 0
    assert result.relationships.f1 == 1.0


def test_unresolved_relationship_endpoint_ids_remain_distinct():
    expected_relationship = {
        "source_case_id": "fixture-agent",
        "target_case_id": "fixture-model",
        "relationship_type": "uses_model",
    }
    predicted_relationship = {
        "source_instance_id": "unknown-agent",
        "target_instance_id": "unknown-model",
        "relationship_type": "uses_model",
    }

    expected_identity = canonical_relationship_identity(expected_relationship)
    predicted_identity = canonical_relationship_identity(predicted_relationship)
    result = evaluate_decisions(
        predicted_components=[],
        expected_components=[],
        predicted_relationships=[predicted_relationship],
        expected_relationships=[expected_relationship],
    )

    assert predicted_identity != expected_identity
    assert json.loads(expected_identity.source) == {"id": "fixture-agent"}
    assert json.loads(predicted_identity.source) == {"id": "unknown-agent"}
    assert result.relationships.true_positives == 0
    assert result.relationships.false_positives == 1
    assert result.relationships.false_negatives == 1


def test_relationship_type_mismatch_is_one_false_positive_and_negative():
    components = [
        _component("agent", "router", "agent", instance_id="router"),
        _component("model", "gpt", "model", instance_id="model"),
    ]
    endpoints = {
        "source_instance_id": "router",
        "target_instance_id": "model",
    }

    result = evaluate_decisions(
        predicted_components=components,
        expected_components=components,
        predicted_relationships=[{**endpoints, "relationship_type": "calls"}],
        expected_relationships=[{**endpoints, "relationship_type": "uses_model"}],
    )

    assert result.relationships.true_positives == 0
    assert result.relationships.false_positives == 1
    assert result.relationships.false_negatives == 1


def test_relationship_recall_lift_uses_optional_deterministic_baseline():
    components = [
        _component("agent", "router", "agent", instance_id="router"),
        _component("model", "gpt", "model", instance_id="model"),
        _component("tool", "search", "tool", instance_id="tool"),
    ]
    uses_model = {
        "source_instance_id": "router",
        "target_instance_id": "model",
        "relationship_type": "uses_model",
    }
    uses_tool = {
        "source_instance_id": "router",
        "target_instance_id": "tool",
        "relationship_type": "uses_tool",
    }

    result = evaluate_decisions(
        predicted_components=components,
        expected_components=components,
        predicted_relationships=[uses_model, uses_tool],
        expected_relationships=[uses_model, uses_tool],
        deterministic_relationships=[uses_model],
        repository="repo",
    )

    assert result.relationships.recall == 1.0
    assert result.baseline_relationships is not None
    assert result.baseline_relationships.recall == pytest.approx(0.5)
    assert result.relationship_recall_lift == pytest.approx(0.5)
    assert result.details.baseline_relationships is not None
    assert len(result.details.baseline_relationships.false_negative_ids) == 1


def test_relationship_identity_remains_stable_across_endpoint_reclassification():
    deterministic = [
        _component("router", "router", "tool", instance_id="router-runtime"),
        _component("model", "gpt", "model", instance_id="model-runtime"),
    ]
    final = [
        _component("router", "router", "agent", instance_id="router-runtime"),
        deterministic[1],
    ]
    # Runtime AIBOM components are not required to carry evaluation-only case
    # IDs; their instance IDs must still make relationship identity stable.
    for component in [*deterministic, *final]:
        component.pop("metadata", None)
    edge = {
        "source_id": "router-runtime",
        "target_id": "model-runtime",
        "relationship_type": "uses_model",
    }

    result = evaluate_decisions(
        predicted_components=final,
        expected_components=final,
        deterministic_components=deterministic,
        predicted_relationships=[edge],
        expected_relationships=[edge],
        deterministic_relationships=[edge],
    )

    assert result.baseline_relationships is not None
    assert result.baseline_relationships.recall == 1.0
    assert result.relationship_recall_lift == 0.0


def test_explicit_negative_relationship_labels_are_scored_as_absence():
    components = [
        _component("agent", "router", "agent", instance_id="router"),
        _component("model", "gpt", "model", instance_id="model"),
        _component("tool", "search", "tool", instance_id="tool"),
    ]
    positive = {
        "source_case_id": "agent",
        "target_case_id": "model",
        "relationship_type": "uses_model",
    }
    negative = {
        "source_case_id": "agent",
        "target_case_id": "tool",
        "relationship_type": "uses_tool",
        "expected_present": False,
    }
    predicted_absent = {**negative, "predicted_present": False}

    safe_result = evaluate_decisions(
        predicted_components=components,
        expected_components=components,
        predicted_relationships=[positive, predicted_absent],
        expected_relationships=[positive, negative],
    )

    assert safe_result.relationships.true_positives == 1
    assert safe_result.relationships.false_positives == 0
    assert safe_result.relationships.false_negatives == 0
    assert safe_result.relationships.predicted_count == 1
    assert safe_result.relationships.expected_count == 1
    assert safe_result.relationships.f1 == 1.0

    violated_result = evaluate_decisions(
        predicted_components=components,
        expected_components=components,
        predicted_relationships=[positive, dict(negative)],
        expected_relationships=[positive, negative],
    )

    assert violated_result.relationships.true_positives == 1
    assert violated_result.relationships.false_positives == 1
    assert violated_result.relationships.false_negatives == 0
    assert violated_result.relationships.precision == pytest.approx(0.5)


def test_risk_identity_and_exact_metrics_ignore_description_and_weight(tmp_path):
    repo_root = tmp_path / "repo"
    expected_match = {
        "flag": "UNRESOLVED MODEL REFERENCE",
        "severity": "HIGH",
        "weight": 100,
        "description": "reviewer wording",
        "file_path": str(repo_root / "src" / "agent.py"),
        "line_number": "19",
        "metadata": {"stable_case_id": "risk-a"},
    }
    predicted_match = {
        "risk_type": "unresolved_model-reference",
        "severity": "high",
        "weight": 1,
        "description": "different generated prose",
        "source_path": "src/agent.py",
        "line": 19,
        "case_id": "risk-a",
    }
    expected_other = {
        "flag": "dynamic tool loading",
        "severity": "medium",
        "file_path": "src/tools.py",
        "line_number": 7,
        "case_id": "risk-b",
    }
    predicted_wrong_severity = {
        **expected_other,
        "severity": "high",
        "description": "severity is part of identity",
    }

    assert canonical_risk_identity(
        predicted_match, repo_root=repo_root
    ) == canonical_risk_identity(expected_match, repo_root=repo_root)

    result = evaluate_decisions(
        predicted_components=[],
        expected_components=[],
        predicted_risks=[predicted_match, predicted_wrong_severity],
        expected_risks=[expected_match, expected_other],
        repo_root=repo_root,
    )

    assert result.risks.true_positives == 1
    assert result.risks.false_positives == 1
    assert result.risks.false_negatives == 1
    assert result.risks.precision == pytest.approx(0.5)
    assert result.risks.recall == pytest.approx(0.5)
    assert result.risks.f1 == pytest.approx(0.5)
    assert len(result.details.risks.true_positive_ids) == 1


def test_explicit_negative_risk_labels_do_not_become_false_negatives():
    positive = {
        "flag": "unresolved model",
        "severity": "high",
        "file_path": "src/a.py",
        "case_id": "positive",
    }
    negative = {
        "flag": "hardcoded secret",
        "severity": "critical",
        "file_path": "src/a.py",
        "case_id": "negative",
        "expected_present": False,
    }
    predicted_absent = {**negative, "predicted_present": False}

    safe_result = evaluate_decisions(
        predicted_components=[],
        expected_components=[],
        predicted_risks=[positive, predicted_absent],
        expected_risks=[positive, negative],
    )

    assert safe_result.risks.true_positives == 1
    assert safe_result.risks.false_positives == 0
    assert safe_result.risks.false_negatives == 0
    assert safe_result.risks.f1 == 1.0

    violated_result = evaluate_decisions(
        predicted_components=[],
        expected_components=[],
        predicted_risks=[positive, dict(negative)],
        expected_risks=[positive, negative],
    )

    assert violated_result.risks.true_positives == 1
    assert violated_result.risks.false_positives == 1
    assert violated_result.risks.false_negatives == 0
    assert violated_result.risks.precision == pytest.approx(0.5)


def test_baseline_recall_lift_and_discovery_quality():
    baseline = [_component("a", "model-a", "model", path="a.py")]
    expected = [
        _component("a", "model-a", "model", path="a.py"),
        _component("c", "model-c", "model", path="c.py"),
    ]
    predicted = [
        _component("a", "model-a", "model", path="a.py"),
        _component("c", "model-c", "model", path="c.py"),
        _component("d", "model-d", "model", path="d.py"),
    ]

    result = evaluate_decisions(
        predicted_components=predicted,
        expected_components=expected,
        deterministic_components=baseline,
        repository="org/repo",
    )

    assert result.baseline_components is not None
    assert result.baseline_components.recall == pytest.approx(0.5)
    assert result.components.recall == 1.0
    assert result.net_recall_lift == pytest.approx(0.5)
    assert result.discoveries is not None
    assert result.discoveries.true_positives == 1
    assert result.discoveries.false_positives == 1
    assert result.discoveries.false_negatives == 0
    assert result.discoveries.precision == pytest.approx(0.5)
    assert result.discoveries.recall == 1.0


def test_over_pruning_and_inferred_action_accuracy():
    baseline = [
        _component("a", "client", "tool", path="a.py"),
        _component("b", "guard", "guardrail", path="b.py"),
    ]
    expected = [
        _component("a", "client", "agent", path="a.py"),
        _component("b", "guard", "guardrail", path="b.py"),
    ]
    predicted = [_component("a", "client", "agent", path="a.py")]

    result = evaluate_decisions(
        predicted_components=predicted,
        expected_components=expected,
        deterministic_components=baseline,
        repository="repo",
    )

    assert result.over_pruning is not None
    assert result.over_pruning.over_pruned_count == 1
    assert result.over_pruning.eligible_baseline_count == 2
    assert result.over_pruning.rate == pytest.approx(0.5)
    assert result.over_pruning.over_pruned_action_keys == ["case:b"]

    assert result.action_accuracy is not None
    assert result.action_accuracy.correct_count == 1
    assert result.action_accuracy.evaluated_count == 2
    assert result.action_accuracy.accuracy == pytest.approx(0.5)
    assert [mismatch.action_key for mismatch in result.action_accuracy.mismatches] == [
        "case:b"
    ]

    assert result.reclassification_accuracy is not None
    assert result.reclassification_accuracy.correct_count == 1
    assert result.reclassification_accuracy.accuracy == 1.0


def test_inferred_enrich_uses_substantive_changes_not_agentic_bookkeeping():
    baseline = [
        _component("model", "client", "model", path="model.py"),
        _component("tool", "guard", "tool", path="guard.py"),
        _component("agent", "router", "agent", path="router.py"),
    ]
    enriched_model = {
        **baseline[0],
        "instance_id": "run-model",
        "model_name": "gpt-5",
        "needs_agentic": False,
        "agentic_confidence": 0.98,
    }
    bookkeeping_only = {
        **baseline[1],
        "instance_id": "run-tool",
        "needs_agentic": False,
        "agentic_hint": "reviewed",
        "agentic_confidence": 0.97,
        "decision_annotation": {"decision": "keep"},
        "metadata": {
            "stable_case_id": "tool",
            "agent_evidence": {"evidence_snippet": "bounded evidence"},
            "agentic_status": "complete",
        },
    }
    enriched_metadata = {
        **baseline[2],
        "metadata": {
            "stable_case_id": "agent",
            "verified_provider": "openai",
        },
    }
    final_components = [enriched_model, bookkeeping_only, enriched_metadata]

    result = evaluate_decisions(
        predicted_components=final_components,
        expected_components=final_components,
        deterministic_components=baseline,
        expected_actions={
            "model": "enrich",
            "tool": "keep",
            "agent": "enrich",
        },
        repository="repo",
    )

    assert result.action_accuracy is not None
    assert result.action_accuracy.correct_count == 3
    assert result.action_accuracy.evaluated_count == 3
    assert result.action_accuracy.accuracy == 1.0
    assert result.action_accuracy.mismatches == []
    assert result.action_macro_f1 == 1.0
    assert result.decision_coverage == 1.0


def test_explicit_reclassification_target_is_scored():
    result = evaluate_decisions(
        predicted_components=[],
        expected_components=[],
        predicted_actions={"candidate-1": {"action": "reclassify", "new_type": "tool"}},
        expected_actions={
            "candidate-1": {"action": "reclassify", "target_type": "agent"}
        },
    )

    assert result.action_accuracy is not None
    assert result.action_accuracy.accuracy == 0.0
    assert result.reclassification_accuracy is not None
    assert result.reclassification_accuracy.accuracy == 0.0
    mismatch = result.reclassification_accuracy.mismatches[0]
    assert mismatch.action_key == "case:candidate-1"
    assert mismatch.expected.target_type == "agent"
    assert mismatch.predicted is not None
    assert mismatch.predicted.target_type == "tool"


def test_partial_actions_report_macro_f1_coverage_and_missing_predictions():
    expected_actions = {
        "a": "keep",
        "b": "remove",
        "c": {"action": "reclassify", "target_type": "agent"},
    }
    predicted_actions = {
        "c": {"action": "reclassify", "target_type": "tool"},
        "a": "keep",
    }

    result = evaluate_decisions(
        predicted_components=[],
        expected_components=[],
        predicted_actions=predicted_actions,
        expected_actions=expected_actions,
    )
    reversed_result = evaluate_decisions(
        predicted_components=[],
        expected_components=[],
        predicted_actions=dict(reversed(list(predicted_actions.items()))),
        expected_actions=dict(reversed(list(expected_actions.items()))),
    )

    assert result.model_dump_json() == reversed_result.model_dump_json()
    assert result.action_accuracy is not None
    assert result.action_accuracy.correct_count == 1
    assert result.action_accuracy.evaluated_count == 3
    assert result.action_accuracy.accuracy == pytest.approx(1 / 3)
    assert [mismatch.action_key for mismatch in result.action_accuracy.mismatches] == [
        "case:b",
        "case:c",
    ]
    assert result.action_accuracy.mismatches[0].predicted is None
    assert result.action_macro_f1 == pytest.approx(2 / 3)
    assert result.decision_coverage == pytest.approx(2 / 3)
    assert result.reclassification_accuracy is not None
    assert result.reclassification_accuracy.accuracy == 0.0


def test_all_missing_explicit_actions_score_zero_without_remove_imputation():
    result = evaluate_decisions(
        predicted_components=[],
        expected_components=[],
        predicted_actions={},
        expected_actions={"a": "keep", "b": "remove"},
    )

    assert result.action_accuracy is not None
    assert result.action_accuracy.accuracy == 0.0
    assert all(
        mismatch.predicted is None for mismatch in result.action_accuracy.mismatches
    )
    assert result.action_macro_f1 == 0.0
    assert result.decision_coverage == 0.0


def test_output_order_and_flat_metric_projection_are_deterministic():
    baseline = [_component("base", "base", "model", path="base.py")]
    expected = [
        _component("a", "alpha", "model", path="a.py"),
        _component("b", "beta", "tool", path="b.py"),
    ]
    predicted = [
        _component("c", "gamma", "agent", path="c.py"),
        _component("a", "alpha", "model", path="a.py"),
    ]

    first = evaluate_decisions(
        predicted_components=predicted,
        expected_components=expected,
        deterministic_components=baseline,
        repository="repo",
    )
    second = evaluate_decisions(
        predicted_components=reversed(predicted),
        expected_components=reversed(expected),
        deterministic_components=reversed(baseline),
        repository="repo",
    )

    assert first.model_dump_json() == second.model_dump_json()
    metrics = first.to_galileo_metrics()
    assert metrics["aibom.components.precision"] == pytest.approx(0.5)
    assert metrics["aibom.components.recall"] == pytest.approx(0.5)
    assert metrics["aibom.components.f1"] == pytest.approx(0.5)
    assert metrics["aibom.net_recall_lift"] == pytest.approx(0.5)
    assert metrics["aibom.discoveries.precision"] == pytest.approx(0.5)
    assert metrics["aibom.over_prune_rate"] == 0.0
    assert all(isinstance(value, float) for value in metrics.values())


def test_flat_metric_projection_includes_relationship_risk_and_action_metrics():
    components = [
        _component("agent", "router", "agent", instance_id="router"),
        _component("model", "gpt", "model", instance_id="model"),
    ]
    relationship = {
        "source_instance_id": "router",
        "target_instance_id": "model",
        "relationship_type": "uses_model",
    }
    risk = {
        "flag": "unresolved model",
        "severity": "high",
        "file_path": "src/a.py",
        "case_id": "risk",
    }
    result = evaluate_decisions(
        predicted_components=components,
        expected_components=components,
        predicted_relationships=[relationship],
        expected_relationships=[relationship],
        deterministic_relationships=[],
        predicted_risks=[risk],
        expected_risks=[risk],
        predicted_actions={"a": "keep"},
        expected_actions={"a": "keep", "b": "remove"},
        repository="repo",
    )

    metrics = result.to_galileo_metrics()

    assert metrics["aibom.risks.precision"] == 1.0
    assert metrics["aibom.risks.recall"] == 1.0
    assert metrics["aibom.risks.f1"] == 1.0
    assert metrics["aibom.baseline_relationships.recall"] == 0.0
    assert metrics["aibom.relationship_recall_lift"] == 1.0
    assert metrics["aibom.action_accuracy"] == pytest.approx(0.5)
    assert metrics["aibom.action_macro_f1"] == pytest.approx(0.5)
    assert metrics["aibom.decision_coverage"] == pytest.approx(0.5)
    assert all(isinstance(value, float) for value in metrics.values())


def test_pydantic_component_inputs_are_supported():
    component = AIComponent(
        name="router",
        component_type=AIComponentType.AGENT,
        file_path="src/router.py",
        line_number=8,
        metadata={"eval_case_id": "router-8"},
    )

    result = evaluate_decisions(
        predicted_components=[component],
        expected_components=[component.model_dump()],
        repository="repo",
    )

    assert result.components.f1 == 1.0
