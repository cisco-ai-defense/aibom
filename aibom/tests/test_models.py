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

from aibom.models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    DetectionSource,
    RelationshipType,
    RiskFlag,
    RiskScore,
    ScanContext,
    ScanResult,
    Severity,
    SourceResult,
)


_EXPECTED_AI_COMPONENT_TYPES: dict[str, str] = {
    "MODEL": "model",
    "LLM_ENDPOINT": "llm_endpoint",
    "MODEL_ENDPOINT": "model_endpoint",
    "AGENT": "agent",
    "TOOL": "tool",
    "MCP_SERVER": "mcp_server",
    "MCP_CLIENT": "mcp_client",
    "MCP_GATEWAY": "mcp_gateway",
    "EMBEDDING": "embedding",
    "VECTOR_STORE": "vector_store",
    "DATASET": "dataset",
    "RETRIEVER": "retriever",
    "KNOWLEDGE_BASE": "knowledge_base",
    "FEATURE_STORE": "feature_store",
    "PROMPT": "prompt",
    "GUARDRAIL": "guardrail",
    "MEMORY": "memory",
    "TRAINING_RUN": "training_run",
    "HYPERPARAMETER": "hyperparameter",
    "MODEL_ARTIFACT": "model_artifact",
    "EXPERIMENT_TRACKER": "experiment_tracker",
    "MODEL_REGISTRY": "model_registry",
    "DATA_VERSIONING": "data_versioning",
    "ML_PIPELINE": "ml_pipeline",
    "SKILL": "skill",
    "OBSERVABILITY": "observability",
    "SECRET": "secret",
    "DEPENDENCY": "dependency",
    "OTHER": "other",
}


def test_ai_component_type_all_members_and_string_values():
    for name, value in _EXPECTED_AI_COMPONENT_TYPES.items():
        member = getattr(AIComponentType, name)
        assert member.value == value
        assert member == value
    assert set(AIComponentType.__members__) == set(_EXPECTED_AI_COMPONENT_TYPES)


def test_severity_ordering_and_numeric():
    assert Severity.CRITICAL > Severity.HIGH > Severity.MEDIUM > Severity.LOW > Severity.INFO
    assert Severity.INFO < Severity.LOW < Severity.MEDIUM < Severity.HIGH < Severity.CRITICAL
    assert Severity.CRITICAL.numeric == 4
    assert Severity.HIGH.numeric == 3
    assert Severity.MEDIUM.numeric == 2
    assert Severity.LOW.numeric == 1
    assert Severity.INFO.numeric == 0


def test_severity_comparison_operators():
    assert Severity.LOW < Severity.HIGH
    assert Severity.LOW <= Severity.HIGH
    assert Severity.HIGH > Severity.MEDIUM
    assert Severity.HIGH >= Severity.MEDIUM
    assert Severity.INFO <= Severity.INFO
    assert Severity.CRITICAL >= Severity.CRITICAL
    assert not (Severity.MEDIUM < Severity.MEDIUM)
    assert Severity.MEDIUM <= Severity.MEDIUM


def test_ai_component_minimal_and_instance_id():
    c = AIComponent(name="m", component_type=AIComponentType.MODEL)
    assert c.instance_id == "m__0"


def test_ai_component_instance_id_preserved_when_set():
    c = AIComponent(
        name="x",
        component_type=AIComponentType.AGENT,
        instance_id="custom",
    )
    assert c.instance_id == "custom"


def test_ai_component_serialization_roundtrip():
    c = AIComponent(
        name="svc",
        component_type=AIComponentType.TOOL,
        file_path="/app/main.py",
        line_number=10,
        framework="langchain",
    )
    data = c.model_dump()
    c2 = AIComponent(**data)
    assert c2 == c


def test_ai_component_optional_defaults():
    c = AIComponent(name="n", component_type=AIComponentType.OTHER)
    assert c.file_path == ""
    assert c.line_number == 0
    assert c.framework == ""
    assert c.detection_source == DetectionSource.CODE_ANALYSIS
    assert c.confidence == 1.0
    assert c.model_name is None
    assert c.embedding_model is None
    assert c.description is None
    assert c.text is None
    assert c.transport is None
    assert c.config_source is None
    assert c.storage_uri is None
    assert c.dataset_source is None
    assert c.skill_format is None
    assert c.hyperparameters == {}
    assert c.training_info is None
    assert c.metrics == {}
    assert c.kb_concept is None
    assert c.kb_label is None
    assert c.sdk_version is None
    assert c.metadata == {}


def test_ai_component_model_dump_json_mode():
    c = AIComponent(name="j", component_type=AIComponentType.PROMPT)
    payload = c.model_dump(mode="json")
    json.dumps(payload)
    assert payload["component_type"] == "prompt"
    assert payload["detection_source"] == "code_analysis"


def test_component_relationship_auto_label_and_dump():
    r = ComponentRelationship(
        source_instance_id="a_1",
        target_instance_id="b_2",
        relationship_type=RelationshipType.USES_MODEL,
    )
    assert r.label == "USES_MODEL"
    d = r.model_dump()
    assert d["label"] == "USES_MODEL"
    assert d["relationship_type"] == RelationshipType.USES_MODEL


def test_component_relationship_explicit_label():
    r = ComponentRelationship(
        source_instance_id="s",
        target_instance_id="t",
        label="custom",
    )
    assert r.label == "custom"


def test_risk_score_starts_zero_info():
    rs = RiskScore()
    assert rs.score == 0
    assert rs.severity == Severity.INFO
    assert rs.flags == []


@pytest.mark.parametrize(
    ("weights", "expected_severity"),
    [
        ([10], Severity.LOW),
        ([25], Severity.LOW),
        ([26], Severity.MEDIUM),
        ([50], Severity.MEDIUM),
        ([51], Severity.HIGH),
        ([75], Severity.HIGH),
        ([76], Severity.CRITICAL),
        ([100], Severity.CRITICAL),
    ],
)
def test_risk_score_severity_bands(weights: list[int], expected_severity: Severity):
    rs = RiskScore()
    for w in weights:
        rs.add_flag(
            RiskFlag(
                flag="f",
                severity=Severity.INFO,
                weight=w,
                description="d",
            )
        )
    assert rs.severity == expected_severity


def test_risk_score_add_flag_increments_and_caps():
    rs = RiskScore()
    rs.add_flag(
        RiskFlag(flag="a", severity=Severity.LOW, weight=40, description="")
    )
    assert rs.score == 40
    rs.add_flag(
        RiskFlag(flag="b", severity=Severity.LOW, weight=40, description="")
    )
    assert rs.score == 80
    rs.add_flag(
        RiskFlag(flag="c", severity=Severity.LOW, weight=50, description="")
    )
    assert rs.score == 100
    assert len(rs.flags) == 3


def test_source_result_with_components_and_relationships():
    c1 = AIComponent(name="c1", component_type=AIComponentType.MODEL)
    c2 = AIComponent(name="c2", component_type=AIComponentType.AGENT)
    r = ComponentRelationship(
        source_instance_id=c1.instance_id,
        target_instance_id=c2.instance_id,
        relationship_type=RelationshipType.USES_MODEL,
    )
    sr = SourceResult(path="/proj", components=[c1, c2], relationships=[r])
    assert len(sr.components) == 2
    assert sr.relationships[0].source_instance_id == c1.instance_id


def test_scan_result_aggregates_and_summary():
    c_a = AIComponent(name="m1", component_type=AIComponentType.MODEL, file_path="a.py")
    c_b = AIComponent(name="m2", component_type=AIComponentType.MODEL, file_path="b.py")
    c_c = AIComponent(name="ag", component_type=AIComponentType.AGENT, file_path="c.py")
    rel = ComponentRelationship(
        source_instance_id=c_c.instance_id,
        target_instance_id=c_a.instance_id,
    )
    s1 = SourceResult(path="a.py", components=[c_a], relationships=[])
    s2 = SourceResult(path="dir", components=[c_b, c_c], relationships=[rel])
    risk = RiskScore()
    risk.add_flag(
        RiskFlag(flag="x", severity=Severity.HIGH, weight=60, description="")
    )
    scan = ScanResult(sources=[s1, s2], risk=risk)
    comps = scan.all_components
    assert len(comps) == 3
    assert {c.name for c in comps} == {"m1", "m2", "ag"}
    rels = scan.all_relationships
    assert len(rels) == 1
    assert rels[0].target_instance_id == c_a.instance_id
    summ = scan.summary
    assert summ["total_sources"] == 2
    assert summ["total_components"] == 3
    assert summ["component_types"] == {"model": 2, "agent": 1}
    assert summ["total_relationships"] == 1
    assert summ["risk_score"] == 60
    assert summ["risk_severity"] == "high"


def test_scan_context_defaults():
    ctx = ScanContext(paths=["/scan"])
    assert ctx.paths == ["/scan"]
    assert ctx.exclude_patterns == []
    assert ctx.output_format == "json"
    assert ctx.output_file is None
    assert ctx.config == {}
    assert ctx.kb_path is None
    assert ctx.llm_config is None
    assert ctx.fail_on is None
    assert ctx.min_severity == Severity.INFO
