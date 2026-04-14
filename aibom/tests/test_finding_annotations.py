from __future__ import annotations

from aibom.finding_annotations import annotate_findings
from aibom.models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    DecisionAnnotation,
    RelationshipType,
)
from aibom.models.scan import RiskFlag


def test_annotate_findings_adds_default_annotations(tmp_path) -> None:
    source_file = tmp_path / "app.py"
    source_file.write_text("agent = RouterAgent()\nagent.run(task)\n", encoding="utf-8")

    component = AIComponent(
        name="router_agent",
        component_type=AIComponentType.AGENT,
        file_path=str(source_file),
        line_number=1,
        instance_id="agent-1",
    )
    model = AIComponent(
        name="gpt-4o",
        component_type=AIComponentType.MODEL,
        file_path=str(source_file),
        line_number=2,
        instance_id="model-1",
    )
    relationship = ComponentRelationship(
        source_instance_id=component.instance_id,
        target_instance_id=model.instance_id,
        source_name=component.name,
        target_name=model.name,
        relationship_type=RelationshipType.USES_MODEL,
        source_type=component.component_type,
        target_type=model.component_type,
    )
    risk_flag = RiskFlag(
        flag="sensitive_prompt",
        severity="medium",
        weight=5,
        description="Prompt content should be reviewed before deployment.",
        file_path=str(source_file),
        line_number=2,
    )

    components, relationships, risk_flags = annotate_findings(
        [component, model],
        [relationship],
        [risk_flag],
        include_code_snippets=False,
    )

    assert all(component.decision_annotation is not None for component in components)
    assert relationships[0].decision_annotation is not None
    assert relationships[0].decision_annotation.decision == "derived"
    assert risk_flags[0].decision_annotation is not None
    assert risk_flags[0].decision_annotation.decision == "flagged"
    assert risk_flags[0].decision_annotation.code_snippet is None


def test_annotate_findings_hydrates_snippets_when_enabled(tmp_path) -> None:
    source_file = tmp_path / "flow.py"
    source_file.write_text(
        "client = OpenAI()\nresponse = client.responses.create()\n",
        encoding="utf-8",
    )
    component = AIComponent(
        name="openai_client",
        component_type=AIComponentType.LLM_ENDPOINT,
        file_path=str(source_file),
        line_number=1,
        instance_id="endpoint-1",
        decision_annotation=DecisionAnnotation(
            decision="confirmed",
            justification="Client creation is present in the request path.",
            evidence_kinds=["code_context"],
            evidence_locations=[],
        ),
    )

    components, _, _ = annotate_findings(
        [component],
        [],
        [],
        include_code_snippets=True,
    )

    assert components[0].decision_annotation is not None
    assert components[0].decision_annotation.code_snippet is not None
    assert "client = OpenAI()" in components[0].decision_annotation.code_snippet.text


def test_snippet_blocked_for_out_of_repo_path(tmp_path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    in_repo = repo_dir / "app.py"
    in_repo.write_text("agent = RouterAgent()\n", encoding="utf-8")

    outside = tmp_path / "secret.txt"
    outside.write_text("TOP SECRET DATA\n", encoding="utf-8")

    legit = AIComponent(
        name="router",
        component_type=AIComponentType.AGENT,
        file_path=str(in_repo),
        line_number=1,
        instance_id="a-1",
    )
    hostile = AIComponent(
        name="exfil",
        component_type=AIComponentType.AGENT,
        file_path=str(outside),
        line_number=1,
        instance_id="a-2",
    )

    components, _, _ = annotate_findings(
        [legit, hostile],
        [],
        [],
        include_code_snippets=True,
        allowed_roots=[str(repo_dir)],
    )

    assert components[0].decision_annotation.code_snippet is not None
    assert "RouterAgent" in components[0].decision_annotation.code_snippet.text

    assert components[1].decision_annotation.code_snippet is None
