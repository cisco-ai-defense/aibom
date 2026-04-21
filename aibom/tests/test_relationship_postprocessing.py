# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for relationship post-processing helpers in ``scan_pipeline``.

Covers three correctness fixes flagged during code review of PR #37:

* ``_dedup_relationships`` must not collapse edges that share source/target
  names but point to distinct component instances (different files).
* ``_propagate_model_from_relationships`` must not broadcast a
  ``model_name`` across every component that happens to share a name.
* ``_resolve_relationship_types`` must not leak an owner's
  ``component_type`` onto that owner's ``model_name`` — only
  model-related owners may contribute a type alias for their model id.
"""

from __future__ import annotations

from aibom.models.enums import AIComponentType, RelationshipType
from aibom.models.scan import AIComponent, ComponentRelationship
from aibom.scan_pipeline import (
    _dedup_relationships,
    _propagate_model_from_relationships,
    _resolve_relationship_types,
)


class TestDedupRelationships:
    """``_dedup_relationships`` should key on instance id, not component name."""

    def test_distinct_instances_with_same_name_are_kept(self) -> None:
        """Two edges whose sources share a name but live in different files must both survive."""
        rel_a = ComponentRelationship(
            source_instance_id="agent_a.py_10",
            target_instance_id="gpt-4_a.py_10",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        rel_b = ComponentRelationship(
            source_instance_id="agent_b.py_20",
            target_instance_id="gpt-4_b.py_20",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _dedup_relationships([rel_a, rel_b])
        assert len(out) == 2
        assert {r.source_instance_id for r in out} == {
            "agent_a.py_10",
            "agent_b.py_20",
        }

    def test_exact_duplicates_are_collapsed(self) -> None:
        rel = ComponentRelationship(
            source_instance_id="agent_a.py_10",
            target_instance_id="gpt-4_a.py_10",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _dedup_relationships([rel, rel.model_copy()])
        assert len(out) == 1

    def test_llm_edges_with_blank_instance_ids_fall_back_to_names(self) -> None:
        """LLM-produced edges with empty instance ids should dedup on names only."""
        rel_a = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        rel_b = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _dedup_relationships([rel_a, rel_b])
        assert len(out) == 1

    def test_scanner_and_llm_edges_do_not_collide(self) -> None:
        """An id-keyed scanner edge and a name-keyed LLM edge must not collapse."""
        scanner_rel = ComponentRelationship(
            source_instance_id="agent_a.py_10",
            target_instance_id="gpt-4_a.py_10",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        llm_rel = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _dedup_relationships([scanner_rel, llm_rel])
        assert len(out) == 2


class TestPropagateModelFromRelationships:
    """``_propagate_model_from_relationships`` should target one instance, not every match."""

    def test_propagates_only_to_matching_instance(self) -> None:
        """Two agents share a name; only the one referenced by the edge gets the model."""
        agent_a = AIComponent(
            name="agent",
            component_type=AIComponentType.AGENT,
            file_path="a.py",
            line_number=10,
        )
        agent_b = AIComponent(
            name="agent",
            component_type=AIComponentType.AGENT,
            file_path="b.py",
            line_number=20,
        )
        rel = ComponentRelationship(
            source_instance_id=agent_a.instance_id,
            target_instance_id="gpt-4_a.py_10",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _propagate_model_from_relationships([agent_a, agent_b], [rel])
        by_id = {c.instance_id: c for c in out}
        assert by_id[agent_a.instance_id].model_name == "gpt-4"
        assert by_id[agent_b.instance_id].model_name is None

    def test_does_not_overwrite_existing_model_name(self) -> None:
        agent = AIComponent(
            name="agent",
            component_type=AIComponentType.AGENT,
            file_path="a.py",
            line_number=10,
            model_name="claude-3-opus",
        )
        rel = ComponentRelationship(
            source_instance_id=agent.instance_id,
            target_instance_id="gpt-4_a.py_10",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _propagate_model_from_relationships([agent], [rel])
        assert out[0].model_name == "claude-3-opus"

    def test_uses_embedding_relationship_is_respected(self) -> None:
        agent = AIComponent(
            name="agent",
            component_type=AIComponentType.AGENT,
            file_path="a.py",
            line_number=10,
        )
        rel = ComponentRelationship(
            source_instance_id=agent.instance_id,
            target_instance_id="ada-002_a.py_10",
            source_name="agent",
            target_name="text-embedding-ada-002",
            relationship_type=RelationshipType.USES_EMBEDDING,
        )
        out = _propagate_model_from_relationships([agent], [rel])
        assert out[0].model_name == "text-embedding-ada-002"

    def test_llm_edge_with_blank_instance_id_falls_back_to_name(self) -> None:
        """With no instance id, a single agent matching by name still gets the model."""
        agent = AIComponent(
            name="agent",
            component_type=AIComponentType.AGENT,
            file_path="a.py",
            line_number=10,
        )
        rel = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name="agent",
            target_name="gpt-4",
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _propagate_model_from_relationships([agent], [rel])
        assert out[0].model_name == "gpt-4"

    def test_unrelated_relationship_type_is_ignored(self) -> None:
        agent = AIComponent(
            name="agent",
            component_type=AIComponentType.AGENT,
            file_path="a.py",
            line_number=10,
        )
        rel = ComponentRelationship(
            source_instance_id=agent.instance_id,
            target_instance_id="tool_a.py_10",
            source_name="agent",
            target_name="some-tool",
            relationship_type=RelationshipType.USES_TOOL,
        )
        out = _propagate_model_from_relationships([agent], [rel])
        assert out[0].model_name is None


class TestResolveRelationshipTypes:
    """``_resolve_relationship_types`` must not attribute an owner's type to its ``model_name``."""

    def test_agent_owner_does_not_leak_type_onto_its_model_name(self) -> None:
        """Agent with model_name='gpt-4o' must not register 'gpt-4o' as an AGENT alias."""
        agent = AIComponent(
            name="my_agent",
            component_type=AIComponentType.AGENT,
            file_path="a.py",
            line_number=10,
            model_name="gpt-4o",
        )
        rel = ComponentRelationship(
            source_instance_id="other_a.py_30",
            target_instance_id="",
            source_name="some_caller",
            target_name="gpt-4o",
            source_type=AIComponentType.OTHER,
            target_type=AIComponentType.OTHER,
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _resolve_relationship_types([rel], [agent])
        assert out[0].target_type != AIComponentType.AGENT

    def test_model_owner_contributes_type_alias_for_its_model_name(self) -> None:
        """A MODEL component with model_name='gpt-4o' is a genuine alias → keep the alias."""
        model_comp = AIComponent(
            name="canonical_model",
            component_type=AIComponentType.MODEL,
            file_path="a.py",
            line_number=10,
            model_name="gpt-4o",
        )
        rel = ComponentRelationship(
            source_instance_id="other_a.py_30",
            target_instance_id="",
            source_name="caller",
            target_name="gpt-4o",
            source_type=AIComponentType.OTHER,
            target_type=AIComponentType.OTHER,
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _resolve_relationship_types([rel], [model_comp])
        assert out[0].target_type == AIComponentType.MODEL

    def test_llm_endpoint_owner_contributes_type_alias(self) -> None:
        endpoint = AIComponent(
            name="openai_endpoint",
            component_type=AIComponentType.LLM_ENDPOINT,
            file_path="a.py",
            line_number=10,
            model_name="gpt-4o",
        )
        rel = ComponentRelationship(
            source_instance_id="other_a.py_30",
            target_instance_id="",
            source_name="caller",
            target_name="gpt-4o",
            source_type=AIComponentType.OTHER,
            target_type=AIComponentType.OTHER,
            relationship_type=RelationshipType.USES_MODEL,
        )
        out = _resolve_relationship_types([rel], [endpoint])
        assert out[0].target_type == AIComponentType.LLM_ENDPOINT

    def test_component_name_still_resolves(self) -> None:
        """Primary name-to-type mapping must still work for non-model owners."""
        agent = AIComponent(
            name="my_agent",
            component_type=AIComponentType.AGENT,
            file_path="a.py",
            line_number=10,
        )
        rel = ComponentRelationship(
            source_instance_id="caller_a.py_30",
            target_instance_id=agent.instance_id,
            source_name="caller",
            target_name="my_agent",
            source_type=AIComponentType.OTHER,
            target_type=AIComponentType.OTHER,
            relationship_type=RelationshipType.CUSTOM,
        )
        out = _resolve_relationship_types([rel], [agent])
        assert out[0].target_type == AIComponentType.AGENT
