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

"""Tests for middleware post-processing gates."""

from __future__ import annotations

from pathlib import Path

from aibom.agentic.agent import _Relationship
from aibom.agentic.middleware import (
    AIBOMScannerMiddleware,
    _cap_confidence_if_unresolved,
    _drop_env_placeholder_identifiers,
    _is_class_name_not_model_id,
    _is_negating_justification,
    _is_sentinel_name,
    _reject_class_name_models,
    _remove_unresolved_embedders,
    _rewrite_if_ungrounded_endpoint,
    _sanitize_metadata,
    _should_protect_deterministic_model_removal,
    _should_reject_tool_from_helm,
)
from aibom.models.enums import AIComponentType, DetectionSource, RelationshipType
from aibom.models.scan import AIComponent, ComponentRelationship


class TestRelationshipSchema:
    """Fix 1: _Relationship Pydantic model accepts source_type/target_type."""

    def test_source_type_and_target_type_default_empty(self):
        rel = _Relationship(
            source_name="MyAgent",
            target_name="gpt-4o",
            relationship_type="USES_MODEL",
        )
        assert rel.source_type == ""
        assert rel.target_type == ""

    def test_source_type_and_target_type_set(self):
        rel = _Relationship(
            source_name="MyAgent",
            target_name="gpt-4o",
            relationship_type="USES_MODEL",
            source_type="agent",
            target_type="model",
        )
        assert rel.source_type == "agent"
        assert rel.target_type == "model"

    def test_model_dump_includes_type_fields(self):
        rel = _Relationship(
            source_name="Router",
            target_name="gpt-4o",
            relationship_type="USES_MODEL",
            source_type="agent",
            target_type="model",
        )
        data = rel.model_dump()
        assert data["source_type"] == "agent"
        assert data["target_type"] == "model"


def _comp(name: str, comp_type: AIComponentType, **kwargs) -> AIComponent:
    return AIComponent(
        name=name,
        component_type=comp_type,
        file_path="/test.py",
        line_number=1,
        **kwargs,
    )


class TestIsClassName:
    def test_pascal_case(self):
        assert _is_class_name_not_model_id("MyLargeLanguageModel") is True

    def test_model_id_with_slash(self):
        assert _is_class_name_not_model_id("meta-llama/Llama-3-70B") is False

    def test_model_id_with_dash(self):
        assert _is_class_name_not_model_id("gpt-4o") is False

    def test_model_id_with_dots(self):
        assert _is_class_name_not_model_id("text-embedding-ada-002") is False

    def test_single_word_lowercase(self):
        assert _is_class_name_not_model_id("agent") is False

    def test_camel_case_wrapper_class(self):
        assert _is_class_name_not_model_id("ChatCompletionClient") is True


class TestRejectClassNameModels:
    def test_removes_class_name_models(self):
        comps = [
            _comp("MyLargeLanguageModel", AIComponentType.MODEL),
            _comp("gpt-4o", AIComponentType.MODEL),
        ]
        result = _reject_class_name_models(comps)
        assert len(result) == 1
        assert result[0].name == "gpt-4o"

    def test_keeps_non_model_class_names(self):
        comps = [
            _comp("ChatCompletionClient", AIComponentType.AGENT),
        ]
        result = _reject_class_name_models(comps)
        assert len(result) == 1


class TestRemoveUnresolvedEmbedders:
    def test_removes_embedder_without_model(self):
        comps = [_comp("SyntheticEmbedderWrapper", AIComponentType.EMBEDDING)]
        result = _remove_unresolved_embedders(comps, [])
        assert len(result) == 0

    def test_keeps_embedder_with_model_name(self):
        comps = [
            _comp(
                "SyntheticEmbedderWrapper",
                AIComponentType.EMBEDDING,
                model_name="text-embedding-ada-002",
            )
        ]
        result = _remove_unresolved_embedders(comps, [])
        assert len(result) == 1

    def test_keeps_embedder_with_relationship(self):
        comps = [_comp("SyntheticEmbedderWrapper", AIComponentType.EMBEDDING)]
        rels = [
            ComponentRelationship(
                source_instance_id="",
                target_instance_id="",
                source_name="SyntheticEmbedderWrapper",
                target_name="text-embedding-ada-002",
                relationship_type=RelationshipType.USES_EMBEDDING,
            )
        ]
        result = _remove_unresolved_embedders(comps, rels)
        assert len(result) == 1


class TestDropDependencyMcpClientRels:
    def test_drops_dependency_typed_uses_mcp_client(self):
        rels = [
            ComponentRelationship(
                source_instance_id="",
                target_instance_id="",
                source_name="my-mcp-server",
                target_name="mcp",
                source_type=AIComponentType.MCP_SERVER,
                target_type=AIComponentType.DEPENDENCY,
                relationship_type=RelationshipType.USES_MCP_CLIENT,
            ),
        ]
        result = AIBOMScannerMiddleware._drop_dependency_mcp_client_rels(rels)
        assert len(result) == 0

    def test_keeps_real_uses_mcp_client(self):
        rels = [
            ComponentRelationship(
                source_instance_id="",
                target_instance_id="",
                source_name="AgenticRouter",
                target_name="MCPClient",
                source_type=AIComponentType.AGENT,
                target_type=AIComponentType.MCP_CLIENT,
                relationship_type=RelationshipType.USES_MCP_CLIENT,
            ),
        ]
        result = AIBOMScannerMiddleware._drop_dependency_mcp_client_rels(rels)
        assert len(result) == 1

    def test_keeps_other_relationship_types(self):
        rels = [
            ComponentRelationship(
                source_instance_id="",
                target_instance_id="",
                source_name="AgentA",
                target_name="gpt-4o",
                relationship_type=RelationshipType.USES_MODEL,
            ),
        ]
        result = AIBOMScannerMiddleware._drop_dependency_mcp_client_rels(rels)
        assert len(result) == 1


class TestDropDepToDepUsesModel:
    """Middleware gate: USES_MODEL with both sides as dependency is spurious."""

    def test_drops_dep_to_dep_uses_model(self):
        rels = [
            ComponentRelationship(
                source_instance_id="",
                target_instance_id="",
                source_name="openai-agents",
                target_name="openai",
                source_type=AIComponentType.DEPENDENCY,
                target_type=AIComponentType.DEPENDENCY,
                relationship_type=RelationshipType.USES_MODEL,
            ),
        ]
        result = AIBOMScannerMiddleware._drop_dep_to_dep_uses_model(rels)
        assert len(result) == 0

    def test_keeps_agent_to_model_uses_model(self):
        rels = [
            ComponentRelationship(
                source_instance_id="",
                target_instance_id="",
                source_name="MyAgent",
                target_name="gpt-4o",
                source_type=AIComponentType.AGENT,
                target_type=AIComponentType.MODEL,
                relationship_type=RelationshipType.USES_MODEL,
            ),
        ]
        result = AIBOMScannerMiddleware._drop_dep_to_dep_uses_model(rels)
        assert len(result) == 1

    def test_keeps_dep_to_model_uses_model(self):
        rels = [
            ComponentRelationship(
                source_instance_id="",
                target_instance_id="",
                source_name="langchain-openai",
                target_name="gpt-4o",
                source_type=AIComponentType.DEPENDENCY,
                target_type=AIComponentType.MODEL,
                relationship_type=RelationshipType.USES_MODEL,
            ),
        ]
        result = AIBOMScannerMiddleware._drop_dep_to_dep_uses_model(rels)
        assert len(result) == 1

    def test_keeps_dep_to_dep_other_relationship(self):
        rels = [
            ComponentRelationship(
                source_instance_id="",
                target_instance_id="",
                source_name="pkg-a",
                target_name="pkg-b",
                source_type=AIComponentType.DEPENDENCY,
                target_type=AIComponentType.DEPENDENCY,
                relationship_type=RelationshipType.USES_TOOL,
            ),
        ]
        result = AIBOMScannerMiddleware._drop_dep_to_dep_uses_model(rels)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Issue 1: sentinel / self-contradicting LLM output rejection
# ---------------------------------------------------------------------------


class TestIsSentinelName:
    """Detector for LLM placeholder / sentinel names."""

    def test_real_names_not_sentinel(self):
        for name in [
            "gpt-4o",
            "text-embedding-3-large",
            "AgenticRouter",
            "my-mcp-server",
            "claude-3-5-sonnet",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]:
            assert _is_sentinel_name(name) is False, name

    def test_sentinel_names_rejected(self):
        for name in [
            # The exact rogue value observed in e2e logs.
            "USES_MODEL placeholder skipped",
            "placeholder",
            "placeholder - skipped",
            "None found",
            "no match",
            "no component",
            "no relationship",
            "no suitable model",
            "Nothing to add",
            "N/A",
            "not applicable",
            "Unknown",
            "omitted",
            "",
            "   ",
        ]:
            assert _is_sentinel_name(name) is True, name

    def test_non_string_rejected(self):
        assert _is_sentinel_name(None) is True
        assert _is_sentinel_name(42) is True
        assert _is_sentinel_name([]) is True


class TestIsNegatingJustification:
    """Detector for self-contradicting LLM justifications."""

    def test_affirmative_justifications_kept(self):
        for just in [
            "Observed OpenAI().chat.completions.create() call using gpt-4o.",
            "The AgenticRouter class extends LangChain's AgentExecutor.",
            "Evidence in handler.py line 42.",
            "Matched framework parent class and ReAct loop.",
        ]:
            assert _is_negating_justification(just) is False, just

    def test_negating_justifications_flagged(self):
        for just in [
            "No suitable evidence of a gpt-4o call was found.",
            "Not a real component.",
            "None of the candidates match.",
            "Cannot verify the class inheritance.",
            "Unable to confirm agent behavior.",
            "Nothing to add at this location.",
            "Insufficient evidence for reclassification.",
            "Placeholder entry — should be omitted.",
            "Skipped because evidence is weak.",
            "N/A at this location.",
        ]:
            assert _is_negating_justification(just) is True, just

    def test_empty_justification_not_flagged(self):
        # Empty justification is a separate concern (the decision-annotation
        # schema may be enforced elsewhere). This helper only fires on
        # explicit negation.
        assert _is_negating_justification("") is False
        assert _is_negating_justification(None) is False


class TestMiddlewareRejectsSentinelComponents:
    """``_extract_new_components`` must drop sentinel-named entries."""

    def _run(self, items: list[dict]) -> list[AIComponent]:
        mw = AIBOMScannerMiddleware()
        return mw._extract_new_components({"new_components": items})

    def test_drops_rogue_placeholder_tool(self):
        """The exact rogue entry seen in the e2e log must be rejected."""
        out = self._run([
            {
                "name": "USES_MODEL placeholder skipped",
                "component_type": "tool",
                "decision_annotation": {
                    "decision": "added",
                    "justification": "No model call observed.",
                },
            }
        ])
        assert out == []

    def test_drops_negating_justification(self):
        """A real-looking name is rejected when the LLM's justification
        explicitly says nothing should be added."""
        out = self._run([
            {
                "name": "gpt-4o",
                "component_type": "model",
                "decision_annotation": {
                    "decision": "added",
                    "justification": "No suitable gpt-4o call found in this file.",
                },
            }
        ])
        assert out == []

    def test_keeps_real_component_with_affirmative_justification(self):
        out = self._run([
            {
                "name": "gpt-4o",
                "component_type": "model",
                "file_path": "handler.py",
                "line_number": 42,
                "decision_annotation": {
                    "decision": "added",
                    "justification": "Observed in client.chat.completions.create(model=...).",
                },
            }
        ])
        assert len(out) == 1
        assert out[0].name == "gpt-4o"
        assert out[0].component_type == AIComponentType.MODEL


class TestMiddlewareRejectsSentinelRelationships:
    """``_extract_relationships`` must drop sentinel-endpoint relationships."""

    def _run(self, items: list[dict]) -> list[ComponentRelationship]:
        mw = AIBOMScannerMiddleware()
        return mw._extract_relationships({"new_relationships": items})

    def test_drops_sentinel_source_name(self):
        out = self._run([
            {
                "source_name": "placeholder - skipped",
                "target_name": "gpt-4o",
                "relationship_type": "USES_MODEL",
                "source_type": "agent",
                "target_type": "model",
            }
        ])
        assert out == []

    def test_drops_sentinel_target_name(self):
        out = self._run([
            {
                "source_name": "AgenticRouter",
                "target_name": "None found",
                "relationship_type": "USES_MODEL",
                "source_type": "agent",
                "target_type": "model",
            }
        ])
        assert out == []

    def test_drops_negating_justification(self):
        out = self._run([
            {
                "source_name": "AgenticRouter",
                "target_name": "gpt-4o",
                "relationship_type": "USES_MODEL",
                "source_type": "agent",
                "target_type": "model",
                "decision_annotation": {
                    "decision": "added",
                    "justification": "Not able to confirm model binding.",
                },
            }
        ])
        assert out == []

    def test_keeps_real_relationship(self):
        out = self._run([
            {
                "source_name": "AgenticRouter",
                "target_name": "gpt-4o",
                "relationship_type": "USES_MODEL",
                "source_type": "agent",
                "target_type": "model",
                "decision_annotation": {
                    "decision": "added",
                    "justification": "Observed AgenticRouter.llm set to gpt-4o.",
                },
            }
        ])
        assert len(out) == 1
        assert out[0].source_name == "AgenticRouter"
        assert out[0].target_name == "gpt-4o"
        assert out[0].relationship_type == RelationshipType.USES_MODEL


class TestSanitizeMetadata:
    """``_sanitize_metadata`` strips hallucinated keys but keeps schema keys."""

    def test_keeps_allow_list_keys(self):
        cleaned = _sanitize_metadata(
            {
                "env_var": "OPENAI_API_KEY",
                "framework": "langchain",
                "model_family": "gpt-4",
                "service_name": "classifier",
            },
            component_name="gpt-4o",
            component_type="model",
        )
        assert cleaned == {
            "env_var": "OPENAI_API_KEY",
            "framework": "langchain",
            "model_family": "gpt-4",
            "service_name": "classifier",
        }

    def test_strips_unknown_keys(self):
        cleaned = _sanitize_metadata(
            {
                "env_var": "OPENAI_API_KEY",
                "resolution": "inferred from docstring",
                "llm_notes": "likely Azure endpoint",
                "inferred_from": "validation error message",
            },
            component_name="env:OPENAI_API_KEY",
            component_type="llm_endpoint",
        )
        assert cleaned == {"env_var": "OPENAI_API_KEY"}

    def test_non_dict_returns_empty(self):
        assert _sanitize_metadata(None, component_name="x", component_type="y") == {}
        assert _sanitize_metadata("string", component_name="x", component_type="y") == {}
        assert _sanitize_metadata(["list"], component_name="x", component_type="y") == {}

    def test_non_string_keys_skipped(self):
        cleaned = _sanitize_metadata(
            {"env_var": "FOO", 123: "should_drop"},
            component_name="x",
            component_type="model",
        )
        assert cleaned == {"env_var": "FOO"}


class TestRejectToolFromHelm:
    """``_should_reject_tool_from_helm`` distinguishes agent tools from K8s services."""

    def _tool(self, **kwargs) -> AIComponent:
        defaults = {
            "name": "some-tool",
            "component_type": AIComponentType.TOOL,
            "file_path": "",
        }
        defaults.update(kwargs)
        return AIComponent(**defaults)

    def test_rejects_tool_with_framework_helm(self):
        comp = self._tool(framework="helm")
        rejected, reason = _should_reject_tool_from_helm(comp)
        assert rejected is True
        assert "helm" in reason.lower()

    def test_rejects_tool_with_service_name_metadata(self):
        comp = self._tool(metadata={"service_name": "billing-api"})
        rejected, reason = _should_reject_tool_from_helm(comp)
        assert rejected is True
        assert "service_name" in reason

    def test_rejects_tool_with_helm_key_metadata(self):
        comp = self._tool(metadata={"helm_key": "authServer"})
        rejected, _ = _should_reject_tool_from_helm(comp)
        assert rejected is True

    def test_rejects_tool_with_chart_path_metadata(self):
        comp = self._tool(metadata={"chart_path": "charts/foo/values.yaml"})
        rejected, _ = _should_reject_tool_from_helm(comp)
        assert rejected is True

    def test_rejects_tool_with_kubernetes_kind_metadata(self):
        comp = self._tool(metadata={"kubernetes_kind": "Service"})
        rejected, _ = _should_reject_tool_from_helm(comp)
        assert rejected is True

    def test_rejects_tool_with_values_yaml_path(self):
        comp = self._tool(file_path="helm/my-service/values.yaml")
        rejected, reason = _should_reject_tool_from_helm(comp)
        assert rejected is True
        assert "values.yaml" in reason

    def test_rejects_tool_under_charts_directory(self):
        comp = self._tool(file_path="infra/charts/service/templates/deployment.yaml")
        rejected, reason = _should_reject_tool_from_helm(comp)
        assert rejected is True
        assert "charts" in reason.lower()

    def test_keeps_real_tool(self):
        comp = self._tool(
            name="fetch_user",
            file_path="src/agents/tools.py",
            framework="strands",
        )
        rejected, _ = _should_reject_tool_from_helm(comp)
        assert rejected is False

    def test_ignores_non_tool_types(self):
        comp = AIComponent(
            name="gpt-4",
            component_type=AIComponentType.MODEL,
            file_path="values.yaml",
            framework="helm",
        )
        rejected, _ = _should_reject_tool_from_helm(comp)
        assert rejected is False


class TestRewriteIfUngroundedEndpoint:
    """``_rewrite_if_ungrounded_endpoint`` removes hallucinated URLs."""

    def _endpoint(self, name: str, **kwargs) -> AIComponent:
        defaults = {
            "name": name,
            "component_type": AIComponentType.LLM_ENDPOINT,
            "file_path": "",
        }
        defaults.update(kwargs)
        return AIComponent(**defaults)

    def test_rewrites_url_not_in_live_code(self, tmp_path: Path):
        source = tmp_path / "handler.py"
        source.write_text(
            "import os\n"
            "endpoint = os.environ['CODEX_ENDPOINT']\n"
        )
        comp = self._endpoint(
            "https://api.openai.azure.com",
            metadata={"env_var": "CODEX_ENDPOINT"},
            heuristic_confidence=1.0,
        )
        out = _rewrite_if_ungrounded_endpoint(comp, allowed_roots=[str(tmp_path)])
        assert out.name == "env:CODEX_ENDPOINT"
        assert out.model_name is None
        assert out.heuristic_confidence <= 0.5

    def test_preserves_url_grounded_in_live_code(self, tmp_path: Path):
        source = tmp_path / "config.py"
        source.write_text(
            'ENDPOINT = "https://api.openai.com/v1"\n'
        )
        comp = self._endpoint(
            "https://api.openai.com/v1",
            metadata={"env_var": "OPENAI_ENDPOINT"},
            heuristic_confidence=0.9,
        )
        out = _rewrite_if_ungrounded_endpoint(comp, allowed_roots=[str(tmp_path)])
        assert out.name == "https://api.openai.com/v1"
        assert out.heuristic_confidence == 0.9

    def test_url_in_docstring_not_grounded(self, tmp_path: Path):
        source = tmp_path / "handler.py"
        source.write_text(
            'def configure():\n'
            '    """Example URL: https://api.openai.azure.com/v1"""\n'
            "    return os.environ['CODEX_ENDPOINT']\n"
        )
        comp = self._endpoint(
            "https://api.openai.azure.com/v1",
            metadata={"env_var": "CODEX_ENDPOINT"},
        )
        out = _rewrite_if_ungrounded_endpoint(comp, allowed_roots=[str(tmp_path)])
        assert out.name == "env:CODEX_ENDPOINT"

    def test_url_in_line_comment_not_grounded(self, tmp_path: Path):
        source = tmp_path / "handler.go"
        source.write_text(
            '// see https://api.openai.azure.com/v1 for details\n'
            'var endpoint = os.Getenv("CODEX_ENDPOINT")\n'
        )
        comp = self._endpoint(
            "https://api.openai.azure.com/v1",
            metadata={"env_var": "CODEX_ENDPOINT"},
        )
        out = _rewrite_if_ungrounded_endpoint(comp, allowed_roots=[str(tmp_path)])
        assert out.name == "env:CODEX_ENDPOINT"

    def test_skip_when_no_env_var_metadata(self, tmp_path: Path):
        comp = self._endpoint(
            "https://fabricated.example.com",
            metadata={"framework": "openai"},
        )
        out = _rewrite_if_ungrounded_endpoint(comp, allowed_roots=[str(tmp_path)])
        assert out.name == "https://fabricated.example.com"

    def test_skip_non_url_name(self, tmp_path: Path):
        comp = self._endpoint(
            "env:FOO",
            metadata={"env_var": "FOO"},
        )
        out = _rewrite_if_ungrounded_endpoint(comp, allowed_roots=[str(tmp_path)])
        assert out.name == "env:FOO"

    def test_skip_non_endpoint_type(self, tmp_path: Path):
        comp = AIComponent(
            name="https://some.url",
            component_type=AIComponentType.MODEL,
            file_path="",
            metadata={"env_var": "FOO"},
        )
        out = _rewrite_if_ungrounded_endpoint(comp, allowed_roots=[str(tmp_path)])
        assert out.name == "https://some.url"

    def test_uses_dockerfile_env_metadata_key(self, tmp_path: Path):
        source = tmp_path / "Dockerfile"
        source.write_text(
            "FROM python:3.11\n"
            "ENV CODEX_ENDPOINT=https://api.openai.com/v1\n"
        )
        comp = self._endpoint(
            "https://fabricated.example.com/v1",
            metadata={"env": "CODEX_ENDPOINT"},
        )
        out = _rewrite_if_ungrounded_endpoint(comp, allowed_roots=[str(tmp_path)])
        assert out.name == "env:CODEX_ENDPOINT"


class TestCapConfidenceIfUnresolved:
    """``_cap_confidence_if_unresolved`` limits unresolved env components."""

    def test_caps_high_confidence_for_unresolved_env_model(self):
        comp = AIComponent(
            name="env:ANTHROPIC_MODEL",
            component_type=AIComponentType.MODEL,
            file_path="",
            metadata={"env_var": "ANTHROPIC_MODEL"},
            heuristic_confidence=1.0,
        )
        out = _cap_confidence_if_unresolved(comp)
        assert out.heuristic_confidence == 0.5

    def test_preserves_low_confidence_for_unresolved(self):
        comp = AIComponent(
            name="env:FOO",
            component_type=AIComponentType.MODEL,
            file_path="",
            metadata={"env_var": "FOO"},
            heuristic_confidence=0.3,
        )
        out = _cap_confidence_if_unresolved(comp)
        assert out.heuristic_confidence == 0.3

    def test_keeps_confidence_for_resolved_model(self):
        comp = AIComponent(
            name="claude-sonnet-4-20250514",
            component_type=AIComponentType.MODEL,
            file_path="",
            model_name="claude-sonnet-4-20250514",
            metadata={"env_var": "ANTHROPIC_MODEL"},
            heuristic_confidence=1.0,
        )
        out = _cap_confidence_if_unresolved(comp)
        assert out.heuristic_confidence == 1.0

    def test_keeps_confidence_for_resolved_url(self):
        comp = AIComponent(
            name="https://api.openai.com/v1",
            component_type=AIComponentType.LLM_ENDPOINT,
            file_path="",
            metadata={"env_var": "OPENAI_ENDPOINT"},
            heuristic_confidence=1.0,
        )
        out = _cap_confidence_if_unresolved(comp)
        assert out.heuristic_confidence == 1.0

    def test_no_env_var_no_cap(self):
        comp = AIComponent(
            name="gpt-4",
            component_type=AIComponentType.MODEL,
            file_path="",
            model_name="gpt-4",
            heuristic_confidence=1.0,
        )
        out = _cap_confidence_if_unresolved(comp)
        assert out.heuristic_confidence == 1.0

    def test_placeholder_model_name_still_unresolved(self):
        comp = AIComponent(
            name="env:MODEL_NAME",
            component_type=AIComponentType.MODEL,
            file_path="",
            model_name="${MODEL_NAME}",
            metadata={"env_var": "MODEL_NAME"},
            heuristic_confidence=1.0,
        )
        out = _cap_confidence_if_unresolved(comp)
        assert out.heuristic_confidence == 0.5


class TestMiddlewareRejectsHelmTools:
    """End-to-end: ``_extract_new_components`` rejects Helm/K8s tools."""

    def _run(self, items: list[dict]) -> list[AIComponent]:
        mw = AIBOMScannerMiddleware()
        return mw._extract_new_components({"new_components": items})

    def test_rejects_kebab_service_name_shaped_tool(self):
        out = self._run([
            {
                "name": "billing-api",
                "component_type": "tool",
                "file_path": "helm/my-service/values.yaml",
                "framework": "helm",
                "metadata": {
                    "service_name": "billing-api",
                    "helm_key": "billing-api",
                },
                "decision_annotation": {
                    "decision": "added",
                    "justification": "Found billing-api key in values.yaml.",
                },
            }
        ])
        assert out == []

    def test_rejects_camelcase_helm_key_shaped_tool(self):
        out = self._run([
            {
                "name": "authServer",
                "component_type": "tool",
                "file_path": "infra/charts/my-service/values.yaml",
                "metadata": {"helm_key": "authServer", "kubernetes_kind": "Service"},
                "decision_annotation": {
                    "decision": "added",
                    "justification": "authServer Service referenced in values.yaml.",
                },
            }
        ])
        assert out == []

    def test_keeps_real_tool_from_source(self):
        out = self._run([
            {
                "name": "fetch_user",
                "component_type": "tool",
                "file_path": "src/agents/tools.py",
                "line_number": 12,
                "framework": "strands",
                "decision_annotation": {
                    "decision": "added",
                    "justification": "Decorated with @tool and registered in agent.",
                },
            }
        ])
        assert len(out) == 1
        assert out[0].name == "fetch_user"
        assert out[0].component_type == AIComponentType.TOOL


class TestMiddlewareRejectsUngroundedEndpoint:
    """End-to-end: ``_extract_new_components`` rewrites hallucinated URLs."""

    def test_rewrites_hallucinated_azure_endpoint(self, tmp_path: Path):
        source = tmp_path / "client.py"
        source.write_text(
            "import os\n"
            "endpoint = os.environ['CODEX_ENDPOINT']\n"
        )
        mw = AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)])
        out = mw._extract_new_components({
            "new_components": [
                {
                    "name": "https://api.openai.azure.com",
                    "component_type": "llm_endpoint",
                    "file_path": "client.py",
                    "line_number": 2,
                    "metadata": {"env_var": "CODEX_ENDPOINT"},
                    "decision_annotation": {
                        "decision": "added",
                        "justification": "Endpoint resolved from env var.",
                    },
                }
            ]
        })
        assert len(out) == 1
        comp = out[0]
        assert comp.name == "env:CODEX_ENDPOINT"
        assert comp.heuristic_confidence <= 0.5

    def test_strips_hallucinated_metadata_keys(self, tmp_path: Path):
        mw = AIBOMScannerMiddleware(allowed_roots=[str(tmp_path)])
        out = mw._extract_new_components({
            "new_components": [
                {
                    "name": "gpt-4o",
                    "component_type": "model",
                    "file_path": "client.py",
                    "line_number": 5,
                    "metadata": {
                        "env_var": "OPENAI_MODEL",
                        "resolution": "inferred from docstring",
                        "llm_notes": "likely default",
                    },
                    "decision_annotation": {
                        "decision": "added",
                        "justification": "Literal gpt-4o observed.",
                    },
                }
            ]
        })
        assert len(out) == 1
        comp = out[0]
        assert "resolution" not in comp.metadata
        assert "llm_notes" not in comp.metadata
        assert comp.metadata.get("env_var") == "OPENAI_MODEL"


def _det_model(name: str, **kwargs) -> AIComponent:
    """Build a deterministically-scanned MODEL component for protection tests."""
    defaults: dict = {
        "name": name,
        "component_type": AIComponentType.MODEL,
        "file_path": "charts/svc/values-prod.yaml",
        "line_number": 10,
        "detection_source": DetectionSource.CONFIG_FILE,
    }
    defaults.update(kwargs)
    return AIComponent(**defaults)


class TestShouldProtectDeterministicModelRemoval:
    """``_should_protect_deterministic_model_removal`` prevents agentic
    erasure of concrete private/deployment model strings sourced from
    deterministic scanners when the agent cites a registry miss."""

    def test_protects_private_vllm_model_name(self):
        comp = _det_model("org/custom-model/stable")
        protect, why = _should_protect_deterministic_model_removal(
            comp,
            "Model registry lookup failed and the config shows a "
            "private/custom deployment-style name with insufficient evidence.",
        )
        assert protect is True
        assert "concrete deterministic model" in why

    def test_protects_azure_deployment_alias(self):
        comp = _det_model("prod-chat-gpt4o-westus")
        protect, why = _should_protect_deterministic_model_removal(
            comp,
            "Azure OpenAI deployment name alias does not resolve as a "
            "concrete model identifier; backing model cannot be confirmed.",
        )
        assert protect is True
        assert "does not resolve" in why or "deployment" in why

    def test_protects_versioned_bedrock_id(self):
        comp = _det_model("anthropic.claude-3-5-haiku-20241022-v1:0")
        protect, _ = _should_protect_deterministic_model_removal(
            comp,
            "Registry lookup did not resolve this identifier to a canonical public model.",
        )
        assert protect is True

    def test_does_not_protect_camelcase_class_name(self):
        comp = _det_model("OpenAILLM")
        protect, _ = _should_protect_deterministic_model_removal(
            comp,
            "Not in registry; looks like a Python wrapper class.",
        )
        assert protect is False

    def test_does_not_protect_env_placeholder(self):
        comp = _det_model("env:ANTHROPIC_MODEL")
        protect, _ = _should_protect_deterministic_model_removal(
            comp,
            "Unresolved env var not registered in the model registry.",
        )
        assert protect is False

    def test_does_not_protect_agentic_sourced_model(self):
        comp = _det_model(
            "gpt-4-proposed",
            detection_source=DetectionSource.AGENTIC,
        )
        protect, _ = _should_protect_deterministic_model_removal(
            comp,
            "Does not resolve in registry; remove.",
        )
        assert protect is False

    def test_does_not_protect_non_model_types(self):
        comp = _det_model(
            "org/custom-model/stable",
            component_type=AIComponentType.TOOL,
        )
        protect, _ = _should_protect_deterministic_model_removal(
            comp,
            "Does not resolve in registry.",
        )
        assert protect is False

    def test_does_not_protect_when_reason_has_no_registry_marker(self):
        comp = _det_model("org/custom-model/stable")
        protect, _ = _should_protect_deterministic_model_removal(
            comp,
            "This is a test fixture string, not a live model.",
        )
        assert protect is False

    def test_protects_regardless_of_scanner_flavor(self):
        for source in (
            DetectionSource.CODE_ANALYSIS,
            DetectionSource.CONFIG_FILE,
            DetectionSource.DEPENDENCY_MANIFEST,
            DetectionSource.KB_ENRICHMENT,
            DetectionSource.BASE_CLASS_RULE,
        ):
            comp = _det_model("org/custom-model/stable", detection_source=source)
            protect, _ = _should_protect_deterministic_model_removal(
                comp,
                "private custom name does not resolve in registry",
            )
            assert protect is True, f"must protect for source={source.value}"


class TestMiddlewareProtectsDeterministicModelsFromRemoval:
    """End-to-end: ``apply_enrichments_from_dict`` drops agent removals that
    would erase deterministic concrete model strings citing a registry miss."""

    def _run(
        self,
        existing: list[AIComponent],
        remove_items: list[dict],
    ) -> list[AIComponent]:
        mw = AIBOMScannerMiddleware()
        return mw.apply_enrichments_from_dict(
            existing,
            {"remove_components": remove_items},
        )

    def test_keeps_private_vllm_model_agent_tried_to_remove(self):
        iid = "org/custom-model/stable_charts/foo/values.yaml_75"
        comp = _det_model(
            "org/custom-model/stable",
            file_path="charts/foo/values.yaml",
            line_number=75,
        )
        comp = comp.model_copy(update={"instance_id": iid})
        out = self._run(
            [comp],
            [
                {
                    "instance_id": iid,
                    "reason": "Configured VLLM model name does not resolve in "
                              "the model registry and may be private.",
                }
            ],
        )
        assert len(out) == 1
        assert out[0].name == "org/custom-model/stable"

    def test_keeps_azure_deployment_alias_agent_tried_to_remove(self):
        iid = "prod-chat-gpt4o-westus_charts/bar/values-prod.yaml_59"
        comp = _det_model(
            "prod-chat-gpt4o-westus",
            file_path="charts/bar/values-prod.yaml",
            line_number=59,
        )
        comp = comp.model_copy(update={"instance_id": iid})
        out = self._run(
            [comp],
            [
                {
                    "instance_id": iid,
                    "reason": "Azure OpenAI deployment name alias does not "
                              "resolve as a concrete model identifier.",
                }
            ],
        )
        assert len(out) == 1
        assert out[0].name == "prod-chat-gpt4o-westus"

    def test_still_allows_removal_of_camelcase_wrapper_class(self):
        iid = "OpenAIClient_src/client.py_5"
        comp = AIComponent(
            name="OpenAIClient",
            component_type=AIComponentType.MODEL,
            file_path="src/client.py",
            line_number=5,
            detection_source=DetectionSource.CODE_ANALYSIS,
        )
        comp = comp.model_copy(update={"instance_id": iid})
        out = self._run(
            [comp],
            [
                {
                    "instance_id": iid,
                    "reason": "CamelCase wrapper class, not a model id; "
                              "does not resolve in registry.",
                }
            ],
        )
        assert out == []

    def test_still_allows_removal_with_unrelated_reason(self):
        """If the agent removes for a non-registry reason (e.g. 'test
        fixture', 'false positive'), the guard should not protect."""
        iid = "org/custom-model/stable_tests/fixture.yaml_5"
        comp = _det_model(
            "org/custom-model/stable",
            file_path="tests/fixture.yaml",
            line_number=5,
        )
        comp = comp.model_copy(update={"instance_id": iid})
        out = self._run(
            [comp],
            [
                {
                    "instance_id": iid,
                    "reason": "This appears in a test fixture file, not "
                              "production code; false positive.",
                }
            ],
        )
        assert out == []


class TestDropEnvPlaceholderIdentifiers:
    """``_drop_env_placeholder_identifiers`` removes ``env:<VAR>``-named
    MODEL / MODEL_ENDPOINT / LLM_ENDPOINT components so unresolved env
    placeholders cannot leak into the final BOM where the ``name`` is
    supposed to be a concrete model id or URL. SECRET is preserved
    because the env-var name IS its primary identifier."""

    def _comp(
        self,
        name: str,
        component_type: AIComponentType,
        *,
        model_name: str | None = None,
        file_path: str = "src/app.py",
        line_number: int = 1,
    ) -> AIComponent:
        return AIComponent(
            name=name,
            component_type=component_type,
            file_path=file_path,
            line_number=line_number,
            detection_source=DetectionSource.CODE_ANALYSIS,
            model_name=model_name,
        )

    def test_drops_env_prefixed_model(self):
        comp = self._comp("env:CODEX_MODEL", AIComponentType.MODEL)
        assert _drop_env_placeholder_identifiers([comp]) == []

    def test_drops_env_prefixed_model_endpoint(self):
        comp = self._comp("env:MODEL_ENDPOINT_URL", AIComponentType.MODEL_ENDPOINT)
        assert _drop_env_placeholder_identifiers([comp]) == []

    def test_drops_env_prefixed_llm_endpoint(self):
        comp = self._comp("env:CODEX_ENDPOINT", AIComponentType.LLM_ENDPOINT)
        assert _drop_env_placeholder_identifiers([comp]) == []

    def test_keeps_env_prefixed_secret(self):
        """Secret var name IS the primary identifier — BOM consumer needs
        to know the app reads ``OPENAI_API_KEY``."""
        comp = self._comp("env:OPENAI_API_KEY", AIComponentType.SECRET)
        assert _drop_env_placeholder_identifiers([comp]) == [comp]

    def test_keeps_concrete_model(self):
        comp = self._comp("gpt-4o", AIComponentType.MODEL)
        assert _drop_env_placeholder_identifiers([comp]) == [comp]

    def test_keeps_concrete_model_endpoint(self):
        comp = self._comp(
            "https://api.example.com/v1/chat", AIComponentType.MODEL_ENDPOINT
        )
        assert _drop_env_placeholder_identifiers([comp]) == [comp]

    def test_keeps_concrete_llm_endpoint(self):
        comp = self._comp(
            "https://api.example.com/v1/chat", AIComponentType.LLM_ENDPOINT
        )
        assert _drop_env_placeholder_identifiers([comp]) == [comp]

    def test_keeps_component_with_resolved_model_name(self):
        """If the scanner resolved ``model_name`` to a concrete value on a
        later pass — even though ``name`` still carries the ``env:`` shape
        — the component is preserved."""
        comp = self._comp(
            "env:CODEX_MODEL",
            AIComponentType.MODEL,
            model_name="gpt-4o",
        )
        assert _drop_env_placeholder_identifiers([comp]) == [comp]

    def test_drops_component_when_model_name_is_also_placeholder(self):
        comp = self._comp(
            "env:CODEX_MODEL",
            AIComponentType.MODEL,
            model_name="env:CODEX_MODEL",
        )
        assert _drop_env_placeholder_identifiers([comp]) == []

    def test_keeps_other_types_with_env_name(self):
        """Only identifier types (model, *_endpoint) are targeted; every
        other type with an ``env:`` name passes through unchanged."""
        kept_types = [
            AIComponentType.TOOL,
            AIComponentType.AGENT,
            AIComponentType.DEPENDENCY,
            AIComponentType.VECTOR_STORE,
            AIComponentType.EMBEDDING,
            AIComponentType.MCP_SERVER,
            AIComponentType.OBSERVABILITY,
            AIComponentType.GUARDRAIL,
            AIComponentType.SECRET,
        ]
        comps = [self._comp(f"env:SOMETHING_{t.value}", t) for t in kept_types]
        assert _drop_env_placeholder_identifiers(comps) == comps

    def test_mixed_batch_drops_only_identifier_env_placeholders(self):
        keep_concrete = self._comp("gpt-4o", AIComponentType.MODEL)
        drop_llm_ep = self._comp("env:CODEX_ENDPOINT", AIComponentType.LLM_ENDPOINT)
        keep_secret = self._comp("env:OPENAI_API_KEY", AIComponentType.SECRET)
        drop_model_ep = self._comp(
            "env:MODEL_ENDPOINT_URL", AIComponentType.MODEL_ENDPOINT
        )
        drop_model = self._comp("env:CODEX_MODEL", AIComponentType.MODEL)
        keep_tool = self._comp("env:TOOL_CFG", AIComponentType.TOOL)
        out = _drop_env_placeholder_identifiers(
            [
                keep_concrete,
                drop_llm_ep,
                keep_secret,
                drop_model_ep,
                drop_model,
                keep_tool,
            ]
        )
        names = [c.name for c in out]
        assert names == ["gpt-4o", "env:OPENAI_API_KEY", "env:TOOL_CFG"]
