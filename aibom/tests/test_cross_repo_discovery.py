# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-repo relationship discovery fixes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aibom.models import AIComponent, AIComponentType
from aibom.models.enums import CrossRepoLinkType, RelationshipType
from aibom.models.scan import ComponentRelationship, CrossRepoLink, RepoOccurrence
from aibom.scanners.deployment_detector import (
    DeploymentDetector,
    _build_helm_env_var_map,
    _resolve_helm_env_var,
)

from .conftest import run_scanner


class TestHelmEnvVarMap:
    """Fix 1: Verify Helm template env var resolution."""

    def test_parses_deployment_template_env_mapping(self, tmp_path: Path) -> None:
        chart = tmp_path / "my-chart"
        chart.mkdir()
        (chart / "Chart.yaml").write_text("name: my-chart\nversion: 0.1.0\n")

        templates = chart / "templates"
        templates.mkdir()
        (templates / "deployment.yaml").write_text(
            """apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    spec:
      containers:
        - name: app
          env:
            - name: PROD_LLM_ENDPOINT
              value: {{ .Values.config.llm.ENDPOINT }}
            - name: VECTOR_DB_URL
              value: {{ .Values.config.vectordb.URL }}
            - name: STATIC_VALUE
              value: "hardcoded"
"""
        )

        result = _build_helm_env_var_map(chart)
        assert result["config.llm.ENDPOINT"] == "PROD_LLM_ENDPOINT"
        assert result["config.vectordb.URL"] == "VECTOR_DB_URL"
        assert "STATIC_VALUE" not in result.values() or "hardcoded" not in result

    def test_fallback_to_leaf_when_no_templates(self, tmp_path: Path) -> None:
        chart = tmp_path / "bare-chart"
        chart.mkdir()
        (chart / "Chart.yaml").write_text("name: bare\n")

        resolved = _resolve_helm_env_var(
            chart / "values.yaml", "config.llm.ENDPOINT", "ENDPOINT",
        )
        assert resolved == "ENDPOINT"

    def test_resolve_uses_template_mapping(self, tmp_path: Path) -> None:
        chart = tmp_path / "svc-chart"
        chart.mkdir()
        (chart / "Chart.yaml").write_text("name: svc\n")

        templates = chart / "templates"
        templates.mkdir()
        (templates / "deploy.yaml").write_text(
            """env:
  - name: MY_LLM_ENDPOINT
    value: {{ .Values.config.llm.ENDPOINT }}
"""
        )

        values = chart / "values.yaml"
        values.write_text("config:\n  llm:\n    ENDPOINT: https://llm.example.com\n")

        resolved = _resolve_helm_env_var(
            values, "config.llm.ENDPOINT", "ENDPOINT",
        )
        assert resolved == "MY_LLM_ENDPOINT"

    def test_emitted_component_has_resolved_env_var(self, tmp_path: Path) -> None:
        chart = tmp_path / "deploy-chart"
        chart.mkdir()
        (chart / "Chart.yaml").write_text("name: deploy\n")

        templates = chart / "templates"
        templates.mkdir()
        (templates / "deployment.yaml").write_text(
            """env:
  - name: VECTOR_STORE_URL
    value: {{ .Values.env.vector.ENDPOINT }}
"""
        )

        values_content = """\
env:
  vector:
    ENDPOINT: https://vector.internal:8080
"""
        comps, _ = run_scanner(
            DeploymentDetector, chart,
            {"values.yaml": values_content},
        )
        eps = [
            c for c in comps
            if c.component_type in (
                AIComponentType.LLM_ENDPOINT,
                AIComponentType.MODEL_ENDPOINT,
                AIComponentType.VECTOR_STORE,
            )
        ]
        assert eps, f"No endpoint components found. Components: {[c.name for c in comps]}"
        ep = eps[0]
        assert ep.metadata.get("env_var") == "VECTOR_STORE_URL"
        assert ep.metadata.get("endpoint_url") == "https://vector.internal:8080"


class TestCoordinatorRepoAttribution:
    """Fix 2: Verify source_repo/target_repo on ComponentRelationship."""

    def test_component_relationship_stores_repo_fields(self) -> None:
        rel = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name="AgentRouter",
            target_name="gpt-4o",
            relationship_type=RelationshipType.USES_MODEL,
            source_repo="/repos/app-services",
            target_repo="/repos/infra-ops",
        )
        assert rel.source_repo == "/repos/app-services"
        assert rel.target_repo == "/repos/infra-ops"

    def test_repo_fields_default_empty(self) -> None:
        rel = ComponentRelationship(
            source_instance_id="",
            target_instance_id="",
            source_name="A",
            target_name="B",
        )
        assert rel.source_repo == ""
        assert rel.target_repo == ""


class TestResolveRepoRef:
    """Fix 2: Verify _resolve_repo_ref resolves LLM repo names to scan paths."""

    def test_exact_match(self) -> None:
        from aibom.cli import _resolve_repo_ref

        paths = ["/repos/alpha", "/repos/beta"]
        assert _resolve_repo_ref("/repos/alpha", paths) == "/repos/alpha"

    def test_suffix_match(self) -> None:
        from aibom.cli import _resolve_repo_ref

        paths = ["/home/user/repos/alpha", "/home/user/repos/beta"]
        assert _resolve_repo_ref("alpha", paths) == "/home/user/repos/alpha"

    def test_substring_match(self) -> None:
        from aibom.cli import _resolve_repo_ref

        paths = ["/home/user/repos/app-services", "/home/user/repos/infra-ops"]
        assert _resolve_repo_ref("infra", paths) == "/home/user/repos/infra-ops"

    def test_empty_ref(self) -> None:
        from aibom.cli import _resolve_repo_ref

        paths = ["/repos/a"]
        assert _resolve_repo_ref("", paths) == ""

    def test_no_match(self) -> None:
        from aibom.cli import _resolve_repo_ref

        paths = ["/repos/alpha"]
        assert _resolve_repo_ref("nonexistent", paths) == ""


class TestSharedEndpointLinks:
    """Fix 4: Verify _build_shared_endpoint_links matches URLs across repos."""

    def test_same_url_two_repos_creates_link(self) -> None:
        from aibom.cross_repo_links import _build_shared_endpoint_links

        results = {
            "/repos/app": {
                "components": [
                    AIComponent(
                        name="https://llm.example.com/v1",
                        component_type=AIComponentType.LLM_ENDPOINT,
                        file_path="src/config.py",
                        line_number=10,
                        instance_id="ep-1",
                        metadata={"env_var": "LLM_ENDPOINT"},
                    ),
                ],
            },
            "/repos/infra": {
                "components": [
                    AIComponent(
                        name="https://llm.example.com/v1",
                        component_type=AIComponentType.LLM_ENDPOINT,
                        file_path="helm/values.yaml",
                        line_number=5,
                        instance_id="ep-2",
                        metadata={"env_var": "LLM_ENDPOINT"},
                    ),
                ],
            },
        }

        links = _build_shared_endpoint_links(results)
        assert len(links) == 1
        assert links[0].link_type == CrossRepoLinkType.ENV_VAR_BINDING
        repos = {o.repo_path for o in links[0].occurrences}
        assert repos == {"/repos/app", "/repos/infra"}

    def test_same_url_single_repo_no_link(self) -> None:
        from aibom.cross_repo_links import _build_shared_endpoint_links

        results = {
            "/repos/app": {
                "components": [
                    AIComponent(
                        name="https://llm.example.com/v1",
                        component_type=AIComponentType.LLM_ENDPOINT,
                        file_path="config.py",
                        line_number=1,
                        instance_id="ep-1",
                    ),
                ],
            },
        }
        links = _build_shared_endpoint_links(results)
        assert len(links) == 0

    def test_non_url_endpoints_ignored(self) -> None:
        from aibom.cross_repo_links import _build_shared_endpoint_links

        results = {
            "/repos/app": {
                "components": [
                    AIComponent(
                        name="env:SOME_VAR",
                        component_type=AIComponentType.LLM_ENDPOINT,
                        file_path="f.py",
                        line_number=1,
                        instance_id="ep-1",
                    ),
                ],
            },
            "/repos/infra": {
                "components": [
                    AIComponent(
                        name="env:SOME_VAR",
                        component_type=AIComponentType.LLM_ENDPOINT,
                        file_path="v.yaml",
                        line_number=1,
                        instance_id="ep-2",
                    ),
                ],
            },
        }
        links = _build_shared_endpoint_links(results)
        assert len(links) == 0

    def test_trailing_slash_normalization(self) -> None:
        from aibom.cross_repo_links import _build_shared_endpoint_links

        results = {
            "/repos/a": {
                "components": [
                    AIComponent(
                        name="https://api.example.com/",
                        component_type=AIComponentType.MODEL_ENDPOINT,
                        file_path="a.py",
                        line_number=1,
                        instance_id="ep-a",
                    ),
                ],
            },
            "/repos/b": {
                "components": [
                    AIComponent(
                        name="https://api.example.com",
                        component_type=AIComponentType.MODEL_ENDPOINT,
                        file_path="b.yaml",
                        line_number=1,
                        instance_id="ep-b",
                    ),
                ],
            },
        }
        links = _build_shared_endpoint_links(results)
        assert len(links) == 1


class TestAgentModelCrossLinks:
    """Fix 4: Deterministic Agent→Model cross-repo links."""

    def test_agent_uses_model_across_repos(self) -> None:
        from aibom.cross_repo_links import _build_agent_model_cross_links

        shared = [
            CrossRepoLink(
                link_type=CrossRepoLinkType.SHARED_MODEL,
                identifier="gpt-4o",
                resolved_value="gpt-4o",
                occurrences=[
                    RepoOccurrence(repo_path="/repos/app", component_name="gpt-4o", role="shared"),
                    RepoOccurrence(repo_path="/repos/infra", component_name="gpt-4o", role="shared"),
                ],
            ),
        ]
        per_repo = {
            "/repos/app": {
                "components": [],
                "relationships": [
                    ComponentRelationship(
                        source_instance_id="",
                        target_instance_id="",
                        source_name="MainAgent",
                        target_name="gpt-4o",
                        relationship_type=RelationshipType.USES_MODEL,
                        source_type=AIComponentType.AGENT,
                        target_type=AIComponentType.MODEL,
                    ),
                ],
            },
            "/repos/infra": {
                "components": [],
                "relationships": [],
            },
        }
        links = _build_agent_model_cross_links(shared, per_repo)
        assert len(links) == 1
        assert "MainAgent" in links[0].identifier
        assert "gpt-4o" in links[0].identifier
        repos = {o.repo_path for o in links[0].occurrences}
        assert repos == {"/repos/app", "/repos/infra"}

    def test_no_link_when_model_in_same_repo(self) -> None:
        from aibom.cross_repo_links import _build_agent_model_cross_links

        shared = [
            CrossRepoLink(
                link_type=CrossRepoLinkType.SHARED_MODEL,
                identifier="gpt-4o",
                occurrences=[
                    RepoOccurrence(repo_path="/repos/only", component_name="gpt-4o", role="shared"),
                ],
            ),
        ]
        per_repo = {
            "/repos/only": {
                "components": [],
                "relationships": [
                    ComponentRelationship(
                        source_instance_id="",
                        target_instance_id="",
                        source_name="Agent",
                        target_name="gpt-4o",
                        relationship_type=RelationshipType.USES_MODEL,
                        source_type=AIComponentType.AGENT,
                        target_type=AIComponentType.MODEL,
                    ),
                ],
            },
        }
        links = _build_agent_model_cross_links(shared, per_repo)
        assert len(links) == 0

    def test_non_agent_relationship_ignored(self) -> None:
        from aibom.cross_repo_links import _build_agent_model_cross_links

        shared = [
            CrossRepoLink(
                link_type=CrossRepoLinkType.SHARED_MODEL,
                identifier="gpt-4o",
                occurrences=[
                    RepoOccurrence(repo_path="/repos/app", component_name="gpt-4o", role="shared"),
                    RepoOccurrence(repo_path="/repos/infra", component_name="gpt-4o", role="shared"),
                ],
            ),
        ]
        per_repo = {
            "/repos/app": {
                "components": [],
                "relationships": [
                    ComponentRelationship(
                        source_instance_id="",
                        target_instance_id="",
                        source_name="openai",
                        target_name="gpt-4o",
                        relationship_type=RelationshipType.USES_MODEL,
                        source_type=AIComponentType.DEPENDENCY,
                        target_type=AIComponentType.MODEL,
                    ),
                ],
            },
            "/repos/infra": {"components": [], "relationships": []},
        }
        links = _build_agent_model_cross_links(shared, per_repo)
        assert len(links) == 0


class TestHelmEnvVarCrossRefIndexing:
    """Fix 5: Verify Helm env var names are indexed in cross_ref."""

    def test_resolved_env_var_indexed_alongside_leaf_key(self, tmp_path: Path) -> None:
        from aibom.cross_ref import build_env_index

        chart = tmp_path / "my-chart"
        chart.mkdir()
        (chart / "Chart.yaml").write_text("name: my-chart\nversion: 0.1.0\n")

        templates = chart / "templates"
        templates.mkdir()
        (templates / "deployment.yaml").write_text(
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "spec:\n"
            "  template:\n"
            "    spec:\n"
            "      containers:\n"
            "        - name: app\n"
            "          env:\n"
            "            - name: WEAVIATE_ENDPOINT\n"
            "              value: {{ .Values.global.weaviate.endpoint }}\n"
            "            - name: LLM_API_KEY\n"
            "              value: {{ .Values.config.llm.api_key }}\n"
        )

        values = chart / "values.yaml"
        values.write_text(
            "global:\n"
            "  weaviate:\n"
            "    endpoint: http://weaviate:8080\n"
            "config:\n"
            "  llm:\n"
            "    api_key: sk-test-key\n"
        )

        idx = build_env_index([str(tmp_path)])

        assert "WEAVIATE_ENDPOINT" in idx.env
        assert idx.env["WEAVIATE_ENDPOINT"][0].value == "http://weaviate:8080"

        assert "endpoint" in idx.env
        assert idx.env["endpoint"][0].value == "http://weaviate:8080"

        assert "LLM_API_KEY" in idx.env
        assert idx.env["LLM_API_KEY"][0].value == "sk-test-key"

    def test_fallback_when_no_chart_dir(self, tmp_path: Path) -> None:
        from aibom.cross_ref import build_env_index

        values = tmp_path / "values.yaml"
        values.write_text(
            "global:\n"
            "  endpoint: http://example.com\n"
        )

        idx = build_env_index([str(tmp_path)])

        assert "endpoint" in idx.env
        assert "WEAVIATE_ENDPOINT" not in idx.env
