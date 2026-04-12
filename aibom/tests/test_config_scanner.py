# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aibom.models import AIComponentType
from aibom.scanners.config_scanner import ConfigScanner

from .conftest import run_scanner


class TestConfigScanner:
    def test_docker_compose_ollama_image(self, tmp_path: Path) -> None:
        yml = "services:\n  llm:\n    image: ollama/ollama:latest\n"
        comps, rels = run_scanner(ConfigScanner, tmp_path, {"docker-compose.yml": yml})
        assert rels == []
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert models
        assert any("ollama" in (c.model_name or "").lower() for c in models)

    def test_dockerfile_from_vllm_base_image(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            ConfigScanner, tmp_path, {"Dockerfile": "FROM vllm/vllm-openai:latest\n"}
        )
        assert any(
            c.component_type == AIComponentType.MODEL
            and c.model_name
            and "vllm" in c.model_name.lower()
            for c in comps
        )

    def test_env_model_vars_not_secrets(self, tmp_path: Path) -> None:
        env = "OPENAI_MODEL=gpt-4o\nLLM_NAME=meta-llama/Llama-3.2-1B\n"
        comps, _ = run_scanner(ConfigScanner, tmp_path, {".env": env})
        secrets = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert secrets == []
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert {c.metadata.get("env_var") for c in models if c.metadata.get("env_var")} >= {
            "OPENAI_MODEL",
            "LLM_NAME",
        }

    def test_env_embedding_model_is_embedding(self, tmp_path: Path) -> None:
        env = "OPENAI_EMBEDDING_MODEL=text-embedding-3-large\n"
        comps, _ = run_scanner(ConfigScanner, tmp_path, {".env": env})
        assert any(
            c.component_type == AIComponentType.EMBEDDING
            and c.model_name == "text-embedding-3-large"
            for c in comps
        )
        assert not any(
            c.component_type == AIComponentType.MODEL
            and c.model_name == "text-embedding-3-large"
            for c in comps
        )

    def test_env_weaviate_endpoint_is_vector_store(self, tmp_path: Path) -> None:
        env = "WEAVIATE_CLOUD_ENDPOINT=https://cluster.example.weaviate.cloud\n"
        comps, _ = run_scanner(ConfigScanner, tmp_path, {".env": env})
        assert any(
            c.component_type == AIComponentType.VECTOR_STORE
            and c.metadata.get("env_var") == "WEAVIATE_CLOUD_ENDPOINT"
            for c in comps
        )
        assert not any(
            c.component_type == AIComponentType.LLM_ENDPOINT
            and c.metadata.get("env_var") == "WEAVIATE_CLOUD_ENDPOINT"
            for c in comps
        )

    def test_crewai_yaml_agents(self, tmp_path: Path) -> None:
        cfg = "agents:\n  - name: Researcher\n    role: analyst\n"
        comps, _ = run_scanner(ConfigScanner, tmp_path, {"crewai.yaml": cfg})
        agents = [c for c in comps if c.component_type == AIComponentType.AGENT]
        assert any("Researcher" in c.name or "researcher" in c.name.lower() for c in agents)
        crew_named = [c for c in agents if "crewai_agent" in c.name]
        assert crew_named
        assert all(c.framework == "crewai" for c in crew_named)

    def test_langgraph_json_model_string(self, tmp_path: Path) -> None:
        j = '{"graphs": {"app": "./graph.py"}, "assistant_model": "gpt-4o-mini"}'
        comps, _ = run_scanner(ConfigScanner, tmp_path, {"langgraph.json": j})
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert any(c.model_name and "gpt-4o-mini" in c.model_name for c in models)
        assert any(c.framework == "langgraph" for c in models)
