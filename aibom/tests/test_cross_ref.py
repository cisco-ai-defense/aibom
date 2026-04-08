# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aibom.cross_ref import (
    CrossRefIndex,
    EnvVarEntry,
    build_env_index,
    build_package_index,
    resolve_components,
)
from aibom.models.enums import AIComponentType, DetectionSource
from aibom.models.scan import AIComponent


class TestEnvVarIndexing:
    def test_dotenv_parsing(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text(
            """MODEL_NAME=gpt-4o
OPENAI_API_KEY="sk-abc123"
# comment
EMPTY_VAR=
EMBEDDING_MODEL='text-embedding-3-small'
""",
            encoding="utf-8",
        )
        idx = build_env_index([str(tmp_path)])
        assert idx.env["MODEL_NAME"][0].value == "gpt-4o"
        assert idx.env["OPENAI_API_KEY"][0].value == "sk-abc123"
        assert idx.env["EMBEDDING_MODEL"][0].value == "text-embedding-3-small"
        assert idx.env["EMPTY_VAR"][0].value == ""

    def test_docker_compose_env_dict(self, tmp_path: Path) -> None:
        (tmp_path / "docker-compose.yml").write_text(
            """services:
  api:
    image: app:latest
    environment:
      MODEL_NAME: gpt-4o
      OPENAI_API_KEY: sk-xyz
""",
            encoding="utf-8",
        )
        idx = build_env_index([str(tmp_path)])
        assert idx.env["MODEL_NAME"][0].value == "gpt-4o"
        assert idx.env["OPENAI_API_KEY"][0].value == "sk-xyz"
        assert idx.env["MODEL_NAME"][0].source_type == "docker-compose"
        assert idx.env["OPENAI_API_KEY"][0].source_type == "docker-compose"

    def test_docker_compose_env_list(self, tmp_path: Path) -> None:
        (tmp_path / "docker-compose.yml").write_text(
            """services:
  api:
    environment:
      - MODEL_NAME=claude-3-sonnet
      - DEBUG=true
""",
            encoding="utf-8",
        )
        idx = build_env_index([str(tmp_path)])
        assert idx.env["MODEL_NAME"][0].value == "claude-3-sonnet"
        assert idx.env["DEBUG"][0].value == "true"

    def test_tfvars_parsing(self, tmp_path: Path) -> None:
        (tmp_path / "terraform.tfvars").write_text(
            """model_name = "anthropic.claude-3-sonnet"
region     = "us-east-1"
gpu_count  = 2
""",
            encoding="utf-8",
        )
        idx = build_env_index([str(tmp_path)])
        assert idx.env["model_name"][0].value == "anthropic.claude-3-sonnet"
        assert idx.env["region"][0].value == "us-east-1"

    def test_helm_values(self, tmp_path: Path) -> None:
        (tmp_path / "values.yaml").write_text(
            """inference:
  modelName: gpt-4o-mini
  endpoint: https://api.openai.com
  replicas: 3
""",
            encoding="utf-8",
        )
        idx = build_env_index([str(tmp_path)])
        assert idx.env["modelName"][0].value == "gpt-4o-mini"

    def test_k8s_configmap(self, tmp_path: Path) -> None:
        (tmp_path / "configmap.yaml").write_text(
            """apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-config
data:
  MODEL_NAME: gpt-4o
  LOG_LEVEL: debug
""",
            encoding="utf-8",
        )
        idx = build_env_index([str(tmp_path)])
        assert idx.env["MODEL_NAME"][0].value == "gpt-4o"
        assert idx.env["MODEL_NAME"][0].source_type == "k8s-configmap"

    def test_multiple_sources_same_var(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / ".env").write_text("MODEL_NAME=first\n", encoding="utf-8")
        (b / ".env").write_text("MODEL_NAME=second\n", encoding="utf-8")
        idx = build_env_index([str(tmp_path)])
        values = {e.value for e in idx.env["MODEL_NAME"]}
        assert values == {"first", "second"}

    def test_empty_dir(self, tmp_path: Path) -> None:
        idx = build_env_index([str(tmp_path)])
        assert idx.env == {}


class TestPackageIndexing:
    def test_requirements_txt(self, tmp_path: Path) -> None:
        (tmp_path / "requirements.txt").write_text(
            """langchain>=0.2.0
openai==1.30.0
requests>=2.31.0
transformers
""",
            encoding="utf-8",
        )
        idx = build_package_index([str(tmp_path)])
        assert "langchain" in idx.packages
        assert "openai" in idx.packages
        assert "transformers" in idx.packages
        assert "requests" in idx.packages

    def test_package_json(self, tmp_path: Path) -> None:
        payload = {
            "dependencies": {
                "@langchain/core": "^0.2.0",
                "express": "^4.18.0",
                "@anthropic-ai/sdk": "^0.20.0",
            }
        }
        (tmp_path / "package.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        idx = build_package_index([str(tmp_path)])
        assert "@langchain/core" in idx.packages
        assert "@anthropic-ai/sdk" in idx.packages
        assert "express" in idx.packages

    def test_pyproject_toml(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            """[project]
dependencies = ["torch>=2.0", "pydantic>=2.0"]
[project.optional-dependencies]
ml = ["transformers>=4.40"]
""",
            encoding="utf-8",
        )
        idx = build_package_index([str(tmp_path)])
        assert "torch" in idx.packages
        assert "transformers" in idx.packages
        assert "pydantic" in idx.packages

    def test_go_mod(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text(
            """module github.com/org/service
require (
    github.com/openai/openai-go v0.1.0
    github.com/gin-gonic/gin v1.9.0
)
""",
            encoding="utf-8",
        )
        idx = build_package_index([str(tmp_path)])
        assert "github.com/openai/openai-go" in idx.packages
        assert "github.com/gin-gonic/gin" in idx.packages


class TestResolveComponents:
    def test_resolves_env_var_model_name(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("MODEL_NAME=gpt-4o\n", encoding="utf-8")
        idx = build_env_index([str(tmp_path)])
        comp = AIComponent(
            name="m",
            component_type=AIComponentType.MODEL,
            detection_source=DetectionSource.CONFIG_FILE,
            model_name=None,
            metadata={"env": "MODEL_NAME"},
        )
        out = resolve_components([comp], idx)
        assert out[0].model_name == "gpt-4o"

    def test_resolves_dollar_var_in_model_name(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("MODEL_NAME=claude-3-sonnet\n", encoding="utf-8")
        idx = build_env_index([str(tmp_path)])
        comp = AIComponent(
            name="m",
            component_type=AIComponentType.MODEL,
            detection_source=DetectionSource.CONFIG_FILE,
            model_name="${MODEL_NAME}",
            metadata={},
        )
        out = resolve_components([comp], idx)
        assert out[0].model_name == "claude-3-sonnet"

    def test_no_resolution_when_already_set(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("MODEL_NAME=other\n", encoding="utf-8")
        idx = build_env_index([str(tmp_path)])
        comp = AIComponent(
            name="m",
            component_type=AIComponentType.MODEL,
            detection_source=DetectionSource.CONFIG_FILE,
            model_name="gpt-4o",
            metadata={"env": "MODEL_NAME"},
        )
        out = resolve_components([comp], idx)
        assert out[0].model_name == "gpt-4o"

    def test_no_resolution_when_var_missing(self, tmp_path: Path) -> None:
        idx = build_env_index([str(tmp_path)])
        comp = AIComponent(
            name="m",
            component_type=AIComponentType.MODEL,
            detection_source=DetectionSource.CONFIG_FILE,
            model_name=None,
            metadata={"env": "UNKNOWN_VAR"},
        )
        out = resolve_components([comp], idx)
        assert out[0].model_name is None

    def test_resolves_config_key(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("MODEL_NAME=gpt-4o\n", encoding="utf-8")
        idx = build_env_index([str(tmp_path)])
        comp = AIComponent(
            name="m",
            component_type=AIComponentType.MODEL,
            detection_source=DetectionSource.CONFIG_FILE,
            model_name=None,
            metadata={"config_key": "MODEL_NAME"},
        )
        out = resolve_components([comp], idx)
        assert out[0].model_name == "gpt-4o"

    def test_provenance_metadata_on_resolution(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("LLM_MODEL=gpt-4o\n", encoding="utf-8")
        idx = build_env_index([str(tmp_path)])
        comp = AIComponent(
            name="env:LLM_MODEL",
            component_type=AIComponentType.MODEL,
            detection_source=DetectionSource.CODE_ANALYSIS,
            model_name=None,
            needs_agentic=True,
            metadata={"env": "LLM_MODEL", "env_context": "model_kwarg"},
        )
        out = resolve_components([comp], idx)
        assert out[0].model_name == "gpt-4o"
        assert out[0].metadata.get("resolved_from") == "dotenv"
        assert "resolved_source_file" in out[0].metadata
        assert out[0].metadata.get("resolved_env_var") == "LLM_MODEL"

    def test_unresolved_env_gets_agentic_hint(self) -> None:
        idx = CrossRefIndex()
        comp = AIComponent(
            name="env:MISSING_VAR",
            component_type=AIComponentType.MODEL,
            detection_source=DetectionSource.CODE_ANALYSIS,
            model_name=None,
            needs_agentic=True,
            metadata={"env": "MISSING_VAR", "env_context": "model_kwarg"},
        )
        out = resolve_components([comp], idx)
        assert out[0].needs_agentic is True
        assert "MISSING_VAR" in out[0].agentic_hint
        assert "not found in any config source" in out[0].agentic_hint
        assert out[0].metadata.get("unresolved_env_var") == "MISSING_VAR"

    def test_resolution_upgrades_confidence_on_registry_hit(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("LLM_MODEL=gpt-4o\n", encoding="utf-8")
        idx = build_env_index([str(tmp_path)])
        comp = AIComponent(
            name="env:LLM_MODEL",
            component_type=AIComponentType.MODEL,
            detection_source=DetectionSource.CODE_ANALYSIS,
            confidence=0.3,
            needs_agentic=True,
            model_name=None,
            metadata={"env": "LLM_MODEL", "env_context": "model_kwarg"},
        )
        out = resolve_components([comp], idx)
        assert out[0].confidence >= 0.5
