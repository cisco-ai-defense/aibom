# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aibom.models import AIComponentType, DetectionSource, ScanContext
from aibom.scanners.base import scanner_registry
from aibom.scanners.deployment_detector import DeploymentDetector

from .conftest import run_scanner


class TestHelmK8sDetection:
    def test_detects_ai_container_image_in_deployment(self, tmp_path: Path) -> None:
        yml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
spec:
  template:
    spec:
      containers:
        - name: inference
          image: vllm/vllm-openai:latest
          resources:
            limits:
              nvidia.com/gpu: "1"
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"deploy.yaml": yml})
        deps = [c for c in comps if c.component_type == AIComponentType.DEPENDENCY]
        assert deps
        assert any("vllm" in (c.metadata.get("image") or "").lower() for c in deps)
        assert any(c.metadata.get("gpu") == "1" for c in deps)

    def test_detects_model_name_in_configmap(self, tmp_path: Path) -> None:
        yml = """apiVersion: v1
kind: ConfigMap
metadata:
  name: cfg
data:
  MODEL_NAME: "gpt-4o"
  OPENAI_API_KEY: "sk-xxx"
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"cm.yaml": yml})
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert any(c.model_name == "gpt-4o" for c in models)

    def test_detects_model_in_helm_values(self, tmp_path: Path) -> None:
        yml = """inference:
  model: gpt-4o-mini
  image: vllm/vllm-openai:v0.4.0
  resources:
    gpu: 1
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"values.yaml": yml})
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        deps = [c for c in comps if c.component_type == AIComponentType.DEPENDENCY]
        assert any(c.model_name == "gpt-4o-mini" for c in models)
        assert any("vllm" in (c.metadata.get("image") or "").lower() for c in deps)

    def test_helm_weaviate_endpoint_is_vector_store(self, tmp_path: Path) -> None:
        yml = """env:
  WEAVIATE:
    CLOUD_ENDPOINT: https://cluster.example.weaviate.cloud
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"values.yaml": yml})
        url = "https://cluster.example.weaviate.cloud"
        assert any(
            c.component_type == AIComponentType.VECTOR_STORE
            and c.metadata.get("endpoint_url") == url
            for c in comps
        )
        assert not any(
            c.component_type == AIComponentType.LLM_ENDPOINT
            and c.metadata.get("endpoint_url") == url
            for c in comps
        )

    def test_helm_conflicting_vector_backend_requires_agentic_review(self, tmp_path: Path) -> None:
        yml = """orchestrator:
  env:
    WEAVIATE:
      WEAVIATE_ENDPOINT: http://vector.example.internal
      VECTOR_DB_TYPE: chroma
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"values.yaml": yml})
        matches = [c for c in comps if c.name == "http://vector.example.internal"]
        assert len(matches) == 1
        comp = matches[0]
        assert comp.component_type == AIComponentType.VECTOR_STORE
        assert comp.metadata.get("env_var") == "WEAVIATE_ENDPOINT"
        assert comp.needs_agentic is True
        assert comp.metadata.get("store_technology") == "chromadb"
        assert "vector_db_type" in comp.agentic_hint.lower()
        assert "weaviate_endpoint" in comp.agentic_hint.lower()

    def test_helm_embedding_endpoint_and_engine_use_correct_types(self, tmp_path: Path) -> None:
        yml = """env:
  AZURE:
    EMBEDDING:
      LARGE3:
        ENDPOINT: https://example.openai.azure.com/
        ENGINE: text-embedding-3-large
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"values.yaml": yml})
        endpoint_url = "https://example.openai.azure.com/"
        assert any(
            c.component_type == AIComponentType.LLM_ENDPOINT
            and c.metadata.get("endpoint_url") == endpoint_url
            for c in comps
        )
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

    def test_helm_azure_endpoint_with_sibling_engine_is_llm_endpoint(self, tmp_path: Path) -> None:
        yml = """env:
  AZURE:
    CHAT_SERVICE:
      ENDPOINT: https://example.openai.azure.com/
      ENGINE: chat-model-deployment
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"values.yaml": yml})
        endpoint_url = "https://example.openai.azure.com/"
        assert any(
            c.component_type == AIComponentType.LLM_ENDPOINT
            and c.metadata.get("endpoint_url") == endpoint_url
            for c in comps
        )

    def test_helm_azure_endpoint_with_sibling_model_name_is_llm_endpoint(self, tmp_path: Path) -> None:
        yml = """env:
  CHAT_SERVICE:
    ENDPOINT: https://example.openai.azure.com/
    MODEL_NAME: assistant-model
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"values.yaml": yml})
        endpoint_url = "https://example.openai.azure.com/"
        assert any(
            c.component_type == AIComponentType.LLM_ENDPOINT
            and c.metadata.get("endpoint_url") == endpoint_url
            for c in comps
        )

    def test_detects_training_job(self, tmp_path: Path) -> None:
        yml = """apiVersion: batch/v1
kind: Job
metadata:
  name: train
spec:
  template:
    spec:
      containers:
        - name: trainer
          image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
          resources:
            limits:
              nvidia.com/gpu: "1"
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"job.yaml": yml})
        runs = [c for c in comps if c.component_type == AIComponentType.TRAINING_RUN]
        assert runs
        assert any("pytorch" in (c.metadata.get("image") or "").lower() for c in runs)

    def test_ignores_non_ai_deployment(self, tmp_path: Path) -> None:
        yml = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  template:
    spec:
      containers:
        - name: nginx
          image: nginx:latest
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"deploy.yaml": yml})
        assert comps == []


class TestTerraformDetection:
    def test_detects_sagemaker_endpoint(self, tmp_path: Path) -> None:
        hcl = '''resource "aws_sagemaker_endpoint" "inference" {
  endpoint_config_name = aws_sagemaker_endpoint_configuration.my_config.name
}
resource "aws_sagemaker_model" "my_model" {
  name = "my-model"
  primary_container {
    image = "123456789.dkr.ecr.us-east-1.amazonaws.com/sagemaker-huggingface:latest"
    model_data_url = "s3://my-bucket/models/model.tar.gz"
  }
}
'''
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"main.tf": hcl})
        assert len(comps) >= 2
        assert any(
            "sagemaker" in (c.metadata.get("terraform_resource_type") or "").lower()
            for c in comps
        )

    def test_detects_bedrock_agent(self, tmp_path: Path) -> None:
        hcl = '''resource "aws_bedrock_agent_agent" "assistant" {
  model_id = "anthropic.claude-3-sonnet"
}
'''
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"bedrock.tf": hcl})
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert any(c.model_name == "anthropic.claude-3-sonnet" for c in models)

    def test_detects_azure_openai_terraform(self, tmp_path: Path) -> None:
        hcl = '''resource "azurerm_cognitive_account" "openai" {
  kind = "OpenAI"
}
'''
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"azure.tf": hcl})
        assert comps
        assert any(
            c.metadata.get("terraform_resource_type") == "azurerm_cognitive_account"
            for c in comps
        )

    def test_detects_gcp_vertex(self, tmp_path: Path) -> None:
        hcl = '''resource "google_vertex_ai_endpoint" "endpoint" {
  display_name = "inference"
}
'''
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"gcp.tf": hcl})
        assert comps
        assert any(
            c.metadata.get("terraform_resource_type") == "google_vertex_ai_endpoint"
            for c in comps
        )

    def test_detects_gpu_instance_type(self, tmp_path: Path) -> None:
        hcl = '''resource "aws_sagemaker_notebook_instance" "nb" {
  name          = "nb"
  instance_type = "ml.p4d.24xlarge"
}
'''
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"gpu.tf": hcl})
        gpuish = [
            c
            for c in comps
            if c.metadata.get("instance_type")
            and "ml.p4d" in str(c.metadata.get("instance_type"))
        ]
        assert gpuish

    def test_detects_variable_defaults(self, tmp_path: Path) -> None:
        hcl = '''variable "model_name" { default = "claude-3-sonnet" }
'''
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"vars.tf": hcl})
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert any(c.model_name == "claude-3-sonnet" for c in models)


class TestCloudFormationDetection:
    def test_detects_cfn_sagemaker(self, tmp_path: Path) -> None:
        yml = """AWSTemplateFormatVersion: "2010-09-09"
Resources:
  MyEndpoint:
    Type: AWS::SageMaker::Endpoint
    Properties:
      EndpointConfigName: !Ref MyConfig
  MyModel:
    Type: AWS::SageMaker::Model
    Properties:
      ModelName: my-inference-model
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"cfn.yaml": yml})
        assert any(c.metadata.get("cfn_type") == "AWS::SageMaker::Endpoint" for c in comps)
        assert any(c.metadata.get("cfn_type") == "AWS::SageMaker::Model" for c in comps)

    def test_detects_cfn_bedrock(self, tmp_path: Path) -> None:
        yml = """AWSTemplateFormatVersion: "2010-09-09"
Resources:
  Agent:
    Type: AWS::Bedrock::Agent
    Properties: {}
  Guard:
    Type: AWS::Bedrock::Guardrail
    Properties: {}
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"bedrock.yaml": yml})
        types = {c.metadata.get("cfn_type") for c in comps}
        assert "AWS::Bedrock::Agent" in types
        assert "AWS::Bedrock::Guardrail" in types

    def test_detects_cfn_parameters(self, tmp_path: Path) -> None:
        yml = """AWSTemplateFormatVersion: "2010-09-09"
Parameters:
  ModelName:
    Type: String
    Default: gpt-4o
Resources: {}
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"params.yaml": yml})
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert any(c.model_name == "gpt-4o" for c in models)

    def test_detects_cfn_json(self, tmp_path: Path) -> None:
        doc = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Resources": {
                "MyEndpoint": {
                    "Type": "AWS::SageMaker::Endpoint",
                    "Properties": {"EndpointConfigName": {"Ref": "MyConfig"}},
                },
                "MyModel": {
                    "Type": "AWS::SageMaker::Model",
                    "Properties": {"ModelName": "my-inference-model"},
                },
            },
        }
        comps, _ = run_scanner(
            DeploymentDetector, tmp_path, {"cfn.json": json.dumps(doc, indent=2)}
        )
        assert any(c.metadata.get("cfn_type") == "AWS::SageMaker::Endpoint" for c in comps)
        assert any(c.metadata.get("cfn_type") == "AWS::SageMaker::Model" for c in comps)


class TestAzureARMDetection:
    def test_detects_arm_cognitive_services(self, tmp_path: Path) -> None:
        doc = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "resources": [
                {
                    "type": "Microsoft.CognitiveServices/accounts",
                    "name": "my-openai",
                    "properties": {
                        "kind": "OpenAI",
                        "deployments": [
                            {
                                "model": {
                                    "name": "gpt-4o",
                                    "version": "2024-05-13",
                                }
                            }
                        ],
                    },
                }
            ],
        }
        comps, _ = run_scanner(
            DeploymentDetector, tmp_path, {"arm.json": json.dumps(doc)}
        )
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert any(c.model_name == "gpt-4o" for c in models)

    def test_detects_arm_ml_workspace(self, tmp_path: Path) -> None:
        doc = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "resources": [
                {
                    "type": "Microsoft.MachineLearningServices/workspaces",
                    "name": "ws1",
                    "properties": {},
                }
            ],
        }
        comps, _ = run_scanner(
            DeploymentDetector, tmp_path, {"ml.json": json.dumps(doc)}
        )
        assert any(
            c.metadata.get("arm_type") == "Microsoft.MachineLearningServices/workspaces"
            for c in comps
        )


class TestBicepDetection:
    def test_detects_bicep_cognitive_services(self, tmp_path: Path) -> None:
        src = """resource openai 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: 'my-openai'
  kind: 'OpenAI'
}
param modelName string = 'gpt-4o'
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"main.bicep": src})
        assert any(
            c.metadata.get("bicep_type") == "Microsoft.CognitiveServices/accounts"
            for c in comps
        )
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert any(c.model_name == "gpt-4o" for c in models)


class TestScannerIntegration:
    def test_scanner_registered(self) -> None:
        assert DeploymentDetector in scanner_registry

    def test_supports_always_true(self, tmp_path: Path) -> None:
        ctx = ScanContext(paths=[str(tmp_path)])
        assert DeploymentDetector().supports(ctx) is True

    def test_mixed_iac_directory(self, tmp_path: Path) -> None:
        files = {
            "infra/main.tf": '''resource "google_vertex_ai_endpoint" "e" {
  display_name = "x"
}
''',
            "k8s/cm.yaml": """apiVersion: v1
kind: ConfigMap
metadata:
  name: m
data:
  MODEL_NAME: "gpt-4o"
""",
            "helm/values.yaml": """inference:
  model: gpt-4o-mini
  image: vllm/vllm-openai:v0.4.0
""",
        }
        comps, rels = run_scanner(DeploymentDetector, tmp_path, files)
        assert rels == []
        assert any(
            c.metadata.get("terraform_resource_type") == "google_vertex_ai_endpoint"
            for c in comps
        )
        assert any(c.model_name == "gpt-4o" for c in comps)
        assert any(c.model_name == "gpt-4o-mini" for c in comps)

    def test_empty_directory(self, tmp_path: Path) -> None:
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, rels = DeploymentDetector().scan(ctx)
        assert comps == [] and rels == []

    def test_malformed_yaml_handled(self, tmp_path: Path) -> None:
        comps, rels = run_scanner(
            DeploymentDetector, tmp_path, {"broken.yaml": "foo: [\n  x"}
        )
        assert rels == []
        assert isinstance(comps, list)

    def test_detection_source_is_config_file(self, tmp_path: Path) -> None:
        yml = """apiVersion: v1
kind: ConfigMap
metadata:
  name: c
data:
  MODEL_NAME: "gpt-4o"
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"x.yaml": yml})
        assert comps
        assert all(c.detection_source == DetectionSource.CONFIG_FILE for c in comps)

    def test_vendored_venv_skipped(self, tmp_path: Path) -> None:
        venv_dir = tmp_path / "myapp_venv" / "lib" / "python3.11" / "site-packages"
        venv_dir.mkdir(parents=True)
        (venv_dir / "values.yaml").write_text(
            "image: vllm/vllm-openai:latest\n"
        )
        (tmp_path / "real.yaml").write_text(
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\n"
            "spec:\n  template:\n    spec:\n      containers:\n"
            "        - image: vllm/vllm-openai:latest\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = DeploymentDetector().scan(ctx)
        paths = [c.file_path for c in comps]
        assert not any("myapp_venv" in p for p in paths)

    def test_site_packages_skipped(self, tmp_path: Path) -> None:
        sp_dir = tmp_path / "site-packages" / "openai"
        sp_dir.mkdir(parents=True)
        (sp_dir / "deploy.yaml").write_text(
            "apiVersion: v1\nkind: ConfigMap\ndata:\n  MODEL: gpt-4o\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = DeploymentDetector().scan(ctx)
        assert len(comps) == 0


class TestObservabilityEndpointDetection:
    """Fix 2: Helm keys matching observability platform tokens emit OBSERVABILITY."""

    def test_langsmith_endpoint_classified_as_observability(self, tmp_path: Path) -> None:
        yml = """env:
  LANGSMITH:
    API_ENDPOINT: https://tracing.example.com/api/v1
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"values.yaml": yml})
        obs = [c for c in comps if c.component_type == AIComponentType.OBSERVABILITY]
        assert len(obs) >= 1
        assert any(c.name == "https://tracing.example.com/api/v1" for c in obs)

    def test_mlflow_endpoint_classified_as_observability(self, tmp_path: Path) -> None:
        yml = """env:
  MLFLOW:
    TRACKING_ENDPOINT: https://mlflow.internal.example.com
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"values.yaml": yml})
        obs = [c for c in comps if c.component_type == AIComponentType.OBSERVABILITY]
        assert len(obs) >= 1

    def test_non_observability_key_not_affected(self, tmp_path: Path) -> None:
        yml = """env:
  AZURE:
    CHAT_SERVICE:
      ENDPOINT: https://example.openai.azure.com/
"""
        comps, _ = run_scanner(DeploymentDetector, tmp_path, {"values.yaml": yml})
        obs = [c for c in comps if c.component_type == AIComponentType.OBSERVABILITY]
        assert len(obs) == 0


class TestHelmKeyPathGate:
    """Fix 6: ``_helm_key_is_definitely_not_model`` and walker integration."""

    def test_helm_key_gate_blocks_tag_version_threshold_ip_keys(self) -> None:
        from aibom.scanners.deployment_detector import (
            _helm_key_is_definitely_not_model,
        )

        deny = [
            "image.tag",
            "sops.version",
            "runbook.image.tag",
            "widget_mcp_server.env.CONFIDENCE_SCHEMA_MISMATCH_THRESHOLD",
            "aigateway.env.MY_NODE_IP",
            "env.ENVIRONMENT",
            "orchestrator.image.tag",
            "servingEngineSpec.defaultModelConfig.tag",
            "servingEngineSpec.modelSpec.0.tag",
            "meraki_mcp_server.image.tag",
            "widget_mcp_server.image.tag",
            "sample.config.image_tag",
            "foo.bar_url",
            "cache.redis_host",
            "service.api_port",
        ]
        for key in deny:
            assert _helm_key_is_definitely_not_model(key), (
                f"key-path gate must reject {key!r}"
            )

        allow = [
            "env.TASK_CLASSIFIER.LLM_MODEL_NAME",
            "env.AZURE.GPT4o.ENGINE",
            "widget_mcp_server.env.MODEL_NAME",
            "lapse.env.LAPSE_LLM_MODEL_NAME",
            "servingEngineSpec.modelSpec.0.annotations.model",
        ]
        for key in allow:
            assert not _helm_key_is_definitely_not_model(key), (
                f"key-path gate must NOT reject {key!r}"
            )

    def test_helm_walker_drops_garbage_models_at_tag_version_keys(
        self, tmp_path: Path,
    ) -> None:
        yml = """sops:
  version: "3.10.2"
image:
  tag: "release-0.3.1"
aigateway:
  env:
    MY_NODE_IP: "127.0.0.1"
widget_mcp_server:
  image:
    tag: "2.0.6"
  env:
    CONFIDENCE_SCHEMA_MISMATCH_THRESHOLD: "0.5"
orchestrator:
  image:
    tag: "1.1.1"
servingEngineSpec:
  defaultModelConfig:
    tag: "1.1"
  modelSpec:
    - tag: "v4.5.4"
meraki_mcp_server:
  image:
    tag: "v1.6"
runbook:
  image:
    tag: "3.2.2"
env:
  ENVIRONMENT: "dev"
"""
        comps, _ = run_scanner(
            DeploymentDetector, tmp_path, {"values.yaml": yml},
        )
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert models == [], (
            "garbage values at non-model helm key paths must not produce "
            f"MODEL components; got {[c.model_name for c in models]}"
        )

    def test_helm_walker_keeps_real_models_at_model_keys(
        self, tmp_path: Path,
    ) -> None:
        yml = """env:
  TASK_CLASSIFIER:
    LLM_MODEL_NAME: "gpt-4o"
  AZURE:
    GPT4o:
      ENGINE: "gpt-4o"
widget_mcp_server:
  env:
    MODEL_NAME: "anthropic/claude-3-5-sonnet"
"""
        comps, _ = run_scanner(
            DeploymentDetector, tmp_path, {"values.yaml": yml},
        )
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        names = {c.model_name for c in models}
        assert "gpt-4o" in names
        assert "anthropic/claude-3-5-sonnet" in names
        for c in models:
            assert "helm_key" in (c.metadata or {}), (
                f"helm-walker MODEL component must carry helm_key metadata; "
                f"got {c.metadata!r}"
            )

    def test_helm_walker_high_conf_model_routes_through_agent(
        self, tmp_path: Path,
    ) -> None:
        """Fix 11: high-conf helm model branch must go through the agent."""
        yml = """env:
  MODEL_NAME: "gpt-4o"
"""
        comps, _ = run_scanner(
            DeploymentDetector, tmp_path, {"values.yaml": yml},
        )
        models = [
            c
            for c in comps
            if c.component_type == AIComponentType.MODEL
            and c.model_name == "gpt-4o"
        ]
        assert models, "expected gpt-4o MODEL component from env.MODEL_NAME"
        c = models[0]
        assert c.needs_agentic is True, (
            "high-confidence helm-walker model emission must route through "
            "the agent so the value-literal prompt rule has a chance to fire"
        )
        hint = c.agentic_hint or ""
        assert "deployment artifact" in hint.lower(), (
            f"agentic_hint must reference the deployment-artifact escape "
            f"clause from prompt section 4; got {hint!r}"
        )
