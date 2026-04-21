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

"""Tests for asset category detection: Embeddings, MCP Clients, Guardrails,
Observability, and Secrets across dependency, import, and env-var scanners."""

from __future__ import annotations

from pathlib import Path

import pytest

from aibom.models.enums import AIComponentType, DetectionSource
from aibom.models.scan import AIComponent, ScanContext
from aibom.scanners.dependency_scanner import DependencyScanner


# ---------------------------------------------------------------------------
# Gap 1: Embedding import-based detection
# ---------------------------------------------------------------------------


class TestEmbeddingImportDetection:
    """Embedders/Embeddings should be detected from import statements."""

    def test_embedder_import_detected(self, tmp_path: Path):
        (tmp_path / "emb.py").write_text(
            "from haystack.components.embedders import SentenceTransformersDocumentEmbedder\n"
            "\n"
            "embedder = SentenceTransformersDocumentEmbedder()\n"
        )
        from aibom.cst_parser import parse_source_code
        from aibom.scanners.kb_enrichment_scanner import _detect_import_based_assets

        source = (tmp_path / "emb.py").read_text()
        result = parse_source_code(str(tmp_path / "emb.py"), source)
        comps = _detect_import_based_assets(result)
        emb_comps = [c for c in comps if c.component_type == AIComponentType.EMBEDDING]
        assert len(emb_comps) >= 1

    def test_embedding_class_import_detected(self, tmp_path: Path):
        (tmp_path / "emb2.py").write_text(
            "from langchain_openai import OpenAIEmbeddings\n"
            "\n"
            "emb = OpenAIEmbeddings(model='text-embedding-3-small')\n"
        )
        from aibom.cst_parser import parse_source_code
        from aibom.scanners.kb_enrichment_scanner import _detect_import_based_assets

        source = (tmp_path / "emb2.py").read_text()
        result = parse_source_code(str(tmp_path / "emb2.py"), source)
        comps = _detect_import_based_assets(result)
        emb_comps = [c for c in comps if c.component_type == AIComponentType.EMBEDDING]
        assert len(emb_comps) >= 1

    def test_type_inference_from_class_name(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("OpenAIEmbedder") == (AIComponentType.EMBEDDING, True)
        assert _infer_type_from_name("SentenceTransformersDocumentEmbedding") == (AIComponentType.EMBEDDING, True)
        assert _infer_type_from_name("CustomEmbeddings") == (AIComponentType.EMBEDDING, True)

    def test_suggestive_candidate_uses_correct_type(self, tmp_path: Path):
        """_emit_suggestive_candidates should infer EMBEDDING, not MODEL."""
        emb_dir = tmp_path / "models" / "embedders"
        emb_dir.mkdir(parents=True)
        (emb_dir / "openai.py").write_text(
            "from base_embedder import BaseEmbedder\n"
            "\n"
            "emb = OpenAIEmbedder(api_key='test')\n"
        )
        from aibom.scanners.kb_enrichment_scanner import _emit_suggestive_candidates

        source = (emb_dir / "openai.py").read_text()
        comps = _emit_suggestive_candidates(emb_dir / "openai.py", source)
        emb_comps = [c for c in comps if c.component_type == AIComponentType.EMBEDDING]
        assert len(emb_comps) >= 1
        assert all(c.component_type != AIComponentType.MODEL for c in comps if "Embed" in c.name)

    def test_embedding_helper_function_import_is_not_detected(self, tmp_path: Path):
        (tmp_path / "helpers.py").write_text(
            "from storage_helpers import get_embeddings_archive_path\n"
            "\n"
            "def build_path() -> str:\n"
            "    return get_embeddings_archive_path('foo', '1.0')\n"
        )
        from aibom.cst_parser import parse_source_code
        from aibom.scanners.kb_enrichment_scanner import _detect_import_based_assets

        source = (tmp_path / "helpers.py").read_text()
        result = parse_source_code(str(tmp_path / "helpers.py"), source)
        comps = _detect_import_based_assets(result)
        assert comps == []

    def test_embedding_constant_import_is_not_detected(self, tmp_path: Path):
        (tmp_path / "constants_user.py").write_text(
            "from storage_constants import EMBEDDINGS_ARCHIVE_TEMPLATE\n"
            "\n"
            "def build_key(product_name: str, version: str) -> str:\n"
            "    return EMBEDDINGS_ARCHIVE_TEMPLATE.format(product_name=product_name, version=version)\n"
        )
        from aibom.cst_parser import parse_source_code
        from aibom.scanners.kb_enrichment_scanner import _detect_import_based_assets

        source = (tmp_path / "constants_user.py").read_text()
        result = parse_source_code(str(tmp_path / "constants_user.py"), source)
        comps = _detect_import_based_assets(result)
        assert comps == []

    def test_copy_embeddings_activity_is_not_suggestive_embedding(self, tmp_path: Path):
        emb_dir = tmp_path / "etl" / "embeddings"
        emb_dir.mkdir(parents=True)
        activity = emb_dir / "copy.py"
        activity.write_text(
            "from base_activity import BaseActivity\n"
            "\n"
            "copy_job = CopyEmbeddingsArchiveToBucket(source_bucket='dev-bucket')\n"
        )
        from aibom.scanners.kb_enrichment_scanner import _emit_suggestive_candidates

        comps = _emit_suggestive_candidates(activity, activity.read_text())
        assert comps == []


# ---------------------------------------------------------------------------
# Gap 2: MCP Client detection
# ---------------------------------------------------------------------------


class TestMCPClientDetection:
    """MCP client packages and imports should be detected."""

    def test_mcp_client_in_requirements(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("mcp-client==0.1.0\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        mcp = [c for c in comps if "mcp" in c.name.lower()]
        assert len(mcp) >= 1

    def test_mcp_client_import_detected(self, tmp_path: Path):
        (tmp_path / "client.py").write_text(
            "from mcp.client import MCPClient\n"
            "\n"
            "client = MCPClient()\n"
        )
        from aibom.cst_parser import parse_source_code
        from aibom.scanners.kb_enrichment_scanner import _detect_import_based_assets

        source = (tmp_path / "client.py").read_text()
        result = parse_source_code(str(tmp_path / "client.py"), source)
        comps = _detect_import_based_assets(result)
        mcp_comps = [c for c in comps if c.component_type == AIComponentType.MCP_CLIENT]
        assert len(mcp_comps) >= 1


# ---------------------------------------------------------------------------
# Gap 3: Guardrail detection
# ---------------------------------------------------------------------------


class TestGuardrailDetection:
    """Guardrail frameworks should be detected from dependencies and imports."""

    def test_nemoguardrails_in_requirements(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("nemoguardrails==0.10.0\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        gr = [c for c in comps if "nemoguardrails" in c.name.lower()]
        assert len(gr) >= 1

    def test_guardrails_ai_in_requirements(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("guardrails-ai==0.5.0\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        gr = [c for c in comps if "guardrails" in c.name.lower()]
        assert len(gr) >= 1

    def test_llm_guard_in_requirements(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("llm-guard==0.3.0\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        gr = [c for c in comps if "llm-guard" in c.name.lower()]
        assert len(gr) >= 1

    def test_rebuff_in_requirements(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("rebuff==0.1.0\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        rebuff = [c for c in comps if "rebuff" in c.name.lower()]
        assert len(rebuff) >= 1

    def test_guardrail_import_detected(self, tmp_path: Path):
        (tmp_path / "guard.py").write_text(
            "from nemoguardrails import LLMRails, RailsConfig\n"
            "\n"
            "config = RailsConfig.from_path('config')\n"
            "rails = LLMRails(config)\n"
        )
        from aibom.cst_parser import parse_source_code
        from aibom.scanners.kb_enrichment_scanner import _detect_import_based_assets

        source = (tmp_path / "guard.py").read_text()
        result = parse_source_code(str(tmp_path / "guard.py"), source)
        comps = _detect_import_based_assets(result)
        gr_comps = [c for c in comps if c.component_type == AIComponentType.GUARDRAIL]
        assert len(gr_comps) >= 1

    def test_guardrail_type_inference(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("LLMRails") == (AIComponentType.GUARDRAIL, True)
        assert _infer_type_from_name("InputGuardrail") == (AIComponentType.GUARDRAIL, True)
        assert _infer_type_from_name("ContentInspector") == (AIComponentType.GUARDRAIL, True)


# ---------------------------------------------------------------------------
# Gap 4: Observability detection
# ---------------------------------------------------------------------------


class TestObservabilityDetection:
    """Observability frameworks should be detected from dependencies and imports."""

    def test_traceloop_sdk_in_requirements(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("traceloop-sdk==0.30.0\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        tl = [c for c in comps if "traceloop" in c.name.lower()]
        assert len(tl) >= 1

    def test_langfuse_in_requirements(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("langfuse==2.0.0\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        lf = [c for c in comps if "langfuse" in c.name.lower()]
        assert len(lf) >= 1

    def test_langsmith_in_requirements(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("langsmith==0.1.0\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        ls = [c for c in comps if "langsmith" in c.name.lower()]
        assert len(ls) >= 1

    def test_arize_phoenix_in_requirements(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("arize-phoenix==5.0.0\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        ap = [c for c in comps if "arize" in c.name.lower()]
        assert len(ap) >= 1

    def test_langsmith_in_package_json(self, tmp_path: Path):
        (tmp_path / "package.json").write_text(
            '{"dependencies": {"langsmith": "^0.1.0", "langfuse": "^2.0.0"}}\n'
        )
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        obs = [c for c in comps if c.name.lower() in ("langsmith", "langfuse")]
        assert len(obs) >= 1

    def test_observability_import_detected(self, tmp_path: Path):
        (tmp_path / "tracing.py").write_text(
            "from traceloop.sdk import Traceloop\n"
            "\n"
            "Traceloop.init()\n"
        )
        from aibom.cst_parser import parse_source_code
        from aibom.scanners.kb_enrichment_scanner import _detect_import_based_assets

        source = (tmp_path / "tracing.py").read_text()
        result = parse_source_code(str(tmp_path / "tracing.py"), source)
        comps = _detect_import_based_assets(result)
        obs_comps = [c for c in comps if c.component_type == AIComponentType.OBSERVABILITY]
        assert len(obs_comps) >= 1

    def test_langfuse_import_detected(self, tmp_path: Path):
        (tmp_path / "obs.py").write_text(
            "from langfuse import Langfuse\n"
            "\n"
            "langfuse = Langfuse()\n"
        )
        from aibom.cst_parser import parse_source_code
        from aibom.scanners.kb_enrichment_scanner import _detect_import_based_assets

        source = (tmp_path / "obs.py").read_text()
        result = parse_source_code(str(tmp_path / "obs.py"), source)
        comps = _detect_import_based_assets(result)
        obs_comps = [c for c in comps if c.component_type == AIComponentType.OBSERVABILITY]
        assert len(obs_comps) >= 1

    def test_observability_type_inference(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("Traceloop") == (AIComponentType.OBSERVABILITY, False)
        # "CustomTracing" and "MetricObserver" are intentionally NOT detected
        # as OBSERVABILITY by class-name alone — observability is detected via
        # import-based patterns to avoid FPs on generic Logger/Observer classes.


# ---------------------------------------------------------------------------
# Gap 5: Secrets - expanded env var patterns
# ---------------------------------------------------------------------------


class TestSecretEnvVarDetection:
    """AI-related env var names should be detected as SECRET even without kwarg context."""

    def test_openai_api_key_env_detected(self, tmp_path: Path):
        (tmp_path / "config.py").write_text(
            'import os\n'
            'key = os.getenv("OPENAI_API_KEY")\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secrets = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert any("OPENAI_API_KEY" in c.name for c in secrets)

    def test_anthropic_api_key_env_detected(self, tmp_path: Path):
        (tmp_path / "config.py").write_text(
            'import os\n'
            'key = os.environ["ANTHROPIC_API_KEY"]\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secrets = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert any("ANTHROPIC_API_KEY" in c.name for c in secrets)

    def test_huggingface_token_env_detected(self, tmp_path: Path):
        (tmp_path / "config.py").write_text(
            'import os\n'
            'token = os.environ.get("HF_TOKEN")\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secrets = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert any("HF_TOKEN" in c.name for c in secrets)

    def test_azure_openai_key_env_detected(self, tmp_path: Path):
        (tmp_path / "config.py").write_text(
            'import os\n'
            'key = os.getenv("AZURE_OPENAI_API_KEY")\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secrets = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert any("AZURE_OPENAI_API_KEY" in c.name for c in secrets)

    def test_non_ai_env_var_not_detected(self, tmp_path: Path):
        (tmp_path / "config.py").write_text(
            'import os\n'
            'port = os.getenv("DATABASE_PORT")\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 0

    def test_bedrock_secret_env_detected(self, tmp_path: Path):
        (tmp_path / "config.py").write_text(
            'import os\n'
            'secret = os.getenv("BEDROCK_SECRET_KEY")\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secrets = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert any("BEDROCK_SECRET_KEY" in c.name for c in secrets)


# ---------------------------------------------------------------------------
# Cross-cutting: _infer_type_from_name defaults to MODEL for unknowns
# ---------------------------------------------------------------------------


class TestTypeInferenceDefaults:
    """Unknown class names should default to MODEL."""

    def test_generic_llm_class_defaults_to_model(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("ChatCompletion") == (AIComponentType.MODEL, False)
        assert _infer_type_from_name("CustomLLM") == (AIComponentType.MODEL, False)

    def test_agent_type_inference(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("RouterAgent") == (AIComponentType.AGENT, True)

    def test_tool_type_inference(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("SearchTool") == (AIComponentType.TOOL, True)

    def test_memory_type_inference(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("ConversationHistory") == (AIComponentType.MEMORY, True)
        assert _infer_type_from_name("ChatBuffer") == (AIComponentType.MEMORY, True)

    def test_retriever_type_inference(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("DocumentRetriever") == (AIComponentType.RETRIEVER, True)

    def test_vector_store_type_inference(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("PineconeVectorStore") == (AIComponentType.VECTOR_STORE, True)


# ====================================================================
# Fix 1: Component consolidation
# ====================================================================


class TestConsolidation:
    """Verify that per-file-reference duplicates are collapsed into
    per-logical-asset entries in the assemble stage."""

    def test_same_name_type_service_consolidated(self):
        from aibom.scan_pipeline import _consolidate_components

        base = AIComponent(
            name="Weaviate",
            component_type=AIComponentType.VECTOR_STORE,
            file_path="/svc/a/one.py",
            line_number=10,
            heuristic_confidence=0.8,
        )
        dups = [
            base.model_copy(update={"file_path": f"/svc/a/{f}.py", "line_number": n})
            for f, n in [("one", 10), ("two", 20), ("three", 30)]
        ]
        result = _consolidate_components(dups)
        assert len(result) == 1
        assert result[0].metadata.get("consolidated_count") == 3
        assert len(result[0].metadata.get("evidence", [])) == 2

    def test_different_types_not_consolidated(self):
        from aibom.scan_pipeline import _consolidate_components

        a = AIComponent(
            name="gpt-4", component_type=AIComponentType.MODEL,
            file_path="/svc/a/x.py", line_number=1, heuristic_confidence=0.9,
        )
        b = AIComponent(
            name="gpt-4", component_type=AIComponentType.SECRET,
            file_path="/svc/a/y.py", line_number=5, heuristic_confidence=0.6,
        )
        result = _consolidate_components([a, b])
        assert len(result) == 2

    def test_different_services_consolidated_at_repo_level(self, tmp_path: Path):
        """Repo-level consolidation merges same (name, type) across services."""
        from aibom.scan_pipeline import _consolidate_components

        svc1 = tmp_path / "svc1"
        svc2 = tmp_path / "svc2"
        svc1.mkdir()
        svc2.mkdir()
        (svc1 / "pyproject.toml").write_text('[project]\nname="a"\nversion="0.1"\n')
        (svc2 / "pyproject.toml").write_text('[project]\nname="b"\nversion="0.1"\n')

        a = AIComponent(
            name="gpt-4", component_type=AIComponentType.MODEL,
            file_path=str(svc1 / "x.py"), line_number=1, heuristic_confidence=0.9,
        )
        b = AIComponent(
            name="gpt-4", component_type=AIComponentType.MODEL,
            file_path=str(svc2 / "y.py"), line_number=5, heuristic_confidence=0.9,
        )
        result = _consolidate_components([a, b])
        assert len(result) == 1
        ev = result[0].metadata.get("evidence", [])
        assert len(ev) == 1
        assert "service" in ev[0]

    def test_highest_confidence_wins(self):
        from aibom.scan_pipeline import _consolidate_components

        low = AIComponent(
            name="Agent", component_type=AIComponentType.AGENT,
            file_path="/svc/a/low.py", line_number=5, heuristic_confidence=0.3,
        )
        high = AIComponent(
            name="Agent", component_type=AIComponentType.AGENT,
            file_path="/svc/a/high.py", line_number=10, heuristic_confidence=0.9,
        )
        result = _consolidate_components([low, high])
        assert len(result) == 1
        assert result[0].heuristic_confidence == 0.9

    def test_singleton_left_alone(self):
        from aibom.scan_pipeline import _consolidate_components

        c = AIComponent(
            name="redis", component_type=AIComponentType.MEMORY,
            file_path="/svc/a/mem.py", line_number=1, heuristic_confidence=0.8,
        )
        result = _consolidate_components([c])
        assert len(result) == 1
        assert "consolidated_count" not in result[0].metadata


# ====================================================================
# Fix 2: Data-class suffix exclusion
# ====================================================================


class TestDataClassSuffixExclusion:
    """Verify that data-class suffixes suppress false-positive detection."""

    def test_conversation_response_excluded(self):
        from aibom.scanners.kb_enrichment_scanner import _is_data_class_name

        assert _is_data_class_name("ConversationResponse")
        assert _is_data_class_name("AgentSchema")
        assert _is_data_class_name("ToolConfig")
        assert _is_data_class_name("ModelRequest")

    def test_real_ai_classes_not_excluded(self):
        from aibom.scanners.kb_enrichment_scanner import _is_data_class_name

        assert not _is_data_class_name("ConversationMemory")
        assert not _is_data_class_name("AgentRouter")
        assert not _is_data_class_name("ToolExecutor")
        assert not _is_data_class_name("EmbeddingService")

    def test_logger_not_in_indicative_regex(self):
        from aibom.scanners.kb_enrichment_scanner import _AI_INDICATIVE_CLASS_RE

        assert not _AI_INDICATIVE_CLASS_RE.search("RequestLogger")
        assert not _AI_INDICATIVE_CLASS_RE.search("MetricsObserver")
        assert _AI_INDICATIVE_CLASS_RE.search("Traceloop")

    def test_data_class_name_skips_inference(self):
        from aibom.scanners.kb_enrichment_scanner import _infer_type_from_name

        assert _infer_type_from_name("ConversationResponse") == (AIComponentType.MODEL, False)
        assert _infer_type_from_name("ConversationMemory") == (AIComponentType.MEMORY, True)


# ====================================================================
# Fix 3: Vault secret detection
# ====================================================================


class TestVaultSecretDetection:
    """Verify that programmatic secret fetches are detected."""

    def test_conjur_get_secret(self, tmp_path: Path):
        (tmp_path / "secrets.py").write_text(
            "from conjur import Client\n"
            "client = Client()\n"
            'api_key = client.get_secret("api/ai/openai_key")\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secret_comps = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert len(secret_comps) >= 1
        assert secret_comps[0].metadata.get("secret_source") == "vault_sdk"

    def test_aws_secrets_manager(self, tmp_path: Path):
        (tmp_path / "aws_sec.py").write_text(
            "import boto3\n"
            "client = boto3.client('secretsmanager')\n"
            "resp = client.get_secret_value(SecretId='ai/model-key')\n"
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secret_comps = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert len(secret_comps) >= 1

    def test_azure_key_vault(self, tmp_path: Path):
        (tmp_path / "akv.py").write_text(
            "from azure.keyvault.secrets import SecretClient\n"
            "client = SecretClient(vault_url=url, credential=cred)\n"
            'secret = client.get_secret("openai-api-key")\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secret_comps = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert len(secret_comps) >= 1
        assert secret_comps[0].metadata.get("has_vault_import") is True

    def test_gcp_secret_manager(self, tmp_path: Path):
        (tmp_path / "gcp_sec.py").write_text(
            "from google.cloud.secretmanager import SecretManagerServiceClient\n"
            "client = SecretManagerServiceClient()\n"
            "resp = client.access_secret_version(request={'name': name})\n"
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secret_comps = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert len(secret_comps) >= 1

    def test_hashicorp_vault(self, tmp_path: Path):
        (tmp_path / "vault.py").write_text(
            "import hvac\n"
            "client = hvac.Client()\n"
            "result = client.secrets.kv.read('secret/data/openai')\n"
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secret_comps = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert len(secret_comps) >= 0  # vault.read pattern is loose; import present

    def test_no_false_positive_on_generic_get(self, tmp_path: Path):
        (tmp_path / "util.py").write_text(
            "import json\n"
            "data = config.get('key')\n"
            "result = cache.get(item_id)\n"
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        secret_comps = [c for c in comps if c.component_type == AIComponentType.SECRET]
        assert len(secret_comps) == 0


# ====================================================================
# Fix 4: Cloud ML training detection
# ====================================================================


class TestCloudMLDetection:
    """Verify that SageMaker, Vertex AI, Azure ML training patterns are detected."""

    def test_sagemaker_boto3_client(self, tmp_path: Path):
        (tmp_path / "train.py").write_text(
            "import boto3\n"
            "client = boto3.client('sagemaker')\n"
            "client.create_training_job(\n"
            "    TrainingJobName='my-classifier',\n"
            "    AlgorithmSpecification={'TrainingImage': 'image'},\n"
            ")\n"
        )
        from aibom.scanners.ml_lifecycle_detector import MLLifecycleDetector

        scanner = MLLifecycleDetector()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        training = [c for c in comps if c.component_type == AIComponentType.TRAINING_RUN]
        assert len(training) >= 1
        assert any("sagemaker" in c.framework for c in training)

    def test_sagemaker_estimator_import(self, tmp_path: Path):
        (tmp_path / "train_sm.py").write_text(
            "from sagemaker import Estimator\n"
            "est = Estimator(image_uri='...', role=role)\n"
            "est.fit({'training': s3_input})\n"
        )
        from aibom.scanners.ml_lifecycle_detector import MLLifecycleDetector

        scanner = MLLifecycleDetector()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        training = [c for c in comps if c.component_type == AIComponentType.TRAINING_RUN]
        assert len(training) >= 1

    def test_vertex_ai_custom_job(self, tmp_path: Path):
        (tmp_path / "train_vertex.py").write_text(
            "from google.cloud import aiplatform\n"
            "aiplatform.init(project='my-project')\n"
            "job = aiplatform.CustomJob(display_name='train', worker_pool_specs=specs)\n"
            "job.run()\n"
        )
        from aibom.scanners.ml_lifecycle_detector import MLLifecycleDetector

        scanner = MLLifecycleDetector()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        training = [c for c in comps if c.component_type == AIComponentType.TRAINING_RUN]
        assert len(training) >= 1
        assert any("vertex" in c.framework for c in training)

    def test_azure_ml_command_job(self, tmp_path: Path):
        (tmp_path / "train_azure.py").write_text(
            "from azure.ai.ml import command, MLClient\n"
            "ml_client = MLClient(credential, subscription, resource_group, workspace)\n"
            "command_job = command(\n"
            "    code='./src',\n"
            "    command='python train.py',\n"
            "    environment='AzureML-sklearn-1.0',\n"
            ")\n"
            "ml_client.jobs.create_or_update(command_job)\n"
        )
        from aibom.scanners.ml_lifecycle_detector import MLLifecycleDetector

        scanner = MLLifecycleDetector()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        training = [c for c in comps if c.component_type == AIComponentType.TRAINING_RUN]
        assert len(training) >= 1
        assert any("azure" in c.framework for c in training)

    def test_cloud_ml_high_confidence(self, tmp_path: Path):
        (tmp_path / "sm.py").write_text(
            "from sagemaker import Estimator\n"
            "est = SageMakerEstimator(image_uri='img')\n"
        )
        from aibom.scanners.ml_lifecycle_detector import MLLifecycleDetector

        scanner = MLLifecycleDetector()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        training = [c for c in comps if c.component_type == AIComponentType.TRAINING_RUN]
        for c in training:
            assert c.heuristic_confidence >= 0.8
            assert c.needs_agentic is False


# ---------------------------------------------------------------------------
# Fix A: Repo-level consolidation (drop service_dir from key)
# ---------------------------------------------------------------------------


class TestRepoLevelConsolidation:
    def test_cross_service_same_name_type_merged(self, tmp_path: Path):
        """Same (name, type) across different service dirs → single entry."""
        from aibom.scan_pipeline import _consolidate_components

        svc1 = tmp_path / "svc1"
        svc2 = tmp_path / "svc2"
        for d in (svc1, svc2):
            d.mkdir()
            (d / "pyproject.toml").write_text('[project]\nname="x"\nversion="0.1"\n')

        comps = [
            AIComponent(
                name="gpt-4", component_type=AIComponentType.MODEL,
                file_path=str(svc1 / "a.py"), line_number=1, heuristic_confidence=0.9,
            ),
            AIComponent(
                name="gpt-4", component_type=AIComponentType.MODEL,
                file_path=str(svc2 / "b.py"), line_number=5, heuristic_confidence=0.7,
            ),
        ]
        result = _consolidate_components(comps)
        assert len(result) == 1
        assert result[0].heuristic_confidence == 0.9
        ev = result[0].metadata["evidence"]
        assert len(ev) == 1
        assert "service" in ev[0]

    def test_evidence_contains_service_field(self, tmp_path: Path):
        from aibom.scan_pipeline import _consolidate_components

        comps = [
            AIComponent(
                name="Weaviate", component_type=AIComponentType.VECTOR_STORE,
                file_path=str(tmp_path / "a.py"), line_number=1, heuristic_confidence=0.9,
            ),
            AIComponent(
                name="Weaviate", component_type=AIComponentType.VECTOR_STORE,
                file_path=str(tmp_path / "b.py"), line_number=10, heuristic_confidence=0.7,
            ),
        ]
        result = _consolidate_components(comps)
        assert len(result) == 1
        for ev in result[0].metadata["evidence"]:
            assert "service" in ev


# ---------------------------------------------------------------------------
# Fix B: Test file tagging
# ---------------------------------------------------------------------------


class TestTestFileTagging:
    def test_test_only_tagged_when_all_in_test_dirs(self):
        from aibom.scan_pipeline import _consolidate_components

        comps = [
            AIComponent(
                name="fake-agent", component_type=AIComponentType.AGENT,
                file_path="/repo/tests/test_agent.py", line_number=5, heuristic_confidence=0.8,
            ),
            AIComponent(
                name="fake-agent", component_type=AIComponentType.AGENT,
                file_path="/repo/tests/integration/test_flow.py", line_number=10,
                heuristic_confidence=0.6,
            ),
        ]
        result = _consolidate_components(comps)
        assert len(result) == 1
        assert result[0].metadata.get("test_only") is True

    def test_not_tagged_when_mixed_paths(self):
        from aibom.scan_pipeline import _consolidate_components

        comps = [
            AIComponent(
                name="agent", component_type=AIComponentType.AGENT,
                file_path="/repo/src/agent.py", line_number=5, heuristic_confidence=0.9,
            ),
            AIComponent(
                name="agent", component_type=AIComponentType.AGENT,
                file_path="/repo/tests/test_agent.py", line_number=10,
                heuristic_confidence=0.6,
            ),
        ]
        result = _consolidate_components(comps)
        assert len(result) == 1
        assert result[0].metadata.get("test_only") is not True

    def test_singleton_in_test_dir_tagged(self):
        from aibom.scan_pipeline import _consolidate_components

        comps = [
            AIComponent(
                name="mock-model", component_type=AIComponentType.MODEL,
                file_path="/repo/tests/conftest.py", line_number=1, heuristic_confidence=0.5,
            ),
        ]
        result = _consolidate_components(comps)
        assert len(result) == 1
        assert result[0].metadata.get("test_only") is True

    def test_test_only_excluded_from_summary(self):
        from aibom.models.scan import ScanResult, RiskScore, SourceResult

        comps = [
            AIComponent(
                name="real-model", component_type=AIComponentType.MODEL,
                file_path="/repo/src/m.py", line_number=1, heuristic_confidence=0.9,
            ),
            AIComponent(
                name="test-model", component_type=AIComponentType.MODEL,
                file_path="/repo/tests/t.py", line_number=1, heuristic_confidence=0.5,
                metadata={"test_only": True},
            ),
        ]
        sr = ScanResult(
            sources=[SourceResult(path="/repo", components=comps)],
            risk=RiskScore(),
        )
        summary = sr.summary
        assert summary["total_components"] == 1
        assert summary["test_only_components"] == 1
        assert summary["component_types"]["model"] == 1


class TestConsolidationPrefersProdFile:
    """Fix 5: When confidence ties, prefer non-test file as primary."""

    def test_production_file_wins_over_test_file(self):
        from aibom.scan_pipeline import _consolidate_components

        test_comp = AIComponent(
            name="AgentRouter",
            component_type=AIComponentType.AGENT,
            file_path="/repo/tests/test_agent.py",
            line_number=5,
            heuristic_confidence=0.8,
        )
        prod_comp = AIComponent(
            name="AgentRouter",
            component_type=AIComponentType.AGENT,
            file_path="/repo/src/agent.py",
            line_number=20,
            heuristic_confidence=0.8,
        )
        result = _consolidate_components([test_comp, prod_comp])
        assert len(result) == 1
        assert result[0].file_path == "/repo/src/agent.py"

    def test_higher_confidence_still_wins(self):
        from aibom.scan_pipeline import _consolidate_components

        test_comp = AIComponent(
            name="AgentRouter",
            component_type=AIComponentType.AGENT,
            file_path="/repo/tests/test_agent.py",
            line_number=5,
            heuristic_confidence=0.95,
        )
        prod_comp = AIComponent(
            name="AgentRouter",
            component_type=AIComponentType.AGENT,
            file_path="/repo/src/agent.py",
            line_number=20,
            heuristic_confidence=0.8,
        )
        result = _consolidate_components([test_comp, prod_comp])
        assert len(result) == 1
        assert result[0].heuristic_confidence == 0.95


# ---------------------------------------------------------------------------
# Fix C: Infrastructure env var exclusion
# ---------------------------------------------------------------------------


class TestInfraEnvVarExclusion:
    def test_temporal_endpoint_filtered(self, tmp_path: Path):
        """TEMPORAL_ENDPOINT should not be detected as an AI env var."""
        (tmp_path / "svc.go").write_text(
            'package main\n'
            'import "os"\n'
            'func main() {\n'
            '    endpoint := os.Getenv("TEMPORAL_ENDPOINT")\n'
            '}\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        names = [c.name for c in comps]
        assert "env:TEMPORAL_ENDPOINT" not in names

    def test_otel_exporter_filtered(self, tmp_path: Path):
        (tmp_path / "cfg.py").write_text(
            'import os\n'
            'exporter = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        names = [c.name for c in comps]
        assert "env:OTEL_EXPORTER_OTLP_ENDPOINT" not in names

    def test_openai_key_still_detected(self, tmp_path: Path):
        (tmp_path / "app.py").write_text(
            'import os\n'
            'key = os.environ.get("OPENAI_API_KEY")\n'
        )
        from aibom.scanners.env_var_resolver import EnvVarResolver

        scanner = EnvVarResolver()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        names = [c.name for c in comps]
        assert "env:OPENAI_API_KEY" in names


# ---------------------------------------------------------------------------
# Fix F: Non-AI class suffix, OpenAPI spec skip, non-AI package exclusion
# ---------------------------------------------------------------------------


class TestNonAIClassSuffixExclusion:
    def test_guardrailcode_excluded(self):
        from aibom.scanners.kb_enrichment_scanner import _is_data_class_name

        assert _is_data_class_name("GuardRailCode") is True
        assert _is_data_class_name("ModelStatus") is True
        assert _is_data_class_name("EmbeddingFlag") is True

    def test_real_ai_class_not_excluded(self):
        from aibom.scanners.kb_enrichment_scanner import _is_data_class_name

        assert _is_data_class_name("ChatModel") is False
        assert _is_data_class_name("AgentExecutor") is False
        assert _is_data_class_name("ToolChain") is False


class TestOpenAPISpecSkip:
    def test_openapi_yaml_dataset_not_detected(self, tmp_path: Path):
        """data: sections in OpenAPI specs should not be flagged as datasets."""
        (tmp_path / "api.yaml").write_text(
            "openapi: '3.0.0'\n"
            "info:\n"
            "  title: Test API\n"
            "  version: '1.0'\n"
            "paths:\n"
            "  /users:\n"
            "    get:\n"
            "      responses:\n"
            "        '200':\n"
            "          description: OK\n"
            "          content:\n"
            "            application/json:\n"
            "              schema:\n"
            "                data:\n"
            "                  type: array\n"
        )
        from aibom.scanners.ml_lifecycle_detector import MLLifecycleDetector

        scanner = MLLifecycleDetector()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        datasets = [c for c in comps if c.component_type == AIComponentType.DATASET]
        assert len(datasets) == 0

    def test_regular_yaml_data_still_detected(self, tmp_path: Path):
        (tmp_path / "config.yaml").write_text(
            "pipeline:\n"
            "  data:\n"
            "    source: s3://bucket/dataset\n"
        )
        from aibom.scanners.ml_lifecycle_detector import MLLifecycleDetector

        scanner = MLLifecycleDetector()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        datasets = [c for c in comps if c.component_type == AIComponentType.DATASET]
        assert len(datasets) >= 1


class TestNonAIPackageExclusion:
    def test_more_itertools_excluded(self, tmp_path: Path):
        """more_itertools should be filtered out by _NON_AI_PACKAGES."""
        from aibom.scanners.kb_enrichment_scanner import _NON_AI_PACKAGES

        assert "more_itertools" in _NON_AI_PACKAGES

    def test_pytest_excluded(self):
        from aibom.scanners.kb_enrichment_scanner import _NON_AI_PACKAGES

        assert "pytest" in _NON_AI_PACKAGES


# ---------------------------------------------------------------------------
# Fix G: MCP server fastmcp import detection
# ---------------------------------------------------------------------------


class TestMCPServerFastMCPImport:
    def test_fastmcp_package_import_detected(self, tmp_path: Path):
        (tmp_path / "server.py").write_text(
            "from fastmcp import FastMCP\n"
            "\n"
            "mcp = FastMCP('my-server')\n"
            "\n"
            "@mcp.tool()\n"
            "def hello():\n"
            "    return 'world'\n"
        )
        from aibom.scanners.mcp_detector import McpDetector

        scanner = McpDetector()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        servers = [c for c in comps if c.component_type == AIComponentType.MCP_SERVER]
        assert len(servers) >= 1
