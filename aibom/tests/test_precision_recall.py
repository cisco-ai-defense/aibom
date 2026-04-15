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

"""Tests for Phase 6 precision/recall improvements (6.0a-6.0g)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aibom.models.enums import AIComponentType, DetectionSource
from aibom.models.scan import AIComponent, ScanContext
from aibom.scanners.dependency_scanner import DependencyScanner


class TestDedupDependencies:
    """6.0a: Dedup duplicate dependencies."""

    def test_same_package_in_two_manifests_deduped(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("langchain==0.3.1\n")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "0.1"\n'
            "dependencies = [\n"
            '    "langchain>=0.3.0",\n'
            "]\n"
        )
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        langchain_comps = [c for c in comps if "langchain" in c.name.lower()]
        assert len(langchain_comps) == 1

    def test_pinned_version_preserved_on_dedup(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "0.1"\n'
            "dependencies = [\n"
            '    "torch>=2.0",\n'
            "]\n"
        )
        (tmp_path / "requirements.txt").write_text("torch==2.5.1\n")
        scanner = DependencyScanner()
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        torch_comps = [c for c in comps if "torch" in c.name.lower()]
        assert len(torch_comps) == 1
        assert torch_comps[0].sdk_version is not None


class TestToolSchemaDetection:
    """6.0b: Tool schema detection via call context."""

    def test_tool_decorator_from_langchain(self, tmp_path: Path):
        (tmp_path / "tools.py").write_text(
            "from langchain_core.tools import tool\n"
            "\n"
            "@tool\n"
            "def search_web(query: str) -> str:\n"
            "    return query\n"
        )
        from aibom.scanners.kb_enrichment_scanner import _detect_tool_schemas
        from aibom.cst_parser import parse_source_code
        from aibom.scanners.file_cache import read_text_cached

        source = (tmp_path / "tools.py").read_text()
        result = parse_source_code(str(tmp_path / "tools.py"), source)
        comps = _detect_tool_schemas(result)
        tool_comps = [c for c in comps if c.component_type == AIComponentType.TOOL]
        assert any(c.name == "search_web" for c in tool_comps)

    def test_function_to_schema_call(self, tmp_path: Path):
        (tmp_path / "schemas.py").write_text(
            "from langchain_core.utils.function_calling import function_to_schema\n"
            "schema = function_to_schema(my_function)\n"
        )
        from aibom.scanners.kb_enrichment_scanner import _detect_tool_schemas
        from aibom.cst_parser import parse_source_code

        source = (tmp_path / "schemas.py").read_text()
        result = parse_source_code(str(tmp_path / "schemas.py"), source)
        comps = _detect_tool_schemas(result)
        assert any(c.component_type == AIComponentType.TOOL for c in comps)


class TestPromptDetection:
    """6.0c: Prompt detection via call context."""

    def test_system_prompt_kwarg(self, tmp_path: Path):
        (tmp_path / "agent.py").write_text(
            "from deepagents import create_deep_agent\n"
            'agent = create_deep_agent(system_prompt="You are a helpful assistant.")\n'
        )
        from aibom.scanners.kb_enrichment_scanner import _detect_prompt_kwargs
        from aibom.cst_parser import parse_source_code

        source = (tmp_path / "agent.py").read_text()
        result = parse_source_code(str(tmp_path / "agent.py"), source)
        comps = _detect_prompt_kwargs(result)
        prompt_comps = [c for c in comps if c.component_type == AIComponentType.PROMPT]
        assert len(prompt_comps) >= 1
        assert prompt_comps[0].text == "You are a helpful assistant."

    def test_resolves_ai_client_chain_for_prompt_kwargs(self, tmp_path: Path):
        (tmp_path / "client.py").write_text(
            "import openai\n"
            "client = openai.OpenAI()\n"
            'response = client.responses.create(instructions="Answer briefly.")\n'
        )
        from aibom.scanners.kb_enrichment_scanner import _detect_prompt_kwargs
        from aibom.cst_parser import parse_source_code

        source = (tmp_path / "client.py").read_text()
        result = parse_source_code(str(tmp_path / "client.py"), source)
        comps = _detect_prompt_kwargs(result)
        prompt_comps = [c for c in comps if c.component_type == AIComponentType.PROMPT]
        assert len(prompt_comps) == 1
        assert prompt_comps[0].text == "Answer briefly."
        assert prompt_comps[0].metadata["enclosing_call"] == "openai.OpenAI.responses.create"

    def test_ignores_prompt_kwargs_on_non_ai_helpers(self, tmp_path: Path):
        (tmp_path / "helpers.py").write_text(
            "def render_prompt(prompt: str):\n"
            "    return prompt\n"
            "\n"
            'render_prompt(prompt="local helper text")\n'
        )
        from aibom.scanners.kb_enrichment_scanner import _detect_prompt_kwargs
        from aibom.cst_parser import parse_source_code

        source = (tmp_path / "helpers.py").read_text()
        result = parse_source_code(str(tmp_path / "helpers.py"), source)
        comps = _detect_prompt_kwargs(result)
        assert [c for c in comps if c.component_type == AIComponentType.PROMPT] == []

    def test_ignores_variable_messages_on_non_ai_methods(self, tmp_path: Path):
        (tmp_path / "helpers.py").write_text(
            "class RequestBuilder:\n"
            "    def create(self, messages):\n"
            "        return messages\n"
            "\n"
            "payload = ['status']\n"
            "builder = RequestBuilder()\n"
            "request = builder.create(messages=payload)\n"
        )
        from aibom.scanners.kb_enrichment_scanner import _detect_prompt_kwargs
        from aibom.cst_parser import parse_source_code

        source = (tmp_path / "helpers.py").read_text()
        result = parse_source_code(str(tmp_path / "helpers.py"), source)
        comps = _detect_prompt_kwargs(result)
        assert [c for c in comps if c.component_type == AIComponentType.PROMPT] == []


class TestEmbeddingDetection:
    """6.0d: Cross-repo embedding detection via suggestive-signal regex."""

    def test_embedder_class_matches(self):
        import re
        from aibom.scanners.kb_enrichment_scanner import _AI_INDICATIVE_CLASS_RE

        assert _AI_INDICATIVE_CLASS_RE.search("OpenAIEmbedder")
        assert _AI_INDICATIVE_CLASS_RE.search("CustomEmbedding")
        assert _AI_INDICATIVE_CLASS_RE.search("VectorEncoder")
        assert not _AI_INDICATIVE_CLASS_RE.search("HttpClient")


class TestMemoryDetection:
    """6.0e: Memory detection via indicative class regex."""

    def test_memory_classes_match(self):
        from aibom.scanners.kb_enrichment_scanner import _AI_INDICATIVE_CLASS_RE

        assert _AI_INDICATIVE_CLASS_RE.search("ConversationHistory")
        assert _AI_INDICATIVE_CLASS_RE.search("ChatBuffer")
        assert _AI_INDICATIVE_CLASS_RE.search("SessionMemory")


class TestCacheCoOccurrence:
    """6.0e: Import co-occurrence detection."""

    def test_has_cache_imports(self):
        from aibom.scanners.import_context import has_cache_imports

        assert has_cache_imports("import redis\n")
        assert has_cache_imports("from cachetools import TTLCache\n")
        assert not has_cache_imports("import json\n")


class TestVectorStoreEnvVar:
    """6.0g: Vector store env var resolution."""

    def test_vector_store_resolved(self):
        from aibom.cross_ref import resolve_components, CrossRefIndex, EnvVarEntry

        comp = AIComponent(
            name="WeaviateVectorIndex",
            component_type=AIComponentType.VECTOR_STORE,
            file_path="test.py",
            line_number=1,
            metadata={"env": "WEAVIATE_CLASS_NAME"},
            needs_agentic=True,
        )
        env_idx = CrossRefIndex()
        env_idx.env["WEAVIATE_CLASS_NAME"] = [
            EnvVarEntry(name="WEAVIATE_CLASS_NAME", value="FMC_Annotations", source_type="env_file", source_path=".env")
        ]
        resolved = resolve_components([comp], env_idx)
        assert len(resolved) == 1
        assert resolved[0].metadata.get("index_name") == "FMC_Annotations"
        assert resolved[0].needs_agentic is False
