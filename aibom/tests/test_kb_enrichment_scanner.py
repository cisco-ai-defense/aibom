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

"""Tests for the KB Enrichment Scanner.

These tests exercise the scanner's filtering, matching tiers, framework
disambiguation, and graceful fallback behaviour.  Tests that hit the live
DuckDB KB (from ``~/.aibom/catalogs/``) are marked ``integration`` so they
can be skipped in CI where the KB may not be present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from aibom.cst_parser import parse_source_code
from aibom.models import AIComponentType, DetectionSource, ScanContext
from aibom.scanners.kb_enrichment_scanner import (
    ALLOWED_CONCEPTS,
    KBEnrichmentScanner,
    _AGENT_FRAMEWORK_PREFIXES,
    _BARE_CLIENT_HINTS,
    _KNOWN_AI_CLIENT_CLASSES,
    _MatchResult,
    _build_kb_patterns,
    _detect_model_kwargs,
    _emit_suggestive_candidates,
    _extract_class_segment,
    _extract_leaf_class,
    _frameworks_related,
    _has_suggestive_signal,
    _match_observation_rich,
    _process_file_with_cache,
    _resolve_kb_path,
)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestExtractClassSegment:
    def test_simple_dotted(self):
        assert _extract_class_segment("langchain_openai.ChatOpenAI") == "ChatOpenAI"

    def test_factory_method(self):
        assert _extract_class_segment("langchain_community.vectorstores.FAISS.from_texts") == "FAISS"

    def test_deep_method_chain(self):
        assert _extract_class_segment("openai.OpenAI.chat.completions.create") == "OpenAI"

    def test_single_name(self):
        assert _extract_class_segment("ChatOpenAI") is None

    def test_all_lowercase(self):
        assert _extract_class_segment("foo.bar.baz") is None

    def test_module_only(self):
        assert _extract_class_segment("crewai.agent") is None


class TestFrameworksRelated:
    def test_same_family(self):
        assert _frameworks_related("langchain_openai", "langchain_community") is True

    def test_exact_match(self):
        assert _frameworks_related("crewai", "crewai") is True

    def test_unrelated(self):
        assert _frameworks_related("openai", "langchain_community") is False

    def test_empty_module(self):
        assert _frameworks_related("", "langchain_community") is True

    def test_empty_framework(self):
        assert _frameworks_related("openai", "") is True


class TestMatchObservationRich:
    """Unit tests for _match_observation_rich with a fake kb_by_id."""

    @pytest.fixture()
    def sample_kb(self) -> dict[str, dict[str, Any]]:
        return {
            "langchain_openai.ChatOpenAI": {
                "id": "langchain_openai.ChatOpenAI",
                "concept": "model",
                "framework": "langchain_openai",
            },
            "langchain_community.vectorstores.faiss.FAISS": {
                "id": "langchain_community.vectorstores.faiss.FAISS",
                "concept": "datastore",
                "framework": "langchain_community",
            },
            "crewai.Agent": {
                "id": "crewai.Agent",
                "concept": "agent",
                "framework": "crewai",
            },
        }

    def test_tier1_exact(self, sample_kb: dict):
        result = _match_observation_rich("crewai.Agent", sample_kb, {"crewai"})
        assert result.is_confirmed
        assert result.entry is not None
        assert result.entry["concept"] == "agent"

    def test_tier2_suffix(self, sample_kb: dict):
        result = _match_observation_rich(
            "ChatOpenAI",
            {"langchain_openai.ChatOpenAI": sample_kb["langchain_openai.ChatOpenAI"]},
            {"langchain_openai"},
        )
        assert result.is_confirmed
        assert result.entry is not None
        assert result.entry["concept"] == "model"

    def test_tier3_class_segment(self, sample_kb: dict):
        result = _match_observation_rich(
            "langchain_community.vectorstores.FAISS.from_texts",
            sample_kb,
            {"langchain_community"},
        )
        assert result.is_confirmed
        assert result.entry is not None
        assert result.entry["concept"] == "datastore"

    def test_no_match(self, sample_kb: dict):
        result = _match_observation_rich("totally.Unknown", sample_kb, set())
        assert not result.is_confirmed
        assert not result.is_partial

    def test_partial_match_framework_mismatch(self):
        """Leaf class matches KB but import module differs — MAYBE path."""
        kb = {
            "langchain_community.llms.openai.OpenAI": {
                "id": "langchain_community.llms.openai.OpenAI",
                "concept": "model",
                "framework": "langchain_community",
            },
        }
        result = _match_observation_rich("models.OpenAI", kb, {"models"})
        assert result.is_partial
        assert result.partial_kb_id == "langchain_community.llms.openai.OpenAI"
        assert result.partial_kb_framework == "langchain_community"
        assert result.obs_module == "models"


class TestMethodLabelFiltering:
    def test_kb_method_match_does_not_emit_vector_store_asset(self, tmp_path: Path):
        helper = tmp_path / "helper.py"
        helper.write_text(
            "def create_store_adapter(config):\n"
            "    return config\n"
            "\n"
            "store = create_store_adapter(config={})\n"
        )

        result = parse_source_code(str(helper), helper.read_text())
        kb_entries = {
            "langchain_community.utilities.fake.StoreFactory.create_store_adapter": {
                "id": "langchain_community.utilities.fake.StoreFactory.create_store_adapter",
                "concept": "datastore",
                "framework": "langchain_community",
                "label": "method",
            },
        }

        comps = _process_file_with_cache(result, kb_entries)
        assert [c for c in comps if c.component_type == AIComponentType.VECTOR_STORE] == []

    def test_kb_class_match_still_emits_vector_store_asset(self, tmp_path: Path):
        helper = tmp_path / "store.py"
        helper.write_text(
            "class FakeVectorStore:\n"
            "    pass\n"
            "\n"
            "store = FakeVectorStore()\n"
        )

        result = parse_source_code(str(helper), helper.read_text())
        kb_entries = {
            "langchain_community.vectorstores.fake.FakeVectorStore": {
                "id": "langchain_community.vectorstores.fake.FakeVectorStore",
                "concept": "datastore",
                "framework": "langchain_community",
                "label": "class",
            },
        }

        comps = _process_file_with_cache(result, kb_entries)
        stores = [c for c in comps if c.component_type == AIComponentType.VECTOR_STORE]
        assert len(stores) == 1
        assert stores[0].name == "FakeVectorStore"

    def test_partial_match_wrapper_library(self):
        """Wrapper class that matches KB leaf via class segment extraction."""
        kb = {
            "langchain_openai.ChatOpenAI": {
                "id": "langchain_openai.ChatOpenAI",
                "concept": "model",
                "framework": "langchain_openai",
            },
        }
        result = _match_observation_rich(
            "models.llm.openai.ChatOpenAI.invoke", kb, set(),
        )
        assert result.is_partial
        assert "ChatOpenAI" in (result.partial_kb_id or "")

    def test_confirmed_when_framework_matches(self):
        kb = {
            "langchain_openai.ChatOpenAI": {
                "id": "langchain_openai.ChatOpenAI",
                "concept": "model",
                "framework": "langchain_openai",
            },
        }
        result = _match_observation_rich("langchain_openai.ChatOpenAI", kb, {"langchain_openai"})
        assert result.is_confirmed

    def test_tier3_lowercase_short_name_skipped(self):
        kb = {
            "crewai.cli.cli.ToolCommand.create": {
                "id": "crewai.cli.cli.ToolCommand.create",
                "concept": "tool",
                "framework": "crewai",
            },
        }
        result = _match_observation_rich(
            "openai.OpenAI.chat.completions.create", kb, {"openai"}
        )
        assert not result.is_confirmed


def _kb_available() -> bool:
    ctx = ScanContext(paths=["/tmp"])
    return _resolve_kb_path(ctx) is not None


class TestExtractLeafClass:
    """Unit tests for _extract_leaf_class KB id parsing."""

    def test_valid_class(self):
        assert _extract_leaf_class("langchain_core.tools.tool.Tool") == "Tool"

    def test_valid_deep_class(self):
        assert (
            _extract_leaf_class("langchain_community.tools.ddg_search.tool.DuckDuckGoSearchRun")
            == "DuckDuckGoSearchRun"
        )

    def test_rejects_lowercase_leaf(self):
        assert _extract_leaf_class("langchain.tools.render.format_tool_to_openai_function") is None

    def test_rejects_all_uppercase(self):
        assert _extract_leaf_class("langchain.constants.API") is None

    def test_rejects_short_leaf(self):
        assert _extract_leaf_class("some.module.OK") is None

    def test_rejects_generic_class(self):
        assert _extract_leaf_class("langchain.tools.base.Awaitable") is None

    def test_rejects_data_class_suffix(self):
        assert _extract_leaf_class("langchain.agents.schema.AgentAction") is None

    def test_rejects_method_on_class(self):
        assert _extract_leaf_class("langchain_core.tools.Tool.from_function") is None

    def test_no_dot(self):
        assert _extract_leaf_class("Tool") is None

    def test_excluded_class_blocked(self):
        assert _extract_leaf_class("langchain.text_splitter.RecursiveCharacterTextSplitter") is None


@pytest.mark.skipif(not _kb_available(), reason="No KB DuckDB installed")
class TestBuildKBPatterns:
    """Integration tests for _build_kb_patterns against real KB."""

    def test_returns_all_component_types(self, tmp_path: Path):
        from aibom.catalog_db import CatalogDB

        ctx = ScanContext(paths=[str(tmp_path)])
        kb_path = _resolve_kb_path(ctx)
        assert kb_path is not None

        with CatalogDB(kb_path) as db:
            patterns = _build_kb_patterns(db)
            assert AIComponentType.TOOL in patterns
            assert AIComponentType.MEMORY in patterns
            assert AIComponentType.PROMPT in patterns
            assert AIComponentType.AGENT in patterns

    def test_tool_patterns_exceed_static(self, tmp_path: Path):
        from aibom.catalog_db import CatalogDB

        ctx = ScanContext(paths=[str(tmp_path)])
        kb_path = _resolve_kb_path(ctx)
        assert kb_path is not None

        with CatalogDB(kb_path) as db:
            patterns = _build_kb_patterns(db)
            assert len(patterns[AIComponentType.TOOL]) > 10

    def test_patterns_include_static_fallbacks(self, tmp_path: Path):
        from aibom.catalog_db import CatalogDB

        ctx = ScanContext(paths=[str(tmp_path)])
        kb_path = _resolve_kb_path(ctx)
        assert kb_path is not None

        with CatalogDB(kb_path) as db:
            patterns = _build_kb_patterns(db)
            assert "Tool" in patterns[AIComponentType.TOOL]
            assert "StructuredTool" in patterns[AIComponentType.TOOL]
            assert "ConversationBufferMemory" in patterns[AIComponentType.MEMORY]
            assert "PromptTemplate" in patterns[AIComponentType.PROMPT]

    def test_excluded_classes_not_in_patterns(self, tmp_path: Path):
        from aibom.catalog_db import CatalogDB

        ctx = ScanContext(paths=[str(tmp_path)])
        kb_path = _resolve_kb_path(ctx)
        assert kb_path is not None

        with CatalogDB(kb_path) as db:
            patterns = _build_kb_patterns(db)
            all_names = set()
            for names in patterns.values():
                all_names |= names
            assert "RecursiveCharacterTextSplitter" not in all_names
            assert "ABC" not in all_names

    def test_detects_duckduckgo_tool_from_kb(self, tmp_path: Path):
        """KB-derived patterns should include DuckDuckGoSearchRun without
        needing it in the static list."""
        (tmp_path / "search.py").write_text(
            "from langchain_community.tools import DuckDuckGoSearchRun\n"
            "search = DuckDuckGoSearchRun()\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        tools = [c for c in comps if c.component_type == AIComponentType.TOOL]
        assert any("DuckDuckGoSearchRun" in t.name or "search" in t.name for t in tools)


class TestAllowedConcepts:
    def test_expected_concepts(self):
        assert "agent" in ALLOWED_CONCEPTS
        assert "model" in ALLOWED_CONCEPTS
        assert "tool" in ALLOWED_CONCEPTS
        assert "datastore" in ALLOWED_CONCEPTS
        assert "embedding" in ALLOWED_CONCEPTS
        assert "prompt" in ALLOWED_CONCEPTS
        assert "memory" in ALLOWED_CONCEPTS
        assert "retriever" in ALLOWED_CONCEPTS

    def test_other_excluded(self):
        assert "other" not in ALLOWED_CONCEPTS


# ---------------------------------------------------------------------------
# Scanner-level tests (KB presence required)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _kb_available(), reason="No KB DuckDB installed")
class TestKBEnrichmentScannerIntegration:
    """Integration tests that exercise the scanner against the real KB."""

    def test_supports_with_kb(self, tmp_path: Path):
        ctx = ScanContext(paths=[str(tmp_path)])
        assert KBEnrichmentScanner().supports(ctx) is True

    def test_empty_directory(self, tmp_path: Path):
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, rels = KBEnrichmentScanner().scan(ctx)
        assert comps == []
        assert rels == []

    def test_detects_model_usage(self, tmp_path: Path):
        (tmp_path / "app.py").write_text(
            "from langchain_openai import ChatOpenAI\n"
            "llm = ChatOpenAI(model='gpt-4o')\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert len(models) >= 1
        assert models[0].kb_concept == "model"
        assert models[0].detection_source == DetectionSource.KB_ENRICHMENT

    def test_detects_tool_decorator(self, tmp_path: Path):
        (tmp_path / "tools.py").write_text(
            "from langchain_core.tools import tool\n"
            "@tool\n"
            "def greet(name: str) -> str:\n"
            "    return f'Hello {name}'\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        tools = [c for c in comps if c.component_type == AIComponentType.TOOL]
        assert len(tools) >= 1
        assert tools[0].name == "greet"

    def test_detects_agent(self, tmp_path: Path):
        (tmp_path / "graph.py").write_text(
            "from langgraph.graph import StateGraph\n"
            "graph = StateGraph()\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        agents = [c for c in comps if c.component_type == AIComponentType.AGENT]
        assert len(agents) >= 1
        assert any("StateGraph" in a.name for a in agents)

    def test_detects_embedding(self, tmp_path: Path):
        (tmp_path / "embed.py").write_text(
            "from langchain_openai import OpenAIEmbeddings\n"
            "embeddings = OpenAIEmbeddings()\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        embeddings = [c for c in comps if c.component_type == AIComponentType.EMBEDDING]
        assert len(embeddings) >= 1

    def test_detects_vector_store(self, tmp_path: Path):
        (tmp_path / "store.py").write_text(
            "from langchain_community.vectorstores import FAISS\n"
            "from langchain_openai import OpenAIEmbeddings\n"
            "vs = FAISS.from_texts(['hello'], OpenAIEmbeddings())\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        stores = [c for c in comps if c.component_type == AIComponentType.VECTOR_STORE]
        assert len(stores) >= 1

    def test_filters_other_concept(self, tmp_path: Path):
        (tmp_path / "client.py").write_text(
            "import openai\n"
            "client = openai.OpenAI()\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        # openai.OpenAI has concept=other in KB → should be filtered out
        assert all(c.kb_concept != "other" for c in comps)

    def test_method_call_noise_suppressed(self, tmp_path: Path):
        (tmp_path / "app.py").write_text(
            "import openai\n"
            "client = openai.OpenAI()\n"
            "resp = client.chat.completions.create(model='gpt-4o')\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        # The completions.create call should not be emitted as a component
        names = [c.name for c in comps]
        assert not any("create" in n for n in names)

    def test_dedup_same_line(self, tmp_path: Path):
        (tmp_path / "app.py").write_text(
            "from crewai import Agent\n"
            "a = Agent(role='r')\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        line_2 = [c for c in comps if c.line_number == 2]
        assert len(line_2) <= 1

    def test_skips_non_python_files(self, tmp_path: Path):
        (tmp_path / "config.yaml").write_text("agent: true\n")
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        assert comps == []

    def test_multiple_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text(
            "from crewai import Agent\n"
            "a = Agent(role='r')\n"
        )
        (tmp_path / "b.py").write_text(
            "from langgraph.graph import StateGraph\n"
            "g = StateGraph()\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        files = {c.file_path for c in comps}
        assert len(files) == 2


class TestHasSuggestiveSignal:
    """Unit tests for _has_suggestive_signal."""

    def test_ai_dir_models(self, tmp_path: Path):
        f = tmp_path / "models" / "llm.py"
        f.parent.mkdir()
        f.write_text("class MyLLM: pass")
        assert _has_suggestive_signal(f, "class MyLLM: pass")

    def test_ai_dir_agents(self, tmp_path: Path):
        f = tmp_path / "agents" / "main.py"
        f.parent.mkdir()
        f.write_text("pass")
        assert _has_suggestive_signal(f, "pass")

    def test_import_has_openai(self, tmp_path: Path):
        f = tmp_path / "utils" / "helper.py"
        f.parent.mkdir()
        source = "from internal.openai_wrapper import get_client\n"
        f.write_text(source)
        assert _has_suggestive_signal(f, source)

    def test_no_signal_plain_util(self, tmp_path: Path):
        f = tmp_path / "utils" / "strings.py"
        f.parent.mkdir()
        source = "import os\ndef strip(s): return s.strip()\n"
        f.write_text(source)
        assert not _has_suggestive_signal(f, source)

    def test_no_signal_test_file(self, tmp_path: Path):
        f = tmp_path / "tests" / "test_foo.py"
        f.parent.mkdir()
        source = "import pytest\ndef test_x(): pass\n"
        f.write_text(source)
        assert not _has_suggestive_signal(f, source)


class TestEmitSuggestiveCandidates:
    """Unit tests for _emit_suggestive_candidates."""

    def test_emits_for_indicative_class(self, tmp_path: Path):
        f = tmp_path / "models" / "chat.py"
        f.parent.mkdir()
        source = "router = RouterAgent(api_key='x')\n"
        f.write_text(source)
        candidates = _emit_suggestive_candidates(f, source)
        assert len(candidates) == 1
        assert candidates[0].component_type == AIComponentType.AGENT
        assert candidates[0].needs_agentic is True
        assert candidates[0].heuristic_confidence == 0.2
        assert "suggestive_signal" in candidates[0].metadata

    def test_skips_generic_class(self, tmp_path: Path):
        f = tmp_path / "models" / "data.py"
        f.parent.mkdir()
        source = "obj = Dict(key='val')\n"
        f.write_text(source)
        candidates = _emit_suggestive_candidates(f, source)
        assert len(candidates) == 0

    def test_skips_non_indicative_class(self, tmp_path: Path):
        f = tmp_path / "models" / "user.py"
        f.parent.mkdir()
        source = "user = UserProfile(name='x')\n"
        f.write_text(source)
        candidates = _emit_suggestive_candidates(f, source)
        assert len(candidates) == 0

    def test_multiple_candidates(self, tmp_path: Path):
        f = tmp_path / "agents" / "multi.py"
        f.parent.mkdir()
        source = (
            "router = RouterAgent(api_key='x')\n"
            "search = SearchTool(name='docs')\n"
        )
        f.write_text(source)
        candidates = _emit_suggestive_candidates(f, source)
        assert len(candidates) == 2


@pytest.mark.skipif(not _kb_available(), reason="No KB DuckDB installed")
class TestPartialMatchIntegration:
    """Integration test: wrapper code emits agentic candidates via Gate 3 MAYBE."""

    def test_wrapper_emits_agentic_candidate(self, tmp_path: Path):
        wrapper_dir = tmp_path / "models" / "llm"
        wrapper_dir.mkdir(parents=True)
        (wrapper_dir / "__init__.py").write_text("")
        (wrapper_dir / "openai_wrapper.py").write_text(
            "from models.llm.base import BaseLLM\n"
            "class ChatOpenAI:\n"
            "    pass\n"
            "llm = ChatOpenAI()\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        agentic = [c for c in comps if c.needs_agentic]
        assert len(agentic) >= 1
        hints = " ".join(c.agentic_hint for c in agentic)
        assert "remove" in hints.lower()


class TestKBEnrichmentScannerNoKB:
    """Tests for graceful behaviour when no KB is installed."""

    def test_supports_false_without_kb(self, tmp_path: Path):
        ctx = ScanContext(paths=[str(tmp_path)])
        with patch(
            "aibom.scanners.kb_enrichment_scanner._resolve_kb_path", return_value=None
        ):
            assert KBEnrichmentScanner().supports(ctx) is False

    def test_scan_returns_empty_without_kb(self, tmp_path: Path):
        (tmp_path / "app.py").write_text(
            "from langchain_openai import ChatOpenAI\n"
            "llm = ChatOpenAI(model='gpt-4')\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        with patch(
            "aibom.scanners.kb_enrichment_scanner._resolve_kb_path", return_value=None
        ):
            comps, rels = KBEnrichmentScanner().scan(ctx)
            assert comps == []
            assert rels == []


@pytest.mark.skipif(not _kb_available(), reason="No KB DuckDB installed")
class TestAgentSpecificHint:
    """Fix 3d: Import-inferred agents get a specialized hint.

    Requires the live DuckDB KB (``~/.aibom/catalogs/``) because
    ``KBEnrichmentScanner.scan`` short-circuits to ``[], []`` when no
    KB path is resolvable. Marked as integration to keep CI hermetic;
    matches the convention used by ``TestBuildKBPatterns`` and others
    above.
    """

    def test_agent_hint_mentions_three_conditions(self, tmp_path: Path):
        (tmp_path / "router.py").write_text(
            "from agents.security_agent import SecurityAgent\n"
            "agent = SecurityAgent()\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        agents = [c for c in comps if c.component_type == AIComponentType.AGENT]
        assert len(agents) >= 1
        hint = agents[0].agentic_hint
        assert "LLM-driven control flow" in hint
        assert "iterative loop" in hint
        assert "read_file_snippet" in hint

    def test_non_agent_hint_unchanged(self, tmp_path: Path):
        (tmp_path / "app.py").write_text(
            "from stores.vector_store import VectorStore\n"
            "vs = VectorStore()\n"
        )
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = KBEnrichmentScanner().scan(ctx)
        stores = [
            c for c in comps
            if c.component_type == AIComponentType.VECTOR_STORE
        ]
        if stores:
            hint = stores[0].agentic_hint
            assert "REMOVE unless surrounding code proves" in hint


# ---------------------------------------------------------------------------
# Strands provider-class coverage (kb_enrichment_scanner._KNOWN_AI_CLIENT_CLASSES)
# ---------------------------------------------------------------------------


_STRANDS_MODEL_CLASSES = (
    "BedrockModel",
    "AnthropicModel",
    "OpenAIModel",
    "OllamaModel",
    "LiteLLMModel",
    "GeminiModel",
    "MistralModel",
    "SageMakerModel",
    "LlamaAPIModel",
    "WriterModel",
    "LlamaCppModel",
    "CohereModel",
)


class TestStrandsProviderClassCoverage:
    """All Strands built-in model providers must be recognised.

    The agentic Strands framework (``strands.models.*``) exposes one class
    per provider. Each is the canonical construction point for a deployed
    model, so they must appear in ``_KNOWN_AI_CLIENT_CLASSES`` and have a
    matching bare-client hint so that ``_detect_model_kwargs`` can surface
    both resolved and unresolved instantiations.
    """

    @pytest.mark.parametrize("class_name", _STRANDS_MODEL_CLASSES)
    def test_strands_class_is_registered(self, class_name: str) -> None:
        assert class_name in _KNOWN_AI_CLIENT_CLASSES, (
            f"{class_name} missing from _KNOWN_AI_CLIENT_CLASSES"
        )

    @pytest.mark.parametrize("class_name", _STRANDS_MODEL_CLASSES)
    def test_strands_class_has_bare_client_hint(self, class_name: str) -> None:
        assert class_name in _BARE_CLIENT_HINTS, (
            f"{class_name} missing from _BARE_CLIENT_HINTS"
        )
        hint = _BARE_CLIENT_HINTS[class_name]
        assert "Remove" in hint, (
            f"{class_name} bare-client hint must mention removal: {hint!r}"
        )

    def test_bedrock_model_resolves_model_id_kwarg(self, tmp_path: Path) -> None:
        """``BedrockModel(model_id="...")`` should yield a MODEL asset with
        the resolved model name and full (1.0) heuristic confidence, not a
        low-confidence bare-client placeholder."""
        src = tmp_path / "agent.py"
        src.write_text(
            "from strands.models import BedrockModel\n"
            'model = BedrockModel(model_id="us.anthropic.claude-sonnet-4-20250514-v1:0")\n'
        )
        result = parse_source_code(str(src), src.read_text())
        comps = _detect_model_kwargs(result)
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert len(models) == 1
        assert (
            models[0].name == "us.anthropic.claude-sonnet-4-20250514-v1:0"
        ), f"unexpected model name: {models[0].name!r}"
        assert models[0].heuristic_confidence == 1.0
        assert models[0].metadata["constructor"] == "BedrockModel"
        assert models[0].metadata["extracted_from_kwarg"] == "model_id"
        assert (
            models[0].metadata.get("detection_method") != "bare_provider_client"
        )

    def test_bare_bedrock_model_emits_agentic_candidate(self, tmp_path: Path) -> None:
        """``BedrockModel()`` with no model_id is a bare client, so we emit a
        needs_agentic MODEL so the agentic layer can resolve or prune it."""
        src = tmp_path / "agent.py"
        src.write_text(
            "from strands.models import BedrockModel\n"
            "model = BedrockModel()\n"
        )
        result = parse_source_code(str(src), src.read_text())
        comps = _detect_model_kwargs(result)
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert len(models) == 1
        assert models[0].name == "BedrockModel"
        assert models[0].needs_agentic is True
        assert models[0].heuristic_confidence == 0.3
        assert models[0].metadata["detection_method"] == "bare_provider_client"
        assert "BEDROCK_MODEL_ID" in models[0].agentic_hint

    def test_anthropic_model_resolves_model_id_kwarg(self, tmp_path: Path) -> None:
        src = tmp_path / "agent.py"
        src.write_text(
            "from strands.models.anthropic import AnthropicModel\n"
            'model = AnthropicModel(model_id="claude-sonnet-4-20250514")\n'
        )
        result = parse_source_code(str(src), src.read_text())
        comps = _detect_model_kwargs(result)
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert len(models) == 1
        assert models[0].name == "claude-sonnet-4-20250514"
        assert models[0].metadata["provider"] == "anthropic"

    def test_openai_model_resolves_model_id_kwarg(self, tmp_path: Path) -> None:
        src = tmp_path / "agent.py"
        src.write_text(
            "from strands.models.openai import OpenAIModel\n"
            'model = OpenAIModel(model_id="gpt-4o")\n'
        )
        result = parse_source_code(str(src), src.read_text())
        comps = _detect_model_kwargs(result)
        models = [c for c in comps if c.component_type == AIComponentType.MODEL]
        assert len(models) == 1
        assert models[0].name == "gpt-4o"
        assert models[0].metadata["provider"] == "openai"


class TestStrandsAgentFrameworkPrefix:
    """Module-level ``agent = Agent(...)`` from Strands must become an AGENT.

    ``_AGENT_FRAMEWORK_PREFIXES`` gates the call-pattern tier of agent
    emission. Without ``"strands"`` in the set, a script-style Strands
    agent (no surrounding class) is silently dropped even though it uses
    the canonical ``Agent`` entrypoint. These tests lock in the prefix
    and guard against regressions — a ground-truth scan on a published
    open-source Strands sample repository failed on exactly this gap.
    """

    def test_strands_is_registered_framework_prefix(self) -> None:
        assert "strands" in _AGENT_FRAMEWORK_PREFIXES, (
            "strands must be a recognised agent framework prefix so that "
            "module-level ``Agent(...)`` calls are classified."
        )

    def test_module_level_agent_call_is_detected(self, tmp_path: Path) -> None:
        """``agent = Agent(model=...)`` must emit exactly one AGENT component.

        The component's ``name`` is the assignment target (``"agent"``) by
        design of the assignment branch of ``_process_file_with_cache``;
        the call pattern (``Agent`` / ``strands.Agent``) is recorded in
        ``metadata['call_pattern']`` and is the stable signal to assert
        on for the framework routing.
        """
        src = tmp_path / "agent.py"
        src.write_text(
            "from strands import Agent\n"
            "from strands.models import BedrockModel\n"
            "\n"
            'model = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0")\n'
            "agent = Agent(model=model)\n"
        )
        result = parse_source_code(str(src), src.read_text())
        comps = _process_file_with_cache(result, kb_by_id={})
        agents = [c for c in comps if c.component_type == AIComponentType.AGENT]
        assert len(agents) == 1, (
            f"expected exactly one Strands agent; got: "
            f"{[(c.name, c.component_type) for c in comps]}"
        )
        agent = agents[0]
        assert agent.detection_source == DetectionSource.CODE_ANALYSIS
        call_pattern = agent.metadata.get("call_pattern", "")
        assert call_pattern.endswith("Agent"), (
            f"call_pattern should end with 'Agent', got {call_pattern!r}"
        )

    def test_non_strands_non_framework_call_is_not_detected(
        self, tmp_path: Path
    ) -> None:
        """Adding ``strands`` must not also let a stray ``Agent(...)`` from
        an unrelated module register as an agent. This proves the prefix
        still gates the match (framework check > creation-pattern check).
        """
        src = tmp_path / "random.py"
        src.write_text(
            "from some_unrelated_module import Agent\n"
            "agent = Agent(name='not really an agent')\n"
        )
        result = parse_source_code(str(src), src.read_text())
        comps = _process_file_with_cache(result, kb_by_id={})
        agents = [c for c in comps if c.component_type == AIComponentType.AGENT]
        assert agents == [], (
            f"stray Agent() call from unknown framework should NOT be "
            f"promoted; got {[(c.name, c.framework) for c in agents]}"
        )
