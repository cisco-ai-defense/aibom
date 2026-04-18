# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for precision improvements and cross-repo link architecture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aibom.agentic.middleware import (
    AIBOMScannerMiddleware,
    _is_class_name_not_model_id,
)
from aibom.models import (
    AIComponent,
    AIComponentType,
    ComponentRelationship,
    CrossRepoLink,
    CrossRepoLinkType,
    DecisionAnnotation,
    DetectionSource,
    RelationshipType,
    RepoOccurrence,
    ScanResult,
    SourceResult,
)
from aibom.finding_annotations import annotate_findings
from aibom.reporters.json_reporter import _aibom_payload


# ---------------------------------------------------------------------------
# 1. Class-name gate
# ---------------------------------------------------------------------------


class TestIsClassNameNotModelId:
    def test_rejects_camel_case_classes(self):
        assert _is_class_name_not_model_id("OpenAILLM") is True
        assert _is_class_name_not_model_id("ChatOpenAI") is True
        assert _is_class_name_not_model_id("AzureChatOpenAI") is True
        assert _is_class_name_not_model_id("OllamaClient") is True
        assert _is_class_name_not_model_id("HuggingFaceEmbeddings") is True

    def test_accepts_real_model_ids(self):
        assert _is_class_name_not_model_id("gpt-4o") is False
        assert _is_class_name_not_model_id("text-embedding-ada-002") is False
        assert _is_class_name_not_model_id("meta-llama/Llama-3-70B") is False
        assert _is_class_name_not_model_id("claude-sonnet-4-20250514") is False
        assert _is_class_name_not_model_id("all-MiniLM-L6-v2") is False

    def test_accepts_versioned_strings(self):
        assert _is_class_name_not_model_id("gpt-3.5-turbo") is False
        assert _is_class_name_not_model_id("Model3.0Beta") is False

    def test_accepts_slash_paths(self):
        assert _is_class_name_not_model_id("meta-llama/Llama-3-70B") is False

    def test_single_word_lowercase_not_class(self):
        assert _is_class_name_not_model_id("gpt4o") is False
        assert _is_class_name_not_model_id("mistral") is False


class TestMiddlewareClassNameGate:
    def test_new_component_model_class_name_rejected(self):
        mw = AIBOMScannerMiddleware()
        data = {
            "new_components": [
                {
                    "name": "OpenAILLM",
                    "component_type": "model",
                    "file_path": "app.py",
                    "line_number": 10,
                },
                {
                    "name": "gpt-4o",
                    "component_type": "model",
                    "file_path": "app.py",
                    "line_number": 20,
                },
            ],
            "enriched_components": [],
            "remove_components": [],
            "reclassify_components": [],
            "new_relationships": [],
            "risk_findings": [],
        }
        comps, _, _ = mw.extract_findings_from_dict(data)
        names = [c.name for c in comps]
        assert "OpenAILLM" not in names
        assert "gpt-4o" in names

    def test_enriched_model_class_name_removed(self):
        mw = AIBOMScannerMiddleware()
        existing = [
            AIComponent(
                name="OpenAILLM",
                component_type=AIComponentType.MODEL,
                file_path="app.py",
                line_number=10,
                detection_source=DetectionSource.CODE_ANALYSIS,
                needs_agentic=True,
            ),
            AIComponent(
                name="gpt-4o",
                component_type=AIComponentType.MODEL,
                model_name="gpt-4o",
                file_path="app.py",
                line_number=20,
                detection_source=DetectionSource.CODE_ANALYSIS,
                needs_agentic=True,
            ),
        ]
        data = {
            "enriched_components": [
                {"instance_id": existing[0].instance_id, "updates": {}},
                {"instance_id": existing[1].instance_id, "updates": {}},
            ],
            "remove_components": [],
            "reclassify_components": [],
        }
        result = mw.apply_enrichments_from_dict(existing, data)
        names = [c.name for c in result]
        assert "OpenAILLM" not in names
        assert "gpt-4o" in names


# ---------------------------------------------------------------------------
# 2. Endpoint URL exposure
# ---------------------------------------------------------------------------


class TestEndpointUrlExposure:
    def test_env_endpoint_url_in_metadata_and_model_name(self, tmp_path: Path):
        from aibom.scanners.config_scanner import ConfigScanner
        from aibom.models import ScanContext

        env_content = "CODEX_ENDPOINT=https://api.example.com/v1\n"
        (tmp_path / ".env").write_text(env_content)
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = ConfigScanner().scan(ctx)
        endpoints = [c for c in comps if "CODEX_ENDPOINT" in c.name]
        assert endpoints
        ep = endpoints[0]
        assert ep.model_name == "https://api.example.com/v1"
        assert ep.metadata.get("endpoint_url") == "https://api.example.com/v1"
        assert ep.metadata.get("env_var") == "CODEX_ENDPOINT"
        assert ep.metadata.get("redacted") is None


# ---------------------------------------------------------------------------
# 3. Embedding enforcement
# ---------------------------------------------------------------------------


class TestEmbeddingEnforcement:
    def test_unresolved_embedding_wrapper_removed(self):
        mw = AIBOMScannerMiddleware()
        data = {
            "new_components": [
                {
                    "name": "OpenAIEmbedder",
                    "component_type": "embedding",
                    "file_path": "app.py",
                    "line_number": 10,
                },
                {
                    "name": "text-embedding-ada-002",
                    "component_type": "embedding",
                    "file_path": "app.py",
                    "line_number": 20,
                    "model_name": "text-embedding-ada-002",
                },
            ],
            "enriched_components": [],
            "remove_components": [],
            "reclassify_components": [],
            "new_relationships": [],
            "risk_findings": [],
        }
        comps, _, _ = mw.extract_findings_from_dict(data)
        names = [c.name for c in comps]
        assert "OpenAIEmbedder" not in names
        assert "text-embedding-ada-002" in names

    def test_embedding_with_uses_relationship_kept(self):
        mw = AIBOMScannerMiddleware()
        data = {
            "new_components": [
                {
                    "name": "MyEmbedder",
                    "component_type": "embedding",
                    "file_path": "app.py",
                    "line_number": 10,
                },
            ],
            "enriched_components": [],
            "remove_components": [],
            "reclassify_components": [],
            "new_relationships": [
                {
                    "source_name": "MyEmbedder",
                    "target_name": "text-embedding-ada-002",
                    "relationship_type": "USES_EMBEDDING",
                },
            ],
            "risk_findings": [],
        }
        comps, rels, _ = mw.extract_findings_from_dict(data)
        names = [c.name for c in comps]
        assert "MyEmbedder" in names


# ---------------------------------------------------------------------------
# 4. File-loaded prompt annotation
# ---------------------------------------------------------------------------


class TestFileLoadedPromptAnnotation:
    def test_prompt_without_text_gets_limitation_annotation(self):
        comp = AIComponent(
            name="system_prompt",
            component_type=AIComponentType.PROMPT,
            file_path="/repo/prompts.py",
            line_number=5,
            detection_source=DetectionSource.CODE_ANALYSIS,
            text=None,
        )
        annotated, _, _ = annotate_findings([comp], [], [])
        ann = annotated[0].decision_annotation
        assert ann is not None
        assert "file_loaded_limitation" in ann.evidence_kinds
        assert "not extracted" in ann.justification.lower() or "loaded from" in ann.justification.lower()

    def test_prompt_with_text_gets_normal_annotation(self):
        comp = AIComponent(
            name="greeting_prompt",
            component_type=AIComponentType.PROMPT,
            file_path="/repo/prompts.py",
            line_number=10,
            detection_source=DetectionSource.CODE_ANALYSIS,
            text="You are a helpful assistant.",
        )
        annotated, _, _ = annotate_findings([comp], [], [])
        ann = annotated[0].decision_annotation
        assert ann is not None
        assert "file_loaded_limitation" not in ann.evidence_kinds


# ---------------------------------------------------------------------------
# 5. CrossRepoLink model
# ---------------------------------------------------------------------------


class TestCrossRepoLinkModel:
    def test_roundtrip_serialization(self):
        link = CrossRepoLink(
            link_type=CrossRepoLinkType.ENV_VAR_BINDING,
            identifier="OPENAI_API_KEY",
            resolved_value="sk-...",
            occurrences=[
                RepoOccurrence(repo_path="/repo-a", role="producer", file_path="/repo-a/.env"),
                RepoOccurrence(repo_path="/repo-b", role="consumer", component_name="env:OPENAI_API_KEY"),
            ],
            evidence="Defined in repo-a, consumed in repo-b",
        )
        data = link.model_dump(mode="json")
        restored = CrossRepoLink.model_validate(data)
        assert restored.link_type == CrossRepoLinkType.ENV_VAR_BINDING
        assert restored.identifier == "OPENAI_API_KEY"
        assert len(restored.occurrences) == 2
        assert restored.occurrences[0].role == "producer"

    def test_scan_result_includes_cross_repo_links(self):
        sr = ScanResult(
            cross_repo_links=[
                CrossRepoLink(
                    link_type=CrossRepoLinkType.SHARED_MODEL,
                    identifier="gpt-4o",
                    occurrences=[
                        RepoOccurrence(repo_path="/a", role="shared"),
                        RepoOccurrence(repo_path="/b", role="shared"),
                    ],
                )
            ]
        )
        assert len(sr.cross_repo_links) == 1
        assert sr.cross_repo_links[0].link_type == CrossRepoLinkType.SHARED_MODEL


# ---------------------------------------------------------------------------
# 6. Deterministic cross-repo link builder
# ---------------------------------------------------------------------------


class TestDeterministicLinkBuilder:
    def test_env_var_binding_detected(self, tmp_path: Path):
        from aibom.cross_repo_links import build_deterministic_cross_repo_links

        repo_a = tmp_path / "repo-a"
        repo_b = tmp_path / "repo-b"
        repo_a.mkdir()
        repo_b.mkdir()

        (repo_a / ".env").write_text("MODEL_NAME=gpt-4o\n")
        (repo_b / "app.py").write_text('import os\nmodel = os.getenv("MODEL_NAME")\n')

        comp_b = AIComponent(
            name="env:MODEL_NAME",
            component_type=AIComponentType.MODEL,
            file_path=str(repo_b / "app.py"),
            line_number=2,
            metadata={"env": "MODEL_NAME"},
        )
        per_repo = {
            str(repo_a): {"components": [], "_v2": True},
            str(repo_b): {"components": [comp_b], "_v2": True},
        }
        links = build_deterministic_cross_repo_links(
            per_repo, [str(repo_a), str(repo_b)]
        )
        env_links = [l for l in links if l.link_type == CrossRepoLinkType.ENV_VAR_BINDING]
        assert len(env_links) >= 1
        assert any(l.identifier == "MODEL_NAME" for l in env_links)

    def test_shared_model_detected(self):
        from aibom.cross_repo_links import build_deterministic_cross_repo_links

        comp_a = AIComponent(
            name="env_model_CHAT_MODEL",
            component_type=AIComponentType.MODEL,
            model_name="gpt-4o",
            file_path="/repo-a/config.py",
        )
        comp_b = AIComponent(
            name="env_model_LLM",
            component_type=AIComponentType.MODEL,
            model_name="gpt-4o",
            file_path="/repo-b/main.py",
        )
        per_repo = {
            "/repo-a": {"components": [comp_a], "_v2": True},
            "/repo-b": {"components": [comp_b], "_v2": True},
        }
        links = build_deterministic_cross_repo_links(per_repo, ["/repo-a", "/repo-b"])
        model_links = [l for l in links if l.link_type == CrossRepoLinkType.SHARED_MODEL]
        assert len(model_links) == 1
        assert model_links[0].identifier == "gpt-4o"

    def test_single_repo_returns_empty(self):
        from aibom.cross_repo_links import build_deterministic_cross_repo_links

        links = build_deterministic_cross_repo_links(
            {"/repo-a": {"components": []}}, ["/repo-a"]
        )
        assert links == []


# ---------------------------------------------------------------------------
# 7. JSON reporter cross_repo_links
# ---------------------------------------------------------------------------


class TestJsonReporterCrossRepoLinks:
    def test_cross_repo_links_in_output(self):
        sr = ScanResult(
            sources=[SourceResult(path="/repo-a")],
            cross_repo_links=[
                CrossRepoLink(
                    link_type=CrossRepoLinkType.SHARED_MODEL,
                    identifier="gpt-4o",
                    occurrences=[
                        RepoOccurrence(repo_path="/a", role="shared"),
                        RepoOccurrence(repo_path="/b", role="shared"),
                    ],
                ),
            ],
        )
        payload = _aibom_payload(sr)
        analysis = payload["aibom_analysis"]
        assert "cross_repo_links" in analysis
        assert len(analysis["cross_repo_links"]) == 1
        assert analysis["cross_repo_links"][0]["link_type"] == "SHARED_MODEL"

    def test_empty_cross_repo_links_omitted(self):
        sr = ScanResult(sources=[SourceResult(path="/repo-a")])
        payload = _aibom_payload(sr)
        analysis = payload["aibom_analysis"]
        assert "cross_repo_links" not in analysis
