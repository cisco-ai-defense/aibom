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

"""Tests for custom catalog loading, inline annotations, base class detection,
custom categories, excludes, metadata, and custom relationships."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict

import pytest

from aibom.custom_catalog import (
    CustomCatalogConfig,
    CustomComponentEntry,
    CustomRelationshipDef,
    BaseClassRule,
    discover_custom_catalog,
    load_custom_catalog,
    parse_inline_annotation,
)
from aibom.cst_parser import parse_source_code
from aibom.structures import ClassDefObservation, FunctionAnnotationObservation


# ─── Config loader tests ────────────────────────────────────────────────


class TestLoadCustomCatalogYAML:
    """Tests for loading .aibom.yaml files."""

    def test_load_full_yaml(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            components:
              - id: MyLLMWrapper
                concept: model
                label: My Custom LLM
                framework: internal
                metadata:
                  owner: ml-team
                  version: "2.1"
              - id: myproject.tools.SearchTool
                concept: tool
            base_classes:
              - class: BaseTool
                concept: tool
              - class: mylib.BaseAgent
                concept: agent
            excludes:
              - langchain.deprecated.OldAgent
              - some_noisy_function
            relationship_hints:
              tool_arguments:
                - custom_tools
                - plugins
              llm_arguments:
                - language_model
            custom_relationships:
              - label: ROUTES_TO
                source_categories: [router]
                target_categories: [agent]
                argument_hints: [routes, destinations]
              - label: GUARDS
                source_categories: [guardrail]
                target_categories: [model, agent]
                argument_hints: [guarded_by]
        """)
        config_file = tmp_path / ".aibom.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        config = load_custom_catalog(config_file)

        assert config.source_path == str(config_file)
        assert not config.is_empty

        # Components
        assert len(config.components) == 2
        c0 = config.components[0]
        assert c0.id == "MyLLMWrapper"
        assert c0.concept == "model"
        assert c0.label == "My Custom LLM"
        assert c0.framework == "internal"
        assert c0.metadata == {"owner": "ml-team", "version": "2.1"}

        c1 = config.components[1]
        assert c1.id == "myproject.tools.SearchTool"
        assert c1.concept == "tool"

        # Base class rules
        assert len(config.base_class_rules) == 2
        assert config.base_class_rules[0].base_class == "BaseTool"
        assert config.base_class_rules[0].concept == "tool"
        assert config.base_class_rules[1].base_class == "mylib.BaseAgent"

        # Excludes
        assert len(config.excludes) == 2
        assert "langchain.deprecated.OldAgent" in config.excludes

        # Relationship hints
        assert "tool_arguments" in config.relationship_hints
        assert "custom_tools" in config.relationship_hints["tool_arguments"]
        assert "language_model" in config.relationship_hints["llm_arguments"]

        # Custom relationships
        assert len(config.custom_relationships) == 2
        r0 = config.custom_relationships[0]
        assert r0.label == "ROUTES_TO"
        assert r0.source_categories == ["router"]
        assert r0.target_categories == ["agent"]
        assert r0.argument_hints == ["routes", "destinations"]

    def test_load_json(self, tmp_path: Path):
        data = {
            "components": [
                {"id": "MyCustomTool", "concept": "tool"},
            ],
            "excludes": ["unwanted_symbol"],
        }
        config_file = tmp_path / ".aibom.json"
        config_file.write_text(json.dumps(data), encoding="utf-8")

        config = load_custom_catalog(config_file)
        assert len(config.components) == 1
        assert config.components[0].concept == "tool"
        assert config.excludes == ["unwanted_symbol"]

    def test_empty_file_returns_empty_config(self, tmp_path: Path):
        config_file = tmp_path / ".aibom.yaml"
        config_file.write_text("", encoding="utf-8")
        config = load_custom_catalog(config_file)
        assert config.is_empty

    def test_missing_file_returns_empty_config(self, tmp_path: Path):
        config = load_custom_catalog(tmp_path / "nonexistent.yaml")
        assert config.is_empty

    def test_invalid_yaml_returns_empty_config(self, tmp_path: Path):
        config_file = tmp_path / ".aibom.yaml"
        config_file.write_text(":::invalid yaml [[[", encoding="utf-8")
        config = load_custom_catalog(config_file)
        assert config.is_empty

    def test_component_missing_id_skipped(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            components:
              - concept: model
        """)
        config_file = tmp_path / ".aibom.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")
        config = load_custom_catalog(config_file)
        assert len(config.components) == 0

    def test_component_missing_concept_skipped(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            components:
              - id: SomeThing
        """)
        config_file = tmp_path / ".aibom.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")
        config = load_custom_catalog(config_file)
        assert len(config.components) == 0

    def test_custom_category_accepted(self, tmp_path: Path):
        """Custom categories like 'guardrail' or 'router' should be accepted."""
        yaml_content = textwrap.dedent("""\
            components:
              - id: SafetyFilter
                concept: guardrail
              - id: MyRouter
                concept: router
        """)
        config_file = tmp_path / ".aibom.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")
        config = load_custom_catalog(config_file)
        assert len(config.components) == 2
        assert config.components[0].concept == "guardrail"
        assert config.components[1].concept == "router"

    def test_to_catalog_dict(self):
        entry = CustomComponentEntry(
            id="MyTool",
            concept="tool",
            label="My Tool",
            framework="custom",
            metadata={"version": "1.0"},
        )
        d = entry.to_catalog_dict()
        assert d["id"] == "MyTool"
        assert d["concept"] == "tool"
        assert d["label"] == "My Tool"
        assert d["framework"] == "custom"
        assert d["_metadata"] == {"version": "1.0"}


class TestDiscoverCustomCatalog:
    def test_discovers_yaml(self, tmp_path: Path):
        (tmp_path / ".aibom.yaml").write_text("components: []", encoding="utf-8")
        result = discover_custom_catalog(tmp_path)
        assert result is not None
        assert result.name == ".aibom.yaml"

    def test_discovers_yml(self, tmp_path: Path):
        (tmp_path / ".aibom.yml").write_text("components: []", encoding="utf-8")
        result = discover_custom_catalog(tmp_path)
        assert result is not None
        assert result.name == ".aibom.yml"

    def test_discovers_json(self, tmp_path: Path):
        (tmp_path / ".aibom.json").write_text("{}", encoding="utf-8")
        result = discover_custom_catalog(tmp_path)
        assert result is not None
        assert result.name == ".aibom.json"

    def test_yaml_preferred_over_json(self, tmp_path: Path):
        (tmp_path / ".aibom.yaml").write_text("components: []", encoding="utf-8")
        (tmp_path / ".aibom.json").write_text("{}", encoding="utf-8")
        result = discover_custom_catalog(tmp_path)
        assert result.name == ".aibom.yaml"

    def test_returns_none_when_absent(self, tmp_path: Path):
        assert discover_custom_catalog(tmp_path) is None


# ─── Inline annotation tests ────────────────────────────────────────────


class TestParseInlineAnnotation:
    def test_basic_annotation(self):
        result = parse_inline_annotation("# aibom: concept=model")
        assert result == {"concept": "model"}

    def test_annotation_with_extras(self):
        result = parse_inline_annotation("# aibom: concept=tool framework=internal label=MyTool")
        assert result["concept"] == "tool"
        assert result["framework"] == "internal"
        assert result["label"] == "MyTool"

    def test_no_aibom_comment(self):
        assert parse_inline_annotation("# some regular comment") is None

    def test_missing_concept(self):
        assert parse_inline_annotation("# aibom: framework=custom") is None

    def test_empty_string(self):
        assert parse_inline_annotation("") is None

    def test_none_input(self):
        assert parse_inline_annotation(None) is None


# ─── CST parser integration: ClassDef + FunctionDef annotations ──────────


class TestCSTParserClassDef:
    def test_basic_class_with_bases(self):
        code = textwrap.dedent("""\
            class BaseTool:
                pass

            class MyTool(BaseTool):
                pass
        """)
        result = parse_source_code("test.py", code)
        assert len(result.class_defs) == 2

        my_tool = [c for c in result.class_defs if c.class_name == "MyTool"][0]
        assert len(my_tool.base_classes) == 1
        assert "BaseTool" in my_tool.base_classes[0]

    def test_class_with_inline_annotation(self):
        code = textwrap.dedent("""\
            # aibom: concept=guardrail framework=internal
            class SafetyFilter:
                pass
        """)
        result = parse_source_code("test.py", code)
        annotated = [c for c in result.class_defs if c.class_name == "SafetyFilter"]
        assert len(annotated) == 1
        assert annotated[0].aibom_annotation is not None
        assert annotated[0].aibom_annotation["concept"] == "guardrail"
        assert annotated[0].aibom_annotation["framework"] == "internal"

    def test_class_without_annotation(self):
        code = textwrap.dedent("""\
            class PlainClass:
                pass
        """)
        result = parse_source_code("test.py", code)
        assert len(result.class_defs) == 1
        assert result.class_defs[0].aibom_annotation is None


class TestCSTParserFunctionAnnotation:
    def test_function_with_inline_annotation(self):
        code = textwrap.dedent("""\
            # aibom: concept=tool
            def search_web(query: str):
                pass
        """)
        result = parse_source_code("test.py", code)
        assert len(result.function_annotations) == 1
        ann = result.function_annotations[0]
        assert ann.function_name == "search_web"
        assert ann.aibom_annotation["concept"] == "tool"

    def test_function_without_annotation(self):
        code = textwrap.dedent("""\
            def plain_function():
                pass
        """)
        result = parse_source_code("test.py", code)
        assert len(result.function_annotations) == 0


# ─── CatalogDB integration ──────────────────────────────────────────────


class TestCatalogDBCustomEntries:
    """Tests for add_custom_entries and add_excludes on CatalogDB."""

    @pytest.fixture
    def catalog_db(self, tmp_path: Path):
        """Create a minimal DuckDB catalog for testing."""
        import duckdb
        db_file = tmp_path / "test_catalog.duckdb"
        con = duckdb.connect(str(db_file))
        con.execute("""
            CREATE TABLE component_catalog (
                id VARCHAR,
                label VARCHAR,
                concept VARCHAR,
                framework VARCHAR,
                sig_name VARCHAR,
                type VARCHAR,
                catalog_label VARCHAR
            )
        """)
        con.execute("""
            INSERT INTO component_catalog VALUES
            ('torch.nn.Module', 'Module', 'model', 'pytorch', NULL, NULL, NULL),
            ('sklearn.ensemble.RandomForestClassifier', 'RandomForestClassifier', 'model', 'sklearn', NULL, NULL, NULL)
        """)
        con.close()

        from aibom.catalog_db import CatalogDB
        return CatalogDB(db_file)

    def test_add_custom_entries(self, catalog_db):
        custom_entry = {
            "id": "mylib.CustomLLM",
            "label": "CustomLLM",
            "concept": "model",
            "framework": "custom",
            "sig_name": None,
            "type": None,
            "catalog_label": None,
        }
        catalog_db.add_custom_entries([custom_entry])
        results = catalog_db.find_components_by_suffixes(["CustomLLM"])
        assert any(r["id"] == "mylib.CustomLLM" for r in results)

    def test_excludes_filter_results(self, catalog_db):
        catalog_db.add_excludes(["RandomForestClassifier"])
        results = catalog_db.find_components_by_suffixes(
            ["RandomForestClassifier", "Module"]
        )
        ids = [r["id"] for r in results]
        assert "sklearn.ensemble.RandomForestClassifier" not in ids
        assert "torch.nn.Module" in ids

    def test_custom_entries_lower_precedence_than_db(self, catalog_db):
        """Custom entries should not override DuckDB entries."""
        custom_entry = {
            "id": "torch.nn.Module",
            "label": "Override",
            "concept": "tool",
            "framework": "custom",
            "sig_name": None,
            "type": None,
            "catalog_label": None,
        }
        catalog_db.add_custom_entries([custom_entry])
        results = catalog_db.find_components_by_suffixes(["Module"])
        module_entries = [r for r in results if r["id"] == "torch.nn.Module"]
        assert len(module_entries) == 1
        assert module_entries[0]["concept"] == "model"  # DB value wins


# ─── agent_signatures section routing ───────────────────────────────────


class TestCustomCatalogAgentSignatures:
    """The loader must pass the ``agent_signatures:`` section through to
    :func:`parse_user_signatures` and expose it on ``CustomCatalogConfig``.
    """

    def test_agent_signatures_are_loaded(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            agent_signatures:
              frameworks:
                - id: myorg.InternalAgent
                  framework: myorg
                  evidence_pattern: framework_inheritance
                  base_class_names: [InternalAgentBase]
                  import_substrings: [myorg.agents]
              protocols:
                - id: myorg.remote
                  protocol: remote_http
                  evidence_pattern: remote_proxy
                  role: client
                  qualified_name_substrings: [myorg.remote.Client]
              anti_patterns:
                - id: myorg.cron
                  label: cron_job
                  base_class_names: [CronJob]
              verification_policy:
                require_evidence_for_agent: false
                min_react_loop_call_count: 5
        """)
        config_file = tmp_path / ".aibom.yaml"
        config_file.write_text(yaml_content)

        config = load_custom_catalog(config_file)

        assert len(config.agent_signatures.frameworks) == 1
        assert config.agent_signatures.frameworks[0].id == "myorg.InternalAgent"
        assert len(config.agent_signatures.protocols) == 1
        assert config.agent_signatures.protocols[0].id == "myorg.remote"
        assert len(config.agent_signatures.anti_patterns) == 1
        assert config.agent_signatures.anti_patterns[0].id == "myorg.cron"
        assert config.agent_signatures.verification_policy.require_evidence_for_agent is False
        assert config.agent_signatures.verification_policy.min_react_loop_call_count == 5

    def test_agent_signatures_absent_yields_empty_catalog(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            components:
              - id: MyLLMWrapper
                concept: model
        """)
        config_file = tmp_path / ".aibom.yaml"
        config_file.write_text(yaml_content)

        config = load_custom_catalog(config_file)
        assert config.agent_signatures.is_empty is True

    def test_is_empty_accounts_for_agent_signatures(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            agent_signatures:
              frameworks:
                - id: myorg.X
                  framework: myorg
                  evidence_pattern: framework_agent
                  entrypoint_qualified_names: [myorg.X]
        """)
        config_file = tmp_path / ".aibom.yaml"
        config_file.write_text(yaml_content)

        config = load_custom_catalog(config_file)
        assert config.is_empty is False
