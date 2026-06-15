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

import unittest
from collections import namedtuple
from unittest.mock import MagicMock, patch

from aibom.categorizer import (
    _is_symbol_match,
    _reconstruct_code_snippet,
    categorize_symbols,
)
from aibom.structures import (
    AssignmentObservation,
    CallObservation,
    CodeAnalysisResult,
    DecoratorObservation,
)
from aibom.workflow_analyzer import FunctionNode, WorkflowIndex


class TestCategorizer(unittest.TestCase):
    def setUp(self):
        self.mock_connector = MagicMock()
        self.mock_llm_client = MagicMock()

    def test_categorize_symbols_basic(self):
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/file.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="test_model",
                        call=CallObservation(
                            qualified_name="some.lib.Model",
                            arguments={"arg1": "val1"},
                        ),
                        line_number=10,
                    )
                ],
                decorators=[],
                imports=["from some.lib import Model"],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "some.lib.Model", "concept": "model", "label": "Model"}
        ]

        workflow_index = WorkflowIndex()
        workflow_node = FunctionNode(
            function_id="/test/file.py::workflow",
            file_path="/test/file.py",
            qualname="test.workflow",
            start_line=1,
            end_line=50,
            decorators=[],
        )
        workflow_index.add_function(workflow_node)
        workflow_index.finalize()

        result = categorize_symbols(
            analysis_results, self.mock_connector, workflow_index=workflow_index
        )
        components = result.components

        self.mock_connector.find_components_by_suffixes.assert_called_with(
            unittest.mock.ANY,
        )

        self.assertIn("model", components)
        self.assertEqual(len(components["model"]), 1)
        self.assertEqual(components["model"][0]["name"], "some.lib.Model")
        self.assertIn("workflows", components["model"][0])

    def test_standalone_calls_detected(self):
        """Improvement 1: standalone calls (not assigned) are categorized."""
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/utils.py",
                assignments=[],
                calls=[
                    CallObservation(
                        qualified_name="langchain.chat_models.init_chat_model",
                        arguments={"model": "gpt-4"},
                        line_number=14,
                        raw_code='return init_chat_model("gpt-4")',
                    )
                ],
                decorators=[],
                imports=[],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {
                "id": "langchain.chat_models.init_chat_model",
                "concept": "model",
                "label": "init_chat_model",
            }
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)
        self.assertIn("model", result.components)
        self.assertEqual(len(result.components["model"]), 1)
        self.assertEqual(
            result.components["model"][0]["name"],
            "langchain.chat_models.init_chat_model",
        )

    def test_memory_relationship_detected(self):
        """Improvement 3: USES_MEMORY relationship derived from agent args."""
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/agent.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="saver",
                        call=CallObservation(
                            qualified_name="langgraph.checkpoint.memory.MemorySaver",
                            arguments={},
                            line_number=5,
                        ),
                        line_number=5,
                    ),
                    AssignmentObservation(
                        target_qualified_name="agent",
                        call=CallObservation(
                            qualified_name="langgraph.prebuilt.create_react_agent",
                            arguments={"checkpointer": "VARIABLE:saver"},
                            line_number=10,
                        ),
                        line_number=10,
                    ),
                ],
                decorators=[],
                imports=[],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {
                "id": "langgraph.checkpoint.memory.MemorySaver",
                "concept": "memory",
                "label": "MemorySaver",
            },
            {
                "id": "langgraph.prebuilt.create_react_agent",
                "concept": "agent",
                "label": "create_react_agent",
            },
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)
        self.assertTrue(any(r.label == "USES_MEMORY" for r in result.relationships))

    def test_retriever_relationship_detected(self):
        """Improvement 4: USES_RETRIEVER relationship."""
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/rag.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="my_retriever",
                        call=CallObservation(
                            qualified_name="langchain_core.retrievers.BaseRetriever",
                            arguments={},
                            line_number=5,
                        ),
                        line_number=5,
                    ),
                    AssignmentObservation(
                        target_qualified_name="agent",
                        call=CallObservation(
                            qualified_name="langgraph.prebuilt.create_react_agent",
                            arguments={"retriever": "VARIABLE:my_retriever"},
                            line_number=10,
                        ),
                        line_number=10,
                    ),
                ],
                decorators=[],
                imports=[],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {
                "id": "langchain_core.retrievers.BaseRetriever",
                "concept": "retriever",
                "label": "BaseRetriever",
            },
            {
                "id": "langgraph.prebuilt.create_react_agent",
                "concept": "agent",
                "label": "create_react_agent",
            },
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)
        self.assertTrue(any(r.label == "USES_RETRIEVER" for r in result.relationships))

    def test_embedding_relationship_detected(self):
        """Improvement 5: USES_EMBEDDING relationship."""
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/embed.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="embeddings",
                        call=CallObservation(
                            qualified_name="langchain_openai.OpenAIEmbeddings",
                            arguments={},
                            line_number=5,
                        ),
                        line_number=5,
                    ),
                    AssignmentObservation(
                        target_qualified_name="agent",
                        call=CallObservation(
                            qualified_name="langgraph.prebuilt.create_react_agent",
                            arguments={"embedding": "VARIABLE:embeddings"},
                            line_number=10,
                        ),
                        line_number=10,
                    ),
                ],
                decorators=[],
                imports=[],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {
                "id": "langchain_openai.OpenAIEmbeddings",
                "concept": "embedding",
                "label": "OpenAIEmbeddings",
            },
            {
                "id": "langgraph.prebuilt.create_react_agent",
                "concept": "agent",
                "label": "create_react_agent",
            },
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)
        self.assertTrue(any(r.label == "USES_EMBEDDING" for r in result.relationships))

    def test_agent_to_agent_relationship_detected(self):
        """USES_AGENT: a supervisor agent delegating to a sub-agent."""
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/crew.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="worker",
                        call=CallObservation(
                            qualified_name="crewai.Agent",
                            arguments={},
                            line_number=5,
                        ),
                        line_number=5,
                    ),
                    AssignmentObservation(
                        target_qualified_name="supervisor",
                        call=CallObservation(
                            qualified_name="crewai.Agent",
                            arguments={"sub_agents": "VARIABLE:worker"},
                            line_number=10,
                        ),
                        line_number=10,
                    ),
                ],
                decorators=[],
                imports=[],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "crewai.Agent", "concept": "agent", "label": "Agent"},
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)
        agent_edges = [r for r in result.relationships if r.label == "USES_AGENT"]
        self.assertTrue(agent_edges, "expected a USES_AGENT edge")
        # No self-edge: source and target must differ.
        for r in agent_edges:
            self.assertNotEqual(r.source_instance_id, r.target_instance_id)

    def test_agent_to_skill_relationship_detected(self):
        """USES_SKILL: an agent using a skill plugin."""
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/sk.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="my_skill",
                        call=CallObservation(
                            qualified_name="semantic_kernel.MathSkill",
                            arguments={},
                            line_number=5,
                        ),
                        line_number=5,
                    ),
                    AssignmentObservation(
                        target_qualified_name="agent",
                        call=CallObservation(
                            qualified_name="crewai.Agent",
                            arguments={"skills": "VARIABLE:my_skill"},
                            line_number=10,
                        ),
                        line_number=10,
                    ),
                ],
                decorators=[],
                imports=[],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {
                "id": "semantic_kernel.MathSkill",
                "concept": "skill",
                "label": "MathSkill",
            },
            {"id": "crewai.Agent", "concept": "agent", "label": "Agent"},
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)
        self.assertTrue(
            any(r.label == "USES_SKILL" for r in result.relationships),
            "expected a USES_SKILL edge",
        )

    def test_uses_agent_and_uses_skill_enum_members_exist(self):
        from aibom.models.enums import RelationshipType

        assert RelationshipType("USES_AGENT") is RelationshipType.USES_AGENT
        assert RelationshipType("USES_SKILL") is RelationshipType.USES_SKILL

    def test_is_symbol_match(self):
        self.assertTrue(_is_symbol_match("a.b.c", "a.b.c", []))
        self.assertFalse(_is_symbol_match("a.b.c", "x.y.a.b.c", ["import a.b"]))
        self.assertFalse(_is_symbol_match("a.b.c", "d.e.f", []))

    def test_reconstruct_code_snippet(self):
        obs_with_raw = {"raw_code": "MyClass(a=1)"}
        self.assertEqual(_reconstruct_code_snippet(obs_with_raw), "MyClass(a=1)")

        obs_without_raw = {
            "parser_name": "some.lib.MyClass",
            "arguments": {"param1": "hello", "param2": 123},
        }
        reconstructed = _reconstruct_code_snippet(obs_without_raw)
        self.assertIn("MyClass(", reconstructed)
        self.assertIn('param1="hello"', reconstructed)
        self.assertIn("param2=123", reconstructed)

    def test_categorize_symbols_with_decorators(self):
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/decorator_file.py",
                assignments=[],
                decorators=[
                    DecoratorObservation(
                        decorator_qualified_name="fastapi.get",
                        decorated_function_name="my_endpoint",
                        line_number=15,
                    )
                ],
                imports=["from fastapi import FastAPI"],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "fastapi.get", "concept": "tool", "label": "HTTPMethod"}
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)
        components = result.components

        self.assertIn("tool", components)
        self.assertEqual(len(components["tool"]), 1)
        self.assertEqual(components["tool"][0]["name"], "my_endpoint")
        self.assertEqual(components["tool"][0]["decorated_by"], "fastapi.get")

    def test_categorize_symbols_with_prompt_category(self):
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/prompt_file.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="my_prompt",
                        call=CallObservation(
                            qualified_name="langchain.PromptTemplate",
                            arguments={"template": "Hello {name}!"},
                        ),
                        line_number=20,
                    )
                ],
                decorators=[],
                imports=["from langchain import PromptTemplate"],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {
                "id": "langchain.PromptTemplate",
                "concept": "prompt",
                "label": "PromptTemplate",
            }
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)
        components = result.components

        self.assertIn("prompt", components)
        self.assertEqual(len(components["prompt"]), 1)
        self.assertEqual(components["prompt"][0]["name"], "langchain.PromptTemplate")
        self.assertEqual(components["prompt"][0]["text"], "Hello {name}!")

    def test_agent_relationships_detected(self):
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/agent_file.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="search_tool",
                        call=CallObservation(
                            qualified_name="my_library.SearchTool",
                            arguments={},
                            line_number=10,
                        ),
                        line_number=10,
                    ),
                    AssignmentObservation(
                        target_qualified_name="llm_client",
                        call=CallObservation(
                            qualified_name="my_library.AsyncLLM",
                            arguments={},
                            line_number=15,
                        ),
                        line_number=15,
                    ),
                    AssignmentObservation(
                        target_qualified_name="agent_instance",
                        call=CallObservation(
                            qualified_name="my_library.Agent",
                            arguments={
                                "tools": ["VARIABLE:search_tool"],
                                "llm": "VARIABLE:llm_client",
                            },
                            line_number=20,
                        ),
                        line_number=20,
                    ),
                ],
                decorators=[],
                imports=[],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "my_library.SearchTool", "concept": "tool", "label": "Tool"},
            {"id": "my_library.AsyncLLM", "concept": "model", "label": "LLM"},
            {"id": "my_library.Agent", "concept": "agent", "label": "Agent"},
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)

        self.assertEqual(len(result.relationships), 2)
        labels = {rel.label for rel in result.relationships}
        self.assertSetEqual(labels, {"USES_TOOL", "USES_LLM"})
        tool_relationship = next(
            rel for rel in result.relationships if rel.label == "USES_TOOL"
        )
        self.assertEqual(tool_relationship.target_name, "my_library.SearchTool")
        llm_relationship = next(
            rel for rel in result.relationships if rel.label == "USES_LLM"
        )
        self.assertEqual(llm_relationship.target_name, "my_library.AsyncLLM")

    def test_categorize_symbols_with_no_concept(self):
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/other_file.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="unknown_obj",
                        call=CallObservation(
                            qualified_name="unknown.Library", arguments={}
                        ),
                        line_number=25,
                    )
                ],
                decorators=[],
                imports=[],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "unknown.Library", "concept": None, "label": "Library"}
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)
        components = result.components

        self.assertIn("other", components)
        self.assertEqual(len(components["other"]), 1)
        self.assertEqual(components["other"][0]["name"], "unknown.Library")

    def test_categorize_symbols_empty_input(self):
        result = categorize_symbols([], self.mock_connector)
        self.assertEqual(result.components, {})
        self.assertEqual(result.relationships, [])

    def test_categorize_symbols_no_matches(self):
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/nomatch_file.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="no_match",
                        call=CallObservation(
                            qualified_name="nonexistent.Class", arguments={}
                        ),
                        line_number=30,
                    )
                ],
                decorators=[],
                imports=[],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = []

        result = categorize_symbols(analysis_results, self.mock_connector)
        self.assertEqual(result.components, {})
        self.assertEqual(result.relationships, [])

    def test_categorize_symbols_with_llm_config(self):
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/model_file.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="my_model",
                        call=CallObservation(
                            qualified_name="openai.OpenAI",
                            arguments={"model": "gpt-4"},
                            raw_code='OpenAI(model="gpt-4")',
                        ),
                        line_number=35,
                    )
                ],
                decorators=[],
                imports=["from openai import OpenAI"],
            )
        ]

        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "openai.OpenAI", "concept": "model", "label": "LLMModel"}
        ]

        mock_llm_config = {"api_key": "test", "model": "gpt-3.5-turbo"}

        with patch("aibom.categorizer.LLMClient") as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_instance.extract_model_name.return_value = "gpt-4"
            mock_llm_class.return_value = mock_llm_instance

            result = categorize_symbols(
                analysis_results, self.mock_connector, mock_llm_config
            )

        components = result.components
        self.assertIn("model", components)
        self.assertEqual(len(components["model"]), 1)
        self.assertEqual(components["model"][0]["name"], "openai.OpenAI")
        self.assertEqual(components["model"][0]["model_name"], "gpt-4")

    def test_is_symbol_match_edge_cases(self):
        self.assertTrue(_is_symbol_match("", "", []))
        self.assertFalse(_is_symbol_match("a.b.c", "", []))
        self.assertFalse(_is_symbol_match("", "a.b.c", []))
        self.assertFalse(_is_symbol_match("A.B.C", "a.b.c", []))
        self.assertTrue(_is_symbol_match("A.B.C", "A.B.C", []))
        self.assertTrue(_is_symbol_match("a.b_c.d-e", "a.b_c.d-e", []))

    def test_reconstruct_code_snippet_edge_cases(self):
        empty_obs = {}
        result = _reconstruct_code_snippet(empty_obs)
        self.assertIn("()", result)

        no_args_obs = {"parser_name": "module.Class", "arguments": {}}
        result = _reconstruct_code_snippet(no_args_obs)
        self.assertEqual(result, "Class()")

        complex_obs = {
            "parser_name": "module.Class",
            "arguments": {
                "list_arg": [1, 2, 3],
                "dict_arg": {"key": "value"},
                "none_arg": None,
                "bool_arg": True,
            },
        }
        result = _reconstruct_code_snippet(complex_obs)
        self.assertIn("Class(", result)
        self.assertIn("list_arg=[1, 2, 3]", result)
        self.assertIn("dict_arg={", result)
        self.assertIn("none_arg=None", result)
        self.assertIn("bool_arg=True", result)


class TestPrecedenceAndExcludes(unittest.TestCase):
    """Tests for inline-annotation precedence, excludes, and deduplication (Phase 7)."""

    def setUp(self):
        self.mock_connector = MagicMock()

    def test_inline_annotation_overrides_catalog_concept(self):
        """DuckDB says concept='model', inline says concept='tool' -> appears once as 'tool'."""
        from aibom.custom_catalog import CustomCatalogConfig
        from aibom.structures import ClassDefObservation

        # The DuckDB catalog would match this symbol as 'model'
        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "mylib.MyClass", "concept": "model", "label": "MyClass"}
        ]

        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/inline.py",
                # An assignment that would match the catalog entry
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="obj",
                        call=CallObservation(
                            qualified_name="mylib.MyClass",
                            arguments={},
                            line_number=10,
                        ),
                        line_number=10,
                    )
                ],
                # The same class at line 10 also has an inline annotation saying concept=tool
                class_defs=[
                    ClassDefObservation(
                        class_name="MyClass",
                        line_number=10,
                        aibom_annotation={"concept": "tool", "framework": "custom"},
                    )
                ],
                decorators=[],
                imports=[],
            )
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)

        # The inline annotation should win: appears in 'tool', NOT in 'model'
        tool_names = [c["name"] for c in result.components.get("tool", [])]
        model_at_line_10 = [
            c for c in result.components.get("model", []) if c.get("line_number") == 10
        ]
        self.assertIn("MyClass", tool_names)
        self.assertEqual(len(model_at_line_10), 0)

    def test_excludes_filter_inline_annotations(self):
        """Symbol excluded in config, detected via # aibom: -> does NOT appear."""
        from aibom.custom_catalog import CustomCatalogConfig
        from aibom.structures import ClassDefObservation

        self.mock_connector.find_components_by_suffixes.return_value = []

        config = CustomCatalogConfig(excludes=["MyAgent"])

        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/excluded.py",
                class_defs=[
                    ClassDefObservation(
                        class_name="MyAgent",
                        line_number=5,
                        aibom_annotation={"concept": "agent", "framework": "custom"},
                    )
                ],
                imports=[],
            )
        ]

        result = categorize_symbols(
            analysis_results, self.mock_connector, custom_config=config
        )

        # MyAgent should be excluded
        agent_names = [c["name"] for c in result.components.get("agent", [])]
        self.assertNotIn("MyAgent", agent_names)

    def test_excludes_filter_base_class_detections(self):
        """Symbol excluded in config, detected via base-class rule -> does NOT appear."""
        from aibom.custom_catalog import BaseClassRule, CustomCatalogConfig
        from aibom.structures import ClassDefObservation

        self.mock_connector.find_components_by_suffixes.return_value = []

        config = CustomCatalogConfig(
            base_class_rules=[BaseClassRule(base_class="BaseTool", concept="tool")],
            excludes=["InternalTool"],
        )

        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/base.py",
                class_defs=[
                    ClassDefObservation(
                        class_name="InternalTool",
                        base_classes=["BaseTool"],
                        line_number=20,
                    )
                ],
                imports=[],
            )
        ]

        result = categorize_symbols(
            analysis_results, self.mock_connector, custom_config=config
        )

        tool_names = [c["name"] for c in result.components.get("tool", [])]
        self.assertNotIn("InternalTool", tool_names)

    def test_no_duplicates_across_detection_paths(self):
        """Same symbol at same line detected by catalog AND inline -> appears once."""
        from aibom.structures import ClassDefObservation, FunctionAnnotationObservation

        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "mylib.MyTool", "concept": "tool", "label": "MyTool"}
        ]

        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/dedup.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="t",
                        call=CallObservation(
                            qualified_name="mylib.MyTool",
                            arguments={},
                            line_number=5,
                        ),
                        line_number=5,
                    )
                ],
                function_annotations=[
                    FunctionAnnotationObservation(
                        function_name="MyTool",
                        line_number=5,
                        aibom_annotation={"concept": "tool"},
                    )
                ],
                decorators=[],
                imports=[],
            )
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)

        # Should appear exactly once across all categories
        all_at_line_5 = []
        for category, components in result.components.items():
            for c in components:
                if c.get("line_number") == 5 and c.get("file_path") == "/test/dedup.py":
                    all_at_line_5.append(c)
        self.assertEqual(len(all_at_line_5), 1)

    def test_base_class_skipped_when_catalog_detected(self):
        """Symbol detected by DuckDB at same location as base-class rule -> appears once with DuckDB data."""
        from aibom.custom_catalog import BaseClassRule, CustomCatalogConfig
        from aibom.structures import ClassDefObservation

        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "mylib.Agent", "concept": "agent", "label": "Agent"}
        ]

        config = CustomCatalogConfig(
            base_class_rules=[BaseClassRule(base_class="BaseAgent", concept="agent")],
        )

        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/overlap.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="a",
                        call=CallObservation(
                            qualified_name="mylib.Agent",
                            arguments={},
                            line_number=10,
                        ),
                        line_number=10,
                    )
                ],
                class_defs=[
                    ClassDefObservation(
                        class_name="Agent",
                        base_classes=["BaseAgent"],
                        line_number=10,
                    )
                ],
                decorators=[],
                imports=[],
            )
        ]

        result = categorize_symbols(
            analysis_results, self.mock_connector, custom_config=config
        )

        agents = result.components.get("agent", [])
        agents_at_10 = [c for c in agents if c.get("line_number") == 10]
        self.assertEqual(len(agents_at_10), 1)
        # Should be the catalog-detected version (name = full qualified id)
        self.assertEqual(agents_at_10[0]["name"], "mylib.Agent")

    def test_multiple_suffix_match_deterministic(self):
        """When multiple custom entries match via suffix, results are deterministic."""
        self.mock_connector.find_components_by_suffixes.return_value = [
            {"id": "pkg1.Agent", "concept": "agent", "label": "Agent"},
            {"id": "pkg2.Agent", "concept": "agent", "label": "Agent"},
        ]

        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/multi.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="a1",
                        call=CallObservation(
                            qualified_name="pkg1.Agent",
                            arguments={},
                            line_number=5,
                        ),
                        line_number=5,
                    ),
                    AssignmentObservation(
                        target_qualified_name="a2",
                        call=CallObservation(
                            qualified_name="pkg2.Agent",
                            arguments={},
                            line_number=10,
                        ),
                        line_number=10,
                    ),
                ],
                decorators=[],
                imports=[],
            )
        ]

        result = categorize_symbols(analysis_results, self.mock_connector)

        # Both should appear as distinct agents
        agents = result.components.get("agent", [])
        self.assertEqual(len(agents), 2)

        # Run again to verify determinism
        result2 = categorize_symbols(analysis_results, self.mock_connector)
        agents2 = result2.components.get("agent", [])
        self.assertEqual(len(agents2), 2)
        self.assertEqual(
            [a["name"] for a in agents],
            [a["name"] for a in agents2],
        )


if __name__ == "__main__":
    unittest.main()
