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
from unittest.mock import MagicMock, patch
from collections import namedtuple

from aibom.categorizer import (
    categorize_symbols,
    _is_symbol_match,
    _reconstruct_code_snippet,
)
from aibom.structures import (
    CodeAnalysisResult,
    AssignmentObservation,
    CallObservation,
    DecoratorObservation,
)
from aibom.workflow_analyzer import WorkflowIndex, FunctionNode


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

    def test_is_symbol_match(self):
        self.assertTrue(_is_symbol_match("a.b.c", "a.b.c", []))
        self.assertFalse(
            _is_symbol_match("a.b.c", "x.y.a.b.c", ["import a.b"])
        )
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
            {"id": "langchain.PromptTemplate", "concept": "prompt", "label": "PromptTemplate"}
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
        tool_relationship = next(rel for rel in result.relationships if rel.label == "USES_TOOL")
        self.assertEqual(tool_relationship.target_name, "my_library.SearchTool")
        llm_relationship = next(rel for rel in result.relationships if rel.label == "USES_LLM")
        self.assertEqual(llm_relationship.target_name, "my_library.AsyncLLM")

    def test_categorize_symbols_with_no_concept(self):
        analysis_results = [
            CodeAnalysisResult(
                file_path="/test/other_file.py",
                assignments=[
                    AssignmentObservation(
                        target_qualified_name="unknown_obj",
                        call=CallObservation(
                            qualified_name="unknown.Library",
                            arguments={}
                        ),
                        line_number=25
                    )
                ],
                decorators=[],
                imports=[]
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
                            qualified_name="nonexistent.Class",
                            arguments={}
                        ),
                        line_number=30
                    )
                ],
                decorators=[],
                imports=[]
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
                            raw_code='OpenAI(model="gpt-4")'
                        ),
                        line_number=35
                    )
                ],
                decorators=[],
                imports=['from openai import OpenAI']
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

            result = categorize_symbols(analysis_results, self.mock_connector, mock_llm_config)

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

        no_args_obs = {
            "parser_name": "module.Class",
            "arguments": {}
        }
        result = _reconstruct_code_snippet(no_args_obs)
        self.assertEqual(result, "Class()")

        complex_obs = {
            "parser_name": "module.Class",
            "arguments": {
                "list_arg": [1, 2, 3],
                "dict_arg": {"key": "value"},
                "none_arg": None,
                "bool_arg": True
            }
        }
        result = _reconstruct_code_snippet(complex_obs)
        self.assertIn("Class(", result)
        self.assertIn("list_arg=[1, 2, 3]", result)
        self.assertIn("dict_arg={", result)
        self.assertIn("none_arg=None", result)
        self.assertIn("bool_arg=True", result)


if __name__ == "__main__":
    unittest.main()
