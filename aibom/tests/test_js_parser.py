# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the JS/TS parser."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from aibom.js_parser import parse_js_source_code, is_js_file, _deserialise


class TestIsJsFile:
    def test_js(self, tmp_path):
        assert is_js_file(Path("app.js"))

    def test_ts(self):
        assert is_js_file(Path("app.ts"))

    def test_jsx(self):
        assert is_js_file(Path("App.jsx"))

    def test_tsx(self):
        assert is_js_file(Path("App.tsx"))

    def test_py_is_not_js(self):
        assert not is_js_file(Path("app.py"))


class TestDeserialise:
    def test_basic_deserialisation(self):
        data = {
            "file_path": "/test/file.ts",
            "assignments": [
                {
                    "target_qualified_name": "llm",
                    "call": {
                        "qualified_name": "@langchain/openai.ChatOpenAI",
                        "arguments": {"model": "gpt-4o"},
                        "line_number": 5,
                        "raw_code": 'new ChatOpenAI({ model: "gpt-4o" })',
                    },
                    "line_number": 5,
                }
            ],
            "calls": [
                {
                    "qualified_name": "ai.generateText",
                    "arguments": {"prompt": "hello"},
                    "line_number": 10,
                    "raw_code": "generateText({ prompt: 'hello' })",
                }
            ],
            "decorators": [],
            "class_defs": [
                {
                    "class_name": "MyAgent",
                    "qualified_name": "MyAgent",
                    "base_classes": ["BaseAgent"],
                    "line_number": 20,
                    "aibom_annotation": None,
                }
            ],
            "imports": ["from @langchain/openai import ChatOpenAI"],
            "type_annotations": [],
            "context_managers": [],
            "function_annotations": [],
        }
        result = _deserialise(data, "/fallback.ts")
        assert result.file_path == "/test/file.ts"
        assert len(result.assignments) == 1
        assert result.assignments[0].call.qualified_name == "@langchain/openai.ChatOpenAI"
        assert len(result.calls) == 1
        assert result.calls[0].qualified_name == "ai.generateText"
        assert len(result.class_defs) == 1
        assert result.class_defs[0].class_name == "MyAgent"
        assert "ChatOpenAI" in result.imports[0]


class TestParseJsSourceCode:
    def test_returns_empty_when_node_unavailable(self, monkeypatch):
        monkeypatch.setattr("aibom.js_parser._node_available", lambda: False)
        result = parse_js_source_code("/test.ts", "const x = 1;")
        assert result.file_path == "/test.ts"
        assert result.assignments == []
        assert result.calls == []

    def test_returns_empty_for_missing_script(self, monkeypatch):
        monkeypatch.setattr("aibom.js_parser._node_available", lambda: True)
        monkeypatch.setattr("aibom.js_parser._PARSER_SCRIPT", Path("/nonexistent/parse.js"))
        result = parse_js_source_code("/test.ts", "const x = 1;")
        assert result.assignments == []

    def test_handles_subprocess_failure(self, monkeypatch):
        monkeypatch.setattr("aibom.js_parser._node_available", lambda: True)
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        monkeypatch.setattr("aibom.js_parser.subprocess.run", mock_run)
        result = parse_js_source_code("/test.ts", "const x = 1;")
        assert result.assignments == []

    def test_handles_invalid_json(self, monkeypatch):
        monkeypatch.setattr("aibom.js_parser._node_available", lambda: True)
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")
        monkeypatch.setattr("aibom.js_parser.subprocess.run", mock_run)
        result = parse_js_source_code("/test.ts", "const x = 1;")
        assert result.assignments == []

    def test_parses_valid_output(self, monkeypatch):
        monkeypatch.setattr("aibom.js_parser._node_available", lambda: True)
        output = json.dumps({
            "file_path": "/test.ts",
            "assignments": [{
                "target_qualified_name": "llm",
                "call": {
                    "qualified_name": "@langchain/openai.ChatOpenAI",
                    "arguments": {"model": "gpt-4o"},
                    "line_number": 3,
                    "raw_code": "new ChatOpenAI({})",
                },
                "line_number": 3,
            }],
            "calls": [],
            "decorators": [],
            "class_defs": [],
            "imports": ["from @langchain/openai import ChatOpenAI"],
            "type_annotations": [],
            "context_managers": [],
            "function_annotations": [],
        })
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout=output, stderr="")
        monkeypatch.setattr("aibom.js_parser.subprocess.run", mock_run)
        result = parse_js_source_code("/test.ts", "const llm = new ChatOpenAI({});")
        assert len(result.assignments) == 1
        assert result.assignments[0].call.qualified_name == "@langchain/openai.ChatOpenAI"


class TestDeserialiseEnriched:
    """Tests for richer argument structures emitted by the updated parse.js."""

    def test_nested_call_in_arguments(self):
        """Calls nested inside other calls' arguments are captured."""
        data = {
            "file_path": "/test.ts",
            "assignments": [{
                "target_qualified_name": "result",
                "call": {
                    "qualified_name": "ai.generateText",
                    "arguments": {"model": "VARIABLE:openai", "prompt": "hi"},
                    "line_number": 5,
                    "raw_code": "generateText({ model: openai('gpt-4o'), prompt: 'hi' })",
                },
                "line_number": 5,
            }],
            "calls": [{
                "qualified_name": "@ai-sdk/openai.openai",
                "arguments": {},
                "line_number": 5,
                "raw_code": "openai('gpt-4o')",
            }],
            "decorators": [],
            "class_defs": [],
            "imports": ["from ai import generateText", "from @ai-sdk/openai import openai"],
            "type_annotations": [],
            "context_managers": [],
            "function_annotations": [],
        }
        result = _deserialise(data, "/fallback.ts")
        assert len(result.assignments) == 1
        assert result.assignments[0].call.qualified_name == "ai.generateText"
        assert result.assignments[0].call.arguments["model"] == "VARIABLE:openai"
        assert len(result.calls) == 1
        assert result.calls[0].qualified_name == "@ai-sdk/openai.openai"

    def test_await_expression_assignment(self):
        """Assignments wrapping await expressions are correctly unwrapped."""
        data = {
            "file_path": "/test.ts",
            "assignments": [{
                "target_qualified_name": "response",
                "call": {
                    "qualified_name": "ai.generateText",
                    "arguments": {"prompt": "hello"},
                    "line_number": 3,
                    "raw_code": "await generateText({ prompt: 'hello' })",
                },
                "line_number": 3,
            }],
            "calls": [],
            "decorators": [],
            "class_defs": [],
            "imports": [],
            "type_annotations": [],
            "context_managers": [],
            "function_annotations": [],
        }
        result = _deserialise(data, "/fallback.ts")
        assert len(result.assignments) == 1
        assert result.assignments[0].target_qualified_name == "response"
        assert result.assignments[0].call.qualified_name == "ai.generateText"

    def test_object_expression_arguments(self):
        """Object expressions in arguments are preserved as dicts."""
        data = {
            "file_path": "/test.ts",
            "assignments": [{
                "target_qualified_name": "stream",
                "call": {
                    "qualified_name": "ai.streamText",
                    "arguments": {
                        "model": "VARIABLE:openai",
                        "tools": {"weather": "VARIABLE:weatherTool"},
                    },
                    "line_number": 8,
                    "raw_code": "streamText({ model: openai('gpt-4'), tools: { weather: weatherTool } })",
                },
                "line_number": 8,
            }],
            "calls": [],
            "decorators": [],
            "class_defs": [],
            "imports": [],
            "type_annotations": [],
            "context_managers": [],
            "function_annotations": [],
        }
        result = _deserialise(data, "/fallback.ts")
        args = result.assignments[0].call.arguments
        assert args["model"] == "VARIABLE:openai"
        assert isinstance(args["tools"], dict)
        assert args["tools"]["weather"] == "VARIABLE:weatherTool"
