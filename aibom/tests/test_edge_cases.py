# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Edge case tests for parsers: empty files, syntax errors, encodings, etc."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aibom.cst_parser import parse_source_code
from aibom.js_parser import parse_js_source_code, _deserialise


class TestEmptyFiles:
    def test_empty_python_file(self):
        result = parse_source_code("/test/empty.py", "")
        assert result.file_path == "/test/empty.py"
        assert result.assignments == []
        assert result.calls == []
        assert result.imports == []

    def test_empty_js_file(self, monkeypatch):
        monkeypatch.setattr("aibom.js_parser._node_available", lambda: True)
        output = json.dumps({
            "file_path": "/test/empty.ts",
            "assignments": [],
            "calls": [],
            "decorators": [],
            "class_defs": [],
            "imports": [],
            "type_annotations": [],
            "context_managers": [],
            "function_annotations": [],
        })
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout=output, stderr="")
        monkeypatch.setattr("aibom.js_parser.subprocess.run", mock_run)
        result = parse_js_source_code("/test/empty.ts", "")
        assert result.assignments == []
        assert result.calls == []


class TestSyntaxErrors:
    def test_syntax_error_python(self):
        bad_code = "def broken(\n  x = ChatOpenAI(model='gpt-4o'"
        result = parse_source_code("/test/bad.py", bad_code)
        assert result.file_path == "/test/bad.py"

    def test_syntax_error_js(self, monkeypatch):
        monkeypatch.setattr("aibom.js_parser._node_available", lambda: True)
        output = json.dumps({
            "file_path": "/test/bad.ts",
            "assignments": [],
            "calls": [],
            "decorators": [],
            "class_defs": [],
            "imports": [],
            "type_annotations": [],
            "context_managers": [],
            "function_annotations": [],
        })
        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout=output, stderr="")
        monkeypatch.setattr("aibom.js_parser.subprocess.run", mock_run)
        result = parse_js_source_code("/test/bad.ts", "const x = new Foo({")
        assert result.assignments == []


class TestUnicodeAndEncoding:
    def test_unicode_identifiers(self):
        code = """
from langchain_openai import ChatOpenAI
变量 = ChatOpenAI(model="gpt-4o")
"""
        result = parse_source_code("/test/unicode.py", code)
        assert result.file_path == "/test/unicode.py"
        assert len(result.assignments) > 0

    def test_utf8_bom(self):
        bom = "\ufeff"
        code = bom + "from langchain_openai import ChatOpenAI\nllm = ChatOpenAI()\n"
        result = parse_source_code("/test/bom.py", code)
        assert result.file_path == "/test/bom.py"


class TestLargeAndComplex:
    def test_deeply_nested_calls(self):
        code = """
from langchain_openai import ChatOpenAI
result = ChatOpenAI(model=str(int(float("4"))))
"""
        result = parse_source_code("/test/nested.py", code)
        assert result.file_path == "/test/nested.py"

    def test_multiline_arguments(self):
        code = """
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
)
"""
        result = parse_source_code("/test/multiline.py", code)
        assert len(result.assignments) == 1

    def test_star_import(self):
        code = """
from langchain_openai import *
llm = ChatOpenAI(model="gpt-4o")
"""
        result = parse_source_code("/test/star.py", code)
        assert result.file_path == "/test/star.py"
