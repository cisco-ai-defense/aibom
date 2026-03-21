# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aibom.models.enums import AIComponentType
from aibom.scanners.multi_language_scanner import MultiLanguageScanner

from .conftest import run_scanner


def test_typescript_openai_import(tmp_path: Path) -> None:
    comps, rels = run_scanner(
        MultiLanguageScanner, tmp_path, {"src/main.ts": 'import OpenAI from "openai"\n'}
    )
    assert rels == []
    assert any(c.name == "openai" for c in comps if c.component_type == AIComponentType.DEPENDENCY)


def test_javascript_model_literal(tmp_path: Path) -> None:
    comps, _ = run_scanner(
        MultiLanguageScanner, tmp_path, {"app.js": 'const cfg = { model: "gpt-4o" }\n'}
    )
    models = [c for c in comps if c.component_type == AIComponentType.MODEL]
    assert any(c.model_name == "gpt-4o" for c in models)


def test_go_openai_import(tmp_path: Path) -> None:
    comps, _ = run_scanner(
        MultiLanguageScanner,
        tmp_path,
        {"main.go": 'package main\nimport "github.com/sashabaranov/go-openai"\nfunc main() {}\n'},
    )
    assert any(
        c.name == "github.com/sashabaranov/go-openai"
        for c in comps
        if c.component_type == AIComponentType.DEPENDENCY
    )


def test_rust_async_openai(tmp_path: Path) -> None:
    comps, _ = run_scanner(
        MultiLanguageScanner, tmp_path, {"src/lib.rs": "use async_openai::Client;\n"}
    )
    assert any(
        c.name == "async_openai" for c in comps if c.component_type == AIComponentType.DEPENDENCY
    )


def test_csharp_semantic_kernel(tmp_path: Path) -> None:
    comps, _ = run_scanner(
        MultiLanguageScanner, tmp_path, {"Program.cs": "using Microsoft.SemanticKernel;\n"}
    )
    assert any(
        c.name == "Microsoft.SemanticKernel"
        for c in comps
        if c.component_type == AIComponentType.DEPENDENCY
    )


def test_skips_python_files(tmp_path: Path) -> None:
    comps, _ = run_scanner(
        MultiLanguageScanner,
        tmp_path,
        {"app.py": 'import openai\nmodel = "gpt-4o"\n', "side.ts": 'import { foo } from "openai"\n'},
    )
    assert all(not c.file_path.endswith(".py") for c in comps)
    assert any(c.file_path.endswith(".ts") for c in comps)
