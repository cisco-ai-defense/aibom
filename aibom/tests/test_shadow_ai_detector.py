# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aibom.scanners.shadow_ai_detector import ShadowAIDetector

from .conftest import run_scanner


def test_shadow_detects_openai_not_in_requirements(tmp_path: Path) -> None:
    comps, rels = run_scanner(ShadowAIDetector, tmp_path, {"app.py": "import openai\n"})
    assert rels == []
    assert len(comps) == 1
    assert comps[0].name == "openai"
    assert comps[0].metadata.get("shadow_ai") is True


def test_shadow_no_finding_when_openai_declared(tmp_path: Path) -> None:
    comps, _ = run_scanner(
        ShadowAIDetector,
        tmp_path,
        {"requirements.txt": "openai>=1.0.0\n", "app.py": "import openai\n"},
    )
    assert comps == []


def test_shadow_detects_importlib_anthropic(tmp_path: Path) -> None:
    comps, _ = run_scanner(
        ShadowAIDetector,
        tmp_path,
        {"app.py": 'import importlib\nm = importlib.import_module("anthropic")\n'},
    )
    assert len(comps) == 1
    assert comps[0].name == "anthropic"


def test_shadow_langchain_core_without_manifest(tmp_path: Path) -> None:
    comps, _ = run_scanner(
        ShadowAIDetector,
        tmp_path,
        {"app.py": "from langchain_core.runnables import Runnable\n"},
    )
    assert len(comps) >= 1
    names = {c.name for c in comps}
    assert "langchain-core" in names or "langchain" in names
