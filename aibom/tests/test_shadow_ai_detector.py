# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
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


def test_shadow_ai_detector_suppresses_syntax_warning_from_target_source(
    tmp_path: Path,
) -> None:
    """Regression: parsing third-party source that contains invalid escape
    sequences (e.g. ``"\\s"`` inside a regex string literal) used to leak
    ``SyntaxWarning`` messages to the operator's terminal. The detector must
    silence those warnings — they are noise about *scanned* code, not the
    aibom codebase.
    """
    noisy_source = (
        "import re\n"
        "import openai\n"
        # The "\s" literal triggers SyntaxWarning at ast.parse time on 3.12+.
        "PATTERN = re.compile(\"\\sfoo\")\n"
        "DOC = \"\"\"matches \\d+ digits and \\swhitespace\"\"\"\n"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_scanner(ShadowAIDetector, tmp_path, {"app.py": noisy_source})

    syntax_warnings = [w for w in caught if issubclass(w.category, SyntaxWarning)]
    assert syntax_warnings == [], (
        "ShadowAIDetector leaked SyntaxWarning(s) from scanned source: "
        f"{[str(w.message) for w in syntax_warnings]}"
    )
