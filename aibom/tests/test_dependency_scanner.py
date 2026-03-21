# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aibom.scanners.dependency_scanner import DependencyScanner

from .conftest import run_scanner


class TestDependencyScanner:
    def test_parse_requirements_txt_ai_packages(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {"requirements.txt": "openai>=1.0\nlangchain-core==0.2.0\n"},
        )
        names = {c.name for c in comps}
        assert "openai" in names
        assert "langchain-core" in names

    def test_parse_package_json_ai_packages(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "package.json": json.dumps(
                    {"dependencies": {"openai": "4.28.0", "@langchain/core": "0.2.0"}},
                    indent=2,
                ),
            },
        )
        names = {c.name for c in comps}
        assert "openai" in names
        assert "@langchain/core" in names

    def test_parse_pyproject_toml_dependencies(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "pyproject.toml": (
                    "[project.dependencies]\n"
                    '"openai>=1.0"\n'
                    '"anthropic>=0.18"\n'
                ),
            },
        )
        names = {c.name for c in comps}
        assert "openai" in names
        assert "anthropic" in names

    def test_parse_go_mod_ai_packages(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "go.mod": (
                    "module example.com/x\ngo 1.22\nrequire (\n"
                    "\tgithub.com/sashabaranov/go-openai v1.17.9\n)\n"
                ),
            },
        )
        assert any(
            "go-openai" in c.name or c.name == "github.com/sashabaranov/go-openai"
            for c in comps
        )

    def test_parse_cargo_toml_ai_packages(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "Cargo.toml": (
                    "[package]\nname = \"demo\"\n\n[dependencies]\n"
                    'async-openai = "0.28"\n'
                ),
            },
        )
        names = {c.name for c in comps}
        assert "async-openai" in names

    def test_ignores_non_ai_packages(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "requirements.txt": "requests>=2.0\n",
                "package.json": json.dumps({"dependencies": {"lodash": "4.17.21"}}),
            },
        )
        names = {c.name for c in comps}
        assert "requests" not in names
        assert "lodash" not in names
