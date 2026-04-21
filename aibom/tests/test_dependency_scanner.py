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

    def test_parse_poetry_dependencies_uses_key_not_version_value(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "pyproject.toml": (
                    "[tool.poetry]\n"
                    'name = "demo"\n'
                    'version = "0.1.0"\n'
                    "\n"
                    "[tool.poetry.dependencies]\n"
                    'python = "^3.12"\n'
                    'stub-nonai-one = "0.18.0"\n'
                    'stub-nonai-two = "3.9.3"\n'
                    'freeplay = "0.5.4"\n'
                ),
            },
        )
        by_name = {c.name: c for c in comps}
        names = set(by_name)
        assert "freeplay" in names
        assert "stub-nonai-one" not in names
        assert "stub-nonai-two" not in names
        assert "0.18.0" not in names
        assert "3.9.3" not in names
        assert "0.5.4" not in names
        assert by_name["freeplay"].metadata["version_spec"] == "0.5.4"
        assert by_name["freeplay"].sdk_version == "0.5.4"

    def test_parse_pyproject_ignores_bare_version_strings(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "pyproject.toml": (
                    "[tool.poetry]\n"
                    'name = "demo"\n'
                    'version = "0.1.0"\n'
                    "\n"
                    "[tool.poetry.dependencies]\n"
                    'python = "^3.12"\n'
                    '"0.18.0"\n'
                    '"3.9.3"\n'
                    'openai = "^1.99.9"\n'
                ),
            },
        )
        names = {c.name for c in comps}
        assert "openai" in names
        assert "0.18.0" not in names
        assert "3.9.3" not in names

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

    def test_emits_only_ai_packages_from_mixed_manifests(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "requirements.txt": "requests>=2.0\nopenai>=1.0\n",
                "package.json": json.dumps({"dependencies": {"lodash": "4.17.21"}}),
            },
        )
        by_name = {c.name: c for c in comps}
        assert "openai" in by_name
        assert by_name["openai"].metadata["known_ai_package"] is True
        assert "requests" not in by_name
        assert "lodash" not in by_name

    def test_emits_only_ai_packages_from_go_mod(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "go.mod": (
                    "module example.com/x\n"
                    "go 1.22\n"
                    "require (\n"
                    "\tgithub.com/sashabaranov/go-openai v1.17.9\n"
                    "\tgithub.com/google/uuid v1.6.0\n"
                    ")\n"
                ),
            },
        )
        names = {c.name for c in comps}
        assert "github.com/sashabaranov/go-openai" in names
        assert "github.com/google/uuid" not in names

    def test_recognizes_additional_public_ai_packages(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "pyproject.toml": (
                    "[project.dependencies]\n"
                    '"guardrails>=0.5"\n'
                    '"google-genai>=1.0"\n'
                    '"openai-agents>=0.2"\n'
                    '"llmetry>=0.1"\n'
                ),
            },
        )
        names = {c.name for c in comps}
        assert "guardrails" in names
        assert "google-genai" in names
        assert "openai-agents" in names
        assert "llmetry" in names

    def test_recognizes_strands_packages(self, tmp_path: Path) -> None:
        """AWS Strands agent framework packages must be detected.

        Covers both bare names and extras syntax (``strands-agents[otel]``)
        since Strands docs frequently recommend the OpenTelemetry extra.
        """
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {
                "requirements.txt": (
                    "strands-agents>=1.0\n"
                    "strands-agents-tools>=1.0\n"
                    "mcp-proxy-for-aws>=0.1\n"
                ),
            },
        )
        names = {c.name for c in comps}
        assert "strands-agents" in names
        assert "strands-agents-tools" in names
        assert "mcp-proxy-for-aws" in names

    def test_recognizes_strands_with_extras(self, tmp_path: Path) -> None:
        """``strands-agents[otel]`` normalises to ``strands-agents``."""
        comps, _ = run_scanner(
            DependencyScanner,
            tmp_path,
            {"requirements.txt": "strands-agents[otel]>=1.0\n"},
        )
        names = {c.name for c in comps}
        assert "strands-agents" in names
