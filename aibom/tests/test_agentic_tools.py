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

"""Tests for agentic tool implementations (no LangChain/deepagents required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aibom.agentic.tools import (
    _allowed_search_roots,
    analyze_imports_impl,
    lookup_model_impl,
    resolve_env_var_impl,
    scan_directory_impl,
    search_codebase_impl,
    trace_data_flow_impl,
)


@pytest.fixture(autouse=True)
def _clear_search_roots():
    """Reset module-level search roots so earlier test files don't pollute."""
    _allowed_search_roots.clear()
    yield
    _allowed_search_roots.clear()


class TestScanDirectory:
    def test_returns_json_with_components(self, tmp_path: Path):
        py = tmp_path / "app.py"
        py.write_text('from openai import OpenAI\nclient = OpenAI(model="gpt-4o")\n')
        result = json.loads(scan_directory_impl(str(tmp_path)))
        assert result["path"] == str(tmp_path)
        assert isinstance(result["total_components"], int)
        assert isinstance(result["components"], list)

    def test_empty_dir_returns_zero(self, tmp_path: Path):
        result = json.loads(scan_directory_impl(str(tmp_path)))
        assert result["total_components"] == 0


class TestResolveEnvVar:
    def test_finds_in_dotenv(self, tmp_path: Path):
        (tmp_path / ".env").write_text("MODEL_NAME=gpt-4o\nOTHER=123\n")
        result = json.loads(resolve_env_var_impl("MODEL_NAME", [str(tmp_path)]))
        assert result["resolved"] is True
        assert any(m["value"] == "gpt-4o" for m in result["matches"])

    def test_finds_in_docker_compose(self, tmp_path: Path):
        (tmp_path / "docker-compose.yml").write_text(
            "services:\n  app:\n    environment:\n      LLM_MODEL: claude-3\n"
        )
        result = json.loads(resolve_env_var_impl("LLM_MODEL", [str(tmp_path)]))
        assert result["resolved"] is True
        assert any(m["value"] == "claude-3" for m in result["matches"])

    def test_not_found_returns_resolved_false(self, tmp_path: Path):
        (tmp_path / ".env").write_text("OTHER=123\n")
        result = json.loads(resolve_env_var_impl("MODEL_NAME", [str(tmp_path)]))
        assert result["resolved"] is False

    def test_searches_multiple_paths(self, tmp_path: Path):
        d1 = tmp_path / "repo1"
        d2 = tmp_path / "repo2"
        d1.mkdir()
        d2.mkdir()
        (d2 / ".env").write_text("API_KEY=sk-test\n")
        result = json.loads(
            resolve_env_var_impl("API_KEY", [str(d1), str(d2)])
        )
        assert result["resolved"] is True


class TestLookupModel:
    def test_known_model_returns_found(self):
        result = json.loads(lookup_model_impl("gpt-4o"))
        assert result["found"] is True
        assert result["provider"] != ""

    def test_unknown_model_returns_not_found(self):
        result = json.loads(lookup_model_impl("nonexistent-model-xyz-999"))
        assert result["found"] is False


class TestAnalyzeImports:
    def test_python_file_returns_imports(self, tmp_path: Path):
        py = tmp_path / "sample.py"
        py.write_text("import os\nfrom pathlib import Path\nx = Path('.')\n")
        result = json.loads(analyze_imports_impl(str(py)))
        assert any("os" in imp for imp in result["imports"])

    def test_non_python_returns_error(self, tmp_path: Path):
        txt = tmp_path / "readme.txt"
        txt.write_text("hello")
        result = json.loads(analyze_imports_impl(str(txt)))
        assert "error" in result


class TestTraceDataFlow:
    def test_finds_string_literal(self, tmp_path: Path):
        py = tmp_path / "config.py"
        py.write_text('MODEL = "gpt-4o"\nclient = OpenAI(model=MODEL)\n')
        result = json.loads(trace_data_flow_impl("MODEL", str(py)))
        assert result["resolved"] is True
        assert result["concrete_value"] == "gpt-4o"

    def test_finds_env_var_reference(self, tmp_path: Path):
        py = tmp_path / "config.py"
        py.write_text('import os\nMODEL = os.environ["LLM_MODEL"]\n')
        result = json.loads(trace_data_flow_impl("MODEL", str(py)))
        assert len(result["env_var_references"]) > 0
        assert result["env_var_references"][0]["env_var"] == "LLM_MODEL"

    def test_missing_file_returns_error(self):
        result = json.loads(trace_data_flow_impl("X", "/nonexistent/path.py"))
        assert "error" in result


class TestSearchCodebase:
    def test_literal_search(self, tmp_path: Path):
        (tmp_path / "app.py").write_text('model = "gpt-4o"\n')
        result = json.loads(
            search_codebase_impl("gpt-4o", [str(tmp_path)], literal=True)
        )
        assert result["total_matches"] >= 1

    def test_regex_search(self, tmp_path: Path):
        (tmp_path / "app.py").write_text('model = "gpt-4o-2024-08-06"\n')
        result = json.loads(
            search_codebase_impl(r"gpt-4o-\d{4}", [str(tmp_path)])
        )
        assert result["total_matches"] >= 1

    def test_no_matches(self, tmp_path: Path):
        (tmp_path / "app.py").write_text("x = 1\n")
        result = json.loads(
            search_codebase_impl("nonexistent_pattern", [str(tmp_path)])
        )
        assert result["total_matches"] == 0

    def test_invalid_regex_returns_error(self):
        result = json.loads(search_codebase_impl("[invalid", ["/tmp"]))
        assert "error" in result
