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
import threading
from pathlib import Path

import pytest

from aibom.agentic.tools import (
    _reset_tool_stats,
    analyze_imports_impl,
    get_tool_stats,
    list_directory_tree_impl,
    lookup_model_impl,
    reset_allowed_search_roots,
    reset_strict_tool_root_enforcement,
    resolve_env_var_impl,
    scan_directory_impl,
    search_codebase_impl,
    set_allowed_search_roots,
    set_strict_tool_root_enforcement,
    trace_data_flow_impl,
)


@pytest.fixture(autouse=True)
def _clear_search_roots():
    """Reset module-level search roots so earlier test files don't pollute."""
    roots_token = set_allowed_search_roots(None)
    strict_token = set_strict_tool_root_enforcement(False)
    _reset_tool_stats()
    yield
    reset_strict_tool_root_enforcement(strict_token)
    reset_allowed_search_roots(roots_token)
    _reset_tool_stats()


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
        result = json.loads(resolve_env_var_impl("API_KEY", [str(d1), str(d2)]))
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
        result = json.loads(search_codebase_impl(r"gpt-4o-\d{4}", [str(tmp_path)]))
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


class TestReadFileSnippetInBuildTools:
    """Fix 2: read_file_snippet must be in the main scanner tools."""

    def test_build_tools_includes_read_file_snippet(self):
        pytest.importorskip("langchain_core")
        from aibom.agentic.tools import build_tools

        tools = build_tools()
        tool_names = [t.name for t in tools]
        assert "read_file_snippet" in tool_names

    def test_read_file_snippet_reads_file(self, tmp_path: Path):
        from aibom.agentic.tools import read_file_snippet_impl

        test_file = tmp_path / "agent.py"
        test_file.write_text("class SecurityAgent:\n    pass\n")
        result = read_file_snippet_impl(str(test_file))
        assert "SecurityAgent" in result

    def test_read_file_snippet_nonexistent(self):
        from aibom.agentic.tools import read_file_snippet_impl

        result = read_file_snippet_impl("/nonexistent/path.py")
        assert "error" in result


class TestApprovedSourceRootGuard:
    def test_non_raw_mode_preserves_legacy_direct_file_access(
        self, tmp_path: Path
    ) -> None:
        from aibom.agentic.tools import read_file_snippet_impl

        approved = tmp_path / "approved"
        approved.mkdir()
        outside = tmp_path / "outside.py"
        outside.write_text("LEGACY_DIRECT_READ = True\n")
        set_allowed_search_roots([str(approved)])

        assert "LEGACY_DIRECT_READ" in read_file_snippet_impl(str(outside))

    def test_explicitly_empty_root_set_denies_all_access(self, tmp_path: Path) -> None:
        (tmp_path / "canary.py").write_text("PRIVATE_CANARY = True\n")
        set_allowed_search_roots([])
        set_strict_tool_root_enforcement(True)

        output = scan_directory_impl(str(tmp_path))

        assert "PRIVATE_CANARY" not in output
        assert "outside the approved source root" in output
        assert get_tool_stats()["scan_directory"]["guard_denials"] == 1

    def test_file_and_directory_tools_block_and_count_outside_access(
        self, tmp_path: Path
    ) -> None:
        from aibom.agentic.tools import (
            list_directory_tree_impl,
            read_file_snippet_impl,
        )

        approved = tmp_path / "approved"
        approved.mkdir()
        (approved / "inside.py").write_text("inside = True\n")
        outside = tmp_path / "outside.py"
        outside.write_text("PRIVATE_CANARY = 'must-not-be-read'\n")
        set_allowed_search_roots([str(approved)])
        set_strict_tool_root_enforcement(True)

        outputs = [
            analyze_imports_impl(str(outside)),
            trace_data_flow_impl("PRIVATE_CANARY", str(outside)),
            read_file_snippet_impl(str(outside)),
            list_directory_tree_impl(str(tmp_path)),
            search_codebase_impl("must-not-be-read", [str(tmp_path)], literal=True),
        ]

        assert all("must-not-be-read" not in output for output in outputs[:4])
        assert json.loads(outputs[4])["total_matches"] == 0
        assert all(
            "outside the approved source root" in output for output in outputs[:4]
        )
        stats = get_tool_stats()
        for tool_name in (
            "analyze_imports",
            "trace_data_flow",
            "read_file_snippet",
            "list_directory_tree",
            "search_codebase",
        ):
            assert stats[tool_name]["guard_denials"] == 1
            assert stats[tool_name]["calls"] == 1

    def test_recursive_tools_do_not_follow_outside_symlinks(
        self, tmp_path: Path
    ) -> None:
        approved = tmp_path / "approved"
        outside = tmp_path / "outside"
        approved.mkdir()
        outside.mkdir()
        (outside / ".env").write_text("PRIVATE_CANARY=must-not-be-read\n")
        (outside / "settings.yaml").write_text("PRIVATE_CANARY: must-not-be-read\n")
        (outside / "secret.py").write_text("PRIVATE_CANARY = 'must-not-be-read'\n")
        (approved / ".env").symlink_to(outside / ".env")
        (approved / "settings.yaml").symlink_to(outside / "settings.yaml")
        (approved / "secret.py").symlink_to(outside / "secret.py")
        (approved / "outside-dir").symlink_to(outside, target_is_directory=True)
        set_allowed_search_roots([str(approved)])
        set_strict_tool_root_enforcement(True)

        env_result = resolve_env_var_impl("PRIVATE_CANARY", [str(approved)])
        search_result = search_codebase_impl(
            "PRIVATE_CANARY", [str(approved)], literal=True
        )
        tree_result = list_directory_tree_impl(str(approved))

        assert json.loads(env_result)["resolved"] is False
        assert json.loads(search_result)["total_matches"] == 0
        assert "secret.py" not in tree_result
        assert "outside-dir" not in tree_result
        assert "must-not-be-read" not in env_result + search_result + tree_result
        stats = get_tool_stats()
        assert stats["resolve_env_var"]["guard_denials"] >= 1
        assert stats["search_codebase"]["guard_denials"] >= 1
        assert stats["list_directory_tree"]["guard_denials"] >= 1


class TestRunScopedSourceRoots:
    def test_concurrent_scans_keep_independent_roots(self, tmp_path: Path) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "value.txt").write_text("FIRST_ONLY")
        (second / "value.txt").write_text("SECOND_ONLY")
        barrier = threading.Barrier(2)
        outputs: dict[str, dict] = {}

        def _scan(name: str, root: Path, own_value: str) -> None:
            token = set_allowed_search_roots([str(root)])
            try:
                barrier.wait(timeout=2)
                outputs[name] = json.loads(
                    search_codebase_impl(own_value, [str(root)], literal=True)
                )
            finally:
                reset_allowed_search_roots(token)

        threads = [
            threading.Thread(target=_scan, args=("first", first, "FIRST_ONLY")),
            threading.Thread(target=_scan, args=("second", second, "SECOND_ONLY")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=3)

        assert outputs["first"]["total_matches"] == 1
        assert outputs["second"]["total_matches"] == 1
        assert "second" not in json.dumps(outputs["first"])
        assert "first" not in json.dumps(outputs["second"])

    def test_private_async_runner_propagates_roots(self, tmp_path: Path) -> None:
        from aibom.agentic.agent import _run_async_bounded

        approved = tmp_path / "approved"
        outside = tmp_path / "outside"
        approved.mkdir()
        outside.mkdir()
        (outside / "canary.txt").write_text("MUST_NOT_CROSS_CONTEXT")

        async def _search() -> dict:
            return json.loads(
                search_codebase_impl(
                    "MUST_NOT_CROSS_CONTEXT", [str(outside)], literal=True
                )
            )

        token = set_allowed_search_roots([str(approved)])
        try:
            result = _run_async_bounded(_search())
        finally:
            reset_allowed_search_roots(token)

        assert result["total_matches"] == 0
