# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving loop-registered tools to the functions they register."""

from __future__ import annotations

from pathlib import Path

from aibom.models import AIComponent, AIComponentType
from aibom.registration_flow import LoopRegistrationResolver
from aibom.scan_pipeline import _expand_loop_registered_tools


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _names(refs) -> list[tuple[str, str, int]]:
    return [(r.path.name, r.name, r.line) for r in refs]


class TestLoopRegistrationResolver:
    def test_literal_list_in_same_function(self, tmp_path: Path):
        _write(
            tmp_path,
            {
                "server.py": (
                    "def search(q):\n"
                    "    return q\n"
                    "\n"
                    "def lookup(k):\n"
                    "    return k\n"
                    "\n"
                    "def build(mcp):\n"
                    "    for tool in [search, lookup]:\n"
                    "        mcp.add_tool(tool)\n"
                )
            },
        )
        refs = LoopRegistrationResolver(tmp_path).resolve(
            tmp_path / "server.py", 9, "tool"
        )
        assert _names(refs) == [
            ("server.py", "search", 1),
            ("server.py", "lookup", 4),
        ]

    def test_inventory_object_through_parameters_and_imports(self, tmp_path: Path):
        _write(
            tmp_path,
            {
                "pkg/__init__.py": "",
                "pkg/tools/__init__.py": (
                    "from ..spec import ToolSet\n"
                    "from .kv import get_doc, put_doc\n"
                    "\n"
                    "TOOL_SET = ToolSet(read_only=(get_doc,), write=(put_doc,))\n"
                ),
                "pkg/tools/kv.py": (
                    "def get_doc(key):\n"
                    "    return key\n"
                    "\n"
                    "def put_doc(key, value):\n"
                    "    return value\n"
                ),
                "pkg/spec.py": (
                    "from dataclasses import dataclass\n"
                    "\n"
                    "@dataclass\n"
                    "class ToolSet:\n"
                    "    read_only: tuple = ()\n"
                    "    write: tuple = ()\n"
                    "\n"
                    "    def tools_for(self, read_only_mode):\n"
                    "        return list(self.read_only)\n"
                    "\n"
                    "@dataclass\n"
                    "class ServerSpec:\n"
                    "    name: str\n"
                    "    tools: ToolSet\n"
                ),
                "pkg/server_spec.py": (
                    "from .spec import ServerSpec\n"
                    "from .tools import TOOL_SET\n"
                    "\n"
                    'SPEC = ServerSpec(name="kv", tools=TOOL_SET)\n'
                ),
                "pkg/gating.py": (
                    "def gate(spec, read_only_mode):\n"
                    "    tools = spec.tools.tools_for(read_only_mode)\n"
                    "    enabled = [t for t in tools if t]\n"
                    "    final = []\n"
                    "    for t in enabled:\n"
                    "        final.append(t)\n"
                    "    return final\n"
                ),
                "pkg/app.py": (
                    "def build_app(spec, tools):\n"
                    "    for tool in tools:\n"
                    "        mcp.add_tool(tool)\n"
                ),
                "pkg/main.py": (
                    "from .app import build_app\n"
                    "from .gating import gate\n"
                    "\n"
                    "def start(spec):\n"
                    "    build_app(spec, tools=gate(spec, False))\n"
                    "\n"
                    "def main():\n"
                    "    from .server_spec import SPEC\n"
                    "    start(SPEC)\n"
                ),
            },
        )
        refs = LoopRegistrationResolver(tmp_path).resolve(
            tmp_path / "pkg" / "app.py", 3, "tool"
        )
        assert _names(refs) == [("kv.py", "get_doc", 1), ("kv.py", "put_doc", 4)]

    def test_wrapper_returning_closure_passes_function_through(self, tmp_path: Path):
        _write(
            tmp_path,
            {
                "server.py": (
                    "def guarded(fn):\n"
                    "    def inner(*a):\n"
                    "        return fn(*a)\n"
                    "    return inner\n"
                    "\n"
                    "def search(q):\n"
                    "    return q\n"
                    "\n"
                    "TOOLS = [guarded(search)]\n"
                    "\n"
                    "for tool in TOOLS:\n"
                    "    mcp.add_tool(tool)\n"
                )
            },
        )
        refs = LoopRegistrationResolver(tmp_path).resolve(
            tmp_path / "server.py", 12, "tool"
        )
        assert _names(refs) == [("server.py", "search", 6)]

    def test_unresolvable_iterable_yields_empty_list(self, tmp_path: Path):
        _write(
            tmp_path,
            {
                "server.py": (
                    "from plugins import discover\n"
                    "\n"
                    "for tool in discover():\n"
                    "    mcp.add_tool(tool)\n"
                )
            },
        )
        refs = LoopRegistrationResolver(tmp_path).resolve(
            tmp_path / "server.py", 4, "tool"
        )
        assert refs == []

    def test_non_loop_variable_returns_none(self, tmp_path: Path):
        _write(tmp_path, {"server.py": "def f():\n    pass\n\nmcp.add_tool(f)\n"})
        refs = LoopRegistrationResolver(tmp_path).resolve(
            tmp_path / "server.py", 4, "f"
        )
        assert refs is None

    def test_functions_in_test_files_are_ignored(self, tmp_path: Path):
        _write(
            tmp_path,
            {
                "tests/fake_tools.py": "def fake(q):\n    return q\n",
                "server.py": (
                    "from tests.fake_tools import fake\n"
                    "\n"
                    "for tool in [fake]:\n"
                    "    mcp.add_tool(tool)\n"
                ),
            },
        )
        refs = LoopRegistrationResolver(tmp_path).resolve(
            tmp_path / "server.py", 4, "tool"
        )
        assert refs == []


    def test_index_skips_vendored_directories(self, tmp_path: Path):
        from aibom.registration_flow import _RepoIndex

        _write(
            tmp_path,
            {
                "node_modules/pkg/vendored.py": "def vendored(q):\n    return q\n",
                ".venv/lib/site.py": "x = 1\n",
                "app/server.py": "def serve():\n    pass\n",
            },
        )
        files = _RepoIndex(tmp_path).files
        assert files == [tmp_path / "app" / "server.py"]

class TestExpandLoopRegisteredTools:
    @staticmethod
    def _row(path: Path, line: int, via: str = "add_tool(_pos_0)") -> AIComponent:
        return AIComponent(
            name="tool",
            component_type=AIComponentType.TOOL,
            file_path=str(path),
            line_number=line,
            metadata={"extracted_via": via},
        )

    def test_loop_row_becomes_one_tool_per_function(self, tmp_path: Path):
        server = tmp_path / "server.py"
        _write(
            tmp_path,
            {
                "server.py": (
                    "def search(q):\n"
                    "    return q\n"
                    "\n"
                    "for tool in (search,):\n"
                    "    mcp.add_tool(FunctionTool.from_function(tool))\n"
                )
            },
        )
        row = self._row(server, 5, "from_function(_pos_0)")
        [tool] = _expand_loop_registered_tools([row], [str(tmp_path)])
        assert (tool.name, tool.line_number) == ("search", 1)
        assert tool.file_path == str(server.resolve())
        assert tool.instance_id == f"search_{server.resolve()}_1"
        assert tool.metadata["reported_line"] == 5
        assert tool.metadata["registered_in_loop"] is True

    def test_unresolved_loop_row_is_dropped(self, tmp_path: Path):
        server = tmp_path / "server.py"
        _write(
            tmp_path,
            {"server.py": "for tool in load():\n    mcp.add_tool(tool)\n"},
        )
        assert (
            _expand_loop_registered_tools([self._row(server, 2)], [str(tmp_path)]) == []
        )

    def test_agent_tools_kwarg_rows_are_untouched(self, tmp_path: Path):
        server = tmp_path / "agent.py"
        _write(
            tmp_path,
            {"agent.py": "for tool in load():\n    Agent(tools=[tool])\n"},
        )
        row = self._row(server, 2, "agent_tools_kwarg")
        assert _expand_loop_registered_tools([row], [str(tmp_path)]) == [row]

    def test_direct_registration_is_untouched(self, tmp_path: Path):
        server = tmp_path / "server.py"
        _write(tmp_path, {"server.py": "def f():\n    pass\n\nmcp.add_tool(f)\n"})
        row = self._row(server, 4)
        assert _expand_loop_registered_tools([row], [str(tmp_path)]) == [row]
