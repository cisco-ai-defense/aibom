# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from aibom.models import AIComponentType
from aibom.scanners.mcp_detector import McpDetector

from .conftest import run_scanner


class TestMcpDetector:
    def test_detects_mcp_server_from_mcp_json(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            McpDetector,
            tmp_path,
            {
                "mcp.json": json.dumps(
                    {
                        "mcpServers": {
                            "fetch": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-fetch"]},
                        },
                    },
                ),
            },
        )
        names = {c.name for c in comps if c.component_type == AIComponentType.MCP_SERVER}
        assert "fetch" in names

    def test_detects_mcp_server_from_claude_desktop_config(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            McpDetector,
            tmp_path,
            {
                "claude_desktop_config.json": json.dumps(
                    {
                        "mcpServers": {
                            "filesystem": {"command": "mcp-server-fs", "args": []},
                        },
                    },
                ),
            },
        )
        names = {c.name for c in comps if c.component_type == AIComponentType.MCP_SERVER}
        assert "filesystem" in names

    def test_detects_fastmcp_in_python(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            McpDetector,
            tmp_path,
            {"srv.py": "from mcp.server import Server\napp = FastMCP()\n"},
        )
        servers = [c for c in comps if c.component_type == AIComponentType.MCP_SERVER]
        assert len(servers) == 1
        assert "FastMCP" in servers[0].metadata.get("patterns", [])

    def test_detects_mcp_client_in_python(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            McpDetector,
            tmp_path,
            {"cli.py": "from foo import MCPClient\nx = MCPClient()\n"},
        )
        clients = [c for c in comps if c.component_type == AIComponentType.MCP_CLIENT]
        assert len(clients) == 1
        assert "MCPClient" in clients[0].metadata.get("patterns", [])

    def test_malformed_json_config_no_crash(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            McpDetector,
            tmp_path,
            {"mcp.json": "{ not valid json <<<"},
        )
        assert comps == []
