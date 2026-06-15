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
                            "fetch": {
                                "command": "npx",
                                "args": ["-y", "@modelcontextprotocol/server-fetch"],
                            },
                        },
                    },
                ),
            },
        )
        names = {
            c.name for c in comps if c.component_type == AIComponentType.MCP_SERVER
        }
        assert "fetch" in names

    def test_detects_mcp_server_from_claude_desktop_config(
        self, tmp_path: Path
    ) -> None:
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
        names = {
            c.name for c in comps if c.component_type == AIComponentType.MCP_SERVER
        }
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

    def test_detects_streamablehttp_client_with_session(self, tmp_path: Path) -> None:
        # Modern MCP SDK idiom: streamablehttp_client + ClientSession, where
        # the file imports ClientSession from ``mcp`` (not ``mcp.client``).
        src = (
            "from mcp import ClientSession\n"
            "from mcp.client.streamable_http import streamablehttp_client\n"
            "\n"
            "async def run(url):\n"
            "    async with streamablehttp_client(url) as (r, w, _):\n"
            "        async with ClientSession(r, w) as session:\n"
            "            await session.initialize()\n"
        )
        comps, _ = run_scanner(McpDetector, tmp_path, {"agent.py": src})
        clients = [c for c in comps if c.component_type == AIComponentType.MCP_CLIENT]
        assert len(clients) == 1
        patterns = clients[0].metadata.get("patterns", [])
        assert "streamablehttp_client" in patterns
        assert "ClientSession" in patterns

    def test_detects_google_adk_mcp_toolset(self, tmp_path: Path) -> None:
        src = (
            "from google.adk.tools.mcp_tool import McpToolset\n"
            "from google.adk.tools.mcp_tool.mcp_session_manager import (\n"
            "    StreamableHTTPConnectionParams,\n"
            ")\n"
            "\n"
            "def build(url):\n"
            "    return McpToolset(\n"
            "        connection_params=StreamableHTTPConnectionParams(url=url),\n"
            "    )\n"
        )
        comps, _ = run_scanner(McpDetector, tmp_path, {"adk_agent.py": src})
        clients = [c for c in comps if c.component_type == AIComponentType.MCP_CLIENT]
        assert len(clients) == 1
        patterns = clients[0].metadata.get("patterns", [])
        assert "McpToolset" in patterns

    def test_distinct_clients_across_files_not_collapsed(self, tmp_path: Path) -> None:
        src = (
            "from mcp import ClientSession\n"
            "from mcp.client.streamable_http import streamablehttp_client\n"
            "async def run(url):\n"
            "    async with streamablehttp_client(url) as (r, w, _):\n"
            "        async with ClientSession(r, w) as s:\n"
            "            await s.initialize()\n"
        )
        comps, _ = run_scanner(
            McpDetector,
            tmp_path,
            {"a/agent.py": src, "b/agent.py": src},
        )
        clients = [c for c in comps if c.component_type == AIComponentType.MCP_CLIENT]
        assert len(clients) == 2

    def test_plain_clientsession_without_mcp_not_detected(self, tmp_path: Path) -> None:
        # A ClientSession unrelated to MCP (no mcp import / transport) must
        # not be misdetected as an MCP client.
        src = (
            "from aiohttp import ClientSession\n"
            "async def run():\n"
            "    async with ClientSession() as s:\n"
            "        await s.get('https://example.com')\n"
        )
        comps, _ = run_scanner(McpDetector, tmp_path, {"http.py": src})
        clients = [c for c in comps if c.component_type == AIComponentType.MCP_CLIENT]
        assert clients == []
