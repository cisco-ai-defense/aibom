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

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pathspec import PathSpec

from ..models import AIComponent, ComponentRelationship, ScanContext
from ..models.enums import AIComponentType, DetectionSource
from .base import BaseScanner
from .file_cache import read_text_cached

_LOGGER = logging.getLogger(__name__)

_CONFIG_NAMES = frozenset(
    {
        "mcp.json",
        ".mcp.json",
        "mcp_config.json",
        "claude_desktop_config.json",
        "cline_mcp_settings.json",
    }
)

_RE_FROM_MCP_IMPORT = re.compile(r"from\s+mcp\s+import\b")
_RE_FROM_MCP_SERVER_IMPORT = re.compile(r"from\s+mcp\.server\s+import\b")
_RE_FASTMCP = re.compile(r"\bFastMCP\s*\(")
_RE_SERVER_CALL = re.compile(r"\bServer\s*\(")
_RE_MCP_TOOL_DECORATOR = re.compile(
    r"@\s*\w+\s*\.\s*(tool|resource|prompt)\s*\(", re.MULTILINE
)
_RE_MCP_CLIENT = re.compile(r"\bMCPClient\s*\(")
_RE_MULTI_MCP_CLIENT = re.compile(r"\bMultiServerMCPClient\s*\(")
_RE_LANGCHAIN_MCP_ADAPTERS = re.compile(r"from\s+langchain_mcp_adapters\b")
_RE_CLIENT_SESSION = re.compile(r"\bClientSession\s*\(")
_RE_MCP_CLIENT_MODULE = re.compile(r"\bmcp\.client\b")


def _is_mcp_config_path(path: Path) -> bool:
    if path.name in _CONFIG_NAMES:
        return True
    parts = path.parts
    return len(parts) >= 2 and parts[-2] == ".cursor" and path.name == "mcp.json"


def _load_exclude_spec(patterns: list[str]) -> Optional[PathSpec]:
    if not patterns:
        return None
    return PathSpec.from_lines("gitwildmatch", patterns)


def _is_excluded(
    file_path: Path, root: Path, spec: Optional[PathSpec]
) -> bool:
    if not spec:
        return False
    try:
        rel = file_path.relative_to(root).as_posix()
    except ValueError:
        rel = file_path.as_posix()
    return spec.match_file(rel)


def _iter_files_under(root: Path, spec: Optional[PathSpec]) -> list[Path]:
    out: list[Path] = []
    if root.is_file():
        if not _is_excluded(root, root.parent, spec):
            out.append(root)
        return out
    if not root.is_dir():
        return out
    for p in root.rglob("*"):
        if p.is_file() and not _is_excluded(p, root, spec):
            out.append(p)
    return out


def _extract_servers_blob(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    for key in ("mcpServers", "servers"):
        blob = data.get(key)
        if isinstance(blob, dict):
            return blob
    return {}


def _infer_transport(server: dict[str, Any]) -> str:
    raw = server.get("transport") or server.get("type")
    if isinstance(raw, str) and raw.strip():
        t = raw.strip().lower()
        if t in ("stdio", "sse", "streamable-http", "streamable_http"):
            return "streamable-http" if t == "streamable_http" else t
    if server.get("url"):
        url = str(server.get("url", "")).lower()
        if "streamable" in url:
            return "streamable-http"
        return "sse"
    return "stdio"


def _server_metadata(server: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if "command" in server:
        meta["command"] = server.get("command")
    if "args" in server:
        meta["args"] = server.get("args")
    if "url" in server:
        meta["url"] = server.get("url")
    env = server.get("env")
    if isinstance(env, dict):
        meta["env_keys"] = sorted(str(k) for k in env.keys())
    return meta


def _parse_config_file(path: Path) -> tuple[Optional[dict[str, Any]], bool]:
    try:
        text = read_text_cached(path)
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None, False
    servers = _extract_servers_blob(data)
    return data, bool(servers)


def _components_from_config(path: Path) -> list[AIComponent]:
    data, has_servers = _parse_config_file(path)
    if not has_servers or data is None:
        return []
    servers = _extract_servers_blob(data)
    cfg = str(path.resolve())
    out: list[AIComponent] = []
    for name, raw in servers.items():
        if not isinstance(raw, dict):
            continue
        transport = _infer_transport(raw)
        meta = _server_metadata(raw)
        out.append(
            AIComponent(
                name=str(name),
                component_type=AIComponentType.MCP_SERVER,
                file_path=cfg,
                line_number=0,
                framework="mcp",
                detection_source=DetectionSource.CONFIG_FILE,
                transport=transport,
                config_source=cfg,
                metadata=meta,
            )
        )
    return out


def _line_for_match(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _first_match_line(pattern: re.Pattern[str], text: str) -> Optional[int]:
    m = pattern.search(text)
    if not m:
        return None
    return _line_for_match(text, m.start())


def _components_from_python(path: Path) -> list[AIComponent]:
    try:
        text = read_text_cached(path)
    except OSError:
        return []
    fp = str(path.resolve())
    out: list[AIComponent] = []

    server_hits: list[tuple[str, int]] = []
    m_imp = _RE_FROM_MCP_IMPORT.search(text)
    if m_imp:
        server_hits.append(("from_mcp_import", _line_for_match(text, m_imp.start())))
    m_srv_imp = _RE_FROM_MCP_SERVER_IMPORT.search(text)
    if m_srv_imp:
        server_hits.append(
            (
                "from_mcp_server_import",
                _line_for_match(text, m_srv_imp.start()),
            )
        )
    ln = _first_match_line(_RE_FASTMCP, text)
    if ln is not None:
        server_hits.append(("FastMCP", ln))
    m_srv_call = _RE_SERVER_CALL.search(text)
    if m_srv_call and (
        _RE_FROM_MCP_SERVER_IMPORT.search(text) or "mcp.server" in text
    ):
        server_hits.append(("Server", _line_for_match(text, m_srv_call.start())))
    if server_hits:
        server_hits.sort(key=lambda x: x[1])
        kinds = [k for k, _ in server_hits]
        out.append(
            AIComponent(
                name=f"{path.stem}_mcp_server",
                component_type=AIComponentType.MCP_SERVER,
                file_path=fp,
                line_number=server_hits[0][1],
                framework="mcp",
                detection_source=DetectionSource.CODE_ANALYSIS,
                metadata={"patterns": kinds},
            )
        )

    mdec = _RE_MCP_TOOL_DECORATOR.search(text)
    has_mcp = bool(m_imp or m_srv_imp or _RE_FASTMCP.search(text))
    if mdec:
        if has_mcp:
            out.append(
                AIComponent(
                    name=f"{path.stem}_mcp_tooling",
                    component_type=AIComponentType.TOOL,
                    file_path=fp,
                    line_number=_line_for_match(text, mdec.start()),
                    framework="mcp",
                    detection_source=DetectionSource.CODE_ANALYSIS,
                    metadata={"mcp_decorators": True},
                )
            )
        else:
            out.append(
                AIComponent(
                    name=f"{path.stem}_mcp_tooling",
                    component_type=AIComponentType.TOOL,
                    file_path=fp,
                    line_number=_line_for_match(text, mdec.start()),
                    framework="mcp",
                    detection_source=DetectionSource.CODE_ANALYSIS,
                    confidence=0.3,
                    needs_agentic=True,
                    agentic_hint="@tool decorator without MCP imports in file",
                    metadata={"mcp_decorators": True},
                )
            )

    client_hits: list[tuple[str, int]] = []
    for label, pat in (
        ("MCPClient", _RE_MCP_CLIENT),
        ("MultiServerMCPClient", _RE_MULTI_MCP_CLIENT),
    ):
        m = pat.search(text)
        if m:
            client_hits.append((label, _line_for_match(text, m.start())))
    m_lc = _RE_LANGCHAIN_MCP_ADAPTERS.search(text)
    if m_lc:
        client_hits.append(
            ("langchain_mcp_adapters", _line_for_match(text, m_lc.start()))
        )
    m_cs = _RE_CLIENT_SESSION.search(text)
    if m_cs and _RE_MCP_CLIENT_MODULE.search(text):
        client_hits.append(("ClientSession", _line_for_match(text, m_cs.start())))
    if client_hits:
        client_hits.sort(key=lambda x: x[1])
        out.append(
            AIComponent(
                name=f"{path.stem}_mcp_client",
                component_type=AIComponentType.MCP_CLIENT,
                file_path=fp,
                line_number=client_hits[0][1],
                framework="mcp",
                detection_source=DetectionSource.CODE_ANALYSIS,
                metadata={"patterns": [k for k, _ in client_hits]},
            )
        )

    return out


def _run_optional_yara_on_configs(config_paths: list[str]) -> None:
    try:
        from mcpscanner import Config, Scanner
        from mcpscanner.core.models import AnalyzerEnum
    except ImportError:
        return

    async def _scan_all() -> None:
        scanner = Scanner(Config())
        for cfg in config_paths:
            try:
                await scanner.scan_mcp_config_file(
                    cfg,
                    analyzers=[AnalyzerEnum.YARA],
                )
            except Exception:
                _LOGGER.debug(
                    "mcpscanner YARA scan failed for %s", cfg, exc_info=True
                )

    try:
        asyncio.run(_scan_all())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_scan_all())
        finally:
            loop.close()


class McpDetector(BaseScanner):
    name = "mcp_detector"

    def supports(self, context: ScanContext) -> bool:
        return True

    def scan(
        self, context: ScanContext
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        config_paths_seen: set[str] = set()

        idx = context.file_index()
        if idx:
            all_files = [e.path for entries in idx.values() for e in entries]
        else:
            spec = _load_exclude_spec(context.exclude_patterns)
            all_files = []
            for raw in context.paths:
                root = Path(raw).expanduser()
                all_files.extend(_iter_files_under(root, spec))

        for file_path in all_files:
            if _is_mcp_config_path(file_path):
                config_paths_seen.add(str(file_path.resolve()))
                components.extend(_components_from_config(file_path))
            elif file_path.suffix == ".py":
                components.extend(_components_from_python(file_path))

        if config_paths_seen:
            _run_optional_yara_on_configs(sorted(config_paths_seen))

        return components, []
