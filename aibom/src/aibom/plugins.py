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

"""Plugin discovery and management for AIBOM.

Tiered extensibility:

  1. **Catalog Extensions** — ``.aibom.yaml`` (existing).
  2. **Scanner Plugins** — Python ``entry_points`` group ``aibom.scanners``.
  3. **Reporter Plugins** — Python ``entry_points`` group ``aibom.reporters``.
  4. **Agentic Tools** — MCP servers via ``.mcp.json`` discovery.
  5. **Full Plugins** — Python packages with a ``manifest.yaml``.

Plugins are loaded lazily on first call to :func:`discover_scanner_plugins`
or :func:`discover_reporter_plugins`.
"""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

_LOGGER = logging.getLogger(__name__)

_ENTRY_GROUP_SCANNERS = "aibom.scanners"
_ENTRY_GROUP_REPORTERS = "aibom.reporters"


def discover_scanner_plugins() -> list[type]:
    """Load scanner classes from installed entry_points.

    Third-party packages declare scanner plugins like::

        [project.entry-points."aibom.scanners"]
        terraform = "my_plugin:TerraformScanner"

    Returns a list of scanner classes (subclasses of BaseScanner).
    """
    return _load_entry_points(_ENTRY_GROUP_SCANNERS)


def discover_reporter_plugins() -> list[type]:
    """Load reporter classes from installed entry_points.

    Third-party packages declare reporter plugins like::

        [project.entry-points."aibom.reporters"]
        grafana = "my_plugin:GrafanaReporter"

    Returns a list of reporter classes.
    """
    return _load_entry_points(_ENTRY_GROUP_REPORTERS)


def _load_entry_points(group: str) -> list[type]:
    plugins: list[type] = []
    try:
        eps = importlib.metadata.entry_points(group=group)
    except TypeError:
        eps = importlib.metadata.entry_points().get(group, [])

    for ep in eps:
        try:
            cls = ep.load()
            plugins.append(cls)
            _LOGGER.info("Loaded plugin %s from %s", ep.name, ep.value)
        except Exception:
            _LOGGER.warning("Failed to load plugin %s (%s)", ep.name, ep.value, exc_info=True)
    return plugins


def discover_mcp_configs() -> list[dict[str, Any]]:
    """Discover MCP server configs from standard locations.

    Searches in order:
      1. ``~/.aibom/.mcp.json`` (user-level)
      2. ``.aibom/.mcp.json`` (project-level, relative to CWD)

    Returns a list of MCP server configs (dicts with ``name``, ``command``, etc.).
    """
    import json

    configs: list[dict[str, Any]] = []
    candidates = [
        Path.home() / ".aibom" / ".mcp.json",
        Path.cwd() / ".aibom" / ".mcp.json",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            servers = data.get("mcpServers", data.get("servers", {}))
            for name, cfg in servers.items():
                cfg["_name"] = name
                cfg["_source"] = str(p)
                configs.append(cfg)
            _LOGGER.info("Loaded %d MCP server(s) from %s", len(servers), p)
        except (json.JSONDecodeError, OSError) as exc:
            _LOGGER.warning("Failed to parse MCP config %s: %s", p, exc)
    return configs


def load_plugin_manifest(manifest_path: Path) -> Optional[dict[str, Any]]:
    """Parse a plugin manifest YAML file.

    Expected format::

        name: my-terraform-scanner
        version: 1.0.0
        type: scanner  # or reporter, tool, enricher
        entry_point: my_plugin:TerraformScanner
        capabilities:
          - file_extensions: [".tf", ".tfvars"]
          - component_types: [model, endpoint]
    """
    if not manifest_path.is_file():
        return None
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "name" not in data:
            _LOGGER.warning("Invalid plugin manifest: %s", manifest_path)
            return None
        data["_manifest_path"] = str(manifest_path)
        return data
    except (yaml.YAMLError, OSError) as exc:
        _LOGGER.warning("Failed to load manifest %s: %s", manifest_path, exc)
        return None


def discover_plugin_manifests(
    search_dirs: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Discover plugin manifests from standard directories.

    Searches:
      - ``~/.aibom/plugins/*/manifest.yaml``
      - Any additional *search_dirs* provided.
    """
    dirs = [Path.home() / ".aibom" / "plugins"]
    if search_dirs:
        dirs.extend(search_dirs)

    manifests: list[dict[str, Any]] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for manifest_file in d.glob("*/manifest.yaml"):
            m = load_plugin_manifest(manifest_file)
            if m:
                manifests.append(m)
    return manifests


def list_plugins() -> dict[str, list[dict[str, str]]]:
    """List all discovered plugins by category."""
    result: dict[str, list[dict[str, str]]] = {
        "scanners": [],
        "reporters": [],
        "mcp_servers": [],
        "manifests": [],
    }

    for cls in discover_scanner_plugins():
        result["scanners"].append({
            "name": getattr(cls, "name", cls.__name__),
            "class": f"{cls.__module__}:{cls.__name__}",
        })

    for cls in discover_reporter_plugins():
        result["reporters"].append({
            "name": getattr(cls, "name", cls.__name__),
            "class": f"{cls.__module__}:{cls.__name__}",
        })

    for cfg in discover_mcp_configs():
        result["mcp_servers"].append({
            "name": cfg.get("_name", "unknown"),
            "source": cfg.get("_source", "unknown"),
        })

    for m in discover_plugin_manifests():
        result["manifests"].append({
            "name": m.get("name", "unknown"),
            "type": m.get("type", "unknown"),
            "version": m.get("version", "unknown"),
            "path": m.get("_manifest_path", "unknown"),
        })

    return result
