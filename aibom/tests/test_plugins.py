# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aibom.plugins import (
    discover_mcp_configs,
    discover_plugin_manifests,
    discover_reporter_plugins,
    discover_scanner_plugins,
    list_plugins,
    load_plugin_manifest,
)


class TestEntryPointDiscovery:
    def test_no_plugins_returns_empty(self) -> None:
        with patch("aibom.plugins.importlib.metadata.entry_points", return_value=[]):
            assert discover_scanner_plugins() == []
            assert discover_reporter_plugins() == []

    def test_loads_scanner_entry_point(self) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "test_scanner"
        mock_ep.value = "test_pkg:TestScanner"

        class FakeScanner:
            name = "test_scanner"

        mock_ep.load.return_value = FakeScanner
        with patch(
            "aibom.plugins.importlib.metadata.entry_points",
            return_value=[mock_ep],
        ):
            plugins = discover_scanner_plugins()
        assert len(plugins) == 1
        assert plugins[0] is FakeScanner

    def test_handles_failed_load(self) -> None:
        mock_ep = MagicMock()
        mock_ep.name = "bad_plugin"
        mock_ep.value = "nonexistent:Class"
        mock_ep.load.side_effect = ImportError("no module")
        with patch(
            "aibom.plugins.importlib.metadata.entry_points",
            return_value=[mock_ep],
        ):
            plugins = discover_scanner_plugins()
        assert plugins == []


class TestMCPDiscovery:
    def test_discovers_user_mcp_config(self, tmp_path: Path) -> None:
        import json

        mcp_dir = tmp_path / ".aibom"
        mcp_dir.mkdir()
        config = {
            "mcpServers": {
                "terraform": {"command": "npx", "args": ["tf-mcp"]},
            }
        }
        (mcp_dir / ".mcp.json").write_text(json.dumps(config))

        with patch("aibom.plugins.Path.home", return_value=tmp_path):
            with patch("aibom.plugins.Path.cwd", return_value=Path("/nonexistent")):
                configs = discover_mcp_configs()

        assert len(configs) == 1
        assert configs[0]["_name"] == "terraform"

    def test_empty_when_no_config(self, tmp_path: Path) -> None:
        with patch("aibom.plugins.Path.home", return_value=tmp_path):
            with patch("aibom.plugins.Path.cwd", return_value=tmp_path):
                assert discover_mcp_configs() == []


class TestPluginManifest:
    def test_load_valid_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(
            "name: my-scanner\nversion: 1.0.0\ntype: scanner\n"
            "entry_point: my_pkg:MyScanner\n"
        )
        data = load_plugin_manifest(manifest)
        assert data is not None
        assert data["name"] == "my-scanner"
        assert data["type"] == "scanner"

    def test_missing_name_returns_none(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("version: 1.0.0\n")
        assert load_plugin_manifest(manifest) is None

    def test_nonexistent_returns_none(self, tmp_path: Path) -> None:
        assert load_plugin_manifest(tmp_path / "nope.yaml") is None

    def test_discover_manifests(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        (plugin_dir / "manifest.yaml").write_text(
            "name: my-plugin\nversion: 2.0.0\ntype: reporter\n"
        )
        manifests = discover_plugin_manifests(search_dirs=[tmp_path])
        assert len(manifests) == 1
        assert manifests[0]["name"] == "my-plugin"


class TestListPlugins:
    def test_list_empty(self) -> None:
        with patch("aibom.plugins.discover_scanner_plugins", return_value=[]):
            with patch("aibom.plugins.discover_reporter_plugins", return_value=[]):
                with patch("aibom.plugins.discover_mcp_configs", return_value=[]):
                    with patch("aibom.plugins.discover_plugin_manifests", return_value=[]):
                        result = list_plugins()
        assert result["scanners"] == []
        assert result["reporters"] == []
        assert result["mcp_servers"] == []
        assert result["manifests"] == []
