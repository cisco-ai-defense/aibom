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

from pathlib import Path
import tempfile

import duckdb

from aibom.catalog_db import CatalogDB


def _create_catalog(path: Path) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE component_catalog (
            id TEXT PRIMARY KEY,
            label TEXT,
            concept TEXT,
            framework TEXT,
            sig_name TEXT,
            type TEXT,
            catalog_label TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO component_catalog
        VALUES
            ('pkg.Agent', 'Agent', 'agent', 'pkg', 'Agent.__init__', 'class', 'Agent'),
            ('pkg.Tool.run', 'ToolRun', 'tool', 'pkg', 'Tool.run', 'method', 'ToolRun'),
            ('other.Unused', 'Unused', NULL, 'other', 'Unused', 'class', 'Unused');
        """
    )
    con.close()


def test_find_components_by_suffixes_returns_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"
        _create_catalog(db_file)

        with CatalogDB(db_file) as connector:
            results = connector.find_components_by_suffixes(["Agent", "Tool.run"])

        ids = {row["id"] for row in results}
        assert "pkg.Agent" in ids
        assert "pkg.Tool.run" in ids
        assert "other.Unused" not in ids


def test_find_components_by_suffixes_empty_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"
        _create_catalog(db_file)

        with CatalogDB(db_file) as connector:
            results = connector.find_components_by_suffixes([])

        assert results == []


def test_two_tier_precedence_duckdb_over_custom():
    """DuckDB entry beats custom entry with same ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"
        _create_catalog(db_file)

        with CatalogDB(db_file) as connector:
            connector.add_custom_entries([
                {
                    "id": "pkg.Agent",
                    "label": "Agent",
                    "concept": "tool",
                    "framework": "custom",
                    "sig_name": None,
                    "type": None,
                    "catalog_label": None,
                }
            ])
            results = connector.find_components_by_suffixes(["Agent"])

        agent_results = [r for r in results if r["id"] == "pkg.Agent"]
        assert len(agent_results) == 1
        assert agent_results[0]["concept"] == "agent"


def test_custom_entries_added_when_not_in_duckdb():
    """Custom entries are returned when not present in DuckDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"
        _create_catalog(db_file)

        with CatalogDB(db_file) as connector:
            connector.add_custom_entries([
                {
                    "id": "custom.MyModel",
                    "label": "MyModel",
                    "concept": "model",
                    "framework": "custom",
                    "sig_name": None,
                    "type": None,
                    "catalog_label": None,
                }
            ])
            results = connector.find_components_by_suffixes(["MyModel"])

        ids = {row["id"] for row in results}
        assert "custom.MyModel" in ids


def test_excludes_filter_results():
    """Excluded IDs are filtered out of query results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"
        _create_catalog(db_file)

        with CatalogDB(db_file) as connector:
            connector.add_excludes(["pkg.Agent"])
            results = connector.find_components_by_suffixes(["Agent", "Tool.run"])

        ids = {row["id"] for row in results}
        assert "pkg.Agent" not in ids
        assert "pkg.Tool.run" in ids
