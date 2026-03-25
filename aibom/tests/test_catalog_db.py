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


def _create_catalog(path: Path, *, with_token_table: bool = True) -> None:
    """Build a minimal test catalog.

    When *with_token_table* is True (the default), the pre-built
    ``component_catalog_last_seg`` table is included, simulating a
    new-format KB.  Set to False to test the old-KB fallback path.
    """
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
    if with_token_table:
        con.execute(
            """
            CREATE TABLE component_catalog_last_seg AS
            SELECT id, split_part(id, '.', -1) AS last_seg
            FROM component_catalog;
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


# ── Old-KB fallback (no pre-built token table) ───────────────────────


def test_old_kb_fallback_builds_temp_token_table():
    """When the catalog has no component_catalog_last_seg table,
    CatalogDB builds a temporary one and lookups still work."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"
        _create_catalog(db_file, with_token_table=False)

        with CatalogDB(db_file) as connector:
            results = connector.find_components_by_suffixes(["Agent", "run"])

        ids = {row["id"] for row in results}
        assert "pkg.Agent" in ids
        assert "pkg.Tool.run" in ids
        assert "other.Unused" not in ids


def test_old_kb_full_path_exact_match():
    """Full dotted path used as a suffix resolves via exact id match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"
        _create_catalog(db_file, with_token_table=False)

        with CatalogDB(db_file) as connector:
            results = connector.find_components_by_suffixes(["pkg.Tool.run"])

        ids = {row["id"] for row in results}
        assert "pkg.Tool.run" in ids


# ── Large suffix list ─────────────────────────────────────────────────


def test_large_suffix_list():
    """Verify that hundreds of suffixes don't cause query issues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"

        con = duckdb.connect(str(db_file))
        con.execute(
            """
            CREATE TABLE component_catalog (
                id TEXT PRIMARY KEY, label TEXT, concept TEXT,
                framework TEXT, sig_name TEXT, type TEXT, catalog_label TEXT
            );
            """
        )
        entries = []
        for i in range(500):
            entries.append((
                f"fw.mod{i}.Class{i}", f"Class{i}", "model",
                "fw", None, "class", None,
            ))
        con.executemany(
            "INSERT INTO component_catalog VALUES (?, ?, ?, ?, ?, ?, ?)",
            entries,
        )
        con.execute(
            """
            CREATE TABLE component_catalog_last_seg AS
            SELECT id, split_part(id, '.', -1) AS last_seg
            FROM component_catalog;
            """
        )
        con.close()

        suffixes = [f"Class{i}" for i in range(500)]
        suffixes += [f"nonexistent_{i}" for i in range(500)]

        with CatalogDB(db_file) as connector:
            results = connector.find_components_by_suffixes(suffixes)

        ids = {row["id"] for row in results}
        assert len(ids) == 500
        assert "fw.mod0.Class0" in ids
        assert "fw.mod499.Class499" in ids


def test_find_ids_by_path_and_concept():
    """find_ids_by_path_and_concept still works through the VIEW alias."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"
        _create_catalog(db_file)

        with CatalogDB(db_file) as connector:
            ids = connector.find_ids_by_path_and_concept("pkg", ["agent"])

        assert "pkg.Agent" in ids
