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
            id TEXT,
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

        connector = CatalogDB(db_file)
        results = connector.find_components_by_suffixes(["Agent", "Tool.run"])
        connector.close()

        ids = {row["id"] for row in results}
        assert "pkg.Agent" in ids
        assert "pkg.Tool.run" in ids
        assert "other.Unused" not in ids


def test_find_components_by_suffixes_empty_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "catalog.duckdb"
        _create_catalog(db_file)

        connector = CatalogDB(db_file)
        results = connector.find_components_by_suffixes([])
        connector.close()

        assert results == []
