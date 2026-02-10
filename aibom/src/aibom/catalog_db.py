# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""DuckDB catalog access for the prebuilt knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import duckdb

LOGGER = logging.getLogger(__name__)


class CatalogDB:
    """Provides read-only access to the DuckDB component catalog."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(f"DuckDB catalog not found at {self._db_path}")
        try:
            self._connection = duckdb.connect(str(self._db_path), read_only=True)
        except TypeError:
            self._connection = duckdb.connect(str(self._db_path))

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._connection:
            self._connection.close()

    def find_components_by_suffixes(self, suffixes: Sequence[str]) -> List[Dict[str, Any]]:
        """Return catalog entries whose IDs end with any of the provided suffixes."""
        if not suffixes:
            return []

        where_clause = " OR ".join("id LIKE ?" for _ in suffixes)
        params = [f"%{suffix}" for suffix in suffixes]
        query = f"""
            SELECT id, label, concept, framework, sig_name, type, catalog_label
            FROM component_catalog
            WHERE {where_clause}
        """

        cursor = self._connection.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
