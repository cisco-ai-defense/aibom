# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""DuckDB catalog access for the prebuilt knowledge base.

Precedence: DuckDB > custom.  Excluded IDs are filtered out.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import duckdb

LOGGER = logging.getLogger(__name__)


def is_excluded(name: str, patterns: List[str]) -> bool:
    """Return True if *name* suffix-matches or equals any pattern in *patterns*."""
    for pattern in patterns:
        if name.endswith(pattern) or name == pattern:
            return True
    return False


class CatalogDB:
    """Provides read-only access to the DuckDB component catalog,
    augmented with user-defined custom entries from ``.aibom.yaml``."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(f"DuckDB catalog not found at {self._db_path}")
        try:
            self._connection = duckdb.connect(str(self._db_path), read_only=True)
        except TypeError:
            self._connection = duckdb.connect(str(self._db_path))
        self._custom_index: Dict[str, Dict[str, Any]] = {}
        self._excludes: List[str] = []

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._connection:
            self._connection.close()

    def add_custom_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Merge user-provided custom catalog entries.

        All entries are stored in the custom index.  Precedence is enforced
        at query time in ``find_components_by_suffixes``: DuckDB results are
        returned first and custom entries are only included when their ID has
        not already been seen in the DuckDB results.
        """
        for entry in entries:
            entry_id = entry.get("id")
            if not entry_id:
                continue
            self._custom_index[entry_id] = entry
            LOGGER.debug("Added custom catalog entry: %s", entry_id)

    def add_excludes(self, patterns: List[str]) -> None:
        """Register exclude patterns.  Entries whose IDs suffix-match any
        pattern will be filtered out of query results."""
        self._excludes.extend(patterns)

    def _is_excluded(self, entry_id: str) -> bool:
        """Check if *entry_id* matches any exclude pattern (suffix match)."""
        return is_excluded(entry_id, self._excludes)

    def find_components_by_suffixes(self, suffixes: Sequence[str]) -> List[Dict[str, Any]]:
        """Return catalog entries whose IDs end with any of the provided suffixes.

        Results are drawn from the DuckDB catalog and user custom entries.
        Precedence: DuckDB > custom.  Excluded IDs are filtered out.
        """
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
        db_results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        seen_ids = {row["id"] for row in db_results}

        # Add custom entries (lower precedence than DuckDB)
        for suffix in suffixes:
            for entry_id, entry in self._custom_index.items():
                if entry_id.endswith(suffix) and entry_id not in seen_ids:
                    db_results.append(entry)
                    seen_ids.add(entry_id)

        # Apply exclude filtering
        if self._excludes:
            db_results = [r for r in db_results if not self._is_excluded(r["id"])]

        return db_results
