# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""DuckDB catalog access for the prebuilt knowledge base."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import duckdb

from .supplemental_catalog import SUPPLEMENTAL_ENTRIES

LOGGER = logging.getLogger(__name__)


class CatalogDB:
    """Provides read-only access to the DuckDB component catalog,
    augmented with supplemental entries for frameworks not yet in the
    prebuilt artifact (LangGraph, CrewAI, etc.) and user-defined custom
    entries from ``.aibom.yaml``."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(f"DuckDB catalog not found at {self._db_path}")
        try:
            self._connection = duckdb.connect(str(self._db_path), read_only=True)
        except TypeError:
            self._connection = duckdb.connect(str(self._db_path))
        self._supplemental_index: Dict[str, Dict[str, Any]] = {
            entry["id"]: entry for entry in SUPPLEMENTAL_ENTRIES
        }
        self._custom_index: Dict[str, Dict[str, Any]] = {}
        self._excludes: List[str] = []

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._connection:
            self._connection.close()

    def add_custom_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Merge user-provided custom catalog entries.

        Custom entries have the lowest precedence: DuckDB > supplemental > custom.
        An entry is only added if its ``id`` is not already present in
        the supplemental catalog.
        """
        for entry in entries:
            entry_id = entry.get("id")
            if not entry_id:
                continue
            if entry_id not in self._supplemental_index:
                self._custom_index[entry_id] = entry
                LOGGER.debug("Added custom catalog entry: %s", entry_id)

    def add_excludes(self, patterns: List[str]) -> None:
        """Register exclude patterns.  Entries whose IDs suffix-match any
        pattern will be filtered out of query results."""
        self._excludes.extend(patterns)

    def _is_excluded(self, entry_id: str) -> bool:
        """Check if *entry_id* matches any exclude pattern (suffix match)."""
        for pattern in self._excludes:
            if entry_id.endswith(pattern) or entry_id == pattern:
                return True
        return False

    def find_components_by_suffixes(self, suffixes: Sequence[str]) -> List[Dict[str, Any]]:
        """Return catalog entries whose IDs end with any of the provided suffixes.

        Results are drawn from the DuckDB catalog, the in-memory
        supplemental catalog, and user custom entries.  Precedence:
        DuckDB > supplemental > custom.  Excluded IDs are filtered out.
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

        # Add supplemental entries (lower precedence than DuckDB)
        for suffix in suffixes:
            for entry_id, entry in self._supplemental_index.items():
                if entry_id.endswith(suffix) and entry_id not in seen_ids:
                    db_results.append(entry)
                    seen_ids.add(entry_id)

        # Add custom entries (lowest precedence)
        for suffix in suffixes:
            for entry_id, entry in self._custom_index.items():
                if entry_id.endswith(suffix) and entry_id not in seen_ids:
                    db_results.append(entry)
                    seen_ids.add(entry_id)

        # Apply exclude filtering
        if self._excludes:
            db_results = [r for r in db_results if not self._is_excluded(r["id"])]

        return db_results
