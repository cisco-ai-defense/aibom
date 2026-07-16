# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""DuckDB catalog access for the prebuilt knowledge base.

Uses an in-memory DuckDB connection with the catalog file ATTACHed as
read-only.  A last-segment token table enables O(S+K) hash-join lookups
instead of the previous O(S*N) ``LIKE '%suffix'`` full-scan pattern.

Precedence: DuckDB > custom.  Excluded IDs are filtered out.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import duckdb

LOGGER = logging.getLogger(__name__)

_CATALOG_COLS = "id, label, concept, framework, sig_name, type, catalog_label"


def is_excluded(name: str, patterns: List[str]) -> bool:
    """Return True if *name* suffix-matches or equals any pattern in *patterns*."""
    for pattern in patterns:
        if name.endswith(pattern) or name == pattern:
            return True
    return False


class CatalogDB:
    """Provides read-only access to the DuckDB component catalog,
    augmented with user-defined custom entries from ``.aibom.yaml``.

    The catalog file is ATTACHed read-only to an in-memory DuckDB
    connection so that temporary tables (for the last-segment token
    index) can be created without modifying the catalog file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(f"DuckDB catalog not found at {self._db_path}")

        self._connection = duckdb.connect()
        safe_path = str(self._db_path).replace("'", "''")
        self._connection.execute(f"ATTACH '{safe_path}' AS kb (READ_ONLY)")

        self._connection.execute(
            f"CREATE VIEW component_catalog AS SELECT * FROM kb.component_catalog"
        )

        self._custom_index: Dict[str, Dict[str, Any]] = {}
        self._excludes: List[str] = []
        self._token_table: str | None = None

    def __enter__(self) -> "CatalogDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        """Close the DuckDB connection (idempotent)."""
        if self._connection is not None:
            try:
                self._connection.execute("DETACH kb")
            except Exception:  # noqa: BLE001
                pass
            self._connection.close()
            self._connection = None

    # ------------------------------------------------------------------
    # Token table management
    # ------------------------------------------------------------------

    def _ensure_token_table(self) -> str:
        """Lazily initialise the last-segment token table.

        If the catalog already ships a pre-built
        ``component_catalog_last_seg`` table (new KBs), use it directly but
        join against ``kb.component_catalog`` so stale ``label='parameter'``
        rows (only present in pre-M1 KBs) are ignored and never pollute the
        suffix index.  Otherwise build a temporary table from
        ``kb.component_catalog`` with the same guard applied.

        Returns the qualified table name to use in queries.
        """
        if self._token_table is not None:
            return self._token_table

        label_filter_sql = self._ignore_stale_parameter_sql("cc")
        build_filter_sql = self._ignore_stale_parameter_sql()

        try:
            self._connection.execute(
                "SELECT 1 FROM kb.component_catalog_last_seg LIMIT 1"
            )
            LOGGER.debug(
                "Pre-built token table found; ignoring stale parameter rows in temp view"
            )
            self._connection.execute(
                "CREATE TEMP TABLE _last_seg AS "
                "SELECT ls.id, ls.last_seg "
                "FROM kb.component_catalog_last_seg ls "
                "JOIN kb.component_catalog cc USING (id) "
                f"WHERE {label_filter_sql}"
            )
            self._token_table = "_last_seg"
        except duckdb.CatalogException:
            LOGGER.debug("Building temp last-segment token table")
            self._connection.execute(
                "CREATE TEMP TABLE _last_seg AS "
                "SELECT id, split_part(id, '.', -1) AS last_seg "
                "FROM kb.component_catalog "
                f"WHERE {build_filter_sql}"
            )
            self._token_table = "_last_seg"

        return self._token_table

    @staticmethod
    def _ignore_stale_parameter_sql(alias: str = "") -> str:
        """SQL fragment that ignores stale ``label='parameter'`` rows.

        The server-side KB pipeline now strips parameter rows at build time, so
        a current KB never contains them and this guard is a no-op there. It is
        retained purely as a defensive fallback: if a stale KB that still
        carries parameter rows is loaded, those rows are silently ignored so
        scan output matches a current KB. Parameter rows are never top-level AI
        components and were a source of suffix-match false positives.
        """
        col = f"{alias}.label" if alias else "label"
        return f"COALESCE({col}, '') <> 'parameter'"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def distinct_frameworks(self) -> List[str]:
        """Return the union of framework values from DuckDB and custom entries.

        Used by KB enrichment helpers to build framework-prefix allowlists
        that recognise custom/built-in catalog supplements (e.g. Strands)
        alongside the prebuilt DuckDB snapshot.
        """
        names: set[str] = set()
        try:
            rows = self._connection.execute(
                "SELECT DISTINCT framework FROM component_catalog "
                "WHERE framework IS NOT NULL"
            ).fetchall()
            names.update(fw for (fw,) in rows if fw)
        except Exception:  # noqa: BLE001
            LOGGER.debug("distinct_frameworks DuckDB query failed", exc_info=True)

        for entry in self._custom_index.values():
            fw = entry.get("framework")
            if fw:
                names.add(fw)
        return sorted(names)

    def find_ids_in_custom_by_path_and_concept(
        self, path_segment: str, concepts: Sequence[str]
    ) -> List[str]:
        """Scan the in-memory custom index for entries whose ID contains
        *path_segment* and whose concept is in *concepts*.

        Mirrors :meth:`find_ids_by_path_and_concept` but over the custom
        index only, so pattern discovery also picks up built-in
        supplements (e.g. Strands tool/model class names).
        """
        if not concepts:
            return []
        concept_lower = {c.lower() for c in concepts}
        results: list[str] = []
        for entry_id, entry in self._custom_index.items():
            concept = (entry.get("concept") or "").lower()
            if concept in concept_lower and path_segment in entry_id:
                results.append(entry_id)
        return results

    def find_ids_by_path_and_concept(
        self, path_segment: str, concepts: Sequence[str]
    ) -> List[str]:
        """Return distinct entry IDs whose path contains *path_segment*
        and whose concept is in *concepts*."""
        if not concepts:
            return []
        placeholders = ",".join("?" for _ in concepts)
        label_filter = self._ignore_stale_parameter_sql()
        query = f"""
            SELECT DISTINCT id FROM component_catalog
            WHERE id LIKE ? AND LOWER(concept) IN ({placeholders})
              AND {label_filter}
        """
        params: list[str] = [f"%{path_segment}%"] + [c.lower() for c in concepts]
        rows = self._connection.execute(query, params).fetchall()
        return [r[0] for r in rows]

    def find_components_by_suffixes(
        self,
        suffixes: Sequence[str],
    ) -> List[Dict[str, Any]]:
        """Return catalog entries whose IDs match any of the provided suffixes.

        Uses a two-tier strategy for O(S+K) performance:

        1. **Full-path suffixes** (contain a dot): exact ``id IN (...)`` match.
        2. **Single-token suffixes** (no dot): ``last_seg IN (...)`` hash join
           against the pre-built token table.

        Results are drawn from the DuckDB catalog and user custom entries.
        Precedence: DuckDB > custom.  Excluded IDs are filtered out.
        """
        if not suffixes:
            return []

        token_tbl = self._ensure_token_table()

        full_paths: list[str] = []
        tokens: set[str] = set()
        for s in suffixes:
            if "." in s:
                full_paths.append(s)
                tokens.add(s.rsplit(".", 1)[-1])
            else:
                tokens.add(s)

        matched_ids: set[str] = set()
        label_filter = self._ignore_stale_parameter_sql()

        if full_paths:
            rows = self._connection.execute(
                "SELECT id FROM kb.component_catalog "
                "WHERE id IN (SELECT UNNEST(?)) "
                f"AND {label_filter}",
                [full_paths],
            ).fetchall()
            matched_ids.update(r[0] for r in rows)

        if tokens:
            token_list = list(tokens)
            rows = self._connection.execute(
                f"SELECT DISTINCT id FROM {token_tbl} "
                "WHERE last_seg IN (SELECT UNNEST(?))",
                [token_list],
            ).fetchall()
            matched_ids.update(r[0] for r in rows)

        if matched_ids:
            id_list = list(matched_ids)
            cursor = self._connection.execute(
                f"SELECT {_CATALOG_COLS} FROM kb.component_catalog "
                "WHERE id IN (SELECT UNNEST(?)) "
                f"AND {label_filter}",
                [id_list],
            )
            columns = [desc[0] for desc in cursor.description]
            db_results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        else:
            db_results = []

        seen_ids = {row["id"] for row in db_results}

        for s in suffixes:
            for entry_id, entry in self._custom_index.items():
                if entry_id not in seen_ids and self._custom_entry_matches_suffix(
                    entry_id, s
                ):
                    db_results.append(entry)
                    seen_ids.add(entry_id)

        if self._excludes:
            db_results = [r for r in db_results if not self._is_excluded(r["id"])]

        return db_results

    @staticmethod
    def _custom_entry_matches_suffix(entry_id: str, suffix: str) -> bool:
        """Mirror the two-tier SQL matching for custom (in-memory) entries.

        * Full-path suffixes (contain a dot) must match as an ``endswith``
          exact path suffix.
        * Single-token suffixes match only the last segment of the ID.
        This mirrors ``find_components_by_suffixes``' DuckDB behaviour so
        custom entries do not get picked up by unrelated suffixes.
        """
        if "." in suffix:
            return entry_id == suffix or entry_id.endswith("." + suffix)
        last_seg = entry_id.rsplit(".", 1)[-1]
        return last_seg == suffix
