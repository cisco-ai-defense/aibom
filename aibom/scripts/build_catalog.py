#!/usr/bin/env python3
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

"""Build the DuckDB catalog from the catalog_entries data module.

Usage:
    python scripts/build_catalog.py --output path/to/catalog.duckdb
    python scripts/build_catalog.py --output path/to/catalog.duckdb --merge-existing old.duckdb
    python scripts/build_catalog.py --output path/to/catalog.duckdb --update-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb

# Allow running from the repo root or from within the scripts directory.
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from catalog_entries import get_all_entries


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_catalog(
    output_path: Path,
    merge_existing: Path | None = None,
) -> str:
    """Build a DuckDB catalog file and return its SHA-256 digest."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing output to start fresh
    if output_path.exists():
        output_path.unlink()

    con = duckdb.connect(str(output_path))

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS component_catalog (
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

    # If merging with an existing catalog, copy its entries first
    if merge_existing and merge_existing.exists():
        print(f"Merging existing catalog from {merge_existing}")
        # Use parameterized path to avoid SQL injection from paths with quotes
        safe_path = str(merge_existing).replace("'", "''")
        con.execute(f"ATTACH '{safe_path}' AS old_db (READ_ONLY)")
        con.execute(
            """
            INSERT OR IGNORE INTO component_catalog
            SELECT id, label, concept, framework, sig_name, type, catalog_label
            FROM old_db.component_catalog;
            """
        )
        con.execute("DETACH old_db")

    # Insert all entries from the data module in a single batch
    entries = get_all_entries()
    con.executemany(
        """
        INSERT OR IGNORE INTO component_catalog
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                entry["id"],
                entry.get("label"),
                entry.get("concept"),
                entry.get("framework"),
                entry.get("sig_name"),
                entry.get("type"),
                entry.get("catalog_label"),
            )
            for entry in entries
        ],
    )

    count = con.execute("SELECT COUNT(*) FROM component_catalog").fetchone()[0]
    print(f"Catalog built with {count} entries.")

    con.execute(
        """
        CREATE TABLE component_catalog_last_seg AS
        SELECT id, split_part(id, '.', -1) AS last_seg
        FROM component_catalog;
        """
    )
    print("Built component_catalog_last_seg token table.")

    con.close()

    digest = _sha256(output_path)
    print(f"SHA-256: {digest}")
    return digest


def update_manifest(digest: str, output_path: Path) -> None:
    """Update the manifest.json with the new SHA-256 and file path."""
    manifest_path = _SCRIPTS_DIR.parent / "src" / "aibom" / "manifest.json"
    if not manifest_path.exists():
        print(f"Warning: manifest.json not found at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["duckdb_sha256"] = digest
    # Use the tilde path form for the manifest
    duckdb_filename = output_path.name
    manifest["duckdb_file"] = f"~/.aibom/catalogs/{duckdb_filename}"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"Updated manifest.json: duckdb_sha256={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AIBOM DuckDB catalog")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the output DuckDB file.",
    )
    parser.add_argument(
        "--merge-existing",
        type=Path,
        default=None,
        help="Optional path to an existing DuckDB to merge entries from.",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Update src/aibom/manifest.json with the new SHA-256 and file path.",
    )
    args = parser.parse_args()

    digest = build_catalog(args.output, args.merge_existing)

    if args.update_manifest:
        update_manifest(digest, args.output)


if __name__ == "__main__":
    main()
