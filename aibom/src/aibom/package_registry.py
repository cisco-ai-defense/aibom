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

"""Read package liveness snapshots from the selected DuckDB catalog."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import platformdirs

from .db_loader import _load_manifest, _resolve_db_path


@dataclass(frozen=True, order=True)
class PackageCoordinate:
    """A normalized package identity."""

    ecosystem: str
    name: str


@dataclass(frozen=True)
class PackageSnapshot:
    """Liveness fields frozen into one knowledge-base build."""

    coordinate: PackageCoordinate
    liveness_status: str | None
    liveness_snapshot_at: str | None
    certification: dict[str, Any] | None


def normalize_package_name(name: str, ecosystem: str) -> str:
    """Normalize a package name using ecosystem comparison rules."""

    normalized = name.strip()
    normalized_ecosystem = ecosystem.strip().lower()
    if normalized_ecosystem in {"pypi", "npm", "cargo", "rubygems", "nuget"}:
        normalized = normalized.lower()
    if normalized_ecosystem == "pypi":
        normalized = re.sub(r"[-_.]+", "-", normalized)
    return normalized


def package_coordinate(ecosystem: str, name: str) -> PackageCoordinate:
    """Construct a normalized package coordinate."""

    normalized_ecosystem = ecosystem.strip().lower()
    return PackageCoordinate(
        ecosystem=normalized_ecosystem,
        name=normalize_package_name(name, normalized_ecosystem),
    )


def resolve_package_catalog_path() -> Path | None:
    """Resolve the selected catalog without requiring schema-v2 loading."""

    env_path = os.environ.get("AIBOM_DB_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    manifest, manifest_path = _load_manifest()
    if not manifest:
        return None

    db_file = manifest.get("duckdb_file")
    if isinstance(db_file, str) and db_file:
        return _resolve_db_path(db_file, False, manifest_path)

    duckdb_entry = manifest.get("duckdb")
    if isinstance(duckdb_entry, dict):
        filename = duckdb_entry.get("filename")
        if isinstance(filename, str) and filename and manifest_path:
            return (manifest_path.parent / filename).resolve()

    kb_version = manifest.get("kb_version")
    if isinstance(kb_version, str) and kb_version:
        user_root = Path(platformdirs.user_data_dir("aibom"))
        return user_root / "catalogs" / f"kb-{kb_version}.duckdb"
    return None


def read_package_snapshots(
    db_path: Path | str | None,
    coordinates: Iterable[PackageCoordinate],
) -> dict[PackageCoordinate, PackageSnapshot]:
    """Read only the package coordinates already present in the user's BOM."""

    requested = sorted(set(coordinates))
    if not requested or db_path is None:
        return {}
    path = Path(db_path).expanduser()
    if not path.is_file():
        return {}

    try:
        connection = duckdb.connect(str(path), read_only=True)
    except duckdb.Error:
        return {}

    snapshots: dict[PackageCoordinate, PackageSnapshot] = {}
    try:
        tables = {
            str(row[0])
            for row in connection.execute("SHOW TABLES").fetchall()
        }
        if "package_catalog" not in tables:
            return {}
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info('package_catalog')"
            ).fetchall()
        }
        required = {
            "package_name",
            "ecosystem",
            "liveness_status",
            "liveness_snapshot_at",
        }
        if not required.issubset(columns):
            return {}
        certification_sql = (
            "liveness_certification"
            if "liveness_certification" in columns
            else "NULL"
        )
        connection.execute(
            """
            CREATE TEMP TABLE requested_packages (
                ecosystem VARCHAR,
                package_name VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO requested_packages VALUES (?, ?)",
            [
                (coordinate.ecosystem, coordinate.name.lower())
                for coordinate in requested
            ],
        )
        rows = connection.execute(
            f"""
            SELECT requested.ecosystem, requested.package_name,
                   catalog.package_name, catalog.ecosystem,
                   catalog.liveness_status, catalog.liveness_snapshot_at,
                   {certification_sql}
              FROM requested_packages AS requested
              JOIN package_catalog AS catalog
                ON lower(catalog.ecosystem) = requested.ecosystem
               AND lower(catalog.package_name) = requested.package_name
            """
        ).fetchall()
        for row in rows:
            coordinate = package_coordinate(str(row[0]), str(row[1]))
            if coordinate in snapshots:
                continue
            matched = package_coordinate(str(row[3]), str(row[2]))
            snapshots[coordinate] = PackageSnapshot(
                coordinate=matched,
                liveness_status=str(row[4]) if row[4] is not None else None,
                liveness_snapshot_at=_to_iso8601(row[5]),
                certification=_json_object(row[6]),
            )
    except duckdb.Error:
        return {}
    finally:
        connection.close()
    return snapshots


def _to_iso8601(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return (
            timestamp.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if isinstance(value, date):
        timestamp = datetime.combine(value, datetime.min.time(), timezone.utc)
        return timestamp.isoformat().replace("+00:00", "Z")
    text = str(value).strip()
    return text or None


def _json_object(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
