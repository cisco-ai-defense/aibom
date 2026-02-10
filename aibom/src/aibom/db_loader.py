# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

"""Resolve and verify a local DuckDB knowledge base file."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from importlib import resources as importlib_resources
from rich.console import Console

_LOGGER = logging.getLogger(__name__)

BUNDLED_DB_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"


class DatabaseLoadError(RuntimeError):
    """Raised when the DuckDB knowledge base cannot be loaded or verified."""


def ensure_local_database(console: Optional[Console] = None) -> Path:
    """Resolve and verify the configured local DuckDB knowledge base.

    Args:
        console: Unused placeholder kept for API compatibility.

    Returns:
        Absolute path to the local DuckDB file.
    """
    del console

    manifest, manifest_path = _load_manifest()
    env_db_path = os.environ.get("AIBOM_DB_PATH")
    has_env_db_path = bool(env_db_path)
    env_db_sha = os.environ.get("AIBOM_DB_SHA256")

    db_sha = env_db_sha or (manifest or {}).get("duckdb_sha256") or BUNDLED_DB_SHA256
    if not db_sha or set(db_sha) == {"0"}:
        raise DatabaseLoadError(
            "duckdb_sha256 is missing. Provide a checksum via manifest.json or AIBOM_DB_SHA256."
        )
    db_path_value = env_db_path if has_env_db_path else (manifest or {}).get("duckdb_file")
    if not db_path_value:
        raise DatabaseLoadError(
            "AIBOM_DB_PATH/duckdb_file is not set in env or manifest.json. Configure the local DuckDB file path before running the analyzer."
        )
    db_path = _resolve_db_path(db_path_value, has_env_db_path, manifest_path)
    if not db_path.exists():
        raise DatabaseLoadError(
            f"DuckDB catalog file not found at {db_path}. Set AIBOM_DB_PATH or update duckdb_file in manifest.json."
        )
    if not _verify_checksum(db_path, db_sha):
        raise DatabaseLoadError(
            "Local DuckDB checksum mismatch. Verify AIBOM_DB_SHA256."
        )

    _LOGGER.info("Using DuckDB knowledge base at %s", db_path)
    return db_path


def _resolve_db_path(
    db_path_value: str, is_env_override: bool, manifest_path: Optional[Path]
) -> Path:
    """Resolve local DB path relative to CWD or manifest location."""
    candidate = Path(db_path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    if is_env_override:
        return (Path.cwd() / candidate).resolve()
    if manifest_path:
        return (manifest_path.parent / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def _verify_checksum(path: Path, expected_sha: str) -> bool:
    """Verify file checksum matches expected SHA-256."""
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    computed = digest.hexdigest()
    matches = computed.lower() == expected_sha.lower()
    if not matches:
        _LOGGER.error(
            "Checksum mismatch for %s; expected %s, got %s",
            path,
            expected_sha,
            computed,
        )
    return matches


def _load_manifest() -> Tuple[Optional[dict], Optional[Path]]:
    """Load manifest.json if present, returning parsed manifest and source path."""
    path: Optional[Path] = None

    env_manifest = os.environ.get("AIBOM_MANIFEST_PATH")
    if env_manifest:
        path = Path(env_manifest).expanduser()
    else:
        path = _default_manifest_path()

    if not path or not path.exists():
        return None, None

    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj), path
    except Exception as exc:  # noqa: BLE001
        _LOGGER.debug("Failed to load manifest.json from %s: %s", path, exc)
        return None, path


def _default_manifest_path() -> Optional[Path]:
    """Resolve a default manifest, preferring packaged configuration."""

    try:
        pkg_root = importlib_resources.files(__package__.split(".")[0])
        pkg_root_path = Path(str(pkg_root))
        packaged = pkg_root_path / "manifest.json"
        if packaged.exists():
            return packaged
    except Exception:
        pass

    cwd_manifest = Path.cwd() / "manifest.json"
    if cwd_manifest.exists():
        return cwd_manifest

    module_manifest = Path(__file__).resolve().parent / "manifest.json"
    if module_manifest.exists():
        return module_manifest

    return None
