# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from aibom import db_loader


def _write_dummy_db(path: Path, content: bytes = b"duckdb") -> str:
    path.write_bytes(content)
    digest = hashlib.sha256()
    digest.update(content)
    return digest.hexdigest()


def test_ensure_local_database_uses_env_path_and_verifies_checksum(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "aibom_catalog-1.0.0.duckdb"
        checksum = _write_dummy_db(db_path)

        monkeypatch.setenv("AIBOM_DB_PATH", str(db_path))
        monkeypatch.setenv("AIBOM_DB_SHA256", checksum)

        resolved = db_loader.ensure_local_database(console=None)
        assert resolved == db_path


def test_ensure_local_database_uses_manifest_local_path(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        db_path = root / "aibom_catalog-1.0.0.duckdb"
        checksum = _write_dummy_db(db_path)

        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            f"""{{
  "duckdb_sha256": "{checksum}",
  "duckdb_file": "{db_path.name}"
}}""",
            encoding="utf-8",
        )

        monkeypatch.setenv("AIBOM_MANIFEST_PATH", str(manifest_path))

        resolved = db_loader.ensure_local_database(console=None)
        assert resolved == db_path.resolve()


def test_ensure_local_database_requires_path(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        manifest_path.write_text(
            """{
  "duckdb_sha256": "abcdef"
}""",
            encoding="utf-8",
        )
        monkeypatch.setenv("AIBOM_MANIFEST_PATH", str(manifest_path))

        with pytest.raises(db_loader.DatabaseLoadError, match="AIBOM_DB_PATH/duckdb_file"):
            db_loader.ensure_local_database(console=None)


def test_ensure_local_database_checksum_mismatch_triggers_error(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "aibom_catalog-1.0.0.duckdb"
        db_path.write_bytes(b"wrong-data")

        monkeypatch.setenv("AIBOM_DB_PATH", str(db_path))
        monkeypatch.setenv("AIBOM_DB_SHA256", "abcdef")

        with pytest.raises(db_loader.DatabaseLoadError):
            db_loader.ensure_local_database(console=None)


def test_ensure_local_database_uses_manifest_checksum_when_env_sha_missing(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        db_path = root / "aibom_catalog-1.0.0.duckdb"
        checksum = _write_dummy_db(db_path)

        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            f"""{{
  "duckdb_sha256": "{checksum}",
  "duckdb_file": "~/.aibom/catalogs/aibom_catalog-1.0.0.duckdb"
}}""",
            encoding="utf-8",
        )

        monkeypatch.setenv("AIBOM_MANIFEST_PATH", str(manifest_path))
        monkeypatch.setenv("AIBOM_DB_PATH", str(db_path))
        monkeypatch.delenv("AIBOM_DB_SHA256", raising=False)

        resolved = db_loader.ensure_local_database(console=None)
        assert resolved == db_path


def test_schema_v2_manifest_requires_cli_upgrade(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """{
  "schema_version": 2,
  "duckdb_sha256": "unused",
  "duckdb_file": "candidate.duckdb"
}""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AIBOM_MANIFEST_PATH", str(manifest_path))

    with pytest.raises(
        db_loader.UnsupportedDatabaseSchemaError,
        match=r"requires cisco-aibom 2\.x",
    ):
        db_loader.ensure_local_database(console=None)


@pytest.mark.parametrize("schema_version", ["1.0.9", 1])
def test_schema_v1_manifest_remains_supported(
    monkeypatch,
    tmp_path,
    schema_version,
):
    db_path = tmp_path / "frozen-v1.duckdb"
    checksum = _write_dummy_db(db_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        f"""{{
  "schema_version": {json.dumps(schema_version)},
  "duckdb_sha256": "{checksum}",
  "duckdb_file": "{db_path.name}"
}}""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AIBOM_MANIFEST_PATH", str(manifest_path))

    assert db_loader.ensure_local_database(console=None) == db_path.resolve()


def test_default_manifest_path_prefers_packaged_over_cwd(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cwd = root / "cwd"
        cwd.mkdir()
        (cwd / "manifest.json").write_text('{"duckdb_sha256":"from-cwd"}', encoding="utf-8")

        package_root = root / "site-packages" / "aibom"
        package_root.mkdir(parents=True)
        packaged_manifest = package_root / "manifest.json"
        packaged_manifest.write_text(
            '{"duckdb_sha256":"from-package"}', encoding="utf-8"
        )

        monkeypatch.chdir(cwd)
        monkeypatch.setattr(
            db_loader.importlib_resources, "files", lambda _: package_root
        )

        resolved = db_loader._default_manifest_path()
        assert resolved == packaged_manifest
