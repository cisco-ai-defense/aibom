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

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
from unittest.mock import MagicMock, patch

import duckdb
import httpx
import pytest
import respx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from aibom.kb.manager import DEFAULT_MANIFEST_URL, KBError, KBManager
from aibom.kb.manifest import KBManifest, KBManifestIndex

HELLO_SHA256 = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

_S3_ORIGIN = "https://example-public.s3.us-west-2.amazonaws.com"


def _schema_v2_bundle(tmp_path, *, kb_version="2.0.0", min_cli_version="2.0.0"):
    db_path = tmp_path / "source.duckdb"
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute("CREATE TABLE component_catalog (id INTEGER)")
        connection.execute("INSERT INTO component_catalog VALUES (1)")
    finally:
        connection.close()

    artifact = tmp_path / "aibom_catalog.duckdb.gz"
    with db_path.open("rb") as source, gzip.open(artifact, "wb") as output:
        output.write(source.read())

    private_key = ec.generate_private_key(ec.SECP256R1())
    signature = private_key.sign(artifact.read_bytes(), ec.ECDSA(hashes.SHA256()))
    signature_path = tmp_path / "aibom_catalog.duckdb.gz.sig"
    signature_path.write_bytes(base64.b64encode(signature) + b"\n")
    public_key_path = tmp_path / "cosign.pub"
    public_key_path.write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    prefix = f"{_S3_ORIGIN}/aibom/kb/schema-v2/v{kb_version}"
    manifest = {
        "artifact_state": "authoritative",
        "authoritative": True,
        "schema_version": 2,
        "vocabulary_version": "2.0.0",
        "min_cli_version": min_cli_version,
        "kb_version": kb_version,
        "build_type": "floor",
        "generated_at": "2026-08-01T00:00:00Z",
        "duckdb": {
            "url": f"{prefix}/aibom_catalog.duckdb.gz",
            "size_bytes": artifact.stat().st_size,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        "signature": {
            "algorithm": "ECDSA_SHA_256",
            "url": f"{prefix}/aibom_catalog.duckdb.gz.sig",
            "public_key_url": f"{prefix}/cosign.pub",
            "key_id": "test-signing-key",
        },
        "freshness_api": "https://api.example.com/api/v1/aibom/kb/packages/freshness",
        "has_enrichment": True,
        "contents": {"component_catalog": {"rows": 1}},
        "provenance": {
            "source_commit": "test-source",
            "pipeline_version": "test-pipeline",
            "config_version": "test-config",
            "model_version": "test-model",
            "tools_version": "test-tools",
            "input_fingerprint": "test-fingerprint",
        },
        "source_candidate": {
            "build_id": "test-build",
            "validation_report_url": f"{prefix}/validation-report.json",
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, manifest_path, artifact, signature_path, public_key_path


def test_kb_manifest_model_creation_and_serialization():
    m = KBManifest(
        kb_version="1.0.0",
        min_cli_version="0.5.0",
        duckdb_sha256="abc",
        duckdb_url="https://example.com/kb.duckdb",
        size_bytes=100,
        entity_count=42,
        created_at="2026-01-01T00:00:00Z",
        sdk_versions={"python": "3.12"},
    )
    dumped = m.model_dump()
    assert dumped["kb_version"] == "1.0.0"
    assert dumped["duckdb_sha256"] == "abc"
    roundtrip = KBManifest.model_validate(dumped)
    assert roundtrip.model_dump_json() == m.model_dump_json()

    older = KBManifest(
        kb_version="0.9.0",
        duckdb_sha256="def",
        duckdb_url="https://example.com/old.duckdb",
    )
    idx = KBManifestIndex(latest=m, versions=[older])
    raw = json.loads(idx.model_dump_json())
    restored = KBManifestIndex.model_validate(raw)
    assert restored.latest.kb_version == "1.0.0"
    assert len(restored.versions) == 1
    assert restored.versions[0].kb_version == "0.9.0"


def test_kb_manager_compute_sha256(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"hello")
    mgr = KBManager()
    assert mgr._compute_sha256(p) == HELLO_SHA256


def test_kb_manager_verify_true_when_checksum_matches(tmp_path):
    root = tmp_path
    catalogs = root / "catalogs"
    catalogs.mkdir(parents=True)
    db_path = catalogs / "kb-1.0.0.duckdb"
    db_path.write_bytes(b"kb-data")
    sha = hashlib.sha256(b"kb-data").hexdigest()
    manifest = {
        "kb_version": "1.0.0",
        "duckdb_sha256": sha,
        "duckdb_url": "https://example.com/k.duckdb",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mgr = KBManager()
    with (
        patch.object(mgr, "_user_root", return_value=root),
        patch.object(mgr, "_local_manifest_path", return_value=root / "manifest.json"),
        patch.object(mgr, "_kb_duckdb_path", return_value=db_path),
    ):
        assert mgr.verify() is True


def test_kb_manager_verify_false_when_checksum_mismatches(tmp_path):
    root = tmp_path
    catalogs = root / "catalogs"
    catalogs.mkdir(parents=True)
    db_path = catalogs / "kb-1.0.0.duckdb"
    db_path.write_bytes(b"wrong")
    manifest = {
        "kb_version": "1.0.0",
        "duckdb_sha256": hashlib.sha256(b"kb-data").hexdigest(),
        "duckdb_url": "https://example.com/k.duckdb",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mgr = KBManager()
    with (
        patch.object(mgr, "_user_root", return_value=root),
        patch.object(mgr, "_local_manifest_path", return_value=root / "manifest.json"),
        patch.object(mgr, "_kb_duckdb_path", return_value=db_path),
    ):
        assert mgr.verify() is False


def test_kb_manager_info_returns_expected_keys(tmp_path):
    root = tmp_path
    catalogs = root / "catalogs"
    catalogs.mkdir(parents=True)
    db_path = catalogs / "kb-2.1.0.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE component_catalog (id INTEGER)")
        con.execute("INSERT INTO component_catalog VALUES (1), (2), (3)")
    finally:
        con.close()

    manifest = {
        "kb_version": "2.1.0",
        "duckdb_sha256": "unused",
        "duckdb_url": "https://example.com/k.duckdb",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mgr = KBManager()
    with (
        patch.object(mgr, "_user_root", return_value=root),
        patch.object(mgr, "_local_manifest_path", return_value=root / "manifest.json"),
        patch.object(mgr, "_kb_duckdb_path", return_value=db_path),
    ):
        info = mgr.info()

    assert set(info.keys()) == {
        "version",
        "path",
        "sha256",
        "size_bytes",
        "entity_count",
        "last_modified",
    }
    assert info["version"] == "2.1.0"
    assert info["entity_count"] == 3
    assert info["path"] == str(db_path.resolve())
    assert info["sha256"] == hashlib.sha256(db_path.read_bytes()).hexdigest()
    assert info["size_bytes"] == db_path.stat().st_size


def test_kb_manager_info_surfaces_schema_v2_candidate_offline(tmp_path):
    root = tmp_path
    db_path = root / "aibom_catalog.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE component_catalog (id INTEGER)")
        con.execute("INSERT INTO component_catalog VALUES (1)")
    finally:
        con.close()

    manifest = {
        "artifact_state": "candidate",
        "authoritative": False,
        "schema_version": 2,
        "build_id": "candidate-2026-07-24",
        "min_cli_version": "2.0.0",
        "vocabulary_version": "v2.0",
        "duckdb": {
            "filename": db_path.name,
            "size_bytes": db_path.stat().st_size,
            "sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mgr = KBManager()
    with patch.object(
        mgr,
        "_local_manifest_path",
        return_value=root / "manifest.json",
    ):
        info = mgr.info()

    assert info["version"] == "candidate-2026-07-24"
    assert info["path"] == str(db_path.resolve())
    assert info["schema_version"] == 2
    assert info["vocabulary_version"] == "v2.0"
    assert info["concept_count"] == 20
    assert "document_loader" in info["concepts"]


@respx.mock
def test_kb_manager_check_update_available_when_remote_newer(tmp_path):
    root = tmp_path
    manifest_url = "https://check.test/manifest.json"
    local = KBManifest(
        kb_version="1.0.0",
        duckdb_sha256="a",
        duckdb_url="https://x/k.duckdb",
    )
    (root / "manifest.json").write_text(
        local.model_dump_json(),
        encoding="utf-8",
    )
    remote_index = {
        "latest": {
            "kb_version": "2.0.0",
            "duckdb_sha256": "b",
            "duckdb_url": "https://x/new.duckdb",
        },
        "versions": [],
    }
    respx.get(manifest_url).mock(return_value=httpx.Response(200, json=remote_index))
    mgr = KBManager(manifest_url=manifest_url)
    with patch.object(mgr, "_user_root", return_value=root):
        out = mgr.check()
    assert out["update_available"] is True
    assert out["current_version"] == "1.0.0"
    assert out["latest_version"] == "2.0.0"
    assert out["download_url"] == "https://x/new.duckdb"


@respx.mock
def test_kb_manager_check_no_update_when_same_version(tmp_path):
    root = tmp_path
    manifest_url = "https://check.test/manifest.json"
    local = KBManifest(
        kb_version="2.0.0",
        duckdb_sha256="a",
        duckdb_url="https://x/k.duckdb",
    )
    (root / "manifest.json").write_text(
        local.model_dump_json(),
        encoding="utf-8",
    )
    remote_index = {
        "latest": {
            "kb_version": "2.0.0",
            "duckdb_sha256": "b",
            "duckdb_url": "https://x/k.duckdb",
        },
        "versions": [],
    }
    respx.get(manifest_url).mock(return_value=httpx.Response(200, json=remote_index))
    mgr = KBManager(manifest_url=manifest_url)
    with patch.object(mgr, "_user_root", return_value=root):
        out = mgr.check()
    assert out["update_available"] is False
    assert out["current_version"] == "2.0.0"
    assert out["latest_version"] == "2.0.0"


@respx.mock
def test_kb_manager_download_writes_db_manifest_and_verifies_checksum(tmp_path):
    root = tmp_path
    body = b"duckdb-payload"
    sha = hashlib.sha256(body).hexdigest()
    manifest_url = "https://kb.test/manifest.json"
    duck_url = "https://kb.test/file.duckdb"
    index = {
        "latest": {
            "kb_version": "3.4.5",
            "duckdb_sha256": sha,
            "duckdb_url": duck_url,
            "size_bytes": 0,
        },
        "versions": [],
    }
    respx.get(manifest_url).mock(return_value=httpx.Response(200, json=index))
    respx.get(duck_url).mock(return_value=httpx.Response(200, content=body))

    mgr = KBManager(manifest_url=manifest_url)
    with patch.object(mgr, "_user_root", return_value=root):
        dest = mgr.download()

    assert dest == root / "catalogs" / "kb-3.4.5.duckdb"
    assert dest.read_bytes() == body
    mp = root / "manifest.json"
    saved = KBManifest.model_validate_json(mp.read_text(encoding="utf-8"))
    assert saved.kb_version == "3.4.5"
    assert saved.duckdb_sha256.lower() == sha.lower()


def test_schema_v2_prefetched_install_verifies_signature_and_cache(tmp_path):
    _, manifest_path, artifact, signature, public_key = _schema_v2_bundle(tmp_path)
    user_root = tmp_path / "user"
    manager = KBManager(
        manifest_url=f"{_S3_ORIGIN}/aibom/kb/manifest.json",
        installed_cli_version="2.0.0",
    )
    with patch.object(manager, "_user_root", return_value=user_root):
        installed = manager.install_prefetched(
            manifest_path,
            artifact_path=artifact,
            signature_path=signature,
            public_key_path=public_key,
        )
        assert installed.exists()
        assert manager.verify() is True
        saved = json.loads((user_root / "manifest.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["build_type"] == "floor"


def test_schema_v2_rejects_tampered_signature_and_preserves_last_good(tmp_path):
    _, manifest_path, artifact, signature, public_key = _schema_v2_bundle(tmp_path)
    user_root = tmp_path / "user"
    manager = KBManager(
        manifest_url=f"{_S3_ORIGIN}/aibom/kb/manifest.json",
        installed_cli_version="2.0.0",
    )
    with patch.object(manager, "_user_root", return_value=user_root):
        installed = manager.install_prefetched(
            manifest_path,
            artifact_path=artifact,
            signature_path=signature,
            public_key_path=public_key,
        )
        before = installed.read_bytes()
        signature.write_bytes(base64.b64encode(b"not-a-valid-signature"))
        with pytest.raises(KBError, match="signature verification failed"):
            manager.install_prefetched(
                manifest_path,
                artifact_path=artifact,
                signature_path=signature,
                public_key_path=public_key,
            )
        assert installed.read_bytes() == before
        assert manager.verify() is True


@respx.mock
def test_schema_v2_rejects_min_cli_version_before_artifact_download(tmp_path):
    manifest, _, _, _, _ = _schema_v2_bundle(
        tmp_path,
        min_cli_version="2.1.0",
    )
    manifest_url = f"{_S3_ORIGIN}/aibom/kb/manifest.json"
    artifact_route = respx.get(manifest["duckdb"]["url"]).mock(
        return_value=httpx.Response(200, content=b"must-not-download")
    )
    respx.get(manifest_url).mock(return_value=httpx.Response(200, json=manifest))
    manager = KBManager(
        manifest_url=manifest_url,
        installed_cli_version="2.0.0",
    )
    with patch.object(manager, "_user_root", return_value=tmp_path / "user"):
        with pytest.raises(KBError, match="installed 2.0.0, required 2.1.0"):
            manager.download()
    assert artifact_route.called is False


@respx.mock
def test_schema_v2_rejects_non_s3_object_url_before_download(tmp_path):
    manifest, _, _, _, _ = _schema_v2_bundle(tmp_path)
    manifest["duckdb"]["url"] = "https://downloads.example.com/aibom_catalog.duckdb.gz"
    manifest_url = f"{_S3_ORIGIN}/aibom/kb/manifest.json"
    respx.get(manifest_url).mock(return_value=httpx.Response(200, json=manifest))
    manager = KBManager(
        manifest_url=manifest_url,
        installed_cli_version="2.0.0",
    )
    with pytest.raises(KBError, match="direct regional S3 HTTPS URL"):
        manager.download()


def test_schema_v2_manifest_rejects_unknown_fields(tmp_path):
    manifest, manifest_path, artifact, signature, public_key = _schema_v2_bundle(
        tmp_path
    )
    manifest["unexpected"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manager = KBManager(
        manifest_url=f"{_S3_ORIGIN}/aibom/kb/manifest.json",
        installed_cli_version="2.0.0",
    )
    with pytest.raises(KBError, match="extra_forbidden"):
        manager.install_prefetched(
            manifest_path,
            artifact_path=artifact,
            signature_path=signature,
            public_key_path=public_key,
        )


def test_schema_v2_manifest_rejects_coerced_field_types(tmp_path):
    manifest, manifest_path, artifact, signature, public_key = _schema_v2_bundle(
        tmp_path
    )
    manifest["schema_version"] = "2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manager = KBManager(
        manifest_url=f"{_S3_ORIGIN}/aibom/kb/manifest.json",
        installed_cli_version="2.0.0",
    )
    with pytest.raises(KBError, match="valid integer"):
        manager.install_prefetched(
            manifest_path,
            artifact_path=artifact,
            signature_path=signature,
            public_key_path=public_key,
        )


def test_manifest_selection_precedence_is_explicit_then_pin_then_pointer(tmp_path):
    pointer = f"{_S3_ORIGIN}/aibom/kb/manifest.json"
    manager = KBManager(manifest_url=pointer, installed_cli_version="2.0.0")
    config_path = tmp_path / "kb.json"
    explicit_url = f"{_S3_ORIGIN}/aibom/kb/schema-v2/v2.8.0/manifest.json"
    with patch.object(manager, "_config_path", return_value=config_path):
        manager.set_pin("2.4.0")
        assert manager._selected_manifest_url(version=None, url=None) == (
            f"{_S3_ORIGIN}/aibom/kb/schema-v2/v2.4.0/manifest.json"
        )
        assert manager._selected_manifest_url(version="2.6.0", url=None) == (
            f"{_S3_ORIGIN}/aibom/kb/schema-v2/v2.6.0/manifest.json"
        )
        assert manager._selected_manifest_url(version=None, url=explicit_url) == (
            explicit_url
        )


def test_unsigned_rehearsal_requires_explicit_override(tmp_path):
    manifest, manifest_path, artifact, _, _ = _schema_v2_bundle(tmp_path)
    manifest["artifact_state"] = "rehearsal"
    manifest["authoritative"] = False
    manifest["signature"]["algorithm"] = "disabled"
    manifest["signature"]["public_key_url"] = ""
    manifest["signature"]["key_id"] = "disabled"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    user_root = tmp_path / "user"
    manager = KBManager(
        manifest_url=f"{_S3_ORIGIN}/aibom/kb/manifest.json",
        installed_cli_version="2.0.0",
    )
    with patch.object(manager, "_user_root", return_value=user_root):
        with pytest.raises(KBError, match="Unsigned KB artifacts are disabled"):
            manager.install_prefetched(manifest_path, artifact_path=artifact)
        installed = manager.install_prefetched(
            manifest_path,
            artifact_path=artifact,
            allow_unsigned=True,
        )
    assert installed.exists()


def test_schema_v2_rejects_checksum_mismatch(tmp_path):
    manifest, manifest_path, artifact, signature, public_key = _schema_v2_bundle(
        tmp_path
    )
    manifest["duckdb"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manager = KBManager(
        manifest_url=f"{_S3_ORIGIN}/aibom/kb/manifest.json",
        installed_cli_version="2.0.0",
    )
    with pytest.raises(KBError, match="checksum mismatch"):
        manager.install_prefetched(
            manifest_path,
            artifact_path=artifact,
            signature_path=signature,
            public_key_path=public_key,
        )


def test_schema_v2_rejects_wrong_public_key(tmp_path):
    _, manifest_path, artifact, signature, public_key = _schema_v2_bundle(tmp_path)
    wrong_key = ec.generate_private_key(ec.SECP256R1()).public_key()
    public_key.write_bytes(
        wrong_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    manager = KBManager(
        manifest_url=f"{_S3_ORIGIN}/aibom/kb/manifest.json",
        installed_cli_version="2.0.0",
    )
    with pytest.raises(KBError, match="signature verification failed"):
        manager.install_prefetched(
            manifest_path,
            artifact_path=artifact,
            signature_path=signature,
            public_key_path=public_key,
        )


def test_kb_manager_request_build_calls_endpoint_and_payload():
    mgr = KBManager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "request": {"request_id": "req-1", "state": "queued"},
        "coalesced": False,
        "quota_remaining": 9,
    }
    mock_resp.is_error = False
    mock_resp.headers = {
        "x-ratelimit-limit": "10",
        "x-ratelimit-remaining": "9",
        "x-ratelimit-reset": "1234",
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("aibom.kb.manager.httpx.Client", return_value=mock_client):
        out = mgr.request_build(
            "pypi",
            "openai",
            ["openai.OpenAI"],
            api_key="k",
            api_base="https://api.example",
        )

    expected_url = "https://api.example/api/v1/aibom/kb/requests"
    mock_client.post.assert_called_once()
    call_kw = mock_client.post.call_args
    assert call_kw[0][0] == expected_url
    assert call_kw[1]["json"] == {
        "ecosystem": "pypi",
        "package_name": "openai",
        "symbols": ["openai.OpenAI"],
    }
    headers = call_kw[1]["headers"]
    assert headers["x-cisco-ai-defense-tenant-api-key"] == "k"
    assert headers["Content-Type"] == "application/json"
    assert out["request"]["request_id"] == "req-1"
    assert out["rate_limit"] == {
        "limit": "10",
        "remaining": "9",
        "reset": "1234",
    }


def test_kb_manager_request_build_missing_api_key_raises():
    mgr = KBManager()
    env = {k: v for k, v in os.environ.items() if k != "CISCO_AI_DEFENSE_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(KBError, match="API key required"):
            mgr.request_build(
                "pypi",
                "x",
                ["x.Client"],
                api_key=None,
                api_base="https://api.example",
            )


def test_kb_manager_request_build_missing_api_base_raises():
    mgr = KBManager()
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CISCO_AI_DEFENSE_API_BASE", "CISCO_AI_DEFENSE_API_KEY"}
    }
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(KBError, match="API base URL required"):
            mgr.request_build("pypi", "x", ["x.Client"], api_key="k")


def test_kb_manager_download_uses_schema_v2_default_manifest():
    env = {k: v for k, v in os.environ.items() if k != "CISCO_AIBOM_MANIFEST_URL"}
    with patch.dict(os.environ, env, clear=True):
        mgr = KBManager()
        assert (
            mgr._selected_manifest_url(version=None, url=None) == DEFAULT_MANIFEST_URL
        )


def test_kb_manager_check_uses_schema_v2_default_manifest():
    env = {k: v for k, v in os.environ.items() if k != "CISCO_AIBOM_MANIFEST_URL"}
    with patch.dict(os.environ, env, clear=True):
        mgr = KBManager()
        assert (
            mgr._selected_manifest_url(version=None, url=None) == DEFAULT_MANIFEST_URL
        )


def test_kb_manager_request_status_calls_get_endpoint():
    mgr = KBManager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "request_id": "rid",
        "state": "available",
        "requests": [],
    }
    mock_resp.is_error = False
    mock_resp.headers = {}
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("aibom.kb.manager.httpx.Client", return_value=mock_client):
        out = mgr.request_status(
            "rid",
            api_key="k",
            api_base="https://api.example",
        )

    expected = "https://api.example/api/v1/aibom/kb/requests/rid"
    mock_client.get.assert_called_once_with(
        expected,
        headers={"x-cisco-ai-defense-tenant-api-key": "k"},
    )
    assert out["request_id"] == "rid"
    assert out["state"] == "available"


def test_kb_manager_list_requests_returns_list():
    mgr = KBManager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "requests": [
            {"id": "a", "status": "x"},
            {"id": "b"},
        ]
    }
    mock_resp.is_error = False
    mock_resp.headers = {}
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("aibom.kb.manager.httpx.Client", return_value=mock_client):
        items = mgr.list_requests(api_key="k", api_base="https://api.example")

    expected = "https://api.example/api/v1/aibom/kb/requests"
    mock_client.get.assert_called_once_with(
        expected,
        headers={"x-cisco-ai-defense-tenant-api-key": "k"},
        params={"limit": 50},
    )
    assert items == [
        {"id": "a", "status": "x"},
        {"id": "b"},
    ]


def test_kb_manager_list_requests_accepts_plain_list():
    mgr = KBManager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"r": 1}]
    mock_resp.is_error = False
    mock_resp.headers = {}
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("aibom.kb.manager.httpx.Client", return_value=mock_client):
        items = mgr.list_requests(api_key="k", api_base="https://api.example")
    assert items == [{"r": 1}]


def test_kb_error_is_exception_subclass():
    assert issubclass(KBError, Exception)
    e = KBError("msg")
    assert str(e) == "msg"
