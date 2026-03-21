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

import hashlib
import json
import os
from unittest.mock import MagicMock, patch

import duckdb
import httpx
import pytest
import respx

from aibom.kb.manager import DEFAULT_API_BASE, KBError, KBManager
from aibom.kb.manifest import KBManifest, KBManifestIndex

HELLO_SHA256 = (
    "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
)


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


def test_kb_manager_request_build_calls_endpoint_and_payload():
    mgr = KBManager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "request_id": "req-1",
        "status": "queued",
        "message": "ok",
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("aibom.kb.manager.httpx.Client", return_value=mock_client):
        out = mgr.request_build("openai", "1.0", api_key="k", api_base="https://api.example")

    expected_url = "https://api.example/api/v1/aibom/kb/requests"
    mock_client.post.assert_called_once()
    call_kw = mock_client.post.call_args
    assert call_kw[0][0] == expected_url
    assert call_kw[1]["json"] == {
        "sdk": "openai",
        "version": "1.0",
        "language": "python",
    }
    headers = call_kw[1]["headers"]
    assert headers["x-cisco-ai-defense-tenant-api-key"] == "k"
    assert headers["Content-Type"] == "application/json"
    assert out == {"request_id": "req-1", "status": "queued", "message": "ok"}


def test_kb_manager_request_build_missing_api_key_raises():
    mgr = KBManager()
    env = {k: v for k, v in os.environ.items() if k != "CISCO_AI_DEFENSE_API_KEY"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(KBError, match="API key required"):
            mgr.request_build("x", "1.0", api_key=None)


def test_kb_manager_request_status_calls_get_endpoint():
    mgr = KBManager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "request_id": "rid",
        "status": "done",
        "sdk": "openai",
        "version": "1.0",
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("aibom.kb.manager.httpx.Client", return_value=mock_client):
        out = mgr.request_status("rid", api_key="k")

    base = DEFAULT_API_BASE.rstrip("/")
    expected = f"{base}/api/v1/aibom/kb/requests/rid"
    mock_client.get.assert_called_once_with(
        expected,
        headers={"x-cisco-ai-defense-tenant-api-key": "k"},
    )
    assert out["request_id"] == "rid"
    assert out["status"] == "done"
    assert out["sdk"] == "openai"
    assert out["version"] == "1.0"


def test_kb_manager_list_requests_returns_list():
    mgr = KBManager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "requests": [
            {"id": "a", "status": "x"},
            {"id": "b"},
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("aibom.kb.manager.httpx.Client", return_value=mock_client):
        items = mgr.list_requests(api_key="k")

    base = DEFAULT_API_BASE.rstrip("/")
    expected = f"{base}/api/v1/aibom/kb/requests"
    mock_client.get.assert_called_once_with(
        expected,
        headers={"x-cisco-ai-defense-tenant-api-key": "k"},
    )
    assert items == [
        {"id": "a", "status": "x"},
        {"id": "b"},
    ]


def test_kb_manager_list_requests_accepts_plain_list():
    mgr = KBManager()
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"r": 1}]
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    with patch("aibom.kb.manager.httpx.Client", return_value=mock_client):
        items = mgr.list_requests(api_key="k")
    assert items == [{"r": 1}]


def test_kb_error_is_exception_subclass():
    assert issubclass(KBError, Exception)
    e = KBError("msg")
    assert str(e) == "msg"
