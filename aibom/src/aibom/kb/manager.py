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
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import httpx
import platformdirs
from pydantic import ValidationError

from .manifest import KBManifest, KBManifestIndex

LOGGER = logging.getLogger(__name__)

DEFAULT_MANIFEST_URL: str | None = None
DEFAULT_API_BASE: str | None = None
HTTP_TIMEOUT = httpx.Timeout(120.0)

_REGIONAL_ENDPOINT_HINT = (
    "Regional API hosts follow the same pattern as AIBOM_POST_URL — "
    "api.security.cisco.com (US), api.eu.security.cisco.com (EU), "
    "api.apj.security.cisco.com (APJ), api.uae.security.cisco.com (UAE)."
)


class KBError(Exception):
    pass


class KBManager:
    def __init__(self, *, manifest_url: str | None = None) -> None:
        self._default_manifest_url = (
            manifest_url
            or os.environ.get("CISCO_AIBOM_MANIFEST_URL")
            or DEFAULT_MANIFEST_URL
        )

    def _resolve_manifest_url(self, url: str | None) -> str:
        resolved = url or self._default_manifest_url
        if not resolved:
            raise KBError(
                "KB manifest URL required: pass --url, set "
                "CISCO_AIBOM_MANIFEST_URL, or construct KBManager(manifest_url=...). "
                + _REGIONAL_ENDPOINT_HINT
            )
        return resolved

    def _user_root(self) -> Path:
        return Path(platformdirs.user_data_dir("aibom"))

    def _local_manifest_path(self) -> Path:
        return self._user_root() / "manifest.json"

    def _local_kb_dir(self) -> Path:
        d = self._user_root() / "catalogs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _compute_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _api_headers(self, api_key: str, *, json_body: bool = False) -> dict[str, str]:
        h: dict[str, str] = {"x-cisco-ai-defense-tenant-api-key": api_key}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _get_manifest(self, url: str) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            LOGGER.error("Failed to fetch manifest from %s: %s", url, e)
            raise KBError(f"Failed to fetch manifest: {e}") from e
        except ValueError as e:
            LOGGER.error("Invalid JSON in manifest response from %s: %s", url, e)
            raise KBError("Manifest response is not valid JSON") from e
        if not isinstance(data, dict):
            raise KBError("Manifest root must be a JSON object")
        return data

    def _resolve_api_key(self, api_key: str | None) -> str:
        key = api_key or os.environ.get("CISCO_AI_DEFENSE_API_KEY")
        if not key:
            raise KBError(
                "API key required: pass api_key or set CISCO_AI_DEFENSE_API_KEY"
            )
        return key

    def _resolve_api_base(self, api_base: str | None) -> str:
        base = api_base or os.environ.get("CISCO_AI_DEFENSE_API_BASE") or DEFAULT_API_BASE
        if not base:
            raise KBError(
                "API base URL required: pass --api-base or set "
                "CISCO_AI_DEFENSE_API_BASE. " + _REGIONAL_ENDPOINT_HINT
            )
        return base.rstrip("/")

    def _select_manifest(self, index: KBManifestIndex, version: str | None) -> KBManifest:
        if version is None:
            return index.latest
        if index.latest.kb_version == version:
            return index.latest
        for v in index.versions:
            if v.kb_version == version:
                return v
        raise KBError(f"Unknown KB version: {version}")

    def _kb_duckdb_path(self, kb_version: str) -> Path:
        return self._local_kb_dir() / f"kb-{kb_version}.duckdb"

    def _version_newer(self, latest: str, current: str) -> bool:
        if not current:
            return True
        try:
            from packaging.version import Version

            return Version(latest) > Version(current)
        except Exception:
            return latest != current

    def download(self, version: str | None = None, url: str | None = None) -> Path:
        manifest_url = self._resolve_manifest_url(url)
        raw = self._get_manifest(manifest_url)
        try:
            index = KBManifestIndex.model_validate(raw)
        except ValidationError as e:
            LOGGER.error("Invalid manifest schema: %s", e)
            raise KBError(f"Invalid manifest: {e}") from e

        chosen = self._select_manifest(index, version)
        dest = self._kb_duckdb_path(chosen.kb_version)
        expected_hash = chosen.duckdb_sha256.lower()

        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                with client.stream("GET", chosen.duckdb_url) as resp:
                    resp.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    digest = hashlib.sha256()
                    nbytes = 0
                    with tmp.open("wb") as out:
                        for chunk in resp.iter_bytes():
                            digest.update(chunk)
                            out.write(chunk)
                            nbytes += len(chunk)
                    got = digest.hexdigest().lower()
                    if got != expected_hash:
                        tmp.unlink(missing_ok=True)
                        raise KBError(
                            "SHA-256 checksum mismatch for downloaded KB "
                            f"(expected {expected_hash}, got {got})"
                        )
                    if chosen.size_bytes and nbytes != chosen.size_bytes:
                        tmp.unlink(missing_ok=True)
                        raise KBError(
                            f"Downloaded size {nbytes} does not match "
                            f"manifest size_bytes {chosen.size_bytes}"
                        )
                    tmp.replace(dest)
        except httpx.HTTPError as e:
            LOGGER.error("KB download failed: %s", e)
            raise KBError(f"Failed to download KB: {e}") from e

        self._user_root().mkdir(parents=True, exist_ok=True)
        self._local_manifest_path().write_text(
            chosen.model_dump_json(indent=2),
            encoding="utf-8",
        )
        LOGGER.info("KB %s saved to %s", chosen.kb_version, dest)
        return dest

    def check(self) -> dict[str, Any]:
        manifest_url = self._resolve_manifest_url(None)
        raw = self._get_manifest(manifest_url)
        try:
            index = KBManifestIndex.model_validate(raw)
        except ValidationError as e:
            raise KBError(f"Invalid manifest: {e}") from e

        latest_ver = index.latest.kb_version
        download_url = index.latest.duckdb_url
        current_ver = ""
        mp = self._local_manifest_path()
        if mp.exists():
            try:
                installed = KBManifest.model_validate(
                    json.loads(mp.read_text(encoding="utf-8"))
                )
                current_ver = installed.kb_version
            except (json.JSONDecodeError, ValidationError) as e:
                LOGGER.warning("Could not read local KB manifest: %s", e)

        update_available = self._version_newer(latest_ver, current_ver)
        return {
            "current_version": current_ver,
            "latest_version": latest_ver,
            "update_available": update_available,
            "download_url": download_url,
        }

    def info(self) -> dict[str, Any]:
        mp = self._local_manifest_path()
        if not mp.exists():
            raise KBError("No local KB manifest found; run download first")
        try:
            m = KBManifest.model_validate(json.loads(mp.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValidationError) as e:
            raise KBError(f"Invalid local manifest: {e}") from e

        db_path = self._kb_duckdb_path(m.kb_version)
        if not db_path.exists():
            raise KBError(f"Local KB file missing at {db_path}")

        stat = db_path.stat()
        sha256 = self._compute_sha256(db_path)
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                row = con.execute(
                    "SELECT COUNT(*) FROM component_catalog"
                ).fetchone()
                entity_count = int(row[0]) if row else 0
            finally:
                con.close()
        except Exception as e:
            LOGGER.error("Failed to query entity count: %s", e)
            raise KBError(f"Failed to query KB database: {e}") from e

        last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        return {
            "version": m.kb_version,
            "path": str(db_path.resolve()),
            "sha256": sha256,
            "size_bytes": stat.st_size,
            "entity_count": entity_count,
            "last_modified": last_modified,
        }

    def verify(self) -> bool:
        mp = self._local_manifest_path()
        if not mp.exists():
            LOGGER.warning("verify: no local manifest")
            return False
        try:
            m = KBManifest.model_validate(json.loads(mp.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValidationError) as e:
            LOGGER.warning("verify: invalid manifest: %s", e)
            return False

        db_path = self._kb_duckdb_path(m.kb_version)
        if not db_path.exists():
            LOGGER.warning("verify: KB file missing at %s", db_path)
            return False

        try:
            got = self._compute_sha256(db_path).lower()
            expected = m.duckdb_sha256.lower()
            if got != expected:
                LOGGER.warning(
                    "verify: SHA-256 mismatch (expected %s, got %s)", expected, got
                )
                return False
            return True
        except OSError as e:
            LOGGER.warning("verify: read error: %s", e)
            return False

    def request_build(
        self,
        sdk: str,
        version: str,
        language: str = "python",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, Any]:
        base = self._resolve_api_base(api_base)
        key = self._resolve_api_key(api_key)
        endpoint = f"{base}/api/ai-defense/v1/aibom/kb/requests"
        payload = {"sdk": sdk, "version": version, "language": language}
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.post(
                    endpoint,
                    json=payload,
                    headers=self._api_headers(key, json_body=True),
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            LOGGER.error("request_build failed: %s", e)
            raise KBError(f"KB build request failed: {e}") from e
        if not isinstance(data, dict):
            raise KBError("Unexpected response from KB build API")
        return {
            "request_id": str(
                data.get("request_id", data.get("id", ""))
            ),
            "status": str(data.get("status", "")),
            "message": str(data.get("message", "")),
        }

    def request_status(
        self,
        request_id: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, Any]:
        base = self._resolve_api_base(api_base)
        key = self._resolve_api_key(api_key)
        endpoint = f"{base}/api/ai-defense/v1/aibom/kb/requests/{request_id}"
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.get(endpoint, headers=self._api_headers(key))
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            LOGGER.error("request_status failed: %s", e)
            raise KBError(f"KB request status failed: {e}") from e
        if not isinstance(data, dict):
            raise KBError("Unexpected response from KB status API")
        return {
            "request_id": str(data.get("request_id", data.get("id", request_id))),
            "status": str(data.get("status", "")),
            "sdk": str(data.get("sdk", "")),
            "version": str(data.get("version", "")),
        }

    def list_requests(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> list[dict[str, Any]]:
        base = self._resolve_api_base(api_base)
        key = self._resolve_api_key(api_key)
        endpoint = f"{base}/api/ai-defense/v1/aibom/kb/requests"
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.get(endpoint, headers=self._api_headers(key))
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            LOGGER.error("list_requests failed: %s", e)
            raise KBError(f"KB list requests failed: {e}") from e

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for k in ("requests", "items", "data", "results"):
                inner = data.get(k)
                if isinstance(inner, list):
                    return [x for x in inner if isinstance(x, dict)]
            return []
        raise KBError("Unexpected response shape from KB list API")
