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
import binascii
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import duckdb
import httpx
import platformdirs
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from packaging.version import InvalidVersion, Version
from pydantic import ValidationError

from ..utils.version import resolve_package_version
from .manifest import KBManifest, KBManifestIndex, KBManifestV2
from .vocabulary import CONCEPTS, SCHEMA_VERSION, VOCABULARY_VERSION, schema_major

LOGGER = logging.getLogger(__name__)

DEFAULT_MANIFEST_URL = (
    "https://cisco-ai-defense-public.s3.us-west-2.amazonaws.com/"
    "aibom/kb/manifest.json"
)
DEFAULT_API_BASE: str | None = None
HTTP_TIMEOUT = httpx.Timeout(120.0)
MAX_REDIRECTS = 3
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_SIGNATURE_BYTES = 64 * 1024
MAX_PUBLIC_KEY_BYTES = 64 * 1024
MAX_COMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 16 * 1024 * 1024 * 1024
_DIRECT_S3_HOST = re.compile(
    r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]\.s3\."
    r"[a-z]{2}-[a-z0-9-]+-[0-9]+\.amazonaws\.com$"
)

_REGIONAL_ENDPOINT_HINT = (
    "Regional API hosts follow the same pattern as AIBOM_POST_URL — "
    "api.security.cisco.com (US), api.eu.security.cisco.com (EU), "
    "api.apj.security.cisco.com (APJ), api.uae.security.cisco.com (UAE)."
)


class KBError(Exception):
    pass


class KBManager:
    def __init__(
        self,
        *,
        manifest_url: str | None = None,
        installed_cli_version: str | None = None,
    ) -> None:
        self._default_manifest_url = (
            manifest_url
            or os.environ.get("CISCO_AIBOM_MANIFEST_URL")
            or DEFAULT_MANIFEST_URL
        )
        self._installed_cli_version = installed_cli_version or resolve_package_version(
            "cisco-aibom"
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

    def _config_path(self) -> Path:
        return Path(platformdirs.user_config_dir("aibom")) / "kb.json"

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

    def _compressed_path(self, kb_version: str) -> Path:
        return self._local_kb_dir() / f"kb-{kb_version}.duckdb.gz"

    def _signature_path(self, kb_version: str) -> Path:
        return self._local_kb_dir() / f"kb-{kb_version}.duckdb.gz.sig"

    def _public_key_path(self, kb_version: str) -> Path:
        return self._local_kb_dir() / f"kb-{kb_version}.pub"

    def configured_pin(self) -> str | None:
        path = self._config_path()
        try:
            exists = path.exists()
        except OSError as exc:
            raise KBError(f"Could not read KB configuration: {exc}") from exc
        if not exists:
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KBError(f"Invalid KB configuration: {exc}") from exc
        pin = raw.get("kb_pin") if isinstance(raw, dict) else None
        return pin.strip() if isinstance(pin, str) and pin.strip() else None

    def set_pin(self, pin: str | None) -> Path:
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        value = pin.strip() if pin else ""
        if value.startswith("https://"):
            self._validate_direct_s3_url(value, field="kb_pin")
            if not value.endswith("/manifest.json"):
                raise KBError("Pinned manifest URL must end with /manifest.json")
        elif value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
            raise KBError("Pinned KB version is invalid")
        path.write_text(
            json.dumps({"kb_pin": value or None}, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _api_headers(self, api_key: str, *, json_body: bool = False) -> dict[str, str]:
        h: dict[str, str] = {"x-cisco-ai-defense-tenant-api-key": api_key}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _validate_direct_s3_url(
        self,
        raw: str,
        *,
        field: str,
        expected_origin: tuple[str, str] | None = None,
    ) -> tuple[str, str]:
        parsed = urlsplit(raw)
        try:
            port = parsed.port
        except ValueError as exc:
            raise KBError(f"{field} contains an invalid port") from exc
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.query
            or parsed.fragment
            or not _DIRECT_S3_HOST.fullmatch(parsed.hostname)
            or not parsed.path.startswith("/aibom/kb/")
            or parsed.path.count("/aibom/kb/") != 1
            or "//" in parsed.path
            or "%" in parsed.path
        ):
            raise KBError(
                f"{field} must be a direct regional S3 HTTPS URL under /aibom/kb/"
            )
        origin = (parsed.scheme, parsed.hostname)
        if expected_origin and origin != expected_origin:
            raise KBError(f"{field} must use the manifest's S3 origin")
        return origin

    def _validate_https_url(self, raw: str, *, field: str) -> None:
        parsed = urlsplit(raw)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise KBError(f"{field} must be an absolute HTTPS URL")

    def _version_manifest_url(self, pointer_url: str, version: str) -> str:
        parsed = urlsplit(pointer_url)
        if not parsed.path.endswith("/manifest.json"):
            raise KBError("KB pointer URL must end with /manifest.json")
        base_path = parsed.path[: -len("manifest.json")]
        path = f"{base_path}schema-v2/v{version}/manifest.json"
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    def _selected_manifest_url(
        self,
        *,
        version: str | None,
        url: str | None,
    ) -> str:
        if version and url:
            raise KBError("Pass either a KB version or manifest URL, not both")
        if url:
            return url
        pointer = self._resolve_manifest_url(None)
        if version:
            return self._version_manifest_url(pointer, version)
        pin = self.configured_pin()
        if pin:
            if pin.startswith("https://"):
                return pin
            return self._version_manifest_url(pointer, pin)
        return pointer

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=HTTP_TIMEOUT,
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
        )

    def _response_bytes(
        self,
        response: httpx.Response,
        *,
        limit: int,
        label: str,
    ) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > limit:
                    raise KBError(f"{label} exceeds the {limit}-byte limit")
            except ValueError as exc:
                raise KBError(f"{label} has an invalid Content-Length") from exc
        data = bytearray()
        for chunk in response.iter_bytes():
            if len(data) + len(chunk) > limit:
                raise KBError(f"{label} exceeds the {limit}-byte limit")
            data.extend(chunk)
        return bytes(data)

    def _get_bytes(self, url: str, *, limit: int, label: str) -> bytes:
        try:
            with self._client() as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    for prior in [*response.history, response]:
                        self._validate_direct_s3_url(
                            str(prior.url),
                            field=f"{label} response URL",
                        )
                    return self._response_bytes(response, limit=limit, label=label)
        except KBError:
            raise
        except httpx.HTTPError as exc:
            raise KBError(f"Failed to fetch {label}: {exc}") from exc

    def _get_manifest(self, url: str) -> dict[str, Any]:
        try:
            with self._client() as client:
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    source = urlsplit(url)
                    if source.hostname and _DIRECT_S3_HOST.fullmatch(source.hostname):
                        for prior in [*resp.history, resp]:
                            self._validate_direct_s3_url(
                                str(prior.url),
                                field="manifest response URL",
                            )
                    body = self._response_bytes(
                        resp,
                        limit=MAX_MANIFEST_BYTES,
                        label="KB manifest",
                    )
            data = json.loads(body)
        except httpx.HTTPError as e:
            LOGGER.error("Failed to fetch manifest from %s: %s", url, e)
            raise KBError(f"Failed to fetch manifest: {e}") from e
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            LOGGER.error("Invalid JSON in manifest response from %s: %s", url, e)
            raise KBError("Manifest response is not valid JSON") from e
        if not isinstance(data, dict):
            raise KBError("Manifest root must be a JSON object")
        return data

    def _validate_manifest_v2_urls(
        self,
        manifest: KBManifestV2,
        manifest_url: str,
    ) -> None:
        origin = self._validate_direct_s3_url(
            manifest_url,
            field="manifest URL",
        )
        prefix = f"/aibom/kb/schema-v2/v{manifest.kb_version}/"
        expected = {
            "duckdb.url": prefix + "aibom_catalog.duckdb.gz",
            "signature.url": prefix + "aibom_catalog.duckdb.gz.sig",
            "source_candidate.validation_report_url": (
                prefix + "validation-report.json"
            ),
        }
        values = {
            "duckdb.url": manifest.duckdb.url,
            "signature.url": manifest.signature.url,
            "source_candidate.validation_report_url": (
                manifest.source_candidate.validation_report_url
            ),
        }
        if manifest.signature.algorithm == "ECDSA_SHA_256":
            expected["signature.public_key_url"] = prefix + "cosign.pub"
            values["signature.public_key_url"] = manifest.signature.public_key_url
        for field, value in values.items():
            self._validate_direct_s3_url(
                value,
                field=field,
                expected_origin=origin,
            )
            if urlsplit(value).path != expected[field]:
                raise KBError(f"{field} must resolve to {expected[field]}")
        if manifest.freshness_api:
            self._validate_https_url(
                manifest.freshness_api,
                field="freshness_api",
            )

    def _require_supported_cli(self, manifest: KBManifestV2) -> None:
        try:
            installed = Version(self._installed_cli_version)
            required = Version(manifest.min_cli_version)
        except InvalidVersion as exc:
            raise KBError(
                f"Invalid CLI version in distribution contract: {exc}"
            ) from exc
        if installed < required:
            raise KBError(
                "This KB requires a newer CLI "
                f"(installed {installed}, required {required}). "
                "Upgrade with: pip install --upgrade cisco-aibom. "
                f"Manifest: {manifest.duckdb.url.rsplit('/', 1)[0]}/manifest.json"
            )

    def _resolve_api_key(self, api_key: str | None) -> str:
        key = api_key or os.environ.get("CISCO_AI_DEFENSE_API_KEY")
        if not key:
            raise KBError(
                "API key required: pass api_key or set CISCO_AI_DEFENSE_API_KEY"
            )
        return key

    def _resolve_api_base(self, api_base: str | None) -> str:
        base = (
            api_base or os.environ.get("CISCO_AI_DEFENSE_API_BASE") or DEFAULT_API_BASE
        )
        if not base:
            raise KBError(
                "API base URL required: pass --api-base or set "
                "CISCO_AI_DEFENSE_API_BASE. " + _REGIONAL_ENDPOINT_HINT
            )
        return base.rstrip("/")

    def _select_manifest(
        self, index: KBManifestIndex, version: str | None
    ) -> KBManifest:
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

    def download(
        self,
        version: str | None = None,
        url: str | None = None,
        *,
        allow_unsigned: bool = False,
    ) -> Path:
        manifest_url = self._selected_manifest_url(version=version, url=url)
        raw = self._get_manifest(manifest_url)
        if schema_major(raw.get("schema_version")) == SCHEMA_VERSION:
            try:
                manifest = KBManifestV2.model_validate(raw)
            except ValidationError as exc:
                raise KBError(f"Invalid schema-v2 manifest: {exc}") from exc
            self._validate_manifest_v2_urls(manifest, manifest_url)
            self._require_supported_cli(manifest)
            return self._download_v2(manifest, allow_unsigned=allow_unsigned)

        return self._download_legacy(raw, version=version)

    def _download_legacy(
        self,
        raw: dict[str, Any],
        *,
        version: str | None,
    ) -> Path:
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

    def _download_file(
        self,
        url: str,
        destination: Path,
        *,
        limit: int,
        expected_size: int,
        label: str,
    ) -> tuple[bytes, int]:
        digest = hashlib.sha256()
        size = 0
        try:
            with self._client() as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    for prior in [*response.history, response]:
                        self._validate_direct_s3_url(
                            str(prior.url),
                            field=f"{label} response URL",
                        )
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            declared = int(content_length)
                        except ValueError as exc:
                            raise KBError(
                                f"{label} has an invalid Content-Length"
                            ) from exc
                        if declared > limit:
                            raise KBError(f"{label} exceeds the {limit}-byte limit")
                    with destination.open("wb") as output:
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > limit:
                                raise KBError(f"{label} exceeds the {limit}-byte limit")
                            digest.update(chunk)
                            output.write(chunk)
        except KBError:
            destination.unlink(missing_ok=True)
            raise
        except (httpx.HTTPError, OSError) as exc:
            destination.unlink(missing_ok=True)
            raise KBError(f"Failed to fetch {label}: {exc}") from exc
        if size != expected_size:
            destination.unlink(missing_ok=True)
            raise KBError(
                f"{label} size {size} does not match manifest size_bytes "
                f"{expected_size}"
            )
        return digest.digest(), size

    def _verify_v2_signature(
        self,
        digest: bytes,
        signature_text: bytes,
        public_key_bytes: bytes,
    ) -> None:
        try:
            signature = base64.b64decode(signature_text, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise KBError("Detached signature is not valid base64") from exc
        try:
            public_key = serialization.load_pem_public_key(public_key_bytes)
        except (ValueError, TypeError) as exc:
            raise KBError("Public key is not valid PEM") from exc
        if not isinstance(public_key, ec.EllipticCurvePublicKey) or not isinstance(
            public_key.curve, ec.SECP256R1
        ):
            raise KBError("Public key must be an ECDSA P-256 key")
        try:
            public_key.verify(
                signature,
                digest,
                ec.ECDSA(utils.Prehashed(hashes.SHA256())),
            )
        except InvalidSignature as exc:
            raise KBError("Detached signature verification failed") from exc

    def _decompress_v2(self, source: Path, destination: Path) -> None:
        written = 0
        try:
            with (
                gzip.open(source, "rb") as compressed,
                destination.open("wb") as output,
            ):
                while True:
                    chunk = compressed.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_DECOMPRESSED_BYTES:
                        raise KBError(
                            "Decompressed KB exceeds the configured size limit"
                        )
                    output.write(chunk)
        except (gzip.BadGzipFile, EOFError, OSError) as exc:
            destination.unlink(missing_ok=True)
            raise KBError(f"Failed to decompress KB artifact: {exc}") from exc

    def _validate_duckdb(self, path: Path) -> None:
        try:
            connection = duckdb.connect(str(path), read_only=True)
            try:
                tables = {
                    str(row[0]) for row in connection.execute("SHOW TABLES").fetchall()
                }
            finally:
                connection.close()
        except Exception as exc:
            raise KBError(
                f"Downloaded KB is not a readable DuckDB database: {exc}"
            ) from exc
        if "component_catalog" not in tables:
            raise KBError("Downloaded KB is missing component_catalog")

    def _install_v2(
        self,
        manifest: KBManifestV2,
        compressed_part: Path,
        digest: bytes,
        signature_bytes: bytes,
        public_key_bytes: bytes,
        *,
        allow_unsigned: bool,
    ) -> Path:
        expected = bytes.fromhex(manifest.duckdb.sha256)
        if digest != expected:
            raise KBError(
                "SHA-256 checksum mismatch for compressed KB "
                f"(expected {manifest.duckdb.sha256}, got {digest.hex()})"
            )
        if manifest.signature.algorithm == "disabled":
            if not allow_unsigned:
                raise KBError(
                    "Unsigned KB artifacts are disabled; use the explicit "
                    "lower-environment override only for rehearsals"
                )
            if manifest.authoritative:
                raise KBError("Authoritative KB artifacts must be signed")
        else:
            self._verify_v2_signature(digest, signature_bytes, public_key_bytes)

        destination = self._kb_duckdb_path(manifest.kb_version)
        db_part = destination.with_suffix(destination.suffix + ".part")
        self._decompress_v2(compressed_part, db_part)
        try:
            self._validate_duckdb(db_part)
            destination.parent.mkdir(parents=True, exist_ok=True)
            compressed_dest = self._compressed_path(manifest.kb_version)
            signature_dest = self._signature_path(manifest.kb_version)
            public_key_dest = self._public_key_path(manifest.kb_version)
            compressed_part.replace(compressed_dest)
            signature_dest.write_bytes(signature_bytes)
            if public_key_bytes:
                public_key_dest.write_bytes(public_key_bytes)
            else:
                public_key_dest.unlink(missing_ok=True)
            db_part.replace(destination)
            self._user_root().mkdir(parents=True, exist_ok=True)
            manifest_part = self._local_manifest_path().with_suffix(".json.part")
            manifest_part.write_text(
                manifest.model_dump_json(indent=2),
                encoding="utf-8",
            )
            manifest_part.replace(self._local_manifest_path())
        except Exception:
            db_part.unlink(missing_ok=True)
            raise
        return destination

    def _download_v2(
        self,
        manifest: KBManifestV2,
        *,
        allow_unsigned: bool,
    ) -> Path:
        compressed_part = self._compressed_path(manifest.kb_version).with_suffix(
            ".gz.part"
        )
        compressed_part.parent.mkdir(parents=True, exist_ok=True)
        try:
            digest, _ = self._download_file(
                manifest.duckdb.url,
                compressed_part,
                limit=MAX_COMPRESSED_BYTES,
                expected_size=manifest.duckdb.size_bytes,
                label="compressed KB",
            )
            signature_bytes = b""
            public_key_bytes = b""
            if manifest.signature.algorithm != "disabled":
                signature_bytes = self._get_bytes(
                    manifest.signature.url,
                    limit=MAX_SIGNATURE_BYTES,
                    label="detached signature",
                ).strip()
                public_key_bytes = self._get_bytes(
                    manifest.signature.public_key_url,
                    limit=MAX_PUBLIC_KEY_BYTES,
                    label="public key",
                )
            return self._install_v2(
                manifest,
                compressed_part,
                digest,
                signature_bytes,
                public_key_bytes,
                allow_unsigned=allow_unsigned,
            )
        except Exception:
            compressed_part.unlink(missing_ok=True)
            raise

    def install_prefetched(
        self,
        manifest_path: Path,
        *,
        artifact_path: Path | None = None,
        signature_path: Path | None = None,
        public_key_path: Path | None = None,
        allow_unsigned: bool = False,
    ) -> Path:
        """Install prefetched bytes through the production validation path."""

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = KBManifestV2.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise KBError(f"Invalid prefetched manifest: {exc}") from exc
        manifest_url = self._version_manifest_url(
            self._resolve_manifest_url(None),
            manifest.kb_version,
        )
        self._validate_manifest_v2_urls(manifest, manifest_url)
        self._require_supported_cli(manifest)
        artifact = artifact_path or manifest_path.parent / "aibom_catalog.duckdb.gz"
        signature = (
            signature_path or manifest_path.parent / "aibom_catalog.duckdb.gz.sig"
        )
        public_key = public_key_path or manifest_path.parent / "cosign.pub"
        try:
            size = artifact.stat().st_size
            if size > MAX_COMPRESSED_BYTES:
                raise KBError("Prefetched KB exceeds the compressed size limit")
            if size != manifest.duckdb.size_bytes:
                raise KBError(
                    f"Prefetched KB size {size} does not match manifest size_bytes "
                    f"{manifest.duckdb.size_bytes}"
                )
            digest = bytes.fromhex(self._compute_sha256(artifact))
            signature_bytes = (
                signature.read_bytes().strip()
                if manifest.signature.algorithm != "disabled"
                else b""
            )
            public_key_bytes = (
                public_key.read_bytes()
                if manifest.signature.algorithm != "disabled"
                else b""
            )
            compressed_part = self._compressed_path(manifest.kb_version).with_suffix(
                ".gz.part"
            )
            compressed_part.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(artifact, compressed_part)
        except OSError as exc:
            raise KBError(f"Failed to read prefetched KB objects: {exc}") from exc
        try:
            return self._install_v2(
                manifest,
                compressed_part,
                digest,
                signature_bytes,
                public_key_bytes,
                allow_unsigned=allow_unsigned,
            )
        except Exception:
            compressed_part.unlink(missing_ok=True)
            raise

    def check(self) -> dict[str, Any]:
        manifest_url = self._selected_manifest_url(version=None, url=None)
        raw = self._get_manifest(manifest_url)
        if schema_major(raw.get("schema_version")) == SCHEMA_VERSION:
            try:
                manifest = KBManifestV2.model_validate(raw)
            except ValidationError as exc:
                raise KBError(f"Invalid schema-v2 manifest: {exc}") from exc
            self._validate_manifest_v2_urls(manifest, manifest_url)
            self._require_supported_cli(manifest)
            current_ver = ""
            local = self._local_manifest_path()
            if local.exists():
                try:
                    installed = json.loads(local.read_text(encoding="utf-8"))
                    current_ver = str(
                        installed.get("kb_version") or installed.get("build_id") or ""
                    )
                except (OSError, json.JSONDecodeError):
                    current_ver = ""
            return {
                "current_version": current_ver,
                "latest_version": manifest.kb_version,
                "update_available": self._version_newer(
                    manifest.kb_version,
                    current_ver,
                ),
                "download_url": manifest.duckdb.url,
                "build_type": manifest.build_type,
                "parent_kb_version": manifest.parent_kb_version,
            }
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
            manifest = json.loads(mp.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise KBError("Invalid local manifest: root must be an object")
        except json.JSONDecodeError as e:
            raise KBError(f"Invalid local manifest: {e}") from e

        manifest_schema = schema_major(manifest.get("schema_version"))
        if manifest_schema == SCHEMA_VERSION:
            version, db_path = self._schema_v2_info_source(manifest, mp)
            schema_version = manifest.get("schema_version")
            vocabulary_version = manifest.get("vocabulary_version")
        else:
            try:
                legacy = KBManifest.model_validate(manifest)
            except ValidationError as e:
                raise KBError(f"Invalid local manifest: {e}") from e
            version = legacy.kb_version
            db_path = self._kb_duckdb_path(version)
            schema_version = legacy.schema_version
            vocabulary_version = legacy.vocabulary_version

        if not db_path.exists():
            raise KBError(f"Local KB file missing at {db_path}")

        stat = db_path.stat()
        sha256 = self._compute_sha256(db_path)
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                row = con.execute("SELECT COUNT(*) FROM component_catalog").fetchone()
                entity_count = int(row[0]) if row else 0
            finally:
                con.close()
        except Exception as e:
            LOGGER.error("Failed to query entity count: %s", e)
            raise KBError(f"Failed to query KB database: {e}") from e

        last_modified = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()
        info: dict[str, Any] = {
            "version": version,
            "path": str(db_path.resolve()),
            "sha256": sha256,
            "size_bytes": stat.st_size,
            "entity_count": entity_count,
            "last_modified": last_modified,
        }
        if manifest_schema == SCHEMA_VERSION:
            info.update(
                {
                    "schema_version": schema_version,
                    "vocabulary_version": (
                        vocabulary_version
                        if isinstance(vocabulary_version, str) and vocabulary_version
                        else VOCABULARY_VERSION
                    ),
                    "concept_count": len(CONCEPTS),
                    "concepts": ", ".join(CONCEPTS),
                }
            )
        return info

    def output_metadata(self) -> dict[str, Any]:
        """Return stable KB/CLI identity without opening the DuckDB file."""

        metadata: dict[str, Any] = {
            "kb_version": "",
            "build_type": "",
            "schema_version": "",
            "cli_version": self._installed_cli_version,
        }
        path = self._local_manifest_path()
        try:
            exists = path.exists()
        except OSError:
            return metadata
        if not exists:
            return metadata
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return metadata
        if not isinstance(raw, dict):
            return metadata
        metadata.update(
            {
                "kb_version": raw.get("kb_version") or raw.get("build_id") or "",
                "build_type": raw.get("build_type") or "legacy",
                "schema_version": raw.get("schema_version") or "",
            }
        )
        return metadata

    def _schema_v2_info_source(
        self,
        manifest: dict[str, Any],
        manifest_path: Path,
    ) -> tuple[str, Path]:
        """Resolve offline display identity and DuckDB path for schema v2."""

        raw_version = manifest.get("kb_version") or manifest.get("build_id")
        if not isinstance(raw_version, str) or not raw_version.strip():
            raise KBError(
                "Invalid local schema-v2 manifest: "
                "kb_version or build_id is required"
            )
        version = raw_version.strip()

        duckdb_entry = manifest.get("duckdb")
        if not isinstance(duckdb_entry, dict):
            raise KBError("Invalid local schema-v2 manifest: duckdb object is required")
        filename = duckdb_entry.get("filename")
        if isinstance(filename, str) and filename.strip():
            candidate = Path(filename.strip()).expanduser()
            if candidate.is_absolute():
                return version, candidate
            return version, (manifest_path.parent / candidate).resolve()
        return version, self._kb_duckdb_path(version)

    def verify(self) -> bool:
        mp = self._local_manifest_path()
        if not mp.exists():
            LOGGER.warning("verify: no local manifest")
            return False
        try:
            raw = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            LOGGER.warning("verify: invalid manifest: %s", e)
            return False

        if schema_major(raw.get("schema_version")) == SCHEMA_VERSION:
            try:
                manifest = KBManifestV2.model_validate(raw)
                compressed = self._compressed_path(manifest.kb_version)
                database = self._kb_duckdb_path(manifest.kb_version)
                if not compressed.exists() or not database.exists():
                    return False
                digest = bytes.fromhex(self._compute_sha256(compressed))
                if digest != bytes.fromhex(manifest.duckdb.sha256):
                    return False
                if manifest.signature.algorithm == "disabled":
                    return False
                self._verify_v2_signature(
                    digest,
                    self._signature_path(manifest.kb_version).read_bytes().strip(),
                    self._public_key_path(manifest.kb_version).read_bytes(),
                )
                self._validate_duckdb(database)
                return True
            except (KBError, OSError, ValidationError, ValueError) as exc:
                LOGGER.warning("verify: schema-v2 verification failed: %s", exc)
                return False

        try:
            m = KBManifest.model_validate(raw)
        except ValidationError as e:
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
        ecosystem: str,
        package_name: str,
        symbols: list[str],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, Any]:
        base = self._resolve_api_base(api_base)
        key = self._resolve_api_key(api_key)
        endpoint = f"{base}/api/v1/aibom/kb/requests"
        payload = {
            "ecosystem": ecosystem,
            "package_name": package_name,
            "symbols": symbols,
        }
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.post(
                    endpoint,
                    json=payload,
                    headers=self._api_headers(key, json_body=True),
                )
                data = self._api_response(resp, operation="KB build request")
        except httpx.HTTPError as e:
            LOGGER.error("request_build failed: %s", e)
            raise KBError(f"KB build request failed: {e}") from e
        if not isinstance(data, dict):
            raise KBError("Unexpected response from KB build API")
        return self._with_rate_limit(data, resp)

    def request_bulk(
        self,
        requests: list[dict[str, Any]],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, Any]:
        if not 1 <= len(requests) <= 20:
            raise KBError("Bulk requests require between 1 and 20 packages")
        total_symbols = sum(len(item.get("symbols", [])) for item in requests)
        if total_symbols > 200:
            raise KBError("Bulk requests support at most 200 symbols total")
        base = self._resolve_api_base(api_base)
        key = self._resolve_api_key(api_key)
        endpoint = f"{base}/api/v1/aibom/kb/requests/bulk"
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.post(
                    endpoint,
                    json={"requests": requests},
                    headers=self._api_headers(key, json_body=True),
                )
                data = self._api_response(resp, operation="bulk KB build request")
        except httpx.HTTPError as exc:
            raise KBError(f"Bulk KB build request failed: {exc}") from exc
        if not isinstance(data, dict):
            raise KBError("Unexpected response from bulk KB build API")
        return self._with_rate_limit(data, resp)

    def _api_response(self, response: httpx.Response, *, operation: str) -> Any:
        try:
            data = response.json()
        except ValueError as exc:
            raise KBError(f"{operation} returned invalid JSON") from exc
        if response.is_error:
            if isinstance(data, dict):
                detail = (
                    data.get("message") or data.get("error") or response.reason_phrase
                )
            else:
                detail = response.reason_phrase
            retry_after = response.headers.get("retry-after")
            suffix = f" Retry after {retry_after} seconds." if retry_after else ""
            raise KBError(
                f"{operation} failed ({response.status_code}): {detail}.{suffix}"
            )
        return data

    def _with_rate_limit(
        self,
        data: dict[str, Any],
        response: httpx.Response,
    ) -> dict[str, Any]:
        result = dict(data)
        result["rate_limit"] = {
            "limit": response.headers.get("x-ratelimit-limit", ""),
            "remaining": response.headers.get("x-ratelimit-remaining", ""),
            "reset": response.headers.get("x-ratelimit-reset", ""),
        }
        return result

    def request_status(
        self,
        request_id: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, Any]:
        base = self._resolve_api_base(api_base)
        key = self._resolve_api_key(api_key)
        endpoint = f"{base}/api/v1/aibom/kb/requests/{request_id}"
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.get(endpoint, headers=self._api_headers(key))
                data = self._api_response(resp, operation="KB request status")
        except httpx.HTTPError as e:
            LOGGER.error("request_status failed: %s", e)
            raise KBError(f"KB request status failed: {e}") from e
        if not isinstance(data, dict):
            raise KBError("Unexpected response from KB status API")
        return data

    def list_requests(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise KBError("Request list limit must be between 1 and 100")
        base = self._resolve_api_base(api_base)
        key = self._resolve_api_key(api_key)
        endpoint = f"{base}/api/v1/aibom/kb/requests"
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                resp = client.get(
                    endpoint,
                    headers=self._api_headers(key),
                    params={"limit": limit},
                )
                data = self._api_response(resp, operation="KB request list")
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
