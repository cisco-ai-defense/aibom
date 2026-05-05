# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from aibom.scanners.secret_detector import SecretDetector, _HAS_DETECT_SECRETS

from .conftest import run_scanner


class TestSecretDetector:
    def test_detects_openai_api_key_pattern(self, tmp_path: Path) -> None:
        secret = "sk-abcdefghijklmnopqrstuvwxyz12"
        comps, _ = run_scanner(
            SecretDetector,
            tmp_path,
            {"keys.env": f"API_KEY={secret}\n"},
        )
        assert any(c.name == "openai_api_key" for c in comps)

    def test_detects_anthropic_api_key_pattern(self, tmp_path: Path) -> None:
        secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890AB"
        comps, _ = run_scanner(
            SecretDetector,
            tmp_path,
            {".env": f"KEY={secret}\n"},
        )
        assert any(c.name == "anthropic_api_key" for c in comps)

    def test_detects_huggingface_token(self, tmp_path: Path) -> None:
        secret = "hf_abcdefghijklmnopqrstuvwxyz12"
        comps, _ = run_scanner(
            SecretDetector,
            tmp_path,
            {"cfg.txt": f"token={secret}\n"},
        )
        assert any(c.name == "huggingface_api_key" for c in comps)

    def test_detects_aws_access_key(self, tmp_path: Path) -> None:
        secret = "AKIAIOSFODNN7EXAMPLE"
        comps, _ = run_scanner(
            SecretDetector,
            tmp_path,
            {"infra.tf": f'key = "{secret}"\n'},
        )
        assert any(c.name == "aws_access_key_id" for c in comps)

    def test_secret_value_not_in_component_serialization(self, tmp_path: Path) -> None:
        secret = "sk-abcdefghijklmnopqrstuvwxyz12"
        comps, _ = run_scanner(
            SecretDetector,
            tmp_path,
            {"leak.py": f'x = "{secret}"\n'},
        )
        assert comps
        blob = "".join(c.model_dump_json() for c in comps)
        assert secret not in blob
        assert all(c.text is None for c in comps)


@pytest.mark.skipif(
    not _HAS_DETECT_SECRETS,
    reason="detect_secrets not installed (analysis extra missing)",
)
def test_secret_detector_does_not_emit_no_plugins_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Regression: when ``analysis`` extra is installed, ``SecretDetector``
    must register the bundled detect_secrets plugins for the scan duration.
    Without that, ``detect_secrets.core.scan.scan_file`` logs
    ``"No plugins to scan with!"`` once per file at ERROR level.
    """
    caplog.set_level(logging.ERROR, logger="detect_secrets.core.scan")

    run_scanner(
        SecretDetector,
        tmp_path,
        {
            "a.py": "x = 1\n",
            "b.py": "y = 2\n",
            "c.py": "z = 3\n",
        },
    )

    offending = [
        r.getMessage()
        for r in caplog.records
        if "No plugins to scan with" in r.getMessage()
    ]
    assert offending == [], (
        "detect_secrets emitted plugin-missing errors during SecretDetector "
        f"scan: {offending}"
    )


@pytest.mark.skipif(
    not _HAS_DETECT_SECRETS,
    reason="detect_secrets not installed (analysis extra missing)",
)
def test_secret_detector_finds_private_key_via_detect_secrets(tmp_path: Path) -> None:
    """When detect_secrets plugins are active, an obvious PEM private key
    should produce a finding tagged ``detection_method=detect_secrets``."""
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDfn"
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\n"
        "-----END PRIVATE KEY-----\n"
    )
    comps, _ = run_scanner(SecretDetector, tmp_path, {"id_rsa.pem": pem})

    methods = {c.metadata.get("detection_method") for c in comps}
    assert "detect_secrets" in methods, (
        f"expected detect_secrets-sourced finding for PEM key, got methods={methods}"
    )
