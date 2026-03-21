# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aibom.models import ScanContext
from aibom.models.enums import AIComponentType, Severity
from aibom.scanners.vuln_scanner import OsvProvider, PackageRef, Vulnerability, VulnScanner


def _httpx_client_cm(inner_client: MagicMock) -> MagicMock:
    cm = MagicMock()
    cm.__enter__.return_value = inner_client
    cm.__exit__.return_value = None
    return cm


@pytest.fixture
def osv_response_openai() -> dict:
    return {
        "results": [
            {
                "vulns": [
                    {
                        "id": "CVE-2024-1234",
                        "summary": "Test vulnerability in openai",
                        "severity": [{"type": "CVSS_V3", "score": "7.5"}],
                        "affected": [
                            {
                                "package": {"name": "openai", "ecosystem": "PyPI"},
                                "ranges": [{"events": [{"fixed": "1.3.0"}]}],
                            }
                        ],
                    }
                ]
            }
        ]
    }


def test_osv_provider_batch_query_payload_and_parse(osv_response_openai: dict) -> None:
    inner = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = osv_response_openai
    inner.post.return_value = resp
    ref = PackageRef(name="openai", version="1.0.0", ecosystem="PyPI")
    with patch(
        "aibom.scanners.vuln_scanner.httpx.Client",
        return_value=_httpx_client_cm(inner),
    ):
        prov = OsvProvider()
        out = prov.batch_query([ref])
    inner.post.assert_called_once()
    args, kwargs = inner.post.call_args
    assert args[0] == "https://api.osv.dev/v1/querybatch"
    key = "PyPI:openai@1.0.0"
    assert key in out
    v = out[key][0]
    assert v.id == "CVE-2024-1234"
    assert v.severity == Severity.HIGH
    assert v.fixed_version == "1.3.0"


@patch("aibom.scanners.vuln_scanner.httpx.Client")
def test_vuln_scanner_uses_detected_packages(
    mock_client_cls: MagicMock, osv_response_openai: dict, tmp_path: Path
) -> None:
    inner = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = osv_response_openai
    inner.post.return_value = resp
    mock_client_cls.return_value = _httpx_client_cm(inner)
    ctx = ScanContext(
        paths=[str(tmp_path)],
        config={"detected_packages": [{"name": "openai", "version": "1.0.0", "ecosystem": "pypi"}]},
    )
    comps, rels = VulnScanner().scan(ctx)
    assert rels == []
    assert len(comps) == 1
    assert comps[0].name == "openai"
    vulns = comps[0].metadata.get("vulnerabilities")
    assert vulns and vulns[0]["id"] == "CVE-2024-1234"


@patch("aibom.scanners.vuln_scanner.httpx.Client")
def test_vuln_scanner_falls_back_to_dependency_scanner(
    mock_client_cls: MagicMock, osv_response_openai: dict, tmp_path: Path
) -> None:
    inner = MagicMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = osv_response_openai
    inner.post.return_value = resp
    mock_client_cls.return_value = _httpx_client_cm(inner)
    (tmp_path / "requirements.txt").write_text("openai==1.0.0\n", encoding="utf-8")
    comps, _ = VulnScanner().scan(ScanContext(paths=[str(tmp_path)], config={}))
    assert len(comps) == 1
    assert any(v["id"] == "CVE-2024-1234" for v in comps[0].metadata["vulnerabilities"])


def test_vulnerability_model_dump_and_validate() -> None:
    v = Vulnerability(
        id="GHSA-abc", summary="x", severity=Severity.MEDIUM, cvss_score=5.0,
        affected_package="pkg", affected_version="2.0.0", fixed_version="2.1.0",
        source="osv", url="https://example.com",
    )
    data = v.model_dump(mode="json")
    assert data["id"] == "GHSA-abc"
    v2 = Vulnerability.model_validate(data)
    assert v2 == v
