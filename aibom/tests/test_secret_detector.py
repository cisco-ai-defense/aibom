# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aibom.scanners.secret_detector import SecretDetector

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
