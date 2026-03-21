# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aibom.scanners.model_detector import ModelDetector, _litellm_alias_keys

from .conftest import run_scanner


class TestLiteLLMAliasKeys:
    """Unit tests for _litellm_alias_keys normalization."""

    @pytest.mark.parametrize(
        "raw_id, expected_aliases",
        [
            # Plain vendor-native: just lowercase
            ("gpt-4o", ["gpt-4o"]),
            # Slash-prefixed Azure: strip prefix, strip date
            (
                "azure/gpt-4o-2024-08-06",
                ["azure/gpt-4o-2024-08-06", "gpt-4o-2024-08-06", "gpt-4o"],
            ),
            # Multi-segment Azure: strip to final segment, strip date
            (
                "azure/eu/gpt-4o-2024-11-20",
                ["azure/eu/gpt-4o-2024-11-20", "gpt-4o-2024-11-20", "gpt-4o"],
            ),
            # Bedrock ARN: dot-separated provider, version suffix
            (
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                [
                    "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "claude-3-5-sonnet-20241022-v2:0",
                    "claude-3-5-sonnet-20241022-v2",
                    "claude-3-5-sonnet",
                ],
            ),
            # Region-prefixed Bedrock ARN
            (
                "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                [
                    "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                    "anthropic.claude-3-7-sonnet-20250219-v1:0",
                    "claude-3-7-sonnet-20250219-v1:0",
                    "claude-3-7-sonnet-20250219-v1",
                    "claude-3-7-sonnet",
                ],
            ),
            # Fine-tune prefix
            (
                "ft:gpt-4o-2024-08-06",
                ["ft:gpt-4o-2024-08-06", "gpt-4o-2024-08-06", "gpt-4o"],
            ),
            # No date suffix to strip: stays as-is
            ("gemini-2.5-pro", ["gemini-2.5-pro"]),
            # Date without -vN
            (
                "claude-4-opus-20250514",
                ["claude-4-opus-20250514", "claude-4-opus"],
            ),
        ],
    )
    def test_alias_generation(self, raw_id: str, expected_aliases: list[str]) -> None:
        result = _litellm_alias_keys(raw_id)
        for alias in expected_aliases:
            assert alias in result, f"{alias!r} not in {result}"


class TestModelDetector:
    def test_detects_model_kwarg_gpt4o_in_python(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            ModelDetector,
            tmp_path,
            {"app.py": 'resp = client.chat(model="gpt-4o", messages=[])\n'},
        )
        assert len(comps) == 1
        assert comps[0].model_name == "gpt-4o"
        assert comps[0].confidence == 1.0
        assert comps[0].metadata.get("provider") == "openai"

    def test_detects_model_in_env_file(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            ModelDetector,
            tmp_path,
            {".env": "MODEL=claude-3-5-sonnet\n"},
        )
        assert len(comps) == 1
        assert comps[0].model_name == "claude-3-5-sonnet"
        assert comps[0].metadata.get("provider") == "anthropic"

    def test_detects_model_in_yaml_config(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            ModelDetector,
            tmp_path,
            {"config.yaml": "llm:\n  model: gemini-1.5-pro\n"},
        )
        assert len(comps) == 1
        assert comps[0].model_name == "gemini-1.5-pro"
        assert comps[0].metadata.get("provider") == "google"

    def test_unknown_model_lower_confidence(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            ModelDetector,
            tmp_path,
            {"app.py": 'x = foo(model="totally-unknown-custom-llm-id", y=1)\n'},
        )
        assert len(comps) == 1
        assert comps[0].model_name == "totally-unknown-custom-llm-id"
        assert comps[0].confidence == 0.7
        assert comps[0].metadata.get("provider") == "unknown"

    def test_model_card_url_open_vs_closed(self, tmp_path: Path) -> None:
        closed, _ = run_scanner(
            ModelDetector,
            tmp_path / "closed",
            {"a.py": 'model="gpt-4o"\n'},
        )
        open_meta, _ = run_scanner(
            ModelDetector,
            tmp_path / "open",
            {"b.py": 'model="llama-3.1-8b-instruct"\n'},
        )
        assert closed[0].metadata["model_card_url"].startswith("https://platform.openai.com")
        assert "huggingface.co" in open_meta[0].metadata["model_card_url"]

    def test_no_false_positive_on_skip_strings(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            ModelDetector,
            tmp_path,
            {"app.py": 'call(model="auto")\nMODEL_NAME = "default"\n'},
        )
        assert comps == []

    def test_litellm_catalog_used_when_available(self, tmp_path: Path) -> None:
        fake_registry = {
            "brand-new-model-2026": {
                "provider": "newco",
                "family": "brand-new-model",
                "deprecated": False,
            }
        }
        with patch("aibom.scanners.model_detector._get_live_registry", return_value=fake_registry):
            comps, _ = run_scanner(
                ModelDetector,
                tmp_path,
                {"app.py": 'x = chat(model="brand-new-model-2026")\n'},
            )
        assert len(comps) == 1
        assert comps[0].confidence == 1.0
        assert comps[0].metadata["provider"] == "newco"

    def test_builtin_fallback_when_live_empty(self, tmp_path: Path) -> None:
        with patch("aibom.scanners.model_detector._get_live_registry", return_value={}):
            comps, _ = run_scanner(
                ModelDetector,
                tmp_path,
                {"app.py": 'model="gpt-4o"\n'},
            )
        assert len(comps) == 1
        assert comps[0].metadata["provider"] == "openai"

    def test_hf_hub_lookup_for_org_slash_model(self, tmp_path: Path) -> None:
        fake_hf = {
            "provider": "meta-llama",
            "family": "Llama-3.1-8B",
            "deprecated": False,
            "source": "huggingface",
            "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
            "pipeline_tag": "text-generation",
            "license": "llama3.1",
            "downloads": 5000000,
            "tags": ["llama", "text-generation"],
        }
        with (
            patch("aibom.scanners.model_detector._get_live_registry", return_value={}),
            patch("aibom.scanners.model_detector._query_hf_hub", return_value=fake_hf),
        ):
            comps, _ = run_scanner(
                ModelDetector,
                tmp_path,
                {"app.py": 'model="meta-llama/Llama-3.1-8B-Instruct"\n'},
            )
        assert len(comps) == 1
        assert comps[0].confidence == 1.0
        assert comps[0].metadata["registry_source"] == "huggingface"
        assert comps[0].metadata["hf_id"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert comps[0].metadata["model_card_url"] == "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
        assert comps[0].metadata["pipeline_tag"] == "text-generation"

    def test_hf_hub_not_queried_for_plain_model_ids(self, tmp_path: Path) -> None:
        with (
            patch("aibom.scanners.model_detector._get_live_registry", return_value={}),
            patch("aibom.scanners.model_detector._query_hf_hub") as mock_hf,
        ):
            run_scanner(
                ModelDetector,
                tmp_path,
                {"app.py": 'model="gpt-4o"\n'},
            )
        mock_hf.assert_not_called()

    def test_hf_hub_miss_returns_unknown(self, tmp_path: Path) -> None:
        with (
            patch("aibom.scanners.model_detector._get_live_registry", return_value={}),
            patch("aibom.scanners.model_detector._query_hf_hub", return_value=None),
        ):
            comps, _ = run_scanner(
                ModelDetector,
                tmp_path,
                {"app.py": 'model="someorg/nonexistent-model"\n'},
            )
        assert len(comps) == 1
        assert comps[0].confidence == 0.7
        assert comps[0].metadata["registry_source"] == "none"

    def test_bedrock_arn_resolves_via_alias(self, tmp_path: Path) -> None:
        meta = {"provider": "bedrock", "family": "claude-3-5-sonnet", "deprecated": False, "source": "litellm"}
        fake_registry = {
            "anthropic.claude-3-5-sonnet-20241022-v2:0": meta,
            "claude-3-5-sonnet-20241022-v2:0": meta,
            "claude-3-5-sonnet-20241022-v2": meta,
            "claude-3-5-sonnet": meta,
        }
        with patch("aibom.scanners.model_detector._get_live_registry", return_value=fake_registry):
            comps, _ = run_scanner(
                ModelDetector,
                tmp_path,
                {"app.py": 'model="anthropic.claude-3-5-sonnet-20241022-v2:0"\n'},
            )
        assert len(comps) == 1
        assert comps[0].confidence == 1.0
        assert comps[0].metadata["registry_source"] == "litellm"

    def test_shorthand_resolves_via_alias(self, tmp_path: Path) -> None:
        meta = {"provider": "anthropic", "family": "claude-3-5-sonnet", "deprecated": False, "source": "litellm"}
        fake_registry = {
            "claude-3-5-sonnet-20241022": meta,
            "claude-3-5-sonnet": meta,
        }
        with patch("aibom.scanners.model_detector._get_live_registry", return_value=fake_registry):
            comps, _ = run_scanner(
                ModelDetector,
                tmp_path,
                {".env": "MODEL=claude-3-5-sonnet\n"},
            )
        assert len(comps) == 1
        assert comps[0].confidence == 1.0
        assert comps[0].metadata["registry_source"] == "litellm"

    def test_azure_prefixed_model_resolves(self, tmp_path: Path) -> None:
        meta = {"provider": "azure", "family": "gpt-4o", "deprecated": False, "source": "litellm"}
        fake_registry = {
            "azure/gpt-4o-2024-08-06": meta,
            "gpt-4o-2024-08-06": meta,
            "gpt-4o": meta,
        }
        with patch("aibom.scanners.model_detector._get_live_registry", return_value=fake_registry):
            comps, _ = run_scanner(
                ModelDetector,
                tmp_path,
                {"config.yaml": "model: gpt-4o\n"},
            )
        assert len(comps) == 1
        assert comps[0].confidence == 1.0
