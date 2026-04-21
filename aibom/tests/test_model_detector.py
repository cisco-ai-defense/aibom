# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aibom.scanners.model_detector import (
    ModelDetector,
    _model_alias_keys,
    _registry_cache,
    is_known_embedding_model_name,
)

from .conftest import run_scanner


@pytest.fixture(autouse=True)
def _clear_registry_cache():
    """Ensure the model registry cache is fresh for every test."""
    _registry_cache.clear()
    yield
    _registry_cache.clear()


class TestModelAliasKeys:
    """Unit tests for _model_alias_keys normalization."""

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
        result = _model_alias_keys(raw_id)
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
        assert comps[0].heuristic_confidence == 1.0
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
        assert comps[0].heuristic_confidence == 0.4
        assert comps[0].needs_agentic is True
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

    def test_model_catalog_used_when_available(self, tmp_path: Path) -> None:
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
        assert comps[0].heuristic_confidence == 1.0
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
        assert comps[0].heuristic_confidence == 1.0
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
        assert comps[0].heuristic_confidence == 0.4
        assert comps[0].needs_agentic is True
        assert comps[0].metadata["registry_source"] == "none"

    def test_bedrock_arn_resolves_via_alias(self, tmp_path: Path) -> None:
        meta = {"provider": "bedrock", "family": "claude-3-5-sonnet", "deprecated": False, "source": "model_catalog"}
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
        assert comps[0].heuristic_confidence == 1.0
        assert comps[0].metadata["registry_source"] == "model_catalog"

    def test_shorthand_resolves_via_alias(self, tmp_path: Path) -> None:
        meta = {"provider": "anthropic", "family": "claude-3-5-sonnet", "deprecated": False, "source": "model_catalog"}
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
        assert comps[0].heuristic_confidence == 1.0
        assert comps[0].metadata["registry_source"] == "model_catalog"

    def test_azure_prefixed_model_resolves(self, tmp_path: Path) -> None:
        meta = {"provider": "azure", "family": "gpt-4o", "deprecated": False, "source": "model_catalog"}
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
        assert comps[0].heuristic_confidence == 1.0

    @pytest.mark.parametrize(
        "source_line",
        [
            # Prefixed UPPER_SNAKE with model-related suffix
            'BEDROCK_MODEL_ID = "gpt-4o"\n',
            'CLAUDE_MODEL = "gpt-4o"\n',
            'SONNET_MODEL_NAME = "gpt-4o"\n',
            'ANTHROPIC_MODEL_ID = "gpt-4o"\n',
            'OPENAI_DEPLOYMENT = "gpt-4o"\n',
            # Bare UPPER_SNAKE model-related identifiers
            'MODEL = "gpt-4o"\n',
            'MODEL_ID = "gpt-4o"\n',
            'MODEL_NAME = "gpt-4o"\n',
            'DEPLOYMENT = "gpt-4o"\n',
            'DEPLOYMENT_NAME = "gpt-4o"\n',
            'LLM_MODEL = "gpt-4o"\n',
            # Lowercase identifiers
            'model = "gpt-4o"\n',
            'model_name = "gpt-4o"\n',
            'model_id = "gpt-4o"\n',
            'deployment_name = "gpt-4o"\n',
        ],
    )
    def test_py_assign_captures_model_suffixed_identifiers(
        self, tmp_path: Path, source_line: str
    ) -> None:
        comps, _ = run_scanner(ModelDetector, tmp_path, {"app.py": source_line})
        assert len(comps) == 1, f"did not capture {source_line!r}"
        assert comps[0].model_name == "gpt-4o"
        assert comps[0].metadata["provider"] == "openai"

    @pytest.mark.parametrize(
        "non_model_line",
        [
            'LOG_LEVEL = "info"\n',
            'API_URL = "https://example.com"\n',
            'TIMEOUT = "30s"\n',
            'FOO_BAR = "baz"\n',
        ],
    )
    def test_py_assign_ignores_unrelated_identifiers(
        self, tmp_path: Path, non_model_line: str
    ) -> None:
        """UPPER_SNAKE identifiers without a model-related suffix must not match.

        Guards against the suffix-based widening of ``_PY_ASSIGN_RE`` turning
        every caps constant assignment into a model candidate.
        """
        comps, _ = run_scanner(ModelDetector, tmp_path, {"app.py": non_model_line})
        assert comps == [], f"false positive on {non_model_line!r}"

    @pytest.mark.parametrize(
        "source_line",
        [
            # os.getenv with quoted default
            'MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "gpt-4o")\n',
            'MODEL_ID = os.getenv("CLAUDE_MODEL", "gpt-4o")\n',
            # os.environ.get variant
            'MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "gpt-4o")\n',
            # unqualified getenv (common after ``from os import getenv``)
            'MODEL_ID = getenv("BEDROCK_MODEL_ID", "gpt-4o")\n',
            # inside function call arg
            'Agent(model=os.getenv("MY_MODEL_ID", "gpt-4o"))\n',
        ],
    )
    def test_py_getenv_default_captures_literal_fallback(
        self, tmp_path: Path, source_line: str
    ) -> None:
        comps, _ = run_scanner(ModelDetector, tmp_path, {"app.py": source_line})
        # A single literal "gpt-4o" should be detected; the kwarg/assign regex
        # may also see it but dedup happens via the canonical value.
        assert any(c.model_name == "gpt-4o" for c in comps), (
            f"did not capture literal default from {source_line!r}: {[c.model_name for c in comps]}"
        )
        gpt4o = next(c for c in comps if c.model_name == "gpt-4o")
        assert gpt4o.metadata["provider"] == "openai"

    def test_py_getenv_default_ignores_non_model_env_vars(
        self, tmp_path: Path
    ) -> None:
        """``os.getenv("LOG_LEVEL", "info")`` must not register "info" as a model."""
        comps, _ = run_scanner(
            ModelDetector,
            tmp_path,
            {"app.py": 'level = os.getenv("LOG_LEVEL", "info")\n'},
        )
        assert comps == []

    @pytest.mark.parametrize(
        "model_id, expected_provider, expected_family",
        [
            # Claude 3.7 Sonnet (both hyphen-before-minor and version-first naming)
            ("claude-3-7-sonnet", "anthropic", "claude-3-7-sonnet"),
            ("claude-sonnet-4", "anthropic", "claude-sonnet-4"),
            ("claude-opus-4", "anthropic", "claude-opus-4"),
            ("claude-haiku-4", "anthropic", "claude-haiku-4"),
            # Bedrock-prefixed variants resolve via builtin alias walk
            ("us.anthropic.claude-3-7-sonnet-20250219-v1:0", "anthropic", "claude-3-7-sonnet"),
            ("us.anthropic.claude-sonnet-4-20250514-v1:0", "anthropic", "claude-sonnet-4"),
            # Amazon Nova family
            ("amazon.nova-pro-v1:0", "amazon", "nova-pro"),
            ("amazon.nova-lite-v1:0", "amazon", "nova-lite"),
            ("amazon.nova-micro-v1:0", "amazon", "nova-micro"),
            ("amazon.nova-canvas-v1:0", "amazon", "nova-canvas"),
            # Amazon Titan
            ("amazon.titan-text-express-v1", "amazon", "titan-text"),
            ("amazon.titan-embed-text-v2:0", "amazon", "titan-embed-text"),
            # Meta Llama on Bedrock (no separator between "llama" and the version)
            ("meta.llama3-70b-instruct-v1:0", "meta", "llama-3"),
            ("us.meta.llama3-1-70b-instruct-v1:0", "meta", "llama-3.1"),
            ("meta.llama3-2-11b-instruct-v1:0", "meta", "llama-3.2"),
            ("meta.llama3-3-70b-instruct-v1:0", "meta", "llama-3.3"),
            ("us.meta.llama4-maverick-17b-instruct-v1:0", "meta", "llama-4"),
            # Cohere on Bedrock
            ("cohere.command-r-plus-v1:0", "cohere", "command-r-plus"),
            ("cohere.command-r-v1:0", "cohere", "command-r"),
            ("cohere.command-text-v14", "cohere", "command"),
            ("cohere.embed-english-v3", "cohere", "embed-english"),
            ("cohere.embed-multilingual-v3", "cohere", "embed-multilingual"),
            # Mistral on Bedrock
            ("mistral.mistral-large-2402-v1:0", "mistral", "mistral-large"),
            ("mistral.mixtral-8x7b-instruct-v0:1", "mistral", "mixtral"),
        ],
    )
    def test_builtin_registry_covers_bedrock_ids_when_live_empty(
        self,
        tmp_path: Path,
        model_id: str,
        expected_provider: str,
        expected_family: str,
    ) -> None:
        """Bedrock-formatted IDs must resolve via the builtin regex when the live
        catalog is unavailable (offline mode / fetch failure)."""
        with patch("aibom.scanners.model_detector._get_live_registry", return_value={}):
            comps, _ = run_scanner(
                ModelDetector,
                tmp_path,
                {"app.py": f'model="{model_id}"\n'},
            )
        assert len(comps) == 1, f"{model_id} did not resolve"
        assert comps[0].metadata["provider"] == expected_provider, (
            f"{model_id}: provider={comps[0].metadata.get('provider')!r}"
        )
        assert comps[0].metadata["family"] == expected_family, (
            f"{model_id}: family={comps[0].metadata.get('family')!r}"
        )
        assert comps[0].metadata["registry_source"] == "builtin"


class TestIsKnownEmbeddingModelName:
    """Unit tests for is_known_embedding_model_name.

    The helper must defer to the model registry (LiteLLM ``mode`` field or
    builtin ``family`` annotation) and never apply standalone regex.
    """

    @pytest.mark.parametrize(
        "name",
        [
            # LiteLLM mode=embedding (commercial catalog)
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
            "voyage-3-large",
            "cohere.embed-english-v3",
        ],
    )
    def test_embeddings_return_true(self, name: str) -> None:
        """Known embedding model identifiers must return True."""
        assert is_known_embedding_model_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            # Chat models (mode=chat in LiteLLM)
            "gpt-4o",
            "gpt-4-turbo",
            "claude-3-5-sonnet-20241022",
            # Not a model at all
            "random-nonsense-xyz-not-a-model",
            "",
            "   ",
        ],
    )
    def test_non_embeddings_return_false(self, name: str) -> None:
        """Non-embedding models and junk strings must return False."""
        assert is_known_embedding_model_name(name) is False


class TestStagedEmbeddingReclassification:
    """End-to-end: the scan pipeline must reclassify MODEL→EMBEDDING and
    then collapse duplicates so ``text-embedding-3-large`` never appears
    twice (once as ``model``, once as ``embedding``)."""

    def test_mixed_type_same_name_dedupes_to_single_embedding(
        self, tmp_path: Path
    ) -> None:
        """Two components with the same name but different types (MODEL,
        EMBEDDING) must consolidate into a single EMBEDDING entry."""
        from aibom.models import AIComponent, AIComponentType
        from aibom.scan_pipeline import ScanPipeline

        pipeline = ScanPipeline(scan_paths=[str(tmp_path)])
        components = [
            AIComponent(
                name="text-embedding-3-large",
                component_type=AIComponentType.MODEL,
                file_path="a.py",
                line_number=1,
            ),
            AIComponent(
                name="text-embedding-3-large",
                component_type=AIComponentType.EMBEDDING,
                file_path="b.py",
                line_number=1,
            ),
        ]
        out, _ = pipeline._stage_assemble(components)

        names = [(c.name, c.component_type.value) for c in out]
        assert ("text-embedding-3-large", "embedding") in names
        assert ("text-embedding-3-large", "model") not in names
        # Exactly one entry for this name across all types.
        assert sum(1 for n, _ in names if n == "text-embedding-3-large") == 1
