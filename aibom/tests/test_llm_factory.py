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

import unittest
from unittest.mock import MagicMock, patch


@patch(
    "aibom.llm_factory.ensure_llm_runtime_available",
    side_effect=lambda model_string, *, provider=None: (
        __import__("aibom.llm_factory", fromlist=["resolve_provider"]).resolve_provider(
            model_string, provider
        )
    ),
)
class TestBuildChatModel(unittest.TestCase):
    """Tests for ``aibom.llm_factory.build_chat_model``."""

    @patch("aibom.llm_factory.init_chat_model")
    def test_plain_model_no_provider(self, mock_init, _mock_preflight):
        """LangChain infers provider when neither flag nor prefix is given."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("gpt-4o", api_key="k")
        args, kwargs = mock_init.call_args
        self.assertEqual(args[0], "gpt-4o")
        self.assertNotIn("model_provider", kwargs)
        self.assertEqual(kwargs["api_key"], "k")

    @patch("aibom.llm_factory.init_chat_model")
    def test_explicit_provider_flag(self, mock_init, _mock_preflight):
        """--llm-provider bedrock sets model_provider without touching model_string."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model(
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            provider="bedrock",
        )
        args, kwargs = mock_init.call_args
        self.assertEqual(args[0], "us.anthropic.claude-sonnet-4-20250514-v1:0")
        self.assertEqual(kwargs["model_provider"], "bedrock")

    @patch("aibom.llm_factory.init_chat_model")
    def test_slash_prefix_backward_compat(self, mock_init, _mock_preflight):
        """Legacy provider/model slash convention still works."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("bedrock/anthropic.claude-3-5-sonnet")
        args, kwargs = mock_init.call_args
        self.assertEqual(args[0], "anthropic.claude-3-5-sonnet")
        self.assertEqual(kwargs["model_provider"], "bedrock")

    @patch("aibom.llm_factory.init_chat_model")
    def test_explicit_provider_overrides_slash(self, mock_init, _mock_preflight):
        """--llm-provider takes precedence over a slash in the model string."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model(
            "bedrock/anthropic.claude-3-5-sonnet",
            provider="bedrock_converse",
        )
        args, kwargs = mock_init.call_args
        self.assertEqual(args[0], "bedrock/anthropic.claude-3-5-sonnet")
        self.assertEqual(kwargs["model_provider"], "bedrock_converse")

    @patch("aibom.llm_factory.init_chat_model")
    def test_azure_endpoint_routing(self, mock_init, _mock_preflight):
        """azure_openai maps api_base to azure_endpoint, passes api_version."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model(
            "gpt-4o",
            provider="azure_openai",
            api_base="https://my.openai.azure.com",
            api_key="az-key",
            api_version="2024-02-01",
        )
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs["model_provider"], "azure_openai")
        self.assertEqual(kwargs["azure_endpoint"], "https://my.openai.azure.com")
        self.assertEqual(kwargs["api_version"], "2024-02-01")
        self.assertNotIn("base_url", kwargs)

    @patch("aibom.llm_factory.init_chat_model")
    def test_base_url_for_non_azure(self, mock_init, _mock_preflight):
        """Non-azure providers get api_base mapped to base_url."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model(
            "llama3",
            provider="ollama",
            api_base="http://localhost:11434",
        )
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs["base_url"], "http://localhost:11434")
        self.assertNotIn("azure_endpoint", kwargs)

    @patch("aibom.llm_factory.init_chat_model")
    def test_api_version_ignored_for_non_azure(self, mock_init, _mock_preflight):
        """api_version is only passed for azure_openai provider."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("gpt-4o", provider="openai", api_version="2024-01-01")
        _, kwargs = mock_init.call_args
        self.assertNotIn("api_version", kwargs)

    @patch("aibom.llm_factory.init_chat_model")
    def test_temperature_default(self, mock_init, _mock_preflight):
        """Default temperature is 0.0."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("gpt-4o")
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs["temperature"], 0.0)

    @patch("aibom.llm_factory.init_chat_model")
    def test_max_tokens_passed_when_set(self, mock_init, _mock_preflight):
        """max_tokens forwarded only when explicitly provided."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("gpt-4o", max_tokens=100)
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs["max_tokens"], 100)

    @patch("aibom.llm_factory.init_chat_model")
    def test_max_tokens_omitted_when_none(self, mock_init, _mock_preflight):
        """max_tokens not in kwargs when left as None."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("gpt-4o")
        _, kwargs = mock_init.call_args
        self.assertNotIn("max_tokens", kwargs)

    @patch("aibom.llm_factory.init_chat_model")
    def test_rate_limiter_forwarded(self, mock_init, _mock_preflight):
        """rate_limiter object is passed through to init_chat_model."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        limiter = MagicMock()
        build_chat_model("gpt-4o", rate_limiter=limiter)
        _, kwargs = mock_init.call_args
        self.assertIs(kwargs["rate_limiter"], limiter)

    @patch("aibom.llm_factory.init_chat_model")
    def test_rate_limiter_omitted_when_none(self, mock_init, _mock_preflight):
        """rate_limiter not in kwargs when left as None."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("gpt-4o")
        _, kwargs = mock_init.call_args
        self.assertNotIn("rate_limiter", kwargs)

    @patch("aibom.llm_factory.init_chat_model")
    def test_bedrock_model_without_slash(self, mock_init, _mock_preflight):
        """Bedrock model ID without slash + explicit provider works."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model(
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            provider="bedrock",
        )
        args, kwargs = mock_init.call_args
        self.assertEqual(args[0], "us.anthropic.claude-sonnet-4-20250514-v1:0")
        self.assertEqual(kwargs["model_provider"], "bedrock")
        self.assertNotIn("base_url", kwargs)
        self.assertNotIn("azure_endpoint", kwargs)

    # --- Reasoning-model parameter handling ---

    @patch("aibom.llm_factory.init_chat_model")
    def test_reasoning_model_uses_max_completion_tokens(
        self, mock_init, _mock_preflight
    ):
        """Reasoning models (gpt-5.x) reject ``max_tokens`` and require
        ``max_completion_tokens``."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("gpt-5.5", max_tokens=256)
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs.get("max_completion_tokens"), 256)
        self.assertNotIn("max_tokens", kwargs)

    @patch("aibom.llm_factory.init_chat_model")
    def test_reasoning_model_omits_temperature(self, mock_init, _mock_preflight):
        """Reasoning models reject a non-default ``temperature``; omit it."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("gpt-5.5", temperature=0.0)
        _, kwargs = mock_init.call_args
        self.assertNotIn("temperature", kwargs)

    @patch("aibom.llm_factory.init_chat_model")
    def test_reasoning_model_detected_via_provider_slash(
        self, mock_init, _mock_preflight
    ):
        """Detection works on the bare model id after stripping a provider/
        prefix (e.g. ``openai/o3-mini``)."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("openai/o3-mini", max_tokens=128)
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs.get("max_completion_tokens"), 128)
        self.assertNotIn("max_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)

    @patch("aibom.llm_factory.init_chat_model")
    def test_non_reasoning_model_keeps_max_tokens_and_temperature(
        self, mock_init, _mock_preflight
    ):
        """Regression: non-reasoning models are unaffected — they keep
        ``max_tokens`` and ``temperature``."""
        from aibom.llm_factory import build_chat_model

        mock_init.return_value = MagicMock()
        build_chat_model("gpt-4o", max_tokens=256, temperature=0.2)
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs["max_tokens"], 256)
        self.assertEqual(kwargs["temperature"], 0.2)
        self.assertNotIn("max_completion_tokens", kwargs)


import pytest


def _preflight(model_string, *, provider=None):
    from aibom.llm_factory import resolve_provider

    return resolve_provider(model_string, provider)


# temperature must be sent ONLY where the provider/model accepts an
# explicit value. Determinism (0.0) is preserved everywhere it is accepted;
# omitted for OpenAI/Azure reasoning models and newer Claude (Opus 4.7+, Sonnet
# 5+, Fable 5+) on both bedrock and native anthropic. Matrix verified against
# installed langchain source (router does no kwarg filtering).
_TEMPERATURE_MATRIX = [
    # OpenAI (native + vLLM via base_url)
    ("gpt-4o", "openai", True),
    ("gpt-4o-mini", "openai", True),
    ("o1", "openai", False),
    ("o3-mini", "openai", False),
    ("o4-mini", "openai", False),
    ("gpt-5.4", "openai", False),
    ("gpt-5-chat", "openai", False),
    ("qwen2.5-coder", "openai", True),  # vLLM-served open model
    ("mistral-7b-instruct", "openai", True),
    # Azure (deployment names carrying the model family)
    ("gpt-4o", "azure_openai", True),
    ("o3", "azure_openai", False),
    ("gpt-5.3-codex", "azure_openai", False),
    # Bedrock — older Claude / Titan / Nova keep temperature
    ("us.anthropic.claude-sonnet-4-20250514-v1:0", "bedrock", True),
    ("anthropic.claude-3-5-sonnet-20241022-v2:0", "bedrock", True),
    ("amazon.titan-text-express-v1", "bedrock", True),
    ("us.amazon.nova-pro-v1:0", "bedrock", True),
    # Bedrock — newer Claude reject explicit temperature (incl. regional prefixes)
    ("us.anthropic.claude-opus-4-8", "bedrock", False),
    ("anthropic.claude-opus-4-7", "bedrock", False),
    ("eu.anthropic.claude-sonnet-5", "bedrock", False),
    ("apac.anthropic.claude-fable-5", "bedrock", False),
    # Native Anthropic
    ("claude-3-5-sonnet-20241022", "anthropic", True),
    ("claude-opus-4-1", "anthropic", True),  # 4.1 < 4.7 boundary
    ("claude-opus-4-8", "anthropic", False),
    ("claude-sonnet-5", "anthropic", False),
    ("claude-fable-5", "anthropic", False),
    # Google + Ollama accept temperature for every model
    ("gemini-2.5-pro", "google_genai", True),
    ("gemini-3-pro", "google_genai", True),
    ("llama3.1", "ollama", True),
    # Inference / slash paths with no explicit provider
    ("bedrock/us.anthropic.claude-opus-4-8", None, False),
    ("claude-opus-4-8", None, False),
]


@pytest.mark.parametrize("model_string,provider,temp_present", _TEMPERATURE_MATRIX)
@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_temperature_presence_matrix(
    mock_init, _mock_preflight, model_string, provider, temp_present
):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model(model_string, provider=provider)
    _, kwargs = mock_init.call_args
    if temp_present:
        assert kwargs.get("temperature") == 0.0, f"{model_string}/{provider}"
    else:
        assert "temperature" not in kwargs, f"{model_string}/{provider}"


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_temperature_and_max_tokens_are_orthogonal_reasoning(
    mock_init, _mock_preflight
):
    """Reasoning model: max_completion_tokens set, temperature + max_tokens
    absent — the two axes are handled independently."""
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model("o3", provider="openai", max_tokens=4096)
    _, kwargs = mock_init.call_args
    assert kwargs.get("max_completion_tokens") == 4096
    assert "temperature" not in kwargs
    assert "max_tokens" not in kwargs


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_temperature_and_max_tokens_are_orthogonal_bedrock_new_claude(
    mock_init, _mock_preflight
):
    """Newer Bedrock Claude: temperature omitted but max_tokens still sent
    under its normal name (not max_completion_tokens)."""
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model(
        "us.anthropic.claude-opus-4-8", provider="bedrock", max_tokens=4096
    )
    _, kwargs = mock_init.call_args
    assert "temperature" not in kwargs
    assert kwargs.get("max_tokens") == 4096
    assert "max_completion_tokens" not in kwargs


# --llm-reasoning off must emit the CORRECT per-provider
# thinking-disable kwarg, keyed on the resolved provider so an OpenAI-only key
# (extra_body) never leaks to a non-OpenAI provider.
@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_off_openai_vllm_uses_chat_template_kwargs(
    mock_init, _mock_preflight
):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model(
        "zai-org/GLM-5.2-FP8",
        provider="openai",
        api_base="http://localhost:8000/v1",
        reasoning="off",
    )
    _, kwargs = mock_init.call_args
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_off_openai_native_reasoning_uses_reasoning_effort(
    mock_init, _mock_preflight
):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model("gpt-5.5", provider="openai", reasoning="off")
    _, kwargs = mock_init.call_args
    assert kwargs.get("reasoning_effort") == "minimal"
    assert "extra_body" not in kwargs


# ``chat_template_kwargs`` is a vLLM / self-hosted OpenAI-compatible extension
# (signalled by a custom ``api_base``). Native OpenAI (api.openai.com) and Azure
# OpenAI reject it, and their non-reasoning models have no thinking to toggle, so
# --llm-reasoning off/on must be a no-op there — never emit ``extra_body``.
@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_off_native_openai_nonreasoning_is_noop(mock_init, _mock_preflight):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    # Native OpenAI: no custom api_base -> not a vLLM-compatible endpoint.
    build_chat_model("gpt-4o", provider="openai", reasoning="off")
    _, kwargs = mock_init.call_args
    assert "extra_body" not in kwargs
    assert "reasoning_effort" not in kwargs


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_off_azure_nonreasoning_is_noop(mock_init, _mock_preflight):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    # Azure has a custom endpoint but is NOT an OpenAI-compatible/vLLM backend.
    build_chat_model(
        "gpt-4o",
        provider="azure_openai",
        api_base="https://example.openai.azure.com",
        api_version="2024-12-01-preview",
        reasoning="off",
    )
    _, kwargs = mock_init.call_args
    assert "extra_body" not in kwargs


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_on_native_openai_nonreasoning_is_noop(mock_init, _mock_preflight):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model("gpt-4o", provider="openai", reasoning="on")
    _, kwargs = mock_init.call_args
    assert "extra_body" not in kwargs


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_on_openai_vllm_uses_chat_template_kwargs(mock_init, _mock_preflight):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model(
        "zai-org/GLM-5.2-FP8",
        provider="openai",
        api_base="http://localhost:8000/v1",
        reasoning="on",
    )
    _, kwargs = mock_init.call_args
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_off_anthropic_uses_thinking_disabled(mock_init, _mock_preflight):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model(
        "claude-3-5-sonnet-20241022", provider="anthropic", reasoning="off"
    )
    _, kwargs = mock_init.call_args
    assert kwargs["thinking"] == {"type": "disabled"}
    assert "extra_body" not in kwargs


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_off_bedrock_uses_model_kwargs_thinking(mock_init, _mock_preflight):
    # ChatBedrock uses the InvokeModel path (Anthropic body) — the thinking
    # control goes in model_kwargs; additional_model_request_fields is
    # Converse-only and the InvokeModel API rejects it (verified live on Opus 4.8).
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model(
        "us.anthropic.claude-opus-4-8", provider="bedrock", reasoning="off"
    )
    _, kwargs = mock_init.call_args
    assert kwargs["model_kwargs"] == {"thinking": {"type": "disabled"}}
    # OpenAI-only kwarg must NOT leak to bedrock, nor the Converse-only field.
    assert "extra_body" not in kwargs
    assert "additional_model_request_fields" not in kwargs


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_off_google_uses_thinking_budget(mock_init, _mock_preflight):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model("gemini-2.5-pro", provider="google_genai", reasoning="off")
    _, kwargs = mock_init.call_args
    assert kwargs.get("thinking_budget") == 0
    assert "extra_body" not in kwargs


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_off_ollama_is_noop(mock_init, _mock_preflight):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model("llama3.1", provider="ollama", reasoning="off")
    _, kwargs = mock_init.call_args
    for key in (
        "extra_body",
        "thinking",
        "additional_model_request_fields",
        "thinking_budget",
    ):
        assert key not in kwargs


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_reasoning_auto_is_unchanged(mock_init, _mock_preflight):
    """Default reasoning=auto injects nothing (no behavior change)."""
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model("zai-org/GLM-5.2-FP8", provider="openai")
    _, kwargs = mock_init.call_args
    for key in (
        "extra_body",
        "thinking",
        "additional_model_request_fields",
        "thinking_budget",
        "reasoning_effort",
    ):
        assert key not in kwargs


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_init_kwargs_extra_merged_verbatim(mock_init, _mock_preflight):
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model(
        "gpt-4o", provider="openai", init_kwargs_extra={"top_p": 0.9, "seed": 7}
    )
    _, kwargs = mock_init.call_args
    assert kwargs["top_p"] == 0.9
    assert kwargs["seed"] == 7


@patch("aibom.llm_factory.ensure_llm_runtime_available", side_effect=_preflight)
@patch("aibom.llm_factory.init_chat_model")
def test_init_kwargs_extra_overrides_reasoning(mock_init, _mock_preflight):
    """User passthrough is merged last and wins over auto-derived kwargs."""
    from aibom.llm_factory import build_chat_model

    mock_init.return_value = MagicMock()
    build_chat_model(
        "zai-org/GLM-5.2-FP8",
        provider="openai",
        api_base="http://localhost:8000/v1",
        reasoning="off",
        init_kwargs_extra={"extra_body": {"custom": 1}},
    )
    _, kwargs = mock_init.call_args
    assert kwargs["extra_body"] == {"custom": 1}


if __name__ == "__main__":
    unittest.main()
