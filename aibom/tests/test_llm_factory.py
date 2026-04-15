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
        __import__("aibom.llm_factory", fromlist=["resolve_provider"])
        .resolve_provider(model_string, provider)
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


if __name__ == "__main__":
    unittest.main()
