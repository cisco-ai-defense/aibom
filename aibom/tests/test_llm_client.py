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
from unittest.mock import patch, MagicMock

from aibom.llm_client import LLMClient, _build_chat_model


@patch("aibom.llm_client._has_agentic_extras", True)
class TestLLMClient(unittest.TestCase):

    @patch("aibom.llm_client._has_agentic_extras", True)
    @patch("aibom.llm_factory.init_chat_model")
    def setUp(self, mock_init):
        self.mock_model = MagicMock()
        mock_init.return_value = self.mock_model
        self.config = {
            "model": "gpt-test",
            "api_key": "test_key",
            "api_base": "https://test.api.base",
        }
        self.client = LLMClient(self.config)

    def test_init_no_model_failure(self):
        with self.assertRaises(ValueError):
            with patch("aibom.llm_factory.init_chat_model", MagicMock()):
                LLMClient({"api_key": "some_key"})

    def test_init_no_agentic_extras(self):
        with patch("aibom.llm_client._has_agentic_extras", False):
            with self.assertRaises(ImportError) as ctx:
                LLMClient({"model": "gpt-4o", "api_key": "k"})
            self.assertIn("agentic extras", str(ctx.exception))

    def test_extract_model_name_success(self):
        mock_response = MagicMock()
        mock_response.content = "gpt-4"
        self.mock_model.invoke.return_value = mock_response

        model_name = self.client.extract_model_name("code_snippet", "ClassName")

        self.assertEqual(model_name, "gpt-4")
        self.mock_model.invoke.assert_called_once()
        prompt = self.mock_model.invoke.call_args[0][0]
        self.assertIn("Extract the model name", prompt)
        self.assertIn('class called "ClassName"', prompt)

    def test_extract_model_name_returns_none(self):
        mock_response = MagicMock()
        mock_response.content = "NONE"
        self.mock_model.invoke.return_value = mock_response

        model_name = self.client.extract_model_name("code_snippet", "ClassName")

        self.assertIsNone(model_name)

    def test_extract_embedding_model_success(self):
        mock_response = MagicMock()
        mock_response.content = "text-embedding-ada-002"
        self.mock_model.invoke.return_value = mock_response

        model_name = self.client.extract_embedding_model("code_snippet", "EmbeddingClass")

        self.assertEqual(model_name, "text-embedding-ada-002")
        prompt = self.mock_model.invoke.call_args[0][0]
        self.assertIn("Extract the embedding model name", prompt)
        self.assertIn('embedding class called "EmbeddingClass"', prompt)

    def test_llm_call_failure(self):
        self.mock_model.invoke.side_effect = Exception("API Error")

        model_name = self.client.extract_model_name("code_snippet", "ClassName")
        self.assertIsNone(model_name)

    def test_invoke_method(self):
        mock_response = MagicMock()
        mock_response.content = "tool description here"
        self.mock_model.invoke.return_value = mock_response

        result = self.client.invoke("describe this tool")
        self.assertEqual(result, "tool description here")


@patch(
    "aibom.llm_factory.ensure_llm_runtime_available",
    side_effect=lambda model_string, *, provider=None: (
        __import__("aibom.llm_factory", fromlist=["resolve_provider"])
        .resolve_provider(model_string, provider)
    ),
)
class TestBuildChatModel(unittest.TestCase):
    """Tests that _build_chat_model delegates correctly to llm_factory."""

    @patch("aibom.llm_factory.init_chat_model")
    def test_plain_model(self, mock_init, _mock_preflight):
        mock_init.return_value = MagicMock()
        _build_chat_model({"model": "gpt-4o", "api_key": "k"})
        _, kwargs = mock_init.call_args
        self.assertNotIn("model_provider", kwargs)
        self.assertEqual(kwargs["api_key"], "k")

    @patch("aibom.llm_factory.init_chat_model")
    def test_provider_prefix_bedrock(self, mock_init, _mock_preflight):
        mock_init.return_value = MagicMock()
        _build_chat_model({"model": "bedrock/anthropic.claude-3-5-sonnet"})
        args, kwargs = mock_init.call_args
        self.assertEqual(args[0], "anthropic.claude-3-5-sonnet")
        self.assertEqual(kwargs["model_provider"], "bedrock")

    @patch("aibom.llm_factory.init_chat_model")
    def test_explicit_provider_in_config(self, mock_init, _mock_preflight):
        """provider key in config is forwarded to build_chat_model."""
        mock_init.return_value = MagicMock()
        _build_chat_model({
            "model": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "provider": "bedrock",
        })
        args, kwargs = mock_init.call_args
        self.assertEqual(args[0], "us.anthropic.claude-sonnet-4-20250514-v1:0")
        self.assertEqual(kwargs["model_provider"], "bedrock")

    @patch("aibom.llm_factory.init_chat_model")
    def test_azure_endpoint_routing(self, mock_init, _mock_preflight):
        mock_init.return_value = MagicMock()
        _build_chat_model({
            "model": "gpt-4o",
            "provider": "azure_openai",
            "api_base": "https://my.openai.azure.com",
            "api_key": "az-key",
            "api_version": "2024-02-01",
        })
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs["model_provider"], "azure_openai")
        self.assertEqual(kwargs["azure_endpoint"], "https://my.openai.azure.com")
        self.assertEqual(kwargs["api_version"], "2024-02-01")
        self.assertNotIn("base_url", kwargs)

    @patch("aibom.llm_factory.init_chat_model")
    def test_base_url_passthrough(self, mock_init, _mock_preflight):
        mock_init.return_value = MagicMock()
        _build_chat_model({
            "model": "llama3",
            "provider": "ollama",
            "api_base": "http://localhost:11434",
        })
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs["base_url"], "http://localhost:11434")

    @patch("aibom.llm_factory.init_chat_model")
    def test_max_tokens_100(self, mock_init, _mock_preflight):
        """llm_client sets max_tokens=100 for short extraction prompts."""
        mock_init.return_value = MagicMock()
        _build_chat_model({"model": "gpt-4o"})
        _, kwargs = mock_init.call_args
        self.assertEqual(kwargs["max_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
