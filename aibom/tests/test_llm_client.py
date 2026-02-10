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

from aibom.llm_client import LLMClient

class TestLLMClient(unittest.TestCase):

    def setUp(self):
        self.config = {
            "model": "gpt-test",
            "api_key": "test_key",
            "api_base": "https://test.api.base"
        }
        self.client = LLMClient(self.config)

    def test_init_success(self):
        self.assertEqual(self.client.model, "gpt-test")
        self.assertEqual(self.client.api_key, "test_key")

    def test_init_no_model_failure(self):
        with self.assertRaises(ValueError):
            LLMClient({"api_key": "some_key"})

    @patch('litellm.completion')
    def test_extract_model_name_success(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = 'gpt-4'
        mock_completion.return_value = mock_response

        model_name = self.client.extract_model_name("code_snippet", "ClassName")
        
        self.assertEqual(model_name, 'gpt-4')
        mock_completion.assert_called_once()
        prompt = mock_completion.call_args[1]['messages'][0]['content']
        self.assertIn('Extract the model name', prompt)
        self.assertIn('class called \"ClassName\"', prompt)
        self.assertIn('code_snippet', prompt)

    @patch('litellm.completion')
    def test_extract_model_name_returns_none(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = 'NONE'
        mock_completion.return_value = mock_response

        model_name = self.client.extract_model_name("code_snippet", "ClassName")
        
        self.assertIsNone(model_name)

    @patch('litellm.completion')
    def test_extract_embedding_model_success(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = 'text-embedding-ada-002'
        mock_completion.return_value = mock_response

        model_name = self.client.extract_embedding_model("code_snippet", "EmbeddingClass")
        
        self.assertEqual(model_name, 'text-embedding-ada-002')
        prompt = mock_completion.call_args[1]['messages'][0]['content']
        self.assertIn('Extract the embedding model name', prompt)
        self.assertIn('embedding class called \"EmbeddingClass\"', prompt)

    @patch('litellm.completion', side_effect=Exception("API Error"))
    def test_llm_call_failure(self, mock_completion):
        model_name = self.client.extract_model_name("code_snippet", "ClassName")
        self.assertIsNone(model_name)

if __name__ == '__main__':
    unittest.main()
