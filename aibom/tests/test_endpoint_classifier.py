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

"""Tests for endpoint_classifier.classify_endpoint."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aibom.scanners.endpoint_classifier import classify_endpoint


@pytest.fixture(autouse=True)
def _mock_registry(monkeypatch):
    """Prevent real HTTP calls to model registries during tests."""

    def fake_lookup(model_id: str):
        known = {
            "text-embedding-ada-002": {"family": "embedding", "provider": "openai"},
            "all-MiniLM-L6-v2": {"pipeline_tag": "sentence-similarity"},
            "gpt-4o": {"family": "gpt-4", "provider": "openai"},
            "claude-sonnet-4-20250514": {"family": "claude", "provider": "anthropic"},
        }
        return known.get(model_id)

    monkeypatch.setattr(
        "aibom.scanners.endpoint_classifier.registry_lookup", fake_lookup
    )


class TestVectorStoreURLs:
    def test_weaviate(self):
        assert classify_endpoint("https://my-weaviate.example.com/v1") == "vector_store"

    def test_pinecone(self):
        assert classify_endpoint("https://index-xyz.svc.pinecone.io") == "vector_store"

    def test_qdrant(self):
        assert classify_endpoint("https://qdrant.internal:6333") == "vector_store"

    def test_chroma(self):
        assert classify_endpoint("http://chroma-server:8000") == "vector_store"

    def test_milvus(self):
        assert classify_endpoint("http://milvus.svc.cluster.local:19530") == "vector_store"


class TestPairedModel:
    def test_cloud_url_wins_over_embedding_paired_model(self):
        result = classify_endpoint(
            "https://my-endpoint.openai.azure.com/",
            paired_model="text-embedding-ada-002",
        )
        assert result == "llm_endpoint"

    def test_embedding_model_on_non_cloud_url(self):
        result = classify_endpoint(
            "https://custom-server.example.com/v1",
            paired_model="text-embedding-ada-002",
        )
        assert result == "model_endpoint"

    def test_llm_model_returns_llm_endpoint(self):
        result = classify_endpoint(
            "https://some-server.example.com/v1",
            paired_model="gpt-4o",
        )
        assert result == "llm_endpoint"

    def test_unknown_model_falls_through(self):
        result = classify_endpoint(
            "https://some-server.example.com/v1",
            paired_model="totally-unknown-model-xyz",
        )
        assert result == "model_endpoint"


class TestContextKey:
    def test_embedding_key(self):
        result = classify_endpoint(
            "https://custom-server.example.com/v1",
            context_key="services.embedding_endpoint",
        )
        assert result == "model_endpoint"

    def test_llm_key(self):
        result = classify_endpoint(
            "https://custom-server.example.com/v1",
            context_key="services.chat_completion_endpoint",
        )
        assert result == "llm_endpoint"

    def test_vector_key(self):
        result = classify_endpoint(
            "https://custom-server.example.com/v1",
            context_key="services.vector_store_url",
        )
        assert result == "model_endpoint"


class TestCloudProviders:
    def test_azure_openai(self):
        assert classify_endpoint("https://my-resource.openai.azure.com/openai/deployments/gpt4") == "llm_endpoint"

    def test_openai(self):
        assert classify_endpoint("https://api.openai.com/v1/chat/completions") == "llm_endpoint"

    def test_anthropic(self):
        assert classify_endpoint("https://api.anthropic.com/v1/messages") == "llm_endpoint"

    def test_aws_bedrock(self):
        assert classify_endpoint("https://bedrock-runtime.us-east-1.amazonaws.com/model/invoke") == "llm_endpoint"

    def test_aws_bedrock_fips(self):
        assert classify_endpoint("https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com/model/invoke") == "llm_endpoint"

    def test_aws_bedrock_china(self):
        assert classify_endpoint("https://bedrock-runtime.cn-north-1.amazonaws.com.cn/model/invoke") == "llm_endpoint"

    def test_google_vertex(self):
        assert classify_endpoint("https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/l/publishers/google/models/gemini-pro") == "llm_endpoint"

    def test_groq(self):
        assert classify_endpoint("https://api.groq.com/openai/v1/chat/completions") == "llm_endpoint"


class TestFallback:
    def test_vllm_defaults_to_model_endpoint(self):
        assert classify_endpoint("http://vllm-server.internal:8000/v1") == "model_endpoint"

    def test_unknown_url_defaults_to_model_endpoint(self):
        assert classify_endpoint("https://custom-inference.example.com/predict") == "model_endpoint"


class TestSignalPriority:
    def test_vector_store_beats_cloud_provider(self):
        result = classify_endpoint("https://weaviate.openai.azure.com/v1")
        assert result == "vector_store"

    def test_cloud_url_beats_embedding_context_key(self):
        result = classify_endpoint(
            "https://my-resource.openai.azure.com/",
            context_key="dataVectorizer.env.AZURE.ENDPOINT",
        )
        assert result == "llm_endpoint"

    def test_cloud_url_beats_paired_embedding_model(self):
        result = classify_endpoint(
            "https://my-resource.openai.azure.com/",
            paired_model="text-embedding-ada-002",
        )
        assert result == "llm_endpoint"

    def test_paired_model_beats_context_key_on_non_cloud(self):
        result = classify_endpoint(
            "https://custom.example.com/v1",
            context_key="services.embedding_url",
            paired_model="gpt-4o",
        )
        assert result == "llm_endpoint"
