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

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from aibom.api.server import create_server


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        'id': ['1', '2', '3'],
        'name': ['OpenAI GPT', 'HuggingFace Embeddings', 'LangChain Agent'],
        'file_path': ['/app/test.py', '/app/embeddings.py', '/app/agent.py'],
        'line_number': [10, 25, 50],
        'type': ['llm_models', 'embeddings', 'agents'],
        'text': [None, 'embedding text', 'agent prompt'],
        'model_name': ['gpt-3.5-turbo', None, None],
        'embedding_model': [None, 'sentence-transformers/all-MiniLM-L6-v2', None],
        'source': ['test_source', 'test_source', 'test_source']
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_client(sample_dataframe):
    """Create a test client with sample data."""
    app = create_server(sample_dataframe)
    return TestClient(app)


def test_get_all_components(test_client):
    """Test getting all components."""
    response = test_client.get("/api/components")
    assert response.status_code == 200
    
    data = response.json()
    assert "components" in data
    assert "total" in data
    assert data["total"] == 3
    assert len(data["components"]) == 3


def test_get_component_by_id(test_client):
    """Test getting a specific component by ID."""
    response = test_client.get("/api/components/1")
    assert response.status_code == 200
    
    component = response.json()
    assert component["id"] == "1"
    assert component["name"] == "OpenAI GPT"
    assert component["file_path"] == "/app/test.py"
    assert component["line_number"] == 10
    assert component["type"] == "llm_models"


def test_get_component_not_found(test_client):
    """Test getting a non-existent component."""
    response = test_client.get("/api/components/999")
    assert response.status_code == 404
    assert response.json()["detail"] == "Component not found"


def test_get_component_types(test_client):
    """Test getting unique component types."""
    response = test_client.get("/api/components/types")
    assert response.status_code == 200
    
    data = response.json()
    assert "types" in data
    assert set(data["types"]) == {"llm_models", "embeddings", "agents"}


def test_filter_by_type(test_client):
    # Test filtering by component type
    response = test_client.get("/api/components?type=llm_models")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert data["components"][0]["type"] == "llm_models"


def test_filter_by_file_path(test_client):
    """Test filtering components by file path."""
    response = test_client.get("/api/components?file_path=embeddings")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total"] == 1
    assert len(data["components"]) == 1
    assert "embeddings.py" in data["components"][0]["file_path"]


def test_filter_by_multiple_params(test_client):
    """Test filtering by both type and file path."""
    response = test_client.get("/api/components?type=embeddings&file_path=embeddings")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total"] == 1
    assert data["components"][0]["type"] == "embeddings"
    assert "embeddings.py" in data["components"][0]["file_path"]


def test_health_check(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_components"] == 3


def test_empty_dataframe():
    """Test server with empty DataFrame."""
    empty_df = pd.DataFrame(columns=['id', 'name', 'file_path', 'line_number', 'component', 'text', 'model_name', 'embedding_model', 'source'])
    app = create_server(empty_df)
    client = TestClient(app)
    
    response = client.get("/api/components")
    assert response.status_code == 200
    
    data = response.json()
    assert data["total"] == 0
    assert len(data["components"]) == 0
