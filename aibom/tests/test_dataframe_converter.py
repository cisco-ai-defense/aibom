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
from aibom.utils.dataframe_converter import (
    convert_to_dataframe,
    dataframe_to_components,
    dataframe_to_json,
)
from aibom.models.component import ComponentModel


def test_convert_to_dataframe():
    """Test conversion of categorized components to DataFrame."""
    # Sample categorized components data
    all_categorized_components = {
        "test_source": {
            "llm_models": [
                {
                    "name": "OpenAI GPT",
                    "file_path": "/app/test.py",
                    "line_number": 10,
                    "model_name": "gpt-3.5-turbo"
                }
            ],
            "embeddings": [
                {
                    "name": "HuggingFace Embeddings",
                    "file_path": "/app/embeddings.py",
                    "line_number": 25,
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            ]
        }
    }
    
    df = convert_to_dataframe(all_categorized_components)
    
    # Check DataFrame structure
    assert not df.empty
    assert len(df) == 2
    assert list(df.columns) == ['id', 'name', 'file_path', 'line_number', 'type', 'text', 'model_name', 'embedding_model', 'source']
    
    # Check first row (LLM model)
    llm_row = df[df['type'] == 'llm_models'].iloc[0]
    assert llm_row['name'] == "OpenAI GPT"
    assert llm_row['file_path'] == "/app/test.py"
    assert llm_row['line_number'] == 10
    assert llm_row['model_name'] == "gpt-3.5-turbo"
    
    # Check second row (embeddings)
    embed_row = df[df['type'] == 'embeddings'].iloc[0]
    assert embed_row['name'] == "HuggingFace Embeddings"
    assert embed_row['file_path'] == "/app/embeddings.py"
    assert embed_row['line_number'] == 25
    assert embed_row['embedding_model'] == "sentence-transformers/all-MiniLM-L6-v2"


def test_dataframe_to_components():
    """Test conversion of DataFrame to ComponentModel instances."""
    # Create test DataFrame
    data = {
        'id': ['1', '2'],
        'name': ['Test Component 1', 'Test Component 2'],
        'file_path': ['/app/test1.py', '/app/test2.py'],
        'line_number': [10, 20],
        'type': ['llm_models', 'embeddings'],
        'text': [None, 'test text'],
        'model_name': ['gpt-3.5-turbo', None],
        'embedding_model': [None, 'test-embedding'],
        'source': ['test_source', 'test_source']
    }
    df = pd.DataFrame(data)
    
    components = dataframe_to_components(df)
    
    # Check results
    assert len(components) == 2
    assert all(isinstance(comp, ComponentModel) for comp in components)
    
    # Check first component
    comp1 = components[0]
    assert comp1.id == '1'
    assert comp1.name == 'Test Component 1'
    assert comp1.file_path == '/app/test1.py'
    assert comp1.line_number == 10
    assert comp1.type == 'llm_models'
    assert comp1.model_name == 'gpt-3.5-turbo'
    assert comp1.text is None
    
    # Check second component
    comp2 = components[1]
    assert comp2.id == '2'
    assert comp2.name == 'Test Component 2'
    assert comp2.file_path == '/app/test2.py'
    assert comp2.line_number == 20
    assert comp2.type == 'embeddings'
    assert comp2.text == 'test text'
    assert comp2.embedding_model == 'test-embedding'


def test_empty_components():
    """Test handling of empty components."""
    empty_components = {}
    df = convert_to_dataframe(empty_components)
    
    assert df.empty
    assert len(df) == 0
    
    components = dataframe_to_components(df)
    assert len(components) == 0


def test_dataframe_to_json():
    """Test conversion of DataFrame to JSON string."""
    data = {
        'id': ['1'],
        'name': ['Test Component'],
        'file_path': ['/app/test.py'],
        'line_number': [10],
        'type': ['llm_models']
    }
    df = pd.DataFrame(data)
    
    json_output = dataframe_to_json(df)
    
    # Check that the output is a valid JSON string representing a list with one object
    import json
    parsed_json = json.loads(json_output)
    assert isinstance(parsed_json, list)
    assert len(parsed_json) == 1
    assert parsed_json[0]['name'] == 'Test Component'
