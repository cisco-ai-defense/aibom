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

import pandas as pd
from typing import Dict, List, Any
from ..models.component import ComponentModel


def convert_to_dataframe(all_categorized_components: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> pd.DataFrame:
    """
    Convert categorized components to a pandas DataFrame.
    
    Args:
        all_categorized_components: Dictionary with structure {source: {category: [components]}}
        
    Returns:
        pandas.DataFrame with columns: name, file_path, line_number, component
    """
    rows = []
    component_id = 1
    
    for source, categorized_components in all_categorized_components.items():
        for category, components in categorized_components.items():
            for component in components:
                row = {
                    'id': str(component_id),
                    'name': component.get('name', ''),
                    'file_path': component.get('file_path', ''),
                    'line_number': component.get('line_number', 0),
                    'type': category,
                    'text': component.get('text', None),
                    'model_name': component.get('model_name', None),
                    'embedding_model': component.get('embedding_model', None),
                    'source': source
                }
                rows.append(row)
                component_id += 1
    
    return pd.DataFrame(rows)


def dataframe_to_components(df: pd.DataFrame) -> List[ComponentModel]:
    """
    Convert DataFrame rows to ComponentModel instances.
    
    Args:
        df: pandas DataFrame with component data
        
    Returns:
        List of ComponentModel instances
    """
    # Convert DataFrame to list of dictionaries (much simpler!)
    records = df.to_dict('records')
    
    components = []
    for record in records:
        # Build additional_data from non-null fields
        additional_data = {}
        for field in ['text', 'model_name', 'embedding_model', 'source']:
            if pd.notna(record.get(field)):
                additional_data[field] = record[field]
        
        component = ComponentModel(
            id=str(record.get('id', '')),
            name=str(record.get('name', '')),
            file_path=str(record.get('file_path', '')),
            line_number=int(record.get('line_number', 0)),
            type=str(record.get('type', '')),
            text=record.get('text') if pd.notna(record.get('text')) else None,
            model_name=record.get('model_name') if pd.notna(record.get('model_name')) else None,
            embedding_model=record.get('embedding_model') if pd.notna(record.get('embedding_model')) else None,
            additional_data=additional_data if additional_data else None
        )
        components.append(component)
    
    return components


def dataframe_to_json(df: pd.DataFrame) -> str:
    """
    Convert DataFrame directly to JSON string.
    
    Args:
        df: pandas DataFrame with component data
        
    Returns:
        JSON string representation of the DataFrame
    """
    # Option 1: Records format (list of objects) - most common for APIs
    return df.to_json(orient='records', indent=2)

    # Other useful orientations:
    # orient='index' - {index: {column: value}}
    # orient='values' - [[row1_values], [row2_values]]
    # orient='split' - {index: [...], columns: [...], data: [[...]]}
    # orient='table' - JSON Table Schema format
