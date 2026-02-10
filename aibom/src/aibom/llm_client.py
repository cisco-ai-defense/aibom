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

"""
LLM client for semantic model name extraction using litellm.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Any, Optional

# Set litellm log level to WARNING via environment variable
os.environ["LITELLM_LOG"] = "WARNING"
try:
    import litellm
except ImportError:  # pragma: no cover
    litellm = None  # type: ignore


class LLMClient:
    """Client for interacting with LLM APIs for semantic parsing using litellm."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM client with configuration."""
        if litellm is None:
            raise ImportError(
                "The 'litellm' package is required for LLM-assisted analysis. "
                "Install it via 'pip install litellm'."
            )
        self.model = config.get("model")
        self.api_key = config.get("api_key")
        self.api_base = config.get("api_base")
        self.api_version = config.get("api_version")
        
        if not self.model:
            raise ValueError("LLM model must be provided.")
    
    def extract_model_name(self, code_snippet: str, class_name: str) -> Optional[str]:
        """
        Extract model name from a code snippet containing class instantiation.
        
        Args:
            code_snippet: The code snippet containing the class instantiation
            class_name: The name of the class being instantiated
            
        Returns:
            The extracted model name or None if not found
        """
        prompt = f"""
You are a code analysis expert. Extract the model name from the following Python code snippet.

The code instantiates a class called "{class_name}". Look for parameters that specify the model name.
Common parameter names include: "model", "model_name", "model_id", "name".

Code snippet:
```python
{code_snippet}
```

Return ONLY the model name as a string (without quotes), or "NONE" if no model name is found.
Examples:
- If you see model="gpt-3.5-turbo", return: gpt-3.5-turbo
- If you see model_name="claude-3-sonnet", return: claude-3-sonnet
- If no model parameter is found, return: NONE
"""
        
        try:
            response = self._call_llm(prompt)
            if response and response.strip().upper() != "NONE":
                return response.strip().strip('"\'')
            return None
        except Exception as e:
            logging.error(f"Error extracting model name: {e}")
            return None
    
    def extract_embedding_model(self, code_snippet: str, class_name: str) -> Optional[str]:
        """
        Extract embedding model name from a code snippet.
        
        Args:
            code_snippet: The code snippet containing the embedding class instantiation
            class_name: The name of the embedding class being instantiated
            
        Returns:
            The extracted embedding model name or None if not found
        """
        prompt = f"""
You are a code analysis expert. Extract the embedding model name from the following Python code snippet.

The code instantiates an embedding class called "{class_name}". Look for parameters that specify the embedding model name.
Common parameter names include: "model", "model_name", "model_id", "name".

Code snippet:
```python
{code_snippet}
```

Return ONLY the embedding model name as a string (without quotes), or "NONE" if no model name is found.
Examples:
- If you see model="text-embedding-ada-002", return: text-embedding-ada-002
- If you see model_name="sentence-transformers/all-MiniLM-L6-v2", return: sentence-transformers/all-MiniLM-L6-v2
- If no model parameter is found, return: NONE
"""
        
        try:
            response = self._call_llm(prompt)
            if response and response.strip().upper() != "NONE":
                return response.strip().strip('"\'')
            return None
        except Exception as e:
            logging.error(f"Error extracting embedding model name: {e}")
            return None
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Make a call to the LLM API using litellm.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response or None if failed
        """
        try:
            logging.info(f"Querying LLM for model extraction (model: {self.model})...")
            response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.api_key,
                base_url=self.api_base,
                api_version=self.api_version,
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content.strip()
            logging.info(f"LLM response: {content}")
            return content
            
        except Exception as e:
            logging.error(f"LLM API call failed: {e}", exc_info=True)
            return None
