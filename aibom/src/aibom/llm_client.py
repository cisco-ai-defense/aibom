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
LLM client for semantic model name extraction.

Uses LangChain's ``init_chat_model`` (available via the ``[agentic]`` extras)
to support all major providers: OpenAI, Anthropic, AWS Bedrock, Azure OpenAI,
Google Vertex AI, Ollama, Cohere, Mistral, and others.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)

_has_agentic_extras = False
try:
    from langchain.chat_models import init_chat_model as _init_chat_model  # noqa: F401
    _has_agentic_extras = True
except ImportError:
    pass


def _build_chat_model(config: Dict[str, Any]):
    """Construct a LangChain chat model from CLI-style config.

    Delegates to :func:`aibom.llm_factory.build_chat_model` for
    centralised provider routing.
    """
    from .llm_factory import build_chat_model

    return build_chat_model(
        config["model"],
        provider=config.get("provider"),
        api_key=config.get("api_key"),
        api_base=config.get("api_base"),
        api_version=config.get("api_version"),
        max_tokens=100,
    )


class LLMClient:
    """Client for interacting with LLM APIs for semantic parsing."""

    def __init__(self, config: Dict[str, Any]):
        if not _has_agentic_extras:
            raise ImportError(
                "LLM-assisted analysis requires the agentic extras. "
                "Install via: pip install 'cisco-aibom[agentic]'"
            )
        if not config.get("model"):
            raise ValueError("LLM model must be provided.")

        self.model = _build_chat_model(config)

    def invoke(self, prompt: str) -> Optional[str]:
        """Send a free-form prompt and return the text response."""
        return self._call_llm(prompt)

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
            _logger.error(f"Error extracting model name: {e}")
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
            _logger.error(f"Error extracting embedding model name: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Send a prompt to the LLM and return the text content."""
        try:
            _logger.info("Querying LLM for model extraction...")
            response = self.model.invoke(prompt)
            content = response.content.strip()  # type: ignore[union-attr]
            _logger.info(f"LLM response: {content}")
            return content

        except Exception as e:
            _logger.error(f"LLM API call failed: {e}", exc_info=True)
            return None
