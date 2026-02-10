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

from typing import Optional, Dict, Any
from pydantic import BaseModel


class ComponentModel(BaseModel):
    """Pydantic model for AI components."""
    id: Optional[str] = None
    name: str
    file_path: str
    line_number: int
    type: str  # category/type of component
    text: Optional[str] = None
    model_name: Optional[str] = None
    embedding_model: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class ComponentResponse(BaseModel):
    """Response model for component API."""
    components: list[ComponentModel]
    total: int


class ComponentTypesResponse(BaseModel):
    """Response model for component types API."""
    types: list[str]
