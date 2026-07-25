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

from __future__ import annotations

from pydantic import BaseModel, Field


class KBManifest(BaseModel):
    kb_version: str
    schema_version: int | str | None = None
    vocabulary_version: str = ""
    min_cli_version: str = ""
    duckdb_sha256: str
    duckdb_url: str
    size_bytes: int = 0
    entity_count: int = 0
    created_at: str = ""
    sdk_versions: dict[str, str] = Field(default_factory=dict)


class KBManifestIndex(BaseModel):
    latest: KBManifest
    versions: list[KBManifest] = Field(default_factory=list)
