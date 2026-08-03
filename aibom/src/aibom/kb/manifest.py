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

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class KBManifest(BaseModel):
    kb_version: str
    schema_version: int | str | None = None
    vocabulary_version: str = ""
    freshness_api: str = ""
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


class KBArtifact(BaseModel):
    """One immutable object referenced by a schema-v2 manifest."""

    model_config = ConfigDict(extra="forbid", strict=True)

    url: str
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KBSignature(BaseModel):
    """Detached signature metadata for the compressed DuckDB object."""

    model_config = ConfigDict(extra="forbid", strict=True)

    algorithm: Literal["ECDSA_SHA_256", "disabled"]
    url: str
    public_key_url: str
    key_id: str

    @model_validator(mode="after")
    def validate_signing_fields(self) -> "KBSignature":
        if self.algorithm == "disabled":
            if self.key_id != "disabled" or self.public_key_url:
                raise ValueError(
                    "disabled signatures require key_id=disabled and no public key URL"
                )
        elif not self.key_id.strip() or not self.public_key_url.strip():
            raise ValueError(
                "signed artifacts require a key identity and public key URL"
            )
        return self


class KBSourceCandidate(BaseModel):
    """Traceability link from a published artifact to its validated candidate."""

    model_config = ConfigDict(extra="forbid", strict=True)

    build_id: str = Field(min_length=1, max_length=128)
    validation_report_url: str


class KBManifestV2(BaseModel):
    """Strict schema-v2 distribution contract emitted by the publisher."""

    model_config = ConfigDict(extra="forbid", strict=True)

    artifact_state: Literal["authoritative", "rehearsal"]
    authoritative: bool
    schema_version: int = Field(ge=2, le=2)
    vocabulary_version: str = Field(min_length=1)
    min_cli_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+].+)?$")
    kb_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    build_type: Literal["floor", "delta"]
    parent_kb_version: str = ""
    generated_at: str = Field(
        pattern=(
            r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:"
            r"[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
        )
    )
    duckdb: KBArtifact
    signature: KBSignature
    freshness_api: str = ""
    has_enrichment: bool
    contents: dict[str, Any]
    provenance: dict[str, str]
    source_candidate: KBSourceCandidate

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, value: dict[str, str]) -> dict[str, str]:
        required = {
            "source_commit",
            "pipeline_version",
            "config_version",
            "model_version",
            "tools_version",
            "input_fingerprint",
        }
        missing = sorted(k for k in required if not value.get(k, "").strip())
        if missing:
            raise ValueError(f"missing provenance fields: {', '.join(missing)}")
        return value

    @model_validator(mode="after")
    def validate_build_relationship(self) -> "KBManifestV2":
        if self.artifact_state == "authoritative" and not self.authoritative:
            raise ValueError("authoritative artifacts require authoritative=true")
        if self.artifact_state == "rehearsal" and self.authoritative:
            raise ValueError("rehearsal artifacts require authoritative=false")
        if self.build_type == "floor" and self.parent_kb_version:
            raise ValueError("floor builds must not declare parent_kb_version")
        if self.build_type == "delta" and not self.parent_kb_version.strip():
            raise ValueError("delta builds require parent_kb_version")
        if not self.contents:
            raise ValueError("contents must not be empty")
        return self
