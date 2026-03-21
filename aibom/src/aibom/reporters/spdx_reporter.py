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

import hashlib
import json
from datetime import datetime, timezone
from typing import IO, Any

from ..models import AIComponent, AIComponentType, ScanResult
from .base import BaseReporter

SPDX_CONTEXT = "https://spdx.org/rdf/3.0.1/spdx-context.jsonld"
CISCO_AIBOM_NS = "https://cisco.com/ns/aibom#"

_AI_PACKAGE_TYPES: frozenset[AIComponentType] = frozenset(
    {
        AIComponentType.MODEL,
        AIComponentType.EMBEDDING,
        AIComponentType.MODEL_ARTIFACT,
        AIComponentType.TRAINING_RUN,
    }
)


def _spdx_id(component_type: str, digest: str) -> str:
    return f"urn:spdx:aibom-{component_type}-{digest}"


def _digest(instance_id: str) -> str:
    return hashlib.sha256(instance_id.encode("utf-8")).hexdigest()


def _software_primary_purpose(t: AIComponentType) -> str:
    if t in (
        AIComponentType.AGENT,
        AIComponentType.MCP_SERVER,
        AIComponentType.MCP_CLIENT,
        AIComponentType.SKILL,
    ):
        return "application"
    if t in (
        AIComponentType.TOOL,
        AIComponentType.DEPENDENCY,
        AIComponentType.HYPERPARAMETER,
        AIComponentType.MODEL_REGISTRY,
        AIComponentType.EXPERIMENT_TRACKER,
        AIComponentType.ML_PIPELINE,
        AIComponentType.RETRIEVER,
    ):
        return "library"
    if t in (
        AIComponentType.VECTOR_STORE,
        AIComponentType.MEMORY,
        AIComponentType.DATA_VERSIONING,
        AIComponentType.SECRET,
    ):
        return "data"
    return "file"


def _cdx_extension(comp: AIComponent) -> dict[str, Any]:
    return {
        "type": "CdxPropertiesExtension",
        "extensionProperties": {
            "cisco-aibom:componentType": comp.component_type.value,
            "cisco-aibom:instanceId": comp.instance_id,
            "cisco-aibom:name": comp.name,
            "cisco-aibom:detectionSource": comp.detection_source.value,
        },
    }


def _ai_package(comp: AIComponent, spdx_id: str) -> dict[str, Any]:
    hyper = comp.hyperparameters or {}
    body: dict[str, Any] = {
        "type": "ai_AIPackage",
        "spdxId": spdx_id,
        "name": comp.name,
        "typeOfModel": comp.model_name or comp.name,
    }
    if hyper:
        body["hyperparameter"] = hyper
    if comp.training_info:
        body["informationAboutTraining"] = comp.training_info
    elif comp.text:
        body["informationAboutTraining"] = comp.text
    return body


def _dataset_package(comp: AIComponent, spdx_id: str) -> dict[str, Any]:
    dtype = comp.dataset_source or comp.description or comp.name
    return {
        "type": "dataset_DatasetPackage",
        "spdxId": spdx_id,
        "name": comp.name,
        "datasetType": dtype or "unknown",
    }


def _software_package(comp: AIComponent, spdx_id: str) -> dict[str, Any]:
    return {
        "type": "software_Package",
        "spdxId": spdx_id,
        "name": comp.name,
        "primaryPurpose": _software_primary_purpose(comp.component_type),
        "extension_cisco_aibom": _cdx_extension(comp),
    }


def _element_for_component(comp: AIComponent) -> dict[str, Any]:
    ct = comp.component_type.value
    sid = _spdx_id(ct, _digest(comp.instance_id))
    if comp.component_type in _AI_PACKAGE_TYPES:
        return _ai_package(comp, sid)
    if comp.component_type == AIComponentType.DATASET:
        return _dataset_package(comp, sid)
    return _software_package(comp, sid)


class SpdxReporter(BaseReporter):
    name = "spdx"
    file_extension = ".spdx.json"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        dt = datetime.now(timezone.utc).replace(microsecond=0)
        created = dt.isoformat().replace("+00:00", "Z")
        components = result.all_components
        element_ids = [
            _spdx_id(c.component_type.value, _digest(c.instance_id))
            for c in components
        ]

        doc: dict[str, Any] = {
            "type": "SpdxDocument",
            "spdxId": "urn:spdx:aibom-document",
            "name": "AIBOM Scan Report",
            "creationInfo": {
                "type": "CreationInfo",
                "created": created,
                "creators": ["Tool: cisco-aibom"],
            },
            "element": element_ids,
        }

        graph: list[dict[str, Any]] = [doc]
        graph.extend(_element_for_component(c) for c in components)

        payload: dict[str, Any] = {
            "@context": [SPDX_CONTEXT, {"cisco-aibom": CISCO_AIBOM_NS}],
            "@graph": graph,
        }
        json.dump(payload, output, indent=2)
        output.write("\n")
