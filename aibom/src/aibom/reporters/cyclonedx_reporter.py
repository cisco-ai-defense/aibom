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

from typing import IO
from uuid import uuid4

from cyclonedx.model import Property
from cyclonedx.model.bom import Bom, BomMetaData
from cyclonedx.model.component import Component, ComponentType
from cyclonedx.model.tool import Tool
from cyclonedx.model.vulnerability import Vulnerability, VulnerabilitySource
from cyclonedx.output.json import JsonV1Dot6
from cyclonedx.schema import SchemaVersion
from cyclonedx.validation.json import JsonValidator

from ..models import AIComponent, AIComponentType, ScanResult
from .base import BaseReporter

_NATIVE_CDX_TYPES: frozenset[AIComponentType] = frozenset(
    {
        AIComponentType.MODEL,
        AIComponentType.LLM_ENDPOINT,
        AIComponentType.MODEL_ENDPOINT,
        AIComponentType.EMBEDDING,
        AIComponentType.MODEL_ARTIFACT,
        AIComponentType.DATASET,
        AIComponentType.VECTOR_STORE,
    }
)


def _cyclonedx_component_type(kind: AIComponentType) -> ComponentType:
    if kind in (
        AIComponentType.MODEL,
        AIComponentType.LLM_ENDPOINT,
        AIComponentType.MODEL_ENDPOINT,
        AIComponentType.EMBEDDING,
        AIComponentType.MODEL_ARTIFACT,
    ):
        return ComponentType.MACHINE_LEARNING_MODEL
    if kind in (
        AIComponentType.AGENT,
        AIComponentType.MCP_SERVER,
        AIComponentType.MCP_CLIENT,
        AIComponentType.TRAINING_RUN,
        AIComponentType.EXPERIMENT_TRACKER,
        AIComponentType.MODEL_REGISTRY,
        AIComponentType.DATA_VERSIONING,
        AIComponentType.ML_PIPELINE,
    ):
        return ComponentType.APPLICATION
    if kind in (AIComponentType.TOOL, AIComponentType.DEPENDENCY):
        return ComponentType.LIBRARY
    if kind in (AIComponentType.DATASET, AIComponentType.VECTOR_STORE):
        return ComponentType.DATA
    if kind is AIComponentType.SKILL:
        return ComponentType.FILE
    return ComponentType.APPLICATION


def _prop(name: str, value: str) -> Property:
    return Property(name=name, value=value)


def _optional_aibom_properties(
    comp: AIComponent, include_type: bool
) -> list[Property]:
    props: list[Property] = []
    if include_type:
        props.append(_prop("cisco-aibom:type", comp.component_type.value))
    if comp.framework:
        props.append(_prop("cisco-aibom:framework", comp.framework))
    if comp.file_path:
        props.append(_prop("cisco-aibom:file", comp.file_path))
    if comp.line_number:
        props.append(_prop("cisco-aibom:line", str(comp.line_number)))
    if comp.transport:
        props.append(_prop("cisco-aibom:transport", comp.transport))
    if comp.config_source:
        props.append(_prop("cisco-aibom:config_source", comp.config_source))
    if comp.storage_uri:
        props.append(_prop("cisco-aibom:storage_uri", comp.storage_uri))
    if comp.dataset_source:
        props.append(_prop("cisco-aibom:dataset_source", comp.dataset_source))
    if comp.skill_format:
        props.append(_prop("cisco-aibom:skill_format", comp.skill_format))
    return props


def _component_description(comp: AIComponent) -> str:
    if comp.description:
        return comp.description
    parts = [comp.name, comp.component_type.value]
    if comp.framework:
        parts.append(comp.framework)
    return " — ".join(parts)


def _ai_component_to_cyclonedx(comp: AIComponent) -> Component:
    include_type = comp.component_type not in _NATIVE_CDX_TYPES
    return Component(
        bom_ref=comp.instance_id,
        name=comp.name,
        type=_cyclonedx_component_type(comp.component_type),
        description=_component_description(comp),
        properties=_optional_aibom_properties(comp, include_type),
    )


def _secret_to_vulnerability(comp: AIComponent) -> Vulnerability:
    props = _optional_aibom_properties(comp, include_type=True)
    return Vulnerability(
        id=f"aibom-secret-{comp.instance_id}",
        source=VulnerabilitySource(name="cisco-aibom"),
        description=_component_description(comp),
        detail=comp.description or f"Secret reference: {comp.name}",
        properties=props,
    )


def _tool_version(meta: dict[str, object]) -> str | None:
    for key in ("analyzer_version", "version", "tool_version"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            return v
    return None


def _build_bom(result: ScanResult) -> Bom:
    components: list[Component] = []
    vulnerabilities: list[Vulnerability] = []
    for comp in result.all_components:
        if comp.component_type is AIComponentType.SECRET:
            vulnerabilities.append(_secret_to_vulnerability(comp))
        else:
            components.append(_ai_component_to_cyclonedx(comp))
    meta = result.metadata
    tool = Tool(
        vendor="Cisco",
        name="cisco-aibom",
        version=_tool_version(meta),
    )
    bom_meta = BomMetaData(tools=[tool])
    return Bom(
        serial_number=uuid4(),
        version=1,
        metadata=bom_meta,
        components=components,
        vulnerabilities=vulnerabilities,
    )


def _serialize_bom(bom: Bom) -> str:
    out = JsonV1Dot6(bom=bom)
    out.generate()
    return out.output_as_string(indent=2)


def _json_validation_errors(json_str: str) -> list[str]:
    validator = JsonValidator(SchemaVersion.V1_6)
    errs = validator.validate_str(json_str, all_errors=True)
    if errs is None:
        return []
    if isinstance(errs, (str, bytes)):
        return [str(errs)]
    return [str(e) for e in errs]


class CycloneDxReporter(BaseReporter):
    name = "cyclonedx"
    file_extension = ".cdx.json"

    def render(self, result: ScanResult, output: IO[str]) -> None:
        bom = _build_bom(result)
        output.write(_serialize_bom(bom))

    def validate(self, result: ScanResult) -> list[str]:
        try:
            bom = _build_bom(result)
            payload = _serialize_bom(bom)
        except Exception as exc:
            return [str(exc)]
        return _json_validation_errors(payload)
