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

"""Cloud Scanner -- lightweight point-in-time probe for AI assets in cloud
environments.

Requires ``cisco-aibom[cloud]`` extras. When cloud SDKs are not installed,
the scanner gracefully skips.

Supported providers:
* AWS (SageMaker endpoints, Bedrock models)
* GCP (Vertex AI endpoints, models)
* Azure (Azure OpenAI deployments, Cognitive Services)
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import AIComponent, AIComponentType, ComponentRelationship
from ..models.enums import DetectionSource
from ..models.scan import ScanContext
from .base import BaseScanner

_LOGGER = logging.getLogger(__name__)


class CloudScanner(BaseScanner):
    name = "cloud_scanner"

    def supports(self, context: ScanContext) -> bool:
        return context.config.get("cloud_scan", False)

    def scan(
        self, context: ScanContext,
    ) -> tuple[list[AIComponent], list[ComponentRelationship]]:
        components: list[AIComponent] = []
        providers = context.config.get("cloud_providers", ["aws", "gcp", "azure"])

        if "aws" in providers:
            components.extend(_scan_aws())
        if "gcp" in providers:
            components.extend(_scan_gcp())
        if "azure" in providers:
            components.extend(_scan_azure())

        return components, []


def _scan_aws() -> list[AIComponent]:
    components: list[AIComponent] = []
    try:
        import boto3
    except ImportError:
        _LOGGER.debug("Cloud scanner: boto3 not installed, skipping AWS")
        return components

    try:
        sm = boto3.client("sagemaker")
        endpoints = sm.list_endpoints(MaxResults=100).get("Endpoints", [])
        for ep in endpoints:
            name = ep.get("EndpointName", "")
            components.append(
                AIComponent(
                    name=name,
                    component_type=AIComponentType.MODEL,
                    file_path="aws:sagemaker",
                    line_number=0,
                    model_name=name,
                    framework="sagemaker",
                    detection_source=DetectionSource.API,
                    metadata={
                        "cloud_provider": "aws",
                        "service": "sagemaker",
                        "endpoint_status": ep.get("EndpointStatus", ""),
                        "creation_time": str(ep.get("CreationTime", "")),
                    },
                )
            )
    except Exception as exc:
        _LOGGER.debug("Cloud scanner: AWS SageMaker probe failed: %s", exc)

    try:
        br = boto3.client("bedrock")
        models = br.list_foundation_models().get("modelSummaries", [])
        for m in models:
            mid = m.get("modelId", "")
            components.append(
                AIComponent(
                    name=mid,
                    component_type=AIComponentType.MODEL,
                    file_path="aws:bedrock",
                    line_number=0,
                    model_name=mid,
                    framework="bedrock",
                    detection_source=DetectionSource.API,
                    metadata={
                        "cloud_provider": "aws",
                        "service": "bedrock",
                        "provider_name": m.get("providerName", ""),
                        "model_name": m.get("modelName", ""),
                    },
                )
            )
    except Exception as exc:
        _LOGGER.debug("Cloud scanner: AWS Bedrock probe failed: %s", exc)

    return components


def _scan_gcp() -> list[AIComponent]:
    components: list[AIComponent] = []
    try:
        from google.cloud import aiplatform
    except ImportError:
        _LOGGER.debug("Cloud scanner: google-cloud-aiplatform not installed, skipping GCP")
        return components

    try:
        aiplatform.init()
        endpoints = aiplatform.Endpoint.list()
        for ep in endpoints:
            components.append(
                AIComponent(
                    name=ep.display_name,
                    component_type=AIComponentType.MODEL,
                    file_path="gcp:vertex_ai",
                    line_number=0,
                    model_name=ep.display_name,
                    framework="vertex_ai",
                    detection_source=DetectionSource.API,
                    metadata={
                        "cloud_provider": "gcp",
                        "service": "vertex_ai",
                        "resource_name": ep.resource_name,
                    },
                )
            )
    except Exception as exc:
        _LOGGER.debug("Cloud scanner: GCP Vertex AI probe failed: %s", exc)

    return components


def _scan_azure() -> list[AIComponent]:
    components: list[AIComponent] = []
    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
    except ImportError:
        _LOGGER.debug("Cloud scanner: azure packages not installed, skipping Azure")
        return components

    try:
        import os

        sub_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
        if not sub_id:
            _LOGGER.debug("Cloud scanner: AZURE_SUBSCRIPTION_ID not set, skipping Azure")
            return components

        credential = DefaultAzureCredential()
        client = CognitiveServicesManagementClient(credential, sub_id)
        accounts = client.accounts.list()
        for acct in accounts:
            if acct.kind and "openai" in acct.kind.lower():
                components.append(
                    AIComponent(
                        name=acct.name or "",
                        component_type=AIComponentType.MODEL,
                        file_path="azure:cognitive_services",
                        line_number=0,
                        model_name=acct.name or "",
                        framework="azure_openai",
                        detection_source=DetectionSource.API,
                        metadata={
                            "cloud_provider": "azure",
                            "service": "cognitive_services",
                            "kind": acct.kind or "",
                            "location": acct.location or "",
                        },
                    )
                )
    except Exception as exc:
        _LOGGER.debug("Cloud scanner: Azure probe failed: %s", exc)

    return components
