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

import json
from pathlib import Path

import pytest

from aibom.models.enums import AIComponentType
from aibom.models.scan import ScanContext
from aibom.scanners.workflow_scanner import WorkflowScanner


@pytest.fixture
def scanner():
    return WorkflowScanner()


class TestN8nWorkflow:
    def test_detects_openai_node(self, tmp_path: Path, scanner):
        wf = {
            "nodes": [
                {
                    "type": "n8n-nodes-base.openAi",
                    "name": "GPT Node",
                    "parameters": {"model": "gpt-4o"},
                }
            ]
        }
        (tmp_path / "workflow.json").write_text(json.dumps(wf))
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].model_name == "gpt-4o"
        assert comps[0].metadata["workflow_type"] == "n8n"

    def test_detects_langchain_agent(self, tmp_path: Path, scanner):
        wf = {
            "nodes": [
                {
                    "type": "@n8n/n8n-nodes-langchain.agent",
                    "name": "AI Agent",
                    "parameters": {},
                }
            ]
        }
        (tmp_path / "agent.json").write_text(json.dumps(wf))
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].component_type == AIComponentType.AGENT

    def test_detects_vector_store_node(self, tmp_path: Path, scanner):
        wf = {
            "nodes": [
                {
                    "type": "@n8n/n8n-nodes-langchain.vectorStorePinecone",
                    "name": "Pinecone Store",
                    "parameters": {},
                }
            ]
        }
        (tmp_path / "rag.json").write_text(json.dumps(wf))
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 1
        assert comps[0].component_type == AIComponentType.VECTOR_STORE

    def test_skips_non_ai_nodes(self, tmp_path: Path, scanner):
        wf = {
            "nodes": [
                {
                    "type": "n8n-nodes-base.set",
                    "name": "Set Values",
                    "parameters": {},
                },
                {
                    "type": "n8n-nodes-base.webhook",
                    "name": "Webhook",
                    "parameters": {},
                },
            ]
        }
        (tmp_path / "basic.json").write_text(json.dumps(wf))
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 0

    def test_skips_non_n8n_json(self, tmp_path: Path, scanner):
        (tmp_path / "config.json").write_text(json.dumps({"key": "value"}))
        ctx = ScanContext(paths=[str(tmp_path)])
        comps, _ = scanner.scan(ctx)
        assert len(comps) == 0
