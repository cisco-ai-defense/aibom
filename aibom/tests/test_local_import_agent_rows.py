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

"""Agents are anchored at construction sites, not at imports of project code."""

from __future__ import annotations

from pathlib import Path

from aibom.cst_parser import parse_source_code
from aibom.models.enums import AIComponentType, DetectionSource
from aibom.models.scan import AIComponent
from aibom.scan_pipeline import _consolidate_components
from aibom.scanners.kb_enrichment_scanner import (
    _detect_import_based_assets,
    _imports_local_module,
)
from aibom.scanners.model_detector import _extract_python_models


def _import_agents(path: Path) -> list[AIComponent]:
    result = parse_source_code(str(path), path.read_text())
    return [
        c
        for c in _detect_import_based_assets(result)
        if c.component_type == AIComponentType.AGENT
    ]


class TestLocalImportAgentRows:
    def test_absolute_import_of_project_agent_is_not_an_agent(self, tmp_path: Path):
        (tmp_path / "agents").mkdir()
        (tmp_path / "agents" / "__init__.py").write_text("")
        (tmp_path / "agents" / "worker.py").write_text("class WorkerAgent:\n    pass\n")
        app = tmp_path / "app.py"
        app.write_text(
            "from agents.worker import WorkerAgent\n\nagent = WorkerAgent()\n"
        )
        assert _import_agents(app) == []

    def test_relative_import_of_project_agent_is_not_an_agent(self, tmp_path: Path):
        pkg = tmp_path / "ui"
        pkg.mkdir()
        (pkg / "components.py").write_text("class AgentTable:\n    pass\n")
        layout = pkg / "layout.py"
        layout.write_text("from .components import AgentTable\n")
        assert _import_agents(layout) == []

    def test_third_party_agent_import_is_still_reported(self, tmp_path: Path):
        app = tmp_path / "app.py"
        app.write_text("from somevendor.agents import PlannerAgent\n")
        agents = _import_agents(app)
        assert [c.name for c in agents] == ["PlannerAgent"]

    def test_local_module_resolution(self, tmp_path: Path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "mod.py").write_text("")
        importer = str(tmp_path / "pkg" / "user.py")
        assert _imports_local_module(importer, "from pkg.mod import X")
        assert _imports_local_module(importer, "import pkg.mod")
        assert _imports_local_module(importer, "from . import mod")
        assert not _imports_local_module(importer, "from elsewhere import X")
        assert not _imports_local_module(importer, "not python")


class TestEnvHelperModelDefault:
    def test_get_env_helper_default_is_a_model(self):
        text = 'MODEL = get_env_variable("OPENAI_MODEL", "gpt-4o-mini", False)\n'
        assert ("gpt-4o-mini", 1) in _extract_python_models(text)

    def test_get_env_helper_non_model_var_is_ignored(self):
        text = 'LEVEL = get_env_variable("LOG_LEVEL", "info")\n'
        assert _extract_python_models(text) == []


def _agent(path: str, line: int) -> AIComponent:
    return AIComponent(
        name="WorkerAgent",
        component_type=AIComponentType.AGENT,
        file_path=path,
        line_number=line,
        framework="",
        detection_source=DetectionSource.KB_ENRICHMENT,
        heuristic_confidence=0.2,
    )


class TestAgentAnchorTieBreak:
    def test_anchor_prefers_file_creating_the_agent_most(self, tmp_path: Path):
        factory = str(tmp_path / "lifecycle.py")
        loader = str(tmp_path / "loader.py")
        comps = [_agent(loader, 10), _agent(factory, 40), _agent(factory, 80)]
        (merged,) = _consolidate_components(comps)
        assert (merged.file_path, merged.line_number) == (factory, 40)
