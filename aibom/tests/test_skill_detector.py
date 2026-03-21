# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aibom.models import AIComponentType
from aibom.scanners.skill_detector import SkillDetector

from .conftest import run_scanner


class TestSkillDetector:
    def test_cursor_skill_skill_md(self, tmp_path: Path) -> None:
        md = "# My Cursor Skill\n\nBody.\n"
        comps, _ = run_scanner(
            SkillDetector, tmp_path, {"proj/.cursor/skills/my-skill/SKILL.md": md}
        )
        assert len(comps) == 1
        assert comps[0].skill_format == "cursor"
        assert comps[0].component_type == AIComponentType.SKILL

    def test_codex_skill_skill_md(self, tmp_path: Path) -> None:
        md = "# Codex Helper\n\nDesc.\n"
        comps, _ = run_scanner(
            SkillDetector, tmp_path, {"proj/.codex/skills/my-skill/SKILL.md": md}
        )
        assert len(comps) == 1
        assert comps[0].skill_format == "codex"

    def test_agents_md_format(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            SkillDetector, tmp_path, {"AGENTS.md": "# Repo Agents\n\nRules here.\n"}
        )
        assert len(comps) == 1
        assert comps[0].skill_format == "agents_md"
        assert comps[0].name == "Repo Agents"

    def test_skill_name_from_title_heading(self, tmp_path: Path) -> None:
        md = "#   Parsed Title Here\n\nMore text.\n"
        comps, _ = run_scanner(
            SkillDetector, tmp_path, {"proj/.cursor/skills/x/SKILL.md": md}
        )
        assert comps[0].name == "Parsed Title Here"

    def test_trigger_patterns_use_when(self, tmp_path: Path) -> None:
        md = "# T\n\nUse when the user asks about testing.\nActivate when debugging fails.\n"
        comps, _ = run_scanner(
            SkillDetector, tmp_path, {"proj/.codex/skills/t/SKILL.md": md}
        )
        triggers = comps[0].metadata.get("trigger_patterns") or []
        assert any("Use when" in t for t in triggers)
        assert any("Activate when" in t for t in triggers)
