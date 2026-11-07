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

    def test_claude_skill_skill_md(self, tmp_path: Path) -> None:
        md = "# Claude Skill\n\nDesc.\n"
        comps, _ = run_scanner(
            SkillDetector, tmp_path, {"proj/.claude/skills/my-skill/SKILL.md": md}
        )
        assert len(comps) == 1
        assert comps[0].skill_format == "claude"
        assert comps[0].component_type == AIComponentType.SKILL
        assert comps[0].name == "Claude Skill"

    def test_claude_skill_nested_skill_md_not_double_counted(
        self, tmp_path: Path
    ) -> None:
        # Claude skills are single-level (.claude/skills/<name>/SKILL.md). A
        # nested SKILL.md inside a skill's reference subdir must not register a
        # second component.
        comps, _ = run_scanner(
            SkillDetector,
            tmp_path,
            {
                "proj/.claude/skills/my-skill/SKILL.md": "# Real Skill\n\nDesc.\n",
                "proj/.claude/skills/my-skill/references/SKILL.md": "# Nested\n",
            },
        )
        assert len(comps) == 1
        assert comps[0].name == "Real Skill"

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


class TestPortableSkills:
    _SKILL = (
        "---\nname: {name}\ndescription: Use when working with {name}.\n---\n"
        "\n# {name}\n"
    )

    def test_skill_md_with_frontmatter_outside_tool_dirs(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            SkillDetector,
            tmp_path,
            {
                "skills/pdf/SKILL.md": self._SKILL.format(name="pdf"),
                "power/skills/setup/SKILL.md": self._SKILL.format(name="setup"),
            },
        )
        by_name = {c.name: c for c in comps}
        assert set(by_name) == {"pdf", "setup"}
        assert {c.skill_format for c in comps} == {"agent_skill"}
        assert all(c.line_number == 1 for c in comps)

    def test_skill_md_without_frontmatter_is_not_a_skill(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            SkillDetector, tmp_path, {"docs/SKILL.md": "# How we hire\n\nNotes.\n"}
        )
        assert comps == []

    def test_nested_and_vendored_skill_md_skipped(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            SkillDetector,
            tmp_path,
            {
                "skills/pdf/SKILL.md": self._SKILL.format(name="pdf"),
                "skills/pdf/refs/SKILL.md": self._SKILL.format(name="nested"),
                "node_modules/pkg/SKILL.md": self._SKILL.format(name="vendored"),
            },
        )
        assert [c.name for c in comps] == ["pdf"]

    def test_tool_specific_skill_reported_once(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            SkillDetector,
            tmp_path,
            {".claude/skills/review/SKILL.md": self._SKILL.format(name="review")},
        )
        assert [(c.skill_format, c.line_number) for c in comps] == [("claude", 1)]

    def test_unrecognized_skill_md_does_not_hide_skills_below_it(
        self, tmp_path: Path
    ) -> None:
        comps, _ = run_scanner(
            SkillDetector,
            tmp_path,
            {
                "SKILL.md": "# Team notes\n\nNot a skill.\n",
                "skills/pdf/SKILL.md": self._SKILL.format(name="pdf"),
            },
        )
        assert [c.name for c in comps] == ["pdf"]

    def test_plugin_skill_with_frontmatter_reported_once(self, tmp_path: Path) -> None:
        comps, _ = run_scanner(
            SkillDetector,
            tmp_path,
            {
                ".claude/plugins/review/SKILL.md": self._SKILL.format(name="review"),
                ".devin/triage/SKILL.md": self._SKILL.format(name="triage"),
            },
        )
        assert sorted((c.skill_format, c.name) for c in comps) == [
            ("claude", "review"),
            ("devin", "triage"),
        ]
        assert len({c.instance_id for c in comps}) == len(comps)
