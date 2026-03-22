# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from aibom.models import ScanContext
from aibom.models.enums import AIComponentType
from aibom.scanners.env_var_resolver import EnvVarResolver


def _make_ctx(tmp_path: Path) -> ScanContext:
    return ScanContext(paths=[str(tmp_path)])


class TestPythonEnvVarExtraction:
    def test_os_getenv_in_model_kwarg(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(
            'from openai import OpenAI\nclient = OpenAI()\n'
            'resp = client.chat.completions.create(\n'
            '    model=os.getenv("LLM_MODEL"),\n'
            '    messages=[]\n'
            ')\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].name == "env:LLM_MODEL"
        assert comps[0].metadata["env"] == "LLM_MODEL"
        assert comps[0].metadata["env_context"] == "model_kwarg"
        assert comps[0].needs_agentic is True

    def test_os_environ_bracket_in_api_key(self, tmp_path: Path) -> None:
        (tmp_path / "config.py").write_text(
            'client = Client(api_key=os.environ["OPENAI_KEY"])\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env"] == "OPENAI_KEY"
        assert comps[0].metadata["env_context"] == "api_key_kwarg"
        assert comps[0].component_type == AIComponentType.SECRET

    def test_os_environ_get_in_endpoint(self, tmp_path: Path) -> None:
        (tmp_path / "svc.py").write_text(
            'client = AzureOpenAI(azure_endpoint=os.environ.get("AZURE_ENDPOINT"))\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env"] == "AZURE_ENDPOINT"
        assert comps[0].metadata["env_context"] == "endpoint_kwarg"

    def test_model_name_assignment(self, tmp_path: Path) -> None:
        (tmp_path / "run.py").write_text(
            'model_name = os.getenv("MODEL_ID")\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env_context"] == "model_kwarg"

    def test_generic_var_model_matches_kwarg(self, tmp_path: Path) -> None:
        """'model' is in both kwarg set and assignment set; kwarg wins."""
        (tmp_path / "run.py").write_text(
            'model = os.getenv("CHOSEN_MODEL")\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env_context"] == "model_kwarg"

    def test_llm_model_assignment(self, tmp_path: Path) -> None:
        """Variable 'llm_model' not in kwarg set => falls through to assignment."""
        (tmp_path / "run.py").write_text(
            'llm_model = os.getenv("MY_LLM")\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env_context"] == "model_assignment"

    def test_ignores_unrelated_getenv(self, tmp_path: Path) -> None:
        (tmp_path / "unrelated.py").write_text(
            'log_level = os.getenv("LOG_LEVEL")\n'
            'home = os.environ["HOME"]\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 0

    def test_dedup_same_env_same_context(self, tmp_path: Path) -> None:
        (tmp_path / "dup.py").write_text(
            'a = client.create(model=os.getenv("LLM_MODEL"))\n'
            'b = client.create(model=os.getenv("LLM_MODEL"))\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1


class TestJavaScriptEnvVarExtraction:
    def test_process_env_dot(self, tmp_path: Path) -> None:
        (tmp_path / "app.ts").write_text(
            'const client = new OpenAI({ model: process.env.OPENAI_MODEL });\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env"] == "OPENAI_MODEL"

    def test_process_env_bracket(self, tmp_path: Path) -> None:
        (tmp_path / "svc.js").write_text(
            'const key = process.env["API_KEY"];\n'
            'client.init({ api_key: key });\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 0  # not directly in kwarg context


class TestGoEnvVarExtraction:
    def test_os_getenv_in_model_context(self, tmp_path: Path) -> None:
        (tmp_path / "main.go").write_text(
            'func run() {\n'
            '    model = os.Getenv("LLM_MODEL")\n'
            '}\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env"] == "LLM_MODEL"


class TestJavaEnvVarExtraction:
    def test_system_getenv(self, tmp_path: Path) -> None:
        (tmp_path / "App.java").write_text(
            'String model = System.getenv("AI_MODEL");\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env"] == "AI_MODEL"


class TestRubyEnvVarExtraction:
    def test_env_bracket(self, tmp_path: Path) -> None:
        (tmp_path / "app.rb").write_text(
            'model_name = ENV["RUBY_MODEL"]\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env"] == "RUBY_MODEL"

    def test_env_fetch(self, tmp_path: Path) -> None:
        (tmp_path / "config.rb").write_text(
            'model = ENV.fetch("LLM_MODEL")\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1


class TestCommentSkipping:
    def test_python_comment_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "app.py").write_text(
            '# model=os.getenv("COMMENTED_MODEL")\n'
            'real = client.create(model=os.getenv("REAL_MODEL"))\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env"] == "REAL_MODEL"

    def test_js_comment_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "app.ts").write_text(
            '// const m = { model: process.env.COMMENTED };\n'
            'const real = { model: process.env.ACTUAL_MODEL };\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env"] == "ACTUAL_MODEL"

    def test_indented_comment_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "svc.py").write_text(
            '    # api_key=os.getenv("OLD_KEY")\n'
            '    api_key=os.getenv("NEW_KEY")\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 1
        assert comps[0].metadata["env"] == "NEW_KEY"


class TestUnsupportedLanguage:
    def test_txt_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text(
            'model=os.getenv("SECRET_MODEL")\n'
        )
        scanner = EnvVarResolver()
        comps, _ = scanner.scan(_make_ctx(tmp_path))
        assert len(comps) == 0
