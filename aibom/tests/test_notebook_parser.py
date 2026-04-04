# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

import pytest

from aibom.notebook_parser import extract_code_from_notebook
from aibom.scanners.file_cache import (
    clear_cache,
    is_python_source,
    read_python_source,
)


def _write_notebook(cells, *, tmp_dir=None):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"language": "python"}},
        "cells": cells,
    }
    kwargs = {"suffix": ".ipynb", "delete": False, "mode": "w", "encoding": "utf-8"}
    if tmp_dir:
        kwargs["dir"] = str(tmp_dir)
    tmp = tempfile.NamedTemporaryFile(**kwargs)
    json.dump(nb, tmp)
    tmp.close()
    return Path(tmp.name)


class TestNotebookParser(unittest.TestCase):

    def test_extracts_code_cells(self):
        nb_path = _write_notebook(
            [
                {"cell_type": "markdown", "source": ["# Heading"], "metadata": {}},
                {"cell_type": "code", "source": ["import os\n", "print('hi')"], "metadata": {}},
                {"cell_type": "code", "source": ["x = 1"], "metadata": {}},
            ]
        )
        code = extract_code_from_notebook(nb_path)
        self.assertIn("import os", code)
        self.assertIn("print('hi')", code)
        self.assertIn("x = 1", code)
        nb_path.unlink()

    def test_comments_out_magic_lines(self):
        nb_path = _write_notebook(
            [
                {"cell_type": "code", "source": ["%pip install foo\n", "import foo"], "metadata": {}},
                {"cell_type": "code", "source": ["!ls -la"], "metadata": {}},
            ]
        )
        code = extract_code_from_notebook(nb_path)
        self.assertIn("# MAGIC: %pip install foo", code)
        self.assertIn("# MAGIC: !ls -la", code)
        self.assertIn("import foo", code)
        nb_path.unlink()

    def test_empty_notebook(self):
        nb_path = _write_notebook([])
        code = extract_code_from_notebook(nb_path)
        self.assertEqual(code, "")
        nb_path.unlink()

    def test_nonexistent_file(self):
        code = extract_code_from_notebook("/nonexistent/file.ipynb")
        self.assertEqual(code, "")

    def test_invalid_json(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".ipynb", delete=False, mode="w", encoding="utf-8"
        )
        tmp.write("not valid json")
        tmp.close()
        code = extract_code_from_notebook(tmp.name)
        self.assertEqual(code, "")
        Path(tmp.name).unlink()


# ---------------------------------------------------------------------------
# file_cache helpers
# ---------------------------------------------------------------------------


class TestIsPythonSource(unittest.TestCase):
    def test_py(self):
        assert is_python_source(Path("foo.py"))

    def test_ipynb(self):
        assert is_python_source(Path("analysis.ipynb"))

    def test_yaml(self):
        assert not is_python_source(Path("config.yaml"))


class TestReadPythonSource(unittest.TestCase):
    def setUp(self):
        clear_cache()

    def test_reads_py_file(self):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", delete=False, mode="w", encoding="utf-8"
        )
        tmp.write("import torch\n")
        tmp.close()
        result = read_python_source(tmp.name)
        assert "import torch" in result
        Path(tmp.name).unlink()

    def test_reads_ipynb_file(self):
        nb = _write_notebook(
            [
                {"cell_type": "code", "source": ["from openai import OpenAI"], "metadata": {}},
            ]
        )
        result = read_python_source(nb)
        assert "from openai import OpenAI" in result
        nb.unlink()

    def test_caches_notebook(self):
        nb = _write_notebook(
            [
                {"cell_type": "code", "source": ["x = 42"], "metadata": {}},
            ]
        )
        r1 = read_python_source(nb)
        r2 = read_python_source(nb)
        assert r1 == r2
        assert "x = 42" in r1
        nb.unlink()


# ---------------------------------------------------------------------------
# Scanner integration: verify scanners pick up .ipynb files
# ---------------------------------------------------------------------------


class TestModelDetectorNotebook(unittest.TestCase):
    """ModelDetector should find model names inside notebook code cells."""

    def setUp(self):
        clear_cache()

    def test_detects_model_in_notebook(self):
        from aibom.scanners.model_detector import ModelDetector
        from aibom.models.scan import ScanContext

        with tempfile.TemporaryDirectory() as td:
            nb = _write_notebook(
                [
                    {
                        "cell_type": "code",
                        "source": [
                            'from openai import OpenAI\n',
                            'client = OpenAI()\n',
                            'client.chat.completions.create(model="gpt-4o")\n',
                        ],
                        "metadata": {},
                    },
                ],
                tmp_dir=td,
            )
            ctx = ScanContext(paths=[td])
            detector = ModelDetector()
            components, _errs = detector.scan(ctx)
            names = [c.name for c in components]
            assert any("gpt-4o" in n for n in names), f"Expected gpt-4o in {names}"


class TestKBScannerNotebook(unittest.TestCase):
    """KB enrichment scanner should discover .ipynb files."""

    def setUp(self):
        clear_cache()

    def test_find_python_files_includes_ipynb(self):
        from aibom.scanners.kb_enrichment_scanner import _find_python_files
        from aibom.models.scan import ScanContext

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "script.py").write_text("import os\n")
            _write_notebook(
                [{"cell_type": "code", "source": ["import torch"], "metadata": {}}],
                tmp_dir=td_path,
            )
            ctx = ScanContext(paths=[td])
            files = _find_python_files(ctx)
            suffixes = {f.suffix for f in files}
            assert ".py" in suffixes
            assert ".ipynb" in suffixes


class TestMcpDetectorNotebook(unittest.TestCase):
    """MCP detector should scan .ipynb files for MCP patterns."""

    def setUp(self):
        clear_cache()

    def test_detects_mcp_in_notebook(self):
        from aibom.scanners.mcp_detector import McpDetector
        from aibom.models.scan import ScanContext

        with tempfile.TemporaryDirectory() as td:
            nb = _write_notebook(
                [
                    {
                        "cell_type": "code",
                        "source": [
                            'from mcp.server import Server\n',
                            'server = Server("my-mcp-tool")\n',
                        ],
                        "metadata": {},
                    },
                ],
                tmp_dir=td,
            )
            ctx = ScanContext(paths=[td])
            detector = McpDetector()
            components, _errs = detector.scan(ctx)
            types = [c.component_type for c in components]
            from aibom.models.enums import AIComponentType
            assert any(
                t in (AIComponentType.MCP_SERVER, AIComponentType.MCP_CLIENT)
                for t in types
            ), f"Expected MCP component, got {types}"


if __name__ == "__main__":
    unittest.main()
