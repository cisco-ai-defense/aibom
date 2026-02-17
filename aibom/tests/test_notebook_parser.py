# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from aibom.notebook_parser import extract_code_from_notebook


class TestNotebookParser(unittest.TestCase):

    def _write_notebook(self, cells):
        """Helper to create a temporary .ipynb file."""
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {"kernelspec": {"language": "python"}},
            "cells": cells,
        }
        tmp = tempfile.NamedTemporaryFile(
            suffix=".ipynb", delete=False, mode="w", encoding="utf-8"
        )
        json.dump(nb, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_extracts_code_cells(self):
        nb_path = self._write_notebook(
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
        nb_path = self._write_notebook(
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
        nb_path = self._write_notebook([])
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


if __name__ == "__main__":
    unittest.main()
