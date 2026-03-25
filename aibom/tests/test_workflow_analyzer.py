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

import tempfile
from pathlib import Path
import unittest

from aibom.workflow_analyzer import build_workflow_index


WORKFLOW_SOURCE = """
def workflow_entry():
    helper()

def helper():
    agent_action(42)

def agent_action(value):
    return value
""".strip()


class TestWorkflowAnalyzer(unittest.TestCase):
    def test_build_workflow_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "app.py"
            path.write_text(WORKFLOW_SOURCE, encoding="utf-8")

            index = build_workflow_index([path])
            helper_line = next(
                idx + 1
                for idx, line in enumerate(WORKFLOW_SOURCE.splitlines())
                if "def helper" in line
            )
            workflows = index.get_workflow_context(str(path), helper_line)

            self.assertTrue(any("helper" in wf.get("function", "") for wf in workflows))
            parent = next(
                (wf for wf in workflows if "workflow_entry" in (wf.get("function") or "")),
                None,
            )
            self.assertIsNotNone(parent)
            self.assertIsNotNone(parent.get("callsite_line"))
            self.assertIn("args", parent.get("call_arguments") or {})
            agent_line = next(
                idx + 1
                for idx, line in enumerate(WORKFLOW_SOURCE.splitlines())
                if "def agent_action" in line
            )
            agent_workflows = index.get_workflow_context(str(path), agent_line)
            agent = agent_workflows[0]
            self.assertIn("value", agent.get("parameters", []))


    def test_syntax_error_handled_gracefully(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad = Path(tmp_dir) / "broken.py"
            bad.write_text("def foo(:\n    pass\n", encoding="utf-8")
            index = build_workflow_index([bad])
            assert index.get_workflow_context(str(bad), 1) == []

    def test_multi_file_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            a = Path(tmp_dir) / "a.py"
            a.write_text("def caller():\n    callee()\n", encoding="utf-8")
            b = Path(tmp_dir) / "b.py"
            b.write_text("def callee():\n    return 42\n", encoding="utf-8")
            index = build_workflow_index([a, b])
            ctx_a = index.get_workflow_context(str(a), 1)
            assert isinstance(ctx_a, list)

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty = Path(tmp_dir) / "empty.py"
            empty.write_text("", encoding="utf-8")
            index = build_workflow_index([empty])
            assert index.get_workflow_context(str(empty), 1) == []


if __name__ == "__main__":
    unittest.main()
