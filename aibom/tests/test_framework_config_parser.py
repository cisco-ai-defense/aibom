# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from aibom.framework_config_parser import parse_langgraph_json, parse_project_configs


class TestLangGraphJsonParser(unittest.TestCase):

    def setUp(self):
        self._tmp_dirs: list[Path] = []

    def tearDown(self):
        for d in self._tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)

    def _write_langgraph_json(self, data: dict) -> Path:
        tmp_dir = Path(tempfile.mkdtemp())
        self._tmp_dirs.append(tmp_dir)
        config_path = tmp_dir / "langgraph.json"
        config_path.write_text(json.dumps(data), encoding="utf-8")
        return tmp_dir

    def test_parses_graphs(self):
        root = self._write_langgraph_json(
            {
                "graphs": {
                    "agent": "./src/memory_agent/graph.py:graph",
                }
            }
        )
        result = parse_langgraph_json(root)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.assignments), 1)
        self.assertEqual(result.assignments[0].target_qualified_name, "agent")
        self.assertIn("entrypoint", result.assignments[0].call.arguments)

    def test_parses_store_embed(self):
        root = self._write_langgraph_json(
            {
                "graphs": {},
                "store": {
                    "index": {
                        "embed": "openai:text-embedding-3-small",
                        "dims": 1536,
                    }
                },
            }
        )
        result = parse_langgraph_json(root)
        self.assertIsNotNone(result)
        embedding_obs = [
            a
            for a in result.assignments
            if a.target_qualified_name == "langgraph_config_embedding"
        ]
        self.assertEqual(len(embedding_obs), 1)
        self.assertEqual(embedding_obs[0].call.arguments["model"], "openai:text-embedding-3-small")
        self.assertEqual(embedding_obs[0].call.arguments["dims"], 1536)

    def test_returns_none_when_no_file(self):
        tmp_dir = Path(tempfile.mkdtemp())
        self._tmp_dirs.append(tmp_dir)
        self.assertIsNone(parse_langgraph_json(tmp_dir))

    def test_returns_none_on_invalid_json(self):
        tmp_dir = Path(tempfile.mkdtemp())
        self._tmp_dirs.append(tmp_dir)
        (tmp_dir / "langgraph.json").write_text("not json", encoding="utf-8")
        self.assertIsNone(parse_langgraph_json(tmp_dir))

    def test_parse_project_configs_delegates(self):
        root = self._write_langgraph_json(
            {
                "graphs": {"g": "./g.py:run"},
            }
        )
        results = parse_project_configs(root)
        self.assertEqual(len(results), 1)

    def test_parse_project_configs_with_file_parent(self):
        """Config parsing should work when given a file's parent directory.

        Regression: when the CLI receives a single file path, it should
        pass the parent directory to ``parse_project_configs`` so that
        sibling config files like ``langgraph.json`` are still discovered.
        """
        root = self._write_langgraph_json(
            {
                "graphs": {"agent": "./agent.py:graph"},
            }
        )
        dummy_file = root / "agent.py"
        dummy_file.write_text("x = 1\n", encoding="utf-8")

        # Simulate the CLI fix: use file.parent when input is a file
        config_root = dummy_file.parent
        results = parse_project_configs(config_root)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].assignments[0].target_qualified_name, "agent")


if __name__ == "__main__":
    unittest.main()
