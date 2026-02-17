# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from aibom.config_parser import parse_langgraph_json, parse_project_configs


class TestLangGraphJsonParser(unittest.TestCase):

    def _write_langgraph_json(self, data: dict) -> Path:
        tmp_dir = Path(tempfile.mkdtemp())
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
        self.assertIsNone(parse_langgraph_json(tmp_dir))

    def test_returns_none_on_invalid_json(self):
        tmp_dir = Path(tempfile.mkdtemp())
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


if __name__ == "__main__":
    unittest.main()
