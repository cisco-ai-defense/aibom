# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

import unittest

from aibom.supplemental_catalog import SUPPLEMENTAL_ENTRIES


class TestSupplementalCatalog(unittest.TestCase):

    def test_entries_are_not_empty(self):
        self.assertGreater(len(SUPPLEMENTAL_ENTRIES), 0)

    def test_required_keys_present(self):
        required_keys = {"id", "label", "concept"}
        for entry in SUPPLEMENTAL_ENTRIES:
            for key in required_keys:
                self.assertIn(key, entry, f"Entry {entry.get('id', '?')} missing key '{key}'")

    def test_langgraph_entries_exist(self):
        ids = {e["id"] for e in SUPPLEMENTAL_ENTRIES}
        self.assertIn("langgraph.graph.StateGraph", ids)
        self.assertIn("langgraph.prebuilt.create_react_agent", ids)
        self.assertIn("langgraph.checkpoint.memory.MemorySaver", ids)
        self.assertIn("langgraph.store.memory.InMemoryStore", ids)

    def test_crewai_entries_exist(self):
        ids = {e["id"] for e in SUPPLEMENTAL_ENTRIES}
        self.assertIn("crewai.Agent", ids)
        self.assertIn("crewai.Crew", ids)
        self.assertIn("crewai.Task", ids)

    def test_langchain_tool_entries_exist(self):
        ids = {e["id"] for e in SUPPLEMENTAL_ENTRIES}
        self.assertIn("langchain_core.tools.tool", ids)
        self.assertIn("langchain_core.tools.BaseTool", ids)

    def test_memory_concept_entries(self):
        memory_entries = [e for e in SUPPLEMENTAL_ENTRIES if e["concept"] == "memory"]
        self.assertGreater(len(memory_entries), 5)

    def test_retriever_concept_entries(self):
        retriever_entries = [e for e in SUPPLEMENTAL_ENTRIES if e["concept"] == "retriever"]
        self.assertGreater(len(retriever_entries), 0)

    def test_no_duplicate_ids(self):
        ids = [e["id"] for e in SUPPLEMENTAL_ENTRIES]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate IDs found in supplemental catalog")


if __name__ == "__main__":
    unittest.main()
