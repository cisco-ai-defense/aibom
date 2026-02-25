# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Validate catalog entries used by the build script.

These tests replace the original supplemental_catalog tests.  They exercise
the ``catalog_entries`` data module that feeds ``build_catalog.py`` and
verify that building a temporary DuckDB produces a valid catalog.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import duckdb

# Ensure the scripts directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from catalog_entries import get_all_entries


class TestCatalogEntries(unittest.TestCase):

    def test_entries_are_not_empty(self):
        entries = get_all_entries()
        self.assertGreater(len(entries), 0)

    def test_required_keys_present(self):
        required_keys = {"id", "label", "concept"}
        for entry in get_all_entries():
            for key in required_keys:
                self.assertIn(key, entry, f"Entry {entry.get('id', '?')} missing key '{key}'")

    def test_no_duplicate_ids(self):
        ids = [e["id"] for e in get_all_entries()]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate IDs found in catalog entries")

    def test_langgraph_entries_exist(self):
        ids = {e["id"] for e in get_all_entries()}
        self.assertIn("langgraph.graph.StateGraph", ids)
        self.assertIn("langgraph.prebuilt.create_react_agent", ids)
        self.assertIn("langgraph.checkpoint.memory.MemorySaver", ids)
        self.assertIn("langgraph.store.memory.InMemoryStore", ids)

    def test_crewai_entries_exist(self):
        ids = {e["id"] for e in get_all_entries()}
        self.assertIn("crewai.Agent", ids)
        self.assertIn("crewai.Crew", ids)
        self.assertIn("crewai.Task", ids)

    def test_langchain_tool_entries_exist(self):
        ids = {e["id"] for e in get_all_entries()}
        self.assertIn("langchain_core.tools.tool", ids)
        self.assertIn("langchain_core.tools.BaseTool", ids)

    def test_memory_concept_entries(self):
        memory_entries = [e for e in get_all_entries() if e["concept"] == "memory"]
        self.assertGreater(len(memory_entries), 5)

    def test_retriever_concept_entries(self):
        retriever_entries = [e for e in get_all_entries() if e["concept"] == "retriever"]
        self.assertGreater(len(retriever_entries), 0)

    def test_new_framework_entries_exist(self):
        """Phase 6: verify new framework entries were added."""
        ids = {e["id"] for e in get_all_entries()}
        # AutoGen
        self.assertIn("autogen.AssistantAgent", ids)
        # DSPy
        self.assertIn("dspy.ChainOfThought", ids)
        # Haystack
        self.assertIn("haystack.Pipeline", ids)
        # LlamaIndex
        self.assertIn("llama_index.core.VectorStoreIndex", ids)
        # Semantic Kernel
        self.assertIn("semantic_kernel.Kernel", ids)
        # Smolagents
        self.assertIn("smolagents.CodeAgent", ids)
        # Google GenAI
        self.assertIn("google.generativeai.GenerativeModel", ids)

    def test_build_temp_duckdb(self):
        """Build a temp DuckDB and verify key entries are queryable."""
        entries = get_all_entries()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_catalog.duckdb"
            con = duckdb.connect(str(db_path))
            con.execute(
                """
                CREATE TABLE component_catalog (
                    id TEXT PRIMARY KEY,
                    label TEXT,
                    concept TEXT,
                    framework TEXT,
                    sig_name TEXT,
                    type TEXT,
                    catalog_label TEXT
                );
                """
            )
            for entry in entries:
                con.execute(
                    "INSERT OR IGNORE INTO component_catalog VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        entry["id"],
                        entry.get("label"),
                        entry.get("concept"),
                        entry.get("framework"),
                        entry.get("sig_name"),
                        entry.get("type"),
                        entry.get("catalog_label"),
                    ],
                )

            count = con.execute("SELECT COUNT(*) FROM component_catalog").fetchone()[0]
            self.assertEqual(count, len(entries))

            # Verify suffix query works
            rows = con.execute(
                "SELECT id FROM component_catalog WHERE id LIKE ?",
                ["%StateGraph"],
            ).fetchall()
            state_graph_ids = {row[0] for row in rows}
            self.assertIn("langgraph.graph.StateGraph", state_graph_ids)

            con.close()


if __name__ == "__main__":
    unittest.main()
