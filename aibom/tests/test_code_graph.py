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

import os
import shutil
import tempfile
import unittest
from unittest import mock

from aibom.code_graph import (
    build_code_graph,
    derive_relationships,
    resolve_literal_model_names,
)
from aibom.code_graph.models import CodeGraph, ValueBinding
from aibom.cst_parser import parse_source_code
from aibom.models.enums import (
    AIComponentType,
    DetectionSource,
    EvidenceStrength,
    RelationshipType,
)
from aibom.models.scan import AIComponent, ComponentRelationship
from aibom.scan_pipeline import (
    ScanPipeline,
    _apply_literal_model_names,
    _dedup_relationships,
    _propagate_model_from_relationships,
    _remap_relationship_endpoints,
)


def _parse(path, source):
    return parse_source_code(path, source)


def _component(name, ctype, line, *, path="app.py", assigned=None, args=None):
    metadata = {}
    if assigned is not None:
        metadata["assigned_target"] = assigned
    if args is not None:
        metadata["arguments"] = args
    return AIComponent(
        name=name,
        component_type=ctype,
        file_path=path,
        line_number=line,
        metadata=metadata,
    )


class TestValueAssignmentCapture(unittest.TestCase):
    """Phase 0: non-call right-hand sides must reach the parser output."""

    def _bindings(self, source):
        result = _parse("m.py", source)
        return {
            v.target_qualified_name.split(".")[-1]: v for v in result.value_assignments
        }

    def test_string_literal_is_recorded(self):
        got = self._bindings('model = "gpt-4"\n')
        self.assertEqual(got["model"].value, "gpt-4")
        self.assertEqual(got["model"].value_kind, "literal")

    def test_name_reference_is_tagged_as_variable(self):
        got = self._bindings("DEFAULT = 'x'\nmodel = DEFAULT\n")
        self.assertEqual(got["model"].value, "VARIABLE:DEFAULT")
        self.assertEqual(got["model"].value_kind, "variable")

    def test_subscript_is_rendered(self):
        got = self._bindings('model = CONFIG["model"]\n')
        self.assertEqual(got["model"].value, "SUBSCRIPT:CONFIG[model]")
        self.assertEqual(got["model"].value_kind, "subscript")

    def test_attribute_is_tagged(self):
        got = self._bindings("model = settings.model_name\n")
        self.assertEqual(got["model"].value, "ATTRIBUTE:settings.model_name")

    def test_every_target_of_a_chained_assignment_is_recorded(self):
        got = self._bindings('a = b = "shared"\n')
        self.assertEqual(got["a"].value, "shared")
        self.assertEqual(got["b"].value, "shared")

    def test_unresolvable_shapes_are_skipped(self):
        result = _parse("m.py", "f = lambda x: x\ng = [i for i in range(3)]\n")
        self.assertEqual(result.value_assignments, [])

    def test_scope_owner_distinguishes_module_from_function(self):
        source = 'TOP = "a"\n\ndef build():\n    inner = "b"\n'
        got = self._bindings(source)
        self.assertIsNone(got["TOP"].owner_qualified_name)
        self.assertEqual(got["inner"].owner_qualified_name, "build")

    def test_call_right_hand_side_still_goes_to_assignments(self):
        result = _parse("m.py", "from openai import OpenAI\n\nclient = OpenAI()\n")
        self.assertEqual(len(result.assignments), 1)
        self.assertEqual(result.value_assignments, [])

    def test_await_call_is_not_mistaken_for_a_value(self):
        source = "from x import go\n\nasync def f():\n    r = await go()\n"
        self.assertEqual(_parse("m.py", source).value_assignments, [])


class TestCallGraphResolution(unittest.TestCase):
    """Phase 1: callee name strings must become real edges."""

    AGENT = """
from .tools import search_web
import openai

def helper(x):
    return x

class Agent:
    def run(self):
        while True:
            out = self._step()
            helper(out)
            search_web(out)
        return out

    def _step(self):
        return openai.chat.completions.create(model="gpt-4")
"""
    TOOLS = """
def search_web(q):
    return q
"""

    def setUp(self):
        self.graph = build_code_graph(
            [
                _parse("pkg/agent.py", self.AGENT),
                _parse("pkg/tools.py", self.TOOLS),
            ]
        )
        self.run_id = "pkg/agent.py::Agent.run"

    def test_self_call_resolves_to_sibling_method(self):
        self.assertIn("pkg/agent.py::Agent._step", self.graph.callees_of(self.run_id))

    def test_module_level_helper_resolves_in_same_file(self):
        self.assertIn("pkg/agent.py::helper", self.graph.callees_of(self.run_id))

    def test_relative_import_resolves_across_files(self):
        self.assertIn("pkg/tools.py::search_web", self.graph.callees_of(self.run_id))

    def test_third_party_call_is_recorded_as_unresolved(self):
        unresolved = self.graph.unresolved_calls["pkg/agent.py::Agent._step"]
        self.assertIn("openai.chat.completions.create", unresolved)

    def test_transitive_callees_see_through_one_level_of_indirection(self):
        """The gap that defeats string-matching ReAct detection."""
        reached = {n.node_id for n in self.graph.transitive_callees(self.run_id)}
        self.assertIn("pkg/agent.py::Agent._step", reached)
        self.assertIn("pkg/tools.py::search_web", reached)

    def test_reverse_edges_find_callers(self):
        callers = self.graph.callers_of("pkg/tools.py::search_web")
        self.assertEqual([n.node_id for n in callers], [self.run_id])

    def test_self_call_does_not_create_a_self_loop(self):
        self.assertNotIn(self.run_id, self.graph.callees_of(self.run_id))


class TestAmbiguousModuleResolution(unittest.TestCase):
    def test_same_named_modules_resolve_by_directory(self):
        caller = "a/main.py"
        source = "from .util import run\n\ndef go():\n    run()\n"
        graph = build_code_graph(
            [
                _parse(caller, source),
                _parse("a/util.py", "def run():\n    return 1\n"),
                _parse("b/util.py", "def run():\n    return 2\n"),
            ]
        )
        self.assertIn("a/util.py::run", graph.callees_of("a/main.py::go"))
        self.assertNotIn("b/util.py::run", graph.callees_of("a/main.py::go"))


class TestEnclosingFunction(unittest.TestCase):
    def test_tightest_enclosing_function_wins(self):
        source = "def outer():\n    def inner():\n        pass\n    inner()\n"
        graph = build_code_graph([_parse("m.py", source)])
        node = graph.enclosing_function("m.py", 3)
        self.assertIsNotNone(node)
        self.assertEqual(node.method_name, "inner")

    def test_line_outside_any_function_returns_none(self):
        graph = build_code_graph([_parse("m.py", "X = 1\n\ndef f():\n    pass\n")])
        self.assertIsNone(graph.enclosing_function("m.py", 1))


class TestValueResolution(unittest.TestCase):
    def test_chain_is_followed_to_a_literal(self):
        source = 'BASE = "gpt-4"\nalias = BASE\nfinal = alias\n'
        graph = build_code_graph([_parse("m.py", source)])
        got = graph.resolve_value("m.py", "final")
        self.assertEqual(got.value, "gpt-4")

    def test_function_scope_shadows_module_scope(self):
        source = 'model = "module"\n\ndef build():\n    model = "local"\n'
        graph = build_code_graph([_parse("m.py", source)])
        self.assertEqual(graph.resolve_value("m.py", "model", "build").value, "local")
        self.assertEqual(graph.resolve_value("m.py", "model").value, "module")

    def test_unknown_name_resolves_to_none(self):
        graph = build_code_graph([_parse("m.py", 'a = "x"\n')])
        self.assertIsNone(graph.resolve_value("m.py", "nope"))

    def test_cycle_does_not_hang(self):
        graph = CodeGraph()
        graph.bindings["m.py"] = [
            ValueBinding("a", "VARIABLE:b", "variable", None, 1),
            ValueBinding("b", "VARIABLE:a", "variable", None, 2),
        ]
        self.assertIsNotNone(graph.resolve_value("m.py", "a"))

    def test_subscript_terminates_the_walk_without_pretending_to_resolve(self):
        graph = build_code_graph([_parse("m.py", 'model = CONFIG["model"]\n')])
        got = graph.resolve_value("m.py", "model")
        self.assertEqual(got.kind, "subscript")

    def test_a_rebound_name_resolves_to_the_value_in_effect(self):
        """The reader's line decides, not the file's last word.

        Notebooks reassign one name per experiment. Reading the final
        value for every use reports the last model for all of them.
        """
        source = 'name = "gpt-4o"\nfirst = 1\nname = "claude-3-opus"\nsecond = 2\n'
        graph = build_code_graph([_parse("m.py", source)])
        self.assertEqual(
            graph.resolve_value("m.py", "name", before_line=2).value, "gpt-4o"
        )
        self.assertEqual(
            graph.resolve_value("m.py", "name", before_line=4).value, "claude-3-opus"
        )

    def test_a_name_bound_later_is_not_read_early(self):
        graph = build_code_graph([_parse("m.py", 'x = 1\nname = "gpt-4o"\n')])
        self.assertIsNone(graph.resolve_value("m.py", "name", before_line=1))

    def test_a_chain_link_is_read_at_the_line_that_bound_it(self):
        """``alias`` took its value at line 2, so it cannot see line 4."""
        source = 'BASE = "old"\nalias = BASE\nuse = 1\nBASE = "new"\n'
        graph = build_code_graph([_parse("m.py", source)])
        self.assertEqual(
            graph.resolve_value("m.py", "alias", before_line=3).value, "old"
        )

    def test_line_awareness_is_optional(self):
        source = 'name = "gpt-4o"\nname = "claude-3-opus"\n'
        graph = build_code_graph([_parse("m.py", source)])
        self.assertEqual(graph.resolve_value("m.py", "name").value, "claude-3-opus")


class TestLiteralModelNames(unittest.TestCase):
    """Reading the model out of the variable a keyword names."""

    @staticmethod
    def _resolve(source, components, path="m.py"):
        graph = build_code_graph([_parse(path, source)])
        return resolve_literal_model_names(components, graph)

    def test_a_variable_gives_up_its_model(self):
        source = 'expt = "gpt-4o"\nllm = ChatOpenAI(model=expt)\n'
        comp = _component(
            "ChatOpenAI",
            AIComponentType.MODEL,
            2,
            path="m.py",
            args={"model": "VARIABLE:expt"},
        )
        self.assertEqual(self._resolve(source, [comp])[comp.instance_id], "gpt-4o")

    def test_each_use_gets_the_model_of_its_own_line(self):
        """The bug this exists to stop: one value smeared across a file."""
        source = (
            'expt = "gpt-4o"\n'
            "a = ChatOpenAI(model=expt)\n"
            'expt = "claude-3-opus"\n'
            "b = ChatAnthropic(model=expt)\n"
        )
        first = _component(
            "ChatOpenAI",
            AIComponentType.MODEL,
            2,
            path="m.py",
            args={"model": "VARIABLE:expt"},
        )
        second = _component(
            "ChatAnthropic",
            AIComponentType.MODEL,
            4,
            path="m.py",
            args={"model": "VARIABLE:expt"},
        )
        got = self._resolve(source, [first, second])
        self.assertEqual(got[first.instance_id], "gpt-4o")
        self.assertEqual(got[second.instance_id], "claude-3-opus")

    def test_a_local_beats_a_module_level_name(self):
        source = (
            'expt = "module"\n'
            "def build():\n"
            '    expt = "local"\n'
            "    return ChatOpenAI(model=expt)\n"
        )
        comp = _component(
            "ChatOpenAI",
            AIComponentType.MODEL,
            4,
            path="m.py",
            args={"model": "VARIABLE:expt"},
        )
        self.assertEqual(self._resolve(source, [comp])[comp.instance_id], "local")

    def test_a_chain_is_followed(self):
        source = 'BASE = "gpt-4"\nexpt = BASE\nllm = ChatOpenAI(model=expt)\n'
        comp = _component(
            "ChatOpenAI",
            AIComponentType.MODEL,
            3,
            path="m.py",
            args={"model": "VARIABLE:expt"},
        )
        self.assertEqual(self._resolve(source, [comp])[comp.instance_id], "gpt-4")

    def test_a_value_it_cannot_evaluate_is_left_alone(self):
        source = 'expt = CONFIG["model"]\nllm = ChatOpenAI(model=expt)\n'
        comp = _component(
            "ChatOpenAI",
            AIComponentType.MODEL,
            2,
            path="m.py",
            args={"model": "VARIABLE:expt"},
        )
        self.assertEqual(self._resolve(source, [comp]), {})

    def test_an_unknown_name_reports_nothing(self):
        comp = _component(
            "ChatOpenAI",
            AIComponentType.MODEL,
            1,
            path="m.py",
            args={"model": "VARIABLE:missing"},
        )
        self.assertEqual(self._resolve("x = 1\n", [comp]), {})

    def test_a_keyword_that_is_not_the_model_is_ignored(self):
        """``callbacks=handler`` names a handler, not a model."""
        source = 'handler = "thing"\nllm = ChatOpenAI(callbacks=handler)\n'
        comp = _component(
            "ChatOpenAI",
            AIComponentType.MODEL,
            2,
            path="m.py",
            args={"callbacks": "VARIABLE:handler"},
        )
        self.assertEqual(self._resolve(source, [comp]), {})

    def test_a_component_that_knows_its_model_is_not_second_guessed(self):
        source = 'expt = "gpt-4o"\nllm = ChatOpenAI(model=expt)\n'
        comp = _component(
            "ChatOpenAI",
            AIComponentType.MODEL,
            2,
            path="m.py",
            args={"model": "VARIABLE:expt"},
        )
        comp.model_name = "already-known"
        self.assertEqual(self._resolve(source, [comp]), {})

    def test_an_empty_literal_is_not_a_model(self):
        source = 'expt = ""\nllm = ChatOpenAI(model=expt)\n'
        comp = _component(
            "ChatOpenAI",
            AIComponentType.MODEL,
            2,
            path="m.py",
            args={"model": "VARIABLE:expt"},
        )
        self.assertEqual(self._resolve(source, [comp]), {})


class TestApplyingLiteralModelNames(unittest.TestCase):
    def test_only_the_named_component_is_written_to(self):
        target = _component("A", AIComponentType.MODEL, 1)
        other = _component("B", AIComponentType.MODEL, 2)
        got = _apply_literal_model_names(
            [target, other], {target.instance_id: "gpt-4o"}
        )
        self.assertEqual(got[0].model_name, "gpt-4o")
        self.assertIsNone(got[1].model_name)

    def test_an_established_model_is_not_overwritten(self):
        comp = _component("A", AIComponentType.MODEL, 1)
        comp.model_name = "from-a-scanner"
        got = _apply_literal_model_names([comp], {comp.instance_id: "gpt-4o"})
        self.assertEqual(got[0].model_name, "from-a-scanner")

    def test_nothing_to_apply_returns_the_input(self):
        comps = [_component("A", AIComponentType.MODEL, 1)]
        self.assertIs(_apply_literal_model_names(comps, {}), comps)

    def test_a_stale_id_is_ignored(self):
        comp = _component("A", AIComponentType.MODEL, 1)
        got = _apply_literal_model_names([comp], {"no-such-id": "gpt-4o"})
        self.assertIsNone(got[0].model_name)


class TestDerivedRelationships(unittest.TestCase):
    """Phase 2b: edges read off constructor keywords."""

    def _agent_graph(self):
        return [
            _component("ChatOpenAI", AIComponentType.MODEL, 10, assigned="my_llm"),
            _component("SearchTool", AIComponentType.TOOL, 12, assigned="search"),
            _component("CalcTool", AIComponentType.TOOL, 13, assigned="calc"),
            _component("Chroma", AIComponentType.VECTOR_STORE, 14, assigned="vstore"),
            _component("BufferMemory", AIComponentType.MEMORY, 15, assigned="mem"),
            _component(
                "AgentExecutor",
                AIComponentType.AGENT,
                20,
                args={
                    "llm": "VARIABLE:my_llm",
                    "tools": ["VARIABLE:search", "VARIABLE:calc"],
                    "memory": "VARIABLE:mem",
                    "store": "VARIABLE:vstore",
                    "verbose": True,
                    "callback": "VARIABLE:my_llm",
                    "_pos_0": "VARIABLE:search",
                },
            ),
        ]

    def _edges(self, components=None, graph=None):
        return derive_relationships(components or self._agent_graph(), graph)

    def test_model_keyword_produces_uses_model(self):
        kinds = {(e.relationship_type, e.target_name) for e in self._edges()}
        self.assertIn((RelationshipType.USES_MODEL, "ChatOpenAI"), kinds)

    def test_list_valued_keyword_produces_one_edge_per_element(self):
        tools = {
            e.target_name
            for e in self._edges()
            if e.relationship_type is RelationshipType.USES_TOOL
        }
        self.assertEqual(tools, {"SearchTool", "CalcTool"})

    def test_shared_keyword_is_disambiguated_by_target_type(self):
        """``store=`` is a memory in one framework and a vector store in another."""
        edges = [e for e in self._edges() if e.target_name == "Chroma"]
        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].relationship_type, RelationshipType.USES_VECTOR_STORE)

    def test_non_dependency_keyword_is_ignored(self):
        self.assertFalse(
            any(e.label == "CUSTOM" for e in self._edges()),
        )
        # ``callback=`` points at a real model but is not a dependency kwarg,
        # so it must not create a second USES_MODEL edge.
        models = [
            e
            for e in self._edges()
            if e.relationship_type is RelationshipType.USES_MODEL
        ]
        self.assertEqual(len(models), 1)

    def test_positional_arguments_are_ignored(self):
        edges = self._edges()
        self.assertEqual(len(edges), 5)

    def test_literal_valued_keyword_produces_no_edge(self):
        comps = [
            _component("Agent", AIComponentType.AGENT, 5, args={"llm": "gpt-4"}),
        ]
        self.assertEqual(self._edges(comps), [])

    def test_wrong_target_type_produces_no_edge(self):
        comps = [
            _component("SomeTool", AIComponentType.TOOL, 5, assigned="thing"),
            _component(
                "Agent", AIComponentType.AGENT, 9, args={"llm": "VARIABLE:thing"}
            ),
        ]
        self.assertEqual(self._edges(comps), [])

    def test_agent_in_tools_keyword_is_delegation(self):
        # ``tools=[AgentTool(researcher)]`` -- the reference resolves to an
        # agent, which the tool rule would otherwise reject on type.
        comps = [
            _component("researcher", AIComponentType.AGENT, 5, assigned="researcher"),
            _component(
                "lead",
                AIComponentType.AGENT,
                9,
                args={
                    "tools": [{"_call": "AgentTool", "_args": ["VARIABLE:researcher"]}]
                },
            ),
        ]
        edges = self._edges(comps)
        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].relationship_type, RelationshipType.USES_AGENT)
        self.assertEqual(edges[0].target_name, "researcher")

    def test_agent_exemption_does_not_apply_to_other_keywords(self):
        # The type check still guards every keyword that is not a tool slot.
        comps = [
            _component("researcher", AIComponentType.AGENT, 5, assigned="researcher"),
            _component(
                "lead", AIComponentType.AGENT, 9, args={"llm": "VARIABLE:researcher"}
            ),
        ]
        self.assertEqual(self._edges(comps), [])

    def test_ambiguous_name_is_not_guessed(self):
        comps = [
            _component("Dup", AIComponentType.TOOL, 1, path="a.py"),
            _component("Dup", AIComponentType.TOOL, 1, path="b.py"),
            _component(
                "Agent",
                AIComponentType.AGENT,
                9,
                path="c.py",
                args={"tools": "VARIABLE:Dup"},
            ),
        ]
        self.assertEqual(self._edges(comps), [])

    def test_edges_are_marked_as_code_derived(self):
        for edge in self._edges():
            self.assertIs(edge.detection_source, DetectionSource.CODE_ANALYSIS)
            self.assertTrue(edge.is_code_derived)

    def test_edges_carry_auditable_source_locations(self):
        edge = next(
            e
            for e in self._edges()
            if e.relationship_type is RelationshipType.USES_MODEL
        )
        locations = edge.decision_annotation.evidence_locations
        self.assertEqual(
            [(loc.file_path, loc.start_line) for loc in locations],
            [("app.py", 20), ("app.py", 10)],
        )

    def test_duplicate_references_are_deduped(self):
        comps = [
            _component("T", AIComponentType.TOOL, 1, assigned="t"),
            _component(
                "Agent",
                AIComponentType.AGENT,
                9,
                args={"tools": ["VARIABLE:t", "VARIABLE:t"]},
            ),
        ]
        self.assertEqual(len(self._edges(comps)), 1)

    def test_component_does_not_reference_itself(self):
        comps = [
            _component(
                "Thing",
                AIComponentType.TOOL,
                4,
                assigned="thing",
                args={"tools": "VARIABLE:thing"},
            ),
        ]
        self.assertEqual(self._edges(comps), [])


class TestScopeGate(unittest.TestCase):
    """The graph's contribution to relationship precision."""

    SOURCE = """
def build_a():
    client = make_model()
    return Agent(llm=client)

def build_b():
    client = make_other()
    return client
"""

    def _components(self):
        return [
            _component("OtherModel", AIComponentType.MODEL, 7, assigned="client"),
            _component(
                "Agent", AIComponentType.AGENT, 4, args={"llm": "VARIABLE:client"}
            ),
        ]

    def test_cross_function_reference_is_dropped_with_a_graph(self):
        graph = build_code_graph([_parse("app.py", self.SOURCE)])
        self.assertEqual(derive_relationships(self._components(), graph), [])

    def test_same_scope_reference_is_kept(self):
        source = "def build():\n    client = make()\n    return Agent(llm=client)\n"
        graph = build_code_graph([_parse("app.py", source)])
        comps = [
            _component("Model", AIComponentType.MODEL, 2, assigned="client"),
            _component(
                "Agent", AIComponentType.AGENT, 3, args={"llm": "VARIABLE:client"}
            ),
        ]
        self.assertEqual(len(derive_relationships(comps, graph)), 1)

    def test_module_level_binding_is_visible_from_any_function(self):
        source = "client = make()\n\ndef build():\n    return Agent(llm=client)\n"
        graph = build_code_graph([_parse("app.py", source)])
        comps = [
            _component("Model", AIComponentType.MODEL, 1, assigned="client"),
            _component(
                "Agent", AIComponentType.AGENT, 4, args={"llm": "VARIABLE:client"}
            ),
        ]
        self.assertEqual(len(derive_relationships(comps, graph)), 1)


class TestEndpointRemapping(unittest.TestCase):
    """Edges are built before consolidation and must survive it.

    Consolidation groups components by (name, type) across the whole repo
    and keeps one representative, so a derived edge naming a per-call-site
    instance id would otherwise be discarded as dangling. A benchmark run
    over crewAI-examples lost 13 of 13 edges to exactly this.
    """

    def _edge(self, source_id, target_id):
        return ComponentRelationship(
            source_instance_id=source_id,
            target_instance_id=target_id,
            relationship_type=RelationshipType.USES_AGENT,
            source_name="Task",
            target_name="researcher",
            detection_source=DetectionSource.CODE_ANALYSIS,
        )

    def test_endpoint_follows_its_consolidation_representative(self):
        absorbed = _component("Task", AIComponentType.AGENT, 10, path="a.py")
        survivor = _component("Task", AIComponentType.AGENT, 3, path="b.py")
        target = _component("researcher", AIComponentType.AGENT, 1, path="b.py")

        remapped = _remap_relationship_endpoints(
            [self._edge(absorbed.instance_id, target.instance_id)],
            [absorbed, survivor, target],
            [survivor, target],
        )

        self.assertEqual(len(remapped), 1)
        self.assertEqual(remapped[0].source_instance_id, survivor.instance_id)
        self.assertEqual(remapped[0].target_instance_id, target.instance_id)

    def test_edge_is_dropped_when_an_endpoint_does_not_survive(self):
        source = _component("Task", AIComponentType.AGENT, 10)
        dropped = _component("researcher", AIComponentType.AGENT, 1)

        self.assertEqual(
            _remap_relationship_endpoints(
                [self._edge(source.instance_id, dropped.instance_id)],
                [source, dropped],
                [source],
            ),
            [],
        )

    def test_self_loop_from_shared_representative_is_dropped(self):
        """Two call sites of one class carry no edge once merged."""
        first = _component("Task", AIComponentType.AGENT, 10, path="a.py")
        second = _component("Task", AIComponentType.AGENT, 20, path="b.py")

        self.assertEqual(
            _remap_relationship_endpoints(
                [self._edge(first.instance_id, second.instance_id)],
                [first, second],
                [first],
            ),
            [],
        )

    def test_already_current_endpoints_pass_through(self):
        source = _component("Task", AIComponentType.AGENT, 10)
        target = _component("researcher", AIComponentType.AGENT, 1)
        survivors = [source, target]

        remapped = _remap_relationship_endpoints(
            [self._edge(source.instance_id, target.instance_id)],
            survivors,
            survivors,
        )

        self.assertEqual(len(remapped), 1)
        self.assertEqual(remapped[0].source_instance_id, source.instance_id)


class TestPipelinePrecedence(unittest.TestCase):
    """Code-derived edges must outrank inferred ones wherever they collide."""

    def _pair(self):
        """The same edge, once inferred and once read off the source."""
        common = dict(
            source_instance_id="agent-1",
            target_instance_id="model-1",
            relationship_type=RelationshipType.USES_MODEL,
            source_name="Agent",
            target_name="ChatOpenAI",
        )
        inferred = ComponentRelationship(**common)
        derived = ComponentRelationship(
            **common, detection_source=DetectionSource.CODE_ANALYSIS
        )
        return inferred, derived

    def test_dedup_keeps_the_code_derived_edge_when_it_arrives_second(self):
        inferred, derived = self._pair()
        kept = _dedup_relationships([inferred, derived])
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].is_code_derived)

    def test_dedup_keeps_the_code_derived_edge_when_it_arrives_first(self):
        inferred, derived = self._pair()
        kept = _dedup_relationships([derived, inferred])
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].is_code_derived)

    def test_model_name_prefers_the_code_derived_edge(self):
        """A wrong edge here becomes a wrongly reported model."""
        agent = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="app.py",
            line_number=1,
        )
        agent.instance_id = "agent-1"
        inferred = ComponentRelationship(
            source_instance_id="agent-1",
            target_instance_id="x",
            relationship_type=RelationshipType.USES_MODEL,
            source_name="Agent",
            target_name="hallucinated-model",
        )
        derived = ComponentRelationship(
            source_instance_id="agent-1",
            target_instance_id="y",
            relationship_type=RelationshipType.USES_MODEL,
            source_name="Agent",
            target_name="real-model",
            detection_source=DetectionSource.CODE_ANALYSIS,
        )
        for order in ([inferred, derived], [derived, inferred]):
            updated = _propagate_model_from_relationships([agent], order)
            self.assertEqual(updated[0].model_name, "real-model")

    def test_inferred_edge_is_still_used_when_no_derived_edge_exists(self):
        agent = AIComponent(
            name="Agent",
            component_type=AIComponentType.AGENT,
            file_path="app.py",
            line_number=1,
        )
        agent.instance_id = "agent-1"
        inferred = ComponentRelationship(
            source_instance_id="agent-1",
            target_instance_id="x",
            relationship_type=RelationshipType.USES_MODEL,
            source_name="Agent",
            target_name="only-guess",
        )
        updated = _propagate_model_from_relationships([agent], [inferred])
        self.assertEqual(updated[0].model_name, "only-guess")


class TestPipelineIntegration(unittest.TestCase):
    SOURCE = """
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

my_llm = ChatOpenAI(model="gpt-4")
executor = AgentExecutor(llm=my_llm)
"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "app.py")
        with open(self.path, "w", encoding="utf-8") as handle:
            handle.write(self.SOURCE)
        self.components = [
            _component(
                "ChatOpenAI",
                AIComponentType.MODEL,
                5,
                path=self.path,
                assigned="my_llm",
            ),
            _component(
                "AgentExecutor",
                AIComponentType.AGENT,
                6,
                path=self.path,
                args={"llm": "VARIABLE:my_llm"},
            ),
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pipeline_derives_edges_from_real_files(self):
        pipeline = ScanPipeline([self.tmp])
        edges, _, _ = pipeline._analyze_code_graph(self.components)
        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].relationship_type, RelationshipType.USES_MODEL)
        self.assertTrue(edges[0].is_code_derived)

    def test_function_passed_as_tool_becomes_a_component(self):
        path = os.path.join(self.tmp, "tools_app.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "def estimate_cost(x):\n"
                "    return x\n"
                "\n"
                "def unused_helper(x):\n"
                "    return x\n"
                "\n"
                "agent = LlmAgent(tools=[estimate_cost])\n"
            )
        components = [
            _component(
                "agent",
                AIComponentType.AGENT,
                7,
                path=path,
                args={"tools": ["VARIABLE:estimate_cost"]},
            ),
        ]
        edges, _, tools = ScanPipeline([self.tmp])._analyze_code_graph(components)

        self.assertEqual([t.name for t in tools], ["estimate_cost"])
        self.assertIs(tools[0].component_type, AIComponentType.TOOL)
        self.assertEqual(tools[0].line_number, 1)
        # The discovered tool is in the index before edges are derived, so
        # the reference that found it also links to it.
        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].relationship_type, RelationshipType.USES_TOOL)
        self.assertEqual(edges[0].target_name, "estimate_cost")

    def test_functions_not_registered_as_tools_are_left_alone(self):
        path = os.path.join(self.tmp, "plain.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("def helper(x):\n    return x\n")
        components = [
            _component("agent", AIComponentType.AGENT, 1, path=path, args={}),
        ]
        _, _, tools = ScanPipeline([self.tmp])._analyze_code_graph(components)
        self.assertEqual(tools, [])

    def test_env_toggle_disables_derivation(self):
        pipeline = ScanPipeline([self.tmp])
        with mock.patch.dict(os.environ, {"AIBOM_CODE_GRAPH": "0"}):
            self.assertEqual(
                pipeline._analyze_code_graph(self.components), ([], {}, [])
            )

    def test_derivation_failure_does_not_break_the_scan(self):
        pipeline = ScanPipeline([self.tmp])
        with mock.patch(
            "aibom.code_graph.derive_relationships", side_effect=RuntimeError("boom")
        ):
            self.assertEqual(
                pipeline._analyze_code_graph(self.components), ([], {}, [])
            )

    def test_unreadable_file_is_skipped_not_fatal(self):
        missing = _component(
            "Ghost",
            AIComponentType.AGENT,
            1,
            path=os.path.join(self.tmp, "gone.py"),
            args={"llm": "VARIABLE:nothing"},
        )
        pipeline = ScanPipeline([self.tmp])
        self.assertEqual(pipeline._analyze_code_graph([missing]), ([], {}, []))

    def test_derivation_runs_before_consolidation(self):
        """Order is load-bearing, not incidental.

        ``_stage_assemble`` merges components by (name, type), which drops
        the per-call-site ``arguments`` metadata the edges are read from.
        Deriving afterwards silently produced almost no edges on real
        repositories even though every unit test still passed.
        """
        pipeline = ScanPipeline([self.tmp], llm_config=None)
        order = []
        real_derive = pipeline._analyze_code_graph
        real_assemble = pipeline._stage_assemble

        def derive(components):
            order.append("derive")
            return real_derive(components)

        def assemble(components):
            order.append("assemble")
            return real_assemble(components)

        pipeline._analyze_code_graph = derive
        pipeline._stage_assemble = assemble
        pipeline.run()

        self.assertEqual(order, ["derive", "assemble"])


class TestGraphWiringEdges(unittest.TestCase):
    """LangGraph attaches nodes with add_node(), not constructor keywords.

    Keyword rules alone found nothing in four of the five benchmark repos,
    because the template repos that make up most modern LangChain code all
    wire their components this way.
    """

    def _edges(self, source, components, path="g.py"):
        graph = build_code_graph([_parse(path, source)])
        return derive_relationships(components, graph)

    def test_component_constructed_inside_the_call_is_attached(self):
        source = "\n".join(
            [
                "from langgraph.graph import StateGraph",
                "from langgraph.prebuilt import ToolNode",
                "builder = StateGraph(State)",
                'builder.add_node("tools", ToolNode(TOOLS))',
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 3, path="g.py", assigned="builder"
            ),
            _component("ToolNode", AIComponentType.TOOL, 2, path="g.py"),
        ]

        edges = self._edges(source, components)

        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].relationship_type, RelationshipType.USES_TOOL)
        self.assertEqual(edges[0].target_name, "ToolNode")
        self.assertTrue(edges[0].is_code_derived)

    def test_component_built_inside_an_attached_function_is_attached(self):
        source = "\n".join(
            [
                "from langgraph.graph import StateGraph",
                "def call_model(state):",
                '    model = ChatOpenAI(model="gpt-4")',
                "    return model",
                "builder = StateGraph(State)",
                "builder.add_node(call_model)",
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 5, path="g.py", assigned="builder"
            ),
            _component("ChatOpenAI", AIComponentType.MODEL, 3, path="g.py"),
        ]

        edges = self._edges(source, components)

        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].relationship_type, RelationshipType.USES_MODEL)

    def test_construction_one_helper_away_is_still_attached(self):
        """The indirection the call graph exists to see through."""
        source = "\n".join(
            [
                "from langgraph.graph import StateGraph",
                "def build_model():",
                '    return ChatOpenAI(model="gpt-4")',
                "def call_model(state):",
                "    return build_model()",
                "builder = StateGraph(State)",
                "builder.add_node(call_model)",
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 6, path="g.py", assigned="builder"
            ),
            _component("ChatOpenAI", AIComponentType.MODEL, 3, path="g.py"),
        ]

        edges = self._edges(source, components)

        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].relationship_type, RelationshipType.USES_MODEL)

    def test_construction_beyond_the_hop_limit_is_not_attached(self):
        source = "\n".join(
            [
                "from langgraph.graph import StateGraph",
                "def deepest():",
                '    return ChatOpenAI(model="gpt-4")',
                "def middle():",
                "    return deepest()",
                "def outer():",
                "    return middle()",
                "def call_model(state):",
                "    return outer()",
                "builder = StateGraph(State)",
                "builder.add_node(call_model)",
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 10, path="g.py", assigned="builder"
            ),
            _component("ChatOpenAI", AIComponentType.MODEL, 3, path="g.py"),
        ]

        self.assertEqual(self._edges(source, components), [])

    def test_add_edge_links_node_names_and_yields_nothing(self):
        source = "\n".join(
            [
                "from langgraph.graph import StateGraph",
                "builder = StateGraph(State)",
                'builder.add_edge("__start__", "ChatOpenAI")',
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 2, path="g.py", assigned="builder"
            ),
            _component("ChatOpenAI", AIComponentType.MODEL, 1, path="g.py"),
        ]

        self.assertEqual(self._edges(source, components), [])

    def test_unresolvable_receiver_yields_nothing(self):
        source = "\n".join(
            [
                "from langgraph.prebuilt import ToolNode",
                'unknown.add_node("tools", ToolNode(TOOLS))',
            ]
        )
        components = [_component("ToolNode", AIComponentType.TOOL, 1, path="g.py")]

        self.assertEqual(self._edges(source, components), [])

    def test_edge_kind_follows_the_target_type(self):
        """add_node() names no role, so the target must classify the edge."""
        source = "\n".join(
            [
                "from langgraph.graph import StateGraph",
                "builder = StateGraph(State)",
                'builder.add_node("store", Chroma(collection_name="c"))',
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 2, path="g.py", assigned="builder"
            ),
            _component("Chroma", AIComponentType.VECTOR_STORE, 1, path="g.py"),
        ]

        edges = self._edges(source, components)

        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].relationship_type, RelationshipType.USES_VECTOR_STORE)

    def test_untyped_target_is_not_attached(self):
        """A dependency or secret in a node body is not a USES_* edge."""
        source = "\n".join(
            [
                "from langgraph.graph import StateGraph",
                "def call_model(state):",
                "    return requests.get(url)",
                "builder = StateGraph(State)",
                "builder.add_node(call_model)",
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 4, path="g.py", assigned="builder"
            ),
            _component("requests", AIComponentType.DEPENDENCY, 3, path="g.py"),
        ]

        self.assertEqual(self._edges(source, components), [])

    def test_wiring_requires_a_graph(self):
        """Wiring calls live only in the graph, so no graph means no edges."""
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 2, path="g.py", assigned="builder"
            ),
            _component("ToolNode", AIComponentType.TOOL, 1, path="g.py"),
        ]
        self.assertEqual(derive_relationships(components, None), [])

    def test_wiring_carries_an_auditable_location(self):
        source = "\n".join(
            [
                "from langgraph.prebuilt import ToolNode",
                "builder = StateGraph(State)",
                'builder.add_node("tools", ToolNode(TOOLS))',
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 2, path="g.py", assigned="builder"
            ),
            _component("ToolNode", AIComponentType.TOOL, 1, path="g.py"),
        ]

        annotation = self._edges(source, components)[0].decision_annotation

        self.assertIn("graph_wiring", annotation.evidence_kinds)
        roles = {loc.role for loc in annotation.evidence_locations}
        self.assertEqual(roles, {"wiring_call", "attached_component"})
        self.assertEqual(annotation.evidence_locations[0].start_line, 3)


class TestReceiverTypeResolution(unittest.TestCase):
    """``obj.method()`` resolves through the receiver's type, not its name.

    Matching a dotted call on its trailing name alone fired 26 times across
    the benchmark repos and was wrong every time: ``meeting_flow.kickoff()``
    resolved to a module-level ``kickoff()`` sitting in the same file.
    """

    def _graph(self, source):
        return build_code_graph([_parse("m.py", source)])

    def test_call_on_a_locally_built_object_resolves_to_its_class(self):
        graph = self._graph(
            "\n".join(
                [
                    "class Engine:",
                    "    def start(self): pass",
                    "def main():",
                    "    eng = Engine()",
                    "    eng.start()",
                ]
            )
        )

        self.assertIn("m.py::Engine.start", graph.callees_of("m.py::main"))

    def test_call_on_a_library_object_does_not_match_a_local_name(self):
        """The bug the trailing-name fallback caused, as a test."""
        graph = self._graph(
            "\n".join(
                [
                    "class Engine:",
                    "    def start(self): pass",
                    "def main():",
                    "    flow = SomeLibraryFlow()",
                    "    flow.start()",
                ]
            )
        )

        self.assertEqual(graph.callees_of("m.py::main"), set())
        self.assertIn("flow.start", graph.unresolved_calls["m.py::main"])

    def test_a_variable_rebound_to_two_classes_is_not_typed(self):
        graph = self._graph(
            "\n".join(
                [
                    "class A:",
                    "    def run(self): pass",
                    "class B:",
                    "    def run(self): pass",
                    "def main():",
                    "    x = A()",
                    "    x = B()",
                    "    x.run()",
                ]
            )
        )

        self.assertEqual(graph.callees_of("m.py::main"), set())

    def test_two_classes_of_one_name_make_the_lookup_decline(self):
        results = [
            _parse("a.py", "class Dup:\n    def go(self): pass\n"),
            _parse("b.py", "class Dup:\n    def go(self): pass\n"),
            _parse("c.py", "def main():\n    d = Dup()\n    d.go()\n"),
        ]

        graph = build_code_graph(results)

        self.assertEqual(graph.callees_of("c.py::main"), set())

    def test_method_name_alone_never_creates_an_edge(self):
        """A same-file function with the right name is not a match."""
        graph = self._graph(
            "\n".join(
                [
                    "def kickoff():",
                    "    pass",
                    "def main():",
                    "    meeting_flow.kickoff()",
                ]
            )
        )

        self.assertEqual(graph.callees_of("m.py::main"), set())


class TestEvidenceStrength(unittest.TestCase):
    """Not every code-derived edge was read with the same directness."""

    def test_constructor_keyword_is_stated(self):
        source = "\n".join(
            [
                "client = ChatOpenAI(model='gpt-4')",
                "agent = AgentExecutor(llm=client)",
            ]
        )
        components = [
            _component(
                "ChatOpenAI", AIComponentType.MODEL, 1, path="g.py", assigned="client"
            ),
            _component(
                "AgentExecutor",
                AIComponentType.AGENT,
                2,
                path="g.py",
                args={"llm": "VARIABLE:client"},
            ),
        ]
        graph = build_code_graph([_parse("g.py", source)])

        edge = derive_relationships(components, graph)[0]

        self.assertIs(edge.evidence_strength, EvidenceStrength.STATED)
        self.assertTrue(edge.is_stated_in_source)

    def test_component_reached_through_a_node_function_is_not_stated(self):
        source = "\n".join(
            [
                "from langgraph.graph import StateGraph",
                "def call_model(state):",
                "    return ChatOpenAI(model='gpt-4')",
                "builder = StateGraph(State)",
                "builder.add_node(call_model)",
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 4, path="g.py", assigned="builder"
            ),
            _component("ChatOpenAI", AIComponentType.MODEL, 3, path="g.py"),
        ]
        graph = build_code_graph([_parse("g.py", source)])

        edge = derive_relationships(components, graph)[0]

        self.assertIs(edge.evidence_strength, EvidenceStrength.REACHED)
        self.assertTrue(edge.is_code_derived)
        self.assertFalse(edge.is_stated_in_source)

    def test_component_built_inside_the_wiring_call_is_stated(self):
        source = "\n".join(
            [
                "builder = StateGraph(State)",
                "builder.add_node('tools', ToolNode(TOOLS))",
            ]
        )
        components = [
            _component(
                "StateGraph", AIComponentType.AGENT, 1, path="g.py", assigned="builder"
            ),
            _component("ToolNode", AIComponentType.TOOL, 1, path="g.py"),
        ]
        graph = build_code_graph([_parse("g.py", source)])

        edge = derive_relationships(components, graph)[0]

        self.assertIs(edge.evidence_strength, EvidenceStrength.STATED)


class TestAmbiguousReferences(unittest.TestCase):
    """A name matching several components is reported, not discarded."""

    def _components(self):
        """Two same-named models in one file, and an agent naming one."""
        return [
            _component("llm", AIComponentType.MODEL, 1, path="g.py"),
            _component("llm", AIComponentType.MODEL, 2, path="g.py"),
            _component(
                "Agent",
                AIComponentType.AGENT,
                3,
                path="g.py",
                args={"llm": "VARIABLE:llm"},
            ),
        ]

    def test_every_candidate_is_emitted_as_ambiguous(self):
        edges = derive_relationships(self._components())

        self.assertEqual(len(edges), 2)
        for edge in edges:
            self.assertIs(edge.evidence_strength, EvidenceStrength.AMBIGUOUS)
            self.assertEqual(edge.decision_annotation.decision, "REVIEW")

    def test_ambiguous_edge_names_its_rivals_for_review(self):
        annotation = derive_relationships(self._components())[0].decision_annotation

        self.assertIn("ambiguous_reference", annotation.evidence_kinds)
        self.assertIn("matches 2 components", annotation.justification)

    def test_a_single_clear_match_is_not_ambiguous(self):
        components = [
            _component("llm", AIComponentType.MODEL, 1, path="g.py"),
            _component(
                "Agent",
                AIComponentType.AGENT,
                3,
                path="g.py",
                args={"llm": "VARIABLE:llm"},
            ),
        ]

        edges = derive_relationships(components)

        self.assertEqual(len(edges), 1)
        self.assertIs(edges[0].evidence_strength, EvidenceStrength.STATED)

    def test_candidates_of_the_wrong_type_are_not_reported(self):
        components = [
            _component("llm", AIComponentType.TOOL, 1, path="g.py"),
            _component("llm", AIComponentType.TOOL, 2, path="g.py"),
            _component(
                "Agent",
                AIComponentType.AGENT,
                3,
                path="g.py",
                args={"llm": "VARIABLE:llm"},
            ),
        ]

        self.assertEqual(derive_relationships(components), [])

    def test_a_common_word_is_not_worth_listing(self):
        components = [
            _component("llm", AIComponentType.MODEL, n, path="g.py")
            for n in range(1, 7)
        ] + [
            _component(
                "Agent",
                AIComponentType.AGENT,
                9,
                path="g.py",
                args={"llm": "VARIABLE:llm"},
            )
        ]

        self.assertEqual(derive_relationships(components), [])


class TestEvidenceRanking(unittest.TestCase):
    """Stronger evidence wins wherever two edges disagree."""

    def _rel(self, target, strength, source_id="agent-1"):
        return ComponentRelationship(
            source_instance_id=source_id,
            target_instance_id=f"t-{target}",
            relationship_type=RelationshipType.USES_MODEL,
            source_name="Agent",
            target_name=target,
            detection_source=(DetectionSource.CODE_ANALYSIS if strength else None),
            evidence_strength=strength,
        )

    def test_stated_beats_reached_when_writing_model_name(self):
        agent = _component("Agent", AIComponentType.AGENT, 1)
        rels = [
            self._rel("reached-guess", EvidenceStrength.REACHED, agent.instance_id),
            self._rel("stated-truth", EvidenceStrength.STATED, agent.instance_id),
        ]

        updated = _propagate_model_from_relationships([agent], rels)

        self.assertEqual(updated[0].model_name, "stated-truth")

    def test_reached_does_not_lose_to_a_later_llm_guess(self):
        agent = _component("Agent", AIComponentType.AGENT, 1)
        rels = [
            self._rel("from-source", EvidenceStrength.REACHED, agent.instance_id),
            self._rel("llm-guess", None, agent.instance_id),
        ]

        updated = _propagate_model_from_relationships([agent], rels)

        self.assertEqual(updated[0].model_name, "from-source")

    def test_ambiguous_never_writes_a_model_name(self):
        agent = _component("Agent", AIComponentType.AGENT, 1)
        rels = [self._rel("a-guess", EvidenceStrength.AMBIGUOUS, agent.instance_id)]

        updated = _propagate_model_from_relationships([agent], rels)

        self.assertIsNone(updated[0].model_name)

    def test_equally_evidenced_rivals_write_nothing(self):
        """A provider picked by config is not a model the BOM can report.

        The retrieval templates reach both OpenAIEmbeddings and
        CohereEmbeddings from one graph node; last-wins reported whichever
        was seen last as fact.
        """
        agent = _component("Agent", AIComponentType.AGENT, 1)
        rels = [
            self._rel("OpenAIEmbeddings", EvidenceStrength.REACHED, agent.instance_id),
            self._rel("CohereEmbeddings", EvidenceStrength.REACHED, agent.instance_id),
        ]

        updated = _propagate_model_from_relationships([agent], rels)

        self.assertIsNone(updated[0].model_name)

    def test_a_stated_edge_settles_a_disagreement_among_weaker_ones(self):
        agent = _component("Agent", AIComponentType.AGENT, 1)
        rels = [
            self._rel("OpenAIEmbeddings", EvidenceStrength.REACHED, agent.instance_id),
            self._rel("CohereEmbeddings", EvidenceStrength.REACHED, agent.instance_id),
            self._rel("TheRealOne", EvidenceStrength.STATED, agent.instance_id),
        ]

        updated = _propagate_model_from_relationships([agent], rels)

        self.assertEqual(updated[0].model_name, "TheRealOne")

    def test_dedup_keeps_the_best_evidenced_copy(self):
        weak = self._rel("ChatOpenAI", EvidenceStrength.REACHED)
        strong = self._rel("ChatOpenAI", EvidenceStrength.STATED)

        kept = _dedup_relationships([weak, strong])

        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0].evidence_strength, EvidenceStrength.STATED)

    def test_dedup_is_order_independent(self):
        weak = self._rel("ChatOpenAI", EvidenceStrength.REACHED)
        strong = self._rel("ChatOpenAI", EvidenceStrength.STATED)

        kept = _dedup_relationships([strong, weak])

        self.assertIs(kept[0].evidence_strength, EvidenceStrength.STATED)


class TestEmptyInputs(unittest.TestCase):
    def test_no_results_yields_empty_graph(self):
        self.assertEqual(build_code_graph([]).stats()["functions"], 0)

    def test_no_components_yields_no_edges(self):
        self.assertEqual(derive_relationships([]), [])

    def test_components_without_metadata_yield_no_edges(self):
        comps = [_component("X", AIComponentType.MODEL, 1)]
        self.assertEqual(derive_relationships(comps), [])


if __name__ == "__main__":
    unittest.main()
