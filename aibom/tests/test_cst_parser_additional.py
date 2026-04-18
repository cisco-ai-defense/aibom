# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# SPDX-License-Identifier: Apache-2.0

from aibom import cst_parser


def test_parse_source_code_collects_imports_decorators_and_context():
    source = """
import os
import pkg
import contextlib

@pkg.get("/route")
def handler():
    with contextlib.nullcontext() as ctx:
        pass
"""
    result = cst_parser.parse_source_code("file.py", source)

    # Imports captured (tuples of (line_number, import_string))
    import_strs = [stmt for _, stmt in result.imports]
    assert "import os" in import_strs
    assert "import pkg" in import_strs

    # Decorator captured
    assert len(result.decorators) == 1
    assert result.decorators[0].decorator_qualified_name.endswith("pkg.get")
    assert result.decorators[0].decorated_function_name == "handler"

    # Context manager captured
    assert len(result.context_managers) == 1
    ctx = result.context_managers[0]
    assert ctx.context_expr_qualified_name.endswith("contextlib.nullcontext")
    assert ctx.as_target.endswith("ctx")


def test_parse_source_code_handles_annotations_and_raw_code():
    source = '''
from pkg import Model

typed: Model = Model(param="value")
'''
    result = cst_parser.parse_source_code("typed.py", source)

    # Annotated assignment captured
    assert len(result.type_annotations) == 1
    ann = result.type_annotations[0]
    assert ann.target_qualified_name == "typed"
    assert ann.annotation_qualified_name.endswith("Model")

    # Assignment call captured with raw code snippet
    assert len(result.assignments) == 1
    assignment = result.assignments[0]
    assert assignment.call.qualified_name.endswith("Model")
    assert 'param="value"' in assignment.call.raw_code


# ---------------------------------------------------------------------------
# Phase 1: structural observations
# ---------------------------------------------------------------------------


def _find_shape(result, method_name):
    return [s for s in result.method_shapes if s.method_name == method_name]


def _find_control_flow(result, owner_method):
    return [c for c in result.control_flows if c.owner_method_name == owner_method]


# -- ControlFlowObservation --------------------------------------------------


def test_control_flow_captures_while_loop_body_calls():
    source = """
def run():
    while not done():
        result = llm.invoke(prompt)
        tool.execute(result)
"""
    result = cst_parser.parse_source_code("react.py", source)

    flows = _find_control_flow(result, "run")
    assert len(flows) == 1
    flow = flows[0]
    assert flow.loop_kind == "while"
    assert flow.owner_class_name is None
    body_calls = set(flow.body_call_qualified_names)
    assert any(c.endswith("llm.invoke") for c in body_calls)
    assert any(c.endswith("tool.execute") for c in body_calls)
    # ``done()`` appears only in the ``while`` *test* expression, not in the
    # loop body. It must NOT be counted as a body call — otherwise loop
    # predicates would inflate the react-loop detector's distinct-callee
    # count (e.g. ``for _ in range(10):`` would also pull ``range`` in).
    assert not any(c.endswith("done") for c in body_calls)


def test_control_flow_captures_for_loop_and_marks_branch():
    source = """
def step():
    for step in range(10):
        if should_stop():
            break
        act()
"""
    result = cst_parser.parse_source_code("loop.py", source)

    flows = _find_control_flow(result, "step")
    assert len(flows) == 1
    flow = flows[0]
    assert flow.loop_kind == "for"
    assert flow.has_branch is True
    assert any(c.endswith("act") for c in flow.body_call_qualified_names)


def test_control_flow_attributes_nested_loops_separately():
    source = """
def nested():
    for i in range(3):
        while running():
            tool_one()
"""
    result = cst_parser.parse_source_code("nested.py", source)

    flows = _find_control_flow(result, "nested")
    kinds = sorted(f.loop_kind for f in flows)
    assert kinds == ["for", "while"]
    while_flow = next(f for f in flows if f.loop_kind == "while")
    assert any(c.endswith("tool_one") for c in while_flow.body_call_qualified_names)


def test_control_flow_async_for_recorded_as_async_for():
    source = """
async def consume():
    async for msg in stream():
        handle(msg)
"""
    result = cst_parser.parse_source_code("async.py", source)

    flows = _find_control_flow(result, "consume")
    assert len(flows) == 1
    assert flows[0].loop_kind == "async for"


def test_control_flow_does_not_capture_module_level_loops():
    source = """
while True:
    do_thing()
"""
    result = cst_parser.parse_source_code("module.py", source)
    assert result.control_flows == []


# -- MethodBodyShapeObservation ---------------------------------------------


def test_method_shape_single_llm_call_wrapper():
    source = """
def ask(prompt):
    return llm.invoke(prompt)
"""
    result = cst_parser.parse_source_code("wrapper.py", source)

    shapes = _find_shape(result, "ask")
    assert len(shapes) == 1
    shape = shapes[0]
    assert shape.loop_count == 0
    assert shape.call_count == 1
    assert shape.return_count == 1
    assert shape.branch_count == 0
    assert any(c.endswith("llm.invoke") for c in shape.called_qualified_names)


def test_method_shape_react_loop_signature():
    source = """
def run(self):
    while not self.done:
        if self.should_stop():
            break
        out = self.llm.invoke(self.prompt)
        self.dispatch(out)
        return out
"""
    result = cst_parser.parse_source_code("react.py", source)

    shapes = _find_shape(result, "run")
    assert len(shapes) == 1
    shape = shapes[0]
    assert shape.loop_count == 1
    assert shape.call_count >= 3
    assert shape.branch_count >= 1
    assert shape.return_count == 1


def test_method_shape_class_name_attribution():
    source = """
class Orchestrator:
    def run(self):
        while True:
            self.tick()
"""
    result = cst_parser.parse_source_code("orch.py", source)

    shapes = _find_shape(result, "run")
    assert len(shapes) == 1
    assert shapes[0].owner_class_name == "Orchestrator"

    flows = _find_control_flow(result, "run")
    assert len(flows) == 1
    assert flows[0].owner_class_name == "Orchestrator"


def test_method_shape_nested_function_does_not_leak_to_outer():
    source = """
def outer():
    def inner():
        foo()
        bar()
    return inner
"""
    result = cst_parser.parse_source_code("nested.py", source)

    outer_shapes = _find_shape(result, "outer")
    inner_shapes = _find_shape(result, "inner")
    assert len(outer_shapes) == 1
    assert len(inner_shapes) == 1

    outer_calls = outer_shapes[0].called_qualified_names
    assert not any(c.endswith("foo") for c in outer_calls)
    assert not any(c.endswith("bar") for c in outer_calls)

    inner_calls = inner_shapes[0].called_qualified_names
    assert any(c.endswith("foo") for c in inner_calls)
    assert any(c.endswith("bar") for c in inner_calls)


def test_method_shape_lambda_body_calls_do_not_leak_to_outer():
    source = """
def outer():
    fn = lambda x: secret_call(x)
    return fn
"""
    result = cst_parser.parse_source_code("lam.py", source)

    outer_shapes = _find_shape(result, "outer")
    assert len(outer_shapes) == 1
    outer_calls = outer_shapes[0].called_qualified_names
    assert not any(c.endswith("secret_call") for c in outer_calls)


def test_method_shape_dedupes_called_qualified_names():
    source = """
def chatty():
    log("a")
    log("b")
    log("c")
"""
    result = cst_parser.parse_source_code("chatty.py", source)

    shapes = _find_shape(result, "chatty")
    assert len(shapes) == 1
    assert shapes[0].call_count == 3
    assert sum(1 for c in shapes[0].called_qualified_names if c.endswith("log")) == 1


def test_method_shape_async_functions_captured():
    source = """
async def fetch():
    await client.get("/x")
"""
    result = cst_parser.parse_source_code("afetch.py", source)

    shapes = _find_shape(result, "fetch")
    assert len(shapes) == 1
    assert any(c.endswith("client.get") for c in shapes[0].called_qualified_names)


# -- StringLiteralObservation -----------------------------------------------


def test_protocol_strings_capture_urls_and_jsonrpc_methods():
    source = """
def send():
    url = "https://agents.example.com/.well-known/agent.json"
    path = "/.well-known/agent.json"
    rpc = "message/send"
    plain = "this is just a description"
"""
    result = cst_parser.parse_source_code("proto.py", source)

    values = [s.value for s in result.protocol_strings]
    assert "https://agents.example.com/.well-known/agent.json" in values
    assert "/.well-known/agent.json" in values
    assert "message/send" in values
    assert "this is just a description" not in values


def test_protocol_strings_captured_at_class_and_method_scope_but_not_module():
    """Module-level string literals are still ignored — they are rarely part
    of protocol configuration that the evidence builder cares about. But
    class-level string literals (e.g. an A2A Agent Card path assigned to a
    class attribute) and method-local literals are both captured.
    """
    source = """
MODULE_URL = "https://example.com/hook"

class Svc:
    CLASS_URL = "https://cls.example.com/hook"

    def m(self):
        local = "https://local.example.com/hook"
"""
    result = cst_parser.parse_source_code("scope.py", source)

    values = [s.value for s in result.protocol_strings]
    assert "https://example.com/hook" not in values
    assert "https://cls.example.com/hook" in values
    assert "https://local.example.com/hook" in values

    by_value = {s.value: s for s in result.protocol_strings}
    cls_attr = by_value["https://cls.example.com/hook"]
    assert cls_attr.owner_class_name == "Svc"
    assert cls_attr.owner_method_name is None
    method_local = by_value["https://local.example.com/hook"]
    assert method_local.owner_class_name == "Svc"
    assert method_local.owner_method_name == "m"


def test_protocol_strings_reject_oversized_literals():
    long_url = "https://example.com/" + ("x" * 600)
    source = f'''
def m():
    u = "{long_url}"
'''
    result = cst_parser.parse_source_code("big.py", source)
    assert result.protocol_strings == []


# -- ClassBodyFactsObservation ----------------------------------------------


def test_class_body_facts_collect_methods_and_base_classes():
    source = """
class Base:
    pass

class Orchestrator(Base, Mixin):
    def __init__(self):
        self.state = 0

    def run(self):
        pass

    async def aclose(self):
        pass
"""
    result = cst_parser.parse_source_code("cbf.py", source)

    assert len(result.class_bodies) == 2
    by_name = {c.class_name: c for c in result.class_bodies}

    assert by_name["Base"].base_classes == []
    assert by_name["Base"].method_names == []

    orch = by_name["Orchestrator"]
    assert orch.base_classes == ["Base", "Mixin"]
    assert orch.method_names == ["__init__", "run", "aclose"]


def test_class_body_facts_include_full_body_source():
    source = """
class Runner:
    def run(self):
        return 42
"""
    result = cst_parser.parse_source_code("r.py", source)

    assert len(result.class_bodies) == 1
    facts = result.class_bodies[0]
    assert "class Runner" in facts.body_source
    assert "def run" in facts.body_source
    assert "return 42" in facts.body_source
    assert facts.start_line < facts.end_line


def test_class_body_facts_nested_class_in_function_still_captured():
    source = """
def make():
    class Tmp:
        def do(self):
            pass
    return Tmp
"""
    result = cst_parser.parse_source_code("nested_cls.py", source)

    names = [c.class_name for c in result.class_bodies]
    assert names == ["Tmp"]


# -- Interaction: calls inside class bodies do not leak to enclosing functions
#    and calls inside class-body methods DO count in the method --------------


def test_class_body_calls_do_not_leak_into_enclosing_function():
    source = """
def outer():
    class Inner:
        x = make_thing()

    foo()
"""
    result = cst_parser.parse_source_code("leak.py", source)

    outer_shape = _find_shape(result, "outer")[0]
    assert any(c.endswith("foo") for c in outer_shape.called_qualified_names)
    assert not any(c.endswith("make_thing") for c in outer_shape.called_qualified_names)
