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
