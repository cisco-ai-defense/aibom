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

import unittest
import libcst as cst

from aibom.cst_parser import parse_source_code, _extract_argument_value

class TestCstParser(unittest.TestCase):

    def test_parse_assignment(self):
        code = "from my_lib import MyClass\n\nmy_instance = MyClass(name='test', value=123)"
        result = parse_source_code('test.py', code)

        self.assertEqual(len(result.assignments), 1)
        assignment = result.assignments[0]
        self.assertEqual(assignment.target_qualified_name, 'my_instance')
        self.assertEqual(assignment.call.qualified_name, 'my_lib.MyClass')
        self.assertEqual(assignment.call.arguments, {'name': 'test', 'value': 123})

    def test_parse_decorator(self):
        code = "from my_decorators import tool\n\n@tool\ndef my_function():\n    pass"
        result = parse_source_code('test.py', code)

        self.assertEqual(len(result.decorators), 1)
        decorator = result.decorators[0]
        self.assertEqual(decorator.decorator_qualified_name, 'my_decorators.tool')
        self.assertEqual(decorator.decorated_function_name, 'my_function')

    def test_parse_imports(self):
        code = "import os\nfrom sys import argv\nimport pandas as pd"
        result = parse_source_code('test.py', code)

        self.assertIn('import os', result.imports)
        self.assertIn('from sys import argv', result.imports)
        self.assertIn('import pandas as pd', result.imports)

    def test_extract_argument_value(self):
        self.assertEqual(_extract_argument_value(cst.SimpleString('"hello"')), 'hello')
        self.assertEqual(_extract_argument_value(cst.Integer('123')), 123)
        self.assertEqual(_extract_argument_value(cst.Name('True')), True)
        self.assertEqual(_extract_argument_value(cst.Name('my_var')), 'VARIABLE:my_var')
        # A valid Call node requires a function
        dummy_call_node = cst.Call(func=cst.Name("dummy"))
        self.assertEqual(_extract_argument_value(dummy_call_node), 'COMPLEX_TYPE:Call')

    def test_syntax_error(self):
        code = "a = (\n"
        result = parse_source_code('test.py', code)
        self.assertEqual(len(result.assignments), 0)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(len(result.decorators), 0)

if __name__ == '__main__':
    unittest.main()
