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
from unittest.mock import patch, MagicMock
import pandas as pd

from aibom.ui_handler import start_ui_server, run_ui_server_async

class TestUiHandler(unittest.TestCase):

    @patch('uvicorn.run')
    @patch('aibom.ui_handler.create_server')
    @patch('aibom.ui_handler.convert_to_dataframe')
    def test_start_ui_server_success(self, mock_convert, mock_create_server, mock_run):
        # Arrange
        mock_df = pd.DataFrame({'col1': [1, 2]})
        mock_convert.return_value = mock_df
        mock_app = MagicMock()
        mock_create_server.return_value = mock_app
        categorized_components = {'app': {'type': [{'data': 'dummy'}]}}

        # Act
        start_ui_server(categorized_components, host='localhost', port=8080)

        # Assert
        mock_convert.assert_called_once_with(categorized_components)
        mock_create_server.assert_called_once_with(mock_df)
        mock_run.assert_called_once_with(mock_app, host='localhost', port=8080, log_level="info")

    @patch('uvicorn.run')
    @patch('aibom.ui_handler.create_server')
    @patch('aibom.ui_handler.convert_to_dataframe')
    def test_start_ui_server_empty_dataframe(self, mock_convert, mock_create_server, mock_run):
        # Arrange
        mock_convert.return_value = pd.DataFrame()
        categorized_components = {}

        # Act
        start_ui_server(categorized_components)

        # Assert
        mock_convert.assert_called_once_with(categorized_components)
        mock_create_server.assert_not_called()
        mock_run.assert_not_called()

    @patch('time.sleep')
    @patch('threading.Thread')
    @patch('aibom.ui_handler.start_ui_server')
    def test_run_ui_server_async(self, mock_start_server, mock_thread_class, mock_sleep):
        # Arrange
        mock_thread_instance = MagicMock()
        mock_thread_class.return_value = mock_thread_instance
        categorized_components = {'app': {'type': [{'data': 'dummy'}]}}

        # Act
        thread = run_ui_server_async(categorized_components, host='127.0.0.1', port=8000)

        # Assert
        self.assertEqual(thread, mock_thread_instance)
        mock_thread_class.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        mock_sleep.assert_called_once_with(2)
        # Check that start_ui_server would have been called in the thread
        thread_target_func = mock_thread_class.call_args.kwargs['target']
        thread_target_func() # Manually call the target function
        mock_start_server.assert_called_once_with(categorized_components, '127.0.0.1', 8000)

if __name__ == '__main__':
    unittest.main()
