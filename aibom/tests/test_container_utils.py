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
import subprocess
import os

from aibom.container_utils import (
    is_docker_image,
    extract_app_from_docker,
    get_docker_image_info
)

class TestContainerUtils(unittest.TestCase):

    @patch('subprocess.run')
    @patch('os.path.isfile')
    @patch('os.path.isdir')
    def test_is_docker_image(self, mock_isdir, mock_isfile, mock_run):
        # Test with a local directory
        mock_isdir.return_value = True
        self.assertFalse(is_docker_image('/some/local/dir'))

        # Test with a local file
        mock_isdir.return_value = False
        mock_isfile.return_value = True
        self.assertFalse(is_docker_image('/some/local/file'))

        # Test with a docker image that exists locally
        mock_isfile.return_value = False
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(is_docker_image('my-image:latest'))

        # Test with a docker image that needs to be pulled
        mock_run.side_effect = [
            MagicMock(returncode=0), # docker --version succeeds
            subprocess.CalledProcessError(1, 'docker'), # docker image inspect fails
            MagicMock(returncode=0) # docker pull succeeds
        ]
        self.assertTrue(is_docker_image('another-image:latest'))

    @patch('aibom.container_utils.get_docker_image_info', return_value={})
    @patch('subprocess.check_output')
    @patch('subprocess.run')
    @patch('os.makedirs')
    def test_extract_app_from_docker_app_directory(self, mock_makedirs, mock_run, mock_check_output, mock_get_info):
        mock_check_output.side_effect = [
            'container123', # docker run
            'exists'      # check for /app
        ]
        result = extract_app_from_docker('my-image', './output')
        self.assertEqual(result['extracted_to'], os.path.join('./output', 'app'))
        # Check cleanup calls
        self.assertIn('stop', mock_run.call_args_list[-2][0][0])
        self.assertIn('rm', mock_run.call_args_list[-1][0][0])

    @patch('json.loads', return_value=['/usr/lib/python3.8/site-packages'])
    @patch('aibom.container_utils.get_docker_image_info', return_value={})
    @patch('subprocess.check_output')
    @patch('subprocess.run')
    @patch('os.makedirs')
    def test_extract_app_from_docker_with_site_packages(self, mock_makedirs, mock_run, mock_check_output, mock_get_info, mock_json_loads):
        mock_check_output.side_effect = [
            b'container123\n', # docker run
            b'not_exists\n', # check for /app
            b'["/path/to/site-packages"]' # get site-packages path
        ]
        result = extract_app_from_docker('my-image', './output')
        self.assertEqual(result['extracted_to'], './output')

    @patch('subprocess.check_output')
    def test_get_docker_image_info_success(self, mock_check_output):
        mock_check_output.return_value = b'[{"Id": "123"}]'
        info = get_docker_image_info('my-image')
        self.assertEqual(info['Id'], '123')

    @patch('subprocess.check_output', side_effect=subprocess.CalledProcessError(1, 'docker'))
    def test_get_docker_image_info_fail(self, mock_check_output):
        info = get_docker_image_info('my-image')
        self.assertIsNone(info)

if __name__ == '__main__':
    unittest.main()
