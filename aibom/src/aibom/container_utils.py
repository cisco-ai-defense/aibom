#!/usr/bin/env python3
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

"""
Docker Utilities

This module provides utility functions for working with Docker images,
including detecting Docker images, extracting files from Docker containers,
and inspecting Docker images.
"""

import os
import json
import logging
import subprocess
from typing import Dict, Any, Optional


def is_docker_image(source: str) -> bool:
    """
    Auto-detect if source is a Docker image name or local directory.
    
    Args:
        source: Path or Docker image name to check
        
    Returns:
        True if it looks like a Docker image, False otherwise
    """
    # If it's an existing directory, it's not a Docker image
    if os.path.isdir(source):
        return False
    
    # If it's an existing file, it's not a Docker image
    if os.path.isfile(source):
        return False
    
    # Check if Docker is available
    try:
        subprocess.run(['docker', '--version'], 
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.warning("Docker not found. Treating source as a local directory.")
        return False
    
    # Try to inspect the image to see if it exists
    try:
        result = subprocess.run(['docker', 'image', 'inspect', source], 
                              capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        # Image doesn't exist locally, try to pull it
        logging.info(f"Docker image '{source}' not found locally. Attempting to pull...")
        try:
            subprocess.run(['docker', 'pull', source], 
                          capture_output=True, check=True)
            logging.info(f"Successfully pulled Docker image: {source}")
            return True
        except subprocess.CalledProcessError:
            logging.warning(f"Could not pull Docker image '{source}'. Treating as a local directory.")
            return False


def extract_app_from_docker(image_name: str, output_dir: str) -> Dict[str, Any]:
    """
    Extracts files from a Docker image for analysis.
    If an /app directory exists, it extracts that. Otherwise, it extracts site-packages.

    Args:
        image_name: Name of the Docker image.
        output_dir: Directory to extract files to.

    Returns:
        A dictionary containing extraction information or an error.
    """
    container_id = None
    try:
        logging.info(f"Starting a long-running container from image: {image_name}")
        container_id = subprocess.check_output(
            ['docker', 'run', '-d', '--entrypoint', 'sleep', image_name, 'infinity'],
            text=True
        ).strip()

        path_to_scan = None
        app_check_command = 'if [ -d /app ]; then echo "exists"; else echo "not_exists"; fi'
        app_exists = False
        try:
            app_exists_output = subprocess.check_output(
                ['docker', 'exec', container_id, 'sh', '-c', app_check_command], text=True
            ).strip()
            if app_exists_output == "exists":
                app_exists = True
        except subprocess.CalledProcessError as e:
            logging.warning(f"Could not check for /app directory: {e}. Assuming it doesn't exist.")

        if app_exists:
            logging.info("Found /app directory. It will be the exclusive target for scanning.")
            app_dir = os.path.join(output_dir, 'app')
            os.makedirs(app_dir, exist_ok=True)
            subprocess.run(['docker', 'cp', f'{container_id}:/app/.', app_dir], check=True)
            path_to_scan = app_dir
        else:
            logging.info("/app directory not found. Scanning site-packages instead.")
            try:
                site_packages_output = subprocess.check_output([
                    'docker', 'exec', container_id, 'python', '-c',
                    'import site, json; print(json.dumps(site.getsitepackages()))'
                ], text=True).strip()
                site_packages_paths = json.loads(site_packages_output)

                for i, sp_path in enumerate(site_packages_paths):
                    if not sp_path:
                        continue
                    logging.info(f"Copying site-packages from {sp_path}")
                    sp_dir = os.path.join(output_dir, f'site-packages-{i}')
                    os.makedirs(sp_dir, exist_ok=True)
                    subprocess.run(['docker', 'cp', f'{container_id}:{sp_path}/.', sp_dir], check=True)
                path_to_scan = output_dir
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                logging.warning(f"Could not find or copy site-packages: {e}")
                return {'error': f'Could not find site-packages: {e}'}

        return {
            'image_name': image_name,
            'extracted_to': path_to_scan,
            'image_info': get_docker_image_info(image_name)
        }

    except subprocess.CalledProcessError as e:
        logging.error(f"Error during Docker operation: {e}")
        return {'error': str(e)}
    finally:
        if container_id:
            logging.info(f"Cleaning up container: {container_id}")
            subprocess.run(['docker', 'stop', container_id], capture_output=True, check=False)
            subprocess.run(['docker', 'rm', container_id], capture_output=True, check=False)


def get_docker_image_info(image_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a Docker image.
    
    Args:
        image_name: Name of the Docker image
        
    Returns:
        Dictionary containing image information or None if error
    """
    try:
        # Get image info
        image_info = subprocess.check_output([
            'docker', 'inspect', image_name
        ], text=True)
        
        return json.loads(image_info)[0]
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting Docker image info: {e}")
        return None
