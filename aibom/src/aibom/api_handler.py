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

import uvicorn
import logging
import threading
import time
from typing import Dict, List, Any

from .utils.dataframe_converter import convert_to_dataframe
from .api.server import create_server


def start_api_server(
    all_categorized_components: Dict[str, Dict[str, List[Dict[str, Any]]]],
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """
    Start the FastAPI API server with the analyzed components.

    Args:
        all_categorized_components: The categorized components from analysis
        host: Server host address
        port: Server port
    """
    # Convert to DataFrame
    df = convert_to_dataframe(all_categorized_components)

    if df.empty:
        logging.warning(
            "No components found to serve. Starting API server with empty data."
        )

    logging.info(f"Starting AI BOM API server with {len(df)} components...")
    logging.info(f"Server will be available at: http://{host}:{port}")
    logging.info("\nAPI Endpoints:")
    logging.info(f"  - GET http://{host}:{port}/api/components - Get all components")
    logging.info(f"  - GET http://{host}:{port}/api/components/{{id}} - Get specific component")
    logging.info(f"  - GET http://{host}:{port}/api/components/types - Get component types")
    logging.info(f"  - GET http://{host}:{port}/api/components?type={{type}} - Filter by type")
    logging.info(f"  - GET http://{host}:{port}/api/components?file_path={{path}} - Filter by file path")
    logging.info(f"  - GET http://{host}:{port}/health - Health check")

    # Create FastAPI app
    app = create_server(df)

    # Start server
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        logging.info("\nServer stopped by user.")
    except Exception as e:
        logging.error(f"Server error: {e}")


def run_api_server_async(
    all_categorized_components: Dict[str, Dict[str, List[Dict[str, Any]]]],
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """
    Run the API server in a separate thread (for testing purposes).

    Args:
        all_categorized_components: The categorized components from analysis
        host: Server host address
        port: Server port
    """

    def server_thread():
        start_api_server(all_categorized_components, host, port)

    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()

    # Give server time to start
    time.sleep(2)
    return thread
