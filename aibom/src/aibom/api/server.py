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

import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..models.component import ComponentModel, ComponentResponse, ComponentTypesResponse
from ..utils.dataframe_converter import dataframe_to_components
from ..utils.version import resolve_package_version


API_VERSION = resolve_package_version("cisco-aibom")


class AIBOMServer:
    """FastAPI server for serving AI BOM component data."""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.app = FastAPI(
            title="AI BOM API",
            description="API for AI BOM component data",
            version=API_VERSION
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/api/components", response_model=ComponentResponse)
        async def get_components(
            type: Optional[str] = Query(None, description="Filter by component type"),
            file_path: Optional[str] = Query(None, description="Filter by file path substring")
        ):
            """Get all components with optional filtering."""
            df_filtered = self.df.copy()
            
            # Apply filters
            if type:
                df_filtered = df_filtered[df_filtered['type'] == type]
            
            if file_path:
                df_filtered = df_filtered[df_filtered['file_path'].str.contains(file_path, na=False)]
            
            components = dataframe_to_components(df_filtered)
            
            return ComponentResponse(
                components=components,
                total=len(components)
            )
        
        @self.app.get("/api/components/types", response_model=ComponentTypesResponse)
        async def get_component_types():
            """Get unique component types."""
            unique_types = self.df['type'].unique().tolist()
            return ComponentTypesResponse(types=unique_types)
        
        @self.app.get("/api/components/{component_id}", response_model=ComponentModel)
        async def get_component(component_id: str):
            """Get a specific component by ID."""
            component_row = self.df[self.df['id'] == component_id]
            
            if component_row.empty:
                raise HTTPException(status_code=404, detail="Component not found")
            
            components = dataframe_to_components(component_row)
            return components[0]
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "total_components": len(self.df)}
    
    def get_app(self):
        """Get the FastAPI application instance."""
        return self.app


def create_server(dataframe: pd.DataFrame) -> FastAPI:
    """Create and return a FastAPI server instance."""
    server = AIBOMServer(dataframe)
    return server.get_app()
