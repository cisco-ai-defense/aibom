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

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

from .structures import (
    CodeAnalysisResult,
    AssignmentObservation,
    CallObservation,
    DecoratorObservation,
    ComponentRelationship,
    CategorizationOutput,
)
from .catalog_db import CatalogDB
from .llm_client import LLMClient
from .workflow_analyzer import WorkflowIndex


AGENT_CATEGORY_HINTS = {"agent"}
TOOL_CATEGORY_HINTS = {"tool"}
LLM_CATEGORY_HINTS = {"llm", "model"}
TOOL_ARGUMENT_HINTS = {"tool", "tools", "skills", "abilities"}
LLM_ARGUMENT_HINTS = {"llm", "language_model", "chat_model", "model"}
RELATIONSHIP_LABEL_TOOL = "USES_TOOL"
RELATIONSHIP_LABEL_LLM = "USES_LLM"

def categorize_symbols(
    analysis_results: List[CodeAnalysisResult],
    connector: CatalogDB,
    llm_config: Optional[Dict[str, Any]] = None,
    workflow_index: Optional[WorkflowIndex] = None,
) -> CategorizationOutput:
    """
    Orchestrates the symbol matching and categorization process using the 'concept' attribute from the catalog.
    Enhanced to handle multiple matches, model extraction, and better import disambiguation.
    """
    # Initialize LLM client if config provided
    llm_client = None
    if llm_config:
        try:
            llm_client = LLMClient(llm_config)
            logging.info("LLM client initialized for model extraction.")
        except Exception as e:
            logging.warning(f"Failed to initialize LLM client: {e}")
    
    # Build variable mapping from assignments FIRST for decorator reconstruction
    variable_map = {}
    for result in analysis_results:
        for assignment in result.assignments:
            if assignment.target_qualified_name and assignment.call.qualified_name:
                variable_map[assignment.target_qualified_name] = assignment.call.qualified_name

    # Collect all observations with enhanced tracking
    all_observations = []
    symbols_to_find = set()
    
    for result in analysis_results:
        for obs_type in ["assignments", "decorators"]:
            for obs in getattr(result, obs_type):
                original_name = obs.decorator_qualified_name if obs_type == "decorators" else obs.call.qualified_name

                reconstructed_name = original_name
                instance_variable = getattr(obs, 'instance_variable', None)
                if obs_type == "decorators" and instance_variable:
                    if instance_variable in variable_map:
                        base_class = variable_map[instance_variable]
                        attribute = original_name.split('.', 1)[-1] if '.' in original_name else original_name
                        reconstructed_name = f"{base_class}.{attribute}"
                        logging.debug(f"Reconstructed decorator: {original_name} -> {reconstructed_name}")

                # Enhanced observation tracking
                observation = {
                    "parser_name": reconstructed_name,
                    "original_parser_name": original_name,
                    "file_path": result.file_path,
                    "line_number": obs.line_number,
                    "arguments": obs.call.arguments if obs_type == "assignments" else {},
                    "decorated_name": obs.decorated_function_name if obs_type == "decorators" else None,
                    "type": obs_type[:-1],
                    "raw_code": getattr(obs, 'raw_code', ''),  # Store raw code for LLM analysis
                    "imports": getattr(result, 'imports', []),  # Track imports for disambiguation
                    "instance_variable": instance_variable,  # Store instance variable for decorator reconstruction
                    "assigned_target": getattr(obs, 'target_qualified_name', None) if obs_type == "assignments" else None,
                }
                
                all_observations.append(observation)

                # Collect reconstructed qualified symbol names for exact database matching
                symbols_to_find.add(reconstructed_name)

        for annotation in getattr(result, "type_annotations", []):
            if not annotation.annotation_qualified_name:
                continue

            observation = {
                "parser_name": annotation.annotation_qualified_name,
                "original_parser_name": annotation.annotation_qualified_name,
                "file_path": result.file_path,
                "line_number": annotation.line_number,
                "arguments": {},
                "decorated_name": None,
                "type": "type_annotation",
                "raw_code": "",
                "imports": getattr(result, 'imports', []),
                "instance_variable": None,
                "annotation_target": annotation.target_qualified_name,
            }
            all_observations.append(observation)
            symbols_to_find.add(annotation.annotation_qualified_name)

        for ctx in getattr(result, "context_managers", []):
            if not ctx.context_expr_qualified_name:
                continue

            observation = {
                "parser_name": ctx.context_expr_qualified_name,
                "original_parser_name": ctx.context_expr_qualified_name,
                "file_path": result.file_path,
                "line_number": ctx.line_number,
                "arguments": {},
                "decorated_name": None,
                "type": "context_manager",
                "raw_code": "",
                "imports": getattr(result, 'imports', []),
                "instance_variable": None,
                "context_target": ctx.as_target,
            }
            all_observations.append(observation)
            symbols_to_find.add(ctx.context_expr_qualified_name)

    # Query the database to find all possible matches for our suffixes
    matched_symbols = connector.find_components_by_suffixes(list(symbols_to_find))
    logging.debug(f"Found {len(matched_symbols)} matching symbols in the database.")

    categorized_components: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    component_lookup_by_var: Dict[Tuple[str, str], Dict[str, Any]] = {}
    component_lookup_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    component_metadata_by_instance: Dict[str, Dict[str, Any]] = {}
    processed_observations = set()  # Track processed observations to avoid duplicates

    # Enhanced matching: Find ALL matching observations for each database symbol
    for db_symbol_data in matched_symbols:
        db_symbol_name = db_symbol_data["id"]
        category = db_symbol_data.get("concept")
        label = db_symbol_data.get("label", "")

        # If the node doesn't have a concept, categorize it as 'other'
        if not category:
            category = "other"

        # Find ALL matching observations (not just the first one)
        matching_observations = []
        for obs in all_observations:
            parser_name = obs["parser_name"]
            
            # Enhanced matching logic with better disambiguation
            if _is_symbol_match(parser_name, db_symbol_name, obs.get("imports", [])):
                matching_observations.append(obs)
        
        if not matching_observations:
            logging.debug(f"Skipping symbol '{db_symbol_name}' because it could not be matched to any code observation.")
            continue

        # Process each matching observation
        for matched_obs in matching_observations:
            obs_key = (matched_obs["file_path"], matched_obs["line_number"], matched_obs["parser_name"])
            if obs_key in processed_observations:
                continue  # Skip if already processed
            
            processed_observations.add(obs_key)
            
            # Create component details
            component_details = {
                "name": db_symbol_name,
                "file_path": matched_obs["file_path"],
                "line_number": matched_obs["line_number"],
            }
            component_details["category"] = category

            description = None
            if category == 'tool':
                # Primary strategy: Get description from arguments
                description = matched_obs.get('arguments', {}).get('description')
                # Fallback strategy: Generate description with LLM if not found
                if not description and llm_client and matched_obs.get('raw_code'):
                    try:
                        code_snippet = _reconstruct_code_snippet(matched_obs)
                        prompt = f"Please provide a concise, one-sentence description for the following tool based on its code. The tool's code is:\n\n{code_snippet}"
                        logging.info(f"Generating description for tool: {db_symbol_name}")
                        description = llm_client.invoke(prompt)
                    except Exception as e:
                        logging.warning(f"Failed to generate description for {db_symbol_name}: {e}")
                if description:
                    component_details['description'] = description

            if matched_obs["type"] == "decorator":
                component_details["name"] = matched_obs["decorated_name"]
                component_details["decorated_by"] = db_symbol_name

            elif matched_obs["type"] == "assignment":
                if category == "prompt":
                    prompt_text = matched_obs["arguments"].get("template") or matched_obs["arguments"].get("prompt")
                    if prompt_text:
                        component_details["text"] = prompt_text
                
                # Enhanced: Extract model names using LLM for model and embedding categories
                elif category in ["model", "embedding"] and llm_client:
                    class_name = db_symbol_name.split('.')[-1]
                    code_snippet = _reconstruct_code_snippet(matched_obs)
                    
                    if category == "model":
                        model_name = llm_client.extract_model_name(code_snippet, class_name)
                        if model_name:
                            component_details["model_name"] = model_name
                    
                    elif category == "embedding":
                        embedding_model = llm_client.extract_embedding_model(code_snippet, class_name)
                        if embedding_model:
                            component_details["model_name"] = embedding_model

            elif matched_obs["type"] == "type_annotation":
                component_details["annotated_target"] = matched_obs.get("annotation_target")

            elif matched_obs["type"] == "context_manager":
                component_details["context_target"] = matched_obs.get("context_target")

            if workflow_index and component_details.get("file_path") and component_details.get("line_number"):
                workflows = workflow_index.get_workflow_context(
                    component_details["file_path"],
                    component_details["line_number"],
                )
                if workflows:
                    component_details["workflows"] = workflows

            categorized_components[category].append(component_details)
            component_details["instance_id"] = _build_instance_id(component_details)
            component_metadata_by_instance[component_details["instance_id"]] = {
                "file_path": matched_obs["file_path"],
                "arguments": matched_obs.get("arguments", {}),
            }

            assigned_target = matched_obs.get("assigned_target")
            if not assigned_target and matched_obs["type"] == "decorator":
                assigned_target = matched_obs.get("decorated_name")
            if assigned_target:
                component_details["assigned_target"] = assigned_target
                _register_component_target(
                    component_lookup_by_var,
                    matched_obs["file_path"],
                    assigned_target,
                    component_details,
                )

            _register_component_name(component_lookup_by_name, component_details["name"], component_details)

    relationships = _derive_relationships(
        categorized_components,
        component_metadata_by_instance,
        component_lookup_by_var,
        component_lookup_by_name,
    )

    return CategorizationOutput(components=dict(categorized_components), relationships=relationships)


def _is_symbol_match(parser_name: str, db_symbol_name: str, imports: List[str]) -> bool:
    """
    Exact symbol matching using direct string comparison.
    
    Args:
        parser_name: The parsed symbol name from code
        db_symbol_name: The database symbol name
        imports: List of import statements from the file (unused but kept for compatibility)
        
    Returns:
        True if the symbols match exactly
    """
    logging.debug(f"EXACT MATCH DEBUG: Comparing parser='{parser_name}' vs db='{db_symbol_name}'")
    
    # Direct exact match (case-sensitive)
    if parser_name == db_symbol_name:
        logging.debug(f"EXACT MATCH DEBUG: ✓ PERFECT match")
        return True
    
    logging.debug(f"EXACT MATCH DEBUG: ✗ NO EXACT MATCH found")
    return False

def _reconstruct_code_snippet(observation: Dict[str, Any]) -> str:
    """
    Reconstruct code snippet from observation for LLM analysis.
    
    Args:
        observation: The observation dictionary
        
    Returns:
        Reconstructed code snippet
    """
    if observation.get("raw_code"):
        return observation["raw_code"]
    
    # Fallback: reconstruct from available information
    parser_name = observation.get("parser_name", "")
    arguments = observation.get("arguments", {})
    
    # Handle empty parser_name
    if not parser_name:
        return "()"
    
    class_name = parser_name.split('.')[-1] if parser_name else ""
    
    # Build argument string
    arg_parts = []
    for key, value in arguments.items():
        if isinstance(value, str):
            arg_parts.append(f'{key}="{value}"')
        else:
            arg_parts.append(f'{key}={value}')
    
    args_str = ', '.join(arg_parts)
    return f"{class_name}({args_str})"


def _build_instance_id(component: Dict[str, Any]) -> str:
    name = component.get("name") or "component"
    line = component.get("line_number") or 0
    return f"{name}_{line}"


def _register_component_target(
    index: Dict[Tuple[str, str], Dict[str, Any]],
    file_path: Optional[str],
    target_name: Optional[str],
    component: Dict[str, Any],
) -> None:
    if not file_path or not target_name:
        return
    for normalized in _normalize_reference_names(target_name):
        index[(file_path, normalized)] = component


def _register_component_name(index: Dict[str, List[Dict[str, Any]]], name: Optional[str], component: Dict[str, Any]) -> None:
    if not name:
        return
    for normalized in _normalize_reference_names(name):
        index.setdefault(normalized, []).append(component)


def _derive_relationships(
    categorized_components: Dict[str, List[Dict[str, Any]]],
    component_metadata_by_instance: Dict[str, Dict[str, Any]],
    component_lookup_by_var: Dict[Tuple[str, str], Dict[str, Any]],
    component_lookup_by_name: Dict[str, List[Dict[str, Any]]],
) -> List[ComponentRelationship]:
    relationships: List[ComponentRelationship] = []
    seen_relationships: Set[Tuple[str, str, str]] = set()

    for category, components in categorized_components.items():
        if not _category_matches(category, AGENT_CATEGORY_HINTS):
            continue
        for component in components:
            instance_id = component.get("instance_id")
            if not instance_id:
                continue
            metadata = component_metadata_by_instance.get(instance_id)
            if not metadata:
                continue
            arguments = metadata.get("arguments") or {}
            agent_file = metadata.get("file_path") or component.get("file_path")

            tool_refs = _collect_references(arguments, TOOL_ARGUMENT_HINTS)
            for reference in tool_refs:
                target_component = _resolve_component_reference(
                    reference, agent_file, component_lookup_by_var, component_lookup_by_name
                )
                if not target_component:
                    continue
                if not _category_matches(target_component.get("category"), TOOL_CATEGORY_HINTS):
                    continue
                _append_relationship(
                    relationships,
                    seen_relationships,
                    component,
                    target_component,
                    RELATIONSHIP_LABEL_TOOL,
                )

            llm_refs = _collect_references(arguments, LLM_ARGUMENT_HINTS)
            for reference in llm_refs:
                target_component = _resolve_component_reference(
                    reference, agent_file, component_lookup_by_var, component_lookup_by_name
                )
                if not target_component:
                    continue
                if not _is_llm_category(target_component.get("category")):
                    continue
                _append_relationship(
                    relationships,
                    seen_relationships,
                    component,
                    target_component,
                    RELATIONSHIP_LABEL_LLM,
                )

    return relationships


def _collect_references(arguments: Dict[str, Any], key_hints: Set[str]) -> Set[str]:
    references: Set[str] = set()
    for key, value in arguments.items():
        key_lower = key.lower()
        if any(hint in key_lower for hint in key_hints):
            references.update(_extract_variable_references(value))
    return references


def _extract_variable_references(value: Any) -> Set[str]:
    references: Set[str] = set()
    if isinstance(value, str):
        if value.startswith("VARIABLE:"):
            references.add(value.split("VARIABLE:", 1)[1])
        elif value.startswith("ATTRIBUTE:"):
            references.add(value.split("ATTRIBUTE:", 1)[1])
    elif isinstance(value, list):
        for item in value:
            references.update(_extract_variable_references(item))
    elif isinstance(value, dict):
        for item in value.values():
            references.update(_extract_variable_references(item))
    return references


def _resolve_component_reference(
    reference: str,
    file_path: Optional[str],
    lookup_by_var: Dict[Tuple[str, str], Dict[str, Any]],
    lookup_by_name: Dict[str, List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    normalized_names = _normalize_reference_names(reference)

    for name in normalized_names:
        if file_path:
            component = lookup_by_var.get((file_path, name))
            if component:
                return component

    for name in normalized_names:
        candidates = lookup_by_name.get(name, [])
        if len(candidates) == 1:
            return candidates[0]

    return None


def _normalize_reference_names(name: Optional[str]) -> List[str]:
    if not name:
        return []
    normalized = [name]
    if "." in name:
        normalized.append(name.split(".")[-1])
    # Preserve order while removing duplicates
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in normalized:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _append_relationship(
    relationships: List[ComponentRelationship],
    seen_relationships: Set[Tuple[str, str, str]],
    source_component: Dict[str, Any],
    target_component: Dict[str, Any],
    label: str,
) -> None:
    source_instance = source_component.get("instance_id")
    target_instance = target_component.get("instance_id")
    if not source_instance or not target_instance:
        return

    key = (source_instance, target_instance, label)
    if key in seen_relationships:
        return

    seen_relationships.add(key)
    relationships.append(
        ComponentRelationship(
            source_instance_id=source_instance,
            target_instance_id=target_instance,
            label=label,
            source_name=source_component.get("name", ""),
            target_name=target_component.get("name", ""),
            source_category=source_component.get("category", ""),
            target_category=target_component.get("category", ""),
        )
    )


def _category_matches(category: Optional[str], hints: Set[str]) -> bool:
    if not category:
        return False
    category_lower = category.lower()
    return any(hint in category_lower for hint in hints)


def _is_llm_category(category: Optional[str]) -> bool:
    if not category:
        return False
    cat_lower = category.lower()
    if "embedding" in cat_lower:
        return False
    return any(hint in cat_lower for hint in LLM_CATEGORY_HINTS)
