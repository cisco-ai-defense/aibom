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
    ClassDefObservation,
    DecoratorObservation,
    FunctionAnnotationObservation,
    ComponentRelationship,
    CategorizationOutput,
)
from .catalog_db import CatalogDB, is_excluded
from .custom_catalog import CustomCatalogConfig
from .llm_client import LLMClient
from .workflow_analyzer import WorkflowIndex


AGENT_CATEGORY_HINTS = {"agent"}
TOOL_CATEGORY_HINTS = {"tool"}
LLM_CATEGORY_HINTS = {"llm", "model"}
MEMORY_CATEGORY_HINTS = {"memory"}
RETRIEVER_CATEGORY_HINTS = {"retriever"}
EMBEDDING_CATEGORY_HINTS = {"embedding"}
DATASTORE_CATEGORY_HINTS = {"datastore"}

TOOL_ARGUMENT_HINTS = {"tool", "tools", "skills", "abilities"}
LLM_ARGUMENT_HINTS = {"llm", "language_model", "chat_model", "model"}
MEMORY_ARGUMENT_HINTS = {"memory", "checkpointer", "store", "saver", "chat_history"}
RETRIEVER_ARGUMENT_HINTS = {"retriever", "retrievers", "search", "search_kwargs"}
EMBEDDING_ARGUMENT_HINTS = {"embedding", "embeddings", "embedding_function", "embed", "embed_model"}

RELATIONSHIP_LABEL_TOOL = "USES_TOOL"
RELATIONSHIP_LABEL_LLM = "USES_LLM"
RELATIONSHIP_LABEL_MEMORY = "USES_MEMORY"
RELATIONSHIP_LABEL_RETRIEVER = "USES_RETRIEVER"
RELATIONSHIP_LABEL_EMBEDDING = "USES_EMBEDDING"


_is_excluded = is_excluded  # re-export for backward compatibility


def categorize_symbols(
    analysis_results: List[CodeAnalysisResult],
    connector: CatalogDB,
    llm_config: Optional[Dict[str, Any]] = None,
    workflow_index: Optional[WorkflowIndex] = None,
    custom_config: Optional[CustomCatalogConfig] = None,
) -> CategorizationOutput:
    """
    Orchestrates the symbol matching and categorization process using the 'concept' attribute from the catalog.
    Enhanced to handle multiple matches, model extraction, better import disambiguation,
    inline annotations, base-class detection, and custom relationship types.
    """
    # Initialize LLM client if config provided
    llm_client = None
    if llm_config:
        try:
            llm_client = LLMClient(llm_config)
            logging.info("LLM client initialized for model extraction.")
        except Exception as e:
            logging.warning("Failed to initialize LLM client: %s", e)

    # Extend argument hint sets with user-supplied hints (additive, scoped to this call)
    effective_tool_hints = set(TOOL_ARGUMENT_HINTS)
    effective_llm_hints = set(LLM_ARGUMENT_HINTS)
    effective_memory_hints = set(MEMORY_ARGUMENT_HINTS)
    effective_retriever_hints = set(RETRIEVER_ARGUMENT_HINTS)
    effective_embedding_hints = set(EMBEDDING_ARGUMENT_HINTS)

    if custom_config and custom_config.relationship_hints:
        hints = custom_config.relationship_hints
        effective_tool_hints.update(hints.get("tool_arguments", []))
        effective_llm_hints.update(hints.get("llm_arguments", []))
        effective_memory_hints.update(hints.get("memory_arguments", []))
        effective_retriever_hints.update(hints.get("retriever_arguments", []))
        effective_embedding_hints.update(hints.get("embedding_arguments", []))

    # Build a metadata lookup from custom component entries
    custom_metadata_by_id: Dict[str, Dict[str, Any]] = {}
    if custom_config:
        for comp in custom_config.components:
            if comp.metadata:
                custom_metadata_by_id[comp.id] = comp.metadata

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
        for obs_type in ["assignments", "decorators", "calls"]:
            for obs in getattr(result, obs_type):
                if obs_type == "calls":
                    original_name = obs.qualified_name
                elif obs_type == "decorators":
                    original_name = obs.decorator_qualified_name
                else:
                    original_name = obs.call.qualified_name

                reconstructed_name = original_name
                instance_variable = getattr(obs, 'instance_variable', None)
                if obs_type == "decorators" and instance_variable:
                    if instance_variable in variable_map:
                        base_class = variable_map[instance_variable]
                        attribute = original_name.split('.', 1)[-1] if '.' in original_name else original_name
                        reconstructed_name = f"{base_class}.{attribute}"
                        logging.debug("Reconstructed decorator: %s -> %s", original_name, reconstructed_name)

                # Method-chain resolution: e.g. "builder.compile" -> "langgraph.graph.StateGraph.compile"
                if "." in reconstructed_name:
                    parts = reconstructed_name.split(".", 1)
                    local_var = parts[0]
                    attr_tail = parts[1]
                    if local_var in variable_map:
                        resolved_base = variable_map[local_var]
                        candidate = f"{resolved_base}.{attr_tail}"
                        reconstructed_name = candidate
                        logging.debug("Method-chain resolved: %s -> %s", original_name, reconstructed_name)

                if obs_type == "calls":
                    obs_arguments = obs.arguments
                    obs_raw_code = obs.raw_code
                elif obs_type == "assignments":
                    obs_arguments = obs.call.arguments
                    obs_raw_code = getattr(obs, 'raw_code', '') or getattr(obs.call, 'raw_code', '')
                else:
                    obs_arguments = {}
                    obs_raw_code = getattr(obs, 'raw_code', '')

                observation = {
                    "parser_name": reconstructed_name,
                    "original_parser_name": original_name,
                    "file_path": result.file_path,
                    "line_number": obs.line_number,
                    "arguments": obs_arguments,
                    "decorated_name": obs.decorated_function_name if obs_type == "decorators" else None,
                    "type": obs_type[:-1] if obs_type != "calls" else "call",
                    "raw_code": obs_raw_code,
                    "imports": getattr(result, 'imports', []),
                    "instance_variable": instance_variable,
                    "assigned_target": getattr(obs, 'target_qualified_name', None) if obs_type == "assignments" else None,
                }
                
                all_observations.append(observation)
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

    # Extract exclude patterns early so they can be applied globally
    exclude_patterns: List[str] = custom_config.excludes if custom_config else []

    # Query the database to find all possible matches for our suffixes
    matched_symbols = connector.find_components_by_suffixes(list(symbols_to_find))
    logging.debug("Found %d matching symbols in the database.", len(matched_symbols))

    categorized_components: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    component_lookup_by_var: Dict[Tuple[str, str], Dict[str, Any]] = {}
    component_lookup_by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    component_metadata_by_instance: Dict[str, Dict[str, Any]] = {}

    # Track detected locations across all passes to prevent duplicates.
    # Precedence: inline annotation > DuckDB catalog > base-class rule.
    detected_locations: Set[Tuple[str, int]] = set()  # (file_path, line_number)

    # ── Pass 1: Inline annotation (highest precedence) ───────────────────
    for result in analysis_results:
        for class_obs in getattr(result, "class_defs", []):
            if class_obs.aibom_annotation:
                concept = class_obs.aibom_annotation.get("concept", "other")
                comp_name = class_obs.class_name
                if _is_excluded(comp_name, exclude_patterns):
                    continue
                comp_details = {
                    "name": comp_name,
                    "file_path": result.file_path,
                    "line_number": class_obs.line_number,
                    "category": concept,
                    "framework": class_obs.aibom_annotation.get("framework", "custom"),
                    "detection_source": "inline_annotation",
                }
                label = class_obs.aibom_annotation.get("label")
                if label:
                    comp_details["name"] = label

                if workflow_index and comp_details.get("file_path") and comp_details.get("line_number"):
                    workflows = workflow_index.get_workflow_context(
                        comp_details["file_path"], comp_details["line_number"],
                    )
                    if workflows:
                        comp_details["workflows"] = workflows

                categorized_components[concept].append(comp_details)
                comp_details["instance_id"] = _build_instance_id(comp_details)
                detected_locations.add((result.file_path, class_obs.line_number))
                _register_component_name(component_lookup_by_name, comp_details["name"], comp_details)
                _register_component_target(
                    component_lookup_by_var, result.file_path, comp_name, comp_details,
                )

        for func_obs in getattr(result, "function_annotations", []):
            concept = func_obs.aibom_annotation.get("concept", "other")
            comp_name = func_obs.function_name
            if _is_excluded(comp_name, exclude_patterns):
                continue
            comp_details = {
                "name": comp_name,
                "file_path": result.file_path,
                "line_number": func_obs.line_number,
                "category": concept,
                "framework": func_obs.aibom_annotation.get("framework", "custom"),
                "detection_source": "inline_annotation",
            }
            label = func_obs.aibom_annotation.get("label")
            if label:
                comp_details["name"] = label

            if workflow_index and comp_details.get("file_path") and comp_details.get("line_number"):
                workflows = workflow_index.get_workflow_context(
                    comp_details["file_path"], comp_details["line_number"],
                )
                if workflows:
                    comp_details["workflows"] = workflows

            categorized_components[concept].append(comp_details)
            comp_details["instance_id"] = _build_instance_id(comp_details)
            detected_locations.add((result.file_path, func_obs.line_number))
            _register_component_name(component_lookup_by_name, comp_details["name"], comp_details)
            _register_component_target(
                component_lookup_by_var, result.file_path, comp_name, comp_details,
            )

    # ── Pass 2: DuckDB catalog matching ──────────────────────────────────
    processed_observations: Set[Tuple[str, int, str]] = set()

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
            logging.debug("Skipping symbol '%s' because it could not be matched to any code observation.", db_symbol_name)
            continue

        # Process each matching observation
        for matched_obs in matching_observations:
            obs_key = (matched_obs["file_path"], matched_obs["line_number"], matched_obs["parser_name"])
            if obs_key in processed_observations:
                continue  # Skip if already processed by another DB symbol

            # Skip if this location was already detected by inline annotation
            loc_key = (matched_obs["file_path"], matched_obs["line_number"])
            if loc_key in detected_locations:
                continue

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
                        logging.info("Generating description for tool: %s", db_symbol_name)
                        description = llm_client.invoke(prompt)
                    except Exception as e:
                        logging.warning("Failed to generate description for %s: %s", db_symbol_name, e)
                if description:
                    component_details['description'] = description

            if matched_obs["type"] == "decorator":
                component_details["name"] = matched_obs["decorated_name"]
                component_details["decorated_by"] = db_symbol_name

            elif matched_obs["type"] in ("assignment", "call"):
                if category == "prompt":
                    prompt_text = matched_obs["arguments"].get("template") or matched_obs["arguments"].get("prompt")
                    if prompt_text:
                        component_details["text"] = prompt_text

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
            detected_locations.add(loc_key)
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

            # Merge user-provided metadata from custom catalog for this component
            for custom_id, meta in custom_metadata_by_id.items():
                if db_symbol_name.endswith(custom_id) or db_symbol_name == custom_id:
                    component_details.update(meta)
                    break

            _register_component_name(component_lookup_by_name, component_details["name"], component_details)

    # ── Pass 3: Base class detection (lowest precedence) ─────────────────
    base_class_rules = custom_config.base_class_rules if custom_config else []
    if base_class_rules:
        for result in analysis_results:
            for class_obs in getattr(result, "class_defs", []):
                if class_obs.aibom_annotation:
                    continue  # already handled by inline pass

                # Skip if already detected by inline or catalog pass
                loc_key = (result.file_path, class_obs.line_number)
                if loc_key in detected_locations:
                    continue

                for rule in base_class_rules:
                    matched = any(
                        base.endswith(rule.base_class) or base == rule.base_class
                        for base in class_obs.base_classes
                    )
                    if matched:
                        comp_name = class_obs.class_name
                        if _is_excluded(comp_name, exclude_patterns):
                            continue
                        comp_details = {
                            "name": comp_name,
                            "file_path": result.file_path,
                            "line_number": class_obs.line_number,
                            "category": rule.concept,
                            "framework": "custom",
                            "detection_source": "base_class_rule",
                            "base_class_matched": rule.base_class,
                        }

                        if workflow_index and comp_details.get("file_path") and comp_details.get("line_number"):
                            workflows = workflow_index.get_workflow_context(
                                comp_details["file_path"], comp_details["line_number"],
                            )
                            if workflows:
                                comp_details["workflows"] = workflows

                        categorized_components[rule.concept].append(comp_details)
                        comp_details["instance_id"] = _build_instance_id(comp_details)
                        detected_locations.add(loc_key)
                        _register_component_name(component_lookup_by_name, comp_details["name"], comp_details)
                        _register_component_target(
                            component_lookup_by_var, result.file_path, comp_name, comp_details,
                        )
                        break  # first matching rule wins

    # ── Safety-net: final exclude sweep before relationship derivation ────
    if exclude_patterns:
        for category in list(categorized_components.keys()):
            categorized_components[category] = [
                c for c in categorized_components[category]
                if not _is_excluded(c.get("name", ""), exclude_patterns)
            ]

    # ── Graph-wiring pass: augment agent metadata from add_node() calls ──
    # LangGraph wires agents to tools/models via builder.add_node() rather
    # than constructor kwargs.  Scan observations for these patterns and
    # merge the variable references into the agent's metadata so that
    # _derive_relationships can discover them.
    _augment_agent_metadata_from_graph_wiring(
        all_observations,
        variable_map,
        categorized_components,
        component_metadata_by_instance,
        component_lookup_by_var,
        component_lookup_by_name,
        analysis_results,
    )

    relationships = _derive_relationships(
        categorized_components,
        component_metadata_by_instance,
        component_lookup_by_var,
        component_lookup_by_name,
        effective_tool_hints=effective_tool_hints,
        effective_llm_hints=effective_llm_hints,
        effective_memory_hints=effective_memory_hints,
        effective_retriever_hints=effective_retriever_hints,
        effective_embedding_hints=effective_embedding_hints,
        custom_relationships=custom_config.custom_relationships if custom_config else [],
    )

    return CategorizationOutput(components=dict(categorized_components), relationships=relationships)


def _augment_agent_metadata_from_graph_wiring(
    all_observations: List[Dict[str, Any]],
    variable_map: Dict[str, str],
    categorized_components: Dict[str, List[Dict[str, Any]]],
    component_metadata_by_instance: Dict[str, Dict[str, Any]],
    component_lookup_by_var: Dict[Tuple[str, str], Dict[str, Any]],
    component_lookup_by_name: Dict[str, List[Dict[str, Any]]],
    analysis_results: List["CodeAnalysisResult"],
) -> None:
    """Augment agent metadata with references found in ``add_node()`` calls.

    LangGraph builds graphs via ``builder.add_node(name, action)`` where
    *action* is a function or a component like ``ToolNode(tools)``.  The
    standard relationship engine only inspects constructor kwargs, so it
    misses these wiring patterns.

    This function:
    1. Scans observations for ``<agent_var>.add_node(...)`` calls.
    2. Resolves direct variable references and nested calls (e.g. ``ToolNode(tools)``).
    3. **Function-scope analysis**: when ``add_node`` references a function,
       scans observations inside that function's scope for model/tool/memory usage.
    4. **Cross-file import resolution**: when a variable can't be resolved locally,
       traces imports to find components defined in other source files.
    5. **Same-file inference**: links non-agent components detected in the same
       file to the agent when no explicit wiring is found.
    """
    # Build two reverse maps for agent components:
    # 1. (file_path, variable_name) -> agent component (for local var lookup)
    # 2. (file_path, qualified_name) -> agent component (for resolved name lookup)
    agent_var_to_component: Dict[Tuple[str, str], Dict[str, Any]] = {}
    agent_name_to_component: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for category, components in categorized_components.items():
        if not _category_matches(category, AGENT_CATEGORY_HINTS):
            continue
        for comp in components:
            fp = comp.get("file_path", "")
            target = comp.get("assigned_target")
            if target and fp:
                agent_var_to_component[(fp, target)] = comp
            name = comp.get("name", "")
            if name and fp:
                agent_name_to_component[(fp, name)] = comp

    if not agent_var_to_component and not agent_name_to_component:
        return

    import_map, components_by_file, func_scope_obs = _build_graph_wiring_indices(
        analysis_results, categorized_components, all_observations,
    )

    # ── Main add_node scanning loop ─────────────────────────────────────

    # Track which agents got explicit wiring (for same-file inference fallback)
    agents_with_wiring: Set[str] = set()

    for obs in all_observations:
        parser_name = obs.get("parser_name", "")
        if ".add_node" not in parser_name:
            continue

        file_path = obs.get("file_path", "")
        # The base is everything before .add_node
        base_name = parser_name.rsplit(".add_node", 1)[0]

        # The parser_name has already been resolved through variable_map
        # (e.g., "builder.add_node" → "langgraph.graph.StateGraph.add_node")
        # So base_name is the resolved qualified name of the class.
        # Match it against agent components by their detected name.
        agent_comp = agent_name_to_component.get((file_path, base_name))

        # Also try matching via assigned_target (local variable name)
        if not agent_comp:
            agent_comp = agent_var_to_component.get((file_path, base_name))

        # Try suffix matching: base_name might be a full path like
        # "langgraph.graph.StateGraph" and agent name could be just "StateGraph"
        if not agent_comp:
            for (fp, name), comp in agent_name_to_component.items():
                if fp == file_path and (
                    base_name.endswith(f".{name.rsplit('.', 1)[-1]}")
                    or base_name == name
                ):
                    agent_comp = comp
                    break

        if not agent_comp:
            continue

        instance_id = agent_comp.get("instance_id")
        if not instance_id:
            continue

        agents_with_wiring.add(instance_id)

        # Extract all variable references from positional and keyword args,
        # and detect nested calls to known components (e.g., ToolNode(tools)).
        arguments = obs.get("arguments", {})
        refs: Set[str] = set()
        nested_call_names: Set[str] = set()
        nested_call_inner_refs: Set[str] = set()
        for key, value in arguments.items():
            refs.update(_extract_variable_references(value))
            # Collect nested call names (e.g., ToolNode from add_node("tools", ToolNode(tools)))
            if isinstance(value, dict) and "_call" in value:
                nested_call_names.add(value["_call"])
                # Also collect inner args of nested calls (e.g., "tools" from ToolNode(tools))
                for inner_arg in value.get("_args", []):
                    if isinstance(inner_arg, str) and inner_arg.startswith("VARIABLE:"):
                        nested_call_inner_refs.add(inner_arg.replace("VARIABLE:", ""))

        # Merge into the agent's metadata arguments so _derive_relationships
        # can discover these wiring connections.
        metadata = component_metadata_by_instance.get(instance_id)
        if not metadata:
            metadata = {
                "file_path": file_path,
                "arguments": {},
            }
            component_metadata_by_instance[instance_id] = metadata

        existing_args = metadata.get("arguments", {})

        # First, check nested call names against detected components.
        # If add_node contains ToolNode(...), look up ToolNode in detected components.
        for call_name in nested_call_names:
            for cat, comp_list in categorized_components.items():
                for comp in comp_list:
                    comp_name = comp.get("name", "")
                    if comp_name.endswith(call_name) or comp_name == call_name:
                        cat_lower = cat.lower()
                        if "tool" in cat_lower:
                            existing_args.setdefault("tools", []).append(f"VARIABLE:{call_name}")
                        elif "model" in cat_lower:
                            existing_args["llm"] = f"VARIABLE:{call_name}"
                        elif "memory" in cat_lower:
                            existing_args["memory"] = f"VARIABLE:{call_name}"
                        break

        # Resolve inner refs of nested calls via cross-file imports.
        # e.g., ToolNode(TOOLS) where TOOLS is imported from another module.
        for inner_ref in nested_call_inner_refs:
            if component_lookup_by_var.get((file_path, inner_ref)):
                continue  # already locally resolved
            # Check imports
            file_imps = import_map.get(file_path, {})
            qualified = file_imps.get(inner_ref)
            if qualified:
                _resolve_cross_file_ref(
                    inner_ref, qualified, components_by_file, analysis_results,
                    categorized_components, existing_args,
                )

        # Check direct variable references
        for ref in refs:
            # Check if the ref is a detected component variable
            target_comp = component_lookup_by_var.get((file_path, ref))
            if target_comp:
                _merge_component_ref_into_args(ref, target_comp, existing_args)
                continue

            # ── Function-scope analysis ─────────────────────────────────
            # If ref is a function name, scan observations inside that function
            # for component usage (models, tools, memory, etc.)
            func_obs_in_file = func_scope_obs.get(file_path, {})
            if ref in func_obs_in_file:
                for inner_obs in func_obs_in_file[ref]:
                    inner_name = inner_obs.get("parser_name", "")
                    # Check if this inner observation matches a detected component
                    matched_inner = False
                    for cat, comp_list in categorized_components.items():
                        for comp in comp_list:
                            if comp.get("name") and (comp["name"] == inner_name or inner_name.endswith(comp["name"])):
                                _merge_component_ref_into_args(
                                    inner_obs.get("assigned_target", "").rsplit(".", 1)[-1] or inner_name,
                                    comp, existing_args,
                                )
                                matched_inner = True
                                break
                        if matched_inner:
                            break
                    if matched_inner:
                        continue
                    # Transitive module-level inference: when the inner call
                    # (e.g., utils.load_chat_model) doesn't match a component
                    # directly, check if the called module contains components.
                    # Try progressively shorter prefixes of the qualified name
                    # to find the module file (e.g., "react_agent.utils.load_chat_model"
                    # → try "react_agent.utils.load_chat_model", "react_agent.utils", "react_agent").
                    if "." in inner_name:
                        parts = inner_name.split(".")
                        found_module = False
                        for trim in range(1, len(parts)):
                            module_part = ".".join(parts[:-trim])
                            if not module_part:
                                break
                            for fp, file_comps in components_by_file.items():
                                fp_norm = fp.replace("/", ".").replace("\\", ".")
                                if fp_norm.endswith(".py"):
                                    fp_norm = fp_norm[:-3]
                                if fp_norm.endswith(module_part) or fp_norm.endswith("." + module_part):
                                    for mod_comp in file_comps:
                                        mod_cat = (mod_comp.get("category") or "").lower()
                                        comp_ref = mod_comp.get("assigned_target") or mod_comp.get("name", "")
                                        if not _category_matches(mod_cat, AGENT_CATEGORY_HINTS):
                                            _merge_component_ref_into_args(comp_ref, mod_comp, existing_args)
                                    found_module = True
                                    break
                            if found_module:
                                break
                continue

            # ── Cross-file import resolution ────────────────────────────
            # If the variable can't be resolved locally, trace through imports
            file_imps = import_map.get(file_path, {})
            qualified = file_imps.get(ref)
            if qualified:
                _resolve_cross_file_ref(
                    ref, qualified, components_by_file, analysis_results,
                    categorized_components, existing_args,
                )
                continue

            # Fallback: check if the variable resolves to a known class
            resolved = variable_map.get(ref, "")
            if resolved:
                for cat, comp_list in categorized_components.items():
                    matched = False
                    for comp in comp_list:
                        if comp.get("name", "") == resolved or comp.get("name", "").endswith(f".{ref}"):
                            _merge_component_ref_into_args(ref, comp, existing_args)
                            matched = True
                            break
                    if matched:
                        break

        metadata["arguments"] = existing_args

    _apply_same_file_inference(
        agents_with_wiring, component_metadata_by_instance, components_by_file,
    )


def _build_graph_wiring_indices(
    analysis_results: List["CodeAnalysisResult"],
    categorized_components: Dict[str, List[Dict[str, Any]]],
    all_observations: List[Dict[str, Any]],
) -> tuple:
    """Build helper indices used by the graph-wiring analysis.

    Returns:
        (import_map, components_by_file, func_scope_obs)
    """
    # Build import map: { file_path: { local_name: qualified_module_path } }
    import_map: Dict[str, Dict[str, str]] = {}
    for result in analysis_results:
        file_imports: Dict[str, str] = {}
        for imp_entry in result.imports:
            imp_str = imp_entry[1] if isinstance(imp_entry, tuple) else imp_entry
            if imp_str.startswith("from "):
                parts = imp_str.split()
                if len(parts) >= 4 and parts[2] == "import":
                    module = parts[1]
                    imported_names = " ".join(parts[3:]).split(",")
                    for name_part in imported_names:
                        name_part = name_part.strip()
                        if " as " in name_part:
                            original, alias = name_part.split(" as ", 1)
                            file_imports[alias.strip()] = f"{module}.{original.strip()}"
                        else:
                            file_imports[name_part] = f"{module}.{name_part}"
            elif imp_str.startswith("import "):
                parts = imp_str.split()
                if " as " in imp_str:
                    idx = parts.index("as")
                    full_module = " ".join(parts[1:idx])
                    alias = parts[idx + 1]
                    file_imports[alias] = full_module
                else:
                    full_module = parts[1]
                    file_imports[full_module] = full_module
        import_map[result.file_path] = file_imports

    # Build per-file component index
    components_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for _cat, comp_list in categorized_components.items():
        for comp in comp_list:
            fp = comp.get("file_path", "")
            if fp:
                components_by_file.setdefault(fp, []).append(comp)

    # Build per-file observation index for function-scope analysis
    func_scope_obs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for obs in all_observations:
        target = obs.get("assigned_target") or ""
        if ".<locals>." in target:
            fp = obs.get("file_path", "")
            func_name = target.split(".<locals>.", 1)[0]
            if "." in func_name:
                func_name = func_name.rsplit(".", 1)[-1]
            func_scope_obs.setdefault(fp, {}).setdefault(func_name, []).append(obs)

    return import_map, components_by_file, func_scope_obs


def _apply_same_file_inference(
    agents_with_wiring: Set[str],
    component_metadata_by_instance: Dict[str, Dict[str, Any]],
    components_by_file: Dict[str, List[Dict[str, Any]]],
) -> None:
    """Fallback: infer relationships from same-file co-occurrence.

    For agents that had ``add_node`` calls but no explicit model/tool/memory
    wiring was discovered, scan for non-agent components detected in the same
    file and create implicit relationships.
    """
    for instance_id in agents_with_wiring:
        metadata = component_metadata_by_instance.get(instance_id)
        if not metadata:
            continue
        existing_args = metadata.get("arguments", {})
        if any(k in existing_args for k in ("llm", "model", "tools", "memory", "retriever", "embedding")):
            continue

        agent_file = metadata.get("file_path", "")
        if not agent_file:
            continue

        for comp in components_by_file.get(agent_file, []):
            cat = (comp.get("category") or "").lower()
            comp_target = comp.get("assigned_target") or comp.get("name", "")
            if _category_matches(cat, AGENT_CATEGORY_HINTS):
                continue
            if "model" in cat or "llm" in cat:
                existing_args.setdefault("llm", f"VARIABLE:{comp_target}")
            elif "tool" in cat:
                existing_args.setdefault("tools", []).append(f"VARIABLE:{comp_target}")
            elif "memory" in cat:
                existing_args.setdefault("memory", f"VARIABLE:{comp_target}")
            elif "retriever" in cat:
                existing_args.setdefault("retriever", f"VARIABLE:{comp_target}")
            elif "embedding" in cat:
                existing_args.setdefault("embedding", f"VARIABLE:{comp_target}")

        metadata["arguments"] = existing_args


def _merge_component_ref_into_args(
    ref: str,
    comp: Dict[str, Any],
    existing_args: Dict[str, Any],
) -> None:
    """Merge a component reference into the agent's metadata arguments dict."""
    cat = (comp.get("category") or "").lower()
    if "tool" in cat:
        existing_args.setdefault("tools", []).append(f"VARIABLE:{ref}")
    elif "model" in cat or "llm" in cat:
        existing_args["llm"] = f"VARIABLE:{ref}"
    elif "memory" in cat:
        existing_args["memory"] = f"VARIABLE:{ref}"
    elif "retriever" in cat:
        existing_args["retriever"] = f"VARIABLE:{ref}"
    elif "embedding" in cat:
        existing_args["embedding"] = f"VARIABLE:{ref}"
    else:
        existing_args.setdefault("tools", []).append(f"VARIABLE:{ref}")


def _resolve_cross_file_ref(
    ref: str,
    qualified_import: str,
    components_by_file: Dict[str, List[Dict[str, Any]]],
    analysis_results: List["CodeAnalysisResult"],
    categorized_components: Dict[str, List[Dict[str, Any]]],
    existing_args: Dict[str, Any],
) -> None:
    """Resolve a cross-file import reference to detected components.

    When a variable like ``TOOLS`` is imported from another module (e.g.,
    ``from react_agent.tools import TOOLS``), trace through analysis results
    to find the source file and look for detected components there that
    are assigned to that variable name.
    """
    # qualified_import is e.g. "react_agent.tools.TOOLS"
    # The variable name in the source file is the last segment
    remote_var = qualified_import.rsplit(".", 1)[-1] if "." in qualified_import else qualified_import
    # The module path is everything before the last segment
    module_path = qualified_import.rsplit(".", 1)[0] if "." in qualified_import else ""

    # Find the source file by matching module paths in analysis_results
    for result in analysis_results:
        fp = result.file_path
        # Convert file path to a module-like dotted name for matching
        # e.g., "/path/to/react_agent/tools.py" → look for "react_agent.tools"
        # Try matching the module_path as a suffix of the file path
        fp_normalized = fp.replace("/", ".").replace("\\", ".")
        if fp_normalized.endswith(".py"):
            fp_normalized = fp_normalized[:-3]

        if not fp_normalized.endswith(module_path) and not fp_normalized.endswith("." + module_path):
            continue

        # Found a candidate source file; look for components assigned to remote_var
        for comp in components_by_file.get(fp, []):
            comp_target = comp.get("assigned_target", "")
            comp_name = comp.get("name", "")
            # Check if this component was assigned to the imported variable name
            if comp_target == remote_var or comp_target.endswith(f".{remote_var}"):
                _merge_component_ref_into_args(ref, comp, existing_args)
                return
            # Also check by name suffix match
            if comp_name.endswith(f".{remote_var}") or comp_name == remote_var:
                _merge_component_ref_into_args(ref, comp, existing_args)
                return

    # If we can't find the source file directly, check if any detected component
    # has a name that matches the qualified import path
    for cat, comp_list in categorized_components.items():
        for comp in comp_list:
            comp_name = comp.get("name", "")
            if comp_name == qualified_import or qualified_import.endswith(comp_name):
                _merge_component_ref_into_args(ref, comp, existing_args)
                return


def _is_symbol_match(parser_name: str, db_symbol_name: str, imports: List[str]) -> bool:
    """Exact symbol matching using direct string comparison.

    Args:
        parser_name: The parsed symbol name from code.
        db_symbol_name: The database symbol name.
        imports: List of import statements (unused, kept for API compatibility).

    Returns:
        True if the symbols match exactly.
    """
    return parser_name == db_symbol_name

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
    effective_tool_hints: Optional[Set[str]] = None,
    effective_llm_hints: Optional[Set[str]] = None,
    effective_memory_hints: Optional[Set[str]] = None,
    effective_retriever_hints: Optional[Set[str]] = None,
    effective_embedding_hints: Optional[Set[str]] = None,
    custom_relationships: Optional[list] = None,
) -> List[ComponentRelationship]:
    relationships: List[ComponentRelationship] = []
    seen_relationships: Set[Tuple[str, str, str]] = set()

    tool_hints = effective_tool_hints or TOOL_ARGUMENT_HINTS
    llm_hints = effective_llm_hints or LLM_ARGUMENT_HINTS
    memory_hints = effective_memory_hints or MEMORY_ARGUMENT_HINTS
    retriever_hints = effective_retriever_hints or RETRIEVER_ARGUMENT_HINTS
    embedding_hints = effective_embedding_hints or EMBEDDING_ARGUMENT_HINTS

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

            tool_refs = _collect_references(arguments, tool_hints)
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

            llm_refs = _collect_references(arguments, llm_hints)
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

            memory_refs = _collect_references(arguments, memory_hints)
            for reference in memory_refs:
                target_component = _resolve_component_reference(
                    reference, agent_file, component_lookup_by_var, component_lookup_by_name
                )
                if not target_component:
                    continue
                if not _category_matches(target_component.get("category"), MEMORY_CATEGORY_HINTS):
                    continue
                _append_relationship(
                    relationships,
                    seen_relationships,
                    component,
                    target_component,
                    RELATIONSHIP_LABEL_MEMORY,
                )

            retriever_refs = _collect_references(arguments, retriever_hints)
            for reference in retriever_refs:
                target_component = _resolve_component_reference(
                    reference, agent_file, component_lookup_by_var, component_lookup_by_name
                )
                if not target_component:
                    continue
                if not _category_matches(target_component.get("category"), RETRIEVER_CATEGORY_HINTS):
                    continue
                _append_relationship(
                    relationships,
                    seen_relationships,
                    component,
                    target_component,
                    RELATIONSHIP_LABEL_RETRIEVER,
                )

            embedding_refs = _collect_references(arguments, embedding_hints)
            for reference in embedding_refs:
                target_component = _resolve_component_reference(
                    reference, agent_file, component_lookup_by_var, component_lookup_by_name
                )
                if not target_component:
                    continue
                if not _category_matches(target_component.get("category"), EMBEDDING_CATEGORY_HINTS):
                    continue
                _append_relationship(
                    relationships,
                    seen_relationships,
                    component,
                    target_component,
                    RELATIONSHIP_LABEL_EMBEDDING,
                )

    # Also derive USES_EMBEDDING from datastore components
    for category, components in categorized_components.items():
        if not _category_matches(category, DATASTORE_CATEGORY_HINTS):
            continue
        for component in components:
            instance_id = component.get("instance_id")
            if not instance_id:
                continue
            metadata = component_metadata_by_instance.get(instance_id)
            if not metadata:
                continue
            arguments = metadata.get("arguments") or {}
            ds_file = metadata.get("file_path") or component.get("file_path")

            embedding_refs = _collect_references(arguments, embedding_hints)
            for reference in embedding_refs:
                target_component = _resolve_component_reference(
                    reference, ds_file, component_lookup_by_var, component_lookup_by_name
                )
                if not target_component:
                    continue
                if not _category_matches(target_component.get("category"), EMBEDDING_CATEGORY_HINTS):
                    continue
                _append_relationship(
                    relationships,
                    seen_relationships,
                    component,
                    target_component,
                    RELATIONSHIP_LABEL_EMBEDDING,
                )

    # ── Custom relationship types from .aibom.yaml ──────────────────────
    if custom_relationships:
        for rel_def in custom_relationships:
            source_cat_set = set(rel_def.source_categories)
            target_cat_set = set(rel_def.target_categories)
            arg_hint_set = set(rel_def.argument_hints)

            if not arg_hint_set:
                continue

            for category, components in categorized_components.items():
                if category.lower() not in source_cat_set:
                    continue
                for component in components:
                    instance_id = component.get("instance_id")
                    if not instance_id:
                        continue
                    metadata = component_metadata_by_instance.get(instance_id)
                    if not metadata:
                        continue
                    arguments = metadata.get("arguments") or {}
                    comp_file = metadata.get("file_path") or component.get("file_path")

                    refs = _collect_references(arguments, arg_hint_set)
                    for reference in refs:
                        target_component = _resolve_component_reference(
                            reference, comp_file, component_lookup_by_var, component_lookup_by_name
                        )
                        if not target_component:
                            continue
                        target_cat = (target_component.get("category") or "").lower()
                        if target_cat_set and target_cat not in target_cat_set:
                            continue
                        _append_relationship(
                            relationships,
                            seen_relationships,
                            component,
                            target_component,
                            rel_def.label,
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
