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

"""Supplemental catalog entries for symbols missing from the prebuilt DuckDB catalog.

These entries augment the catalog at runtime so that LangGraph, LangChain, and
CrewAI symbols are recognized without rebuilding the external catalog artifact.
"""

from typing import Any, Dict, List

SUPPLEMENTAL_ENTRIES: List[Dict[str, Any]] = [
    # ── LangGraph core: agents ──────────────────────────────────────────
    {"id": "langgraph.graph.StateGraph", "label": "StateGraph", "concept": "agent", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.graph.StateGraph.compile", "label": "StateGraph.compile", "concept": "agent", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.graph.state.StateGraph", "label": "StateGraph", "concept": "agent", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.graph.MessageGraph", "label": "MessageGraph", "concept": "agent", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.prebuilt.create_react_agent", "label": "create_react_agent", "concept": "agent", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.prebuilt.chat_agent_executor.create_react_agent", "label": "create_react_agent", "concept": "agent", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangGraph core: memory / checkpointers ──────────────────────────
    {"id": "langgraph.store.base.BaseStore", "label": "BaseStore", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.store.base.BaseStore.asearch", "label": "BaseStore.asearch", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.store.base.BaseStore.aput", "label": "BaseStore.aput", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.store.memory.InMemoryStore", "label": "InMemoryStore", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.checkpoint.memory.MemorySaver", "label": "MemorySaver", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.checkpoint.memory.InMemorySaver", "label": "InMemorySaver", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.checkpoint.base.BaseCheckpointSaver", "label": "BaseCheckpointSaver", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.checkpoint.sqlite.SqliteSaver", "label": "SqliteSaver", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.checkpoint.postgres.PostgresSaver", "label": "PostgresSaver", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver", "label": "AsyncSqliteSaver", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver", "label": "AsyncPostgresSaver", "concept": "memory", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangGraph prebuilt: misc ─────────────────────────────────────────
    {"id": "langgraph.prebuilt.InjectedState", "label": "InjectedState", "concept": "other", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.prebuilt.InjectedStore", "label": "InjectedStore", "concept": "other", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain agents ─────────────────────────────────────────────────
    {"id": "langchain.agents.create_agent", "label": "create_agent", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.agents.AgentExecutor", "label": "AgentExecutor", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.agents.initialize_agent", "label": "initialize_agent", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.agents.create_openai_functions_agent", "label": "create_openai_functions_agent", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.agents.create_openai_tools_agent", "label": "create_openai_tools_agent", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.agents.create_structured_chat_agent", "label": "create_structured_chat_agent", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.agents.create_react_agent", "label": "create_react_agent", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.agents.create_tool_calling_agent", "label": "create_tool_calling_agent", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain tools ──────────────────────────────────────────────────
    {"id": "langchain_core.tools.tool", "label": "tool", "concept": "tool", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_core.tools.BaseTool", "label": "BaseTool", "concept": "tool", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_core.tools.StructuredTool", "label": "StructuredTool", "concept": "tool", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_core.tools.InjectedToolArg", "label": "InjectedToolArg", "concept": "other", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain models ─────────────────────────────────────────────────
    {"id": "langchain.chat_models.init_chat_model", "label": "init_chat_model", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_core.language_models.chat_models.init_chat_model", "label": "init_chat_model", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain memory ─────────────────────────────────────────────────
    {"id": "langchain.memory.ConversationBufferMemory", "label": "ConversationBufferMemory", "concept": "memory", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.memory.ConversationBufferWindowMemory", "label": "ConversationBufferWindowMemory", "concept": "memory", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.memory.ConversationSummaryMemory", "label": "ConversationSummaryMemory", "concept": "memory", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.memory.VectorStoreRetrieverMemory", "label": "VectorStoreRetrieverMemory", "concept": "memory", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.chat_message_histories.ChatMessageHistory", "label": "ChatMessageHistory", "concept": "memory", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain retrievers ─────────────────────────────────────────────
    {"id": "langchain_core.retrievers.BaseRetriever", "label": "BaseRetriever", "concept": "retriever", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.retrievers.multi_query.MultiQueryRetriever", "label": "MultiQueryRetriever", "concept": "retriever", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.retrievers.contextual_compression.ContextualCompressionRetriever", "label": "ContextualCompressionRetriever", "concept": "retriever", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.retrievers.self_query.base.SelfQueryRetriever", "label": "SelfQueryRetriever", "concept": "retriever", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.retrievers.BM25Retriever", "label": "BM25Retriever", "concept": "retriever", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── CrewAI ───────────────────────────────────────────────────────────
    {"id": "crewai.Agent", "label": "Agent", "concept": "agent", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai.Task", "label": "Task", "concept": "other", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai.Crew", "label": "Crew", "concept": "agent", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai.Process", "label": "Process", "concept": "other", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai.tools.BaseTool", "label": "BaseTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai.tools.tool", "label": "tool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.SerperDevTool", "label": "SerperDevTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.ScrapeWebsiteTool", "label": "ScrapeWebsiteTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.WebsiteSearchTool", "label": "WebsiteSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.FileReadTool", "label": "FileReadTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.DirectoryReadTool", "label": "DirectoryReadTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.DirectorySearchTool", "label": "DirectorySearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.CodeDocsSearchTool", "label": "CodeDocsSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.PDFSearchTool", "label": "PDFSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.TXTSearchTool", "label": "TXTSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.CSVSearchTool", "label": "CSVSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.JSONSearchTool", "label": "JSONSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.DOCXSearchTool", "label": "DOCXSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.MDXSearchTool", "label": "MDXSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.GithubSearchTool", "label": "GithubSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.YoutubeVideoSearchTool", "label": "YoutubeVideoSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai_tools.YoutubeChannelSearchTool", "label": "YoutubeChannelSearchTool", "concept": "tool", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai.memory.short_term.ShortTermMemory", "label": "ShortTermMemory", "concept": "memory", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai.memory.long_term.LongTermMemory", "label": "LongTermMemory", "concept": "memory", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "crewai.memory.entity.EntityMemory", "label": "EntityMemory", "concept": "memory", "framework": "crewai", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain vector stores / datastores ─────────────────────────────
    {"id": "langchain_community.vectorstores.FAISS", "label": "FAISS", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.vectorstores.Chroma", "label": "Chroma", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.vectorstores.Pinecone", "label": "Pinecone", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.vectorstores.Qdrant", "label": "Qdrant", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.vectorstores.Weaviate", "label": "Weaviate", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain embeddings ─────────────────────────────────────────────
    {"id": "langchain_openai.OpenAIEmbeddings", "label": "OpenAIEmbeddings", "concept": "embedding", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.embeddings.HuggingFaceEmbeddings", "label": "HuggingFaceEmbeddings", "concept": "embedding", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain chains (common) ────────────────────────────────────────
    {"id": "langchain.chains.RetrievalQA", "label": "RetrievalQA", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.chains.ConversationalRetrievalChain", "label": "ConversationalRetrievalChain", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.chains.LLMChain", "label": "LLMChain", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── Tavily / common tools ────────────────────────────────────────────
    {"id": "langchain_community.tools.tavily_search.TavilySearchResults", "label": "TavilySearchResults", "concept": "tool", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
]
