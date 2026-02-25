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

"""All catalog entries as a Python data module.

This is the single source of truth for framework-specific AI component
entries that are loaded into the DuckDB catalog via ``build_catalog.py``.
"""

from typing import Any, Dict, List


def get_all_entries() -> List[Dict[str, Any]]:
    """Return the complete list of catalog entries.

    Returns a shallow copy of the list.  The inner dicts are shared with
    the module-level data — callers must not mutate them.
    """
    return list(_ALL_ENTRIES)


_ALL_ENTRIES: List[Dict[str, Any]] = [
    # ========================================================================
    # LangGraph
    # ========================================================================

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

    # ── LangGraph prebuilt: tools & misc ───────────────────────────────────
    {"id": "langgraph.prebuilt.ToolNode", "label": "ToolNode", "concept": "tool", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.prebuilt.tools_condition", "label": "tools_condition", "concept": "other", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.prebuilt.InjectedState", "label": "InjectedState", "concept": "other", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langgraph.prebuilt.InjectedStore", "label": "InjectedStore", "concept": "other", "framework": "langgraph", "sig_name": None, "type": None, "catalog_label": None},

    # ========================================================================
    # LangChain
    # ========================================================================

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
    {"id": "langchain_openai.ChatOpenAI", "label": "ChatOpenAI", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_openai.AzureChatOpenAI", "label": "AzureChatOpenAI", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_anthropic.ChatAnthropic", "label": "ChatAnthropic", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_google_genai.ChatGoogleGenerativeAI", "label": "ChatGoogleGenerativeAI", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_google_vertexai.ChatVertexAI", "label": "ChatVertexAI", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_mistralai.ChatMistralAI", "label": "ChatMistralAI", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_fireworks.ChatFireworks", "label": "ChatFireworks", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_groq.ChatGroq", "label": "ChatGroq", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_cohere.ChatCohere", "label": "ChatCohere", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_nvidia_ai_endpoints.ChatNVIDIA", "label": "ChatNVIDIA", "concept": "model", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

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

    # ── LangChain vector stores / datastores ─────────────────────────────
    {"id": "langchain_community.vectorstores.FAISS", "label": "FAISS", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.vectorstores.Chroma", "label": "Chroma", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.vectorstores.Pinecone", "label": "Pinecone", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.vectorstores.Qdrant", "label": "Qdrant", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.vectorstores.Weaviate", "label": "Weaviate", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain embeddings ─────────────────────────────────────────────
    {"id": "langchain_openai.OpenAIEmbeddings", "label": "OpenAIEmbeddings", "concept": "embedding", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_openai.AzureOpenAIEmbeddings", "label": "AzureOpenAIEmbeddings", "concept": "embedding", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_community.embeddings.HuggingFaceEmbeddings", "label": "HuggingFaceEmbeddings", "concept": "embedding", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_cohere.CohereEmbeddings", "label": "CohereEmbeddings", "concept": "embedding", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain chains (common) ────────────────────────────────────────
    {"id": "langchain.chains.RetrievalQA", "label": "RetrievalQA", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.chains.ConversationalRetrievalChain", "label": "ConversationalRetrievalChain", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain.chains.LLMChain", "label": "LLMChain", "concept": "agent", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── Tavily / common tools ────────────────────────────────────────────
    {"id": "langchain_community.tools.tavily_search.TavilySearchResults", "label": "TavilySearchResults", "concept": "tool", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "langchain_tavily.TavilySearch", "label": "TavilySearch", "concept": "tool", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ── LangChain vector stores (additional providers) ────────────────────
    {"id": "langchain_elasticsearch.ElasticsearchStore", "label": "ElasticsearchStore", "concept": "datastore", "framework": "langchain", "sig_name": None, "type": None, "catalog_label": None},

    # ========================================================================
    # CrewAI
    # ========================================================================
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

    # ========================================================================
    # AutoGen  (Phase 6)
    # ========================================================================
    {"id": "autogen.ConversableAgent", "label": "ConversableAgent", "concept": "agent", "framework": "autogen", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "autogen.AssistantAgent", "label": "AssistantAgent", "concept": "agent", "framework": "autogen", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "autogen.UserProxyAgent", "label": "UserProxyAgent", "concept": "agent", "framework": "autogen", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "autogen.GroupChat", "label": "GroupChat", "concept": "agent", "framework": "autogen", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "autogen.GroupChatManager", "label": "GroupChatManager", "concept": "agent", "framework": "autogen", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "autogen.agentchat.ConversableAgent", "label": "ConversableAgent", "concept": "agent", "framework": "autogen", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "autogen.agentchat.AssistantAgent", "label": "AssistantAgent", "concept": "agent", "framework": "autogen", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "autogen.agentchat.UserProxyAgent", "label": "UserProxyAgent", "concept": "agent", "framework": "autogen", "sig_name": None, "type": None, "catalog_label": None},

    # ========================================================================
    # DSPy  (Phase 6)
    # ========================================================================
    {"id": "dspy.Module", "label": "Module", "concept": "agent", "framework": "dspy", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "dspy.ChainOfThought", "label": "ChainOfThought", "concept": "agent", "framework": "dspy", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "dspy.Predict", "label": "Predict", "concept": "agent", "framework": "dspy", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "dspy.Retrieve", "label": "Retrieve", "concept": "retriever", "framework": "dspy", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "dspy.teleprompt.BootstrapFewShot", "label": "BootstrapFewShot", "concept": "other", "framework": "dspy", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "dspy.teleprompt.SignatureOptimizer", "label": "SignatureOptimizer", "concept": "other", "framework": "dspy", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "dspy.OpenAI", "label": "OpenAI", "concept": "model", "framework": "dspy", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "dspy.ColBERTv2", "label": "ColBERTv2", "concept": "retriever", "framework": "dspy", "sig_name": None, "type": None, "catalog_label": None},

    # ========================================================================
    # Haystack  (Phase 6)
    # ========================================================================
    {"id": "haystack.Pipeline", "label": "Pipeline", "concept": "agent", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "haystack.pipeline.Pipeline", "label": "Pipeline", "concept": "agent", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "haystack.components.generators.openai.OpenAIGenerator", "label": "OpenAIGenerator", "concept": "model", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "haystack.components.generators.hugging_face_local.HuggingFaceLocalGenerator", "label": "HuggingFaceLocalGenerator", "concept": "model", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "haystack.components.retrievers.in_memory.InMemoryBM25Retriever", "label": "InMemoryBM25Retriever", "concept": "retriever", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "haystack.components.retrievers.in_memory.InMemoryEmbeddingRetriever", "label": "InMemoryEmbeddingRetriever", "concept": "retriever", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "haystack.document_stores.in_memory.InMemoryDocumentStore", "label": "InMemoryDocumentStore", "concept": "datastore", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "haystack.components.builders.prompt_builder.PromptBuilder", "label": "PromptBuilder", "concept": "other", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "haystack.components.embedders.openai.OpenAITextEmbedder", "label": "OpenAITextEmbedder", "concept": "embedding", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "haystack.components.embedders.openai.OpenAIDocumentEmbedder", "label": "OpenAIDocumentEmbedder", "concept": "embedding", "framework": "haystack", "sig_name": None, "type": None, "catalog_label": None},

    # ========================================================================
    # LlamaIndex  (Phase 6)
    # ========================================================================
    {"id": "llama_index.core.VectorStoreIndex", "label": "VectorStoreIndex", "concept": "datastore", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "llama_index.core.SimpleDirectoryReader", "label": "SimpleDirectoryReader", "concept": "other", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "llama_index.core.query_engine.RetrieverQueryEngine", "label": "RetrieverQueryEngine", "concept": "retriever", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "llama_index.llms.openai.OpenAI", "label": "OpenAI", "concept": "model", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "llama_index.agent.openai.OpenAIAgent", "label": "OpenAIAgent", "concept": "agent", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "llama_index.core.ServiceContext", "label": "ServiceContext", "concept": "other", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "llama_index.core.Settings", "label": "Settings", "concept": "other", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "llama_index.embeddings.openai.OpenAIEmbedding", "label": "OpenAIEmbedding", "concept": "embedding", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "llama_index.core.indices.keyword_table.KeywordTableIndex", "label": "KeywordTableIndex", "concept": "datastore", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "llama_index.agent.react.ReActAgent", "label": "ReActAgent", "concept": "agent", "framework": "llamaindex", "sig_name": None, "type": None, "catalog_label": None},

    # ========================================================================
    # Semantic Kernel  (Phase 6)
    # ========================================================================
    {"id": "semantic_kernel.Kernel", "label": "Kernel", "concept": "agent", "framework": "semantic_kernel", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "semantic_kernel.connectors.ai.open_ai.OpenAIChatCompletion", "label": "OpenAIChatCompletion", "concept": "model", "framework": "semantic_kernel", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "semantic_kernel.connectors.ai.open_ai.AzureChatCompletion", "label": "AzureChatCompletion", "concept": "model", "framework": "semantic_kernel", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "semantic_kernel.functions.kernel_function", "label": "kernel_function", "concept": "tool", "framework": "semantic_kernel", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "semantic_kernel.memory.SemanticTextMemory", "label": "SemanticTextMemory", "concept": "memory", "framework": "semantic_kernel", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "semantic_kernel.connectors.ai.open_ai.OpenAITextEmbedding", "label": "OpenAITextEmbedding", "concept": "embedding", "framework": "semantic_kernel", "sig_name": None, "type": None, "catalog_label": None},

    # ========================================================================
    # Smolagents / Transformers Agents  (Phase 6)
    # ========================================================================
    {"id": "smolagents.CodeAgent", "label": "CodeAgent", "concept": "agent", "framework": "smolagents", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "smolagents.ToolCallingAgent", "label": "ToolCallingAgent", "concept": "agent", "framework": "smolagents", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "smolagents.HfApiModel", "label": "HfApiModel", "concept": "model", "framework": "smolagents", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "smolagents.Tool", "label": "Tool", "concept": "tool", "framework": "smolagents", "sig_name": None, "type": None, "catalog_label": None},

    # ========================================================================
    # Google GenAI / Vertex AI  (Phase 6)
    # ========================================================================
    {"id": "google.generativeai.GenerativeModel", "label": "GenerativeModel", "concept": "model", "framework": "google_genai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "google.generativeai.ChatSession", "label": "ChatSession", "concept": "agent", "framework": "google_genai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "google.generativeai.configure", "label": "configure", "concept": "other", "framework": "google_genai", "sig_name": None, "type": None, "catalog_label": None},
    {"id": "vertexai.generative_models.GenerativeModel", "label": "GenerativeModel", "concept": "model", "framework": "vertexai", "sig_name": None, "type": None, "catalog_label": None},
]
