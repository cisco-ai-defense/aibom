from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
])


@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"


graph = StateGraph(dict)
graph.add_node("agent", llm)
graph.add_node("tools", ToolNode([search]))
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
