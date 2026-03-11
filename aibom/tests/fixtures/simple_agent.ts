import { ChatOpenAI } from "@langchain/openai";
import { StateGraph } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { tool } from "@langchain/core/tools";

const llm = new ChatOpenAI({ model: "gpt-4o-mini" });

const searchTool = tool(async (input: string) => {
  return `Results for: ${input}`;
}, { name: "search", description: "Search the web" });

const graph = new StateGraph({});
graph.addNode("agent", llm);
graph.addNode("tools", new ToolNode([searchTool]));

const memory = new MemorySaver();
const app = graph.compile({ checkpointer: memory });
