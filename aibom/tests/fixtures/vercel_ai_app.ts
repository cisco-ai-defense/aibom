import { generateText, streamText, tool } from "ai";
import { openai } from "@ai-sdk/openai";

const model = openai("gpt-4o");

const result = await generateText({
  model: model,
  prompt: "What is the meaning of life?",
});

const myTool = tool({
  description: "Get weather information",
  parameters: { location: { type: "string" } },
  execute: async ({ location }) => {
    return `Weather in ${location}: Sunny`;
  },
});

const stream = await streamText({
  model: model,
  messages: [{ role: "user", content: "Hello" }],
  tools: { weather: myTool },
});
