from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()

researcher = Agent(
    role="Researcher",
    goal="Research the topic",
    tools=[search_tool],
    llm="gpt-4o-mini",
)

task = Task(
    description="Research AI trends",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
