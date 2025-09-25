from agno.agent import Agent, RunOutput
from agno.exceptions import RetryAgentRun
from agno.tools.tavily import TavilyTools
from fastmcp import FastMCP
import os
import dotenv

dotenv.load_dotenv()

os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
mcp = FastMCP()
# Create research agent with Tavily search
research_agent = Agent(
    name="Research Agent",
    tools=[TavilyTools()],
    instructions="""
        You are a research assistant that helps find accurate information.
        Use Tavily to search for current information and provide comprehensive answers.
        Always cite your sources and provide relevant context.
    """,
    markdown=True
)

retry_count = 0
@mcp.tool()
async def research(message: dict) -> dict:
  input = message['processed_input']
  agent = research_agent
  try:
    response: RunOutput = await agent.arun(input)
    text = response.content if hasattr(response, "content") else str(response) 
    metrics = response.metrics
    print(metrics)
    if text == None and retry_count<2:
      retry_count+=1
      raise RetryAgentRun(
         f"Conduct research based on the users query. users:{input}"
      )
    return {"metrics": metrics.to_dict(), "content": response.content}
  except Exception as e:
    return f"Tool error: {type(e).__name__}: {e}"
  
if __name__ == '__main__':
  mcp.run(transport="streamable-http")