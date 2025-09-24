from typing import Literal, Optional
from datetime import datetime
from fastmcp import FastMCP
import yfinance as yf
from textwrap import dedent
from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Change transport to 'streamable_http'
mcp = FastMCP()
# Define the finance agent
def stock_agent() -> Agent:
    return Agent(
            model=OpenAIChat(id="gpt-4.1-mini"),
            tools=[ReasoningTools(add_instructions=True), YFinanceTools()],
            instructions=dedent("""\
                You are a seasoned Wall Street analyst with deep expertise in market analysis! 📊

                Follow these steps for comprehensive financial analysis:
                1. Market Overview
                - Latest stock price
                - 52-week high and low
                2. Financial Deep Dive
                - Key metrics (P/E, Market Cap, EPS)
                3. Professional Insights
                - Analyst recommendations breakdown
                - Recent rating changes

                4. Market Context
                - Industry trends and positioning
                - Competitive analysis
                - Market sentiment indicators

                Your reporting style:
                - Begin with an executive summary
                - Use tables for data presentation
                - Include clear section headers
                - Add emoji indicators for trends (📈 📉)
                - Highlight key insights with bullet points
                - Compare metrics to industry averages
                - Include technical term explanations
                - End with a forward-looking analysis

                Risk Disclosure:
                - Always highlight potential risk factors
                - Note market uncertainties
                - Mention relevant regulatory concerns\
            """),
            add_datetime_to_context=True,
            markdown=True,
            stream_intermediate_steps=True,
        )

# Register the finance agent as a tool
@mcp.tool
async def finance_analyzer(query: str) -> str:
    """
    Analyze financial information for a given query using the finance agent.
    """
    # Assuming the agent can be invoked with a string query and returns a string response
    # You might need to adapt this based on how your agent is designed to be called
    agent = stock_agent()
    try:
        response: RunOutput = await agent.arun(query)
        return response.content
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"

if __name__ == "__main__":
    # Run with: fastmcp run server.py
    mcp.run(transport="streamable-http")