
import os
import json
from dotenv import load_dotenv
from fastmcp import Client, FastMCP
import requests
from fastapi import FastAPI
from textwrap import dedent
from typing import Dict, Optional, Any
from requests.exceptions import JSONDecodeError
from agno.os import AgentOS
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.tavily import TavilyTools

mcp = FastMCP()

load_dotenv()
os.environ["TAVILY_API_KEY"]= os.getenv("TAVILY_API_KEY")

HISTORY_FILE = "chat_history.json"

# Function to load chat history from file
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# Function to save chat history to file
def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

# Define the stock analysis function
def analyze_stock(query, chat_history):
    # Assuming finance_agent is already defined in a previous cell
    response = budget-agent.print_response(query, stream=True)  # Use stream=False for Gradio output

    # Append the query and response to the chat history
    chat_history.append((query, response))

    # Save the updated chat history
    save_history(chat_history)

    return "", chat_history

# Load existing history when the app starts
initial_history = load_history()


# Creating tool
def get_local_data() -> Optional[Dict[str, Any]]:
  """
  Use this function to get the users location from their IP Address.

  Args:
      None
  Returns:
      str: JSON string of the user's location.
  """
  try:
    response = requests.get("http://ip-api.com/json/")
    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
    return response.json()
  except JSONDecodeError as e:
    print(f"JSONDecodeError: {e}")
    print(f"Raw response text: {response.text}")
    return None
  except requests.exceptions.RequestException as e:
    print(f"RequestException: {e}")
    return None


#Budget-Agent

# Initialize MCP and AgentOS instances

def budget_agent() -> Agent:
  return Agent(
    name="budget-agent",
    model=OpenAIChat(id=os.getenv("OPENAI_MODEL","gpt-3.5-turbo")),
    instructions=dedent("""\
      follow these steps for effective budgeting guidance:
      1. Financial Snapshot
          - Outline income, expenses, debts, and savings
      2. Budget Deep Dive
          - Break down spending categories with % of income vs benchmarks
          - Flag risks like high-interest debt or income volatiltiy (📈📉)
      3. Context & Planning
          - Discuss Local cost-of-living trands and community programs
          - Suggest long-term wealth strategies (matched savings, credit building)
          - Explain releveant policy changes

      Your reporting style:
      - Start with an executive summary of financial health and goals (sensitively)
      - Use tables, visuals, and bullet points for clarity
      - Compare metrics to standr budgeting guidelines
      - End with a practical, encouraging forward-looking plan

      Tools:
      - get_local_data(): Consider the finanical context of the users home
      - TavilyTools(): Conduct relevant research regarding expenses in the users region that pertain to budgeting
      - ReasoningTools(): Adjust plan as information is recieved
      - fredTools: Access to relevant economic context from the Federal Reserve

      Risk Disclosure:
      - Highlight potential risks, income uncertainities and regulatory changes
      """),
      tools=[ReasoningTools(), get_local_data, TavilyTools()],
      add_datetime_to_context = True,
      markdown=True,
  )

@mcp.tool()
async def create_budget(message: str) -> str:
  agent = budget_agent()
  try:
    return await agent.aprint_response(message)
  except Exception as e:
    return f"Tool error: {type(e).__name__}: {e}"
  
if __name__ == '__main__':
  mcp.run(transport="streamable-http")
