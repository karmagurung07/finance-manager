import asyncio
import os
from dotenv import load_dotenv
from fastmcp import Client, FastMCP
import requests
import json
from fastapi import FastAPI
from textwrap import dedent
from typing import Dict, Optional, Any
from requests.exceptions import JSONDecodeError
from agno.os import AgentOS


from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
mcp = FastMCP()
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

HISTORY_FILE = "chat_history.json"

# llm = OpenAIChat(model_provider="gpt-4.1-mini", model_provider="openai")
# mcp_client = Client(mcp)

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

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# Function to save chat history to file
def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

# Define the analysis function that routes based on agent selection
def analyze_query(query, chat_history, agent_type):
    response = ""
    if agent_type == "Finance Agent":
        # Assuming 'client' (fastmcp client) is defined in a previous cell
        if 'client' not in globals():
            response = "Error: fastmcp client is not initialized for Finance Agent."
        else:
            try:
                # Call the analyze_stock_tool using the fastmcp client
                response = mcp_client.analyze_stock_tool(query)
            except Exception as e:
                response = f"Error calling fastmcp tool for Finance Agent: {e}"
    elif agent_type == "Budget Agent":
        # Assuming a budget_agent or a corresponding fastmcp tool exists
        # You would replace this with the actual call to your budget agent or tool
        if 'budget_agent' not in globals(): # Assuming a budget_agent object exists
             response = "Error: Budget Agent is not initialized."
        else:
            try:
                # Example call to a budget agent (replace with your actual code)
                response = budget_agent.process_query(query) # Assuming a method like process_query
            except Exception as e:
                response = f"Error using Budget Agent: {e}"
    else:
        response = "Please select an agent type."
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": response})
    
    # Save the updated chat history
    save_history(chat_history)
    
async def orchestrator_agent(message: str) -> str:
    budget_agent = MCPTools(transport="streamable-http", url="http://127.0.0.1:8000/mcp")
    stock_agent = MCPTools(transport="streamable-http", url="http://127.0.0.1:8001/mcp")

    await budget_agent.connect()
    await stock_agent.connect()

    agent = Agent(
        model=OpenAIChat(),
        tools=[budget_agent, stock_agent],
        markdown=True,
    )
    """Orchestrator a financial advising process with access to a budget-agent"""
    
    try:
        system = (
                "You are an orchestrator. For financial tasks such as budgeting and investing, "
                "call the MCP tools exposed by the sub-agents:\n"
                " - Budget agent tool(s): e.g., `create_budget`\n"
                " - Stock agent tool(s): e.g., `finance_analyzer`\n"
                "Use them to produce a concise, helpful answer for the user."
            )
        prompt = f"{system}\n\nUser: {message}"

        response: RunOutput = await agent.arun(prompt)
        return response.content
    finally:
         # Close the Agent first so it releases the tools
        try:
            await agent.aclose()
        except Exception:
            pass
        # Then close MCP tool sessions (same task that opened them)
        try:
            await stock_agent.close()
        finally:
            await budget_agent.close()

        
    
if __name__ == "__main__":
    userInput = input()
    print(asyncio.run(
       orchestrator_agent(userInput)
    ))
