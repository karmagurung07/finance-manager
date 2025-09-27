import os
import json
import re
from dotenv import load_dotenv
from fastmcp import FastMCP
import requests
from fastapi import FastAPI
from textwrap import dedent
from typing import Dict, List, Optional, Any
from requests.exceptions import JSONDecodeError
from agno.os import AgentOS
from agno.agent import Agent, RunOutput
from agno.exceptions import RetryAgentRun
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools
from agno.tools.tavily import TavilyTools



mcp = FastMCP()
MAX_TURNS = 8
load_dotenv()

# ====== BASIC MEMORY ======
MEMORY_FILE = "budget_memory.json"

def _mem_load():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}  # no memory yet

def _mem_save(mem: dict):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

def get_memory() -> dict:
    # Single-user, minimal defaults
    mem = _mem_load()
    return mem or {
        "currency": "USD",
        "locale": "US-NY",
        "profile": {        # anything stable about finances
            "income_net": None,
            "fixed_expenses": {},
            "debts": [],
            "savings_goal": None
        },
        "last_plan": None
    }

def update_memory(updates: dict):
    # Shallow merge only
    cur = get_memory()
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    _mem_save(cur)

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
  tools = [ReasoningTools(), get_local_data]
  user_ctx = get_memory()

  return Agent(
    name="budget-agent",
    model=OpenAIChat(id="gpt-5-nano"),
    instructions=dedent(f"""
            Concise budgeting assistant. Output under ~500 words.
            1) Snapshot (income, expenses, debts, savings)
            2) Budget by category (% vs common benchmarks)
            3) Risks + 3 quick wins
            4) 4-week action plan
            Only call tools if strictly needed.
                        
            don't use chat_history unless explicitly referenced

            If you learn stable facts (income, fixed bills, debts/APRs, savings goal, locale),
            append a single MEMORY_JSON line at the very end with a small JSON object
            containing ONLY changed fields. Example:
            MEMORY_JSON {{"profile": {{"income_net": 5200}}}}

            USER_CONTEXT:
            {json.dumps(user_ctx, ensure_ascii=False)}
      """),
      tools=tools,
      add_history_to_context=True,
      num_history_runs=3,
      add_datetime_to_context = True,
      markdown=True,
  )

def extract_memory_json(text: str) -> dict:
    # Look for a line starting with 'MEMORY_JSON ' followed by {...}
    m = re.search(r'^MEMORY_JSON\s+(\{.*\})\s*$', text, flags=re.MULTILINE|re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}

  
retry_count = 0
@mcp.tool()
async def create_budget(message: dict) -> dict:
  input = message['processed_input']
  agent = budget_agent()
  try:
    response: RunOutput = await agent.arun(input)
    text = response.content if hasattr(response, "content") else str(response)
    mem_updates = extract_memory_json(text)
    if mem_updates:
        update_memory(mem_updates)
        # Optionally remove the MEMORY_JSON line from the final reply:
        text = re.sub(r'^MEMORY_JSON\s+(\{.*\})\s*$', '', text, flags=re.MULTILINE)
    
    if text == None and retry_count<2:
      retry_count+=1
      raise RetryAgentRun(
         f"Create or provide information about budget based on the users response. users:{input}"
      )
    return {"metrics": response.metrics.to_dict(),"content": text}
  except Exception as e:
    return f"Tool error: {type(e).__name__}: {e}"
  
if __name__ == '__main__':
  mcp.run(transport="streamable-http")