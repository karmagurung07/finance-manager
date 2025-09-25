import asyncio
import os, json
from dotenv import load_dotenv
from typing import Optional
from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
import memory

from nlu_agent import nlu_agent

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def with_context(message: str, mem: dict) -> str:
    ctx = json.dumps(mem, ensure_ascii=False)
    return (
        "Use USER_CONTEXT if helpful; only ask for missing critical info.\n"
        f"USER_CONTEXT:\n{ctx}\n\n"
        f"USER_MESSAGE:\n{message}"
    )

# --- Agents you already have elsewhere ---
# nlu_agent with output_schema=NLUOutput
# create_budget and finance_analyzer are MCP-exposed tools on two MCP servers

# Writer agent (async-friendly)
writer = Agent(
    model=OpenAIChat(),
    instructions=(
        "You are a financial writing agent. Take provided responses (budget/stock/research) "
        "and write clearly and concisely. Include tables for any budget section. "
        "Only include sections that have content."
    ),
    markdown=True,
)

# Persistent MCP tool clients (connect once)
_budget_mcp = MCPTools(transport="streamable-http", url="http://127.0.0.1:8000/mcp", timeout_seconds=120)
_stock_mcp  = MCPTools(transport="streamable-http", url="http://127.0.0.1:8001/mcp", timeout_seconds=120)
_research_mcp = MCPTools(transport="streamable-http", url="http://127.0.0.1:8002/mcp", timeout_seconds=120)
_mcp_connected = False

async def ensure_mcp():
    global _mcp_connected
    if not _mcp_connected:
        await _budget_mcp.connect()
        await _stock_mcp.connect()
        await _research_mcp.connect()
        _mcp_connected = True

async def _safe_close(tool, name: str):
    # Close only if a session exists; swallow errors so others still close
    try:
        if tool is not None and getattr(tool, "session", None):
            await tool.close()
    except Exception as e:
        print(f"[close_tools] Warning: failed to close {name}: {e}")

async def close_tools():
    """
    Close MCP tool sessions in reverse order of connection.
    Keep this in the SAME task/loop that connected them.
    """
    # tools may not all exist depending on your setup; use globals().get to avoid NameError
    research = globals().get("_research_mcp", None)
    stock    = globals().get("_stock_mcp", None)
    budget   = globals().get("_budget_mcp", None)

    # Reverse of ensure/connect order: research -> stock -> budget
    await _safe_close(research, "research_mcp")
    await _safe_close(stock,    "stock_mcp")
    await _safe_close(budget,   "budget_mcp")

def route_intent(intent: str) -> str:
    intent = (intent or "").lower().strip()
    if intent in {"budget", "stock", "research"}:
        return intent
    # default to budget if unclear
    return "budget"

async def orchestrator_agent(message: str) -> str:
    await ensure_mcp()

    mem = memory.get_memory()

    # 1) NLU (await and extract parsed Pydantic or fallback)
    nlu_resp: RunOutput = await nlu_agent.arun(message)
    response = nlu_resp.content
    if hasattr(response, "intent") and response.intent is not None:
        intent = getattr(response, "intent", None)
        intent_specific_queries = getattr(response, "intent_specific_queries", None)
        budget_query = getattr(intent_specific_queries, "budget_query", None)
        stock_query = getattr(intent_specific_queries, "stock_query", None)
        research_query = getattr(intent_specific_queries, "research_query", None)
    else:
        # Fallback if your Agno version doesn’t populate .parsed
        # Try to parse JSON; else just use raw content
        try:
            nlu_json = json.loads(nlu_resp.content)
            intent = nlu_json.get("intent")
        except Exception:
            intent = None
    print(f"intent: {intent}")
    memory.update_memory({"last_intent:": intent})

    # 2) Call sub-agent(s) based on intent
    # NOTE: use the processed input, and only include #research if truly required.
    budget_response: Optional[str] = None
    stock_response: Optional[str] = None
    research_response: Optional[str] = None

    metrics=[]

    try:
        if "budget" in intent:
            # Call the budget tool over MCP
            response.processed_input = budget_query
            budget_call = await _budget_mcp.session.call_tool("create_budget", {"message": response})
            budget_metrics = budget_call if hasattr(budget_call, "metrics") else str(budget_call)
            budget_response = budget_call.content if hasattr(budget_call, "content") else str(budget_call)
            print(f"=========Budget_Metrics=========\n{budget_metrics}")
            print(f"=========Budget_Response=========\n{budget_response}")

            upd = memory.extract_memory_json(budget_response[0].text)
            if upd:
                memory.update_memory(upd)
                
        if "stock" in intent:
            response.processed_input = stock_query
            stock_call = await _stock_mcp.session.call_tool("finance_analyzer", {"query": response})
            stock_response = stock_call.content if hasattr(stock_call, "content") else str(stock_call)
            print(f"=========Stock_Response=========\n{stock_response}")

        if "research" in intent:
            response.processed_input = research_query
            # If you actually have a research tool, call it here.
            # For now, reuse budget with a #research tag if that’s your convention.
            research_call = await _research_mcp.session.call_tool("research", {"message": response})
            research_response = research_call.content if hasattr(research_call, "content") else str(research_call)
            print(f"=========Research_Response=========\n{research_response}")
            upd = memory.extract_memory_json(research_response[0].text)
            if upd:
                memory.update_memory(upd)
    finally:
        print("Successful")

    # 3) Build the writer prompt safely (only include non-empty sections)
    sections = ["Summarize the available sections below. Include tables for any budget section. "
                "Only include sections that are provided."]

    if budget_response:
        sections.append(f"\n=== budget_response ===\n{budget_response}")
    if stock_response:
        sections.append(f"\n=== stock_response ===\n{stock_response}")
    if research_response:
        sections.append(f"\n=== research_response ===\n{research_response}")

    writer_prompt = "\n".join(sections)

    # 4) Ask the writer to produce the final answer
    writer_out: RunOutput = await writer.arun(writer_prompt)
    final_text = writer_out.content

    upd = memory.extract_memory_json(final_text)
    if upd:
        memory.update_memory(upd)
    return final_text

if __name__ == '__main__':
    userInput = input()
    print(asyncio.run(orchestrator_agent(userInput)))