from typing import Optional
from dotenv import load_dotenv
import os, json, asyncio

from agno.tools.mcp import MCPTools
from agno.agent import RunOutput

from workflow_agents import nlu_agent
from workflow_agents import writer

from util import *

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Agents you already have elsewhere ---
# nlu_agent with output_schema=NLUOutput
# create_budget and finance_analyzer are MCP-exposed tools on two MCP servers

# Writer agent (async-friendly)
async def ensure_mcp():
    global _mcp_connected
    if not _mcp_connected:
        await _budget_mcp.connect()
        await _stock_mcp.connect()
        await _research_mcp.connect()
        _mcp_connected = True

# Persistent MCP tool clients (connect once)
_budget_mcp = MCPTools(transport="streamable-http", url="http://127.0.0.1:8000/mcp", timeout_seconds=120)
_stock_mcp  = MCPTools(transport="streamable-http", url="http://127.0.0.1:8001/mcp", timeout_seconds=240)
_research_mcp = MCPTools(transport="streamable-http", url="http://127.0.0.1:8002/mcp", timeout_seconds=120)
_mcp_connected = False

logger = setup_logger()
async def orchestrator_agent(message: str) -> str:
    logger.info(f"Starting orchestration for message: {message[:50]}...")
    # 1) NLU (await and extract parsed Pydantic or fallback)

    await ensure_mcp()
    metrics=[]

    try:
        nlu_resp: RunOutput = await nlu_agent.arun(message)
        response = nlu_resp.content
        if hasattr(response, "intent") and response.intent is not None:
            intent = getattr(response, "intent", None)
            intent_specific_queries = getattr(response, "intent_specific_queries", None)
            budget_query = getattr(intent_specific_queries, "budget_query", None)
            stock_query = getattr(intent_specific_queries, "stock_query", None)
            research_query = getattr(intent_specific_queries, "research_query", None)
            logger.info(f"==========nlu_cost==========\n{nlu_resp.metrics.to_dict()}")
        
        else:
            # Fallback if your Agno version doesn’t populate .parsed
            # Try to parse JSON; else just use raw content
            try:
                nlu_json = json.loads(nlu_resp.content)
                intent = nlu_json.get("intent")
            except Exception:
                intent = None
        logger.info(f"intent: {intent}")
        metrics.append({"nlu_agent_cost": nlu_resp.metrics.to_dict()})
    

        # 2) Call sub-agent(s) based on intent
        # NOTE: use the processed input, and only include #research if truly required.
        budget_response: Optional[str] = None
        stock_response: Optional[str] = None
        research_response: Optional[str] = None

        try:
            if "budget" in intent:
                # Call the budget tool over MCP
                logger.info("Calling budget tool")
                response.processed_input = budget_query
                budget_call = await _budget_mcp.session.call_tool("create_budget", {"message": response})
                budget_response = budget_call.structuredContent if hasattr(budget_call, "content") else str(budget_call)
                logger.info(f"=============Budget tool completed=============\n{calculate_costs(budget_response['metrics'])}")
                metrics.append({"budget_tool_cost": budget_response["metrics"]})
             
            if "stock" in intent:
                logger.info("Calling stock tool")
                response.processed_input = stock_query
                stock_call = await _stock_mcp.session.call_tool("finance_analyzer", {"query": response})
                stock_response = stock_call.structuredContent if hasattr(stock_call, "content") else str(stock_call)
                logger.info(f"=============Stock tool complete=============\n{calculate_costs(stock_response['metrics'])}")
                metrics.append({"stock_tool_cost": stock_response["metrics"]})

            if "research" in intent:
                logger.info("Calling research tool")
                response.processed_input = research_query
                research_call = await _research_mcp.session.call_tool("research", {"message": response})
                research_response = research_call.structuredContent if hasattr(research_call, "content") else str(research_call)
                logger.info(f"=============Research tool complete=============\n{calculate_costs(research_response['metrics'])}")
                metrics.append({"research_tool_cost": research_response["metrics"]})
        finally:
            pass
            

        # 3) Build the writer prompt safely (only include non-empty sections)
        sections = ["Summarize the available sections below. Include tables for any budget section. "
                    "Only include sections that are provided."]

        if budget_response:
            sections.append(f"\n=== budget_response ===\n{budget_response["content"]}")
        if stock_response:
            sections.append(f"\n=== stock_response ===\n{stock_response["content"]}")
        if research_response:
            sections.append(f"\n=== research_response ===\n{research_response["content"]}")

        writer_prompt = "\n".join(sections)

        # 4) Ask the writer to produce the final answer
        logger.info("Writing final resposne")
        writer_out: RunOutput = await writer.arun(writer_prompt)
        
        final_text = writer_out.content
        
        metrics.append({"writing_agent_cost": writer_out.metrics.to_dict()})
        total_metric = total_metrics(metrics)
        
        logger.info(f"Orchestration completed successfully\n====Total Cost====\n{calculate_costs(total_metric)}")
        for metric in metrics:
            name = list(metric.keys())[0]
            logger.info(f"==========={name}===========\n {calculate_costs(metric[name])}")
        return final_text
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        raise
if __name__ == '__main__':
    userInput = input()
   
    print(asyncio.run(orchestrator_agent(userInput)))