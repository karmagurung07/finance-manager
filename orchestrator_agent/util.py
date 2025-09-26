from collections import defaultdict
import logging
from agno.utils.log import configure_agno_logging

def total_metrics(metrics_list):
    """
    Take a list of dictionaries like:
    [{'nlu_agent_metrics': {...}}, {'research_tool_metrics': {...}}]
    and return a single dict summing all like keys (input_tokens, output_tokens…)
    across all agents.
    """
    totals = defaultdict(float)

    for item in metrics_list:
        for inner_dict in item.values():
            for metric, value in inner_dict.items():
                if isinstance(value, (int, float)):
                    totals[metric] += value

    return dict(totals)

def calculate_costs(tokens_dict):
    """
    Calculates the cost for each token type based on gpt-5-nano pricing.
    reasoning_tokens are treated as cached input tokens.
    Returns a dictionary with the same keys but costs instead of token amounts.
    """
    # Prices per 1M tokens
    price_input = 0.050 / 1_000_000       # normal input tokens
    price_cached_input = 0.005 / 1_000_000  # cached input (reasoning tokens)
    price_output = 0.400 / 1_000_000       # output tokens
    input_tokens = tokens_dict.get('input_tokens', 0.0)
    output_tokens = tokens_dict.get('output_tokens', 0.0)
    reasoning_tokens = tokens_dict.get('reasoning_tokens', 0.0)
    total_tokens = tokens_dict.get('total_tokens', 0.0)

    return {
        'input_tokens': input_tokens * price_input,
        'output_tokens': output_tokens * price_output,
        'total_tokens': total_tokens * price_input,  # or recompute sum if you want
        'reasoning_tokens': reasoning_tokens * price_cached_input,
        'duration': tokens_dict.get('duration')
    }



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

def setup_logger():
    logger = logging.getLogger("orchestrator")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Configure Agno to use custom logger
    configure_agno_logging(custom_default_logger=logger)
    return logger
