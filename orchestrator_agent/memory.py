import os 
import json

# ===== BASIC MEMORY (single user) =====
MEMORY_FILE = "orchestrator_memory.json"

def _mem_load() -> dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _mem_save(mem: dict) -> None:
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

def get_memory() -> dict:
    # minimal defaults
    mem = _mem_load()
    return mem or {
        "user_profile": {
            "location": None,
            "income_net": None,
            "preferences": {"entertainment": "normal"}
        },
        "last_intent": None,
        "last_budget_summary": None
    }

def update_memory(updates: dict) -> None:
    cur = get_memory()
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    _mem_save(cur)

# Allow model/tool to append one line to update memory:
# e.g., MEMORY_JSON {"user_profile": {"income_net": 5000, "location": "NYC"}}
import re
def extract_memory_json(text: str) -> dict:
    m = re.search(r"^MEMORY_JSON\s+(\{.*\})\s*$", text, flags=re.MULTILINE|re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}
    

def show_memory() -> str:
    return json.dumps(get_memory(), indent=2)

def clear_memory() -> None:
    _mem_save({})  # reset