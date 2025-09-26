# Objective:
Finance Seer is a gpt-5 powered agent that assists with finanical questions such as budgeting, investing and general finanical literacy. This product is dedicated to providing accessible resources to people from divested and marginalized groups. That is way the this project is built on gpt-5-nano, the cheapest gpt options for power and accessibility.

# Operation Workflow:
- User Submitted generated query
- A Natural Language Understanding Agents parses out key variables
    - Intent
    - Intent Specific Queries
    - Income
    - Location 
    - goal
- The NLU output is ingested by our Orchestrator workflow which will:
    - Call tools based on listed intent
    - Provide tool with intent specific query
- Then the responses are compiled into a prompt
- The Writer Agent then formats the prompt as fit combining all sections. 

# Technologies:
- Framework: Agno
- Language: Python
- Model: gpt-5-nano

# Why Agno?
The main reason for choosing Agno was the it 

# Getting Started
- First copy the repository link and use `git clone {url}` in terminal to download the project
- Add a `.env` file to the root directory finance_manager with your OPENAI_API_KEY and TAVILY_API_KEY
- Set up a venv or conda python envrionment
- Install dependecies from /sub_agents/requirements.txt
- Open up docker desktop
- Now go to finance_manager/sub_agents/ in terminal of choice and write `docker compose up --build`
- Now in another terminal go to finance_manager/orchestrator_agent and write `python run_gradio.py`
- Everything should be load at this point so just go to `http://localhost:7860`
