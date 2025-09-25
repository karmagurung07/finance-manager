from agno.agent import Agent
from agno.models.openai import OpenAIChat
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class UserProfile(BaseModel):
    age_group: str = Field(description="User's age group (child, teen, adult, senior)")
    income: str = Field(description="User's monthly income")
    location: str = Field(description="User's location")
    financial_interests: list[str] = Field(description="List of user's finanical interests")
class IntentSpecificQuery(BaseModel):
    budget_query: str = Field(description="User's budget related part of query")
    stock_query: str = Field(description="User's stock related part of query")
    research_query: str = Field(description="user's research related part of query")
class NLUOutput(BaseModel):
    intent: list[str] = Field(description="List of primary intent of the user ('research', 'budget', or 'stock')")
    confidence: float = Field(description="Confidence score between 0-1")
    user_profile: UserProfile
    processed_input: str = Field(description="Cleaned and normalized input text")
    intent_specific_queries: IntentSpecificQuery

nlu_agent = Agent(
    name="nlu-agent",
    model=OpenAIChat(id="gpt-4.1-mini"),
    description="Extract intent, entities, and user profile from natural language input",
    output_schema=NLUOutput,
    instructions="""
                    Analyze the input to determine user intent (stock, budget or research), 
                    parse the specific questions related to the three possible intents and infer user profile characteristics.
                """
)

if __name__ == '__main__':
    response = nlu_agent.run("I make $7000 net and I live in NYC help me budget for the city with a specific focus on high-yield invests and lots of entertainment funds")
    print(response.content.model_dump())
    print(type(response.content.model_dump()))
    print(type(response))
