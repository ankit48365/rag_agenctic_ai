# >> calls >> config.py

from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from config import PARAMETERS, CLAUDE_3_HAIKU, CLAUDE_3_OPUS, CLAUDE_3_SONNET
import os

################################################################################################
#below import are resonsible for making JSON fornmatted output, i.e. making it API friendly
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Define JSON output structure
# class AIResponse(BaseModel):
#     summary: str = Field(description="Summary of the user's message")
#     sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
#     response: str = Field(description="Suggested response to the user")

class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    category: str = Field(description="Category of the inquiry (e.g., billing, technical, general)")
    action: str = Field(description="Recommended action for the support rep")


# JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)

# JSON BLOCK ENDS###########################################################################################



# Function to initialize a model
def initialize_model(model_id):
    return ChatAnthropic(
        model=model_id,
        api_key= os.getenv("ANTHROPIC_API_KEY"),
        **PARAMETERS
    )

# initialize models
opus_ll = initialize_model(CLAUDE_3_OPUS)
sonnet_ll = initialize_model(CLAUDE_3_SONNET)
haiku_ll = initialize_model(CLAUDE_3_HAIKU)


# Prompt template for Anthropic API (Claude models)
anthropic_template = PromptTemplate(
    template="""
Human: {system_prompt}

{user_prompt}

Assistant:""",
    input_variables=["system_prompt", "user_prompt"]
)

def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model | json_parser
    return chain.invoke({'system_prompt':system_prompt, 'user_prompt':user_prompt})


# Model-specific response functions
def opus_response(system_prompt, user_prompt):
    return get_ai_response(opus_ll, anthropic_template, system_prompt, user_prompt)

def sonet_response(system_prompt, user_prompt):
    return get_ai_response(sonnet_ll, anthropic_template, system_prompt, user_prompt)

def haiku_response(system_prompt, user_prompt):
    return get_ai_response(haiku_ll, anthropic_template, system_prompt, user_prompt)
