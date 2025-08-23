# Anthropic-specific configuration
import os

# Model parameters
PARAMETERS = {
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman:"],
}

# Anthropic API credentials
CREDENTIALS = {
    "api_key": os.getenv("ANTHROPIC_API_KEY"), # questionable as model.py also has its api key, so is it needed here?
    "api_url": "https://api.anthropic.com/v1/messages"
}

# Model IDs
CLAUDE_3_OPUS = "claude-3-opus-20240229"
CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
CLAUDE_3_HAIKU = "claude-3-haiku-20240307"