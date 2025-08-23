# this program runs alone to test the Anthropic API call

#  uv add anthropic
# Let's make our very first call to Anthropic's Claude models.
# This version uses Anthropic's API instead of IBM Watson's Granite model.

# This imports the required modules to authenticate and interact with Anthropic's API.
import anthropic
import os
# Set up Anthropic API key - hardcoded as requested
# Replace "your-anthropic-api-key-here" with your actual Anthropic API key
api_key = str(os.getenv("ANTHROPIC_API_KEY"))
# api_key = input("Enter your Anthropic API key: ")

# Create Anthropic client
client = anthropic.Anthropic(api_key=api_key)

# The parameters for Anthropic's message API:
# model: Specifies which Claude model to use (claude-3-haiku-20240307 is cost-effective and fast)
# max_tokens: This sets the maximum number of tokens the LLM can generate in a response
# temperature: Controls randomness (0 = deterministic, 1 = more creative)

# This sets up a text prompt and uses Anthropic's messages API to get a response
text = input("Enter your prompt: ")

# text = """
# Only reply with the answer. What is the capital of Canada?
# """

try:
    response = client.messages.create(
        model="claude-3-haiku-20240307",  # You can also use "claude-3-sonnet-20240229" or "claude-3-opus-20240229"
        max_tokens=100,
        temperature=0,  # Using 0 for deterministic response, similar to greedy decoding
        messages=[
            {"role": "user", "content": text}
        ]
    )
    
    # Extract and print the generated text
    generated_text = response.content[0].text
    print(generated_text)
    
except anthropic.AuthenticationError:
    print("Error: Invalid Anthropic API key. Please check your API key.")
except anthropic.RateLimitError:
    print("Error: Rate limit exceeded. Please try again later.")
    print("This can happen if you've made too many requests recently.")
    print("Wait a few minutes before trying again, or consider upgrading your Anthropic plan.")
except anthropic.APIError as e:
    print(f"Anthropic API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
