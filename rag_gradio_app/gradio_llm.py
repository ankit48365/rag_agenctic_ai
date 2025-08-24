# This program provides an LLM Q&A function for Gradio interface
# Uses Anthropic's Claude API for text generation

import anthropic
import os
import gradio as gr

def llm_qa_ui(user_input):
    """
    Function to handle LLM Q&A for Gradio interface
    
    Args:
        user_input (str): The user's question/prompt
        
    Returns:
        str: The LLM's response or error message
    """
    # Get Anthropic API key from environment
    api_key = str(os.getenv("ANTHROPIC_API_KEY"))
    
    if not api_key or api_key == "None":
        return "Error: ANTHROPIC_API_KEY environment variable not set. Please set your Anthropic API key."
    
    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Validate input
    if not user_input or user_input.strip() == "":
        return "Please enter a question or prompt."
    
    try:
        # Call Anthropic API
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Cost-effective and fast model
            max_tokens=1000,  # Increased for longer responses
            temperature=0,  # Deterministic response
            messages=[
                {"role": "user", "content": user_input.strip()}
            ]
        )
        
        # Extract and return the generated text
        generated_text = response.content[0].text
        return generated_text
        
    except anthropic.AuthenticationError:
        return "Error: Invalid Anthropic API key. Please check your API key."
    except anthropic.RateLimitError:
        return "Error: Rate limit exceeded. Please try again later. This can happen if you've made too many requests recently."
    except anthropic.APIError as e:
        return f"Anthropic API error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

# Create Gradio interface
chat_application = gr.Interface(
    fn=llm_qa_ui,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
if __name__ == "__main__":
    chat_application.launch(server_name="127.0.0.1", server_port=7860)
