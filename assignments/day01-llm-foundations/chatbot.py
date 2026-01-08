import os
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Choose model
MODEL_NAME = "gemini-2.5-flash"

def chat():
    print("🤖 Gemini LLM Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("👋 Exiting chatbot.")
            break

        prompt = f"""
You are a helpful, concise AI assistant.

User question:
{user_input}

Answer clearly and briefly.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        # Output
        print("\nAssistant:", response.text)

        # Token usage (available in new SDK)
        if response.usage_metadata:
            print("\n--- Token Usage ---")
            print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
            print(f"Completion Tokens: {response.usage_metadata.candidates_token_count}")
            print(f"Total Tokens: {response.usage_metadata.total_token_count}")
            print("-------------------\n")
        else:
            print("\n(Token usage not available)\n")


if __name__ == "__main__":
    chat()
