import os
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-2.5-flash"
INPUT_PATH = "question.txt"
OUTPUT_PATH = "answers.txt"


def is_rate_limit_error(error: Exception) -> bool:
    return "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)


def Answering():
    try:
        with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
             open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:

            question_number = 1

            for line in infile:
                content = line.strip()
                if not content:
                    continue

                prompt = f"""
You are a helpful, concise AI assistant.

User question:
{content}

Answer clearly and briefly.
"""

                while True:
                    try:
                        response = client.models.generate_content(
                            model=MODEL_NAME,
                            contents=prompt
                        )
                        break
                    except Exception as e:
                        if is_rate_limit_error(e):
                            print("⚠️ Rate limit hit. Waiting 40 seconds...")
                            time.sleep(40)
                        else:
                            raise e

                outfile.write(
                    f"Question {question_number}:\n"
                    f"{content}\n\n"
                    f"Answer:\n"
                    f"{response.text.strip()}\n\n"
                    f"{'-' * 40}\n\n"
                )

                question_number += 1
                time.sleep(2)  # gentle throttling

    except FileNotFoundError:
        print(f"❌ File not found: {INPUT_PATH}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    Answering()
    print(f"✅ Answers written to '{OUTPUT_PATH}'.")
