
import os

from dotenv import load_dotenv
from openai import OpenAI, APIError

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
model = os.getenv("LLM_MODEL")

if not api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing from the environment."
    )

if not model:
    raise RuntimeError(
        "LLM_MODEL is missing from the environment."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    timeout=30.0,
    max_retries=0,
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly one word: GROUNDED.",
            }
        ],
    )

    if not response.choices:
        print("The model returned no choices.")
    else:
        result = response.choices[0].message.content

        if result and result.strip():
            print(result)
        else:
            print("The model returned an empty response.")

except APIError as e:
    status_code = getattr(e, "status_code", None)
    error_message = str(e).replace("\n", " ")[:500]

    print(f"Error type: {type(e).__name__}")
    print(f"HTTP status: {status_code}")
    print(f"Details: {error_message}")

except Exception as e:
    error_message = str(e).replace("\n", " ")[:500]

    print(f"Unexpected error type: {type(e).__name__}")
    print(f"Details: {error_message}")
