import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load .env
load_dotenv()

llm = ChatOpenAI(
    model="google/gemini-2.5-flash",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    streaming=True, 
)

response = llm.invoke("Reply with exactly one word: Hello")

print("=" * 50)
print("Response:")
print(response.content)

print("\nMetadata:")
for k, v in response.response_metadata.items():
    print(f"{k}: {v}")

print("\nModel Used:")
print(response.response_metadata.get("model_name", "Not available"))