from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

try:
    vector = embeddings.embed_query("Hello, this is a test.")

    print("✅ Azure connection successful!")
    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")

except Exception as e:
    print("❌ Error:")
    print(e)