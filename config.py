import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="OpenAI_APIkey.env")

class Config:
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-01")
    FAISS_INDEX_PATH = "faiss_index"