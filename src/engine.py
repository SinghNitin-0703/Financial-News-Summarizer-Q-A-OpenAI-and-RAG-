from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from src.config import Config


def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.OPENAI_API_VERSION,
        deployment_name="gpt-4.1-mini-2",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=500,
    )


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        api_key=Config.AZURE_OPENAI_API_KEY,
        api_version=Config.OPENAI_API_VERSION,
        deployment="text-embedding-ada-002",
        chunk_size=16,
    )