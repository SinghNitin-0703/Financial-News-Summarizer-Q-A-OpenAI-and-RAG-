import asyncio
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

async def scrape_and_process_urls(urls):
    loader = AsyncHtmlLoader(urls)
    # Extract
    all_documents = await loader.aload()
    # Transform
    html2text = Html2TextTransformer()
    docs_transformed = list(html2text.transform_documents(all_documents))
    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs_transformed)