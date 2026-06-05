from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def scrape_and_process_urls(urls: list[str]):
    """Scrape URLs, convert HTML to text, and split into chunks."""
    loader = AsyncHtmlLoader(urls)
    all_documents = await loader.aload()

    # Filter out empty / failed pages
    valid_docs = [doc for doc in all_documents if doc.page_content.strip()]
    if not valid_docs:
        raise ValueError("No content could be extracted from the provided URLs.")

    # HTML → plain text
    html2text = Html2TextTransformer()
    docs_transformed = list(html2text.transform_documents(valid_docs))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs_transformed)