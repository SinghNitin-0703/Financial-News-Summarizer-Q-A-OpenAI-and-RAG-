import asyncio
from langchain_community.document_loaders import AsyncHtmlLoader

async def main():
    loader = AsyncHtmlLoader(["https://www.example.com"])
    try:
        docs = await loader.aload()
        print(f"Loaded {len(docs)} documents with aload")
    except Exception as e:
        print(f"Exception aload: {type(e)} {e}")
        try:
            docs = loader.load()
            print(f"Loaded {len(docs)} documents with load")
        except Exception as e2:
            print(f"Exception load: {type(e2)} {e2}")

asyncio.run(main())
