import asyncio
import os
import sys

# Setup environment to match app.py
from src.config import validate_config
from src.processor import scrape_and_process_urls
from src.engine import get_llm, get_embeddings
from src.model import create_vector_store

os.environ["USER_AGENT"] = "FinancialNewsSummarizer/1.0"
validate_config()

urls = ["https://www.moneycontrol.com/news/business/economy/consumer-spending-holds-firm-in-india-s-7-7-gdp-growth-but-watch-out-for-oil-monsoon-hazards-13942371.html"]

async def main():
    try:
        print("Scraping...")
        docs = await scrape_and_process_urls(urls)
        print(f"Scraped {len(docs)} documents.")
        
        print("Getting embeddings...")
        embeddings = get_embeddings()
        
        print("Creating vector store...")
        await asyncio.to_thread(create_vector_store, docs, embeddings)
        print("Success.")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
