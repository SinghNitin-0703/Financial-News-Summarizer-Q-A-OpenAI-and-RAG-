import asyncio
import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

os.environ["USER_AGENT"] = "FinancialNewsSummarizer/1.0"

from src.config import validate_config
from src.engine import get_llm, get_embeddings
from src.model import create_vector_store, get_rag_chain
from src.processor import scrape_and_process_urls

# --- Startup checks ---
validate_config()

# --- FastAPI app ---
app = FastAPI(
    title="Financial News Summarizer API",
    description="Process financial news URLs and answer questions via RAG.",
)

# --- Shared components ---
llm = get_llm()
embeddings = get_embeddings()
app_state = {"chain": None}
state_lock = asyncio.Lock()


# ====================== API ======================

@app.get("/")
async def root():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")


class ProcessRequest(BaseModel):
    urls: list[str]


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/api/process")
async def process_urls_api(request: ProcessRequest):
    """Scrape the given URLs and build the FAISS vector index."""
    if not request.urls:
        raise HTTPException(status_code=400, detail="Provide at least one URL.")

    try:
        docs = await scrape_and_process_urls(request.urls)

        async with state_lock:
            await asyncio.to_thread(create_vector_store, docs, embeddings)
            app_state["chain"] = await asyncio.to_thread(get_rag_chain, llm, embeddings)

        return {"message": f"Successfully processed {len(request.urls)} URLs."}
    except Exception as e:
        error_msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/ask", response_model=QueryResponse)
async def ask_question_api(request: QueryRequest):
    """Ask a question against the processed documents."""
    chain = app_state["chain"]
    if not chain:
        raise HTTPException(status_code=400, detail="Process URLs first.")

    result = await chain.ainvoke({"query": request.query})
    answer = result.get("result", "No answer found.")
    sources = list({
        d.metadata.get("source", "Unknown")
        for d in result.get("source_documents", [])
    })
    return QueryResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)