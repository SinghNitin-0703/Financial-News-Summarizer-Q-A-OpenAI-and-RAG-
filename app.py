import gradio as gr
import asyncio
import nest_asyncio
import os

os.environ["USER_AGENT"] = "FinancialNewsSummarizer/1.0"
from src.engine import get_llm, get_embeddings
from src.processor import scrape_and_process_urls
from src.model import create_vector_store, get_rag_chain

nest_asyncio.apply()

# Initialize components
llm = get_llm()
embeddings = get_embeddings()

async def process_and_store_urls(url_string):
    urls = [url.strip() for url in url_string.split('\n') if url.strip()]
    if not urls:
        raise gr.Error("Please provide at least one URL.")
    
    docs = await scrape_and_process_urls(urls)
    create_vector_store(docs, embeddings)
    chain = get_rag_chain(llm, embeddings)
    
    return f"Successfully processed {len(urls)} URLs.", chain

def get_answer(query, chain):
    if not chain:
        raise gr.Error("Please process URLs first.")
    result = chain.invoke({"query": query})
    answer = result.get('result', 'No answer found.')
    sources = "\n".join({f"- {d.metadata['source']}" for d in result['source_documents']})
    return answer, sources

# Gradio UI Construction
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    chain_state = gr.State()
    gr.Markdown("# 📈 Financial News Summarizer")
    
    with gr.Row():
        with gr.Column():
            url_input = gr.Textbox(lines=5, label="Enter URLs")
            process_btn = gr.Button("🔗 Process")
            status = gr.Label(value="Status: Ready")
            
        with gr.Column():
            query_input = gr.Textbox(label="Question")
            ask_btn = gr.Button("❓ Get Answer")
            ans_out = gr.Markdown()
            src_out = gr.Markdown()

    process_btn.click(process_and_store_urls, [url_input], [status, chain_state])
    ask_btn.click(get_answer, [query_input, chain_state], [ans_out, src_out])

if __name__ == "__main__":
    demo.launch()