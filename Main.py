import os
import asyncio
import gradio as gr
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import Html2TextTransformer
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Allows asyncio to be used in environments that already have a running event loop
import nest_asyncio
nest_asyncio.apply()

# --- 1. Initialize LLM and Embeddings ---
# IMPORTANT: Create a file named 'OpenAI_APIkey.env' in the same directory
# and add your Azure OpenAI credentials to it.
# Example content for OpenAI_APIkey.env:
# AZURE_OPENAI_ENDPOINT="your_endpoint"
# AZURE_OPENAI_API_KEY="your_api_key"
# OPENAI_API_VERSION="2024-02-01"

try:
    load_dotenv(dotenv_path="OpenAI_APIkey.env")

    # Initialize the LLM for chat
    llm = AzureChatOpenAI(
        deployment_name="gpt-4.1-mini-2",
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=500
    )

    # Initialize the Embeddings model
    embeddings = AzureOpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        chunk_size=1
    )
    
    # Path for storing the local vector index
    FAISS_INDEX_PATH = "faiss_index"

except Exception as e:
    print(f"Error initializing Azure services: {e}")
    print("Please ensure your 'OpenAI_APIkey.env' file is correctly configured.")
    llm = None
    embeddings = None


# --- 2. Core Logic Functions ---

async def process_and_store_urls(url_string):
    """
    Loads content from URLs, processes it, and creates a FAISS vector store.
    """
    if not llm or not embeddings:
        raise gr.Error("LLM and Embeddings are not initialized. Check your .env file.")

    urls = [url.strip() for url in url_string.split('\n') if url.strip()]
    if not urls:
        raise gr.Error("Please provide at least one URL.")

    try:
        # Load documents using the async loader
        loader = AsyncHtmlLoader(urls)
        
        async def load_docs():
            return await loader.aload()
        
        all_documents = asyncio.run(load_docs())

        # Transform HTML to plain text
        html2text = Html2TextTransformer()
        docs_transformed = list(html2text.transform_documents(all_documents))

        # Split documents into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs_split = text_splitter.split_documents(docs_transformed)

        if not docs_split:
            return "Could not extract any content from the provided URLs. Please check them.", None

        # Create and save the FAISS vector index
        vector_index = FAISS.from_documents(docs_split, embeddings)
        vector_index.save_local(FAISS_INDEX_PATH)
        
        # Load the index and create the QA chain
        vector_index_loaded = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_index_loaded.as_retriever(),
            return_source_documents=True
        )

        return f"Successfully processed {len(urls)} URLs and created the knowledge base.", chain

    except Exception as e:
        raise gr.Error(f"An error occurred during URL processing: {e}")


def get_answer_from_query(query, chain):
    """
    Executes a query against the RetrievalQA chain and returns the answer and sources.
    """
    if not query:
        return "", ""
    if not chain:
        raise gr.Error("The knowledge base has not been created yet. Please process URLs first.")

    try:
        result = chain.invoke({"query": query})
        
        answer = result.get('result', 'No answer found.').strip()
        
        sources_text = "No sources found."
        if 'source_documents' in result and result['source_documents']:
            unique_sources = {doc.metadata['source'] for doc in result['source_documents']}
            sources_text = "\n".join(f"- {source}" for source in unique_sources)

        return answer, sources_text

    except Exception as e:
        raise gr.Error(f"An error occurred while fetching the answer: {e}")


# --- 3. Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft(), title="Financial News Summarizer") as demo:
    chain_state = gr.State()

    gr.Markdown(
        """
        # 📈 Financial News Summarizer & Q&A
        Enter news article URLs to create a knowledge base, then ask questions about the content.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Step 1: Create Knowledge Base")
            url_input = gr.Textbox(
                lines=5,
                label="Enter Article URLs",
                placeholder="https://www.example.com/news-article-1\nhttps://www.example.com/news-article-2"
            )
            process_button = gr.Button("🔗 Process URLs", variant="primary")
            status_output = gr.Label(value="Status: Waiting for URLs")

        with gr.Column(scale=2):
            gr.Markdown("### Step 2: Ask a Question")
            question_input = gr.Textbox(
                lines=2,
                label="Your Question",
                placeholder="e.g., What was the net direct tax revenue jump?"
            )
            ask_button = gr.Button("❓ Get Answer")
            
            gr.Markdown("---")
            
            gr.Markdown("#### Answer")
            answer_output = gr.Markdown()
            
            gr.Markdown("#### Sources")
            sources_output = gr.Markdown()

    # --- Event Handlers ---
    process_button.click(
        fn=process_and_store_urls,
        inputs=[url_input],
        outputs=[status_output, chain_state]
    )

    ask_button.click(
        fn=get_answer_from_query,
        inputs=[question_input, chain_state],
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)