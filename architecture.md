# 🏗️ Architecture — Financial News Summarizer

> A Retrieval-Augmented Generation (RAG) system that ingests live financial news
> from URLs, builds a searchable knowledge base, and answers user questions
> grounded in the actual article text.

---

## 🧠 Project Mind Map

```mermaid
mindmap
  root((📈 Financial News<br/>Summarizer))
    🌐 Entry Points
      FastAPI REST API
        POST /api/process
        POST /api/ask
        GET /docs — Swagger UI
      Gradio Web UI
        URL input panel
        Q&A panel
        Mounted at root /
    ⚙️ Core Pipeline
      🔗 Scraping — processor.py
        AsyncHtmlLoader
        Html2TextTransformer
        RecursiveCharacterTextSplitter
      🧮 Indexing — model.py
        FAISS.from_documents
        save_local to disk
      🤖 RAG Q&A — model.py
        FAISS.load_local
        RetrievalQA chain
        Source attribution
    🔧 Infrastructure
      🔑 Config — config.py
        load_dotenv — absolute path
        validate_config — fail-fast
        FAISS_INDEX_PATH
      🧪 Engine — engine.py
        AzureChatOpenAI — GPT-4
        AzureOpenAIEmbeddings — Ada-002
      🔒 Concurrency — app.py
        asyncio.Lock
        asyncio.to_thread
    ☁️ External Services
      Azure OpenAI
        GPT-4 — generation
        Ada-002 — embeddings
      Target News URLs
        Financial articles
        Any public webpage
```

---

## 🔄 Request Flow — End to End

The system has **two phases**: **Ingestion** (processing URLs) and **Retrieval** (answering questions).

### Phase 1 — Ingestion (`POST /api/process` or Gradio 🔗 Process)

```mermaid
flowchart LR
    A["👤 User<br/>submits URLs"] --> B["app.py<br/>Validate input"]
    B --> C["processor.py<br/>scrape_and_process_urls"]
    
    subgraph SCRAPE ["🔗 Scraping Pipeline"]
        direction TB
        C1["AsyncHtmlLoader<br/>Fetch HTML concurrently"] --> C2["Filter empty pages"]
        C2 --> C3["Html2TextTransformer<br/>HTML → plain text"]
        C3 --> C4["RecursiveCharacterTextSplitter<br/>Split into 1000-char chunks"]
    end
    
    C --> C1
    C4 --> D["model.py<br/>create_vector_store"]
    
    subgraph INDEX ["🧮 Indexing"]
        direction TB
        D1["Ada-002 embeddings<br/>Text → 1536-dim vectors"] --> D2["FAISS.from_documents<br/>Build similarity index"]
        D2 --> D3["save_local<br/>Persist to disk"]
    end
    
    D --> D1
    D3 --> E["model.py<br/>get_rag_chain"]
    E --> F["✅ Chain ready<br/>Stored in app_state"]

    style SCRAPE fill:#1a1a2e,stroke:#e94560,color:#eee
    style INDEX fill:#1a1a2e,stroke:#0f3460,color:#eee
```

### Phase 2 — Retrieval (`POST /api/ask` or Gradio ❓ Get Answer)

```mermaid
flowchart LR
    A["👤 User<br/>asks a question"] --> B["app.py<br/>Get chain from state"]
    B --> C["chain.ainvoke"]
    
    subgraph RAG ["🤖 RAG Pipeline"]
        direction TB
        R1["Ada-002<br/>Embed the question"] --> R2["FAISS retriever<br/>Find top-k similar chunks"]
        R2 --> R3["Stuff chain<br/>Pack chunks into prompt"]
        R3 --> R4["GPT-4<br/>Generate grounded answer"]
    end
    
    C --> R1
    R4 --> D["Extract answer +<br/>source documents"]
    D --> E["👤 User sees<br/>answer + sources"]

    style RAG fill:#1a1a2e,stroke:#16c79a,color:#eee
```

---

## 📁 Module Breakdown

### File Structure

```
News_summ_RAG_and_fewshot/
│
├── app.py                  ← 🚀 Entry point, API + UI
├── requirements.txt        ← 📦 Dependencies
├── OpenAI_APIkey.env       ← 🔑 Azure credentials (user-created)
├── README.md               ← 📖 Documentation
├── architecture.md         ← 🏗️ This file
│
├── src/
│   ├── __init__.py         ← Package marker
│   ├── config.py           ← ⚙️ Env loading & validation
│   ├── engine.py           ← 🧪 LLM & embedding factory
│   ├── processor.py        ← 🔗 Web scraping pipeline
│   └── model.py            ← 🧮 Vector store & RAG chain
│
└── faiss_index/            ← 💾 Generated at runtime (not committed)
```

---

### 1 · `config.py` — ⚙️ Configuration & Validation

| Aspect | Detail |
|---|---|
| **What it does** | Loads Azure credentials from `OpenAI_APIkey.env`, exposes them via `Config` class, provides `validate_config()` for fail-fast startup |
| **Why necessary** | Centralizes all secrets and paths. Uses `Path(__file__)` to resolve the `.env` path relative to the project root — works regardless of where `python app.py` is run from |
| **Key exports** | `Config.AZURE_OPENAI_ENDPOINT`, `Config.AZURE_OPENAI_API_KEY`, `Config.OPENAI_API_VERSION`, `Config.FAISS_INDEX_PATH`, `validate_config()` |

```
OpenAI_APIkey.env ──load_dotenv──► Config class ──validate_config──► ✅ or ❌ EnvironmentError
```

---

### 2 · `engine.py` — 🧪 LLM & Embedding Factory

| Aspect | Detail |
|---|---|
| **What it does** | Creates and returns pre-configured Azure OpenAI clients: one for text generation (GPT-4), one for embeddings (Ada-002) |
| **Why necessary** | Decouples client creation from business logic. Credentials are explicitly wired from `Config` — no hidden env-var side effects |
| **Key exports** | `get_llm()` → `AzureChatOpenAI`, `get_embeddings()` → `AzureOpenAIEmbeddings` |

```
Config credentials ──► get_llm()        ──► AzureChatOpenAI  (GPT-4, temp=0.7)
                   ──► get_embeddings() ──► AzureOpenAIEmbeddings (Ada-002, batch=16)
```

---

### 3 · `processor.py` — 🔗 Web Scraping Pipeline

| Aspect | Detail |
|---|---|
| **What it does** | Takes a list of URLs → fetches HTML concurrently → converts to plain text → splits into overlapping chunks |
| **Why necessary** | Raw HTML is noisy (scripts, CSS, nav bars). This module extracts only readable content and chunks it to fit the embedding model's context window |
| **Key export** | `scrape_and_process_urls(urls)` → `list[Document]` |

```mermaid
flowchart LR
    A["List of URLs"] --> B["AsyncHtmlLoader<br/>⚡ Concurrent fetch"]
    B --> C{"Filter empty<br/>pages"}
    C -->|"valid"| D["Html2TextTransformer<br/>Strip HTML tags"]
    C -->|"all empty"| E["❌ ValueError"]
    D --> F["RecursiveCharacterTextSplitter<br/>1000 chars, 200 overlap"]
    F --> G["List of Document chunks"]

    style E fill:#c0392b,color:#fff
```

> [!TIP]
> **Why 200-char overlap?** It ensures sentences at chunk boundaries aren't cut in half —
> the retriever can find context that spans two chunks.

---

### 4 · `model.py` — 🧮 Vector Store & RAG Chain

| Aspect | Detail |
|---|---|
| **What it does** | `create_vector_store()` embeds document chunks and saves the FAISS index to disk. `get_rag_chain()` loads the index back and wraps it in a LangChain `RetrievalQA` chain |
| **Why necessary** | FAISS gives O(1) similarity search over thousands of chunks. Persisting to disk means the index survives server restarts. The RetrievalQA chain orchestrates the retrieve → stuff → generate flow |
| **Key exports** | `create_vector_store(docs, embeddings)`, `get_rag_chain(llm, embeddings)` |

```
             ┌──────────────────────────────────────┐
             │         create_vector_store           │
             │                                      │
  docs ─────►│  Ada-002 embed ──► FAISS index       │──► save to disk
             │                                      │
             └──────────────────────────────────────┘

             ┌──────────────────────────────────────┐
             │           get_rag_chain               │
             │                                      │
  disk ─────►│  Load FAISS ──► as_retriever()       │──► RetrievalQA chain
             │                  ──► stuff prompt     │
             │                  ──► GPT-4 generate   │
             └──────────────────────────────────────┘
```

---

### 5 · `app.py` — 🚀 Entry Point (FastAPI + Gradio)

| Aspect | Detail |
|---|---|
| **What it does** | Boots the application: validates config → creates LLM/embeddings → defines REST API endpoints → defines Gradio UI → mounts everything on uvicorn |
| **Why necessary** | Single entry point that exposes **two interfaces** for the same pipeline: a programmatic REST API (for integrations) and a visual Gradio UI (for humans) |
| **Concurrency safety** | Uses `asyncio.Lock` to prevent race conditions when multiple users process URLs simultaneously. Uses `asyncio.to_thread` for sync FAISS operations so the event loop stays free |

```mermaid
flowchart TB
    subgraph STARTUP ["🔑 Startup"]
        S1["validate_config()"] --> S2["get_llm()"]
        S2 --> S3["get_embeddings()"]
    end

    subgraph API ["🌐 FastAPI REST"]
        A1["POST /api/process"]
        A2["POST /api/ask"]
        A3["GET /docs — Swagger"]
    end

    subgraph UI ["🖥️ Gradio Web UI"]
        U1["🔗 Process button"]
        U2["❓ Get Answer button"]
    end

    subgraph SHARED ["🔒 Shared State"]
        ST["app_state + asyncio.Lock"]
    end

    S3 --> API
    S3 --> UI
    A1 --> ST
    A2 --> ST
    U1 --> ST
    U2 --> ST

    style STARTUP fill:#0d1b2a,stroke:#778da9,color:#e0e1dd
    style API fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style UI fill:#1b263b,stroke:#415a77,color:#e0e1dd
    style SHARED fill:#2c1a4a,stroke:#9b59b6,color:#e0e1dd
```

---

## 🔌 API Reference

| Endpoint | Method | Request Body | Response | Purpose |
|---|---|---|---|---|
| `/api/process` | `POST` | `{ "urls": ["..."] }` | `{ "message": "..." }` | Scrape URLs and build FAISS index |
| `/api/ask` | `POST` | `{ "query": "..." }` | `{ "answer": "...", "sources": [...] }` | Ask a question against indexed docs |
| `/docs` | `GET` | — | Swagger UI | Interactive API documentation |
| `/` | `GET` | — | Gradio UI | Visual web interface |

---

## 🛡️ Safety & Error Handling

```mermaid
flowchart TD
    A["App starts"] --> B{"validate_config()"}
    B -->|"Missing keys"| C["❌ EnvironmentError<br/>Clear message about .env"]
    B -->|"All set"| D["Boot LLM + Embeddings"]
    D --> E{"User calls /api/process"}
    E --> F{"URLs empty?"}
    F -->|"Yes"| G["❌ 400: Provide at least one URL"]
    F -->|"No"| H{"Scrape succeeds?"}
    H -->|"All pages empty"| I["❌ ValueError: No content extracted"]
    H -->|"Some pages OK"| J{"Docs list empty?"}
    J -->|"Yes"| K["❌ ValueError: No documents to index"]
    J -->|"No"| L["✅ FAISS index built"]
    L --> M{"User calls /api/ask"}
    M --> N{"Chain exists?"}
    N -->|"No"| O["❌ 400: Process URLs first"]
    N -->|"Yes"| P["✅ Answer + Sources returned"]

    style C fill:#c0392b,color:#fff
    style G fill:#c0392b,color:#fff
    style I fill:#c0392b,color:#fff
    style K fill:#c0392b,color:#fff
    style O fill:#c0392b,color:#fff
    style L fill:#27ae60,color:#fff
    style P fill:#27ae60,color:#fff
```

---

## 📦 Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Web Server** | Uvicorn | ASGI server running the async event loop |
| **API Framework** | FastAPI | REST endpoints with auto-generated Swagger docs |
| **Web UI** | Gradio | Interactive browser UI, mounted on FastAPI |
| **Orchestration** | LangChain | Chains, loaders, splitters, retrievers |
| **LLM** | Azure OpenAI GPT-4 | Generates answers from retrieved context |
| **Embeddings** | Azure OpenAI Ada-002 | Converts text to 1536-dimensional vectors |
| **Vector Store** | FAISS (CPU) | Fast approximate nearest-neighbor search |
| **Scraping** | aiohttp + Html2Text | Concurrent HTML fetch and clean-up |
| **Config** | python-dotenv | Loads `.env` file into `os.environ` |
| **Validation** | Pydantic | Request/response schema enforcement |
