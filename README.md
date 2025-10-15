\ 📈 Financial News Q&A with RAG and Azure OpenAI

This project is a powerful Question-Answering application that leverages
Retrieval-Augmented Generation (RAG) to provide insights from online
financial news articles. Users can input a list of URLs, and the system
builds a dynamic knowledge base to answer questions based \only\ on
the content of those articles. The entire interface is built with Gradio
for easy and interactive use.


✨ **Features**

**Dynamic Knowledge Base**: Ingests content directly from a list of URLs
to create a knowledge base on the fly.

**Asynchronous Processing**: Efficiently loads data from multiple web
pages concurrently.

**RAG Pipeline**: Implements a full Retrieval-Augmented Generation
pipeline to ensure answers are grounded in the provided text, minimizing
hallucinations.

**Source Attribution**: Cites the source URLs from which the answer was
derived.

**Clean & Interactive UI**: A simple and intuitive user interface built
with Gradio allows for easy interaction.

**⚙️ Tech Stack**

**Language**: Python

**Core Framework**: LangChain

**LLM & Embeddings**: Azure OpenAI (\`gpt-4\`,
\`text-embedding-ada-002\`)

**Vector Store**: FAISS (Facebook AI Similarity Search) for efficient
local similarity search.

**Web UI**: Gradio

**Data Handling**: \`AsyncHtmlLoader\`, \`Html2TextTransformer\`,
\`RecursiveCharacterTextSplitter\`.

**Environment Management**: \`python-dotenv\`

**🚀 How It Works**

The application follows a sophisticated Retrieval-Augmented Generation
(RAG) architecture:

1\. **Load**: The system takes a list of URLs and uses
\`AsyncHtmlLoader\` to fetch the raw HTML content from each page.

2\. **Transform**: The raw HTML is parsed and converted into clean,
plain text using \`Html2TextTransformer\`, removing scripts, styles, and
other noise.

3\. **Split**: The cleaned text is divided into smaller, manageable
chunks using \`RecursiveCharacterTextSplitter\`. This is crucial for the
embedding model\'s context window.

4\. **Embed & Store**: Each text chunk is converted into a numerical
vector representation (embedding) using Azure\'s
\`text-embedding-ada-002\` model. These vectors are then stored in a
local FAISS vector index for fast retrieval.

5\. **Retrieve & Generate**: When a user asks a question:

\* The question is also converted into an embedding.

\* The FAISS index performs a similarity search to find the most
relevant text chunks from the original articles.

\* These relevant chunks, along with the user\'s question, are passed as
context to the Azure OpenAI \`gpt-4\` model.

\* The model generates a comprehensive answer based \*only\* on the
provided context, citing the source documents.

