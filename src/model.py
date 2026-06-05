from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from src.config import Config


def create_vector_store(docs, embeddings):
    """Embed documents and save the FAISS index to disk."""
    if not docs:
        raise ValueError("No documents to index. Check that URLs returned valid content.")

    vector_index = FAISS.from_documents(docs, embeddings)
    vector_index.save_local(Config.FAISS_INDEX_PATH)
    return vector_index


def get_rag_chain(llm, embeddings):
    """Load the FAISS index from disk and build a RetrievalQA chain."""
    vector_index = FAISS.load_local(
        Config.FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_index.as_retriever(),
        return_source_documents=True,
    )