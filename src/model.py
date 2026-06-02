from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from src.config import Config

def create_vector_store(docs, embeddings):
    vector_index = FAISS.from_documents(docs, embeddings)
    vector_index.save_local(Config.FAISS_INDEX_PATH)
    return vector_index

def get_rag_chain(llm, embeddings):
    vector_index_loaded = FAISS.load_local(
        Config.FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_index_loaded.as_retriever(),
        return_source_documents=True
    )