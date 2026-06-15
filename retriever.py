import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_groq import ChatGroq

load_dotenv()

CHROMA_DIR = "chroma_db"

# Same embedding model as ingest.py — must match
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

def get_retriever():
    vectorstore = get_vectorstore()

    # Base retriever — fetch top 5 relevant chunks
    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Context compression — removes irrelevant parts from chunks
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    compressor = LLMChainExtractor.from_llm(llm)

    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    return compressed_retriever

def retrieve_docs(query: str):
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return docs

if __name__ == "__main__":
    query = "test query"
    results = retrieve_docs(query)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])
