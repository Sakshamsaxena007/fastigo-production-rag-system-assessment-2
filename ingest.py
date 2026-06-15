import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Paths
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"

# Embedding model (free, local)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Chunking strategy
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)

def load_pdfs():
    docs = []
    for f in DATA_DIR.glob("pdf/*.pdf"):
        loader = PyPDFLoader(str(f))
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} PDF pages")
    return docs

def load_htmls():
    docs = []
    for f in DATA_DIR.glob("html/*.html"):
        loader = BSHTMLLoader(str(f))
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} HTML pages")
    return docs

def load_csvs():
    docs = []
    for f in DATA_DIR.glob("csv/*.csv"):
        loader = CSVLoader(str(f))
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} CSV rows")
    return docs

def ingest_all():
    print("Starting ingestion...")
    all_docs = load_pdfs() + load_htmls() + load_csvs()

    if not all_docs:
        print("No documents found! Add files to data/pdf, data/html, or data/csv folders.")
        return

    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")

    print("Storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"Done! {len(chunks)} chunks stored in ChromaDB.")

if __name__ == "__main__":
    ingest_all()
    